"""Compare checkpoints under the original hand-shaped ("true") reward.

For each policy this reports:
  - the per-term reward rate of the original velocity reward, which the
    task-only and AMP policies never saw during training,
  - velocity tracking errors and falls,
  - how far its AMP features are from the expert data (1D Wasserstein
    distance per feature, in units of the expert std, averaged).

Usage:
  uv run python -m mjlab.tasks.velocity_amp.scripts.evaluate \
    --expert-file logs/amp_expert/g1_velocity_expert.npz \
    --policy expert=path/a.pt --policy task_only=path/b.pt --policy amp=path/c.pt \
    --out logs/amp_eval/results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity_amp.scripts.common import (
  EXPERT_TASK,
  default_device,
  load_policy,
  make_env,
)

_QUANTILES = torch.linspace(0.0, 1.0, 101)


DOWN_HEIGHT = 0.5  # m; G1 pelvis stands at ~0.75 m.
MOVING_SPEED = 0.2  # m/s; gait statistics only while asked to walk.


def feature_gap(policy_feats: torch.Tensor, expert_feats: torch.Tensor) -> float:
  """Mean over features of W1(policy, expert) / std(expert)."""
  q = _QUANTILES.to(policy_feats.device)
  qp = torch.quantile(policy_feats, q, dim=0)
  qe = torch.quantile(expert_feats, q, dim=0)
  w1 = (qp - qe).abs().mean(dim=0)
  return (w1 / expert_feats.std(dim=0).clamp_min(1e-2)).mean().item()


def subsample(x: torch.Tensor, n: int = 200_000) -> torch.Tensor:
  # torch.quantile has an input size limit.
  if x.shape[0] <= n:
    return x
  return x[torch.randperm(x.shape[0], device=x.device)[:n]]


def evaluate(checkpoint: str, args: argparse.Namespace, expert: torch.Tensor) -> dict:
  env, wrapped = make_env(
    args.num_envs, args.seed, args.device, pushes=False, task=args.task
  )
  policy = load_policy(wrapped, checkpoint, args.device)
  robot = env.scene["robot"]
  terms = env.reward_manager.active_terms
  term_sums = torch.zeros(len(terms), device=args.device)
  lin_err = ang_err = 0.0
  falls = 0
  # Progress along the commanded direction, and time spent with the pelvis
  # near the ground: the tilt-based fall check misses a robot that crawls.
  progress = commanded = 0.0
  down = 0.0
  # Gait: touchdowns, swing peak height and air time, for moving commands only.
  prev_contact = None
  prev_air = peak = torch.zeros(0)
  touchdowns = moving_foot_steps = 0
  lift_sum = air_sum = 0.0
  feats = []

  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, dones, extras = wrapped.step(policy(obs))
      # Weighted per-second reward of each term (before the dt scaling).
      term_sums += env.reward_manager._step_reward.mean(dim=0)
      cmd = env.command_manager.get_command("twist")
      assert cmd is not None
      lin_err += (cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]).norm(dim=1).mean()
      ang_err += (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs().mean()
      v_xy = robot.data.root_link_lin_vel_b[:, :2]
      progress += (v_xy * cmd[:, :2]).sum(dim=1).sum()
      commanded += cmd[:, :2].square().sum(dim=1).sum()
      down += (robot.data.root_link_pos_w[:, 2] < DOWN_HEIGHT).float().mean()
      contact = velocity_mdp.foot_contact(env, "feet_ground_contact") > 0.5
      height = velocity_mdp.foot_height(env, "foot_height_scan")
      air = velocity_mdp.foot_air_time(env, "feet_ground_contact")
      moving = (cmd[:, :2].norm(dim=1) > MOVING_SPEED)[:, None].expand_as(contact)
      if prev_contact is None:
        peak = torch.zeros_like(height)
      else:
        peak = torch.maximum(peak, torch.where(contact, 0.0, height))
        td = contact & ~prev_contact & moving & (dones == 0)[:, None]
        touchdowns += int(td.sum())
        lift_sum += float(peak[td].sum())
        air_sum += float(prev_air[td].sum())
        peak = torch.where(contact, 0.0, peak)
      moving_foot_steps += int(moving.sum())
      prev_contact, prev_air = contact, air.clone()
      time_outs = extras.get("time_outs", torch.zeros_like(dones))
      falls += int(((dones == 1) & ~time_outs.bool()).sum())
      feats.append(obs["amp"][dones == 0].clone())
  env.close()

  per_term = {t: (v / args.steps).item() for t, v in zip(terms, term_sums, strict=True)}
  sim_seconds = args.steps * env.step_dt * args.num_envs
  return {
    "checkpoint": checkpoint,
    "true_reward_rate": sum(per_term.values()),
    "reward_terms": per_term,
    "lin_vel_error_m_s": float(lin_err) / args.steps,
    "yaw_vel_error_rad_s": float(ang_err) / args.steps,
    "falls_per_minute": falls / sim_seconds * 60.0,
    # 1.0 = moves exactly at the commanded speed, 0.0 = stands still.
    "speed_ratio": float(progress) / max(float(commanded), 1e-6),
    "down_fraction": float(down) / args.steps,
    # Per foot, while the command is moving.
    "touchdowns_per_s": touchdowns / max(moving_foot_steps * env.step_dt, 1e-6),
    "swing_peak_cm": 100.0 * lift_sum / max(touchdowns, 1),
    "air_time_s": air_sum / max(touchdowns, 1),
    "expert_feature_gap": feature_gap(subsample(torch.cat(feats)), expert),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", action="append", required=True, help="name=path")
  parser.add_argument("--expert-file", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--device", type=str, default=default_device())
  parser.add_argument(
    "--task",
    type=str,
    default=EXPERT_TASK,
    help="evaluation task (hand-written reward + AMP features), e.g. a variant",
  )
  args = parser.parse_args()

  expert = torch.as_tensor(np.load(args.expert_file)["s"], device=args.device)
  expert = subsample(expert)

  results = {}
  for spec in args.policy:
    name, path = spec.split("=", 1)
    print(f"[INFO] Evaluating {name}: {path}")
    results[name] = evaluate(path, args, expert)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=2))

  header = (
    "| policy | true reward/s | lin err (m/s) | yaw err (rad/s) "
    "| falls/min | speed ratio | down | expert gap | steps/s/foot | lift cm "
    "| air s |\n|---|---|---|---|---|---|---|---|---|---|---|"
  )
  print(header)
  for name, r in results.items():
    print(
      f"| {name} | {r['true_reward_rate']:.3f} | {r['lin_vel_error_m_s']:.3f} "
      f"| {r['yaw_vel_error_rad_s']:.3f} | {r['falls_per_minute']:.2f} "
      f"| {r['speed_ratio']:.2f} | {r['down_fraction']:.2f} "
      f"| {r['expert_feature_gap']:.3f} | {r['touchdowns_per_s']:.2f} "
      f"| {r['swing_peak_cm']:.1f} | {r['air_time_s']:.3f} |"
    )
  print(f"[INFO] -> {args.out}")


if __name__ == "__main__":
  main()
