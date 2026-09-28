"""Roll out an expert policy and save its (s, s') AMP-feature transitions.

Usage:
  uv run python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
    <checkpoint.pt> --out logs/amp_expert/g1_velocity_expert.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.velocity_amp.mdp import save_expert_states, snapshot_state
from mjlab.tasks.velocity_amp.scripts.common import (
  EXPERT_TASK,
  default_device,
  load_policy,
  make_env,
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=str)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default=default_device())
  parser.add_argument("--task", type=str, default=EXPERT_TASK)
  parser.add_argument(
    "--states-out",
    type=Path,
    default=None,
    help="Also save robot states every --state-every steps (for RSI).",
  )
  parser.add_argument("--state-every", type=int, default=5)
  args = parser.parse_args()

  # Pushes stay on: the AMP policy is trained with pushes, so the expert data
  # should also contain push recoveries.
  env, wrapped = make_env(
    args.num_envs, args.seed, args.device, pushes=True, task=args.task
  )
  policy = load_policy(wrapped, args.checkpoint, args.device)

  robot = env.scene["robot"]
  s_list, s_next_list, states = [], [], []
  obs = wrapped.get_observations()
  with torch.inference_mode():
    for t in range(args.steps):
      # Skip the first second: states right after reset are not gait states.
      if args.states_out is not None and t >= 50 and t % args.state_every == 0:
        states.append(snapshot_state(robot))
      s = obs["amp"].clone()
      obs, _, dones, _ = wrapped.step(policy(obs))
      valid = dones == 0
      s_list.append(s[valid].cpu())
      s_next_list.append(obs["amp"][valid].cpu())
  env.close()

  s_all = torch.cat(s_list).numpy().astype(np.float32)
  s_next_all = torch.cat(s_next_list).numpy().astype(np.float32)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.out, s=s_all, s_next=s_next_all)
  print(f"[INFO] Saved {s_all.shape[0]} transitions of dim {s_all.shape[1]}")
  print(f"[INFO] -> {args.out}")
  if args.states_out is not None:
    save_expert_states(args.states_out, states)
    print(f"[INFO] Saved {sum(len(d['quat']) for d in states)} states")
    print(f"[INFO] -> {args.states_out}")


if __name__ == "__main__":
  main()
