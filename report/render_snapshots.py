"""Chụp ảnh từng policy với cùng lệnh đi thẳng 0,5 m/s (cho mục định tính).

Mỗi policy cho một dải 5 khung hình ở t = 1..5 s và quãng đường đi được.

Usage (từ thư mục repo, CPU + OSMesa):
  MUJOCO_GL=osmesa PYTHONPATH=src ~/venvs/mjlab/bin/python report/render_snapshots.py \
    --policy expert=path/a.pt --policy amp_v2=path/b.pt
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

TASK = "Mjlab-Velocity-Flat-Unitree-G1-Expert"
FORWARD = 0.5
SNAP_TIMES_S = (1, 2, 3, 4, 5)


def make_env() -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  cfg = load_env_cfg(TASK, play=True)
  cfg.seed = 7
  cfg.scene.num_envs = 1
  cfg.viewer.width, cfg.viewer.height = 320, 400
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x = (FORWARD, FORWARD)  # type: ignore[attr-defined]
  cmd.ranges.lin_vel_y = (0.0, 0.0)  # type: ignore[attr-defined]
  cmd.ranges.ang_vel_z = (0.0, 0.0)  # type: ignore[attr-defined]
  cmd.ranges.heading = None  # type: ignore[attr-defined]
  cmd.heading_command = False  # type: ignore[attr-defined]
  cmd.rel_standing_envs = 0.0  # type: ignore[attr-defined]
  cmd.rel_heading_envs = 0.0  # type: ignore[attr-defined]
  cmd.rel_forward_envs = 0.0  # type: ignore[attr-defined]
  cmd.rel_world_envs = 0.0  # type: ignore[attr-defined]
  cmd.resampling_time_range = (1000.0, 1000.0)
  env = ManagerBasedRlEnv(cfg, device="cpu", render_mode="rgb_array")
  return env, RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)


def snapshot(name: str, checkpoint: str, out_dir: Path) -> dict:
  env, wrapped = make_env()
  runner = VelocityOnPolicyRunner(wrapped, asdict(load_rl_cfg(TASK)), device="cpu")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")
  robot = env.scene["robot"]
  snap_steps = {round(t / env.step_dt) for t in SNAP_TIMES_S}
  frames, fell = [], False
  obs = wrapped.get_observations()
  start = robot.data.root_link_pos_w[0, :2].clone()
  with torch.inference_mode():
    for step in range(1, max(snap_steps) + 1):
      obs, _, dones, _ = wrapped.step(policy(obs))
      fell |= bool(dones[0])
      if step in snap_steps:
        frames.append(np.asarray(env.render()))
  dist = (robot.data.root_link_pos_w[0, :2] - start).norm().item()
  env.close()

  strip = Image.fromarray(np.concatenate(frames, axis=1))
  draw = ImageDraw.Draw(strip)
  for i, t in enumerate(SNAP_TIMES_S):
    draw.text((i * frames[0].shape[1] + 6, 6), f"t={t}s", fill=(0, 0, 0))
  strip.save(out_dir / f"snap_{name}.png")
  return {"distance_m": dist, "fell": fell, "seconds": max(SNAP_TIMES_S)}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", action="append", required=True, help="name=path")
  parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  summary_path = args.out / "snapshots.json"
  summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
  for spec in args.policy:
    name, path = spec.split("=", 1)
    summary[name] = snapshot(name, path, args.out)
    print(f"[INFO] {name}: {summary[name]}", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
