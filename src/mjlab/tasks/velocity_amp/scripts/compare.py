"""Watch several checkpoints side by side in one viewer.

Every policy controls its own robot(s) in the same scene and, by default, all
robots receive the same constant velocity command, so gaits can be compared
directly. With ``--random-commands`` the training command distribution is used
instead (commands then differ between robots).

Usage (CPU is fine for a handful of robots):
  python -m mjlab.tasks.velocity_amp.scripts.compare \
    --policy expert=checkpoints/demo/expert_step.pt \
    --policy v4b=checkpoints/demo/gail_v4b.pt \
    --policy v7c=checkpoints/demo/gail_v7c.pt \
    --device cpu --lin-x 0.5
Then open http://localhost:8080.
"""

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity_amp.scripts.common import (
  EXPERT_TASK,
  default_device,
  load_policy,
)
from mjlab.viewer import ViserPlayViewer


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", action="append", required=True, help="name=path")
  parser.add_argument("--copies", type=int, default=1, help="robots per policy")
  parser.add_argument("--lin-x", type=float, default=0.5)
  parser.add_argument("--lin-y", type=float, default=0.0)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--random-commands", action="store_true")
  parser.add_argument("--spacing", type=float, default=2.0)
  parser.add_argument("--device", type=str, default=default_device())
  args = parser.parse_args()

  specs = [p.split("=", 1) for p in args.policy]
  num_envs = len(specs) * args.copies

  cfg = load_env_cfg(EXPERT_TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.scene.env_spacing = args.spacing
  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  if not args.random_commands:
    # The ranges stay as they are (the viewer's command sliders are sized from
    # them); the sampled command is overwritten below instead.
    cmd.ranges.heading = None
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.rel_world_envs = 0.0

  env = ManagerBasedRlEnv(cfg, device=args.device)
  if not args.random_commands:
    term = env.command_manager.get_term("twist")
    assert isinstance(term, UniformVelocityCommand)
    fixed = torch.tensor([args.lin_x, args.lin_y, args.yaw], device=args.device)
    sample = term._resample_command

    def resample_fixed(env_ids: torch.Tensor) -> None:
      sample(env_ids)
      term.vel_command_b[env_ids] = fixed

    setattr(term, "_resample_command", resample_fixed)  # noqa: B010
    term.vel_command_b[:] = fixed

  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(EXPERT_TASK).clip_actions)
  policies = [load_policy(wrapped, path, args.device) for _, path in specs]

  # Robot i is driven by policy i // copies.
  groups = [
    torch.arange(k * args.copies, (k + 1) * args.copies, device=args.device)
    for k in range(len(specs))
  ]
  origins = env.scene.env_origins.cpu()
  print("\nRobot -> policy (x, y of the robot's start position):")
  for k, (name, path) in enumerate(specs):
    for i in groups[k].tolist():
      x, y = origins[i, 0].item(), origins[i, 1].item()
      print(f"  env {i:2d}  ({x:5.1f}, {y:5.1f})  {name:12s} {path}")
  print()

  num_actions = env.action_manager.total_action_dim

  def policy(obs: torch.Tensor) -> torch.Tensor:
    # The viewer passes the wrapper's observation TensorDict (typed as a Tensor
    # in PolicyProtocol); both support row indexing.
    actions = torch.zeros(num_envs, num_actions, device=args.device)
    for group, pol in zip(groups, policies, strict=True):
      actions[group] = pol(obs[group])
    return actions

  ViserPlayViewer(wrapped, policy).run()
  env.close()


if __name__ == "__main__":
  main()
