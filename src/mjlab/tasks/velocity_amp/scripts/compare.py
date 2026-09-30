"""Watch several checkpoints side by side in one viewer.

Every policy controls its own robot(s) and, by default, all robots receive the
same constant velocity command, so gaits can be compared directly. With
``--random-commands`` the training command distribution is used instead.

By default each policy gets its own view: one simulation drives all robots and
one web viewer per policy (ports ``--port``, ``--port``+1, ...) shows only that
policy's robot, with the camera following it. A page ``compare.html`` puts the
views side by side. Every view has a "Policy" dropdown to switch to any loaded
policy, so many checkpoints can be loaded and a few views (``--views``) used to
pick among them. ``--single`` shows every robot in one interactive viewer.

Usage (CPU is fine for a handful of robots):
  python -m mjlab.tasks.velocity_amp.scripts.compare \
    --policy expert=checkpoints/demo/expert_step.pt \
    --policy v4b=checkpoints/demo/gail_v4b.pt \
    --policy v7c=checkpoints/demo/gail_v7c.pt \
    --device cpu --lin-x 0.5
Then open the printed compare.html (or http://localhost:8080, 8081, ...).
"""

import argparse
import time
from pathlib import Path

import torch
import viser

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
from mjlab.viewer.viser.scene import MjlabViserScene


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
  parser.add_argument("--single", action="store_true", help="all robots in one view")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--columns", type=int, default=3)
  parser.add_argument(
    "--views", type=int, default=0, help="number of views (default: one per policy)"
  )
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

  if args.single:
    ViserPlayViewer(wrapped, policy).run()
  else:
    names = [name for name, _ in specs]
    run_split(env, wrapped, policy, names, args)
  env.close()


def run_split(env, wrapped, policy, names: list[str], args) -> None:
  """One simulation, several web viewers; each shows the first robot of the
  policy selected in its dropdown (initially policy k for view k)."""
  sim = env.sim
  num_views = args.views if args.views > 0 else len(names)
  servers, scenes = [], []
  for k in range(num_views):
    server = viser.ViserServer(port=args.port + k, label=f"view {k}", verbose=False)
    scene = MjlabViserScene(
      server=server,
      mj_model=sim.mj_model,
      num_envs=env.num_envs,
      sim_model=sim.model,
      expanded_fields=sim.expanded_fields,
    )
    scene.env_idx = (k % len(names)) * args.copies
    scene.show_only_selected = True
    add_policy_selector(server, scene, names, k % len(names), args.copies)
    servers.append(server)
    scenes.append(scene)

  page = write_grid_page(num_views, args.port, args.columns)
  last = args.port + num_views - 1
  print(f"Open {page}  (or http://localhost:{args.port}..{last})")
  print("Ctrl+C to stop.")

  obs = wrapped.get_observations()
  step_dt = env.step_dt
  try:
    while True:
      start = time.perf_counter()
      with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
      for server, scene in zip(servers, scenes, strict=True):
        with server.atomic():
          scene.update(sim.data)
        server.flush()
      time.sleep(max(0.0, step_dt - (time.perf_counter() - start)))
  except KeyboardInterrupt:
    pass


def add_policy_selector(
  server: viser.ViserServer,
  scene: MjlabViserScene,
  names: list[str],
  initial: int,
  copies: int,
) -> None:
  dropdown = server.gui.add_dropdown(
    "Policy", tuple(names), initial_value=names[initial]
  )

  @dropdown.on_update
  def _(_) -> None:
    scene.env_idx = names.index(dropdown.value) * copies


def write_grid_page(num_views: int, port: int, columns: int) -> Path:
  cells = "\n".join(
    f'<figure><iframe src="http://localhost:{port + k}"></iframe></figure>'
    for k in range(num_views)
  )
  cols = min(columns, num_views)
  html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Policy comparison</title><style>
body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
main {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 6px;
       padding: 6px; height: 100vh; box-sizing: border-box; }}
figure {{ margin: 0; display: flex; flex-direction: column; min-height: 0; }}
iframe {{ flex: 1; width: 100%; border: 0; }}
</style></head><body><main>
{cells}
</main></body></html>"""
  path = Path("/tmp/mjlab_compare.html")
  path.write_text(html)
  return path


if __name__ == "__main__":
  main()
