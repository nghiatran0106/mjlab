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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from mjlab.viewer.viser.overlays import ViserTermOverlays
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
    "--record",
    type=float,
    default=0.0,
    help="simulate this many seconds first, then replay smoothly (split views)",
  )
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
  fixed = None
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
    run_split(env, wrapped, policy, names, args, fixed)
  env.close()


Terms = list[tuple[str, np.ndarray]]


@dataclass
class Frame:
  """Everything the views show for one control step, for all robots."""

  xpos: np.ndarray
  xmat: np.ndarray
  command: np.ndarray  # (num_envs, 3)
  lin_vel: np.ndarray  # (num_envs, 3), body frame
  yaw_rate: np.ndarray
  height: np.ndarray
  episode_time: np.ndarray
  falls: np.ndarray
  rewards: list[Terms]  # per robot
  metrics: list[Terms]


def capture(env, falls: np.ndarray) -> Frame:
  robot = env.scene["robot"]
  n = env.num_envs
  return Frame(
    xpos=env.sim.data.xpos.cpu().numpy().copy(),
    xmat=env.sim.data.xmat.cpu().numpy().copy(),
    command=env.command_manager.get_command("twist").cpu().numpy().copy(),
    lin_vel=robot.data.root_link_lin_vel_b.cpu().numpy().copy(),
    yaw_rate=robot.data.root_link_ang_vel_b[:, 2].cpu().numpy().copy(),
    height=robot.data.root_link_pos_w[:, 2].cpu().numpy().copy(),
    episode_time=env.episode_length_buf.cpu().numpy() * env.step_dt,
    falls=falls.copy(),
    rewards=[_terms(env.reward_manager, i) for i in range(n)],
    metrics=[_terms(env.metrics_manager, i) for i in range(n)],
  )


def _terms(manager, env_idx: int) -> Terms:
  return [
    (name, np.asarray(v)) for name, v in manager.get_active_iterable_terms(env_idx)
  ]


@dataclass
class Shared:
  """State shared by all views (their GUI callbacks run in other threads)."""

  command: torch.Tensor | None  # live mode: fixed command, edited by sliders
  duration: float  # replay mode: length of the recording in seconds, else 0
  paused: bool = False
  restart: bool = False
  speed: float = 1.0  # target simulated seconds per wall-clock second
  realtime: float = 0.0  # measured
  seek: float | None = None  # replay mode: jump to this time
  time: float = 0.0


class PolicyView:
  """One web viewer: the robot of the selected policy plus the same tabs as the
  play viewer (info, rewards, metrics, visualization, groups)."""

  def __init__(self, k: int, env, wrapped, names: list[str], shared: Shared, args):
    sim = env.sim
    self.names, self.shared, self.copies = names, shared, args.copies
    self.server = viser.ViserServer(
      port=args.port + k, label=f"view {k}", verbose=False
    )
    self.scene = MjlabViserScene(
      server=self.server,
      mj_model=sim.mj_model,
      num_envs=env.num_envs,
      sim_model=sim.model,
      expanded_fields=sim.expanded_fields,
    )
    self.scene.env_idx = (k % len(names)) * args.copies
    self.scene.show_only_selected = True
    self.prev_env_idx = self.scene.env_idx

    gui = self.server.gui
    tabs = gui.add_tab_group()
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      policy = gui.add_dropdown(
        "Policy", tuple(names), initial_value=names[k % len(names)]
      )

      @policy.on_update
      def _(_) -> None:
        self.scene.env_idx = names.index(policy.value) * self.copies

      with gui.add_folder("Info"):
        self.status = gui.add_html("")
      with gui.add_folder("Playback (all views)"):
        self._add_playback_controls()
      if shared.command is not None:
        with gui.add_folder("Command (all robots)"):
          self._add_command_sliders(shared.command)
      with gui.add_folder("Scene"):
        self.scene.create_scene_gui()
    with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
      self.scene.create_overlay_gui()
    self.overlays = ViserTermOverlays(self.server, wrapped, self.scene, env.step_dt)
    self.overlays.setup_tabs(tabs)
    with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
      self.scene.create_groups_gui()

  def _add_playback_controls(self) -> None:
    gui, shared = self.server.gui, self.shared
    pause = gui.add_button("Pause / Play", icon=viser.Icon.PLAYER_PAUSE)
    speed = gui.add_button_group("Speed", options=["Slower", "1x", "Faster"])
    restart = gui.add_button("Restart" if shared.duration else "Reset all robots")

    @pause.on_click
    def _(_) -> None:
      shared.paused = not shared.paused

    @speed.on_click
    def _(event) -> None:
      value = event.target.value
      if value == "1x":
        shared.speed = 1.0
      else:
        factor = 0.5 if value == "Slower" else 2.0
        shared.speed = min(8.0, max(0.125, shared.speed * factor))

    @restart.on_click
    def _(_) -> None:
      shared.restart = True

    if shared.duration:
      self.timeline = gui.add_slider(
        "Time (s)", min=0.0, max=shared.duration, step=0.02, initial_value=0.0
      )

      @self.timeline.on_update
      def _(event) -> None:
        if event.client is not None:  # moved by the user, not by update()
          shared.seek = self.timeline.value

  def _add_command_sliders(self, command: torch.Tensor) -> None:
    for i, label in enumerate(("lin_x (m/s)", "lin_y (m/s)", "yaw (rad/s)")):
      slider = self.server.gui.add_slider(
        label, min=-1.0, max=1.0, step=0.05, initial_value=float(command[i])
      )

      @slider.on_update
      def _(_, i=i, slider=slider) -> None:
        command[i] = slider.value

  def clear_plots(self) -> None:
    self.overlays.clear_histories()

  def feed_plots(self, frame: Frame) -> None:
    """Append one step of reward/metric terms of the selected robot."""
    i = self.scene.env_idx
    if i != self.prev_env_idx:
      self.prev_env_idx = i
      self.overlays.on_env_switch()
    o = self.overlays
    if o.reward_plotter is not None:
      o.reward_plotter.update(frame.rewards[i])
    if o.reward_bar_panel is not None:
      o.reward_bar_panel.update(frame.rewards[i])
    if o.metrics_plotter is not None:
      o.metrics_plotter.update(frame.metrics[i])

  def draw(self, frame: Frame) -> None:
    with self.server.atomic():
      self.scene.update_from_arrays(frame.xpos, frame.xmat)
      if self.shared.duration:
        self.timeline.value = round(self.shared.time, 2)
    self.server.flush()
    self.status.content = self._status_html(frame)

  def _status_html(self, f: Frame) -> str:
    i, s = self.scene.env_idx, self.shared
    cmd, lin = f.command[i], f.lin_vel[i]
    mode = f"replay {s.time:.1f} / {s.duration:.0f} s" if s.duration else "live"
    rows = [
      ("policy", f"<b>{self.names[i // self.copies]}</b> (env {i})"),
      ("mode", mode + ("  (paused)" if s.paused else "")),
      ("episode time", f"{f.episode_time[i]:.1f} s"),
      ("command x, y, yaw", f"{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}"),
      ("actual x, y, yaw", f"{lin[0]:+.2f}, {lin[1]:+.2f}, {f.yaw_rate[i]:+.2f}"),
      ("pelvis height", f"{f.height[i]:.2f} m"),
      ("falls", str(int(f.falls[i]))),
      ("speed (target / actual)", f"{s.speed:g}x / {s.realtime:.2f}x"),
    ]
    cells = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
    return f"<table style='font-size:0.85em'>{cells}</table>"


# Views are redrawn at most RENDER_HZ times per second; the plots still get
# every control step.
RENDER_HZ = 30


def run_split(
  env, wrapped, policy, names: list[str], args, command: torch.Tensor | None
) -> None:
  """One simulation, several web viewers; each shows the first robot of the
  policy selected in its dropdown (initially policy k for view k).

  Live mode simulates while showing, which runs slower than real time on a CPU.
  With ``--record SECONDS`` the simulation runs first and the recording is then
  replayed smoothly at any speed, in a loop, with a time slider.
  """
  step_dt = env.step_dt
  frames: list[Frame] = []
  obs = wrapped.get_observations()
  falls = np.zeros(env.num_envs)

  def step() -> Frame:
    nonlocal obs
    with torch.inference_mode():
      obs, _, dones, extras = wrapped.step(policy(obs))
    falls[(dones.bool() & ~extras["time_outs"].bool()).cpu().numpy()] += 1
    return capture(env, falls)

  if args.record > 0:
    num_steps = round(args.record / step_dt)
    start = time.perf_counter()
    for t in range(num_steps):
      frames.append(step())
      if (t + 1) % 50 == 0 or t + 1 == num_steps:
        elapsed = time.perf_counter() - start
        eta = elapsed / (t + 1) * (num_steps - t - 1)
        print(f"\rRecording {t + 1}/{num_steps} steps, ~{eta:.0f} s left", end="")
    print()
    command = None

  shared = Shared(command=command, duration=len(frames) * step_dt)
  num_views = args.views if args.views > 0 else len(names)
  views = [PolicyView(k, env, wrapped, names, shared, args) for k in range(num_views)]
  page = write_grid_page(num_views, args.port, args.columns)
  last = args.port + num_views - 1
  print(f"Open {page}  (or http://localhost:{args.port}..{last})")
  print("Ctrl+C to stop.")

  frame = frames[0] if frames else capture(env, falls)
  index = 0  # replay position
  last_draw = clock = time.perf_counter()
  shown = 0.0  # simulated seconds shown since `clock`
  try:
    while True:
      tick = time.perf_counter()
      if shared.restart or shared.seek is not None:
        target = shared.seek or 0.0
        shared.restart, shared.seek = False, None
        if frames:
          index = min(len(frames) - 1, round(target / step_dt))
          frame = frames[index]
        else:
          obs, _ = wrapped.reset()
          falls[:] = 0
          frame = capture(env, falls)
        for view in views:
          view.clear_plots()
      if not shared.paused:
        if frames:
          index += 1
          if index >= len(frames):  # loop
            index = 0
            for view in views:
              view.clear_plots()
          frame = frames[index]
        else:
          if command is not None:  # the sliders change it in place
            term = env.command_manager.get_term("twist")
            assert isinstance(term, UniformVelocityCommand)
            term.vel_command_b[:] = command
          frame = step()
        shown += step_dt
        for view in views:
          view.feed_plots(frame)
      shared.time = index * step_dt
      now = time.perf_counter()
      if now - last_draw >= 1.0 / RENDER_HZ:
        last_draw = now
        for view in views:
          view.draw(frame)
      if now - clock >= 1.0:
        shared.realtime, shown, clock = shown / (now - clock), 0.0, now
        print(f"\r{shared.realtime:.2f}x real time", end="", flush=True)
      time.sleep(max(0.0, step_dt / shared.speed - (time.perf_counter() - tick)))
  except KeyboardInterrupt:
    pass


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
