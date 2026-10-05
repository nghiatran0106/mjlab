import os

from rsl_rl.env import VecEnv

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

DEFAULT_EXPERT_FILE = "logs/amp_expert/g1_velocity_expert.npz"


def apply_exploration_overrides(train_cfg: dict) -> None:
  """Optional PPO exploration overrides shared by the task-only and AMP runs.

  - ``MJLAB_INIT_STD``: initial action std of the Gaussian policy.
  - ``MJLAB_ENTROPY_COEF``: entropy bonus coefficient.
  """
  if "MJLAB_INIT_STD" in os.environ:
    dist_cfg = train_cfg["actor"].setdefault("distribution_cfg", {})
    dist_cfg["init_std"] = float(os.environ["MJLAB_INIT_STD"])
  if "MJLAB_ENTROPY_COEF" in os.environ:
    train_cfg["algorithm"]["entropy_coef"] = float(os.environ["MJLAB_ENTROPY_COEF"])
  print(
    "[INFO] Exploration: init_std="
    f"{train_cfg['actor'].get('distribution_cfg', {}).get('init_std')}, "
    f"entropy_coef={train_cfg['algorithm']['entropy_coef']}"
  )


class TaskOnlyOnPolicyRunner(VelocityOnPolicyRunner):
  """Plain PPO runner that honors the exploration overrides."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    apply_exploration_overrides(train_cfg)
    super().__init__(env, train_cfg, log_dir, device)


class FineTuneOnPolicyRunner(VelocityOnPolicyRunner):
  """Plain PPO runner that can start from the weights of another checkpoint.

  ``MJLAB_INIT_CHECKPOINT`` names a checkpoint whose actor and critic (with their
  observation normalizers) initialize the networks. The iteration counter and
  the optimizer state are not loaded, so checkpoints are numbered from 0 as in
  a run from scratch. Without the variable the runner is the stock one.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    path = os.environ.get("MJLAB_INIT_CHECKPOINT", "")
    if path:
      self.load(path, load_cfg={"actor": True, "critic": True}, map_location=device)
      print(f"[INFO] Initialized actor and critic from {path}")


class AmpOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that swaps PPO for :class:`AmpPPO`.

  AMP settings are read from environment variables so they can be changed
  without touching the shared runner config:

  - ``MJLAB_AMP_EXPERT``: expert transitions (default ``DEFAULT_EXPERT_FILE``).
  - ``MJLAB_AMP_LOSS``: ``amp`` (default) or ``gail``.
  - ``MJLAB_AMP_STYLE_WEIGHT``: style reward weight (default 2.0).
  - ``MJLAB_AMP_DISC_LR``: discriminator learning rate (default 1e-4).
  - ``MJLAB_AMP_DISC_UPDATES``: discriminator steps per iteration (default 10).
  - ``MJLAB_AMP_TASK_LERP``: reward = lerp * task + (1 - lerp) * style when in
    [0, 1] (default -1: task + style).
  - ``MJLAB_AMP_REPLAY``: policy-transition replay buffer size (default 0).
  - ``MJLAB_AMP_LERP_START``, ``MJLAB_AMP_LERP_HOLD``, ``MJLAB_AMP_LERP_RAMP``:
    schedule for the task-reward weight (see ``AmpCfg.lerp_start``).
  - ``MJLAB_AMP_BC_FILE``, ``MJLAB_AMP_BC_STEPS``: behavior-cloning pretraining.
  - ``MJLAB_AMP_CRITIC_WARMUP``: iterations that update only the critic.
  - ``MJLAB_AMP_ACTION_RATE_TARGET``: if set, the action-rate penalty weight is
    a Lagrange multiplier adapted toward this per-step ``||a_t - a_{t-1}||^2``
    of the policy mean (see ``AmpCfg.action_rate_target``); with
    ``MJLAB_AMP_ACTION_RATE_INIT``, ``_LR`` and ``_MAX``.

  The exploration overrides of :func:`apply_exploration_overrides` also apply.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    apply_exploration_overrides(train_cfg)
    amp_cfg = {
      "expert_file": os.environ.get("MJLAB_AMP_EXPERT", DEFAULT_EXPERT_FILE),
      "loss_type": os.environ.get("MJLAB_AMP_LOSS", "amp"),
      "style_reward_weight": float(os.environ.get("MJLAB_AMP_STYLE_WEIGHT", "2.0")),
      "learning_rate": float(os.environ.get("MJLAB_AMP_DISC_LR", "1e-4")),
      "num_updates": int(os.environ.get("MJLAB_AMP_DISC_UPDATES", "10")),
      "task_reward_lerp": float(os.environ.get("MJLAB_AMP_TASK_LERP", "-1")),
      "replay_size": int(os.environ.get("MJLAB_AMP_REPLAY", "0")),
      "lerp_start": float(os.environ.get("MJLAB_AMP_LERP_START", "-1")),
      "lerp_hold_iters": int(os.environ.get("MJLAB_AMP_LERP_HOLD", "0")),
      "lerp_ramp_iters": int(os.environ.get("MJLAB_AMP_LERP_RAMP", "0")),
      "bc_file": os.environ.get("MJLAB_AMP_BC_FILE", ""),
      "bc_steps": int(os.environ.get("MJLAB_AMP_BC_STEPS", "2000")),
      "critic_warmup_iters": int(os.environ.get("MJLAB_AMP_CRITIC_WARMUP", "0")),
      "action_rate_target": float(os.environ.get("MJLAB_AMP_ACTION_RATE_TARGET", "-1")),
      "action_rate_init": float(os.environ.get("MJLAB_AMP_ACTION_RATE_INIT", "0.1")),
      "action_rate_lr": float(os.environ.get("MJLAB_AMP_ACTION_RATE_LR", "0.01")),
      "action_rate_max": float(os.environ.get("MJLAB_AMP_ACTION_RATE_MAX", "1.0")),
      "step_dt": env.unwrapped.step_dt,  # type: ignore[attr-defined]
    }
    print(f"[INFO] AMP config: {amp_cfg}")
    train_cfg["algorithm"]["class_name"] = "mjlab.tasks.velocity_amp.rl:AmpPPO"
    train_cfg["algorithm"]["amp_cfg"] = amp_cfg
    super().__init__(env, train_cfg, log_dir, device)
