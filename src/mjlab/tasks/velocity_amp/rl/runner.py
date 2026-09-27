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


class AmpOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that swaps PPO for :class:`AmpPPO`.

  AMP settings are read from environment variables so they can be changed
  without touching the shared runner config:

  - ``MJLAB_AMP_EXPERT``: expert transitions (default ``DEFAULT_EXPERT_FILE``).
  - ``MJLAB_AMP_LOSS``: ``amp`` (default) or ``gail``.
  - ``MJLAB_AMP_STYLE_WEIGHT``: style reward weight (default 2.0).
  - ``MJLAB_AMP_DISC_LR``: discriminator learning rate (default 1e-4).
  - ``MJLAB_AMP_DISC_UPDATES``: discriminator steps per iteration (default 10).

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
      "step_dt": env.unwrapped.step_dt,  # type: ignore[attr-defined]
    }
    print(f"[INFO] AMP config: {amp_cfg}")
    train_cfg["algorithm"]["class_name"] = "mjlab.tasks.velocity_amp.rl:AmpPPO"
    train_cfg["algorithm"]["amp_cfg"] = amp_cfg
    super().__init__(env, train_cfg, log_dir, device)
