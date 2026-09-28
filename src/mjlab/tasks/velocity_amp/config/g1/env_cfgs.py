"""G1 flat velocity configs for the task-only and AMP experiments."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity_amp.mdp import (
  amp_observation_group,
  reset_from_expert_states,
)

# Only the command-tracking terms are kept. Every term that shapes *how* the
# robot moves (pose, foot clearance, slip, action rate, ...) is dropped, so
# gait style must come from the learned reward instead.
TASK_REWARD_TERMS = ("track_linear_velocity", "track_angular_velocity")


def _use_training_command_ranges(cfg: ManagerBasedRlEnvCfg) -> None:
  """Play mode widens the command ranges past what 1000 iterations ever sampled
  (the velocity curriculum had not advanced yet); keep them in-distribution."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)


# Discriminator window for the gait variant: 12 frames = 0.24 s. The lite
# variant (no contact flags, 5 frames) keeps the discriminator from saturating.
GAIT_HISTORY = 12
GAIT_LITE_HISTORY = 5


def unitree_g1_flat_amp_env_cfg(
  play: bool = False,
  key_bodies: bool = False,
  gait: bool = False,
  gait_lite: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.observations["amp"] = _amp_group(key_bodies, gait, gait_lite)
  cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in TASK_REWARD_TERMS}
  if play:
    _use_training_command_ranges(cfg)
  else:
    # Reference state initialization; a no-op unless MJLAB_AMP_RSI_FILE and
    # MJLAB_AMP_RSI_PROB are set, so the default resets apply otherwise.
    cfg.events["reset_from_expert"] = EventTermCfg(
      func=reset_from_expert_states, mode="reset"
    )
  return cfg


def unitree_g1_flat_expert_env_cfg(
  play: bool = False,
  key_bodies: bool = False,
  gait: bool = False,
  gait_lite: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Original hand-shaped reward plus AMP features (expert data, evaluation)."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.observations["amp"] = _amp_group(key_bodies, gait, gait_lite)
  if play:
    _use_training_command_ranges(cfg)
  return cfg


def unitree_g1_flat_expert_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Expert with a clearer stepping gait, as a better demonstration source.

  The stock G1 config disables the air-time reward, and the 1000-iteration
  expert lifts its feet only 2-5 cm (target 10 cm) with short steps. Reward
  air time and penalize low swing height harder; keep the command range fixed
  so the demonstrations match the AMP training distribution.
  """
  cfg = unitree_g1_flat_expert_env_cfg(play=play, key_bodies=True, gait=True)
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["foot_swing_height"].weight = -1.0
  cfg.curriculum.pop("command_vel", None)
  return cfg


def _amp_group(key_bodies: bool, gait: bool, gait_lite: bool = False):
  if gait_lite:
    return amp_observation_group(
      key_bodies=True, feet=True, history=GAIT_LITE_HISTORY, contact=False
    )
  if gait:
    return amp_observation_group(key_bodies=True, feet=True, history=GAIT_HISTORY)
  return amp_observation_group(key_bodies)


def _conditional_group():
  """Key-body features plus the velocity command (conditional discriminator)."""
  return amp_observation_group(key_bodies=True, command=True)


def unitree_g1_flat_amp_cond_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Task reward only, conditional discriminator features, optional RSI.

  RSI is enabled at runtime through ``MJLAB_AMP_RSI_FILE``/``MJLAB_AMP_RSI_PROB``
  (see :func:`reset_from_expert_states`); it runs after the default resets.
  """
  cfg = unitree_g1_flat_amp_env_cfg(play=play)
  cfg.observations["amp"] = _conditional_group()
  return cfg


def unitree_g1_flat_expert_cond_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Original rewards with the conditional features (expert data)."""
  cfg = unitree_g1_flat_expert_env_cfg(play=play)
  cfg.observations["amp"] = _conditional_group()
  return cfg
