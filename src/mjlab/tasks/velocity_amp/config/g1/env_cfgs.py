"""G1 flat velocity configs for the task-only and AMP experiments."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity_amp.mdp import amp_observation_group

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


def unitree_g1_flat_amp_env_cfg(
  play: bool = False, key_bodies: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.observations["amp"] = amp_observation_group(key_bodies)
  cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in TASK_REWARD_TERMS}
  if play:
    _use_training_command_ranges(cfg)
  return cfg


def unitree_g1_flat_expert_env_cfg(
  play: bool = False, key_bodies: bool = False
) -> ManagerBasedRlEnvCfg:
  """Original hand-shaped reward plus AMP features (expert data, evaluation)."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.observations["amp"] = amp_observation_group(key_bodies)
  if play:
    _use_training_command_ranges(cfg)
  return cfg
