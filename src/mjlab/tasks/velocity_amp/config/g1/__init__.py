from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity_amp.rl import AmpOnPolicyRunner, TaskOnlyOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_amp_env_cfg,
  unitree_g1_flat_expert_env_cfg,
  unitree_g1_flat_expert_gait_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-TaskOnly",
  env_cfg=unitree_g1_flat_amp_env_cfg(),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_task_only"),
  runner_cls=TaskOnlyOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AMP",
  env_cfg=unitree_g1_flat_amp_env_cfg(),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_amp"),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Expert",
  env_cfg=unitree_g1_flat_expert_env_cfg(),
  play_env_cfg=unitree_g1_flat_expert_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody",
  env_cfg=unitree_g1_flat_amp_env_cfg(key_bodies=True),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True, key_bodies=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_amp"),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody",
  env_cfg=unitree_g1_flat_expert_env_cfg(key_bodies=True),
  play_env_cfg=unitree_g1_flat_expert_env_cfg(play=True, key_bodies=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Gait variant: discriminator sees a 0.24 s window with foot height/contact.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AMP-Gait",
  env_cfg=unitree_g1_flat_amp_env_cfg(gait=True),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True, gait=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_amp"),
  runner_cls=AmpOnPolicyRunner,
)

# Expert data source with the gait features (for any expert checkpoint).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Expert-Gait",
  env_cfg=unitree_g1_flat_expert_env_cfg(gait=True),
  play_env_cfg=unitree_g1_flat_expert_env_cfg(play=True, gait=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# New expert with clearer steps (air-time reward on, stronger swing height).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-ExpertStep",
  env_cfg=unitree_g1_flat_expert_gait_env_cfg(),
  play_env_cfg=unitree_g1_flat_expert_gait_env_cfg(play=True),
  rl_cfg=replace(
    unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_expert_step"
  ),
  runner_cls=VelocityOnPolicyRunner,
)
