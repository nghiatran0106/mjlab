from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity_amp.rl import AmpOnPolicyRunner, TaskOnlyOnPolicyRunner

from .env_cfgs import (
  VARIANTS,
  apply_variant,
  unitree_g1_flat_amp_cond_env_cfg,
  unitree_g1_flat_amp_env_cfg,
  unitree_g1_flat_expert_cond_env_cfg,
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

# Lite gait features: foot height, 5-frame window, no contact flags.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AMP-GaitLite",
  env_cfg=unitree_g1_flat_amp_env_cfg(gait_lite=True),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True, gait_lite=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_amp"),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Expert-GaitLite",
  env_cfg=unitree_g1_flat_expert_env_cfg(gait_lite=True),
  play_env_cfg=unitree_g1_flat_expert_env_cfg(play=True, gait_lite=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Round 7: conditional discriminator (key bodies + command), optional RSI.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond",
  env_cfg=unitree_g1_flat_amp_cond_env_cfg(),
  play_env_cfg=unitree_g1_flat_amp_cond_env_cfg(play=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_velocity_amp"),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Expert-Cond",
  env_cfg=unitree_g1_flat_expert_cond_env_cfg(),
  play_env_cfg=unitree_g1_flat_expert_cond_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Reuse experiment: every variant gets the PPO baseline (hand-written reward),
# the conditional AMP/GAIL task and an evaluation task, e.g.
# Mjlab-Velocity-Flat-Unitree-G1-ExpertStep-Payload.
for _variant in VARIANTS:
  _suffix = _variant.capitalize()
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Unitree-G1-ExpertStep-{_suffix}",
    env_cfg=apply_variant(unitree_g1_flat_expert_gait_env_cfg(), _variant),
    play_env_cfg=apply_variant(
      unitree_g1_flat_expert_gait_env_cfg(play=True), _variant
    ),
    rl_cfg=replace(
      unitree_g1_ppo_runner_cfg(), experiment_name=f"g1_velocity_{_variant}_ppo"
    ),
    runner_cls=VelocityOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond-{_suffix}",
    env_cfg=apply_variant(unitree_g1_flat_amp_cond_env_cfg(), _variant),
    play_env_cfg=apply_variant(unitree_g1_flat_amp_cond_env_cfg(play=True), _variant),
    rl_cfg=replace(
      unitree_g1_ppo_runner_cfg(), experiment_name=f"g1_velocity_{_variant}_amp"
    ),
    runner_cls=AmpOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Unitree-G1-Expert-{_suffix}",
    env_cfg=apply_variant(unitree_g1_flat_expert_env_cfg(), _variant),
    play_env_cfg=apply_variant(unitree_g1_flat_expert_env_cfg(play=True), _variant),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
