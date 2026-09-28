"""Discriminator features for AMP.

The same observation group is computed for the expert rollouts and for the
policy being trained, so both sides of the discriminator see identical
features by construction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


# Feet and hands, as in the AMP paper's "key body" features.
G1_KEY_BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def key_body_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Key body positions relative to the root, in the root frame (flattened)."""
  asset: Entity = env.scene[asset_cfg.name]
  rel_w = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    - asset.data.root_link_pos_w[:, None, :]
  )
  quat = asset.data.root_link_quat_w[:, None, :].expand(-1, rel_w.shape[1], -1)
  return quat_apply_inverse(quat, rel_w).reshape(rel_w.shape[0], -1)


def amp_observation_group(
  key_bodies: bool = False,
  feet: bool = False,
  history: int = 0,
  contact: bool = True,
  command: bool = False,
) -> ObservationGroupCfg:
  """Noise-free proprioceptive state used by the discriminator.

  With ``key_bodies``, feet and hand positions are appended so the
  discriminator can judge foot placement and arm posture directly. With
  ``feet``, foot clearance and contact flags are appended. ``history`` stacks
  the last ``history`` frames so the discriminator sees part of a gait cycle
  (a single 20 ms step cannot tell short fast steps from long ones).
  ``contact=False`` drops the binary contact flags, which let the
  discriminator separate expert from policy too easily. ``command`` appends
  the velocity command so the discriminator judges the gait *for that
  command* (a conditional discriminator) instead of the average gait.
  """
  group = ObservationGroupCfg(
    terms={
      "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
      "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
      "root_height": ObservationTermCfg(func=root_height),
      "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
      "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  if key_bodies:
    group.terms["key_body_pos"] = ObservationTermCfg(
      func=key_body_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=G1_KEY_BODIES)},
    )
  if feet:
    group.terms["foot_height"] = ObservationTermCfg(
      func=velocity_mdp.foot_height, params={"sensor_name": "foot_height_scan"}
    )
    if contact:
      group.terms["foot_contact"] = ObservationTermCfg(
        func=velocity_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
      )
  if command:
    group.terms["command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "twist"}
    )
  if history > 0:
    group.history_length = history
  return group


# Reference state initialization (RSI) --------------------------------------

RSI_FILE_ENV = "MJLAB_AMP_RSI_FILE"
RSI_PROB_ENV = "MJLAB_AMP_RSI_PROB"
_rsi_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def save_expert_states(path: Path, step_states: list[dict[str, torch.Tensor]]) -> None:
  """Write states gathered with :func:`snapshot_state` to ``path``."""
  arrays = {
    k: torch.cat([d[k] for d in step_states]).cpu().numpy().astype(np.float32)
    for k in step_states[0]
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **arrays)


def snapshot_state(robot: Entity) -> dict[str, torch.Tensor]:
  """Yaw-free robot state: height, orientation, body-frame root velocities and
  joint state. Enough to restart the simulation from this configuration."""
  data = robot.data
  return {
    "height": data.root_link_pos_w[:, 2:3].clone(),
    "quat": data.root_link_quat_w.clone(),
    "lin_vel_b": data.root_link_lin_vel_b.clone(),
    "ang_vel_b": data.root_link_ang_vel_b.clone(),
    "joint_pos": data.joint_pos.clone(),
    "joint_vel": data.joint_vel.clone(),
  }


def reset_from_expert_states(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Start a fraction of episodes from recorded expert states (RSI, as in AMP).

  Configured through ``MJLAB_AMP_RSI_FILE`` (``.npz`` from ``collect_expert
  --states-out``) and ``MJLAB_AMP_RSI_PROB``; a no-op when either is unset,
  so the default reset applies.
  """
  path = os.environ.get(RSI_FILE_ENV, "")
  prob = float(os.environ.get(RSI_PROB_ENV, "0"))
  if not path or prob <= 0.0:
    return
  key = (path, str(env.device))
  if key not in _rsi_cache:
    raw = np.load(path)
    _rsi_cache[key] = {k: torch.as_tensor(raw[k], device=env.device) for k in raw.files}
  states = _rsi_cache[key]

  env_ids = resolve_env_ids(env, env_ids)
  pick = torch.rand(len(env_ids), device=env.device) < prob
  ids = env_ids[pick]
  if len(ids) == 0:
    return
  idx = torch.randint(0, states["quat"].shape[0], (len(ids),), device=env.device)
  quat = states["quat"][idx]
  pos = env.scene.env_origins[ids].clone()
  pos[:, 2] += states["height"][idx, 0]
  lin_w = quat_apply(quat, states["lin_vel_b"][idx])
  ang_w = quat_apply(quat, states["ang_vel_b"][idx])

  asset: Entity = env.scene[asset_cfg.name]
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
  asset.write_root_link_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1), env_ids=ids)
  asset.write_joint_state_to_sim(
    states["joint_pos"][idx], states["joint_vel"][idx], env_ids=ids
  )
