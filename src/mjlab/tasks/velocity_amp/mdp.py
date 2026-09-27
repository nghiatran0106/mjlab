"""Discriminator features for AMP.

The same observation group is computed for the expert rollouts and for the
policy being trained, so both sides of the discriminator see identical
features by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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


def amp_observation_group(key_bodies: bool = False) -> ObservationGroupCfg:
  """Noise-free proprioceptive state used by the discriminator.

  With ``key_bodies``, feet and hand positions are appended so the
  discriminator can judge foot placement and arm posture directly.
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
  return group
