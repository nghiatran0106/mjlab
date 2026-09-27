"""Tests for the AMP discriminator used by the velocity_amp task."""

import numpy as np
import pytest
import torch

from mjlab.tasks.velocity_amp.config.g1.env_cfgs import (
  GAIT_HISTORY,
  unitree_g1_flat_expert_gait_env_cfg,
)
from mjlab.tasks.velocity_amp.mdp import amp_observation_group
from mjlab.tasks.velocity_amp.rl.discriminator import Discriminator, ExpertBuffer


@pytest.mark.parametrize("loss_type", ["amp", "gail"])
def test_discriminator_separates_expert_from_policy(loss_type):
  torch.manual_seed(0)
  dim = 6
  expert = torch.randn(2048, dim) + 2.0
  policy = torch.randn(2048, dim) - 2.0
  disc = Discriminator(dim, hidden_dims=(64, 64), loss_type=loss_type)
  disc.fit_normalizer(expert)
  opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
  for _ in range(200):
    loss, _ = disc.loss(expert, expert, policy, policy, grad_penalty_weight=1.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
  r_expert = disc.style_reward(expert, expert).mean()
  r_policy = disc.style_reward(policy, policy).mean()
  assert r_expert > r_policy + 0.5


def test_amp_style_reward_is_bounded():
  disc = Discriminator(4, hidden_dims=(16,), loss_type="amp")
  s = torch.randn(512, 4) * 100.0
  r = disc.style_reward(s, s)
  assert r.shape == (512,)
  assert torch.all((r >= 0.0) & (r <= 1.0))


def test_expert_buffer_roundtrip(tmp_path):
  s = np.random.randn(100, 5).astype(np.float32)
  path = tmp_path / "expert.npz"
  np.savez(path, s=s, s_next=s + 1.0)
  buf = ExpertBuffer(path, device="cpu")
  assert len(buf) == 100 and buf.obs_dim == 5
  bs, bsn = buf.sample(32)
  assert torch.allclose(bsn - bs, torch.ones_like(bs))


def test_gait_features_add_feet_and_history():
  base = amp_observation_group()
  gait = amp_observation_group(key_bodies=True, feet=True, history=GAIT_HISTORY)
  assert set(gait.terms) - set(base.terms) == {
    "key_body_pos",
    "foot_height",
    "foot_contact",
  }
  assert gait.history_length == GAIT_HISTORY
  assert base.history_length is None


def test_stepping_expert_rewards_air_time_with_fixed_commands():
  cfg = unitree_g1_flat_expert_gait_env_cfg()
  assert cfg.rewards["air_time"].weight > 0
  assert cfg.rewards["foot_swing_height"].weight < -0.25
  assert "command_vel" not in cfg.curriculum
