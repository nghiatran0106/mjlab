"""Tests for the AMP discriminator used by the velocity_amp task."""

import numpy as np
import pytest
import torch

from mjlab.tasks.velocity_amp.config.g1.env_cfgs import (
  GAIT_HISTORY,
  unitree_g1_flat_expert_gait_env_cfg,
)
from mjlab.tasks.velocity_amp.mdp import amp_observation_group
from mjlab.tasks.velocity_amp.rl.amp_ppo import AmpCfg, AmpPPO, scheduled_lerp
from mjlab.tasks.velocity_amp.rl.discriminator import (
  Discriminator,
  ExpertBuffer,
  TransitionReplayBuffer,
)
from mjlab.tasks.velocity_amp.scripts.convergence import summarize


def _metrics(reward: float, lin_err: float, lift: float) -> dict:
  return {
    "true_reward_rate": reward,
    "lin_vel_error_m_s": lin_err,
    "swing_peak_cm": lift,
    "down_fraction": 0.0,
    "falls_per_minute": 0.0,
  }


def test_convergence_reports_first_checkpoint_meeting_target():
  results = {
    "expert_step": _metrics(5.0, 0.15, 7.5),
    "cfg_s1_it250": _metrics(4.0, 0.30, 3.0),
    "cfg_s1_it500": _metrics(4.6, 0.20, 5.5),  # first to meet the target
    "cfg_s1_it750": _metrics(4.7, 0.19, 6.0),
    "cfg_s2_it250": _metrics(3.0, 0.50, 1.0),
    "cfg_s2_it500": _metrics(3.5, 0.40, 2.0),  # never meets it
  }
  summary = summarize(results, expert="expert_step")
  assert summary["per_run"]["cfg_s1"]["iterations_to_target"] == 500
  assert summary["per_run"]["cfg_s2"]["iterations_to_target"] is None
  cfg = summary["per_config"]["cfg"]
  assert (cfg["seeds"], cfg["reached"]) == (2, 1)
  assert cfg["iterations_to_target_all"] == [500, None]
  # A stricter reward threshold (95% of 5.0 = 4.75) is never met here.
  strict = summarize(results, expert="expert_step", min_reward_frac=0.95)
  assert strict["per_run"]["cfg_s1"]["iterations_to_target"] is None


def test_convergence_relative_to_reference_config():
  # Robot variant: the target comes from the converged PPO runs, not the expert.
  results = {
    "ppo_s1_it500": _metrics(2.0, 0.40, 2.0),
    "ppo_s1_it1000": _metrics(4.0, 0.20, 6.0),
    "ppo_s2_it1000": _metrics(4.4, 0.20, 6.0),  # reference: 4.2, 0.20, 6.0
    "bc_s1_it250": _metrics(3.9, 0.24, 5.0),  # 0.9 * 4.2 = 3.78, 0.8 * 6 = 4.8
  }
  summary = summarize(results, expert="unused", reference_config="ppo")
  assert summary["reference"]["reward"] == pytest.approx(4.2)
  assert summary["per_run"]["bc_s1"]["iterations_to_target"] == 250
  assert summary["per_run"]["ppo_s1"]["iterations_to_target"] == 1000


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


def test_replay_buffer_keeps_most_recent_transitions():
  buf = TransitionReplayBuffer(capacity=5, obs_dim=1, device="cpu")
  buf.insert(torch.arange(3.0)[:, None], torch.arange(3.0)[:, None])
  buf.insert(torch.arange(3.0, 7.0)[:, None], torch.arange(3.0, 7.0)[:, None])
  assert len(buf) == 5
  # 0 and 1 were overwritten; 2..6 remain.
  assert sorted(buf.s[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0]
  s, s_next = buf.sample(64)
  assert torch.equal(s, s_next)
  assert s.min() >= 2.0


def test_conditional_features_include_command():
  group = amp_observation_group(key_bodies=True, command=True)
  assert {"key_body_pos", "command"} <= set(group.terms)


def test_task_reward_lerp_schedule():
  # Hold at 0.9 for 250 iterations, then ramp linearly to 0.3 over 500.
  assert scheduled_lerp(0, 0.9, 0.3, 250, 500) == 0.9
  assert scheduled_lerp(250, 0.9, 0.3, 250, 500) == 0.9
  assert scheduled_lerp(500, 0.9, 0.3, 250, 500) == pytest.approx(0.6)
  assert scheduled_lerp(750, 0.9, 0.3, 250, 500) == pytest.approx(0.3)
  assert scheduled_lerp(5000, 0.9, 0.3, 250, 500) == pytest.approx(0.3)
  # No schedule: the end value (or the additive reward, -1) is used as is.
  assert scheduled_lerp(100, -1.0, 0.3, 250, 500) == 0.3
  assert scheduled_lerp(100, 0.9, -1.0, 250, 500) == -1.0


def test_action_rate_multiplier_moves_toward_target():
  # The penalty weight rises while the policy is jerkier than the expert, falls
  # when it is smoother, and stays within [0, max].
  alg = AmpPPO.__new__(AmpPPO)
  alg.amp_cfg = AmpCfg(action_rate_target=1.0, action_rate_lr=0.1, action_rate_max=0.25)
  alg.action_rate_weight = 0.1
  for cost, expected in [(2.0, 0.2), (3.0, 0.25), (0.0, 0.15), (-5.0, 0.0)]:
    alg._mean_cost_sum, alg._mean_cost_count = cost, 1
    stats = alg._update_action_rate_weight()
    assert alg.action_rate_weight == pytest.approx(expected)
    assert stats["amp_action_rate_cost"] == pytest.approx(cost)
