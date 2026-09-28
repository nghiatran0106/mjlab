"""PPO with an adversarial style reward learned from expert transitions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn

from mjlab.tasks.velocity_amp.rl.discriminator import (
  Discriminator,
  ExpertBuffer,
  LossType,
  TransitionReplayBuffer,
)


@dataclass
class AmpCfg:
  expert_file: str = ""
  """Path to an ``.npz`` with expert ``s``/``s_next``. Empty disables the style
  reward (e.g. when only loading a checkpoint for playback)."""
  obs_group: str = "amp"
  """Observation group holding the discriminator features."""
  loss_type: LossType = "amp"
  style_reward_weight: float = 2.0
  """Weight of the style reward, multiplied by ``step_dt`` like the task terms."""
  step_dt: float = 0.02
  hidden_dims: tuple[int, ...] = (1024, 512)
  learning_rate: float = 1e-4
  batch_size: int = 4096
  num_updates: int = 10
  """Discriminator gradient steps per PPO iteration."""
  grad_penalty_weight: float = 10.0
  max_grad_norm: float = 1.0
  task_reward_lerp: float = -1.0
  """If in [0, 1], the reward is ``lerp * task + (1 - lerp) * style`` (as in
  Escontrela et al. 2022, who use 0.3) instead of ``task + style``."""
  replay_size: int = 0
  """Capacity of the policy-transition replay buffer; 0 trains the
  discriminator on the latest rollout only."""


class AmpPPO(PPO):
  """PPO whose per-step reward is ``task + w * dt * style(s, s')``.

  The style term comes from a discriminator trained to tell expert transitions
  from the policy's own, which makes it a learned (inverse-RL) reward.
  """

  def __init__(self, *args, amp_cfg: dict | None = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.amp_cfg = AmpCfg(**(amp_cfg or {}))
    cfg = self.amp_cfg

    obs_dim = self.storage.observations[cfg.obs_group].shape[-1]
    self.discriminator = Discriminator(obs_dim, cfg.hidden_dims, cfg.loss_type).to(
      self.device
    )
    self.disc_optimizer = torch.optim.Adam(
      self.discriminator.parameters(), lr=cfg.learning_rate
    )

    self.expert: ExpertBuffer | None = None
    if cfg.expert_file and Path(cfg.expert_file).exists():
      self.expert = ExpertBuffer(cfg.expert_file, self.device)
      if self.expert.obs_dim != obs_dim:
        raise ValueError(
          f"Expert features have dim {self.expert.obs_dim}, "
          f"but observation group '{cfg.obs_group}' has dim {obs_dim}."
        )
      self.discriminator.fit_normalizer(self.expert.s)
    else:
      warnings.warn(
        f"AMP expert file '{cfg.expert_file}' not found; style reward disabled.",
        stacklevel=2,
      )

    self.replay: TransitionReplayBuffer | None = None
    if cfg.replay_size > 0:
      self.replay = TransitionReplayBuffer(cfg.replay_size, obs_dim, self.device)

    self._amp_prev: torch.Tensor | None = None
    self._policy_s: list[torch.Tensor] = []
    self._policy_s_next: list[torch.Tensor] = []
    self._style_reward_sum = 0.0
    self._style_reward_count = 0

  @property
  def style_enabled(self) -> bool:
    return self.expert is not None and self.amp_cfg.style_reward_weight > 0.0

  def act(self, obs: TensorDict) -> torch.Tensor:
    self._amp_prev = obs[self.amp_cfg.obs_group].clone()
    return super().act(obs)

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    if self.style_enabled and self._amp_prev is not None:
      s = self._amp_prev
      s_next = obs[self.amp_cfg.obs_group]
      assert isinstance(s_next, torch.Tensor)
      # After a reset, s_next belongs to a new episode, so (s, s_next) is not a
      # real transition: give it no style reward and keep it out of training.
      valid = dones.view(-1) == 0
      style = self.discriminator.style_reward(s, s_next) * valid
      style_term = self.amp_cfg.style_reward_weight * self.amp_cfg.step_dt * style
      lerp = self.amp_cfg.task_reward_lerp
      if 0.0 <= lerp <= 1.0:
        rewards = lerp * rewards + (1.0 - lerp) * style_term
      else:
        rewards = rewards + style_term
      self._policy_s.append(s[valid])
      self._policy_s_next.append(s_next[valid].clone())
      self._style_reward_sum += style.sum().item()
      self._style_reward_count += style.numel()
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    loss_dict = super().update()
    if self.style_enabled and self._policy_s:
      loss_dict.update(self._update_discriminator())
      loss_dict["amp_style_reward"] = self._style_reward_sum / max(
        self._style_reward_count, 1
      )
    self._policy_s.clear()
    self._policy_s_next.clear()
    self._style_reward_sum = 0.0
    self._style_reward_count = 0
    return loss_dict

  def _update_discriminator(self) -> dict[str, float]:
    assert self.expert is not None
    cfg = self.amp_cfg
    # Clone to leave inference mode: rollout tensors cannot be used in autograd.
    policy_s = torch.cat(self._policy_s).clone()
    policy_s_next = torch.cat(self._policy_s_next).clone()
    if self.replay is not None:
      self.replay.insert(policy_s, policy_s_next)
    pool = len(self.replay) if self.replay is not None else policy_s.shape[0]
    batch_size = min(cfg.batch_size, pool)
    totals: dict[str, float] = {}
    for _ in range(cfg.num_updates):
      if self.replay is not None:
        batch_s, batch_s_next = self.replay.sample(batch_size)
      else:
        idx = torch.randint(0, policy_s.shape[0], (batch_size,), device=self.device)
        batch_s, batch_s_next = policy_s[idx], policy_s_next[idx]
      expert_s, expert_s_next = self.expert.sample(batch_size)
      loss, stats = self.discriminator.loss(
        expert_s,
        expert_s_next,
        batch_s,
        batch_s_next,
        cfg.grad_penalty_weight,
      )
      self.disc_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), cfg.max_grad_norm)
      self.disc_optimizer.step()
      for k, v in stats.items():
        totals[k] = totals.get(k, 0.0) + v / cfg.num_updates
    return totals

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def save(self) -> dict:
    saved = super().save()
    saved["discriminator_state_dict"] = self.discriminator.state_dict()
    saved["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    wants_disc = load_cfg is None or load_cfg.get("discriminator", False)
    if wants_disc and "discriminator_state_dict" in loaded_dict:
      self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
      self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
    return load_iteration
