"""Adversarial motion prior (AMP) discriminator and expert transition buffer.

References:
  - Ho & Ermon, "Generative Adversarial Imitation Learning" (2016).
  - Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based
    Character Control" (2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LossType = Literal["amp", "gail"]


class ExpertBuffer:
  """Uniformly samples expert (s, s') transitions stored in an ``.npz`` file.

  The file must contain two float arrays ``s`` and ``s_next`` of shape (N, D).
  """

  def __init__(self, path: str | Path, device: str) -> None:
    data = np.load(path)
    self.s = torch.as_tensor(data["s"], dtype=torch.float32, device=device)
    self.s_next = torch.as_tensor(data["s_next"], dtype=torch.float32, device=device)
    if self.s.shape != self.s_next.shape or self.s.ndim != 2:
      raise ValueError(
        f"Expected matching (N, D) arrays, got {self.s.shape} and {self.s_next.shape}"
      )

  def __len__(self) -> int:
    return self.s.shape[0]

  @property
  def obs_dim(self) -> int:
    return self.s.shape[1]

  def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, len(self), (n,), device=self.s.device)
    return self.s[idx], self.s_next[idx]


class Discriminator(nn.Module):
  """Scores state transitions (s, s'): high for expert, low for policy."""

  obs_mean: torch.Tensor
  obs_std: torch.Tensor

  def __init__(
    self,
    obs_dim: int,
    hidden_dims: tuple[int, ...] = (1024, 512),
    loss_type: LossType = "amp",
  ) -> None:
    super().__init__()
    self.loss_type = loss_type
    layers: list[nn.Module] = []
    in_dim = 2 * obs_dim
    for h in hidden_dims:
      layers += [nn.Linear(in_dim, h), nn.ReLU()]
      in_dim = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(in_dim, 1)
    # Fixed input normalization, fitted once on the expert data.
    self.register_buffer("obs_mean", torch.zeros(obs_dim))
    self.register_buffer("obs_std", torch.ones(obs_dim))

  def fit_normalizer(self, expert: torch.Tensor) -> None:
    self.obs_mean.copy_(expert.mean(dim=0))
    self.obs_std.copy_(expert.std(dim=0).clamp_min(1e-2))

  def _normalized_input(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    s = (s - self.obs_mean) / self.obs_std
    s_next = (s_next - self.obs_mean) / self.obs_std
    return torch.cat([s, s_next], dim=-1)

  def _score(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(x)).squeeze(-1)

  def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    return self._score(self._normalized_input(s, s_next))

  @torch.no_grad()
  def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    d = self(s, s_next)
    if self.loss_type == "amp":
      # Least-squares GAN targets are +1 (expert) / -1 (policy); reward in [0, 1].
      return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
    # GAIL: -log(1 - sigmoid(d)).
    return F.softplus(d)

  def loss(
    self,
    expert_s: torch.Tensor,
    expert_s_next: torch.Tensor,
    policy_s: torch.Tensor,
    policy_s_next: torch.Tensor,
    grad_penalty_weight: float,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    x_expert = self._normalized_input(expert_s, expert_s_next).requires_grad_(True)
    d_expert = self._score(x_expert)
    d_policy = self(policy_s, policy_s_next)

    if self.loss_type == "amp":
      disc_loss = (d_expert - 1.0).pow(2).mean() + (d_policy + 1.0).pow(2).mean()
    else:
      disc_loss = F.binary_cross_entropy_with_logits(
        d_expert, torch.ones_like(d_expert)
      ) + F.binary_cross_entropy_with_logits(d_policy, torch.zeros_like(d_policy))

    # Gradient penalty on expert samples keeps the discriminator smooth, which
    # keeps the style reward informative for the policy.
    (grad,) = torch.autograd.grad(d_expert.sum(), x_expert, create_graph=True)
    grad_penalty = grad.pow(2).sum(dim=-1).mean()

    loss = disc_loss + 0.5 * grad_penalty_weight * grad_penalty
    stats = {
      "amp_disc": disc_loss.item(),
      "amp_grad_pen": grad_penalty.item(),
      "amp_d_expert": d_expert.mean().item(),
      "amp_d_policy": d_policy.mean().item(),
    }
    return loss, stats
