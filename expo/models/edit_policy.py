from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class EditPolicy(nn.Module):
    """Squashed-Gaussian edit policy (Implementation Guide §10).

    Inputs (state, base_action) → outputs an edit â ∈ [-β, β]^A and its
    log-probability under the squashed Gaussian. The squash is
    `edit = β · tanh(u)` where `u ~ Normal(μ(s,a), σ(s,a))` (reparameterized).

    Log-prob accounts for the full change of variables
    `edit = β · tanh(u)` — i.e. subtracts `log β + log(1 - tanh(u)^2)`.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, beta: float = 0.05, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.action_dim = action_dim
        self.beta = beta

        layers: list[nn.Module] = [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _features(self, state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([state, base_action], dim=-1))

    def forward(self, state: torch.Tensor, base_action: torch.Tensor):
        """Reparameterized sample. Returns (edit, log_prob)."""
        h = self._features(state, base_action)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        edit = self.beta * torch.tanh(u)

        # log p(edit) = log p(u) - log|d edit/du|
        # |d edit/du| = β · (1 - tanh(u)^2)
        log_prob_u = normal.log_prob(u)
        # Numerically stable log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        log_one_minus_tanh_sq = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_det_jac = math.log(self.beta) + log_one_minus_tanh_sq
        log_prob = (log_prob_u - log_det_jac).sum(dim=-1, keepdim=True)
        return edit, log_prob

    @torch.no_grad()
    def sample(self, state: torch.Tensor, base_action: torch.Tensor,
               deterministic: bool = False) -> torch.Tensor:
        """Sample edit without tracking log-prob; used at inference time."""
        h = self._features(state, base_action)
        mean = self.mean_head(h)
        if deterministic:
            return self.beta * torch.tanh(mean)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        u = mean + log_std.exp() * torch.randn_like(mean)
        return self.beta * torch.tanh(u)
