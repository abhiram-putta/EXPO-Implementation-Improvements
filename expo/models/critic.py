from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q-network with optional first-layer LayerNorm.

    LayerNorm-after-first-Linear is cheap insurance against feature-rank
    collapse / loss of plasticity at high UTD ratios (Improvements §4.1).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 use_layer_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class QEnsemble(nn.Module):
    """REDQ-style ensemble: keep `ensemble_size` Q-nets, take min over a random
    subset of `num_min_q` for pessimistic value estimates."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 ensemble_size: int = 10, num_min_q: int = 2,
                 use_layer_norm: bool = True):
        super().__init__()
        if num_min_q > ensemble_size:
            raise ValueError(
                f"num_min_q={num_min_q} cannot exceed ensemble_size={ensemble_size}"
            )
        self.ensemble_size = ensemble_size
        self.num_min_q = num_min_q
        self.networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, use_layer_norm)
            for _ in range(ensemble_size)
        ])

    def all_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Stack of all ensemble Q-values. Shape: [E, B, 1]."""
        return torch.stack([net(state, action) for net in self.networks], dim=0)

    def min_q(self, state: torch.Tensor, action: torch.Tensor,
              random_subset: bool = True) -> torch.Tensor:
        """Min over a (random or fixed) subset of size `num_min_q`. Shape: [B, 1]."""
        all_q = self.all_q(state, action)
        if random_subset:
            idx = torch.randperm(self.ensemble_size, device=all_q.device)[: self.num_min_q]
            subset = all_q[idx]
        else:
            subset = all_q[: self.num_min_q]
        return subset.min(dim=0).values
