from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional/timestep embedding (Vaswani-style).

    Produces a `dim`-dimensional vector for a [B] tensor of timesteps.
    `dim` must be even.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbed requires even dim, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Two-layer residual block: Linear → LN → Mish → Linear → LN → Mish + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln1(self.fc1(x)))
        h = self.act(self.ln2(self.fc2(h)))
        return x + h
