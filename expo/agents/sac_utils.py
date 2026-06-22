from __future__ import annotations

import torch
import torch.nn as nn


class AutoTunedAlpha(nn.Module):
    """SAC-style auto-tuned entropy temperature.

    Parameterizes `α = exp(log_alpha)` so α stays positive without clamping.
    The loss pushes α up when entropy < target, down when entropy > target.
    """

    def __init__(self, target_entropy: float, init_log_alpha: float = 0.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(float(init_log_alpha)))
        self.target_entropy = float(target_entropy)

    @property
    def value(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        # Minimize:  -log_alpha · (log_prob + target_entropy).detach()
        # → if log_prob + target_entropy < 0 (entropy too high), drive α down.
        return -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging:  target ← (1 - τ)·target + τ·source.

    With τ=0.005 the target tracks the source slowly, as in SAC/REDQ.
    See `DECISIONS.md §8` for note on the paper's flipped notation.
    """
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
