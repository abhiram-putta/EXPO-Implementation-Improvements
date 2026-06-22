"""Progressive β scheduler for the edit policy.

Motivation (Sutton & Barto, Ch. 2): just as ε-greedy methods anneal
exploration over training (high ε early when value estimates are
unreliable, low ε late when the policy is near-optimal), the edit policy's
edit-distance bound β can be annealed:
  - early training: large β → big edits → broad exploration around base actions
  - late training:  small β → tight edits → fine refinement of a tuned policy

Implementation: cosine schedule from beta_start → beta_end over
decay_steps. Constant before warmup_steps and after warmup_steps + decay_steps.
"""
from __future__ import annotations

import math


class ProgressiveBeta:
    def __init__(self, beta_start: float = 0.3, beta_end: float = 0.05,
                 warmup_steps: int = 0, decay_steps: int = 50000):
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError("beta_start and beta_end must be > 0 (no edits "
                             "with β=0; pick a fixed β config instead)")
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.beta_start
        progress = (step - self.warmup_steps) / max(1, self.decay_steps)
        if progress >= 1.0:
            return self.beta_end
        # Cosine anneal: smooth start, smooth end
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.beta_end + (self.beta_start - self.beta_end) * cos
