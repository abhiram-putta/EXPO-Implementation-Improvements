from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.critic import QEnsemble
from ..models.diffusion_policy import DiffusionPolicy
from ..models.edit_policy import EditPolicy
from .sac_utils import AutoTunedAlpha, soft_update


@dataclass
class EXPOConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    diffusion_blocks: int = 3
    diffusion_steps: int = 10
    edit_layers: int = 3
    edit_dropout: float = 0.0
    beta: float = 0.05
    ensemble_size: int = 10
    num_min_q: int = 2
    n_otf_samples: int = 8
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    edit_grad_clip: float = 1.0
    delayed_edit_steps: int = 5000
    # If None, target_entropy is auto-derived to account for the [-β, β]
    # squashing: the standard SAC default of -action_dim assumes uniform
    # is on [-1, 1]; when β < 1 the achievable max entropy shrinks by
    # action_dim · log(2β), so we shift target accordingly to keep auto-α
    # from running away. See DECISIONS.md §15.
    target_entropy: float | None = None


class EXPOAgent:
    """Expressive Policy Optimization agent.

    Three networks:
      - `base_policy`  — diffusion DDPM (IL-trained, never sees Q-grads)
      - `edit_policy`  — squashed-Gaussian (SAC-style, max Q − α H)
      - `critic`       — Q-ensemble with REDQ-style random-min-2

    Action selection uses on-the-fly (OTF) extraction: sample N base
    actions, edit each, evaluate the 2N candidates with the critic, return
    the argmax. OTF is used both for env action selection and for the TD
    target's `a*'`.
    """

    def __init__(self, cfg: EXPOConfig, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.base_policy = DiffusionPolicy(
            state_dim=cfg.state_dim, action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim, num_blocks=cfg.diffusion_blocks,
            num_timesteps=cfg.diffusion_steps,
        ).to(self.device)

        self.edit_policy = EditPolicy(
            state_dim=cfg.state_dim, action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.edit_layers,
            beta=cfg.beta, dropout=cfg.edit_dropout,
        ).to(self.device)

        self.critic = QEnsemble(
            state_dim=cfg.state_dim, action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            ensemble_size=cfg.ensemble_size, num_min_q=cfg.num_min_q,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Adaptive target entropy — see _compute_target_entropy for math.
        # When β changes (progressive β scheduler), call set_beta() which
        # recomputes target_entropy in place.
        target_entropy = self._compute_target_entropy(cfg.beta)
        self.alpha = AutoTunedAlpha(target_entropy=target_entropy).to(self.device)

        self.opt_base = torch.optim.Adam(self.base_policy.parameters(), lr=cfg.lr)
        self.opt_edit = torch.optim.Adam(self.edit_policy.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_alpha = torch.optim.Adam([self.alpha.log_alpha], lr=cfg.lr)

    def _compute_target_entropy(self, beta: float) -> float:
        """SAC's -action_dim assumes actions in [-1, 1]. With tanh squash to
        [-β, β], achievable max entropy shrinks by action_dim · log(2β); shift
        target accordingly. Without this, α inflates trying to reach an
        unachievable entropy. If user provided cfg.target_entropy, honor it."""
        if self.cfg.target_entropy is not None:
            return float(self.cfg.target_entropy)
        sac_default = -float(self.cfg.action_dim)
        if beta > 0:
            squash_aware = sac_default + self.cfg.action_dim * math.log(2.0 * beta)
            return float(min(sac_default, squash_aware))
        return float(sac_default)

    def set_beta(self, beta: float) -> None:
        """Update edit policy's β and recompute α target entropy.
        Called by the training loop's progressive-β scheduler."""
        self.edit_policy.beta = float(beta)
        self.alpha.target_entropy = self._compute_target_entropy(beta)

    # ------------------------------------------------------------------ #
    # OTF action selection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def otf_action(self, state: torch.Tensor, critic: QEnsemble | None = None,
                   N: int | None = None, allow_edits: bool = True) -> torch.Tensor:
        """Vectorized OTF: argmax_a Q(s, a) over N base + N edited candidates.

        - `state`: [B, S]
        - `critic`: which critic to evaluate against (default: target_critic)
        - `N`: number of base samples (default: cfg.n_otf_samples)
        - `allow_edits`: if False, only N base samples are considered
          (used during the edit-policy warmup window)

        Returns: [B, A]. Also exposes a `last_frac_edited` attribute on the
        agent for diagnostics — the fraction of selected actions that came
        from the edited candidate set.
        """
        critic = critic if critic is not None else self.target_critic
        N = N if N is not None else self.cfg.n_otf_samples
        B, S = state.shape
        A = self.cfg.action_dim

        # Repeat each state N times → [B*N, S]
        s_rep = state.unsqueeze(1).expand(B, N, S).reshape(B * N, S)

        base_acts = self.base_policy.sample(s_rep)  # [B*N, A]

        if allow_edits:
            edit = self.edit_policy.sample(s_rep, base_acts)  # [B*N, A]
            edited_acts = (base_acts + edit).clamp(-1.0, 1.0)
            # Group=2 (base, edited); flatten so we can argmax in one shot
            all_acts = torch.cat([base_acts, edited_acts], dim=0)   # [2*B*N, A]
            all_states = torch.cat([s_rep, s_rep], dim=0)           # [2*B*N, S]
            num_groups = 2
        else:
            all_acts = base_acts
            all_states = s_rep
            num_groups = 1

        q = critic.min_q(all_states, all_acts)  # [num_groups*B*N, 1]

        # Reshape so candidates for a given batch element are contiguous.
        # Layout in `all_acts` is [group, batch_in_rep] — we permute to
        # [B, group, N, A] and flatten the last two dims.
        candidates = num_groups * N
        all_acts_view = (
            all_acts.view(num_groups, B, N, A)
                    .permute(1, 0, 2, 3)
                    .reshape(B, candidates, A)
        )
        q_view = (
            q.view(num_groups, B, N)
             .permute(1, 0, 2)
             .reshape(B, candidates)
        )

        best_idx = q_view.argmax(dim=1)  # [B]
        # diagnostic: fraction of picks that landed in the second half (edited)
        if allow_edits:
            self.last_frac_edited = float((best_idx >= N).float().mean().item())
        else:
            self.last_frac_edited = 0.0

        batch_arange = torch.arange(B, device=self.device)
        return all_acts_view[batch_arange, best_idx]

    # ------------------------------------------------------------------ #
    # Update routines
    # ------------------------------------------------------------------ #

    def critic_update(self, batch: dict, allow_edits: bool) -> dict:
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        sp = batch["next_state"]
        d = batch["done"]
        # Per-sample discount: provided by NStepBuffer (γ^k where k = actual
        # n-step chain length). Falls back to fixed γ for plain buffers.
        discount = batch.get("discount", None)
        if discount is None:
            discount = torch.full_like(r, self.cfg.gamma)

        with torch.no_grad():
            next_action = self.otf_action(
                sp, critic=self.target_critic, allow_edits=allow_edits,
            )
            target_q = self.target_critic.min_q(sp, next_action, random_subset=True)
            td_target = r + (1.0 - d) * discount * target_q

        all_q = self.critic.all_q(s, a)  # [E, B, 1]
        td_target_exp = td_target.unsqueeze(0).expand_as(all_q)
        critic_loss = F.mse_loss(all_q, td_target_exp)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        soft_update(self.target_critic, self.critic, self.cfg.tau)

        return {
            "critic/loss": critic_loss.item(),
            "critic/mean_q": all_q.mean().item(),
            "critic/q_ensemble_std": all_q.std(dim=0).mean().item(),
            "critic/td_target_mean": td_target.mean().item(),
        }

    def base_policy_update(self, batch: dict) -> dict:
        loss = self.base_policy.diffusion_loss(batch["state"], batch["action"])
        self.opt_base.zero_grad()
        loss.backward()
        self.opt_base.step()
        return {"base/il_loss": loss.item()}

    def edit_policy_update(self, batch: dict) -> dict:
        s = batch["state"]
        with torch.no_grad():
            base_acts = self.base_policy.sample(s)

        edit, log_prob = self.edit_policy(s, base_acts)
        edited = (base_acts + edit).clamp(-1.0, 1.0)

        # min over a random subset of the *live* critic (not target)
        q_edited = self.critic.min_q(s, edited, random_subset=True)
        alpha = self.alpha.value.detach()
        edit_loss = -(q_edited - alpha * log_prob).mean()

        self.opt_edit.zero_grad()
        edit_loss.backward()
        nn.utils.clip_grad_norm_(self.edit_policy.parameters(), self.cfg.edit_grad_clip)
        self.opt_edit.step()

        # auto-tune α
        alpha_loss = self.alpha.loss(log_prob)
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        return {
            "edit/loss": edit_loss.item(),
            "edit/mean_edit_magnitude": edit.abs().mean().item(),
            "edit/log_prob": log_prob.mean().item(),
            "edit/alpha": self.alpha.value.item(),
            "edit/alpha_loss": alpha_loss.item(),
        }

    def update(self, sampler: Callable[[int], dict], batch_size: int,
               utd: int, env_step: int) -> dict:
        """One env-step's worth of updates: G critic updates, then 1 base, 1 edit.

        Returns the metrics from the final update of each kind (so the logger
        sees one row of each metric per env step at most).
        """
        edits_active = env_step >= self.cfg.delayed_edit_steps
        metrics: dict = {}
        batch = None
        for _ in range(utd):
            batch = sampler(batch_size)
            metrics.update(self.critic_update(batch, allow_edits=edits_active))

        # Base policy IL on the last batch (mixed offline+online)
        metrics.update(self.base_policy_update(batch))

        # Edit policy (skipped during warmup)
        if edits_active:
            metrics.update(self.edit_policy_update(batch))
        else:
            metrics["edit/loss"] = float("nan")
            metrics["edit/alpha"] = self.alpha.value.item()

        return metrics
