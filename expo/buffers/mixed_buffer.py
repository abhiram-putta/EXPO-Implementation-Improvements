from __future__ import annotations

import numpy as np
import torch

from .replay_buffer import ReplayBuffer


class OfflineBuffer:
    """Read-only buffer for pre-collected transitions.

    Tolerates `(s, a)`-only data: when rewards/next_states/dones are missing,
    they default to (0, s, 1). In that mode the offline data should only be
    used for IL pretraining, *not* for the critic loss — caller's
    responsibility.
    """

    def __init__(self, states, actions, rewards=None, next_states=None,
                 dones=None, device: str | torch.device = "cpu",
                 provide_discount_gamma: float | None = None):
        """`provide_discount_gamma`: if set, sample() includes 'discount' = γ
        (since offline data is flat single transitions, no n-step possible).
        Set this when using NStepBuffer for online so MixedBuffer concat
        works."""
        self.device = torch.device(device)
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        n = len(self.states)
        self.has_full_transitions = rewards is not None
        if self.has_full_transitions:
            self.rewards = np.asarray(rewards, dtype=np.float32).reshape(n, 1)
            self.next_states = np.asarray(next_states, dtype=np.float32)
            self.dones = np.asarray(dones, dtype=np.float32).reshape(n, 1)
        else:
            self.rewards = np.zeros((n, 1), dtype=np.float32)
            self.next_states = self.states.copy()
            self.dones = np.ones((n, 1), dtype=np.float32)
        self._gamma = provide_discount_gamma

    def __len__(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self.states), size=batch_size)
        out = {
            "state": torch.as_tensor(self.states[idx], device=self.device),
            "action": torch.as_tensor(self.actions[idx], device=self.device),
            "reward": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_state": torch.as_tensor(self.next_states[idx], device=self.device),
            "done": torch.as_tensor(self.dones[idx], device=self.device),
        }
        if self._gamma is not None:
            out["discount"] = torch.full_like(out["reward"], float(self._gamma))
        return out


class MixedBuffer:
    """Samples a fixed-ratio mix of offline + online (RLPD-style).

    Default 50/50. If the online buffer can't satisfy its share, the
    deficit is drawn from offline (so callers can use this buffer
    immediately at startup).
    """

    def __init__(self, offline: OfflineBuffer, online: ReplayBuffer,
                 online_ratio: float = 0.5):
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in [0, 1], got {online_ratio}")
        self.offline = offline
        self.online = online
        self.online_ratio = online_ratio

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n_online_target = int(round(batch_size * self.online_ratio))
        n_online = min(n_online_target, len(self.online))
        n_offline = batch_size - n_online

        if n_online == 0:
            return self.offline.sample(n_offline)
        if n_offline == 0:
            return self.online.sample(n_online)

        off = self.offline.sample(n_offline)
        on = self.online.sample(n_online)
        return {k: torch.cat([off[k], on[k]], dim=0) for k in off}
