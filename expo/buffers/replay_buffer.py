from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Plain ring-buffer replay for (s, a, r, s', done) transitions.

    Stores numpy arrays; converts to torch tensors at sample time. `done`
    is the *bootstrap* mask — True only on true terminal states, never on
    truncations (caller's responsibility to pass the right thing).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: str | torch.device = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done,
            episode_done: bool = False) -> None:
        """Add a transition. `episode_done` is accepted (and ignored here)
        so that the call site can be uniform across plain and n-step buffers."""
        i = self._idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "state": torch.as_tensor(self.states[idx], device=self.device),
            "action": torch.as_tensor(self.actions[idx], device=self.device),
            "reward": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_state": torch.as_tensor(self.next_states[idx], device=self.device),
            "done": torch.as_tensor(self.dones[idx], device=self.device),
        }
