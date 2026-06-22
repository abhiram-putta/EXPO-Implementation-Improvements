"""N-step return replay buffer.

Augments `ReplayBuffer` with episode-aware n-step target computation. From
Sutton & Barto Ch. 7: n-step TD propagates reward signal n× faster than
1-step TD, at the cost of slightly higher variance. Particularly valuable
for sparse-reward problems like Antmaze where the binary goal reward needs
to influence many earlier states.

Per-sample structure returned:
  - state, action, reward (n-step accumulated), next_state (state after n
    steps), done, discount (γ^k where k = actual chain length, may be < n
    if hit episode boundary)

The 'done' returned is True only if the n-step chain hit a TRUE termination
(self.dones[i] = 1) — episode boundaries from truncation just shorten the
chain but do not zero out bootstrap.

Caller is responsible for passing `episode_done = terminated OR truncated`
to `add()` so the buffer can detect boundaries.
"""
from __future__ import annotations

import numpy as np
import torch

from .replay_buffer import ReplayBuffer


class NStepBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 n_step: int = 1, gamma: float = 0.99,
                 device: str | torch.device = "cpu"):
        super().__init__(capacity, state_dim, action_dim, device)
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        # Track which episode each transition belongs to. Boundaries between
        # consecutive entries with different IDs delimit episodes.
        self.episode_ids = np.zeros(capacity, dtype=np.int64)
        # _next_ep is the ID assigned to the *next* `add()` call's transition.
        # Incremented on episode_done so the following transition gets a new ID.
        self._next_ep = 0

    def add(self, state, action, reward, next_state, done,
            episode_done: bool = False) -> None:
        i = self._idx
        # Parent does the actual numpy writes + cursor advancement
        super().add(state, action, reward, next_state, done)
        self.episode_ids[i] = self._next_ep
        if episode_done or bool(done):
            self._next_ep += 1

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("NStepBuffer is empty")

        idx = np.random.randint(0, self._size, size=batch_size)

        sd = self.next_states.shape[1]
        nstep_rewards = np.zeros((batch_size, 1), dtype=np.float32)
        nstep_discounts = np.full((batch_size, 1), self.gamma ** self.n_step,
                                   dtype=np.float32)
        nstep_next_states = np.zeros((batch_size, sd), dtype=np.float32)
        nstep_dones = np.zeros((batch_size, 1), dtype=np.float32)

        # Vectorize over batch using a Python loop (n is small, batch ~256
        # — well under any throughput threshold).
        full = self._size == self.capacity
        for b in range(batch_size):
            k = int(idx[b])
            ep = int(self.episode_ids[k])
            cumulative_r = 0.0
            actual_n = 0  # how many transitions we successfully chained
            terminated = False

            for j in range(self.n_step):
                step_idx = (k + j) % self.capacity

                # Stop if we'd cross the write head in a full ring buffer
                # (next entry belongs to a wrap-around — different episode
                # by construction since the writer always moves forward in
                # time).
                if j > 0 and full and step_idx == self._idx:
                    break
                # Bounds check for a not-yet-full buffer
                if not full and step_idx >= self._size:
                    break
                # Different episode means we've crossed a boundary.
                if int(self.episode_ids[step_idx]) != ep:
                    break

                cumulative_r += (self.gamma ** j) * float(self.rewards[step_idx, 0])
                actual_n = j + 1

                if float(self.dones[step_idx, 0]) > 0.5:
                    terminated = True
                    break

            nstep_rewards[b, 0] = cumulative_r
            nstep_discounts[b, 0] = self.gamma ** actual_n
            nstep_dones[b, 0] = 1.0 if terminated else 0.0
            # Bootstrap state = next_state of the LAST transition we used
            last_used = (k + actual_n - 1) % self.capacity if actual_n > 0 else k
            nstep_next_states[b] = self.next_states[last_used]

        return {
            "state": torch.as_tensor(self.states[idx], device=self.device),
            "action": torch.as_tensor(self.actions[idx], device=self.device),
            "reward": torch.as_tensor(nstep_rewards, device=self.device),
            "next_state": torch.as_tensor(nstep_next_states, device=self.device),
            "done": torch.as_tensor(nstep_dones, device=self.device),
            "discount": torch.as_tensor(nstep_discounts, device=self.device),
        }
