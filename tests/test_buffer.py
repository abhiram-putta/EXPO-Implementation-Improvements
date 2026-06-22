"""Sanity tests for the replay + mixed buffers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from expo.buffers.mixed_buffer import MixedBuffer, OfflineBuffer
from expo.buffers.replay_buffer import ReplayBuffer


def test_replay_buffer_shapes():
    buf = ReplayBuffer(capacity=100, state_dim=3, action_dim=2)
    assert len(buf) == 0
    for _ in range(50):
        buf.add(
            state=np.zeros(3, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=1.0,
            next_state=np.ones(3, dtype=np.float32),
            done=False,
        )
    assert len(buf) == 50
    batch = buf.sample(8)
    assert batch["state"].shape == (8, 3)
    assert batch["action"].shape == (8, 2)
    assert batch["reward"].shape == (8, 1)
    assert batch["next_state"].shape == (8, 3)
    assert batch["done"].shape == (8, 1)
    print("test_replay_buffer_shapes: OK")


def test_replay_buffer_wraparound():
    buf = ReplayBuffer(capacity=10, state_dim=1, action_dim=1)
    for i in range(25):
        buf.add(
            state=np.array([i], dtype=np.float32),
            action=np.array([0], dtype=np.float32),
            reward=0.0,
            next_state=np.array([i + 1], dtype=np.float32),
            done=False,
        )
    assert len(buf) == 10
    # newest entries should be ids 15..24
    states = buf.states.flatten().tolist()
    assert sorted(int(s) for s in states) == list(range(15, 25))
    print("test_replay_buffer_wraparound: OK")


def test_mixed_buffer_ratio():
    rng = np.random.default_rng(0)
    n_off = 500
    offline = OfflineBuffer(
        states=rng.normal(size=(n_off, 3)).astype(np.float32),
        actions=rng.normal(size=(n_off, 2)).astype(np.float32),
        rewards=np.ones(n_off, dtype=np.float32),
        next_states=rng.normal(size=(n_off, 3)).astype(np.float32),
        dones=np.zeros(n_off, dtype=np.float32),
    )
    online = ReplayBuffer(capacity=200, state_dim=3, action_dim=2)
    # add a distinguishable marker into online actions
    for _ in range(100):
        online.add(
            state=np.zeros(3, dtype=np.float32),
            action=np.full(2, 999.0, dtype=np.float32),
            reward=0.0,
            next_state=np.zeros(3, dtype=np.float32),
            done=False,
        )
    mixed = MixedBuffer(offline, online, online_ratio=0.5)
    batch = mixed.sample(64)
    assert batch["action"].shape == (64, 2)
    # roughly half the actions should be the 999 marker
    n_online_in_batch = (batch["action"][:, 0] == 999.0).sum().item()
    assert 24 <= n_online_in_batch <= 40, f"got {n_online_in_batch}/64 from online"
    print(f"test_mixed_buffer_ratio: OK ({n_online_in_batch}/64 from online)")


def test_mixed_buffer_empty_online():
    rng = np.random.default_rng(0)
    n_off = 100
    offline = OfflineBuffer(
        states=rng.normal(size=(n_off, 3)).astype(np.float32),
        actions=rng.normal(size=(n_off, 2)).astype(np.float32),
        rewards=np.ones(n_off, dtype=np.float32),
        next_states=rng.normal(size=(n_off, 3)).astype(np.float32),
        dones=np.zeros(n_off, dtype=np.float32),
    )
    online = ReplayBuffer(capacity=100, state_dim=3, action_dim=2)
    mixed = MixedBuffer(offline, online, online_ratio=0.5)
    # online empty → all from offline
    batch = mixed.sample(32)
    assert batch["state"].shape == (32, 3)
    print("test_mixed_buffer_empty_online: OK")


if __name__ == "__main__":
    test_replay_buffer_shapes()
    test_replay_buffer_wraparound()
    test_mixed_buffer_ratio()
    test_mixed_buffer_empty_online()
    print("\nAll buffer tests passed.")
