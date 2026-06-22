"""Verify NStepBuffer computes n-step returns correctly across episode
boundaries and matches plain ReplayBuffer when n=1."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from expo.buffers.nstep_buffer import NStepBuffer
from expo.buffers.replay_buffer import ReplayBuffer


def test_n1_matches_plain_buffer():
    """With n=1, NStepBuffer should produce the same rewards/discounts as
    a plain buffer (where reward = r and discount = gamma)."""
    rng = np.random.default_rng(0)
    cap = 100
    n = NStepBuffer(capacity=cap, state_dim=2, action_dim=1, n_step=1, gamma=0.99)
    p = ReplayBuffer(capacity=cap, state_dim=2, action_dim=1)

    for i in range(50):
        s = rng.normal(size=2).astype(np.float32)
        a = rng.normal(size=1).astype(np.float32)
        r = float(rng.normal())
        ns = rng.normal(size=2).astype(np.float32)
        done = (i % 10 == 9)  # episode ends every 10 steps
        n.add(s, a, r, ns, done, episode_done=done)
        p.add(s, a, r, ns, done)

    np.random.seed(42)
    bn = n.sample(8)
    np.random.seed(42)
    bp = p.sample(8)

    assert torch.allclose(bn["reward"], bp["reward"]), \
        f"n=1 reward mismatch: {bn['reward']} vs {bp['reward']}"
    expected_disc = torch.full_like(bn["reward"], 0.99)
    # Where done is True, discount becomes gamma^1 still (we used 1 transition);
    # the real difference is captured by 'done' which zeros the bootstrap
    assert torch.allclose(bn["discount"], expected_disc), \
        f"n=1 discount: {bn['discount']}, expected all 0.99"
    print("test_n1_matches_plain_buffer: OK")


def test_n3_known_sequence():
    """Insert a known reward sequence and verify n-step accumulation."""
    n = NStepBuffer(capacity=100, state_dim=2, action_dim=1, n_step=3, gamma=0.5)
    # Insert one episode of 10 transitions with rewards 1, 2, 3, ..., 10
    # All within one episode (no episode_done until the last one)
    for i in range(10):
        s = np.array([float(i), 0.0], dtype=np.float32)
        a = np.array([float(i)], dtype=np.float32)
        r = float(i + 1)  # 1, 2, ..., 10
        ns = np.array([float(i + 1), 0.0], dtype=np.float32)
        done = False
        episode_done = (i == 9)  # last step is episode end (truncation)
        n.add(s, a, r, ns, done, episode_done=episode_done)

    # Force sample of index 0: 3-step return = r_0 + γ·r_1 + γ²·r_2
    #                                       = 1 + 0.5·2 + 0.25·3 = 2.75
    # Discount returned = γ^3 = 0.125
    # next_state = next_state at index 2 = [3, 0]
    np.random.seed(0)
    # We want index=0 deterministically. Hack: fill the test by directly
    # calling the loop logic. Easiest: sample many and find the one with
    # state matching what we inserted at index 0.
    batch = n.sample(64)
    # Find the row whose 'state' matches index 0's state ([0, 0])
    state_np = batch["state"].numpy()
    matches = np.where(np.all(state_np == np.array([0.0, 0.0]), axis=1))[0]
    assert len(matches) > 0, "didn't sample index 0 in 64 draws"
    row = int(matches[0])
    r0 = float(batch["reward"][row, 0])
    d0 = float(batch["discount"][row, 0])
    expected_r = 1 + 0.5 * 2 + 0.25 * 3
    assert abs(r0 - expected_r) < 1e-5, f"3-step reward at idx 0: {r0}, expected {expected_r}"
    assert abs(d0 - 0.125) < 1e-5, f"discount at idx 0: {d0}, expected 0.125"
    expected_ns = np.array([3.0, 0.0])
    actual_ns = batch["next_state"][row].numpy()
    assert np.allclose(actual_ns, expected_ns), \
        f"next_state at idx 0: {actual_ns}, expected {expected_ns}"
    print(f"test_n3_known_sequence: OK (3-step reward {r0:.4f} = 1 + 0.5·2 + 0.25·3)")


def test_episode_boundary_truncates_chain():
    """If the episode ends before n steps, chain stops at the episode end."""
    n = NStepBuffer(capacity=100, state_dim=2, action_dim=1, n_step=5, gamma=0.5)
    # Episode 1: 3 transitions, then episode_done.
    for i in range(3):
        s = np.array([0.0, float(i)], dtype=np.float32)  # episode 0 marked by [0, *]
        a = np.array([1.0], dtype=np.float32)
        r = float(i + 1)
        ns = np.array([0.0, float(i + 1)], dtype=np.float32)
        episode_done = (i == 2)  # 3rd step ends episode
        n.add(s, a, r, ns, done=False, episode_done=episode_done)
    # Episode 2: 4 transitions
    for i in range(4):
        s = np.array([1.0, float(i)], dtype=np.float32)
        a = np.array([2.0], dtype=np.float32)
        r = float(i + 10)
        ns = np.array([1.0, float(i + 1)], dtype=np.float32)
        episode_done = (i == 3)
        n.add(s, a, r, ns, done=False, episode_done=episode_done)

    # Force sample index 0 (start of episode 1, 3 transitions in this ep,
    # n_step=5 — should chain only 3 steps then stop)
    # Expected reward = 1 + 0.5·2 + 0.25·3 = 2.75 (same as 3-step example)
    # Expected discount = 0.5^3 = 0.125
    batch = n.sample(64)
    state_np = batch["state"].numpy()
    matches = np.where(np.all(state_np == np.array([0.0, 0.0]), axis=1))[0]
    assert len(matches) > 0, "didn't sample index 0 in 64 draws"
    row = int(matches[0])
    r0 = float(batch["reward"][row, 0])
    d0 = float(batch["discount"][row, 0])
    assert abs(r0 - 2.75) < 1e-5, f"truncated chain reward: {r0}, expected 2.75"
    assert abs(d0 - 0.125) < 1e-5, f"truncated chain discount: {d0}, expected 0.125"
    print(f"test_episode_boundary_truncates_chain: OK (chain stops at ep boundary)")


def test_termination_flags_done():
    """If a transition with done=True is in the chain, the n-step batch's
    'done' should be True."""
    n = NStepBuffer(capacity=100, state_dim=2, action_dim=1, n_step=3, gamma=0.5)
    # Episode of 4 transitions, transition 1 has done=True (terminal)
    for i in range(4):
        s = np.array([0.0, float(i)], dtype=np.float32)
        a = np.array([1.0], dtype=np.float32)
        r = float(i + 1)
        ns = np.array([0.0, float(i + 1)], dtype=np.float32)
        done = (i == 1)  # 2nd transition is terminal
        episode_done = done
        n.add(s, a, r, ns, done=done, episode_done=episode_done)
    # Sample index 0: chain is [step 0 (r=1, done=F), step 1 (r=2, done=T)] then stop
    # Expected reward = 1 + 0.5·2 = 2.0, discount = 0.5^2 = 0.25, done=1
    batch = n.sample(64)
    state_np = batch["state"].numpy()
    matches = np.where(np.all(state_np == np.array([0.0, 0.0]), axis=1))[0]
    assert len(matches) > 0
    row = int(matches[0])
    r0 = float(batch["reward"][row, 0])
    d0 = float(batch["discount"][row, 0])
    done0 = float(batch["done"][row, 0])
    assert abs(r0 - 2.0) < 1e-5, f"terminated chain reward: {r0}, expected 2.0"
    assert abs(d0 - 0.25) < 1e-5, f"terminated chain discount: {d0}, expected 0.25"
    assert done0 > 0.5, f"terminated chain done: {done0}, expected 1.0"
    print(f"test_termination_flags_done: OK")


if __name__ == "__main__":
    test_n1_matches_plain_buffer()
    test_n3_known_sequence()
    test_episode_boundary_truncates_chain()
    test_termination_flags_done()
    print("\nAll n-step buffer tests passed.")
