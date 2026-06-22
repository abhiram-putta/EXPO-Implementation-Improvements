"""Convert a Minari dataset into the format `OfflineBuffer` expects.

Minari iterates over episodes; each episode has arrays of length T+1 for
observations and T for actions/rewards/terminations/truncations. We unroll
them into flat (s, a, r, s', done) transitions. `done` is set from
`terminated` only (not truncated) so bootstrapping is correct.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def minari_to_transitions(
    dataset,
    obs_flatten: Callable[[dict | np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Walk every episode in `dataset` and concatenate transitions.

    Args:
        dataset: a `MinariDataset`.
        obs_flatten: callable that turns a dict observation (or np.ndarray)
            into a flat 1-D feature vector for ONE timestep. Will be applied
            to every timestep of every episode.

    Returns dict of arrays: states, actions, rewards, next_states, dones.
    """
    states_chunks: list[np.ndarray] = []
    actions_chunks: list[np.ndarray] = []
    rewards_chunks: list[np.ndarray] = []
    next_states_chunks: list[np.ndarray] = []
    dones_chunks: list[np.ndarray] = []

    for ep in dataset.iterate_episodes():
        obs = ep.observations
        # Convert dict-of-arrays to per-timestep dicts → flatten each
        if isinstance(obs, dict):
            T_plus_1 = len(next(iter(obs.values())))
            flat = np.stack([
                obs_flatten({k: v[t] for k, v in obs.items()})
                for t in range(T_plus_1)
            ], axis=0)
        else:
            flat = np.asarray(obs, dtype=np.float32)

        # actions/rewards/terminations have length T (one less than observations)
        T = len(ep.actions)
        states_chunks.append(flat[:T].astype(np.float32))
        next_states_chunks.append(flat[1:T + 1].astype(np.float32))
        actions_chunks.append(np.asarray(ep.actions, dtype=np.float32))
        rewards_chunks.append(np.asarray(ep.rewards, dtype=np.float32))
        dones_chunks.append(np.asarray(ep.terminations, dtype=np.float32))

    return {
        "states": np.concatenate(states_chunks, axis=0),
        "actions": np.concatenate(actions_chunks, axis=0),
        "rewards": np.concatenate(rewards_chunks, axis=0),
        "next_states": np.concatenate(next_states_chunks, axis=0),
        "dones": np.concatenate(dones_chunks, axis=0),
    }
