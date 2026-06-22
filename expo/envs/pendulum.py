from __future__ import annotations

import gymnasium as gym
import numpy as np


class PendulumEnv:
    """Wrapper around gymnasium `Pendulum-v1` with normalized [-1, 1] actions.

    The underlying env has action range [-2, 2] (torque); we expose a [-1, 1]
    interface so the diffusion + edit policies can use a unit action space.

    `step()` returns `(obs, reward, terminated, truncated)` — Gymnasium's
    five-tuple is collapsed to four (info dropped). For Pendulum-v1
    `terminated` is *always False* (no terminal state); episodes only end
    via `truncated` after 200 steps.
    """

    def __init__(self, seed: int | None = None):
        self.env = gym.make("Pendulum-v1")
        self.action_low = self.env.action_space.low.astype(np.float32)
        self.action_high = self.env.action_space.high.astype(np.float32)
        self.action_dim = int(self.env.action_space.shape[0])
        self.state_dim = int(self.env.observation_space.shape[0])
        self._seed = seed

    def _scale_action(self, agent_action: np.ndarray) -> np.ndarray:
        # [-1, 1] → [low, high]
        a = np.clip(agent_action, -1.0, 1.0).astype(np.float32)
        return self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)

    def reset(self) -> np.ndarray:
        if self._seed is not None:
            obs, _ = self.env.reset(seed=self._seed)
            self._seed = None
        else:
            obs, _ = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        scaled = self._scale_action(np.asarray(action))
        obs, reward, terminated, truncated, _ = self.env.step(scaled)
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated)

    def close(self) -> None:
        self.env.close()


def collect_random_offline_data(env: PendulumEnv, num_transitions: int,
                                 seed: int = 0) -> dict[str, np.ndarray]:
    """Roll out a uniform-random policy for `num_transitions` env steps.

    Used to bootstrap the offline buffer for IL pretraining + replay mixing.
    Returns arrays in the format `OfflineBuffer` expects.
    """
    rng = np.random.default_rng(seed)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    s = env.reset()
    for _ in range(num_transitions):
        a = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
        ns, r, terminated, truncated = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(ns)
        # Bootstrap mask: True only on real terminals (Pendulum has none)
        dones.append(terminated)
        s = env.reset() if (terminated or truncated) else ns
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }
