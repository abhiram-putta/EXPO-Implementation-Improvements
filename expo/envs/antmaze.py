"""D4RL Antmaze wrapper backed by Minari.

The raw env returns a Dict observation:
  - observation:    27D = qpos[2:] (joint angles, EXCLUDES root XY) + qvel[14]
  - achieved_goal:   2D = ant body XY  ← the *only* place body XY appears
  - desired_goal:    2D = target XY

An earlier version of this wrapper dropped achieved_goal under the false
assumption that it was a function of `observation`. It isn't — gymnasium-
robotics' AntMaze runs the underlying Ant with `exclude_current_positions
_from_observation=True`, so root XY is removed from `observation` and lives
only in `achieved_goal`. Without it, the policy is navigating blind.

We flatten to `[observation; desired_goal − achieved_goal]` (29D). The
goal-relative encoding is translation-invariant (the policy sees "the goal
is 14 units northeast" instead of "I am at (-5,-3) and the goal is at
(9,11)") which makes the diffusion sample-efficient across maze positions.

Action is in [-1, 1]^8, no rescaling needed.
"""
from __future__ import annotations

import minari
import numpy as np


def _flatten_obs(obs: dict) -> np.ndarray:
    relative_goal = obs["desired_goal"] - obs["achieved_goal"]
    return np.concatenate(
        [obs["observation"], relative_goal],
        axis=-1,
    ).astype(np.float32)


class AntmazeEnv:
    """Thin wrapper that flattens the dict obs into a 29D vector."""

    def __init__(self, dataset_id: str, seed: int | None = None):
        self._dataset = minari.load_dataset(dataset_id)
        self.env = self._dataset.recover_environment()
        self.action_dim = int(self.env.action_space.shape[0])
        # state_dim = observation + desired_goal
        sample_obs, _ = self.env.reset(seed=seed)
        self.state_dim = int(_flatten_obs(sample_obs).shape[0])
        self._seed = seed
        self._just_seeded = True  # avoid re-seeding on first reset
        self._first_obs = sample_obs

    def reset(self) -> np.ndarray:
        if self._just_seeded:
            self._just_seeded = False
            return _flatten_obs(self._first_obs)
        obs, _ = self.env.reset()
        return _flatten_obs(obs)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self.env.step(a)
        return _flatten_obs(obs), float(reward), bool(terminated), bool(truncated)

    def close(self) -> None:
        self.env.close()

    @property
    def dataset(self):
        return self._dataset
