from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..agents.expo_agent import EXPOAgent


def evaluate(agent: EXPOAgent, env: Any, num_episodes: int = 10,
             max_steps: int = 200) -> dict:
    """Roll out the OTF policy for `num_episodes` and aggregate returns.

    Works with any env exposing `reset() -> obs` and
    `step(action) -> (obs, reward, terminated, truncated)`. For sparse-reward
    tasks (Antmaze, Adroit, Robomimic, MimicGen) `success_rate` is also
    reported — counts an episode as successful if any reward >= 0.5 was
    received during the rollout.
    """
    returns: list[float] = []
    successes: list[bool] = []
    for _ in range(num_episodes):
        s = env.reset()
        ep_ret = 0.0
        ep_success = False
        for _ in range(max_steps):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            a = agent.otf_action(s_t).squeeze(0).cpu().numpy()
            s, r, terminated, truncated = env.step(a)
            ep_ret += r
            if r >= 0.5:
                ep_success = True
            if terminated or truncated:
                break
        returns.append(ep_ret)
        successes.append(ep_success)
    arr = np.asarray(returns)
    return {
        "eval/return_mean": float(arr.mean()),
        "eval/return_std": float(arr.std()),
        "eval/return_min": float(arr.min()),
        "eval/return_max": float(arr.max()),
        "eval/success_rate": float(np.mean(successes)),
    }
