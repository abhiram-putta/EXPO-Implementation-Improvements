"""Env factory: dispatch on env name to build (train_env, eval_env, offline_data)."""
from __future__ import annotations

from typing import Any

from .antmaze import AntmazeEnv, _flatten_obs
from .pendulum import PendulumEnv, collect_random_offline_data
from ..buffers.minari_loader import minari_to_transitions


def build_env_and_offline(cfg: dict, seed: int) -> tuple[Any, Any, dict]:
    """Returns (train_env, eval_env, offline_dict).

    `offline_dict` matches the format `OfflineBuffer` expects (states, actions,
    rewards, next_states, dones).
    """
    env_cfg = cfg["env"]
    name = env_cfg["name"]

    if name == "pendulum":
        train_env = PendulumEnv(seed=seed)
        eval_env = PendulumEnv(seed=seed + 1000)
        offline = collect_random_offline_data(
            PendulumEnv(seed=seed + 7),
            num_transitions=int(env_cfg.get("offline_transitions", 2000)),
            seed=seed + 7,
        )
        return train_env, eval_env, offline

    if name == "antmaze":
        dataset_id = env_cfg["dataset_id"]
        train_env = AntmazeEnv(dataset_id, seed=seed)
        eval_env = AntmazeEnv(dataset_id, seed=seed + 1000)
        offline = minari_to_transitions(train_env.dataset, _flatten_obs)
        return train_env, eval_env, offline

    raise ValueError(f"Unknown env name: {name}")
