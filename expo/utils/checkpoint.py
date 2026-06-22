"""Save / load full training state (agent + buffer + RNG + step).

Goal: a 7-hour GPU run that dies at hour 6 should be recoverable from the
last save without losing all progress. Saves every `save_every_steps` env
steps to `<log_dir>/<run_name>.ckpt`. Resume with `--resume <ckpt_path>`.

What's saved:
  - All four network state_dicts (base, edit, critic, target_critic)
  - All four optimizer state_dicts (base, edit, critic, alpha)
  - α (just log_alpha tensor)
  - Online replay buffer (state arrays + cursor + size)
  - Numpy + torch + python RNG states
  - Current env step + episode-level state (s, ep_return, ep_success, etc.)

What's NOT saved:
  - Offline buffer (re-loaded fresh from dataset)
  - Eval env state (reset on resume)
  - Logger (a new line for resume is appended)
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(path: str | Path, *, agent, online_buffer, env_step: int,
                    episode_state: dict, extra: dict | None = None) -> None:
    """Atomic-ish save: write to .tmp, then rename. Avoids corrupt files on
    interruption mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload: dict[str, Any] = {
        "env_step": env_step,
        "episode_state": episode_state,
        "extra": extra or {},

        "base_policy": agent.base_policy.state_dict(),
        "edit_policy": agent.edit_policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "log_alpha": agent.alpha.log_alpha.detach().cpu(),
        "alpha_target_entropy": agent.alpha.target_entropy,

        "opt_base": agent.opt_base.state_dict(),
        "opt_edit": agent.opt_edit.state_dict(),
        "opt_critic": agent.opt_critic.state_dict(),
        "opt_alpha": agent.opt_alpha.state_dict(),

        # Online buffer: serialize numpy arrays + cursor
        "buffer": {
            "states": online_buffer.states,
            "actions": online_buffer.actions,
            "rewards": online_buffer.rewards,
            "next_states": online_buffer.next_states,
            "dones": online_buffer.dones,
            "_idx": online_buffer._idx,
            "_size": online_buffer._size,
        },

        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        },
    }
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path: str | Path, *, agent, online_buffer) -> dict:
    """Load into the given agent + buffer in place. Returns
    `{env_step, episode_state, extra}` for the training loop to resume from.
    """
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    agent.base_policy.load_state_dict(payload["base_policy"])
    agent.edit_policy.load_state_dict(payload["edit_policy"])
    agent.critic.load_state_dict(payload["critic"])
    agent.target_critic.load_state_dict(payload["target_critic"])
    agent.alpha.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
    if "alpha_target_entropy" in payload:
        agent.alpha.target_entropy = payload["alpha_target_entropy"]

    agent.opt_base.load_state_dict(payload["opt_base"])
    agent.opt_edit.load_state_dict(payload["opt_edit"])
    agent.opt_critic.load_state_dict(payload["opt_critic"])
    agent.opt_alpha.load_state_dict(payload["opt_alpha"])

    buf = payload["buffer"]
    online_buffer.states[:] = buf["states"]
    online_buffer.actions[:] = buf["actions"]
    online_buffer.rewards[:] = buf["rewards"]
    online_buffer.next_states[:] = buf["next_states"]
    online_buffer.dones[:] = buf["dones"]
    online_buffer._idx = int(buf["_idx"])
    online_buffer._size = int(buf["_size"])

    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return {
        "env_step": int(payload["env_step"]),
        "episode_state": payload["episode_state"],
        "extra": payload.get("extra", {}),
    }
