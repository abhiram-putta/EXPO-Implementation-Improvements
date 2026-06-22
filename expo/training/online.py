from __future__ import annotations

import time

import numpy as np
import torch

from pathlib import Path

from ..agents.beta_schedule import ProgressiveBeta
from ..agents.expo_agent import EXPOAgent, EXPOConfig
from ..buffers.mixed_buffer import MixedBuffer, OfflineBuffer
from ..buffers.nstep_buffer import NStepBuffer
from ..buffers.replay_buffer import ReplayBuffer
from ..envs.registry import build_env_and_offline
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logger import Logger
from ..utils.seeding import set_seed
from .eval import evaluate


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train(cfg: dict, resume_path: str | None = None) -> None:
    set_seed(int(cfg["seed"]))
    device = _resolve_device(str(cfg.get("device", "cpu")))
    print(f"Using device: {device}")

    # ---- Env + offline data ----
    env, eval_env, offline_data = build_env_and_offline(cfg, seed=int(cfg["seed"]))
    print(
        f"Env: {cfg['env']['name']}  state_dim={env.state_dim}  "
        f"action_dim={env.action_dim}  offline_transitions={len(offline_data['states'])}"
    )

    # ---- Buffers (choose plain vs n-step based on config) ----
    n_step = int(cfg.get("n_step", 1))
    gamma = float(cfg["agent"]["gamma"])
    if n_step > 1:
        print(f"Using NStepBuffer with n={n_step}, gamma={gamma}")
        online = NStepBuffer(
            capacity=int(cfg["online_capacity"]),
            state_dim=env.state_dim, action_dim=env.action_dim,
            n_step=n_step, gamma=gamma, device=device,
        )
        # Offline buffer must provide 'discount' so MixedBuffer concat works
        offline_gamma = gamma
    else:
        online = ReplayBuffer(
            capacity=int(cfg["online_capacity"]),
            state_dim=env.state_dim, action_dim=env.action_dim, device=device,
        )
        offline_gamma = None  # plain buffers don't include 'discount' field

    offline = OfflineBuffer(
        states=offline_data["states"],
        actions=offline_data["actions"],
        rewards=offline_data["rewards"],
        next_states=offline_data["next_states"],
        dones=offline_data["dones"],
        device=device,
        provide_discount_gamma=offline_gamma,
    )
    mixed = MixedBuffer(offline, online, online_ratio=float(cfg["online_ratio"]))

    # ---- Agent ----
    agent_cfg = EXPOConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        **cfg["agent"],
    )
    agent = EXPOAgent(agent_cfg, device=device)
    print(
        f"Agent: hidden={agent_cfg.hidden_dim}, T={agent_cfg.diffusion_steps}, "
        f"E={agent_cfg.ensemble_size}, N={agent_cfg.n_otf_samples}, "
        f"beta={agent_cfg.beta}, UTD={cfg['utd']}"
    )

    # ---- Logger ----
    logger = Logger(cfg["log_dir"], run_name=cfg.get("run_name"))

    # ---- Resume (optional) ----
    resume_state: dict | None = None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        resume_state = load_checkpoint(resume_path, agent=agent, online_buffer=online)
        print(f"  resumed at env_step={resume_state['env_step']}, "
              f"online_buffer_size={len(online)}")

    # ---- IL pretraining (skip on resume — base policy already loaded) ----
    pretrain_steps = int(cfg.get("pretrain_steps", 0))
    if pretrain_steps > 0 and resume_state is None:
        print(f"IL pretraining base policy for {pretrain_steps} steps...")
        for step in range(pretrain_steps):
            batch = offline.sample(int(cfg["batch_size"]))
            metrics = agent.base_policy_update(batch)
            if step % int(cfg["log_every"]) == 0:
                pretrain_step = step - pretrain_steps
                logger.log(pretrain_step, **metrics)
                logger.commit(pretrain_step)

        # Eval the IL-only baseline before any online learning kicks in.
        # Tells us how much of any final result is from pretrain vs from RL.
        if bool(cfg.get("eval_at_pretrain_end", False)):
            print("Eval at end of pretraining (IL-only baseline)...")
            base_eval = evaluate(
                agent, eval_env,
                num_episodes=int(cfg["eval_episodes"]),
                max_steps=int(cfg["max_episode_steps"]),
            )
            logger.log(0, **{f"pretrain_end/{k.split('/', 1)[1]}": v
                              for k, v in base_eval.items()})
            logger.commit(0)

        # Save a "pretrain-end" checkpoint so multiple online phases (e.g.,
        # vanilla + improved) can resume from the same starting point.
        save_pretrain_to = cfg.get("save_pretrain_checkpoint", None)
        if save_pretrain_to:
            from pathlib import Path as _P
            ckpt_to = _P(save_pretrain_to)
            ckpt_to.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving pretrain-end checkpoint -> {ckpt_to}")
            save_checkpoint(
                ckpt_to,
                agent=agent, online_buffer=online,
                env_step=-1,  # signal: "before any online step"
                episode_state={
                    "s": None, "ep_return": 0.0, "ep_success": False,
                    "rolling_returns": [], "rolling_successes": [],
                },
            )
            if int(cfg.get("total_env_steps", 0)) == 0:
                print("Pretrain-only mode (total_env_steps=0). Exiting.")
                logger.close()
                env.close()
                eval_env.close()
                return

    # ---- Online training ----
    print(f"Online training for {cfg['total_env_steps']} env steps...")
    if resume_state is not None:
        ep_state = resume_state["episode_state"]
        # Env is fresh — reset and discard mid-episode continuity (cheap to
        # lose: at most one episode's worth of env steps).
        s = env.reset()
        ep_return = 0.0
        ep_success = False
        rolling_returns = ep_state.get("rolling_returns", [])
        rolling_successes = ep_state.get("rolling_successes", [])
        start_step = resume_state["env_step"] + 1
        print(f"  resuming at env_step={start_step} "
              f"(env reset; in-progress episode discarded)")
    else:
        s = env.reset()
        ep_return = 0.0
        rolling_returns = []
        rolling_successes = []
        ep_success = False
        start_step = 0
    t0 = time.time()

    total_env_steps = int(cfg["total_env_steps"])
    batch_size = int(cfg["batch_size"])
    utd = int(cfg["utd"])
    log_every = int(cfg["log_every"])
    eval_every = int(cfg["eval_every"])
    eval_episodes = int(cfg["eval_episodes"])
    max_episode_steps = int(cfg["max_episode_steps"])
    min_online = int(cfg["min_online_for_update"])
    save_every = int(cfg.get("save_every_steps", 0))
    ckpt_path = Path(cfg["log_dir"]) / f"{logger.run_name}.ckpt"

    # ---- Progressive β scheduler (optional) ----
    beta_sched_cfg = cfg.get("progressive_beta", None)
    if beta_sched_cfg:
        beta_sched = ProgressiveBeta(
            beta_start=float(beta_sched_cfg["start"]),
            beta_end=float(beta_sched_cfg["end"]),
            warmup_steps=int(beta_sched_cfg.get("warmup_steps", 0)),
            decay_steps=int(beta_sched_cfg.get("decay_steps", 50000)),
        )
        print(f"Progressive beta: {beta_sched.beta_start} -> {beta_sched.beta_end} "
              f"over {beta_sched.decay_steps} steps "
              f"(warmup {beta_sched.warmup_steps})")
    else:
        beta_sched = None

    for step in range(start_step, total_env_steps):
        # Update β if scheduling
        if beta_sched is not None:
            agent.set_beta(beta_sched(step))

        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        a = agent.otf_action(s_t).squeeze(0).cpu().numpy()

        ns, r, terminated, truncated = env.step(a)
        # Pass episode_done so the n-step buffer can detect boundaries
        online.add(s, a, r, ns, terminated,
                   episode_done=(terminated or truncated))
        if r >= 0.5:
            ep_success = True
        s = env.reset() if (terminated or truncated) else ns
        ep_return += r
        if terminated or truncated:
            rolling_returns.append(ep_return)
            rolling_successes.append(ep_success)
            ep_return = 0.0
            ep_success = False

        if len(online) < min_online:
            continue

        metrics = agent.update(
            sampler=mixed.sample,
            batch_size=batch_size,
            utd=utd,
            env_step=step,
        )

        if step % log_every == 0:
            metrics["online/buffer_size"] = float(len(online))
            metrics["online/steps_per_sec"] = float((step + 1) / (time.time() - t0))
            metrics["otf/frac_edited_selected"] = float(getattr(agent, "last_frac_edited", 0.0))
            metrics["agent/beta"] = float(agent.edit_policy.beta)
            metrics["agent/target_entropy"] = float(agent.alpha.target_entropy)
            if rolling_returns:
                metrics["online/episode_return"] = float(np.mean(rolling_returns[-10:]))
                metrics["online/success_rate"] = float(np.mean(rolling_successes[-10:]))
            logger.log(step, **metrics)
            logger.commit(step)

        if step > 0 and step % eval_every == 0:
            eval_metrics = evaluate(
                agent, eval_env,
                num_episodes=eval_episodes,
                max_steps=max_episode_steps,
            )
            logger.log(step, **eval_metrics)
            logger.commit(step)

        if save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(
                ckpt_path,
                agent=agent,
                online_buffer=online,
                env_step=step,
                episode_state={
                    "s": s, "ep_return": ep_return, "ep_success": ep_success,
                    "rolling_returns": rolling_returns,
                    "rolling_successes": rolling_successes,
                },
            )
            print(f"  [step {step}] checkpoint saved -> {ckpt_path}")

    # final eval
    eval_metrics = evaluate(
        agent, eval_env,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps,
    )
    logger.log(total_env_steps, **eval_metrics)
    logger.commit(total_env_steps)
    logger.close()
    env.close()
    eval_env.close()
    print("Training complete.")
