# Implementation Decisions

This file records the choices made for the EXPO reference implementation.
Defaults are intentionally lightweight — designed to validate the algorithm
end-to-end on a single CPU in roughly ten minutes — not to reproduce the
paper's benchmark numbers.

## 1. Benchmark
**Decision:** `gymnasium`'s `Pendulum-v1` with synthetically generated offline
data (random-policy rollouts).

**Why:**
- Zero install friction. Gymnasium's classic-control envs are pure-NumPy and
  ship with the package — no `mujoco_py`, no `d4rl`, no compiled extensions.
- Sufficient to test: continuous actions, dense reward, well-known optimal
  return, standard SAC validation environment.
- D4RL/Robomimic/Adroit/MimicGen are heavyweight installs (Mujoco, Robosuite,
  legacy `gym`) and frequently broken on Windows. Adding them is **purely
  additive**: drop a new wrapper into `expo/envs/`, add a new YAML config —
  no agent or training-loop changes required.

## 2. GPU
**Decision:** Auto-detect (`device: auto` → cuda if available, else cpu).

## 3. Goal
**Decision:** Reference implementation that runs end-to-end and demonstrably
learns. Not paper-faithful reproduction (which would require D4RL benchmarks,
3-seed averages, and ~500k step budgets per task).

## 4. Logging
**Decision:** JSONL on disk + stdout. No external dependency.
- Each step's metrics get one JSON line in `runs/<run_name>.jsonl`.
- Dynamic field set — new metrics can appear at any step.
- W&B / TensorBoard intentionally not used (avoids hard dep). Easy to add
  later by wrapping `Logger.commit`.

## 5. Improvements folded into v1
- **LayerNorm in critic** (Improvements §4.1): adds first-layer LayerNorm to
  every Q-network. Cheap; protects plasticity at high UTD.
- **Edit-policy gradient clipping** (§8.1): `clip_grad_norm_=1.0`.
- **Delayed edit policy start** (§8.2): edit policy update is skipped for the
  first `delayed_edit_steps` env steps; OTF still runs but candidates are just
  base samples (because edit policy is untrained random noise — adding
  random edits would only hurt).
- **Comprehensive diagnostics** (§10.2): `Logger` records the metrics flagged
  in §10.2 — `mean_q`, `q_ensemble_std`, `mean_edit_magnitude`, `alpha`,
  `frac_edited_selected`, etc.

## 6. Improvements designed-for but deferred
- **OTF caching across UTD steps** (§1.1): the `EXPOAgent.otf_action` API
  takes `state` and a critic — it can be wrapped in a caching layer without
  touching callers. Not implemented in v1 because the smoke config uses
  small networks; expected to be needed at paper-scale settings.

## 7. Improvements not implemented
Flow matching, transformer denoiser, mixture edits, hierarchical edits,
prioritized replay, curiosity, multi-task, n-step returns, state-dependent
β, progressive β. All listed in `EXPO_Improvements.md`. Add only when
motivated by a real failure mode.

## 8. Polyak τ convention
The paper writes `ϕ' ← τ · ϕ' + (1 − τ) · ϕ` with τ=0.005. Taken literally,
this means the target nearly equals the source after one step — clearly not
the intent. We use the standard Polyak convention
`target ← (1 − τ)·target + τ·source` with τ=0.005, i.e. the target tracks
the source slowly. This matches the canonical SAC/REDQ reading and is what
the paper's reference codebase actually does.

## 9. Loss conventions
- **Critic**: MSE between every ensemble member's Q and the same TD target
  (per-network MSE summed → averaged via a single `F.mse_loss` over the
  stacked ensemble tensor).
- **Edit policy**: gradients flow through the **live** critic (not target).
  Random subset min for pessimism on the edit-policy loss, matching the
  Improvements doc's API.
- **TD target**: uses **target** critic's random-subset min on the OTF
  action at `s'`.
- **Base policy**: trained on the *mixed* batch (offline + online) — per
  Implementation Guide §6.1, *not* offline-only.
- **α**: SAC auto-tuning, target entropy = -dim(action_space).

## 10. Buffer mixing
50/50 offline/online (RLPD default). When the online buffer is too small to
satisfy its share, the deficit is drawn from offline. No update step runs
until the online buffer holds at least `min_online_for_update` transitions —
guards against degenerate first batches.

## 11. Action range convention
Edit policy outputs are squashed to [-β, β] via `β · tanh(u)`, then added to
the base action and clipped to [-1, 1]. The Pendulum wrapper rescales the
[-1, 1] agent action to the env's native [-2, 2] torque range.

## 12. Pendulum-specific
- Pendulum has no terminal state — episodes only **truncate** (after 200 steps).
- We pass `terminated` (always False for Pendulum) — never `truncated` —
  into the bootstrap mask. Critical: bootstrapping past truncation is required
  for correct Q-learning on time-limit tasks.

## 13. β default
The Implementation Guide gives β=0.05 for tasks with good offline data and
β=0.7 for high-exploration tasks (Adroit). Pendulum with random-policy
offline data sits closer to the high-exploration end (random data is far
from optimal), so the default config uses **β=0.3** as a reasonable middle
ground. Tunable per config.

## 14. Smoke-config sizing
Default `expo/configs/pendulum.yaml` is sized for ~10 minutes on CPU:
hidden_dim=128, diffusion_steps=5, ensemble_size=5, UTD=4, N=4. The paper's
"full" sizing (256, 10, 10, 20, 8) is documented in the YAML as commented
overrides. Scale up once correctness is verified.

## 15. VP β schedule (BUG FIX, post-Antmaze)
**Original**: `vp_beta_schedule(T) = linspace(1e-4, 0.02, T)` — the
standard DDPM-T=1000 schedule applied to T=10. Result: `alpha_bar[T-1] =
0.90`, denoiser only ever sees lightly corrupted inputs.

**Fixed**: Proper VP-SDE-derived schedule.
β(t) = β_min + t·(β_max − β_min) at t_k = k/T, β_min=0.1, β_max=20.
Result: `alpha_bar[T-1] = 4.3e-5` for any T (proper near-pure noise).

**Effect**: IL loss after 30k pretrain steps dropped from 0.66 → 0.027.
Diffusion samples actually condition on state.

## 16. Antmaze observation wrapper (BUG FIX, post-Antmaze)
gymnasium-robotics' AntMaze runs the underlying Ant with
`exclude_current_positions_from_observation=True`. Body XY is *removed*
from `observation` and lives only in `achieved_goal`. Earlier wrapper
dropped `achieved_goal`, leaving the policy navigating blind. Fix:
return `[observation; desired_goal − achieved_goal]` (29 dim,
translation-invariant goal encoding).

## 17. Auto-α target entropy with tight β (BUG FIX, post-Antmaze)
SAC's `target = -action_dim` is calibrated for actions in [-1, 1]. With
β=0.05 squashing to [-0.05, 0.05]^A, achievable max entropy is roughly
`-A · |log(2β)|` ≈ -18.4, BELOW the target of -8. α inflates monotonically
trying to push entropy above an unreachable ceiling.

**Fix**: squash-aware target `min(-A, -A + A·log(2β))`. For β=0.05, A=8:
target = -26.4 instead of -8. Auto-α now stable.

**For progressive β (improvement)**: `agent.set_beta(new_beta)` updates
both `edit_policy.beta` *and* `alpha.target_entropy` in lockstep. Keeps
the entropy regularization sensible across the schedule.

## 18. N-step return implementation (improvement, S&B Ch. 7)
Added `NStepBuffer` extending `ReplayBuffer` with episode-aware n-step
target computation. Episodes are tracked via an `episode_ids[capacity]`
array. On sample, walk forward up to n steps from the sampled index,
stopping at episode boundary or terminal transition. Return per-sample
`discount = γ^k` where k is actual chain length.

The agent's `critic_update` reads `batch['discount']` if present, else
falls back to `cfg.gamma`. Plain buffers are unchanged (they don't include
`discount`); `MixedBuffer` requires both halves to provide `discount` if
either does, so `OfflineBuffer` accepts a `provide_discount_gamma=γ`
constructor arg that activates the field.

## 19. Progressive β implementation (improvement, S&B Ch. 2 ε-decay analogue)
`ProgressiveBeta(start, end, warmup_steps, decay_steps)` returns a cosine
anneal from start to end over decay_steps after warmup_steps. The training
loop calls `agent.set_beta(beta_sched(step))` each env step, which updates
both the edit policy's β and the auto-α target_entropy in lockstep.

## 20. Shared pretrain checkpoint for fair A/B comparison
`save_pretrain_checkpoint: <path>` config option saves a checkpoint at the
end of pretraining specifically. Both vanilla and improved kernels load
from the same `shared.ckpt`, so they start with identical pretrained base
policy + identical (random-init) critic + edit policy + α. The only
difference between them is the n_step + progressive_beta config flags.

