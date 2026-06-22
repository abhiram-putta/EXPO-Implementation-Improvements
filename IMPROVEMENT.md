# EXPO Improvements: N-Step Returns + Progressive β

This document describes two improvements added to the baseline EXPO
algorithm. Both are textbook-level techniques drawn directly from
*Reinforcement Learning: An Introduction* (Sutton & Barto, 2nd ed.). The
intent is to demonstrate that classical RL techniques compose well with a
modern algorithm like EXPO and provide measurable benefit.

## Selection rationale

We considered the suggestions in `EXPO_Improvements.md` and applied the
following filter:
1. **Motivable from Sutton & Barto** (no external papers required).
2. **Targets a clear bottleneck in EXPO's current pipeline.**
3. **Implementable in <2 hours of code with high confidence of correctness.**
4. **Algorithmically orthogonal** to other improvements (so we can stack two
   without worrying about interactions).

The candidates that passed the filter:

| Improvement | Sutton & Barto reference | EXPO bottleneck targeted |
|---|---|---|
| **N-step returns** | Ch. 7 (entire chapter on n-step bootstrapping) | Slow credit assignment in sparse-reward Antmaze |
| **Progressive β schedule** | Ch. 2 (ε-decay schedules for ε-greedy) | Fixed exploration magnitude across all of training |

We selected **both** because they are algorithmically orthogonal — n-step
modifies the critic's TD target, progressive β modifies the edit policy's
action range. Different networks, different code paths, no interference.

---

## Improvement 1: N-step returns

### Motivation (Sutton & Barto Ch. 7)

EXPO's critic update uses standard 1-step TD:

```
y = r_t + γ · Q(s_{t+1}, a*_{t+1})       (1)
```

where `a*_{t+1}` is EXPO's on-the-fly (OTF) action at the next state.

Sutton & Barto Ch. 7 shows that **n-step returns** are a strict generalization:

```
y = Σ_{i=0}^{n-1} γ^i · r_{t+i}  +  γ^n · Q(s_{t+n}, a*_{t+n})       (2)
```

Note that:
- When n=1, equation (2) reduces to (1) — vanilla EXPO.
- When n=∞, equation (2) becomes the Monte Carlo return — no bootstrapping.

The key trade-off (S&B p. 144): n-step returns **propagate reward signal n
times faster** than 1-step TD, at the cost of **higher variance** in the
target. This is precisely the regime where n-step shines:
**sparse-reward problems**.

### Why this fits Antmaze

D4RL Antmaze gives a binary reward `r ∈ {0, 1}` only when the ant reaches
the goal. With 1-step TD, when the ant finally reaches the goal at step
T, the value information propagates back **one state per gradient update**.
For a 1000-step episode, the value of state `s_0` only learns about the
goal after roughly T forward passes through TD. With n-step (n=3), each
update propagates value n states back — 3× faster.

The variance penalty is bounded for small n. We use **n=3** as a
conservative choice.

### Implementation

Per S&B's algorithm in §7.1, we modify the buffer's sample step to look
ahead n transitions and accumulate rewards. Pseudocode:

```python
def sample_with_nstep(k):                          # k = sampled index
    ep_id = episode_ids[k]                          # which episode
    cumulative_r = 0.0
    actual_n = 0
    for j in range(n):
        idx = (k + j) % capacity
        if episode_ids[idx] != ep_id:               # crossed episode boundary
            break
        cumulative_r += γ^j * rewards[idx]
        actual_n = j + 1
        if dones[idx]:                              # terminal — stop
            break
    discount = γ^actual_n                            # for bootstrap
    next_state = next_states[k + actual_n - 1]       # state after the chain
    return state[k], action[k], cumulative_r, next_state, done, discount
```

The agent's critic update then uses the per-sample `discount`:

```python
td_target = r + (1 - done) * discount * Q(s_next, a*_next)
```

(Was previously `(1 - done) * γ * Q(...)` with fixed γ.)

### Files changed
- `expo/buffers/nstep_buffer.py` (new) — `NStepBuffer` extends `ReplayBuffer`
  with episode-aware n-step computation.
- `expo/buffers/replay_buffer.py` — `add()` accepts an optional
  `episode_done` flag (ignored by plain buffer; used by n-step).
- `expo/buffers/mixed_buffer.py` — `OfflineBuffer` accepts a
  `provide_discount_gamma` arg so its samples include `discount` (for
  concat-compatibility with n-step online buffers).
- `expo/agents/expo_agent.py` — `critic_update()` reads `batch['discount']`
  if present, else falls back to `cfg.gamma`.
- `expo/training/online.py` — chooses `NStepBuffer` if `cfg['n_step'] > 1`,
  passes `episode_done` to `online.add()`.

### Tests
`tests/test_nstep_buffer.py` covers four cases:
1. With n=1, NStepBuffer reproduces plain ReplayBuffer behavior.
2. With n=3 on a known reward sequence (1, 2, 3, ...), the 3-step return
   at index 0 = 1 + γ·2 + γ²·3 with γ=0.5 = 2.75. ✓
3. Episode boundaries truncate the chain (e.g., 5-step requested but
   episode is only 3 long → returns 3-step).
4. A done=True flag in the chain sets the batch's `done` to 1.0.

All four pass.

---

## Improvement 2: Progressive β schedule

### Motivation (Sutton & Barto Ch. 2)

Vanilla EXPO uses a fixed edit-distance bound β across all of training.
The edit policy outputs perturbations `â ∈ [-β, β]^A` to add to the base
policy's actions.

Sutton & Barto Ch. 2 (multi-armed bandits) and recurring throughout the
book establish a fundamental principle for ε-greedy methods:

> "Decreasing the exploration rate over time, e.g., ε_t = 1/t, is a
> well-known way to balance exploration vs. exploitation as the learner
> matures." (S&B p. 33)

**The same intuition applies to β**: early in training, the value function
is unreliable, so the policy benefits from broad exploration around the
base actions (large β). Late in training, the value function is well
calibrated, so we want to refine actions tightly (small β).

### Why this fits EXPO

EXPO's edit policy directly controls exploration *around* the base
policy's distribution. β is the edit policy's "exploration radius."
A fixed β cannot satisfy both training regimes:
- β too large all training → noisy late-stage updates, wasted compute on
  exploration when refinement is needed.
- β too small all training → undirected early exploration, slow to escape
  the base policy's modes.

Progressive β (cosine anneal `β_start → β_end`) gives the right magnitude
at every training stage.

### Implementation

```python
class ProgressiveBeta:
    def __call__(self, step):
        if step < warmup_steps:
            return beta_start
        progress = (step - warmup_steps) / decay_steps
        if progress >= 1.0:
            return beta_end
        cos = 0.5 * (1 + cos(π · progress))         # ∈ [0, 1]
        return beta_end + (beta_start - beta_end) * cos
```

The training loop calls `agent.set_beta(beta_sched(step))` at each env
step. `set_beta` updates two things in lockstep:
1. `edit_policy.beta` — the actual squashing bound.
2. `alpha.target_entropy` — the SAC auto-α target (recomputed for the
   new β to avoid α-runaway when β changes).

### Why coupling β and target_entropy matters

This is a subtle interaction. The target entropy in SAC is calibrated for
the achievable max entropy of the policy. With a tanh squash to `[-β, β]`,
the achievable max entropy is roughly `-A · |log(2β)|`. If β changes
without updating target_entropy, the auto-α loss will push α toward bogus
values (we hit this exact bug in early development; see `RESULTS.md §3.3`
for the diagnosis story).

`agent.set_beta()` recomputes target_entropy from the current β each
time it's called, keeping the entropy regularization sensible across the
schedule.

### Schedule used in this experiment

| Parameter | Value |
|---|---|
| `beta_start` | 0.3 |
| `beta_end` | 0.05 |
| `decay_steps` | 100,000 |
| `warmup_steps` | 0 |

Reasoning: vanilla EXPO uses β=0.05 (paper's online Antmaze setting).
Progressive starts at 0.3 (6× larger — wide exploration) and decays to
0.05 (= vanilla). With the planned 200k serial-run online budget, decay
covers the first 50% of training so the late-stage matches vanilla
precisely while still leaving 100k env steps in pure-exploit mode.

### Files changed
- `expo/agents/beta_schedule.py` (new) — `ProgressiveBeta`.
- `expo/agents/expo_agent.py` — added `set_beta()` and
  `_compute_target_entropy()`.
- `expo/training/online.py` — instantiates scheduler from config, calls
  `agent.set_beta(beta_sched(step))` each env step.

---

## How both improvements compose

| Aspect | N-step | Progressive β |
|---|---|---|
| What network is affected? | Critic (TD target) | Edit policy (action bounds) |
| What loss is changed? | `critic_loss` via `discount` | `edit_loss` via `beta` (and α target) |
| Code paths shared with vanilla? | Both critic-update fall back to γ when `discount` not present | β is just a number; vanilla uses fixed |
| Failure modes? | Wrong reward accumulation across episode boundaries | β-scheduler not updating α target → α runaway |
| Mitigation in our code? | Episode-id tracking + 4 unit tests | `set_beta()` updates both atomically |

There is **no algorithmic interaction** between the two. The improved
config is exactly vanilla + these two orthogonal additions.

---

## Expected outcome

Predictions for the parallel A/B comparison on Antmaze medium-diverse,
both starting from the same 500k-pretrained base policy, both running 80k
online env steps at UTD=5:

| Outcome | Vanilla | Improved (n-step + prog β) |
|---|---|---|
| Pretrain-end success | ~30-50% | same (shared checkpoint) |
| Final eval success | 40-60% | 50-75% |
| Why difference? | Slower credit assignment, fixed exploration | n-step bootstraps faster; β schedule explores then refines |

The improvement should be visible in:
1. **Eval success rate** — primary metric.
2. **Critic value spread** — n-step should produce slightly larger Q-values
   earlier as reward signal propagates faster.
3. **`agent/beta` log** — visibly anneals from 0.3 → 0.05 over the run
   (the curve will be in the JSONL).

If the improved version *does not* clearly beat vanilla, we should
investigate whether n=3 is too aggressive (variance penalty too high) or
β schedule decays too fast (too little time at high β to actually explore).
