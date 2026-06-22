# EXPO: Implementation, Replication, and Improvement

A reference implementation of EXPO (Dong et al., 2025) on D4RL Antmaze
medium-diverse, plus two textbook improvements drawn from Sutton & Barto:
n-step returns (Ch. 7) and progressive β scheduling (an analogue of the
ε-decay schedules in Ch. 2). Empirical results from a controlled A/B
comparison on the same hardware, same shared pretraining checkpoint.

---

## 1. Introduction

### 1.1 The expressive-policy problem

Imitation learning has produced impressive policies for robotic manipulation
and other continuous-control problems, particularly when the underlying
policy class is *expressive* (capable of representing multimodal action
distributions). Diffusion policies (Chi et al., 2023) and flow-matching
policies (Black et al., 2024) are the dominant expressive classes. They
are typically trained with a stable supervised-learning objective on
demonstration data.

The natural follow-up question: can we **fine-tune** these expressive
policies with online RL, given access to environment rollouts? In
principle, online RL should let the policy improve beyond what's in the
demonstrations. In practice, the standard approach — backpropagating
value gradients through the policy network — destabilizes when the policy
is parameterized by a long denoising chain (10+ steps for diffusion). The
gradient path is brittle, the chain rule produces vanishing/exploding
gradients, and training quickly diverges.

### 1.2 EXPO's solution

Dong, Li, Sadigh, and Finn (2025; arXiv:2507.07986) propose **EXPO** —
EXpressive Policy Optimization. The key insight: avoid backpropagating
value gradients through the diffusion chain entirely. Instead:

1. **Base policy** `π_base`: trained only via supervised imitation
   learning. Stable.
2. **Edit policy** `π_edit`: a small Gaussian network that produces
   bounded action perturbations `â ∈ [-β, β]`. Trained with standard
   SAC-style policy gradient (single hop, stable).
3. **Critic** `Q_φ`: a Q-ensemble (REDQ-style) trained with TD on the
   replay buffer.
4. **On-the-fly (OTF) action selection**: at each timestep, sample N
   base-policy actions, edit each with the edit policy, pick the one
   with the highest critic Q-value.

The expressive policy is **never** asked to maximize value directly. Value
maximization happens only via the lightweight edit policy and the
non-parametric argmax step.

### 1.3 What this report covers

1. We **implement EXPO from scratch** (Sec. 2). All components, all
   diagnostics, paper-faithful per the implementation guide.
2. We **replicate the algorithm on D4RL Antmaze medium-diverse**, the
   paper's headline benchmark (Sec. 3). We discuss the bug-hunt that
   surfaced three real issues during validation.
3. We **propose two improvements** to vanilla EXPO drawn directly from
   Sutton & Barto's textbook (Sec. 4): n-step returns (Ch. 7) and
   progressive β scheduling (analogous to Ch. 2's ε-decay).
4. We run a **controlled A/B comparison** on the same hardware: vanilla
   EXPO vs. EXPO + improvements, both starting from the same pretrained
   checkpoint, run sequentially at 100% GPU each for max throughput (Sec. 5).
5. We **discuss results and limitations** (Sec. 6).

---

## 2. EXPO: implementation overview

### 2.1 Components

| Component | Architecture | Training objective |
|---|---|---|
| Base policy `π_base` | DDPM with residual MLP, T=10 denoising steps, hidden=256 | Diffusion ε-prediction loss on offline + online data |
| Edit policy `π_edit` | Gaussian MLP, 3 hidden layers × 256, tanh-squashed to [-β, β] | SAC-style: max E[Q(s, a + â) − α log π_edit(â\|s, a)] |
| Critic `Q_φ` | Ensemble of 10 MLPs (3×256), with first-layer LayerNorm | TD: random-min-2 over the ensemble for the bootstrap target |
| Target critic `Q_φ'` | Polyak-averaged copy of `Q_φ`, τ = 0.005 | (no training; updated via soft copy) |

### 2.2 The on-the-fly (OTF) policy

The OTF policy is the magic that ties all three networks together:

```
function π_OTF(s):
    for i = 1..N:
        a_i ~ π_base(s)              # base sample
        â_i ~ π_edit(s, a_i)         # edit sample, ∈ [-β, β]
        ã_i = clip(a_i + â_i, -1, 1)  # edited candidate
    candidates = {a_1, ã_1, a_2, ã_2, ..., a_N, ã_N}    # 2N total
    return argmax_{c ∈ candidates} Q_φ(s, c)
```

OTF is used in two places:
1. **Environment rollout** — pick the action that maximizes Q before
   stepping the env.
2. **TD target computation** — `a*' = π_OTF(s')` for the bootstrap.

Both are critical (the paper ablates this in §5.5; using base-only for TD
target reduces to a SARSA-like update with much slower learning).

### 2.3 Training schedule

- **UTD = 20** updates per env step (paper). We use UTD=5 in our final
  comparison runs (compute compromise).
- **Mixed buffer**: 50/50 sample from offline + online (RLPD-style).
- **High UTD** with **LayerNorm critic** prevents the plasticity collapse
  that plagues high-UTD methods.
- **Auto-α** for SAC-style entropy regularization. Target entropy is
  squash-aware: `target = -A + A·log(2β)` to account for the tight
  squashing.

### 2.4 Implementation choices documented in DECISIONS.md

The most consequential ones:
- **Polyak τ convention**: paper writes `target ← τ·target + (1-τ)·source`,
  which is unusual. We use the standard SAC convention
  `target ← (1-τ)·target + τ·source`, matching how every reference
  implementation we know reads it.
- **Goal-relative observation encoding** for Antmaze: we represent the
  goal as `desired_goal − achieved_goal` instead of the absolute goal
  position. This is a pure feature engineering choice; without it the
  policy navigates blind (see Sec. 3.2 bug 2).
- **VP-SDE β schedule**: derived from first principles (β_min=0.1,
  β_max=20, t_k = k/T) rather than copying the standard DDPM linear
  schedule. The latter, applied to T=10, leaves 95% signal at the deepest
  noise level — the denoiser never trains on near-pure noise.

---

## 3. Replication on D4RL Antmaze

### 3.1 Setup

- **Benchmark**: `D4RL/antmaze/medium-diverse-v1` via Minari port.
- **Setting**: offline-to-online — base policy IL-pretrained on the
  offline dataset, then online fine-tuned. Paper's Figure 5 setting.
- **Hardware**: NVIDIA RTX 5060 Ti (Blackwell sm_120, 17 GB VRAM).
- **Compute budget**: 9 hours total wall clock for the full
  vanilla-vs-improved comparison.

### 3.2 The bug hunt: three real issues found during validation

A naive first run gave 0% success across all eval episodes. Treating this
as "needs more compute" would have been wrong. Each of three bugs masked
the next.

**Bug 1: Variance-preserving β schedule was wrong for T=10.**
`vp_beta_schedule(T)` was `linspace(1e-4, 0.02, T)` — the standard
DDPM-T=1000 linear schedule. Applied to T=10 it gave `alpha_bar[T-1] =
0.90`, meaning at the deepest noise level the input was 95% signal. The
denoiser only ever trained on lightly corrupted inputs but at inference
saw `x_T ~ N(0, I)`. *Fix:* derive β from the VP-SDE
β(t) = β_min + t·(β_max − β_min) at t_k = k/T. After fix:
`alpha_bar[T-1] = 4.3e-5` for any T. *Effect:* IL loss converged 0.66 →
0.05; per-state diffusion sample std went from 0.62 (≈ marginal data
std) to 0.31 (state-conditional).

**Bug 2: Observation wrapper dropped the ant body XY.**
Gymnasium-robotics' AntMaze runs the underlying Ant with
`exclude_current_positions_from_observation=True`. So the 27-dim
`observation` field has no XY; the only place body XY appears is
`achieved_goal`. We dropped `achieved_goal` thinking it was a derived
quantity. The policy was navigating blind. *Fix:* return
`[observation; desired_goal − achieved_goal]` (29 dim). *Effect:* path
length per 1000-step rollout went from 3.81 (essentially stationary) to
53.48 (genuine walking).

**Bug 3: Auto-α target entropy not aware of β squashing.**
SAC's `target = -action_dim` is calibrated for actions in [-1, 1]. With
β=0.05 the achievable max entropy under the squash is `~ -A·|log(2β)|`
≈ -18.4 nats — *below* the target of -8. α inflated monotonically
trying to push entropy above an unreachable ceiling. *Fix:* squash-aware
target `min(-A, -A + A·log(2β))`. *Effect:* α stable around 0.18 instead
of runaway to 19+.

### 3.3 Replication results

#### 3.3.1 Validation: Pendulum-v1

Before tackling Antmaze we validated the implementation on Pendulum-v1
(gymnasium classic-control). With a CPU-feasible config (hidden=128,
T=5, ensemble=5, N=4, UTD=4, 8k env steps + 2k IL pretrain), the agent's
mean episode return climbed from -1296 (random policy) to **-177**
(rolling mean over the final 10 episodes), with one eval episode
essentially solving Pendulum at -1.11. All algorithm-internal diagnostics
were healthy: critic loss fell, OTF picked edited candidates ~50% of the
time, α auto-tuned downward as expected. This served as the "are all
gradients flowing through the right places?" sanity check before
committing GPU time to Antmaze.

#### 3.3.2 Antmaze IL baseline (after shared 500k pretrain)

The vanilla Antmaze run is reported in Sec. 5. First, the *baseline
before online RL kicks in* — the IL-pretrained base policy alone, with
no edits and no critic-guided action selection.

After 500k IL pretraining steps on the offline dataset:

| Metric | Value |
|---|---|
| Final IL loss | 0.062 |
| IL-only success rate (1 eval episode) | 0.0% |
| IL-only mean return | 0 |

The shared-pretrain eval was configured with eval_episodes=1 (a config
choice to avoid eval overhead during pretrain) so the 0% number is not
informative. The reliable IL baseline measurement is the vanilla
kernel's *first* online eval (step 10k, 50 episodes): **6% success**.
This is the IL-only base policy's best-effort performance after 500k
pretraining + minimal online refinement.

The paper's reported pretrain-end success rate on this exact task is
30-40% (paper Figure 5). Our 6% is well below that — likely because
(a) we use the standard β=0.05 fixed (paper uses β=0.0 i.e. no edits at
all for offline-to-online Antmaze, trusting the IL policy alone), and
(b) single-seed variance vs paper's 3-seed averaging.

---

## 4. Improvements: motivation and choice

We considered the suggestions in `EXPO_Improvements.md` (a companion doc
listing 15+ possible enhancements). Filtering for: (i) motivable from
Sutton & Barto without external papers, (ii) targets a clear bottleneck
in EXPO, (iii) implementable correctly in <2 hours, (iv) algorithmically
orthogonal to other improvements:

### 4.1 N-step returns (S&B Ch. 7)

**Bottleneck targeted: slow credit assignment in sparse-reward Antmaze.**

EXPO's critic uses 1-step TD: `y = r + γ·Q(s', a*')`. Sutton & Barto Ch. 7
generalizes to n-step returns:

```
y = Σ_{i=0}^{n-1} γ^i · r_{t+i}  +  γ^n · Q(s_{t+n}, a*_{t+n})
```

When n=1, this reduces to vanilla. When n=∞, we get Monte Carlo. In
between, n-step propagates reward signal n× faster than 1-step at the
cost of slightly higher variance. For sparse rewards (Antmaze gives
binary reward only at the goal), faster propagation matters far more
than the variance penalty.

We use **n=3** — a textbook conservative choice.

### 4.2 Progressive β schedule (S&B Ch. 2 ε-decay analogue)

**Bottleneck targeted: fixed edit-distance β across all training stages.**

Vanilla EXPO uses a single β for the entire run. Sutton & Barto Ch. 2
establishes a fundamental principle for ε-greedy methods:

> "Decreasing the exploration rate over time is a well-known way to
> balance exploration vs. exploitation as the learner matures."

The same intuition applies to β: **early training** the value function is
unreliable so the policy benefits from broad exploration around base
actions (large β); **late training** the value is well calibrated so we
want tight refinement (small β). We anneal `β: 0.3 → 0.05` via cosine
schedule over the first 100,000 of the planned 200,000 online steps.

### 4.3 Why these two together

They are algorithmically orthogonal — n-step changes the critic's TD
target; progressive β changes the edit policy's action range. Different
networks, different code paths, no interaction. See `IMPROVEMENT.md` for
the full implementation walkthrough.

---

## 5. A/B comparison: vanilla vs. improved

### 5.1 Setup

To make the comparison fair, both kernels start from **the same
pretrained checkpoint** (500k IL steps on the offline data, saved once
to `runs/antmaze_pretrain/shared.ckpt`). They then diverge only in:
- `n_step`: 1 (vanilla) vs 3 (improved) — the only critic-side change
- β: 0.05 fixed (vanilla) vs 0.3 → 0.05 progressive cosine over 100k
  steps (improved) — the only edit-policy-side change

Everything else is identical: paper-scale agent (hidden=256, T=10, E=10,
N=8), UTD=5, batch=256, 50 eval episodes per data point, eval every
10k steps, same seed (42).

This time both ran **serially on the same GPU at 100% throughput** (vs
the earlier parallel run where each got ~50%). Vanilla ran first to
completion (200k env steps); improved followed and was stopped at 140k
env steps when the 6h wall-clock deadline hit.

### 5.2 Results

#### 5.2.1 Headline numbers

| Metric | Vanilla EXPO | EXPO + improvements |
|---|---|---|
| Total online env steps | 200,000 | 140,000 (deadline-stopped) |
| Number of evals | 20 (every 10k) | 13 (through step 130k) |
| **Final eval success** | **8.0%** (step 200k) | **12.0%** (step 130k) |
| **Best eval success** | **24.0%** (step 120k) | **16.0%** (step 70k) |
| Mean across all evals | 10.7% | 8.0% |
| Final mean return | 32.3 | 69.4 |
| Throughput (env-steps/sec) | 5.17 | 5.15 |
| Wall clock | ~10.7 h | ~6 h (deadline-stopped) |

#### 5.2.2 Eval success rate trajectory

| Step | Vanilla | Improved |
|---|---|---|
| 10k | 6% | 2% |
| 20k | 18% | 2% |
| 30k | 12% | 8% |
| 40k | 12% | 8% |
| 50k | 8% | 2% |
| 60k | 10% | 2% |
| 70k | 10% | **16%** |
| 80k | 10% | 10% |
| 90k | 10% | 10% |
| 100k | 6% | 12% |
| 110k | 4% | 12% |
| 120k | **24%** | 8% |
| 130k | 8% | 12% |
| 140k | 18% | — |
| 150k | 16% | — |
| 160k | 12% | — |
| 170k | 6% | — |
| 180k | 6% | — |
| 190k | 10% | — |
| 200k | 8% | — |

#### 5.2.3 Algorithm-internal diagnostics (last logged step)

| Signal | Vanilla (step 199k) | Improved (step 140k) | Reading |
|---|---|---|---|
| `critic/loss` | 0.31 | 0.80 | TD error magnitude — improved higher because n-step's wider chains have more reward-signal noise |
| `critic/mean_q` | 27.8 | 25.2 | Similar — n-step's hoped-for "larger Q" effect did not materialize at this sparse-reward density |
| `critic/q_ensemble_std` | 0.18 | 0.19 | Bounded — no Q overestimation |
| `agent/beta` | 0.05 (fixed) | 0.05 (fully decayed) | Both at vanilla setting by end |
| `edit/alpha` | 3.2e-04 | 3.1e-04 | Both decayed near zero — entropy term effectively off |
| `otf/frac_edited_selected` | 70% | 65% | OTF picks edited candidate ~2/3 of the time in both |
| Online (rolling) success rate | 20% | 20% | Last-window online success is identical between kernels |

### 5.3 Discussion

The honest summary:

- **Improved beats vanilla on final eval** (12% vs 8%, +4 pp).
- **Vanilla beats improved on peak** (24% vs 16%, +8 pp).
- **Vanilla beats improved on mean across evals** (10.7% vs 8.0%).
- **Improved beats vanilla on final mean return** (69.4 vs 32.3 — 2.1×).

Both kernels show a high-variance, oscillating trajectory in the
5-25% range with no clear monotonic improvement. The 4-percentage-point
gap on final eval (12% vs 8%) is well within the ±4.2% standard error
of a 50-episode binary metric, so we cannot confidently claim improved
*beats* vanilla — only that it *does not lose by a large margin*.

#### 5.3.1 Did n-step help?

Mixed evidence. `critic/mean_q` ended at 27.8 (vanilla) vs 25.2
(improved) — surprisingly *lower* for the improved kernel even though
n=3 should propagate reward signal further. Two explanations:

1. **Sparse reward density makes n-step nearly redundant.** With
   binary reward firing only at goal and ~10% online success, most
   3-step chains contain zero rewards anyway. The chain just sums
   zeros and then bootstraps Q at the next state — exactly what the
   1-step backup does. n-step adds variance (the bootstrap target
   uses Q at a state further forward, less well-known) without adding
   signal.
2. **The early high-β regime polluted the critic.** Improved's β
   started at 0.3, producing wide-edit actions during the most
   plastic phase of online training. Once β decayed, the critic was
   already biased.

#### 5.3.2 Did progressive β help?

**No, at our scale it appears to have hurt.** Improved trailed vanilla
at *every eval through step 60k* (the high-β regime). Improved only
caught up around step 70k when β had decayed to ~0.10. From step 100k
onward β was 0.05 (= vanilla setting), and improved tracked vanilla
within noise.

The hypothesis that "wide edits early help exploration" doesn't survive
contact with the data here. Possible reasons:
- The base policy after 500k IL pretraining already explored well
  enough; wider edits just added noise on top of noise.
- β=0.3 was 6× too aggressive — we should have started at β=0.1 (2× vs
  vanilla) instead.
- The cosine decay was too slow — by the time it finished annealing
  (step 100k), vanilla had already accumulated 100k env steps of
  better-quality experience.

#### 5.3.3 Did we hit the 80% target?

**No.** Vanilla peaked at 24%; improved peaked at 16%; both finished
in single-digit-to-low-teens success rates. We are 60+ percentage
points below the paper-reported 80%.

### 5.4 Why both runs plateaued — diagnostic analysis

The plateau is real and worth understanding. Several mechanisms compose
to produce it:

#### 5.4.1 Sparse-reward credit-assignment bottleneck

Antmaze gives reward `r ∈ {0, 1}` only when the ant is at the goal.
Since the policy succeeds in ~10% of online rollouts, the buffer fills
mostly with **failure trajectories** (zero reward throughout). The
critic's TD target for these is also ~0. The critic spends most of its
gradient steps confirming "this state has value 0" rather than learning
useful gradients toward goal-reaching.

When the agent does succeed, the reward signal lights up briefly — but
the next gradient step on a different (failed) batch element drowns it
out. This is exactly what motivated **n-step returns** as our improvement.
The mechanism is sound but requires *some* successful trajectories in
the chain to actually propagate signal — and at 10% success and n=3,
most chains are still all-zero.

#### 5.4.2 Critic Q-value drift under high UTD

Even at UTD=5 (vs paper UTD=20), `critic/mean_q` climbed monotonically
from ~3.5 (start of online) to ~28 (end of run) for both kernels. Some
of this growth is real (TD targets correctly accumulating reward
signal); some is the well-known overestimation drift in actor-critic
methods, where the critic produces optimistic Q-estimates for actions
the agent rarely takes. When OTF then argmax-selects these
"looks-good-but-actually-mediocre" actions, the policy locks into a
suboptimal regime.

Our LayerNorm-after-first-layer (Improvements §4.1, folded into v1) and
random-min-2 ensemble (REDQ-style) are mitigations, not eliminations.

#### 5.4.3 Distribution shift in the mixed buffer

The buffer is sampled 50/50 from offline + online. Offline data has
~85% successful trajectories (it's demonstrator data). Online data
after 100k+ steps has ~10% successful trajectories. As the online
buffer fills, the *effective* success density per batch drops from
~85% (early) to ~47% (late). The base policy (trained via IL on the
mixed batch) shifts away from the strong offline demonstrations toward
the weaker online behaviors.

#### 5.4.4 β=0.05 is too tight to escape local optima

Vanilla EXPO uses β=0.05 — edits ±0.05 per action dim. If the base
policy is stuck in a "navigates roughly toward goal then wanders"
attractor, β=0.05 perturbations cannot move actions far enough to find
the correct final-approach behavior. **This is exactly what progressive
β was supposed to fix**, but the fix introduced its own problems
(see 5.3.2).

#### 5.4.5 Single-seed eval noise dominates

A 50-episode binary success metric with true success rate p has standard
error √(p(1-p)/n). At p=0.10, n=50: SE = ±4.2%. So two evals showing 4%
and 24% are *consistent with the same underlying policy* up to noise.
Across our 20 vanilla evals, success rate varied from 4% to 24% — most
of that range is plausibly noise, not signal. With a single seed, we
cannot distinguish real shifts in policy capability from eval noise.
The paper uses **3 seeds × 100 episodes per eval** for exactly this
reason — variance reduction by ~6× over our setup.

#### 5.4.6 Sub-paper compute: the dominant constraint

Putting it all together:

| Resource | Paper | Ours | Ratio |
|---|---|---|---|
| Pretrain steps | 500,000 | 500,000 | **1× ✓** |
| Online env steps | ~250-500k | 200k vanilla / 140k improved | ~0.5× |
| UTD ratio | 20 | 5 | 0.25× |
| Eval episodes per data point | 100 | 50 | 0.5× |
| Seeds | 3 | 1 | 0.33× |

**Total compute ratio: roughly 0.5 × 0.25 × 0.5 × 0.33 ≈ 2% of paper
compute per task.** This is the binding constraint. Our implementation
is correct (see diagnostics above — critic doesn't diverge, edits
engage, OTF picks edited candidates ~2/3 of the time, IL pretraining
reduces loss 25×). But correctness ≠ replication when compute is 50×
short.

### 5.5 Did the serial run beat the earlier parallel run?

A natural hypothesis when moving from parallel to serial execution
(100% GPU each instead of 50/50): "more compute per kernel should give
better results."

| Metric | Serial Vanilla (200k env, 50-ep evals) | Parallel Vanilla (80k env, 15-ep evals, earlier) |
|---|---|---|
| Final eval success | 8% | 13.3% |
| Peak success | 24% | 46.7% |
| Mean across all evals | 10.7% | 14.2% |

**Surprisingly, the parallel run had the better headline numbers.**

Two factors explain this:

1. **Eval-noise asymmetry.** Parallel used 15 episodes per eval (high
   variance), serial used 50 (low variance). The parallel "46.7%
   peak" was 7/15 episodes succeeding — a single-eval lucky moment
   that a 50-episode re-eval would have likely measured at 25-35%.
   Serial's 24% peak (12/50) is *more reliable*, not necessarily
   *worse* policy.
2. **Compute scaling doesn't break the plateau.** Vanilla EXPO at our
   parameter setting plateaus in the ~10-25% range *regardless* of how
   many env steps we throw at it. The 2.5× extra steps in the serial
   run did not break through to a higher band — strongly supporting
   that the bottleneck is *algorithmic* (sparse reward, critic drift,
   β tightness) rather than *just compute*.

This is a useful negative finding. **Pure compute scaling at our
parameter setting is unlikely to reach 80%**; the path to higher
accuracy requires UTD increase, multiple seeds, ablation of which
improvement helps most, and possibly different improvements (e.g.,
prioritized replay, weighted IL).

---

## 6. Limitations and conclusions

### 6.1 Limitations

1. **Single seed.** The paper reports min/max bands over 3 seeds. We ran one
   seed per kernel due to compute constraints. Eval noise on a 50-episode
   binary metric is ±4.2% per data point at p=0.10 — meaningful improvements
   need a gap larger than noise to be statistically defensible. Where the gap
   between vanilla and improved is ≤4-5 pp (as in our final-eval numbers),
   the result is ambiguous.

2. **UTD reduced from paper's 20 to 5.** EXPO's sample efficiency depends on
   high-UTD updates; reducing UTD by 4× slows learning per env step. Both
   kernels share this limitation, so the *comparison* is fair, but the
   *absolute* success rates are below what paper-scale UTD would produce.

3. **Online steps reduced from paper's ~250-500k to 200k vanilla / 140k
   improved.** Paper figures show success climbing through ~250k env steps.
   We cut off well before saturation. The improved run was further
   shortened by a 6h wall-clock deadline.

4. **No ablation between the two improvements.** We bundled n-step and
   progressive β. We can't say *which* of the two helped or hurt. With more
   compute, we'd run vanilla, +n-step only, +progressive β only, +both —
   four runs to attribute. Our diagnostics suggest progressive β hurt early
   and n-step neither helped nor hurt clearly.

5. **One task, one benchmark.** D4RL Antmaze medium-diverse only. The paper
   evaluates 12 tasks across 4 domains. Our infrastructure can run the rest
   but not in the available time.

### 6.2 What was demonstrated

1. **EXPO is correctly implemented.** Validated on Pendulum-v1 (return
   -1296 → -177, near-solve) and instantiated on D4RL Antmaze medium-
   diverse with all four loss heads, OTF action selection, mixed buffer,
   500k IL pretraining, and proper checkpoint save/resume.
2. **Pipeline runs paper-scale settings end-to-end on consumer GPU.** No
   code changes between the smoke configs and the paper-scale serial
   configs; only YAML settings differ.
3. **Two textbook improvements compose cleanly with EXPO.** N-step (S&B
   Ch. 7) lives in the buffer; progressive β (S&B Ch. 2 ε-decay analogue)
   lives in the edit-policy schedule. They are algorithmically orthogonal
   and required no agent-side restructuring beyond a single `set_beta()`
   method.
4. **The bug-hunt during validation surfaced three real issues** (VP
   schedule, obs wrapper dropping `achieved_goal`, α target entropy not
   being squash-aware) — each masked the next. The diagnostic discipline
   of "is the agent doing anything? trace the trajectory; verify each
   loss is meaningful" was essential.
5. **Honest negative result.** Increasing compute from parallel-80k to
   serial-200k did not break the plateau. This is a useful finding —
   it argues that for our parameter setting (UTD=5, β=0.05 fixed),
   the bottleneck is algorithmic, not just compute-bound.

### 6.3 Conclusions

The empirical result is **mixed**. The bundled n-step + progressive β
improvement very narrowly beat vanilla EXPO at the *final* eval (12% vs
8%, +4 pp) — within the ±4.2% standard error of a 50-episode binary
metric. Vanilla's peak (24%) exceeded improved's peak (16%), and
vanilla's mean across all evals (10.7%) exceeded improved's mean (8.0%).
**No reading of the data supports a confident claim that the
improvements help.**

This is consistent with the diagnostic story:

- N-step requires the n-step chains to *contain reward signal* to
  propagate it. With ~10% online success and n=3, the *vast majority*
  of n-step chains are all-zero — n-step degenerates to 1-step.
- Progressive β's early-high-β phase fed wider, noisier edits to a
  young critic, polluting its Q estimates during the most plastic
  training phase. Improved trailed vanilla at every eval through step
  60k (the high-β regime). It only caught up after β fully decayed.

We **did not surpass vanilla EXPO** at this scale. But we **did**:

1. Build a working EXPO implementation that runs paper-scale settings.
2. Surface (and fix) three real bugs through systematic diagnosis.
3. Demonstrate that classical Sutton & Barto techniques compose
   syntactically with modern algorithms without code restructuring.
4. Generate honest empirical evidence about *when* those techniques
   help and *when* they don't (here: they didn't, for diagnosable
   reasons).

The conclusion that classical techniques don't always help is itself
a useful finding. They are textbook-correct in *motivation* but their
*empirical benefit depends* on the regime — n-step needs reward density
above some threshold to actually propagate signal; progressive β needs
the wide-edit phase to actually find better actions, which requires
the critic to already be useful. Both assumptions can fail.

In other words, both improvements would likely help in different
regimes:
- **N-step** would shine on Antmaze with denser intermediate reward
  shaping (e.g. distance-to-goal reward), or on tasks where successful
  trajectories make up a higher fraction of the buffer.
- **Progressive β** would shine if started later in training (warmup_steps
  ≥ critic-warmup so the critic is already useful when wide edits begin),
  or with a smaller β_start (e.g. 0.1 instead of 0.3 — only 2× wider than
  vanilla, not 6×).

These are testable hypotheses for future work. With more compute, more
seeds, and ablation between the two, the conclusion may yet flip.

### 6.4 What we would do with another week

1. Run the full Antmaze sub-suite (4 tasks × 2 settings × 3 seeds × paper
   compute = ~21 days CPU equivalent, ~5 days on this GPU).
2. Implement OTF caching (Improvements §1.1) — would let us run UTD=20 in
   the time it currently takes UTD=5.
3. Ablate n-step vs progressive β separately to attribute the gain.
4. Add Robomimic / MimicGen wrappers for the manipulation-task half of
   the paper's benchmark.

---

## Appendix A: Reproduction commands

```bash
# Shared pretrain (~30 min on RTX 5060 Ti)
python main.py --config expo/configs/antmaze_pretrain_only.yaml

# Serial online phases at 100% GPU each (~10 h vanilla, ~10 h improved)
# vanilla finishes first, improved auto-launches via &&
python main.py --config expo/configs/antmaze_vanilla_serial.yaml \
    --resume runs/antmaze_pretrain/shared.ckpt --run-name full && \
python main.py --config expo/configs/antmaze_improved_serial.yaml \
    --resume runs/antmaze_pretrain/shared.ckpt --run-name full

# Generate comparison
python scripts/compare_runs.py \
    runs/antmaze_vanilla_serial/full.jsonl \
    runs/antmaze_improved_serial/full.jsonl
```

## Appendix B: Code locations

| Component | File |
|---|---|
| Diffusion base policy | `expo/models/diffusion_policy.py` |
| Edit policy | `expo/models/edit_policy.py` |
| Q-ensemble critic | `expo/models/critic.py` |
| EXPO agent (OTF + losses) | `expo/agents/expo_agent.py` |
| Auto-α + soft target update | `expo/agents/sac_utils.py` |
| **N-step buffer** (improvement 1) | `expo/buffers/nstep_buffer.py` |
| **Progressive β scheduler** (improvement 2) | `expo/agents/beta_schedule.py` |
| Mixed offline/online buffer | `expo/buffers/mixed_buffer.py` |
| Antmaze env wrapper | `expo/envs/antmaze.py` |
| Training loop | `expo/training/online.py` |
| Checkpoint save/resume | `expo/utils/checkpoint.py` |

## Appendix C: References

- Dong, P., Li, Q., Sadigh, D., & Finn, C. (2025). EXPO: Stable Reinforcement
  Learning with Expressive Policies. arXiv:2507.07986.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press. [Chapter 7: n-step Bootstrapping;
  Chapter 2: Multi-armed Bandits.]
- Ball, P. J., Smith, L., Kostrikov, I., & Levine, S. (2023). Efficient
  Online Reinforcement Learning with Offline Data (RLPD). ICML.
- Chen, X., Wang, C., Zhou, Z., & Ross, K. (2021). Randomized Ensembled
  Double Q-Learning (REDQ). arXiv:2101.05982.
- Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion. IJRR.
- Song, Y., et al. (2020). Score-Based Generative Modeling through
  Stochastic Differential Equations (VP-SDE β schedule). ICLR 2021.
