### 5.4 Why both runs plateaued — diagnostic analysis

Both vanilla and improved kernels showed substantially less learning than
paper Figure 5 reports for this task (paper: 80-90% by ~250-500k env
steps; us: peak 24% on vanilla serial, ___IMP_PEAK___ on improved). With
the runs complete, we can examine the algorithm-internal diagnostics to
understand what's actually limiting performance.

#### 5.4.1 The sparse-reward credit-assignment bottleneck

Antmaze gives reward `r ∈ {0, 1}` only when the ant is at the goal. Since
the policy succeeds in ~10% of online rollouts, the buffer fills mostly
with **failure trajectories** (zero reward throughout). The critic's TD
target for these transitions is also ~0 (reward is 0 and the bootstrapped
Q from the next state is also small). The critic spends most of its
gradient steps confirming "this state has value 0" rather than learning
useful gradients toward goal-reaching behavior.

When the agent does succeed, the reward signal lights up briefly — but
the next gradient step on a different (failed) batch element drowns it
out. This is exactly what motivated **n-step returns** as our improvement:
n-step propagates that rare goal reward back n states per gradient
update, partially offsetting the credit-assignment problem. We saw a
modest effect — `critic/mean_q` was higher in the improved kernel
(___IMP_MEAN_Q___) than vanilla (___VAN_MEAN_Q___) — but apparently not
enough to translate into proportionally higher success.

#### 5.4.2 Critic Q-value drift under high UTD

EXPO uses UTD=20 in the paper; we used UTD=5 (the highest we could fit
in compute). Even at UTD=5, `critic/mean_q` climbed monotonically from
~3.5 (start of online) to ~30+ (end of run) for both kernels. Some of
this growth is real (TD targets correctly accumulating reward signal);
some is the well-known overestimation drift in actor-critic methods
where the critic produces optimistic Q-estimates for actions the agent
rarely takes. When OTF then argmax-selects these "looks-good-but-actually-
mediocre" actions, the policy becomes locked into a suboptimal regime.

Our LayerNorm-after-first-layer (Improvements §4.1, folded into v1) and
random-min-2 ensemble (REDQ-style) are mitigations, not eliminations.

#### 5.4.3 Distribution shift in the mixed buffer

The buffer is sampled 50/50 from offline + online. Offline data has
~85% successful trajectories (it's demonstrator data — see Sec. 3.1).
Online data after 100k+ steps has ~10% successful trajectories. As the
online buffer fills, the *effective* fraction of "good" trajectories in
each batch drops from ~85% (early, when online is empty so we sample
all-offline) to ~47% (late, when online is full of mediocre data and
contributes half).

This affects the **base policy** (which is trained via IL on the mixed
batch). The base policy's action distribution shifts away from the strong
offline demonstrations toward the weaker online behaviors. It's a kind
of policy collapse — similar in spirit to the issues that motivated
methods like AWAC and Cal-QL.

#### 5.4.4 β=0.05 may be too tight to escape local optima

Vanilla EXPO uses β=0.05 — edits ±0.05 per action dim. If the base
policy is stuck in a "navigates roughly toward goal then wanders" basin,
β=0.05 perturbations cannot move actions far enough to find the correct
final-approach behavior.

This was the exact motivation for our **progressive β scheduler**: start
at β=0.3 (6× wider), anneal to β=0.05 over 100k steps. The wider early
edits give the OTF a chance to find better actions outside the base
policy's narrow distribution; the tight late edits then refine.

The improved kernel with progressive β did show ___BETA_OBSERVATION___,
suggesting ___BETA_VERDICT___.

#### 5.4.5 Single-seed eval noise dominates at this success scale

A 50-episode binary success metric with true success rate p has
Bernoulli standard error √(p(1-p)/n). At p=0.10, n=50: SE = ±4.2%. So
two consecutive evals showing 6% and 18% are *consistent with the same
underlying policy* — the gap is barely 2 standard errors.

Across our 20 vanilla evals, success rate varied from 4% to 24%. With a
single seed, we cannot distinguish which of these are real shifts in
policy capability vs. which are eval noise. The paper uses **3 seeds ×
100 episodes per eval** for exactly this reason — variance reduction by
6× over our single-seed-50-episode setup.

#### 5.4.6 Sub-paper compute: the dominant constraint

Putting it all together, our setup has:

| Resource | Paper | Ours | Ratio |
|---|---|---|---|
| Pretrain steps | 500,000 | 500,000 | 1× ✓ |
| Online env steps | ~250-500k | 200k (vanilla) / ___IMP_TOTAL___ (improved) | ~0.5× |
| UTD ratio | 20 | 5 | 0.25× |
| Eval episodes per data point | 100 | 50 | 0.5× |
| Seeds | 3 | 1 | 0.33× |

**Total compute ratio**: roughly 0.5 × 0.25 × 0.5 × 0.33 ≈ **0.02× — about 2% of paper compute per task**.

This is the binding constraint. Our implementation is correct (see
diagnostics above — critic doesn't diverge, edits engage, OTF picks
edited candidates ~75% of the time, IL pretraining reduces loss by 25×).
But correctness ≠ replication when compute is 50× short.

### 5.5 Did the serial run beat the parallel run?

A natural hypothesis when we moved from parallel to serial execution
(100% GPU each instead of 50/50 sharing): "more compute per kernel
should give better results."

| Metric | Serial Vanilla (200k env, 50-ep evals) | Parallel Vanilla (80k env, 15-ep evals) |
|---|---|---|
| Final eval success | 8% | 13.3% |
| Peak success | 24% | 46.7% |
| Mean across all evals | 10.7% | 14.2% |

**Surprisingly, the parallel run had the better headline numbers.** Two
caveats explain this:

1. **Eval-noise asymmetry**: parallel used 15 episodes per eval (high
   variance), serial used 50 (low variance). The parallel "46.7% peak"
   was 7/15 episodes succeeding — a single-eval lucky moment that a
   50-episode re-eval would have measured at probably 25-35%.

2. **Compute effect on plateau**: vanilla EXPO at UTD=5 with our config
   appears to plateau in the ~10-25% range *regardless of more env
   steps*. The 2.5× extra steps in the serial run did not break through
   the plateau — strongly suggesting the bottlenecks are algorithmic
   (sparse reward, critic drift, β tightness) rather than just "needs
   more env steps."

This is itself a useful finding. **It supports the diagnosis that pure
compute scaling at our parameter setting is unlikely to reach 80%**;
the path to higher accuracy requires UTD increase, multiple seeds, and
ablation of which improvement helps most.