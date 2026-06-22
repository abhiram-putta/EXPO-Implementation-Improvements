# RESULTS_COMPARISON: Vanilla EXPO vs EXPO+Improvements (serial runs)

Both runs started from the same shared 500k IL pretrain checkpoint
(`runs/antmaze_pretrain/shared.ckpt`). Same agent (hidden=256, T=10,
ensemble=10, N=8), same UTD=5, **50 eval episodes per data point** (vs
15 in the earlier parallel comparison), same seed (42).

Both ran **serially on the same RTX 5060 Ti at 100% GPU throughput**
(not in parallel like the earlier comparison): vanilla first to
completion (200k env steps, ~10.7h wall clock), then improved (stopped
at 140k env steps when the 6h wall-clock deadline hit). Improved
therefore has 13 evals vs vanilla's 20.

**Differences (configuration only):**

| Setting | Vanilla | Improved |
|---|---|---|
| `n_step` | 1 | **3** (Sutton & Barto Ch. 7) |
| β | 0.05 fixed | **0.3 → 0.05 cosine** (S&B Ch. 2 ε-decay analogue) |

## Eval success rate over training

| Step | Vanilla success | Improved success | Δ (improved − vanilla) |
|---|---|---|---|
| 10000 | 0.06 | 0.02 | -0.04 |
| 20000 | 0.18 | 0.02 | -0.16 |
| 30000 | 0.12 | 0.08 | -0.04 |
| 40000 | 0.12 | 0.08 | -0.04 |
| 50000 | 0.08 | 0.02 | -0.06 |
| 60000 | 0.1 | 0.02 | -0.08 |
| 70000 | 0.1 | 0.16 | 0.06 |
| 80000 | 0.1 | 0.1 | 0 |
| 90000 | 0.1 | 0.1 | 0 |
| 100000 | 0.06 | 0.12 | 0.06 |
| 110000 | 0.04 | 0.12 | 0.08 |
| 120000 | 0.24 | 0.08 | -0.16 |
| 130000 | 0.08 | 0.12 | 0.04 |
| 140000 | 0.18 | - | - |
| 150000 | 0.16 | - | - |
| 160000 | 0.12 | - | - |
| 170000 | 0.06 | - | - |
| 180000 | 0.06 | - | - |
| 190000 | 0.1 | - | - |
| 200000 | 0.08 | - | - |

## Eval mean return over training

| Step | Vanilla return | Improved return | Δ |
|---|---|---|---|
| 10000 | 19.66 | 8.72 | -10.94 |
| 20000 | 70.58 | 12.94 | -57.64 |
| 30000 | 47.34 | 34.94 | -12.4 |
| 40000 | 62.02 | 30.58 | -31.44 |
| 50000 | 31.56 | 8.3 | -23.26 |
| 60000 | 45.72 | 0.08 | -45.64 |
| 70000 | 37.68 | 89.54 | 51.86 |
| 80000 | 65.94 | 64.72 | -1.22 |
| 90000 | 52.8 | 55.2 | 2.4 |
| 100000 | 32.06 | 78.94 | 46.88 |
| 110000 | 23.66 | 55.94 | 32.28 |
| 120000 | 131.1 | 54.92 | -76.18 |
| 130000 | 24.14 | 69.38 | 45.24 |
| 140000 | 85.78 | - | - |
| 150000 | 78.94 | - | - |
| 160000 | 71.68 | - | - |
| 170000 | 15.94 | - | - |
| 180000 | 32.32 | - | - |
| 190000 | 29.42 | - | - |
| 200000 | 32.28 | - | - |

## Final result

- **Vanilla EXPO**: final success rate = **8.0%**
- **EXPO + Improvements**: final success rate = **12.0%**

**Improved beats vanilla by 4.0 percentage points.**

## Best result over training
- Vanilla peak: 24.0%
- Improved peak: 16.0%

## Algorithm-internal diagnostics

Last logged training step:

| Metric | Vanilla | Improved |
|---|---|---|
| `critic/loss` | 0.3055 | 0.8037 |
| `critic/mean_q` | 27.8 | 25.24 |
| `critic/q_ensemble_std` | 0.1803 | 0.1914 |
| `edit/mean_edit_magnitude` | 0.03129 | 0.03184 |
| `edit/alpha` | 3.23e-04 | 3.08e-04 |
| `agent/beta` | 0.05 | 0.05 |
| `agent/target_entropy` | -26.42 | -26.42 |
| `otf/frac_edited_selected` | 0.6992 | 0.6484 |
| `online/episode_return` | 60.6 | 140.4 |
| `online/success_rate` | 0.2 | 0.2 |
