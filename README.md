# EXPO — Reference Implementation + Improvements

A from-scratch PyTorch implementation of **EXPO (Expressive Policy
Optimization)** — Dong, Li, Sadigh & Finn (2025), *EXPO: Stable
Reinforcement Learning with Expressive Policies*, arXiv:2507.07986 — an
online RL fine-tuner for diffusion / flow-matching policies, plus two
textbook improvements drawn from Sutton & Barto's *Reinforcement Learning:
An Introduction*.

EXPO never backpropagates value gradients through the long diffusion
denoising chain (which is unstable). Instead it samples actions from a
frozen-ish base diffusion policy, perturbs them with a small, cheap-to-train
**edit policy**, and picks the highest-Q candidate with a critic ensemble —
an **on-the-fly (OTF)** policy. Only the lightweight edit policy and critic
ever see policy-gradient/TD updates.

## What's here

- **Full EXPO algorithm**: DDPM base policy, Gaussian edit policy, REDQ-style
  Q-ensemble critic, OTF action selection, mixed offline/online replay,
  auto-α (SAC entropy tuning), checkpoint save/resume.
- **Two improvements** (orthogonal, config-gated, drawn from Sutton & Barto):
  - **N-step returns** (Ch. 7) — faster credit assignment for Antmaze's
    sparse binary reward.
  - **Progressive β schedule** (Ch. 2 ε-decay analogue) — anneal the edit
    policy's exploration radius `β` from wide to tight over training.
- **Two environments**: `Pendulum-v1` (CPU smoke test) and `D4RL
  Antmaze medium-diverse` (via Minari; the paper's headline benchmark).
- **A bug-hunting validation pass** that found and fixed four real issues
  (diffusion noise schedule, a goal-observation bug, an entropy-target bug,
  an eval-horizon bug) — see [`RESULTS.md`](RESULTS.md).
- **A controlled A/B comparison** of vanilla EXPO vs. EXPO + improvements on
  Antmaze, run serially on the same hardware from the same pretrained
  checkpoint — see [`RESULTS_COMPARISON.md`](RESULTS_COMPARISON.md) and
  [`REPORT.md`](REPORT.md).

This is a **reference implementation built to validate correctness**, not a
paper-faithful benchmark reproduction (that would need roughly 50x more
compute and 3-seed averaging across the paper's 12 tasks). See
[Results](#results) and [Limitations](#limitations) below for the honest
read.

## Repo layout

```
expo/
  models/      Diffusion (DDPM) base policy, Gaussian edit policy, Q-ensemble critic
  agents/      EXPO agent (OTF + all loss heads), auto-α, progressive-β scheduler
  buffers/     Ring buffer, n-step buffer, 50/50 offline/online mixed sampler, Minari loader
  envs/        Pendulum + Antmaze (D4RL via Minari) wrappers, env registry
  training/    Online training loop with checkpointing + eval
  utils/       Seeding, config loading, JSONL logger, checkpoint save/resume
  configs/     YAML configs (pendulum, antmaze_vanilla/_improved/_submission/...)
scripts/       summarize_run.py, compare_runs.py, status.py, finalize.py, fill_report.py
tests/         Standalone sanity tests (no pytest dependency)
runs/          Per-run JSONL metric logs (checkpoints are git-ignored, see below)
main.py        CLI entry point
requirements.txt
rl_expo.pdf / rl_expo.txt   The original EXPO paper (PDF + extracted text)
```

## Documentation map

| File | Contents |
|---|---|
| [`EXPO_Implementation_Guide.md`](EXPO_Implementation_Guide.md) | Full algorithm spec this implementation follows |
| [`DECISIONS.md`](DECISIONS.md) | Every implementation choice and why (conventions, bug fixes, defaults) |
| [`IMPROVEMENT.md`](IMPROVEMENT.md) | Motivation + implementation walkthrough for the two improvements |
| [`EXPO_Improvements.md`](EXPO_Improvements.md) | Long list of candidate improvements considered (not all implemented) |
| [`RESULTS.md`](RESULTS.md) | Pendulum + Antmaze validation results, full bug-hunt writeup |
| [`RESULTS_COMPARISON.md`](RESULTS_COMPARISON.md) | Raw vanilla-vs-improved eval/return tables |
| [`REPORT.md`](REPORT.md) | Full writeup: intro, method, replication, improvements, A/B comparison, discussion |
| `rl_expo.pdf` / `rl_expo.txt` | The original EXPO paper |

## Install

```bash
python -m venv .venv
.venv/Scripts/activate    # Windows; use .venv/bin/activate on Unix
pip install -r requirements.txt

# For Antmaze benchmarks (Minari + MuJoCo):
pip install minari[hf,hdf5] gymnasium-robotics mujoco

# For GPU on Blackwell (RTX 50-series): use CUDA 12.8 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Tested on Python 3.10+ (developed on 3.13). CPU-only by default; GPU
auto-detected via `device: auto` in configs.

## Quick start — Pendulum-v1 (CPU smoke test, ~10 min)

```bash
python main.py --config expo/configs/pendulum.yaml
```

Episode return climbs from random (~-1200) toward ~-200; one eval episode
typically reaches close to -1 (near-optimal).

## Real run — D4RL Antmaze (paper benchmark, GPU recommended)

First download the offline dataset:
```bash
python -c "import minari; minari.download_dataset('D4RL/antmaze/medium-diverse-v1')"
```

### Single run (vanilla EXPO)
```bash
python main.py --config expo/configs/antmaze_submission.yaml --save-every 5000
```
~1.5 h on an RTX 5060 Ti.

### Two-run comparison (vanilla vs. +improvements), shared pretrain
```bash
# 1. shared pretrain (~30 min, GPU exclusive)
python main.py --config expo/configs/antmaze_pretrain_only.yaml

# 2. online phases — both resume from the same shared.ckpt
python main.py --config expo/configs/antmaze_vanilla.yaml \
    --resume runs/antmaze_pretrain/shared.ckpt --run-name vanilla
python main.py --config expo/configs/antmaze_improved.yaml \
    --resume runs/antmaze_pretrain/shared.ckpt --run-name improved

# 3. comparison
python scripts/compare_runs.py \
    runs/antmaze_vanilla/vanilla.jsonl \
    runs/antmaze_improved/improved.jsonl
```

## Inspecting runs

```bash
python scripts/status.py                 # live snapshot of in-progress runs
python scripts/status.py --follow        # auto-refresh every 60s
python scripts/summarize_run.py runs/<run>/<name>.jsonl
python scripts/compare_runs.py <vanilla.jsonl> <improved.jsonl>
```

## Tests

Standalone sanity tests, no pytest required:

```bash
python tests/test_buffer.py              # replay + mixed buffer shapes/ratios
python tests/test_otf.py                 # OTF action selection correctness
python tests/test_diffusion_overfit.py   # diffusion policy overfits a tiny dataset
python tests/test_nstep_buffer.py        # n-step return computation vs. hand-checked values
```

## Checkpoint / resume

All long runs auto-checkpoint to `<log_dir>/<run_name>.ckpt` every
`save_every_steps` env steps. Resume any time:

```bash
python main.py --config <yaml> --resume <path>.ckpt --run-name resumed
```

Restores agent (4 networks + α + 4 optimizers) + online buffer + RNG state
+ env_step. The in-progress episode is discarded (env is reset). Pretraining
is skipped on resume.

A `save_pretrain_checkpoint` config option additionally saves a checkpoint
at the end of pretraining specifically — used for the vanilla/improved
comparison so both runs start from identical pretrained weights.

**Note:** `.ckpt` files are large (250–300 MB each) and are excluded from
this repo via `.gitignore` (GitHub's hard limit is 100 MB/file). The
`runs/*/*.jsonl` metric logs are kept since they're small and are what the
analysis scripts (`compare_runs.py`, `summarize_run.py`) read.

## Improvements (n-step + progressive β)

The two improvements are config-driven and orthogonal to vanilla EXPO. To
enable both in a config:

```yaml
n_step: 3                       # n-step TD targets (S&B Ch. 7)
progressive_beta:               # cosine anneal β over training
  start: 0.3
  end: 0.05
  warmup_steps: 0
  decay_steps: 50000
```

Vanilla configs simply omit `n_step` (or set it to 1) and `progressive_beta`.
See [`IMPROVEMENT.md`](IMPROVEMENT.md) for full motivation, code walkthrough,
and Sutton & Barto references.

## Results

Headline numbers from the controlled A/B comparison (same hardware, same
500k-step pretrained checkpoint, same seed, 50 eval episodes/point — full
detail in [`RESULTS_COMPARISON.md`](RESULTS_COMPARISON.md)):

| Metric | Vanilla EXPO | EXPO + improvements |
|---|---|---|
| Final eval success | 8.0% | **12.0%** |
| Best eval success | **24.0%** | 16.0% |
| Final mean return | 32.3 | **69.4** |

**Honest read:** the 4-point final-eval gap is within the noise band of a
50-episode binary metric — not a confident win either way. Pendulum-v1
validation is unambiguous, though: episode return climbs -1296 → -177 with
healthy internal diagnostics throughout (critic value spread bounded, OTF
edits selected the majority of the time, α auto-tuned down). See
[`REPORT.md`](REPORT.md) for the full discussion of *why* the improvements
show mixed results on Antmaze.

The validation pass also caught and fixed four real bugs during
implementation (wrong diffusion noise schedule for small `T`, a dropped
goal-relevant observation field, a squash-unaware SAC entropy target, and a
structurally-too-short eval horizon) — see [`RESULTS.md`](RESULTS.md) for
the full bug-hunt writeup.

## Limitations

- **Single seed** per config (the paper uses 3) — eval noise dominates small gaps.
- **UTD=5** vs. the paper's UTD=20 — slower per-step sample efficiency.
- **Online steps** (140k–200k) well below the paper's ~250–500k.
- **One task** (Antmaze medium-diverse) of the paper's 12-task, 4-domain suite.
- **No ablation** isolating n-step from progressive-β — they were evaluated bundled together.

Full discussion in [`REPORT.md`](REPORT.md).

## References

- Dong, P., Li, Q., Sadigh, D., & Finn, C. (2025). *EXPO: Stable
  Reinforcement Learning with Expressive Policies.* arXiv:2507.07986.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press. — Ch. 2 (ε-decay), Ch. 7 (n-step
  bootstrapping).
- Ball, P. J., Smith, L., Kostrikov, I., & Levine, S. (2023). *Efficient
  Online RL with Offline Data* (RLPD). ICML.
- Chen, X., Wang, C., Zhou, Z., & Ross, K. (2021). *Randomized Ensembled
  Double Q-Learning* (REDQ). arXiv:2101.05982.
