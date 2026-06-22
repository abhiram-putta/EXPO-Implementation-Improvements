"""Generate RESULTS_COMPARISON.md from two run JSONL logs.

Usage:
    python scripts/compare_runs.py runs/antmaze_vanilla/vanilla.jsonl runs/antmaze_improved/improved.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def fmt(v, prec=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{prec}g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vanilla", type=str)
    ap.add_argument("improved", type=str)
    ap.add_argument("--out", type=str, default="RESULTS_COMPARISON.md")
    args = ap.parse_args()

    v_rows = load(Path(args.vanilla))
    i_rows = load(Path(args.improved))

    v_evals = [r for r in v_rows if "eval/return_mean" in r]
    i_evals = [r for r in i_rows if "eval/return_mean" in r]

    v_train = [r for r in v_rows if "critic/loss" in r]
    i_train = [r for r in i_rows if "critic/loss" in r]

    out: list[str] = []
    out.append("# RESULTS_COMPARISON: Vanilla EXPO vs EXPO+Improvements")
    out.append("")
    out.append("Both runs started from the same shared 500k IL pretrain checkpoint.")
    out.append("Same agent (hidden=256, T=10, ensemble=10, N=8), same UTD=5,")
    out.append("same 80k online env steps, same seed, run in parallel on the same GPU.")
    out.append("")
    out.append("**Differences (configuration only):**")
    out.append("")
    out.append("| Setting | Vanilla | Improved |")
    out.append("|---|---|---|")
    out.append("| `n_step` | 1 | **3** (Sutton & Barto Ch. 7) |")
    out.append("| β | 0.05 fixed | **0.3 → 0.05 cosine** (S&B Ch. 2 ε-decay analogue) |")
    out.append("")

    out.append("## Eval success rate over training")
    out.append("")
    out.append("| Step | Vanilla success | Improved success | Δ (improved − vanilla) |")
    out.append("|---|---|---|---|")
    # Align by step
    by_step: dict[int, dict] = {}
    for r in v_evals:
        by_step.setdefault(r["step"], {})["v"] = r
    for r in i_evals:
        by_step.setdefault(r["step"], {})["i"] = r
    for step in sorted(by_step):
        v = by_step[step].get("v", {})
        i = by_step[step].get("i", {})
        vs = v.get("eval/success_rate")
        is_ = i.get("eval/success_rate")
        delta = (is_ - vs) if (vs is not None and is_ is not None) else None
        out.append(f"| {step} | {fmt(vs)} | {fmt(is_)} | {fmt(delta, 2)} |")
    out.append("")

    out.append("## Eval mean return over training")
    out.append("")
    out.append("| Step | Vanilla return | Improved return | Δ |")
    out.append("|---|---|---|---|")
    for step in sorted(by_step):
        v = by_step[step].get("v", {})
        i = by_step[step].get("i", {})
        vr = v.get("eval/return_mean")
        ir = i.get("eval/return_mean")
        delta = (ir - vr) if (vr is not None and ir is not None) else None
        out.append(f"| {step} | {fmt(vr)} | {fmt(ir)} | {fmt(delta)} |")
    out.append("")

    out.append("## Final result")
    out.append("")
    if v_evals and i_evals:
        v_final = v_evals[-1].get("eval/success_rate", 0)
        i_final = i_evals[-1].get("eval/success_rate", 0)
        out.append(f"- **Vanilla EXPO**: final success rate = **{v_final*100:.1f}%**")
        out.append(f"- **EXPO + Improvements**: final success rate = **{i_final*100:.1f}%**")
        out.append("")
        gain = (i_final - v_final) * 100
        if gain > 0:
            out.append(f"**Improved beats vanilla by {gain:.1f} percentage points.**")
        elif gain == 0:
            out.append("**Improved matched vanilla (same final success rate).**")
        else:
            out.append(f"**Vanilla beat improved by {-gain:.1f} percentage points.**")

    out.append("")
    out.append("## Best result over training")
    if v_evals and i_evals:
        v_best = max(r.get("eval/success_rate", 0) for r in v_evals)
        i_best = max(r.get("eval/success_rate", 0) for r in i_evals)
        out.append(f"- Vanilla peak: {v_best*100:.1f}%")
        out.append(f"- Improved peak: {i_best*100:.1f}%")
    out.append("")

    if v_train and i_train:
        out.append("## Algorithm-internal diagnostics")
        out.append("")
        out.append("Last logged training step:")
        out.append("")
        out.append("| Metric | Vanilla | Improved |")
        out.append("|---|---|---|")
        v_last = v_train[-1]
        i_last = i_train[-1]
        for k in ("critic/loss", "critic/mean_q", "critic/q_ensemble_std",
                  "edit/mean_edit_magnitude", "edit/alpha", "agent/beta",
                  "agent/target_entropy", "otf/frac_edited_selected",
                  "online/episode_return", "online/success_rate"):
            if k in v_last or k in i_last:
                out.append(f"| `{k}` | {fmt(v_last.get(k))} | {fmt(i_last.get(k))} |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {args.out}  ({len(out)} lines, {sum(len(l) for l in out)} chars)")


if __name__ == "__main__":
    main()
