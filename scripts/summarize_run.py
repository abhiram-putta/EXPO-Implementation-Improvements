"""Summarize an EXPO JSONL log.

Usage:
    python scripts/summarize_run.py runs/pendulum/full.jsonl

Prints:
  - Final eval return + comparison vs first eval (learning delta)
  - Episode-return trajectory (rolling mean over 10 most recent)
  - Critic / edit-policy diagnostics at the start vs end
  - A simple ASCII sparkline of eval returns over time
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    blocks = ".:-=+*#@"  # ASCII low-to-high
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(len(values) - 1, int(i * step))] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / rng * (len(blocks) - 1)))]
                   for v in sampled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to JSONL log")
    args = ap.parse_args()

    rows = load(Path(args.path))
    if not rows:
        print("Empty log.", file=sys.stderr)
        return

    eval_rows = [(r["step"], r["eval/return_mean"]) for r in rows if "eval/return_mean" in r]
    train_rows = [r for r in rows if "critic/loss" in r]
    pretrain_rows = [r for r in rows if r["step"] < 0]

    print(f"=== {args.path} ===")
    print(f"Total log rows: {len(rows)}  (pretrain: {len(pretrain_rows)}, "
          f"online: {len(rows) - len(pretrain_rows)})")
    print(f"Step range: {rows[0]['step']} -> {rows[-1]['step']}")
    print()

    if eval_rows:
        print("Eval returns:")
        for step, ret in eval_rows:
            print(f"  step {step:>7d}  return_mean = {ret:>9.2f}")
        first, last = eval_rows[0][1], eval_rows[-1][1]
        print(f"\n  delta first->last:  {first:.2f} -> {last:.2f}  (improvement: {last - first:+.2f})")
        sline = sparkline([r for _, r in eval_rows])
        print(f"  trajectory:  {sline}")
        print()

    if train_rows:
        print("Critic + edit diagnostics (first vs last online log):")
        first, last = train_rows[0], train_rows[-1]
        keys = [
            "critic/loss", "critic/mean_q", "critic/q_ensemble_std",
            "edit/mean_edit_magnitude", "edit/alpha", "otf/frac_edited_selected",
            "online/episode_return",
        ]
        for k in keys:
            if k in first or k in last:
                f = first.get(k, float("nan"))
                l = last.get(k, float("nan"))
                print(f"  {k:>32s}: {f:>10.4g}  ->  {l:>10.4g}")

    if pretrain_rows:
        first_il = pretrain_rows[0].get("base/il_loss", float("nan"))
        last_il = pretrain_rows[-1].get("base/il_loss", float("nan"))
        print(f"\nIL pretraining loss: {first_il:.4f} -> {last_il:.4f}")


if __name__ == "__main__":
    main()
