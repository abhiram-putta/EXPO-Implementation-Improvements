"""Generate a markdown table suitable for dropping into RESULTS.md.

Usage:
    python scripts/submission_summary.py runs/antmaze_submission/final.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt(v, prec=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.path).read_text().splitlines()]
    if not rows:
        print("Empty log.", file=sys.stderr)
        return

    pretrain_rows = [r for r in rows if r["step"] < 0]
    online_rows = [r for r in rows if r["step"] >= 0 and "critic/loss" in r]
    pretrain_end = next((r for r in rows if any(k.startswith("pretrain_end/") for k in r)), None)
    eval_rows = [r for r in rows if "eval/return_mean" in r]

    print("## Run summary")
    print()
    print(f"- Log: `{args.path}`")
    print(f"- Pretrain log rows: {len(pretrain_rows)}")
    print(f"- Online log rows: {len(online_rows)}")
    print(f"- Step range: {rows[0]['step']} -> {rows[-1]['step']}")
    print()

    if pretrain_rows:
        first = pretrain_rows[0].get("base/il_loss")
        last = pretrain_rows[-1].get("base/il_loss")
        print(f"### IL pretraining")
        print(f"- Loss: {fmt(first)} -> **{fmt(last)}** (over {len(pretrain_rows)} log rows)")
        print()

    if pretrain_end:
        sr = pretrain_end.get("pretrain_end/success_rate")
        rm = pretrain_end.get("pretrain_end/return_mean")
        print(f"### Eval at end of pretraining (IL-only baseline)")
        print(f"- Success rate: **{fmt(sr)}**")
        print(f"- Mean return: {fmt(rm)}")
        print()

    if eval_rows:
        print(f"### Online evals")
        print()
        print(f"| Step | Mean return | Min | Max | Success |")
        print(f"|---|---|---|---|---|")
        for r in eval_rows:
            print(f"| {r['step']} | {fmt(r.get('eval/return_mean'))} | "
                  f"{fmt(r.get('eval/return_min'))} | {fmt(r.get('eval/return_max'))} | "
                  f"**{fmt(r.get('eval/success_rate'))}** |")
        print()

    if online_rows:
        first, last = online_rows[0], online_rows[-1]
        print(f"### Algorithm-internal diagnostics")
        print()
        print(f"| Signal | Online start (step {first['step']}) | Online end (step {last['step']}) |")
        print(f"|---|---|---|")
        for k in ("critic/loss", "critic/mean_q", "critic/q_ensemble_std",
                  "base/il_loss", "edit/mean_edit_magnitude", "edit/alpha",
                  "otf/frac_edited_selected", "online/episode_return",
                  "online/success_rate", "online/steps_per_sec"):
            if k in first or k in last:
                print(f"| `{k}` | {fmt(first.get(k))} | {fmt(last.get(k))} |")
        print()


if __name__ == "__main__":
    main()
