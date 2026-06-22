"""Live status of the parallel Antmaze runs.

Usage:
    python scripts/status.py
    python scripts/status.py --follow         # auto-refresh every 60s
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


VANILLA = Path("runs/antmaze_vanilla_serial/full.jsonl")
IMPROVED = Path("runs/antmaze_improved_serial/full.jsonl")
VANILLA_CKPT = Path("runs/antmaze_vanilla_serial/full.ckpt")
IMPROVED_CKPT = Path("runs/antmaze_improved_serial/full.ckpt")
TARGET_STEPS = 200000


def load_rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return []


def fmt(v, prec=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    return str(v)


def summarize(name: str, rows: list[dict], ckpt: Path) -> str:
    if not rows:
        return f"  {name}: NOT STARTED (no log file)"
    online = [r for r in rows if r.get("step", 0) >= 0 and "critic/loss" in r]
    evals = [r for r in rows if "eval/return_mean" in r]
    last = online[-1] if online else None
    last_eval = evals[-1] if evals else None
    last_step = last["step"] if last else (rows[-1].get("step", "?"))
    pct = (last_step / TARGET_STEPS * 100) if isinstance(last_step, int) and last_step >= 0 else 0.0

    sps = last.get("online/steps_per_sec", 0) if last else 0
    eta_s = (TARGET_STEPS - last_step) / max(sps, 0.01) if isinstance(last_step, int) else 0
    eta_h = eta_s / 3600

    out = []
    out.append(f"  {name}:")
    out.append(f"    progress      : step {last_step} / {TARGET_STEPS} ({pct:.1f}%)")
    out.append(f"    speed         : {fmt(sps)} env-steps/sec")
    out.append(f"    ETA           : {eta_h:.2f} h")
    if last:
        out.append(f"    critic/loss   : {fmt(last.get('critic/loss'))}")
        out.append(f"    critic/mean_q : {fmt(last.get('critic/mean_q'))}")
        out.append(f"    agent/beta    : {fmt(last.get('agent/beta'))}")
        out.append(f"    edit/alpha    : {fmt(last.get('edit/alpha'))}")
    if last_eval:
        sr = last_eval.get("eval/success_rate", 0)
        out.append(f"    last eval     : step {last_eval['step']}: success={sr*100:.1f}% (return mean {fmt(last_eval.get('eval/return_mean'))})")
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1e6
        mtime_age_s = time.time() - ckpt.stat().st_mtime
        out.append(f"    last ckpt     : {ckpt.name} ({size_mb:.0f} MB, {mtime_age_s/60:.1f} min old)")
    return "\n".join(out)


def show_once() -> None:
    print("=" * 72)
    print(f"EXPO parallel runs status   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print()
    print(summarize("VANILLA  EXPO", load_rows(VANILLA), VANILLA_CKPT))
    print()
    print(summarize("IMPROVED EXPO (n-step + progressive beta)", load_rows(IMPROVED), IMPROVED_CKPT))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true", help="Auto-refresh every 60s")
    args = ap.parse_args()
    if args.follow:
        try:
            while True:
                show_once()
                time.sleep(60)
        except KeyboardInterrupt:
            return
    else:
        show_once()


if __name__ == "__main__":
    main()
