"""End-of-run finalization: produce RESULTS_COMPARISON.md + filled REPORT.md.

Run this once both vanilla and improved online phases have finished.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        print(f"  -> FAILED with exit code {proc.returncode}")
    return proc.returncode


def main() -> None:
    print("=" * 60)
    print("  FINALIZING REPORT")
    print("=" * 60)

    vanilla = REPO / "runs/antmaze_vanilla_serial/full.jsonl"
    improved = REPO / "runs/antmaze_improved_serial/full.jsonl"

    if not vanilla.exists():
        print(f"ERROR: {vanilla} not found")
        sys.exit(1)
    if not improved.exists():
        print(f"ERROR: {improved} not found")
        sys.exit(1)

    print(f"\n[1/2] Generating RESULTS_COMPARISON.md...")
    rc = run([
        sys.executable, "scripts/compare_runs.py",
        str(vanilla), str(improved),
        "--out", "RESULTS_COMPARISON.md",
    ])
    if rc != 0:
        sys.exit(rc)

    print(f"\n[2/2] Filling placeholders in REPORT.md...")
    rc = run([sys.executable, "scripts/fill_report.py"])
    if rc != 0:
        sys.exit(rc)

    # Sanity: any remaining placeholders?
    text = (REPO / "REPORT.md").read_text(encoding="utf-8")
    leftover = re.findall(r"__[A-Z_]+__", text)
    if leftover:
        print(f"\nWARNING: {len(set(leftover))} unique placeholders not filled:")
        for p in sorted(set(leftover)):
            print(f"  {p}")
    else:
        print(f"\nAll placeholders filled. Final REPORT.md ready.")

    word_count = len(text.split())
    print(f"\nFinal REPORT.md word count: {word_count} (target 3000-4000 = 6-8 pages)")
    print(f"\nFiles ready for submission:")
    print(f"  - REPORT.md          ({word_count} words)")
    print(f"  - RESULTS_COMPARISON.md")
    print(f"  - IMPROVEMENT.md")
    print(f"  - RESULTS.md (earlier replication writeup)")
    print(f"  - DECISIONS.md")
    print(f"  - README.md")


if __name__ == "__main__":
    main()
