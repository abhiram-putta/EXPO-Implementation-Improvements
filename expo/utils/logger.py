from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Logger:
    """Stdout + JSONL logger.

    Buffers metrics per-step until `commit(step)` is called. Each step gets
    one JSON line on disk; new fields can appear at any step.
    """

    def __init__(self, log_dir: str | Path, run_name: str | None = None,
                 print_every: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{self.run_name}.jsonl"
        self._fp = self.path.open("w", encoding="utf-8")
        self._buffer: dict[int, dict[str, Any]] = {}
        self.print_every = print_every

    def log(self, step: int, **metrics: Any) -> None:
        row = self._buffer.setdefault(step, {})
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v

    def commit(self, step: int) -> None:
        if step not in self._buffer:
            return
        row = {"step": step, **self._buffer.pop(step)}
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()
        if self.print_every and step % self.print_every == 0:
            kv = " | ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items() if k != "step"
            )
            print(f"[step {step:>7d}] {kv}")

    def close(self) -> None:
        for step in sorted(self._buffer):
            self.commit(step)
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
