from __future__ import annotations

import argparse
import json

from expo.training.online import train
from expo.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="EXPO trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name (else timestamp is used)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .ckpt to resume from")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Override save_every_steps in config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.save_every is not None:
        cfg["save_every_steps"] = args.save_every

    if args.dry_run:
        print(json.dumps(cfg, indent=2, default=str))
        return

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
