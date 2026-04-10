from __future__ import annotations

import argparse

from .config import load_config
from .trainer import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a slow-state cortical regularized world model experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Optional limit for smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)


if __name__ == "__main__":
    main()
