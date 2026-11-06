"""Train the scoring/classification model reproducible.

Usage:
    python scripts/train.py --data data/processed --method baseline
    python scripts/train.py --data data/processed --method transformer [--epochs 3]

Saves artifacts under models/model/ (config.json, weights, metrics.json).
No weights are ever committed to git - the README documents how to reproduce.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.training import main as run_training

VALID_METHODS = ("baseline", "transformer")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the answer-evaluation model.")
    parser.add_argument("--data", default="data/processed", help="processed dataset dir")
    parser.add_argument("--method", choices=VALID_METHODS, default="baseline")
    parser.add_argument("--out", default="models/model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    if args.method == "transformer":
        from importlib.util import find_spec

        if find_spec("torch") is None or find_spec("transformers") is None:
            print(
                "torch/transformers are not installed - install "
                "requirements-ml.txt OR use --method baseline."
            )
            return 1

    print(f"Training `{args.method}` with seed={args.seed} ...")
    metrics = run_training(
        args.data,
        args.out,
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nModel bundle saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())