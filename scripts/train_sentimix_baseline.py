"""
scripts/train_sentimix_baseline.py
==================================
CLI entrypoint to train the SentiMix Hindi-English 3-class sentiment baseline.

Usage:
    python scripts/train_sentimix_baseline.py
    python scripts/train_sentimix_baseline.py --epochs 10 --batch-size 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.sentiment_analysis.sentimix_baseline import train_sentimix_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SentiMix Hindi-English code-mixed sentiment baseline (3 classes)."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "data" / "processed" / "sentimix_train_canonical.csv",
        help="Path to SentiMix canonical training CSV (default: data/processed/sentimix_train_canonical.csv)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=ROOT / "data" / "processed" / "sentimix_test_canonical.csv",
        help="Path to SentiMix canonical test CSV (default: data/processed/sentimix_test_canonical.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "sentiment_training" / "sentimix_hinglish_baseline_v1",
        help="Directory to save versioned artifacts (default: data/sentiment_training/sentimix_hinglish_baseline_v1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs over training data (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Mini-batch size for online SGD training (default: 1000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="L2 regularization constant for SGDClassifier (default: 1e-4)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("PolicyLens: SentiMix Hindi-English Sentiment Training Baseline")
    print("=" * 70)

    res = train_sentimix_baseline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        random_state=args.random_state,
    )

    ev = res["evaluation"]
    print("\n" + "=" * 70)
    print("Final Evaluation Results on Official Held-Out Test Set (3,000 rows):")
    print(f"  Overall Accuracy : {ev['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1         : {ev['macro_f1']:.4f}")
    print(f"  Weighted-F1      : {ev['weighted_f1']:.4f}")
    print("  Per-class metrics:")
    for cls_name, metrics in ev["per_class"].items():
        print(f"    - {cls_name:8s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f} (Support: {metrics['support']})")
    print("  Confusion matrix:")
    for row in ev["confusion_matrix"]["matrix"]:
        print(f"    {row}")
    print("=" * 70)


if __name__ == "__main__":
    main()
