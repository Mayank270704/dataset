"""
scripts/train_sentiment140_baseline.py
======================================
CLI script to train and evaluate the Sentiment140 streaming baseline on all 1,600,000 records.

Usage:
    python scripts/train_sentiment140_baseline.py
    python scripts/train_sentiment140_baseline.py --batch-size 25000 --alpha 1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.sentiment_analysis.streaming_baseline import (
    DISCLAIMER_TEXT,
    train_streaming_sentiment140,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Sentiment140 streaming baseline (HashingVectorizer + SGDClassifier) on full 1.6M rows."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "trainingandtestdata" / "training.1600000.processed.noemoticon.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=ROOT / "trainingandtestdata" / "testdata.manual.2009.06.14.csv",
        help="Path to manual test CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1",
        help="Output directory for model and metrics",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports" / "sentiment140_baseline_report_v1.md",
        help="Path to generated markdown evaluation report",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25000,
        help="Batch size for incremental partial_fit (default: 25,000)",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=2**18,
        help="Number of hash features (default: 262,144)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="L2 regularization parameter (default: 1e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PolicyLens: Sentiment140 Streaming Baseline Training")
    print("=" * 80)
    print(f"Notice: {DISCLAIMER_TEXT}")
    print("-" * 80)

    try:
        results = train_streaming_sentiment140(
            train_path=args.train_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            report_path=args.report_path,
            batch_size=args.batch_size,
            n_features=args.features,
            alpha=args.alpha,
            random_state=args.seed,
        )
    except Exception as e:
        print(f"\nFATAL: Training failed: {e}", file=sys.stderr)
        return 1

    eval_m = results["evaluation"]["binary_evaluation"]
    cm = eval_m["confusion_matrix"]

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Completed Training Rows : {results['total_rows']:,} / 1,600,000 (100% full corpus)")
    print(f"  Training Runtime        : {results['training_time']:.2f} seconds")
    print(f"  Binary Test Size        : {eval_m['test_size']} records (177 Neg, 182 Pos)")
    print(f"  Macro-F1                : {eval_m['macro_f1']:.4f}")
    print(f"  Accuracy                : {eval_m['accuracy'] * 100:.2f}%")
    print(f"  Negative (0) F1         : {eval_m['per_class']['negative']['f1']:.4f} (P: {eval_m['per_class']['negative']['precision']:.4f}, R: {eval_m['per_class']['negative']['recall']:.4f})")
    print(f"  Positive (4) F1         : {eval_m['per_class']['positive']['f1']:.4f} (P: {eval_m['per_class']['positive']['precision']:.4f}, R: {eval_m['per_class']['positive']['recall']:.4f})")
    print(f"  Confusion Matrix        : TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")
    print(f"  Neutral Test Tweets     : {results['evaluation']['neutral_evaluation']['total_neutral_rows']} (Reported separately without forced mapping)")
    print(f"  Output Directory        : {results['output_dir']}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
