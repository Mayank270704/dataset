#!/usr/bin/env python
"""
scripts/run_final_stance_inference_v1.py
========================================
Standalone reproducible script for the final heuristic stance inference pipeline (v1).

Usage
-----
# Run on a real CSV:
    python scripts/run_final_stance_inference_v1.py \\
        --input my_comments.csv \\
        --text-column text \\
        --out results/stance_output.csv \\
        --model linsvm

# Run the built-in smoke test (writes to data/inference/final_stance_inference_v1/):
    python scripts/run_final_stance_inference_v1.py --smoke-test [--model logreg]

Exit codes
----------
0  success
1  error (model load failure, missing input file, etc.)
2  smoke-test output directory already exists (to protect existing artifacts)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── make the project root importable ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.stance_inference_v2 import FinalStancePredictor  # noqa: E402

# ── smoke-test cases ──────────────────────────────────────────────────────────
SMOKE_TEST_TEXTS = [
    "",                                                                        # empty
    "   ",                                                                     # whitespace only
    "ok",                                                                      # too short (2 chars)
    "I strongly support open government data publishing",                      # expected: pos-sentiment, Support
    "This policy is a terrible waste of money and should be stopped",          # expected: neg-sentiment → Oppose
    "The meeting was held on Tuesday",                                         # neutral/factual
    "Le gouvernement devrait publier toutes les données sur les contrats publics.",  # French
    "Yeh policy bilkul galat hai",                                             # Hinglish/Hindi-like, negative
]


def _sha256_file(p: Path) -> str:
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(out_dir: Path, output_csv: Path, summary_json: Path, model: str) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/run_final_stance_inference_v1.py",
        "model_variant": model,
        "stance_run": "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class",
        "sentiment_model": "data/sentiment_training/sentiment140_hashing_sgd_v1",
        "pipeline": "FinalStancePredictor (heuristic_oppose)",
        "disclaimer": (
            "Oppose predictions are heuristic (negative sentiment + non-Support stance); "
            "not produced by a validated Oppose classifier."
        ),
        "outputs_sha256": {
            output_csv.name: _sha256_file(output_csv),
            summary_json.name: _sha256_file(summary_json),
        },
    }
    return manifest


def run_smoke_test(model: str) -> int:
    out_dir = ROOT / "data" / "inference" / "final_stance_inference_v1"
    if out_dir.exists():
        print(
            f"\n[ERROR] Smoke-test output directory already exists: {out_dir}\n"
            "  Refusing to overwrite existing artifacts (exit code 2).\n"
            "  Delete or rename the directory manually if you want to re-run.",
            file=sys.stderr,
        )
        return 2

    print("=" * 72)
    print("  PolicyLens -- Final Stance Inference v1  [SMOKE TEST]")
    print("=" * 72)
    print(f"  Model variant  : {model}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Test cases     : {len(SMOKE_TEST_TEXTS)}")
    print()

    print("Loading models (this may take a few seconds) ...")
    try:
        predictor = FinalStancePredictor(model=model)
    except Exception as exc:
        print(f"\n[ERROR] Model loading failed: {exc}", file=sys.stderr)
        return 1

    print("Running inference ...\n")
    df = predictor.predict(SMOKE_TEST_TEXTS)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "smoke_test_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # ── summary ──────────────────────────────────────────────────────────────
    valid_df = df[df["input_status"] == "ok"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_inputs": len(df),
        "valid_inputs": int(valid_mask_count := (df["input_status"] == "ok").sum()),
        "rejected_inputs": int(len(df) - valid_mask_count),
        "final_stance_distribution": valid_df["final_stance"].value_counts().to_dict(),
        "prediction_method_distribution": valid_df["prediction_method"].value_counts().to_dict(),
        "sentiment_distribution": valid_df["sentiment_prediction"].value_counts().to_dict(),
        "model_variant": model,
        "note": (
            "Oppose is a heuristic (negative sentiment + non-Support stance). "
            "All stance outputs are AI-working-label approximations."
        ),
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = _build_manifest(out_dir, output_csv, summary_json, model)
    manifest_json = out_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ── stdout table ──────────────────────────────────────────────────────────
    print("-" * 72)
    print(f"  {'TEXT (truncated)':<42} {'STATUS':<20} {'STANCE':<10} {'METHOD'}")
    print("-" * 72)
    for _, row in df.iterrows():
        text_trunc = (row["text"][:40] + "...") if len(str(row["text"])) > 40 else str(row["text"])
        status = str(row["input_status"])
        stance = str(row["final_stance"]) if row["final_stance"] is not None else "-"
        method = str(row["prediction_method"]) if row["prediction_method"] is not None else "-"
        print(f"  {text_trunc:<42} {status:<20} {stance:<10} {method}")
    print("-" * 72)
    print()
    print("  Final-stance distribution (valid inputs only):")
    for label, cnt in summary["final_stance_distribution"].items():
        print(f"    {label:<10} {cnt}")
    print()
    print("  Prediction method distribution:")
    for method, cnt in summary["prediction_method_distribution"].items():
        print(f"    {method:<35} {cnt}")
    print()
    print(f"  Output CSV     : {output_csv}")
    print(f"  Summary JSON   : {summary_json}")
    print(f"  Manifest JSON  : {manifest_json}")
    print()
    print("  [SMOKE TEST PASSED]")
    return 0


def run_batch(input_csv: str, text_column: str, output_csv: str, model: str) -> int:
    in_path = Path(input_csv)
    out_path = Path(output_csv)

    if not in_path.is_file():
        print(f"\n[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("  PolicyLens -- Final Stance Inference v1  [BATCH MODE]")
    print("=" * 72)
    print(f"  Input file     : {in_path}")
    print(f"  Text column    : {text_column}")
    print(f"  Output file    : {out_path}")
    print(f"  Model variant  : {model}")
    print()

    df_in = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    if text_column not in df_in.columns:
        print(
            f"\n[ERROR] Column '{text_column}' not found in {in_path.name}. "
            f"Available: {list(df_in.columns)}",
            file=sys.stderr,
        )
        return 1

    texts = df_in[text_column].tolist()
    print(f"  Rows loaded    : {len(texts)}")
    print("Loading models ...")

    try:
        predictor = FinalStancePredictor(model=model)
    except Exception as exc:
        print(f"\n[ERROR] Model loading failed: {exc}", file=sys.stderr)
        return 1

    print("Running inference …")
    results = predictor.predict(texts)

    # Preserve all original columns alongside the inference output
    combined = pd.concat(
        [df_in.reset_index(drop=True), results.drop(columns=["text"]).reset_index(drop=True)],
        axis=1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8")

    valid_df = results[results["input_status"] == "ok"]
    print("\n  Final-stance distribution:")
    for label, cnt in valid_df["final_stance"].value_counts().items():
        print(f"    {label:<10} {cnt}")
    print(f"\n  Results saved to: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PolicyLens final heuristic stance inference pipeline v1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", metavar="CSV", help="Path to input CSV file.")
    parser.add_argument(
        "--text-column", default="text", metavar="COL",
        help="Name of the text column in the input CSV (default: 'text').",
    )
    parser.add_argument("--out", metavar="CSV", help="Path to output CSV file.")
    parser.add_argument(
        "--model", choices=["linsvm", "logreg"], default="linsvm",
        help="Stance model variant to use (default: linsvm).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Run on a small synthetic test set and write output to "
            "data/inference/final_stance_inference_v1/. "
            "Exits with code 2 if the directory already exists."
        ),
    )
    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test(args.model)

    if not args.input or not args.out:
        parser.error("--input and --out are required unless --smoke-test is specified.")

    return run_batch(args.input, args.text_column, args.out, args.model)


if __name__ == "__main__":
    sys.exit(main())
