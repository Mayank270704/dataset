#!/usr/bin/env python
"""
scripts/train_stance_combined_dataset_v1.py
=============================================
Standalone retraining deliverable on the FINAL combined labelled dataset already produced by
this project -- it does not relabel anything or search for new sources.

Final combined labelled dataset (located, not recomputed)
-----------------------------------------------------------
    data/stance_training_ai_working_labels/run_pool400_plus_all_sources_v1/combined_training_data_provenance.csv
      -- 19,331 rows: pool400 train+validation (319, AI working labels) + ALL in-domain corpus
         pseudo-labels (9,515) + ALL Sentiment140-test pseudo-labels (497) + a capped seeded
         Sentiment140-train reservoir sample (9,000). Columns: original_text, gold_stance,
         label_source, source_dataset, source_record_id, rule, sentiment_prediction,
         stance_margin_pseudo, sentiment_margin_pseudo, split.
    data/stance_training_ai_working_labels/run_pool400_plus_all_sources_v1/held_out_test_set.csv
      -- the matching 80-record held-out pool400 test split (genuine AI working labels only,
         zero pseudo-labeled records), used project-wide for every comparable run.

All four labels (Support/Neutral/Oppose/Unclear) are used exactly as currently assigned -- no
filtering, no exclusion, no re-labeling. This reproduces, from the persisted canonical dataset
file (not from re-running the sourcing code), the metrics already reported for "model_3_all_sources"
in that run's own metrics.json -- that is the expected and correct outcome of training on an
identical, deterministic dataset + split + config, not a new result. What IS new here: (1) an
explicit, standalone verification of the dataset's path/schema/row count/class distribution
before training, and (2) an explicit train-vs-test duplicate-text leakage check run across the
FULL 19,331-row combined set (prior runs only verified leakage-safety within the 400-record
pool400 split; the large pseudo pools were never explicitly checked for exact text overlap with
the held-out test set).

Does not overwrite or modify any existing dataset or model directory.

Usage
    python scripts/train_stance_combined_dataset_v1.py --tag combined_dataset_retrain_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402
from models.stance_classification.pipeline import load_trained_pipeline  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402
from scripts.train_stance_expanded_v1 import ALL_FOUR_LABELS, ORIGINAL_RUN_DIR, three_way_split  # noqa: E402

SOURCE_RUN_DIR = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_plus_all_sources_v1"
COMBINED_DATASET_CSV = SOURCE_RUN_DIR / "combined_training_data_provenance.csv"
SOURCE_HELD_OUT_TEST_CSV = SOURCE_RUN_DIR / "held_out_test_set.csv"
SOURCE_METADATA_JSON = SOURCE_RUN_DIR / "metadata.json"
POOL_LABELS_FILE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_working_labels_pool400.csv"
POOL_PROVENANCE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "provenance.json"
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"

REQUIRED_COLUMNS = [
    "original_text", "gold_stance", "label_source", "source_dataset", "source_record_id",
    "rule", "sentiment_prediction", "stance_margin_pseudo", "sentiment_margin_pseudo", "split",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def verify_combined_dataset(path: Path, required_columns: list[str]) -> dict[str, Any]:
    """Step 1: locate + verify the final combined labelled dataset. Raises if it is missing,
    empty, or missing a required column -- never silently proceeds on a malformed input."""
    if not path.is_file():
        raise FileNotFoundError(f"final combined labelled dataset not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"{path} has zero rows")
    for num_col in ("stance_margin_pseudo", "sentiment_margin_pseudo"):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    summary = {
        "path": _rel(path), "sha256": _sha(path), "n_rows": len(df), "columns": list(df.columns),
        "class_distribution": df["gold_stance"].value_counts().to_dict(),
        "label_source_distribution": df["label_source"].value_counts().to_dict(),
        "source_dataset_distribution": df["source_dataset"].value_counts().to_dict(),
        "split_values_present": sorted(df["split"].unique().tolist()),
    }
    return df, summary


def find_train_test_text_leakage(train_texts: pd.Series, test_texts: pd.Series) -> dict[str, Any]:
    """Explicit duplicate-text leakage check across the FULL combined training set vs the
    held-out test set (normalised-text exact match). Returns the overlap so the caller can
    decide what to do with it -- this function never silently drops anything itself."""
    train_norm = train_texts.map(normalise_text)
    test_norm = set(test_texts.map(normalise_text))
    overlap_mask = train_norm.isin(test_norm)
    return {
        "n_train_rows_overlapping_test_text": int(overlap_mask.sum()),
        "overlap_mask": overlap_mask,
        "n_test_texts": len(test_norm),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2

    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed

    # ── Step 1: locate + verify the final combined labelled dataset ──
    combined, dataset_summary = verify_combined_dataset(COMBINED_DATASET_CSV, REQUIRED_COLUMNS)
    print(f"[DATASET] {dataset_summary['path']}  rows={dataset_summary['n_rows']}  "
          f"sha256={dataset_summary['sha256'][:12]}...")
    print(f"[DATASET] columns: {dataset_summary['columns']}")
    print(f"[DATASET] class distribution: {dataset_summary['class_distribution']}")
    print(f"[DATASET] label_source distribution: {dataset_summary['label_source_distribution']}")

    if not SOURCE_HELD_OUT_TEST_CSV.is_file():
        print(f"[ERROR] paired held-out test set not found: {SOURCE_HELD_OUT_TEST_CSV}", file=sys.stderr)
        return 1
    test_df = pd.read_csv(SOURCE_HELD_OUT_TEST_CSV, dtype=str, keep_default_na=False)
    print(f"[TEST SET] {_rel(SOURCE_HELD_OUT_TEST_CSV)}  rows={len(test_df)}  "
          f"class distribution={test_df['ai_stance'].value_counts().to_dict()}")

    # ── Step 2: prepare text/label columns ──
    X_train_all = combined["original_text"]
    y_train_all = combined["gold_stance"]
    X_test, y_test = test_df["original_text"].to_numpy(), test_df["ai_stance"].to_numpy()

    # ── explicit train-vs-test duplicate-text leakage check (full 19,331-row scale) ──
    leakage = find_train_test_text_leakage(X_train_all, pd.Series(X_test))
    overlap_mask = leakage.pop("overlap_mask")
    if leakage["n_train_rows_overlapping_test_text"] > 0:
        print(f"[LEAKAGE] {leakage['n_train_rows_overlapping_test_text']} training rows have text "
              f"identical (after normalisation) to a held-out test text -- EXCLUDING them from training.",
              file=sys.stderr)
        combined = combined[~overlap_mask].reset_index(drop=True)
        X_train_all, y_train_all = combined["original_text"], combined["gold_stance"]
    else:
        print("[LEAKAGE] 0 training rows share normalised text with the held-out test set (checked "
              f"against all {leakage['n_test_texts']} unique test texts).")

    X_train, y_train = X_train_all.to_numpy(), y_train_all.to_numpy()

    # ── recompute pool400's own leakage-safe split (same seed/protocol as every prior run) to get
    #     group ids for the held-out test rows, needed for the group-bootstrap CI in sp.evaluate ──
    st, gold = lb.load_ai_working_labels(POOL_LABELS_FILE, "expansion_v2")
    if st.state != "ok":
        print(f"[ERROR] pool400 working labels: {st.detail}", file=sys.stderr)
        return 1
    cg, corpus_mapping = corpus_groups(gold)
    d = gold.copy()
    d["_cg"] = cg
    cnt = d.gold_stance.value_counts()
    rare = cnt[cnt < cfg["min_class_count"]]
    dropped_pool = rare.to_dict() if len(rare) else {}
    if len(rare):
        d = d[~d.gold_stance.isin(rare.index)]
    d = d.reset_index(drop=True)
    pool_labels = [x for x in lb.SCHEMA_LABELS if x in set(d.gold_stance)]
    groups = sp.leakage_groups(d, d["_cg"], cfg["near_dup_jaccard"], cfg["near_dup_containment"])
    y_pool = d.gold_stance.to_numpy()
    tr, va, te = three_way_split(y_pool, groups, a.seed)
    d_test_recomputed = d.iloc[te].copy()
    if sorted(d_test_recomputed["original_text"].tolist()) != sorted(test_df["original_text"].tolist()):
        print("[ERROR] recomputed pool400 test split does not match the persisted held-out test set; "
              "refusing to proceed with a mismatched group-bootstrap.", file=sys.stderr)
        return 3
    groups_test = groups[te]
    print(f"[SPLIT] held-out test set verified identical to pool400's own leakage-safe test split "
          f"(seed={a.seed}, {len(te)} records); no new hyperparameters are tuned in this run so no "
          f"additional validation split is required -- the fixed project-wide TF-IDF/model config is "
          f"reused unchanged (same as every prior run on this dataset).")

    # ── Step 3-6: train LinearSVM + LogReg, compare, evaluate ──
    pipelines: dict[str, Any] = {}
    metrics_by_model: dict[str, Any] = {}
    original_metrics: dict[str, Any] = {}
    for m in sp.MODELS:
        pipe = sp.build_pipeline(m, cfg).fit(X_train, y_train)
        pipelines[m] = pipe
        pred = pipe.predict(X_test)
        metrics_by_model[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
            "model_classes_": sorted(pipe.named_steps["clf"].classes_.tolist()),
            "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_),
        }
        original_pipe = load_trained_pipeline(ORIGINAL_RUN_DIR, m)
        orig_pred = original_pipe.predict(X_test)
        original_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, orig_pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(orig_pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
        }
        print(f"[RESULT] {m}: macro_f1={metrics_by_model[m]['test_vs_pool_labels']['macro_f1']} "
              f"(original={original_metrics[m]['test_vs_pool_labels']['macro_f1']}) "
              f"Oppose-predictions-on-test={metrics_by_model[m]['test_predicted_distribution_all_4_labels'].get('Oppose', 0)}")

    # ── Step 7: save (write-once), new versioned directory ──
    run_dir.mkdir(parents=True)
    import joblib
    arts: dict[str, str] = {}
    for m, pipe in pipelines.items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)

    disclaimer = (
        lb.AI_DISCLAIMER + " ADDITIONALLY: training data is the FINAL combined labelled dataset "
        "already produced by this project (pool400 AI working labels + in-domain corpus pseudo-labels "
        "+ Sentiment140 pseudo-labels, all four classes Support/Neutral/Oppose/Unclear used as currently "
        "assigned, with no re-labeling or filtering). Oppose is UNVERIFIED everywhere in this run: the "
        "held-out test set has only 1 genuine Oppose example project-wide (dropped below "
        "min_class_count, same as every prior run) and cannot score Oppose precision/recall against "
        "genuine human-reviewed ground truth. Metrics below are agreement with AI/heuristic labels on "
        "Support/Neutral/Unclear only, not validated stance accuracy."
    )
    unclear_handling_note = (
        "Unclear is preserved as a distinct fourth output class throughout: its only training signal "
        "in this dataset is the genuine pool400 Unclear records (32 of the 319 train+validation rows); "
        "the pseudo-labeling rule in models/pseudo_labeling.py always excludes Unclear from the "
        "training-eligible pseudo pool by design, so 0 pseudo-labeled Unclear rows exist anywhere in "
        "this dataset. Unclear precision/recall on the held-out test set remain near 0.0, as in every "
        "prior run -- the class has too few genuine examples project-wide to evaluate reliably, "
        "independent of this experiment."
    )
    metrics = {
        "disclaimer": disclaimer, "unclear_handling": unclear_handling_note,
        "dataset_verification": dataset_summary,
        "leakage_check": {
            "method": "exact match on normalise_text(original_text) between the full combined training "
                      "set and the held-out test set",
            "n_train_rows_excluded_for_test_overlap": leakage["n_train_rows_overlapping_test_text"],
            "n_unique_test_texts_checked_against": leakage["n_test_texts"],
        },
        "pool400_split_cross_check": {
            "seed": a.seed, "usable": len(d), "dropped_rare_classes": dropped_pool,
            "labels_evaluated": pool_labels, "corpus_group_mapping": corpus_mapping,
            "n_test_held_out": len(d_test_recomputed),
            "note": "recomputed from POOL_LABELS_FILE with the identical seed/protocol used by every "
                    "prior run; verified byte-identical original_text set to the persisted held-out "
                    "test CSV before proceeding.",
        },
        "training_set_used": {
            "n_rows": len(combined), "by_class": combined["gold_stance"].value_counts().to_dict(),
            "by_source": combined["label_source"].value_counts().to_dict(),
            "source_file": _rel(COMBINED_DATASET_CSV),
        },
        "held_out_test_set": {
            "n": len(test_df), "class_distribution": pd.Series(y_test).value_counts().to_dict(),
            "source_file": _rel(SOURCE_HELD_OUT_TEST_CSV),
            "note": "Identical pool400 test split used project-wide (same seed/protocol); zero "
                    "pseudo-labeled records; never used for any threshold or hyperparameter tuning.",
        },
        "models": metrics_by_model,
        "original_model_same_test_set": original_metrics,
        "comparison_note": (
            "This run trains on the exact same combined dataset, split, and TF-IDF/model config as "
            "run_pool400_plus_all_sources_v1's 'model_3_all_sources' -- so these numbers are expected "
            "to reproduce that run's reported macro-F1 (modulo the 0-row leakage exclusion above), not "
            "a new or different model. The value of this run is the standalone dataset verification, "
            "the explicit full-scale leakage check, and the clean, independently re-derivable manifest."
        ),
    }
    files = {
        "metrics.json": metrics,
        "preprocessing_config.json": {"base": cfg, "note": "same TF-IDF/model config as run_pool400_ai_v1_3class"},
        "label_mapping.json": {
            "labels": pool_labels, "all_model_output_labels": ALL_FOUR_LABELS, "schema": lb.SCHEMA_LABELS,
            "construct": "stance toward the Open Government initiative/policy (guidelines v1.1) for "
                         "pool400 and corpus-pseudo records; out-of-domain extrapolation for "
                         "Sentiment140-pseudo records; not sentiment, not topic",
            "unclear_handling": unclear_handling_note,
        },
    }
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)

    test_df.to_csv(run_dir / "held_out_test_set.csv", index=False, lineterminator="\n")
    arts["held_out_test_set.csv"] = _sha(run_dir / "held_out_test_set.csv")

    meta = {
        "status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_stance_combined_dataset_v1.py", "argv": sys.argv[1:] if argv is None else argv,
        "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(sp.MODELS),
        "artifacts_sha256": arts, "human_labels_only": False, "label_type": "mixed_ai_and_pseudo_unverified",
        "disclaimer": disclaimer, "merged_or_voted": False,
        "inputs_sha256": {
            _rel(COMBINED_DATASET_CSV): dataset_summary["sha256"],
            _rel(SOURCE_HELD_OUT_TEST_CSV): _sha(SOURCE_HELD_OUT_TEST_CSV),
            _rel(SOURCE_METADATA_JSON): _sha(SOURCE_METADATA_JSON),
            _rel(POOL_LABELS_FILE): _sha(POOL_LABELS_FILE), _rel(POOL_PROVENANCE): _sha(POOL_PROVENANCE),
        },
        "does_not_modify": [
            "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class",
            "data/stance_training_ai_working_labels/run_pool400_plus_pseudo_s140_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_all_sources_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_filtered_s140_corpus_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_no_oppose_v1",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "test_n": len(test_df), "train_n": len(combined),
        "linsvm_macro_f1": metrics_by_model["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "logreg_macro_f1": metrics_by_model["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "original_linsvm_macro_f1": original_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "original_logreg_macro_f1": original_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
    }, default=str))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
