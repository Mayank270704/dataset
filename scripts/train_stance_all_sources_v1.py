#!/usr/bin/env python
"""
scripts/train_stance_all_sources_v1.py
=======================================
Trains a stance classifier on ALL suitable local pseudo-label sources combined:
  - pool400 (319 train+val records; AI-generated, unverified) -- same base as before
  - in-domain corpus remainder pseudo-labels (data/pseudo_labeling/corpus_stance_v1), ALL
    9,515 training-eligible records -- NOT capped, because this is in-domain, relevant
    consultation text, not a large noisy out-of-domain source (see docstring below).
  - Sentiment140 TEST pseudo-labels (data/pseudo_labeling/sentiment140_test_stance_v1), ALL
    497 training-eligible records -- tiny, no cap needed.
  - Sentiment140 TRAIN pseudo-labels (data/pseudo_labeling/sentiment140_stance_v1), the SAME
    seeded 3,000/class reservoir sample used in scripts/train_stance_expanded_v1.py -- capped,
    because this IS the large, noisy, out-of-domain source the project owner was warned about;
    reusing the same cap/seed keeps this comparable to the Sentiment140-only run.

Why the in-domain corpus is not capped but Sentiment140 is
-------------------------------------------------------------
The brief this was built against explicitly warns: "Do not let a very large noisy source
overwhelm the smaller, more relevant consultation data." The corpus remainder (9,515 rows) is
the smaller, more relevant, in-domain source here -- capping it would do the opposite of what
that instruction asks. Sentiment140 is the large, noisy, out-of-domain source; it keeps the same
cap as scripts/train_stance_expanded_v1.py for a clean comparison.

Evaluated against the SAME held-out pool400 test split (80 records) used by:
  - data/stance_training_ai_working_labels/run_pool400_ai_v1_3class       (original model)
  - data/stance_training_ai_working_labels/run_pool400_plus_pseudo_s140_v1 (Sentiment140-only;
    metrics reused from its saved metrics.json rather than recomputed, since the split is
    deterministic given the same seed and will reproduce identically)

Usage
    python scripts/train_stance_all_sources_v1.py --tag pool400_plus_all_sources_v1
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402
from models.stance_classification.pipeline import load_trained_pipeline  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402
from scripts.train_stance_expanded_v1 import (  # noqa: E402
    ALL_FOUR_LABELS, AI_LABEL_SOURCE, ORIGINAL_RUN_DIR, POOL_LABELS_FILE, POOL_PROVENANCE,
    reservoir_sample_pseudo_labels, three_way_split,
)

SENTIMENT140_TRAIN_MANIFEST = ROOT / "data" / "pseudo_labeling" / "sentiment140_stance_v1" / "manifest.json"
SENTIMENT140_TRAIN_CSV = ROOT / "data" / "pseudo_labeling" / "sentiment140_stance_v1" / "pseudo_labels.csv"
SENTIMENT140_TEST_CSV = ROOT / "data" / "pseudo_labeling" / "sentiment140_test_stance_v1" / "pseudo_labels.csv"
CORPUS_PSEUDO_CSV = ROOT / "data" / "pseudo_labeling" / "corpus_stance_v1" / "pseudo_labels.csv"
SENTIMENT140_ONLY_RUN_DIR = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_plus_pseudo_s140_v1"
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"

CORPUS_LABEL_SOURCE = "pseudo_label_corpus_indomain_v1"
S140_LABEL_SOURCE = "pseudo_label_sentiment140_heuristic_v1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def load_simple_pseudo_csv(path: Path, text_col: str, label_source: str) -> pd.DataFrame:
    """Load a small (already written) pseudo-label CSV in full and keep only training-eligible rows."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["eligible_for_training"] == "True"].copy()
    out = pd.DataFrame({
        "original_text": df[text_col].values,
        "gold_stance": df["pseudo_label"].values,
        "label_source": label_source,
        "source_dataset": df["source_dataset"].values,
        "source_record_id": df["source_record_id"].values,
        "rule": df["rule"].values,
        "sentiment_prediction": df["sentiment_prediction"].values,
        "stance_margin_pseudo": df["stance_margin"].astype(float).values,
        "sentiment_margin_pseudo": df["sentiment_margin"].astype(float).values,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-class-cap-sentiment140", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2
    for p in (SENTIMENT140_TRAIN_CSV, SENTIMENT140_TEST_CSV, CORPUS_PSEUDO_CSV):
        if not p.is_file():
            print(f"[ERROR] required pseudo-label file not found: {p}", file=sys.stderr)
            return 1

    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed

    # ── 1. pool400: identical loading + split protocol ──
    st, gold = lb.load_ai_working_labels(POOL_LABELS_FILE, "expansion_v2")
    print(f"[AI WORKING LABELS] {lb.AI_DISCLAIMER}\n[{st.state}] {st.detail}")
    if st.state != "ok":
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
    norm = d.original_text.map(normalise_text)
    for t, g in pd.Series(groups).groupby(norm.values):
        assert g.nunique() == 1, f"normalised-identical text split across groups: {t[:60]!r}"

    y_pool = d.gold_stance.to_numpy()
    tr, va, te = three_way_split(y_pool, groups, a.seed)
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        if set(y_pool[idx]) != set(pool_labels):
            print(f"[BLOCKED] pool400 {name} split lacks classes {sorted(set(pool_labels) - set(y_pool[idx]))}", file=sys.stderr)
            return 3

    d_test = d.iloc[te].copy()
    d_trainval = d.iloc[np.concatenate([tr, va])].copy()
    print(f"pool400 usable records: {len(d)} (dropped {dropped_pool}); "
          f"train+val {len(d_trainval)} / test {len(d_test)} (held out, identical split seed to prior runs)")

    # ── 2. all three pseudo-label sources ──
    s140_sample, s140_report = reservoir_sample_pseudo_labels(
        SENTIMENT140_TRAIN_CSV, a.per_class_cap_sentiment140, a.seed
    )
    print(f"Sentiment140 TRAIN reservoir sample (cap {a.per_class_cap_sentiment140}/class): {json.dumps(s140_report)}")
    s140_train_block = pd.DataFrame({
        "original_text": s140_sample["cleaned_text"].values, "gold_stance": s140_sample["pseudo_label"].values,
        "label_source": S140_LABEL_SOURCE, "source_dataset": s140_sample["source_dataset"].values,
        "source_record_id": s140_sample["source_record_id"].values, "rule": s140_sample["rule"].values,
        "sentiment_prediction": s140_sample["sentiment_prediction"].values,
        "stance_margin_pseudo": s140_sample["stance_margin"].astype(float).values,
        "sentiment_margin_pseudo": s140_sample["sentiment_margin"].astype(float).values,
    })

    s140_test_block = load_simple_pseudo_csv(SENTIMENT140_TEST_CSV, "cleaned_text", S140_LABEL_SOURCE)
    corpus_block = load_simple_pseudo_csv(CORPUS_PSEUDO_CSV, "original_text", CORPUS_LABEL_SOURCE)
    print(f"Sentiment140 TEST pseudo rows (all, training-eligible): {len(s140_test_block)}; "
          f"by class {s140_test_block.gold_stance.value_counts().to_dict()}")
    print(f"In-domain corpus pseudo rows (all, training-eligible): {len(corpus_block)}; "
          f"by class {corpus_block.gold_stance.value_counts().to_dict()}")

    pool_train = pd.DataFrame({
        "original_text": d_trainval["original_text"].values, "gold_stance": d_trainval["gold_stance"].values,
        "label_source": AI_LABEL_SOURCE, "source_dataset": d_trainval["source_dataset"].values,
        "source_record_id": d_trainval["source_record_id"].values,
        "rule": "", "sentiment_prediction": "", "stance_margin_pseudo": np.nan, "sentiment_margin_pseudo": np.nan,
    })

    combined_train = pd.concat([pool_train, corpus_block, s140_test_block, s140_train_block], ignore_index=True)
    train_class_dist = combined_train.gold_stance.value_counts().to_dict()
    train_source_dist = combined_train.label_source.value_counts().to_dict()
    print(f"combined training set: {len(combined_train)} records; by class {train_class_dist}; "
          f"by source {train_source_dist}")

    X_train, y_train = combined_train.original_text.to_numpy(), combined_train.gold_stance.to_numpy()
    X_test, y_test = d_test.original_text.to_numpy(), d_test.gold_stance.to_numpy()
    groups_test = groups[te]

    # ── 3. fit new models ──
    new_pipelines: dict[str, Any] = {}
    new_metrics: dict[str, Any] = {}
    original_metrics: dict[str, Any] = {}
    for m in sp.MODELS:
        pipe = sp.build_pipeline(m, cfg).fit(X_train, y_train)
        new_pipelines[m] = pipe
        pred = pipe.predict(X_test)
        new_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
            "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_),
        }
        original_pipe = load_trained_pipeline(ORIGINAL_RUN_DIR, m)
        orig_pred = original_pipe.predict(X_test)
        original_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, orig_pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(orig_pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
        }

    s140_only_metrics = json.loads((SENTIMENT140_ONLY_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))

    # ── 4. save (write-once) ──
    run_dir.mkdir(parents=True)
    import joblib
    arts: dict[str, str] = {}
    for m, pipe in new_pipelines.items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)

    disclaimer = (
        lb.AI_DISCLAIMER + " ADDITIONALLY: training data mixes pool400 AI-working-labels with pseudo-labels from "
        "THREE sources -- the in-domain consultation corpus remainder (self-training risk: same stance model, "
        "same 3 source datasets), Sentiment140 test tweets, and a capped Sentiment140 train sample (out-of-domain, "
        "heuristic Oppose rule with a known ~98% in-domain false-positive rate). Oppose is UNVERIFIED everywhere in "
        "this run; the held-out test set has only 1 genuine Oppose example project-wide and cannot evaluate it."
    )
    metrics = {
        "label_type": "mixed: ai_generated_unverified + pseudo_label_corpus_indomain_v1 + pseudo_label_sentiment140_heuristic_v1",
        "disclaimer": disclaimer,
        "pool400": {"usable": len(d), "dropped": dropped_pool, "labels_evaluated": pool_labels,
                    "n_train_val": len(d_trainval), "n_test_held_out": len(d_test), "corpus_group_mapping": corpus_mapping},
        "sources": {
            "pool400_train_val": {"n": len(pool_train), "by_class": pool_train.gold_stance.value_counts().to_dict()},
            "corpus_indomain_all": {"n": len(corpus_block), "by_class": corpus_block.gold_stance.value_counts().to_dict(),
                                    "source_manifest": _rel(CORPUS_PSEUDO_CSV.parent / "manifest.json")},
            "sentiment140_test_all": {"n": len(s140_test_block), "by_class": s140_test_block.gold_stance.value_counts().to_dict(),
                                      "source_manifest": _rel(SENTIMENT140_TEST_CSV.parent / "manifest.json")},
            "sentiment140_train_capped_sample": {"n": len(s140_train_block), "per_class_cap": a.per_class_cap_sentiment140,
                                                 "seed": a.seed, "reservoir_report": s140_report,
                                                 "source_manifest": _rel(SENTIMENT140_TRAIN_MANIFEST)},
        },
        "combined_training_set": {"n": len(combined_train), "by_class": train_class_dist, "by_source": train_source_dist},
        "held_out_test_set": {"n": len(d_test), "class_distribution": pd.Series(y_test).value_counts().to_dict(),
                              "note": "Identical pool400 test split used by run_pool400_ai_v1_3class and "
                                      "run_pool400_plus_pseudo_s140_v1 (same seed/protocol); zero pseudo-labeled records."},
        "model_1_original": original_metrics,
        "model_2_sentiment140_only": {
            m: s140_only_metrics["new_model"][m]["test_vs_pool_labels"] for m in sp.MODELS
        } | {"source_run": "run_pool400_plus_pseudo_s140_v1", "note": "reused from its saved metrics.json (identical test split/seed, not recomputed)"},
        "model_3_all_sources": new_metrics,
        "comparison_note": ("All three models evaluated on the identical 80-record held-out pool400 test split. "
                            "macro_f1 computed over classes present in the test set (Support/Neutral/Unclear) for "
                            "comparability. test_predicted_distribution_all_4_labels shows Oppose over-firing "
                            "descriptively (not scored: no genuine Oppose ground truth in the test set)."),
    }
    files = {
        "metrics.json": metrics,
        "preprocessing_config.json": {"base": cfg, "note": "same TF-IDF/model config as run_pool400_ai_v1_3class"},
        "label_mapping.json": {"labels": pool_labels, "all_model_output_labels": ALL_FOUR_LABELS, "schema": lb.SCHEMA_LABELS,
                               "construct": "stance toward the Open Government initiative/policy (guidelines v1.1) for "
                                            "pool400 and corpus-pseudo records; out-of-domain extrapolation for "
                                            "Sentiment140-pseudo records; not sentiment, not topic"},
    }
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)

    combined_train.assign(split="train_or_validation").to_csv(run_dir / "combined_training_data_provenance.csv", index=False, lineterminator="\n")
    d_test[["pilot_id", "source_dataset", "source_record_id", "gold_stance", "original_text"]].assign(split="test").rename(
        columns={"gold_stance": "ai_stance"}
    ).to_csv(run_dir / "held_out_test_set.csv", index=False, lineterminator="\n")
    for f in ("combined_training_data_provenance.csv", "held_out_test_set.csv"):
        arts[f] = _sha(run_dir / f)

    meta = {
        "status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_stance_all_sources_v1.py", "argv": sys.argv[1:] if argv is None else argv,
        "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(sp.MODELS),
        "artifacts_sha256": arts, "human_labels_only": False, "label_type": "mixed_ai_and_pseudo_unverified",
        "disclaimer": disclaimer, "merged_or_voted": False,
        "inputs_sha256": {
            _rel(POOL_LABELS_FILE): _sha(POOL_LABELS_FILE), _rel(POOL_PROVENANCE): _sha(POOL_PROVENANCE),
            _rel(SENTIMENT140_TRAIN_MANIFEST): _sha(SENTIMENT140_TRAIN_MANIFEST),
            _rel(SENTIMENT140_TEST_CSV.parent / "manifest.json"): _sha(SENTIMENT140_TEST_CSV.parent / "manifest.json"),
            _rel(CORPUS_PSEUDO_CSV.parent / "manifest.json"): _sha(CORPUS_PSEUDO_CSV.parent / "manifest.json"),
        },
        "does_not_modify": [
            "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class",
            "data/stance_training_ai_working_labels/run_pool400_plus_pseudo_s140_v1",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "pool400_test_n": len(d_test), "combined_train_n": len(combined_train),
        "model1_original_linsvm_macro_f1": original_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "model2_s140only_linsvm_macro_f1": s140_only_metrics["new_model"]["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "model3_allsources_linsvm_macro_f1": new_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "model1_original_logreg_macro_f1": original_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "model2_s140only_logreg_macro_f1": s140_only_metrics["new_model"]["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "model3_allsources_logreg_macro_f1": new_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
    }, default=str))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
