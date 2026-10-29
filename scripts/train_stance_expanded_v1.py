#!/usr/bin/env python
"""
scripts/train_stance_expanded_v1.py
====================================
Trains a new stance classifier on the original 400-record AI-working-label pool PLUS a
stratified, capped random sample of the Sentiment140 pseudo-labels from
data/pseudo_labeling/sentiment140_stance_v1, and compares it against the original
run_pool400_ai_v1_3class model on a held-out set that contains NO pseudo-labeled records.

Design (why this is safe to run on a memory-constrained laptop without the existing near-dup
leakage-group check becoming O(n^2) over a million rows)
---------------------------------------------------------------------------------------------
The held-out test split is computed ONCE, on the 400-record pool ALONE (same protocol as
scripts/train_stance_ai_labels_v2.py: StratifiedGroupKFold leakage-safe split). The
Sentiment140 pseudo-label sample is added ONLY to the training side, after the split is fixed;
it is never split, so the existing O(n^2) near-duplicate grouping (models/stance_classification/
pipeline.py: leakage_groups) only ever runs on the 400-record pool, exactly as before. There is
no plausible lexical overlap between Canadian consultation text and Sentiment140 tweets, so
skipping near-dup checks between the two sources does not create a leakage risk.

The pseudo-label sample is NOT all ~1.57M accepted rows: it is a seeded, stratified reservoir
sample capped at --per-class-cap (default 3,000) per class (Support/Neutral/Oppose), drawn in
one streaming pass over the large CSV (constant memory). This is a deliberate choice, not a
compute shortcut: dumping in a million heuristic-derived rows would swamp the ~400 higher-quality
AI-working labels and does not, on this project's own evidence, make the Oppose signal more
trustworthy -- see the disclaimer in models/pseudo_labeling.py.

What this does NOT do
----------------------
* Does not evaluate Oppose against genuine ground truth: the 400-record pool has exactly 1
  Oppose example, too few for any held-out Oppose evaluation. Oppose quality is UNVERIFIED;
  this script states that plainly rather than reporting a misleading Oppose metric.
* Does not overwrite data/stance_training_ai_working_labels/run_pool400_ai_v1_3class. The new
  run is written to a new, write-once directory.

Usage
    python scripts/train_stance_expanded_v1.py --tag pool400_plus_pseudo_s140_v1
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
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402
from models.stance_classification.pipeline import load_trained_pipeline  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402

POOL_LABELS_FILE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_working_labels_pool400.csv"
POOL_PROVENANCE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "provenance.json"
ORIGINAL_RUN_DIR = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
PSEUDO_LABELS_CSV = ROOT / "data" / "pseudo_labeling" / "sentiment140_stance_v1" / "pseudo_labels.csv"
PSEUDO_MANIFEST = ROOT / "data" / "pseudo_labeling" / "sentiment140_stance_v1" / "manifest.json"
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"

PSEUDO_LABEL_SOURCE = "pseudo_label_sentiment140_heuristic_v1"
AI_LABEL_SOURCE = lb.AI_LABEL_SOURCE
ALL_FOUR_LABELS = ["Support", "Neutral", "Oppose", "Unclear"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def three_way_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identical protocol to scripts/train_stance_ai_labels_v2.py: test = fold 1 of GroupKFold(5);
    validation = fold 1 of GroupKFold(4) on the remainder; train = the rest. No group crosses a boundary."""
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, te = next(sg.split(np.zeros(len(y)), y, groups))
    sg2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_r, va_r = next(sg2.split(np.zeros(len(rest)), y[rest], groups[rest]))
    tr, va = rest[tr_r], rest[va_r]
    assert not (set(groups[tr]) & set(groups[va]) or set(groups[tr]) & set(groups[te]) or set(groups[va]) & set(groups[te]))
    return tr, va, te


def reservoir_sample_pseudo_labels(csv_path: Path, per_class_cap: int, seed: int,
                                    chunk_rows: int = 50_000) -> pd.DataFrame:
    """
    One streaming pass (constant memory) over the large pseudo-label CSV, keeping a uniform
    random sample of up to `per_class_cap` training-eligible rows per pseudo_label class, via
    classic Algorithm R reservoir sampling applied independently per class.
    """
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = {lab: [] for lab in ("Support", "Neutral", "Oppose")}
    seen_counts: dict[str, int] = {lab: 0 for lab in reservoirs}

    cols = ["record_id", "source_dataset", "source_record_id", "cleaned_text", "pseudo_label", "rule",
            "sentiment_prediction", "sentiment_margin", "stance_raw", "stance_margin", "duplicate_group_id"]
    for chunk in pd.read_csv(csv_path, dtype=str, keep_default_na=False, usecols=cols + ["eligible_for_training"],
                             chunksize=chunk_rows):
        eligible = chunk[chunk["eligible_for_training"] == "True"]
        for label, group in eligible.groupby("pseudo_label"):
            if label not in reservoirs:
                continue
            reservoir, seen = reservoirs[label], seen_counts[label]
            for _, row in group[cols].iterrows():
                seen += 1
                if len(reservoir) < per_class_cap:
                    reservoir.append(row.to_dict())
                else:
                    j = rng.integers(0, seen)
                    if j < per_class_cap:
                        reservoir[j] = row.to_dict()
            seen_counts[label] = seen

    frames = [pd.DataFrame(rows) for rows in reservoirs.values() if rows]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    report = {lab: {"seen": seen_counts[lab], "sampled": len(reservoirs[lab])} for lab in reservoirs}
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-class-cap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2
    if not PSEUDO_LABELS_CSV.is_file():
        print(f"[ERROR] {PSEUDO_LABELS_CSV} not found; run the pseudo-labeling pipeline first.", file=sys.stderr)
        return 1

    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed

    # ── 1. pool400: identical loading + split protocol as train_stance_ai_labels_v2.py ──
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
          f"train+val {len(d_trainval)} / test {len(d_test)} (held out, NEVER touched by pseudo-labels)")

    # ── 2. Sentiment140 pseudo-label reservoir sample (streaming, constant memory) ──
    pseudo, pseudo_report = reservoir_sample_pseudo_labels(PSEUDO_LABELS_CSV, a.per_class_cap, a.seed)
    print(f"pseudo-label reservoir sample (cap {a.per_class_cap}/class, seed {a.seed}): "
          f"{json.dumps(pseudo_report)}")

    # ── 3. combine: pool400 train+val (Support/Neutral/Unclear) + pseudo sample (Support/Neutral/Oppose) ──
    pool_train = pd.DataFrame({
        "original_text": d_trainval["original_text"].values,
        "gold_stance": d_trainval["gold_stance"].values,
        "label_source": AI_LABEL_SOURCE,
        "source_dataset": d_trainval["source_dataset"].values,
        "source_record_id": d_trainval["source_record_id"].values,
        "rule": "", "sentiment_prediction": "", "stance_margin_pseudo": np.nan, "sentiment_margin_pseudo": np.nan,
    })
    pseudo_train = pd.DataFrame({
        "original_text": pseudo["cleaned_text"].values,
        "gold_stance": pseudo["pseudo_label"].values,
        "label_source": PSEUDO_LABEL_SOURCE,
        "source_dataset": pseudo["source_dataset"].values,
        "source_record_id": pseudo["source_record_id"].values,
        "rule": pseudo["rule"].values,
        "sentiment_prediction": pseudo["sentiment_prediction"].values,
        "stance_margin_pseudo": pseudo["stance_margin"].astype(float).values,
        "sentiment_margin_pseudo": pseudo["sentiment_margin"].astype(float).values,
    })
    combined_train = pd.concat([pool_train, pseudo_train], ignore_index=True)
    train_class_dist = combined_train.gold_stance.value_counts().to_dict()
    train_source_dist = combined_train.label_source.value_counts().to_dict()
    print(f"combined training set: {len(combined_train)} records; by class {train_class_dist}; "
          f"by source {train_source_dist}")

    X_train, y_train = combined_train.original_text.to_numpy(), combined_train.gold_stance.to_numpy()
    X_test, y_test = d_test.original_text.to_numpy(), d_test.gold_stance.to_numpy()
    groups_test = groups[te]

    # ── 4. fit new models on the combined training set ──
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
            "note": "original model's own classes are Support/Neutral/Unclear only; it cannot output Oppose",
        }

    # ── 5. save (write-once) ──
    run_dir.mkdir(parents=True)
    import joblib
    arts: dict[str, str] = {}
    for m, pipe in new_pipelines.items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)

    metrics = {
        "label_type": "mixed: ai_generated_unverified (pool400) + pseudo_label_sentiment140_heuristic_v1_unverified",
        "disclaimer": (lb.AI_DISCLAIMER + " ADDITIONALLY: part of the training data is pseudo-labeled Sentiment140 "
                       "tweets (out-of-domain, heuristic Oppose rule with a known high in-domain false-positive rate; "
                       "see models/pseudo_labeling.py). Oppose is UNVERIFIED and NOT evaluated against genuine "
                       "ground truth below: the held-out test set has only 1 genuine Oppose example project-wide, "
                       "too few to evaluate, so it was dropped from the pool like in the original run. Metrics below "
                       "are agreement with AI/heuristic labels, not human-verified stance accuracy."),
        "pool400": {"usable": len(d), "dropped": dropped_pool, "labels_evaluated": pool_labels,
                    "n_train_val": len(d_trainval), "n_test_held_out": len(d_test), "corpus_group_mapping": corpus_mapping},
        "pseudo_label_sample": {"source_manifest": _rel(PSEUDO_MANIFEST), "per_class_cap": a.per_class_cap,
                                "seed": a.seed, "reservoir_report": pseudo_report},
        "combined_training_set": {"n": len(combined_train), "by_class": train_class_dist, "by_source": train_source_dist},
        "held_out_test_set": {"n": len(d_test), "class_distribution": pd.Series(y_test).value_counts().to_dict(),
                              "contains_zero_pseudo_labeled_records": True,
                              "note": "Entirely Canadian consultation text from pool400; never touched by Sentiment140 pseudo-labeling or training."},
        "new_model": new_metrics,
        "original_model_same_test_set": original_metrics,
        "comparison_note": ("Both models evaluated on the identical held-out pool400 test split. macro_f1 is "
                            "computed over the classes present in the test set (Support/Neutral/Unclear) for "
                            "comparability with the originally-documented baseline (0.3641 LR / 0.3958 SVM). "
                            "test_predicted_distribution_all_4_labels additionally shows, purely descriptively, "
                            "whether each model ever predicts Oppose on genuine held-out consultation text; this is "
                            "NOT a scored metric because the test set has no genuine Oppose ground truth."),
    }
    files = {
        "metrics.json": metrics,
        "preprocessing_config.json": {"base": cfg, "note": "same TF-IDF/model config as run_pool400_ai_v1_3class"},
        "label_mapping.json": {"labels": pool_labels, "all_model_output_labels": ALL_FOUR_LABELS, "schema": lb.SCHEMA_LABELS,
                               "construct": "stance toward the Open Government initiative/policy (guidelines v1.1) for "
                                            "pool400 records; general out-of-domain stance extrapolation for pseudo-labeled "
                                            "Sentiment140 records; not sentiment, not topic"},
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
        "script": "scripts/train_stance_expanded_v1.py", "argv": sys.argv[1:] if argv is None else argv,
        "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(sp.MODELS),
        "artifacts_sha256": arts, "human_labels_only": False, "label_type": "mixed_ai_and_pseudo_unverified",
        "disclaimer": metrics["disclaimer"], "merged_or_voted": False,
        "inputs_sha256": {
            _rel(POOL_LABELS_FILE): _sha(POOL_LABELS_FILE), _rel(POOL_PROVENANCE): _sha(POOL_PROVENANCE),
            _rel(PSEUDO_MANIFEST): _sha(PSEUDO_MANIFEST),
        },
        "does_not_modify": "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class (original artifacts untouched)",
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "pool400_test_n": len(d_test), "combined_train_n": len(combined_train),
        "new_linsvm_macro_f1": new_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "original_linsvm_macro_f1": original_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "new_logreg_macro_f1": new_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "original_logreg_macro_f1": original_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
    }, default=str))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
