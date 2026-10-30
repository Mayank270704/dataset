"""Compare stance models against repository reference labels on a fresh grouped holdout.

The previously inspected 80-row split is development data. This script creates a new
source-record/near-duplicate grouped train/validation/test split; selection uses validation
macro-F1 only and evaluates the selected candidate, word+character candidate, and baselines
once on the new test split. Existing labels and source mappings are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from models.stance_classification import pipeline as stance_pipeline
from scripts.train_stance_baselines import corpus_groups

REFERENCE_LABELS_PATH = ROOT / "data/ai_labels/stance_sentiment_ai_v1/ai_working_labels_pool400.csv"
DEFAULT_OUT = ROOT / "reports/stance_reference_model_comparison_v2"
SEED = 20261029
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MODEL_LABELS = ("Support", "Neutral", "Unclear")


def grouped_three_way_split(reference_labels: np.ndarray, groups: np.ndarray, seed: int):
    first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation, test = next(first.split(np.zeros(len(reference_labels)), reference_labels, groups))
    second = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, validation_rel = next(second.split(np.zeros(len(train_validation)), reference_labels[train_validation], groups[train_validation]))
    train, validation = train_validation[train_rel], train_validation[validation_rel]
    assert not (set(groups[train]) & set(groups[validation]))
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[validation]) & set(groups[test]))
    return train, validation, test


def evaluate(reference_labels, predicted_labels, labels):
    p, r, f, n = precision_recall_fscore_support(reference_labels, predicted_labels, labels=labels, zero_division=0)
    return {
        "n": int(len(reference_labels)),
        "accuracy": round(float(accuracy_score(reference_labels, predicted_labels)), 4),
        "macro_f1": round(float(f1_score(reference_labels, predicted_labels, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(reference_labels, predicted_labels, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": {label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                               "f1": round(float(f[i]), 4), "support": int(n[i])} for i, label in enumerate(labels)},
        "confusion_matrix": {"labels": labels,
                             "rows_reference_cols_predicted": confusion_matrix(reference_labels, predicted_labels, labels=labels).tolist()},
    }


def build_model(kind: str, c: float, class_weight: str | None):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           max_features=30000, min_df=2, strip_accents="unicode")
    features = FeatureUnion([("word", word), ("char", char)]) if kind.startswith("word_char_") else word
    classifier = (LinearSVC(C=c, class_weight=class_weight, random_state=SEED, max_iter=10000)
                  if kind.endswith("linsvm") else
                  LogisticRegression(C=c, class_weight=class_weight, random_state=SEED, max_iter=3000))
    return Pipeline([("tfidf", features), ("clf", classifier)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="new output directory; existing paths are refused")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    status, frame = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or frame is None:
        raise RuntimeError(status.detail)
    source_frame = frame.copy()
    # Form leakage groups over every source record before restricting the model's three classes.
    all_counts = source_frame.gold_stance.value_counts()
    outside_model = {label: int(count) for label, count in all_counts.items() if label not in MODEL_LABELS}
    corpus_group, mapping = corpus_groups(source_frame)
    all_leakage_groups = stance_pipeline.leakage_groups(source_frame, corpus_group)
    model_eligible = source_frame.gold_stance.isin(MODEL_LABELS).to_numpy()
    frame = source_frame.loc[model_eligible].reset_index(drop=True)
    leakage_groups = all_leakage_groups[model_eligible]
    reference_labels = frame.gold_stance.to_numpy()
    texts = frame.original_text.to_numpy()
    labels = [label for label in MODEL_LABELS if label in set(reference_labels)]
    train, validation, test = grouped_three_way_split(reference_labels, leakage_groups, args.seed)
    for split_name, indices in (("train", train), ("validation", validation), ("test", test)):
        absent = sorted(set(labels) - set(reference_labels[indices]))
        if absent:
            raise RuntimeError(f"{split_name} split lacks reference labels: {absent}; choose another prespecified split seed")

    # Candidate configurations are fitted and ranked exclusively on train/validation.
    validation_runs = []
    models = {}
    for kind in ("word_linsvm", "word_logreg", "word_char_linsvm", "word_char_logreg"):
        for weight in (None, "balanced"):
            for c in C_GRID:
                name = f"{kind}_weight-{weight or 'none'}_C-{c:g}"
                started = time.perf_counter()
                model = build_model(kind, c, weight).fit(texts[train], reference_labels[train])
                fit_seconds = time.perf_counter() - started
                val_pred = model.predict(texts[validation])
                val_metrics = evaluate(reference_labels[validation], val_pred, labels)
                validation_runs.append({"name": name, "kind": kind, "class_weight": weight, "C": c,
                                        "validation": val_metrics, "fit_seconds": round(fit_seconds, 4)})
                models[name] = model

    # Validation macro-F1 is the selection criterion. Stable tie-breaks favor accuracy and simpler word features.
    best = max(validation_runs, key=lambda row: (row["validation"]["macro_f1"], row["validation"]["accuracy"],
                                                row["kind"].startswith("word_"), -row["C"]))
    best_char = max((row for row in validation_runs if row["kind"].startswith("word_char_")),
                    key=lambda row: (row["validation"]["macro_f1"], row["validation"]["accuracy"], -row["C"]))

    # All configurations evaluated below were frozen by validation before accessing the new test fold.
    train_validation = np.concatenate([train, validation])
    test_models = {"selected": best, "word_char_validation_best": best_char}
    baseline_rows = [row for row in validation_runs if row["kind"] in ("word_linsvm", "word_logreg")
                     and row["class_weight"] == "balanced"]
    for kind in ("word_linsvm", "word_logreg"):
        candidates = [row for row in baseline_rows if row["kind"] == kind]
        test_models[f"word_only_{kind}_validation_best"] = max(candidates, key=lambda row: (row["validation"]["macro_f1"], row["validation"]["accuracy"], -row["C"]))
    results = {}
    test_predictions = {}
    for result_name, config in test_models.items():
        model = build_model(config["kind"], config["C"], config["class_weight"])
        model.fit(texts[train_validation], reference_labels[train_validation])
        pred = model.predict(texts[test])
        results[result_name] = {"configuration": config["name"], "metrics": evaluate(reference_labels[test], pred, labels)}
        test_predictions[result_name] = pred

    majority = pd.Series(reference_labels[train_validation]).value_counts().idxmax()
    results["majority_class"] = {"configuration": f"always predict {majority}",
                                  "metrics": evaluate(reference_labels[test], np.full(len(test), majority), labels)}
    results["test_label_counts"] = pd.Series(reference_labels[test]).value_counts().reindex(labels, fill_value=0).astype(int).to_dict()

    language_results = {}
    for model_name, pred in test_predictions.items():
        language_results[model_name] = {}
        lang_values = frame.source_language.fillna("unknown").astype(str).to_numpy()
        for language in sorted(set(lang_values[test])):
            local = np.where(lang_values[test] == language)[0]
            language_results[model_name][language] = evaluate(reference_labels[test][local], pred[local], labels)

    split = np.full(len(frame), "", dtype=object)
    split[train], split[validation], split[test] = "train", "validation", "test"
    assignments = source_frame.copy()
    assignments["split"] = "excluded_outside_three_class_model"
    assignments["model_eligible"] = model_eligible
    assignments["leakage_group"] = all_leakage_groups
    split_by_id = dict(zip(frame.pilot_id, split))
    assignments.loc[assignments.pilot_id.isin(split_by_id), "split"] = assignments.loc[
        assignments.pilot_id.isin(split_by_id), "pilot_id"].map(split_by_id)
    cols = [c for c in ("pilot_id", "source_dataset", "source_record_id", "source_row_number", "source_field",
                        "question_slug", "source_language", "gold_stance", "label_source", "leakage_group",
                        "model_eligible", "split", "original_text") if c in assignments]
    assignments[cols].rename(columns={"gold_stance": "reference_stance", "label_source": "reference_label_source"}).to_csv(
        out_dir / "split_assignments.csv", index=False, lineterminator="\n")
    pred_frame = frame.iloc[test][[c for c in ("pilot_id", "source_dataset", "source_record_id", "source_language", "gold_stance", "original_text") if c in frame]].copy()
    pred_frame.rename(columns={"gold_stance": "reference_stance"}, inplace=True)
    for name, pred in test_predictions.items():
        pred_frame[f"predicted_{name}"] = pred
    pred_frame.to_csv(out_dir / "test_predictions.csv", index=False, lineterminator="\n")

    metrics = {
        "label_role": "repository_reference_labels",
        "provenance_note": "The existing stance labels are treated as the project reference targets. This terminology does not assert independent human verification; source provenance remains in data/ai_labels/stance_sentiment_ai_v1/provenance.json.",
        "development_split_note": "The seed-42 split has been used during prior development. This run creates a new source-record and near-duplicate grouped test fold for this fit, but the full reference pool has been exposed in prior model development; treat these new-fold results as development estimates, not an untouched independent evaluation.",
        "evaluation_status": "development_estimate_reference_pool_previously_exposed",
        "seed": args.seed,
        "counts": {"reference_records_total": len(source_frame), "reference_records_modeled": len(frame),
                   "train": len(train), "validation": len(validation), "test": len(test),
                   "leakage_groups_all_records": len(np.unique(all_leakage_groups)),
                   "leakage_groups_modeled": len(np.unique(leakage_groups))},
        "labels": labels,
        "excluded_outside_three_class_model": outside_model,
        "label_counts": {part: pd.Series(reference_labels[idx]).value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
                         for part, idx in (("train", train), ("validation", validation), ("test", test))},
        "corpus_group_mapping": mapping,
        "selected_by_validation_macro_f1": best["name"],
        "word_char_selected_by_validation_macro_f1": best_char["name"],
        "test_results": results,
        "language_test_results": language_results,
        "validation_candidates": validation_runs,
        "selection_rule": "validation macro-F1; ties use validation accuracy, then word-only over word+character, then lower C; test evaluated only after configurations frozen",
        "multilingual_pretrained": "not run in this fresh-split comparison; prior MiniLM comparison did not improve held-out results",
        "input_sha256": hashlib.sha256(REFERENCE_LABELS_PATH.read_bytes()).hexdigest(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(validation_runs).to_json(out_dir / "validation_candidates.json", orient="records", indent=2)

    report = ["# Stance model comparison using repository reference labels", "",
              metrics["provenance_note"], "", metrics["development_split_note"], "",
              f"Model schema is fixed to {', '.join(labels)}. All {len(source_frame)} source records retain their reference values and identifiers in `split_assignments.csv`; {json.dumps(outside_model)} labels outside the three-class model are excluded from fitting and recorded, not removed from source data.", "",
              f"New grouped split (seed {args.seed}): {len(train)} train / {len(validation)} validation / {len(test)} test; {len(np.unique(leakage_groups))} leakage groups.",
              "Grouped boundaries join same source records, exact duplicate groups, corpus near-duplicate groups, templates and lexical near-duplicates.",
              "Validation macro-F1 selected configurations before any new-test metrics were computed.", "",
              "| Test model | Frozen configuration | N | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---|---:|---:|---:|---:|"]
    for name, record in results.items():
        if name == "test_label_counts":
            continue
        m = record["metrics"]
        report.append(f"| {name} | {record['configuration']} | {m['n']} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} |")
    for name, record in results.items():
        if name == "test_label_counts":
            continue
        m = record["metrics"]
        report += ["", f"## {name}: per-class test metrics", "", "| Reference class | Precision | Recall | F1 | Test count |", "|---|---:|---:|---:|---:|"]
        report += [f"| {label} | {values['precision']} | {values['recall']} | {values['f1']} | {values['support']} |"
                   for label, values in m["per_class"].items()]
        report.append(f"Confusion matrix, rows reference / columns predicted in {', '.join(labels)} order: `{m['confusion_matrix']['rows_reference_cols_predicted']}`")
    report += ["", "## Language subsets", "", "English and French test subsets are recorded in `metrics.json` for the selected model, the validation-best word+character model, and word-only baselines. Interpret small subsets cautiously.",
               "", "## Remaining limitation", "", "This is a new held-out fold for this fit, but every record in the reference pool has been seen in previous development runs. Treat metrics as development estimates, not untouched test performance. It is also not an unseen consultation-source test. A clean external evaluation requires newly acquired reference labels from source records kept out of all previous development.",
               "", f"Reproduce: `python scripts/experiment_stance_reference_models_v2.py --out-dir reports/stance_reference_model_comparison_rerun1 --seed {args.seed}`"]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "selected": best["name"], "word_char": best_char["name"],
                      "results": {name: value.get("metrics") for name, value in results.items() if name != "test_label_counts"},
                      "test_label_counts": results["test_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
