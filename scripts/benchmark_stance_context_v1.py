"""Compare comment-only and existing-context stance features on saved grouped folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from scripts.benchmark_stance_existing_resources_v1 import fold_map, sha
from scripts.experiment_stance_reference_models_v2 import MODEL_LABELS, REFERENCE_LABELS_PATH

LABELS = list(MODEL_LABELS)
FEATURE_COLUMNS = {
    "comment_only": ("original_text",),
    "question_plus_comment": ("question_wording", "original_text"),
    "context_question_comment": ("question_context", "question_wording", "original_text"),
    "verified_language_comment": ("source_language", "language_basis", "original_text"),
}
FORBIDDEN_COLUMNS = {
    "pilot_id", "source_record_id", "gold_stance", "ai_stance", "ai_review_notes",
    "label_source", "excluded_reason", "round", "stance_A", "stance_B",
    "adjudicated_stance", "review_notes", "reviewer_decision", "sentiment",
}
MODES = tuple(FEATURE_COLUMNS)
CS = (0.3, 1.0, 3.0)


def clean(value, missing_token: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return missing_token
    return " ".join(str(value).split())


def feature_input(frame: pd.DataFrame, mode: str) -> pd.Series:
    """Assemble text exclusively from the explicit feature allowlist for one mode."""
    if mode not in FEATURE_COLUMNS:
        raise ValueError(f"unsupported feature mode: {mode}")
    if set(FEATURE_COLUMNS[mode]) & FORBIDDEN_COLUMNS:
        raise AssertionError("a target, reviewer, or provenance field entered the feature allowlist")
    text = frame.original_text.map(lambda value: clean(value, "[NO_COMMENT]"))
    if mode == "comment_only":
        return text.reset_index(drop=True)
    if mode == "question_plus_comment":
        question = frame.question_wording.map(lambda value: clean(value, "[NO_QUESTION]"))
        return ("[QUESTION] " + question + " [COMMENT] " + text).reset_index(drop=True)
    if mode == "context_question_comment":
        context = frame.question_context.map(lambda value: clean(value, "[NO_CONTEXT]"))
        question = frame.question_wording.map(lambda value: clean(value, "[NO_QUESTION]"))
        return ("[CONTEXT] " + context + " [QUESTION] " + question + " [COMMENT] " + text).reset_index(drop=True)
    basis = frame.language_basis.fillna("").astype(str)
    language = frame.source_language.fillna("").astype(str)
    verified = basis.str.contains("source_language", case=False, regex=False)
    marker = np.where(verified, language.str.casefold().to_numpy(), "unknown")
    return (pd.Series("[VERIFIED_COMMENT_LANGUAGE] " + marker + " [COMMENT] " + text.to_numpy())
            .reset_index(drop=True))


def context_missing_flags(frame: pd.DataFrame) -> pd.Series:
    columns = ("question_wording", "question_context", "context")
    missing = pd.Series(False, index=frame.index)
    for column in columns:
        missing |= frame[column].isna() | frame[column].astype(str).str.strip().eq("")
    return missing


def build_model(*, strip_accents: bool = False, c: float = 1.0):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000,
                           strip_accents="unicode" if strip_accents else None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           max_features=30000, min_df=2, strip_accents="unicode")
    return Pipeline([
        ("tfidf", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LinearSVC(C=c, class_weight="balanced", random_state=20261029, max_iter=10000)),
    ])


def score(y_true, y_pred) -> dict:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)), 4),
        "per_class": {label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                              "f1": round(float(f[i]), 4), "support": int(support[i])}
                      for i, label in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        values = [row[key] for row in rows]
        out[key] = {"mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values, ddof=1)), 4)}
    out["per_class"] = {}
    for label in LABELS:
        out["per_class"][label] = {}
        for key in ("precision", "recall", "f1", "support"):
            values = [row["per_class"][label][key] for row in rows]
            out["per_class"][label][key] = {"mean": round(float(np.mean(values)), 4),
                                            "std": round(float(np.std(values, ddof=1)), 4)}
    out["pooled_confusion_matrix"] = np.sum([row["confusion_matrix"] for row in rows], axis=0).astype(int).tolist()
    out["pooled_n"] = int(sum(row["n"] for row in rows))
    return out


def source_coverage(frame: pd.DataFrame) -> dict:
    fields = ("original_text", "question_wording", "question_context", "context", "source_language",
              "language_basis")
    result = {}
    for source, subset in frame.groupby("source_dataset", sort=True):
        result[source] = {"n": int(len(subset))}
        for field in fields:
            values = subset[field].fillna("").astype(str).str.strip()
            count = int(values.ne("").sum())
            result[source][field] = {"present": count, "missing": int(len(subset) - count),
                                     "coverage": round(count / len(subset), 4)}
    return result


def run(args) -> dict:
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    status, source = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or source is None:
        raise RuntimeError(status.detail)
    frame = source[source.gold_stance.isin(LABELS)].reset_index(drop=True)
    if len(frame) != 399:
        raise AssertionError(f"expected 399 reference rows, got {len(frame)}")

    # Guard the feature boundary before constructing any model input.
    for feature_columns in FEATURE_COLUMNS.values():
        if set(feature_columns) & FORBIDDEN_COLUMNS:
            raise AssertionError("target or review metadata is in the feature allowlist")
    feature_text = {mode: feature_input(frame, mode).to_numpy(dtype=str) for mode in MODES}
    y = frame.gold_stance.astype(str).to_numpy()
    groups_manifest_path = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl"
    saved = pd.read_json(groups_manifest_path, lines=True)
    saved = saved[saved.condition == "shared_original_folds"]
    group_by_id = saved.groupby("record_id", sort=False).leakage_group.first().astype(str)
    groups = frame.pilot_id.astype(str).map(group_by_id).to_numpy()
    if pd.isna(groups).any():
        raise AssertionError("missing saved leakage-group mapping")
    folds = fold_map(frame)

    fold_metrics = {mode: [] for mode in MODES}
    inner_selection = []
    prediction_rows, assignment_rows = [], []
    missing_context = context_missing_flags(frame).to_numpy()
    languages = frame.source_language.fillna("unknown").astype(str).to_numpy()
    for outer in folds:
        train, valid = outer["train"], outer["valid"]
        fold_group_train, fold_group_valid = set(groups[train]), set(groups[valid])
        if fold_group_train & fold_group_valid:
            raise AssertionError("saved outer fold crosses source/duplicate groups")
        seed = 20261029 + outer["repeat"] * 1009 + outer["fold"]
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        inner_splits = list(inner_cv.split(np.zeros(len(train)), y[train], groups[train]))
        for inner_train, inner_valid in inner_splits:
            if set(groups[train[inner_train]]) & set(groups[train[inner_valid]]):
                raise AssertionError("inner validation split crosses source/duplicate groups")

        for index, row_index in enumerate(np.concatenate([train, valid])):
            assignment_rows.append({
                "repeat": outer["repeat"], "fold": outer["fold"], "record_id": str(frame.iloc[row_index].pilot_id),
                "source_dataset": str(frame.iloc[row_index].source_dataset),
                "source_record_id": str(frame.iloc[row_index].source_record_id),
                "leakage_group": str(groups[row_index]), "reference_stance": str(y[row_index]),
                "split": "train" if index < len(train) else "validation",
            })

        for mode in MODES:
            candidates = []
            for c in CS:
                inner_scores = []
                for inner_train, inner_valid in inner_splits:
                    fit_idx, val_idx = train[inner_train], train[inner_valid]
                    strip = mode == "verified_language_comment"
                    model = build_model(strip_accents=strip, c=c)
                    model.fit(feature_text[mode][fit_idx], y[fit_idx])
                    inner_scores.append(score(y[val_idx], model.predict(feature_text[mode][val_idx])))
                candidates.append({"C": c, "macro_f1": float(np.mean([s["macro_f1"] for s in inner_scores])),
                                   "accuracy": float(np.mean([s["accuracy"] for s in inner_scores]))})
            chosen = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"], -row["C"]))
            strip = mode == "verified_language_comment"
            model = build_model(strip_accents=strip, c=chosen["C"])
            model.fit(feature_text[mode][train], y[train])
            pred = model.predict(feature_text[mode][valid])
            outer_score = score(y[valid], pred)
            fold_metrics[mode].append({"repeat": outer["repeat"], "fold": outer["fold"],
                                       "selected_C": chosen["C"], **outer_score})
            inner_selection.append({"mode": mode, "repeat": outer["repeat"], "fold": outer["fold"],
                                    "feature_columns": list(FEATURE_COLUMNS[mode]),
                                    "inner_candidates": candidates, "selected_C": chosen["C"],
                                    "outer_metrics": outer_score})
            for offset, row_index in enumerate(valid):
                prediction_rows.append({
                    "mode": mode, "repeat": outer["repeat"], "fold": outer["fold"],
                    "record_id": str(frame.iloc[row_index].pilot_id),
                    "language": languages[row_index], "context_missing": bool(missing_context[row_index]),
                    "reference_stance": str(y[row_index]), "predicted_stance": str(pred[offset]),
                    "is_error": bool(pred[offset] != y[row_index]),
                })

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_json(out / "grouped_cv_predictions.jsonl", orient="records", lines=True)
    pd.DataFrame(assignment_rows).to_json(out / "fold_assignments.jsonl", orient="records", lines=True)
    summaries = {mode: summarize(rows) for mode, rows in fold_metrics.items()}
    prediction_distribution = {}
    for mode, rows in predictions.groupby("mode", sort=True):
        counts = rows.predicted_stance.value_counts().reindex(LABELS, fill_value=0)
        prediction_distribution[mode] = {
            label: {"count": int(counts[label]), "share": round(float(counts[label] / len(rows)), 4)}
            for label in LABELS
        }
    language_metrics = {}
    missing_metrics = {}
    for mode, rows in predictions.groupby("mode", sort=True):
        language_metrics[mode] = {
            lang: score(group.reference_stance.to_numpy(), group.predicted_stance.to_numpy())
            for lang, group in rows.groupby("language", sort=True)
        }
        missing = rows[rows.context_missing]
        present = rows[~rows.context_missing]
        missing_metrics[mode] = {
            "context_missing": score(missing.reference_stance.to_numpy(), missing.predicted_stance.to_numpy())
                if len(missing) else {"n": 0, "metrics": None, "note": "No modeled rows have missing question/context fields."},
            "context_present": score(present.reference_stance.to_numpy(), present.predicted_stance.to_numpy()),
        }

    # Audit-only description of inputs; reviewer/label fields are not model features.
    language_verified = frame.language_basis.fillna("").astype(str).str.contains(
        "source_language", case=False, regex=False)
    context_equal = frame.context.fillna("").astype(str).eq(frame.question_context.fillna("").astype(str))
    context_inputs = {
        "rows": int(len(frame)),
        "coverage_by_source": source_coverage(frame),
        "overall_present": {field: int(frame[field].fillna("").astype(str).str.strip().ne("").sum())
                            for field in ("original_text", "question_wording", "question_context", "context",
                                          "source_language", "language_basis")},
        "overall_missing": {field: int((frame[field].isna() |
            frame[field].fillna("").astype(str).str.strip().eq("")).sum())
                            for field in ("original_text", "question_wording", "question_context", "context",
                                          "source_language", "language_basis")},
        "question_wording_unique": int(frame.question_wording.nunique()),
        "question_context_unique": int(frame.question_context.nunique()),
        "context_unique": int(frame.context.nunique()),
        "context_equals_question_context_rows": int(context_equal.sum()),
        "verified_comment_language_rows": int(language_verified.sum()),
        "language_not_verified_as_comment_language_rows": int((~language_verified).sum()),
        "language_basis_counts": frame.language_basis.value_counts(dropna=False).to_dict(),
        "missing_context_rows": int(missing_context.sum()),
        "feature_columns_by_mode": {mode: list(cols) for mode, cols in FEATURE_COLUMNS.items()},
        "explicitly_forbidden_feature_columns": sorted(FORBIDDEN_COLUMNS),
    }
    metrics = {
        "evaluation_status": "nested_inner_selection_with_saved_repeated_source_duplicate_grouped_outer_folds; prior_pool_exposure_development_only",
        "seed": 20261029, "reference_n": len(frame), "labels": LABELS,
        "outer_fold_source": str(groups_manifest_path.relative_to(ROOT)),
        "outer_folds": {"repeats": 3, "folds_per_repeat": 5, "total": len(folds),
                        "source_manifest_sha256": sha(groups_manifest_path)},
        "inner_selection": {"kind": "3-fold StratifiedGroupKFold within each outer training fold",
                            "selection_metric": "macro-F1, then accuracy, then lower C",
                            "C_values": list(CS), "class_weight": "balanced"},
        "context_audit": context_inputs,
        "model_configuration": {
            "kind": "word_char_linsvm", "word_ngram_range": [1, 2], "word_max_features": 20000,
            "character_analyzer": "char_wb", "character_ngram_range": [3, 5],
            "character_max_features": 30000, "character_min_df": 2,
            "class_weight": "balanced",
            "language_aware_variant": "verified source-comment language marker only; unknown when language_basis indicates questionnaire/survey language",
            "fit_boundary": "TfidfVectorizer and LinearSVC fitted on each inner training split for selection and each outer training split for scoring",
        },
        "fold_metrics": fold_metrics, "fold_summary_mean_std": summaries,
        "pooled_repeated_oof_prediction_distribution": prediction_distribution,
        "language_oof_metrics": language_metrics, "missing_context_oof_metrics": missing_metrics,
        "inner_configuration_selections": inner_selection,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "predictions_sha256": sha(out / "grouped_cv_predictions.jsonl"),
        "fold_assignments_sha256": sha(out / "fold_assignments.jsonl"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Context feature stance benchmark", "",
        "Existing comment, question, context, and language fields only; no synthetic examples, model downloads, reviewer notes, source labels, or target-derived features. All reported metrics are agreement with previously exposed, unverified repository reference labels, not independent human-verified accuracy.", "",
        "The 3-repeat x 5-fold outer assignments are copied from the v4 benchmark's shared source/duplicate-grouped manifest. Each input mode uses the same 399 rows and folds. C is selected independently within each outer training fold by three-fold grouped validation. Text vectorizers are fitted inside each inner and outer training fold.", "",
        f"Input coverage is complete for comment, question wording, question context, context, and source language on all {len(frame)} rows. Context and question_context are identical in {int(context_equal.sum())}/{len(frame)} rows, so the combined mode includes that field once. Verified comment-language metadata is available on {int(language_verified.sum())} rows; the other {int((~language_verified).sum())} rows describe questionnaire language and are marked unknown for the language-aware mode. No rows have missing question/context values, so missing-context subgroup performance is not estimable from this pool.", "",
        "| Input mode | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Weighted-F1 mean +/- SD | Unclear P/R/F1 mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, summary in summaries.items():
        u = summary["per_class"]["Unclear"]
        lines.append(f"| {mode} | {summary['accuracy']['mean']:.4f} +/- {summary['accuracy']['std']:.4f} | {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f} | {summary['weighted_f1']['mean']:.4f} +/- {summary['weighted_f1']['std']:.4f} | {u['precision']['mean']:.4f} / {u['recall']['mean']:.4f} / {u['f1']['mean']:.4f} |")
    lines += ["", "## Pooled repeated out-of-fold prediction counts", "", "Each source row appears once per outer repeat.", "", "| Input mode | Support | Neutral | Unclear |", "|---|---:|---:|---:|"]
    for mode, counts in prediction_distribution.items():
        lines.append(f"| {mode} | {counts['Support']['count']} ({counts['Support']['share']:.1%}) | {counts['Neutral']['count']} ({counts['Neutral']['share']:.1%}) | {counts['Unclear']['count']} ({counts['Unclear']['share']:.1%}) |")
    lines += ["", "Per-fold, per-class, confusion, language-subset, inner C-selection, missing-context, and prediction-distribution outputs are in metrics.json. The prediction file contains record IDs and outcomes but no raw comment/context text.", "",
              "The context missing subgroup has n=0. The context-present subgroup is reported for every mode; do not infer a missing-context effect from this dataset. Unclear remains a heterogeneous label, including cases where its reason is not recoverable from text.", "",
              "This benchmark does not stage a candidate. It is a feature comparison only; production and existing staging artifacts remain untouched.", "",
              "Reproduce: python scripts/benchmark_stance_context_v1.py --out-dir reports/stance_context_benchmark_v1"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "summaries": {
        mode: {"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}
        for mode, summary in summaries.items()}, "input_audit": context_inputs}, ensure_ascii=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/stance_context_benchmark_v1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
