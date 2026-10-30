"""Train and compare the context-aware staging model using repository resources."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.context_stance_features import build_context_feature
from models.stance_classification.labels import load_stance_reference_labels
from scripts.benchmark_stance_context_v1 import LABELS, build_model, score
from scripts.benchmark_stance_existing_resources_v1 import fold_map, sha
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

OUT = ROOT / "reports/stance_context_staging_v1"
CANDIDATE = ROOT / "models/candidates/stance_context_wordchar_v1"
PRODUCTION = ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class"
CS = (0.3, 1.0, 3.0)
SEED = 20261029


def choose_c(features, y, groups, splits):
    results = []
    for c in CS:
        fold_scores = []
        for train, valid in splits:
            model = build_model(c=c)
            model.fit(features[train], y[train])
            fold_scores.append(score(y[valid], model.predict(features[valid])))
        results.append({"C": c,
                        "macro_f1": float(np.mean([r["macro_f1"] for r in fold_scores])),
                        "accuracy": float(np.mean([r["accuracy"] for r in fold_scores]))})
    selected = max(results, key=lambda row: (row["macro_f1"], row["accuracy"], -row["C"]))
    return selected["C"], results


def summary(rows):
    keys = ("accuracy", "macro_f1", "weighted_f1")
    output = {key: {"mean": round(float(np.mean([row[key] for row in rows])), 4),
                    "std": round(float(np.std([row[key] for row in rows], ddof=1)), 4)} for key in keys}
    output["per_class"] = {}
    for label in LABELS:
        output["per_class"][label] = {}
        for metric in ("precision", "recall", "f1", "support"):
            values = [row["per_class"][label][metric] for row in rows]
            output["per_class"][label][metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)), 4),
            }
    output["pooled_confusion_matrix"] = np.sum([row["confusion_matrix"] for row in rows], axis=0).astype(int).tolist()
    output["pooled_n"] = int(sum(row["n"] for row in rows))
    return output


def production_recipe_fold_model():
    """Fold-local comparator matching deployed production's word-only recipe."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  max_features=20000)),
        ("clf", LinearSVC(C=0.3, class_weight="balanced", random_state=SEED,
                          max_iter=10000)),
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    status, raw = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or raw is None:
        raise RuntimeError(status.detail)
    frame = raw[raw.gold_stance.isin(LABELS)].reset_index(drop=True)
    if len(frame) != 399:
        raise RuntimeError(f"expected 399 eligible source rows, found {len(frame)}")

    # Allowlisted fields only: label, IDs, reviewer decisions, source IDs, sentiment,
    # and provenance are excluded. Language metadata is intentionally not a feature.
    features = np.asarray([
        build_context_feature(row.question_wording, row.question_context, row.original_text)
        for row in frame.itertuples()
    ], dtype=str)
    y = frame.gold_stance.astype(str).to_numpy()
    manifest_path = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl"
    saved = pd.read_json(manifest_path, lines=True)
    saved = saved[saved.condition == "shared_original_folds"]
    groups_by_id = saved.groupby("record_id", sort=False).leakage_group.first().astype(str)
    groups = frame.pilot_id.astype(str).map(groups_by_id).to_numpy()
    if pd.isna(groups).any():
        raise RuntimeError("some reference records have no saved leakage group")
    folds = fold_map(frame)
    # The deployed artifact was trained on this same reference pool, so running it
    # against saved folds would leak its training rows. Compare a fold-refit copy of
    # its frozen recipe instead; retain and hash the actual artifact for traceability.
    production_path = PRODUCTION / "pipeline_linsvm.joblib"
    production_sha = sha(production_path)
    candidate_fold_scores, production_fold_scores = [], []
    selections, prediction_rows, assignments = [], [], []

    for outer in folds:
        train, valid = outer["train"], outer["valid"]
        inner_cv = StratifiedGroupKFold(
            n_splits=3, shuffle=True,
            random_state=SEED + outer["repeat"] * 1009 + outer["fold"],
        )
        inner = list(inner_cv.split(np.zeros(len(train)), y[train], groups[train]))
        if any(set(groups[train[a]]) & set(groups[train[b]]) for a, b in inner):
            raise AssertionError("inner split crossed a source/duplicate group")
        c, inner_results = choose_c(features[train], y[train], groups[train], inner)
        model = build_model(c=c)
        model.fit(features[train], y[train])
        candidate_pred = model.predict(features[valid])
        candidate_metrics = score(y[valid], candidate_pred)
        production_fold_model = production_recipe_fold_model()
        raw_comments = frame.original_text.astype(str).to_numpy()
        production_fold_model.fit(raw_comments[train], y[train])
        production_pred = production_fold_model.predict(raw_comments[valid])
        production_metrics = score(y[valid], production_pred)
        base = {"repeat": outer["repeat"], "fold": outer["fold"]}
        candidate_fold_scores.append({**base, "selected_C": c, **candidate_metrics})
        production_fold_scores.append({**base, **production_metrics})
        selections.append({**base, "selected_C": c, "inner_validation": inner_results})
        for idx, target, cand, prod in zip(valid, y[valid], candidate_pred, production_pred):
            prediction_rows.append({"repeat": outer["repeat"], "fold": outer["fold"],
                                    "record_id": str(frame.iloc[idx].pilot_id),
                                    "source_dataset": str(frame.iloc[idx].source_dataset),
                                    "reference_stance": str(target),
                                    "context_staging_prediction": str(cand),
                                    "production_prediction": str(prod)})
        for idx in np.concatenate((train, valid)):
            assignments.append({"repeat": outer["repeat"], "fold": outer["fold"],
                                "record_id": str(frame.iloc[idx].pilot_id),
                                "leakage_group": str(groups[idx]),
                                "split": "train" if idx in set(train) else "validation"})

    # Select final artifact C using validation only over all existing reference rows.
    final_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    final_splits = list(final_cv.split(np.zeros(len(y)), y, groups))
    selected_c, final_validation = choose_c(features, y, groups, final_splits)
    final_pipeline = build_model(c=selected_c)
    final_pipeline.fit(features, y)
    artifact_path = CANDIDATE / "pipeline_context_wordchar.artifact"
    joblib.dump(final_pipeline, artifact_path, compress=3)

    # Training manifest gives exact source rows and labels without exposing their text.
    training_manifest = pd.DataFrame({
        "record_id": frame.pilot_id.astype(str),
        "source_dataset": frame.source_dataset.astype(str),
        "source_record_id": frame.source_record_id.astype(str),
        "reference_stance": y,
        "leakage_group": groups.astype(str),
    })
    train_manifest_path = CANDIDATE / "training_rows.jsonl"
    training_manifest.to_json(train_manifest_path, orient="records", lines=True)
    fold_path = OUT / "fold_assignments.jsonl"
    pred_path = OUT / "grouped_cv_predictions.jsonl"
    pd.DataFrame(assignments).to_json(fold_path, orient="records", lines=True)
    pd.DataFrame(prediction_rows).to_json(pred_path, orient="records", lines=True)
    prod_sha = production_sha
    manifest = {
        "candidate_id": "stance_context_wordchar_v1", "status": "staging_only_not_production",
        "feature_mode": "question_context + question_wording + original_text",
        "feature_builder": "models.context_stance_features.build_context_feature",
        "feature_allowlist": ["question_wording", "question_context", "original_text"],
        "forbidden_feature_families": ["reference/AI labels", "reviewer decisions/notes", "label provenance",
                                         "source identifiers", "sentiment outputs", "language metadata"],
        "configuration": {"kind": "word_char_linsvm", "word_ngram_range": [1, 2],
                           "word_max_features": 20000, "char_analyzer": "char_wb",
                           "char_ngram_range": [3, 5], "char_max_features": 30000,
                           "char_min_df": 2, "class_weight": "balanced", "C": selected_c,
                           "seed": SEED},
        "training": {"rows": len(frame), "class_counts": frame.gold_stance.value_counts().to_dict(),
                     "reference_labels": "repository expansion_v2, unverified development labels",
                     "training_rows_sha256": sha(train_manifest_path),
                     "reference_data_sha256": sha(REFERENCE_LABELS_PATH)},
        "validation_selection": {"method": "5-fold StratifiedGroupKFold, grouped by saved source/duplicate leakage groups",
                                  "metric": "macro-F1, then accuracy, then lower C",
                                  "candidates": list(CS), "results": final_validation},
        "shared_outer_folds": {"repeats": 3, "folds_per_repeat": 5,
                               "fold_manifest_sha256": sha(manifest_path)},
        "artifact_sha256": sha(artifact_path),
        "production_artifact_sha256": prod_sha,
        "rollback": "No production routing changed. Keep /api/inference/stance and /api/inference/consultation unchanged; disable/remove only the /api/inference/stance/context-staging route or its demo UI panel. Production remains data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (CANDIDATE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result = {
        "evaluation_status": "same_saved_3x5_source_duplicate_grouped_folds; inner grouped validation for C; development-only previously exposed reference labels",
        "labels_order": LABELS, "source_rows": len(frame), "repeated_oof_n": int(sum(r["n"] for r in candidate_fold_scores)),
        "outer_fold_sha256": sha(manifest_path),
        "context_candidate": {"fold_metrics": candidate_fold_scores, "summary": summary(candidate_fold_scores)},
        "production": {"comparison": "fold-local refit of frozen deployed word-only recipe; full-data deployed artifact not scored on its own training pool",
                       "artifact_sha256": prod_sha, "fold_metrics": production_fold_scores,
                       "summary": summary(production_fold_scores)},
        "validation_selected_final_C": selected_c, "final_C_validation_results": final_validation,
        "unclear_confusion_candidate": summary(candidate_fold_scores)["pooled_confusion_matrix"][2],
        "unclear_confusion_production": summary(production_fold_scores)["pooled_confusion_matrix"][2],
        "predictions_sha256": sha(pred_path), "fold_assignments_sha256": sha(fold_path),
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(render_report(result, manifest), encoding="utf-8")
    print(json.dumps({"candidate": str(CANDIDATE), "report": str(OUT),
                      "candidate_summary": result["context_candidate"]["summary"],
                      "production_summary": result["production"]["summary"], "selected_C": selected_c}, indent=2))


def render_report(result, manifest):
    lines = ["# Context-aware stance staging prototype", "",
             "This is a separate staging-only model and route. The production artifact and response contract remain unchanged. Results are agreement with previously exposed, unverified repository reference labels, not external or human-validated accuracy.", "",
             "The context model uses question wording, question context, and comment text. It excludes labels, provenance, source IDs, reviewer notes/decisions, sentiment, and language metadata. All 399 rows and the exact saved 3x5 source/duplicate-grouped folds were used. Each outer training split independently selected C through three-fold grouped validation; all vectorization was fitted within training partitions. The production comparator refits the deployed word-only recipe (word 1-2 gram TF-IDF, C=0.3, balanced) inside each outer training fold. The actual deployed artifact was trained on this reference pool and is not evaluated directly on its own training rows.", "",
             "| Model | OOF rows (repeated) | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Weighted-F1 mean +/- SD |", "|---|---:|---:|---:|---:|"]
    for name in ("context_candidate", "production"):
        sm = result[name]["summary"]
        lines.append(f"| {name} | {sm['pooled_n']} | {sm['accuracy']['mean']:.4f} +/- {sm['accuracy']['std']:.4f} | {sm['macro_f1']['mean']:.4f} +/- {sm['macro_f1']['std']:.4f} | {sm['weighted_f1']['mean']:.4f} +/- {sm['weighted_f1']['std']:.4f} |")
    for name in ("context_candidate", "production"):
        sm = result[name]["summary"]
        lines += ["", f"## {name}: per-class mean precision / recall / F1", "", "| Class | Precision | Recall | F1 | Mean support per fold |", "|---|---:|---:|---:|---:|"]
        for label in LABELS:
            item = sm["per_class"][label]
            lines.append(f"| {label} | {item['precision']['mean']:.4f} | {item['recall']['mean']:.4f} | {item['f1']['mean']:.4f} | {item['support']['mean']:.1f} |")
        lines += ["", f"Pooled repeated out-of-fold confusion matrix (rows reference, columns predicted; order {', '.join(LABELS)}): `{sm['pooled_confusion_matrix']}`."]
    lines += ["", "## Unclear behavior", "",
              f"Context candidate: {result['unclear_confusion_candidate'][2]} correct Unclear predictions from {sum(result['unclear_confusion_candidate'])} reference Unclear instances across repeats. Production: {result['unclear_confusion_production'][2]} correct from {sum(result['unclear_confusion_production'])}. The full row shows errors into Support and Neutral; no threshold or class-balance forcing was used.", "",
              f"Final artifact C={manifest['configuration']['C']} was selected using five-fold grouped validation on the full available training pool (macro-F1, then accuracy, then lower C). The evaluation metrics above remain from the untouched shared outer folds.", "",
              f"Artifact: `models/candidates/stance_context_wordchar_v1/pipeline_context_wordchar.artifact` (SHA-256 `{manifest['artifact_sha256']}`). Training rows manifest SHA-256 `{manifest['training']['training_rows_sha256']}`. Production artifact SHA-256 `{manifest['production_artifact_sha256']}`.", "",
              "Demo route: `POST /api/inference/stance/context-staging`. Rollback: disable the staging route/UI panel; production endpoints and artifact were not switched.", "",
              "Reproduce: `python scripts/build_stance_context_staging_v1.py`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
