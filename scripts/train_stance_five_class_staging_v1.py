"""Grouped-CV comparison and staging build for five-class synthetic stance."""
from __future__ import annotations

import hashlib
import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.context_stance_features import build_context_feature
from models.new_comment_inference import load_verified_pipeline
from models.stance_classification.labels import load_stance_reference_labels
from models.stance_classification.schema import STANCE_LABELS, ORIGINAL_EVALUABLE_LABELS
from scripts.benchmark_stance_existing_resources_v1 import fold_map, sha
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH, SEED

SYNTHETIC_DIR = ROOT / "reports/stance_five_class_synthetic_v1"
SYNTHETIC_PATH = SYNTHETIC_DIR / "synthetic_training_rows.jsonl"
OUT = ROOT / "reports/stance_five_class_staging_v1"
CANDIDATE = ROOT / "models/candidates/stance_context_wordchar_5class_synth_v1"
PRODUCTION = ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class"
C_VALUES = (0.3, 1.0, 3.0)
SYNTHETIC_COUNTS = (40, 80, 120)


def make_model(c: float):
    from scripts.benchmark_stance_context_v1 import build_model
    return build_model(c=c)


def evaluate(y_true, y_pred) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE_LABELS, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=STANCE_LABELS,
                                          average="macro", zero_division=0)), 4),
        "observed_class_macro_f1": round(float(f1_score(y_true, y_pred,
            labels=ORIGINAL_EVALUABLE_LABELS, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=STANCE_LABELS,
                                             average="weighted", zero_division=0)), 4),
        "per_class": {label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                              "f1": round(float(f[i]), 4), "support": int(support[i])}
                      for i, label in enumerate(STANCE_LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=STANCE_LABELS).tolist(),
    }


def aggregate(rows: list[dict]) -> dict:
    y = np.concatenate([np.asarray(row["y"], dtype=str) for row in rows])
    pred = np.concatenate([np.asarray(row["pred"], dtype=str) for row in rows])
    return evaluate(y, pred)


def synthetic_features(synthetic: pd.DataFrame) -> np.ndarray:
    return np.asarray([build_context_feature(r.question_wording, r.question_context, r.original_text)
                       for r in synthetic.itertuples()], dtype=str)


def choose_baseline_c(x, y, groups, splits):
    results = []
    for c in C_VALUES:
        validation_rows = []
        for train, valid in splits:
            model = make_model(c)
            model.fit(x[train], y[train])
            validation_rows.append({"y": y[valid], "pred": model.predict(x[valid])})
        metrics = aggregate(validation_rows)
        results.append({"C": c, "validation_macro_f1": metrics["observed_class_macro_f1"],
                        "validation_accuracy": metrics["accuracy"]})
    selected = max(results, key=lambda row: (row["validation_macro_f1"],
                                              row["validation_accuracy"], -row["C"]))
    return selected["C"], results


def make_augmented_training(train_idx, y, original_x, synth, synth_x,
                            count_per_class: int, seed: int):
    choose_idx = []
    for class_index, label in enumerate(("Oppose", "Mixed", "Unclear")):
        available = np.flatnonzero(synth.reference_stance.to_numpy() == label)
        if len(available) < count_per_class:
            raise RuntimeError(f"synthetic pool lacks {count_per_class} {label} rows")
        rng = np.random.default_rng(seed + class_index * 10007)
        choose_idx.extend(rng.choice(available, count_per_class, replace=False).tolist())
    choose_idx = np.asarray(choose_idx, dtype=int)
    return (np.concatenate([original_x[train_idx], synth_x[choose_idx]]),
            np.concatenate([y[train_idx], synth.reference_stance.to_numpy()[choose_idx]]),
            synth.iloc[choose_idx])


def choose_augmented_config(x, y, groups, synth, synth_x, splits, seed):
    results = []
    for count in SYNTHETIC_COUNTS:
        for c in C_VALUES:
            validation_rows = []
            for split_index, (train, valid) in enumerate(splits):
                train_x, train_y, _ = make_augmented_training(
                    train, y, x, synth, synth_x, count,
                    seed + split_index * 7919 + count * 101,
                )
                model = make_model(c)
                model.fit(train_x, train_y)
                validation_rows.append({"y": y[valid], "pred": model.predict(x[valid])})
            metrics = aggregate(validation_rows)
            results.append({"synthetic_per_class": count, "C": c,
                            "validation_macro_f1": metrics["observed_class_macro_f1"],
                            "validation_accuracy": metrics["accuracy"],
                            "validation_five_class_macro_f1_with_zero_support": metrics["macro_f1"]})
    selected = max(results, key=lambda row: (row["validation_macro_f1"],
                                              row["validation_accuracy"],
                                              -row["synthetic_per_class"], -row["C"]))
    return selected["synthetic_per_class"], selected["C"], results


def main():
    global SYNTHETIC_DIR, SYNTHETIC_PATH, OUT, CANDIDATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE)
    args = parser.parse_args()
    SYNTHETIC_DIR = args.synthetic_dir if args.synthetic_dir.is_absolute() else ROOT / args.synthetic_dir
    SYNTHETIC_PATH = SYNTHETIC_DIR / "synthetic_training_rows.jsonl"
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    CANDIDATE = args.candidate_dir if args.candidate_dir.is_absolute() else ROOT / args.candidate_dir
    OUT.mkdir(parents=True, exist_ok=True)
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    status, all_originals = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or all_originals is None:
        raise RuntimeError(status.detail)
    eligible = all_originals[all_originals.gold_stance.isin(ORIGINAL_EVALUABLE_LABELS)].reset_index(drop=True)
    if len(eligible) != 399:
        raise RuntimeError(f"expected 399 rows in the existing shared three-class folds, found {len(eligible)}")
    synth = pd.read_json(SYNTHETIC_PATH, lines=True)
    if not synth.is_synthetic.all() or not synth.training_only.all() or not synth.model_eligible.all():
        raise AssertionError("synthetic rows must be explicitly eligible, synthetic and training-only")
    if not set(synth.reference_stance).issuperset({"Oppose", "Mixed", "Unclear"}):
        raise AssertionError("synthetic pool does not cover all intended underrepresented classes")

    original_x = np.asarray([build_context_feature(r.question_wording, r.question_context, r.original_text)
                             for r in eligible.itertuples()], dtype=str)
    y = eligible.gold_stance.astype(str).to_numpy()
    saved_manifest = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl"
    saved = pd.read_json(saved_manifest, lines=True)
    saved = saved[saved.condition == "shared_original_folds"]
    groups_by_id = saved.groupby("record_id", sort=False).leakage_group.first().astype(str)
    groups = eligible.pilot_id.astype(str).map(groups_by_id).to_numpy()
    if pd.isna(groups).any():
        raise RuntimeError("some original validation records lack their saved source/duplicate group")
    outer_folds = fold_map(eligible)
    synth_x = synthetic_features(synth)

    baseline_rows, augmented_rows, prediction_rows, assignment_rows = [], [], [], []
    config_selections = []
    for outer in outer_folds:
        train, valid = outer["train"], outer["valid"]
        if set(groups[train]) & set(groups[valid]):
            raise AssertionError("saved outer split crosses a source/duplicate group")
        inner_cv = StratifiedGroupKFold(
            n_splits=3, shuffle=True,
            random_state=SEED + outer["repeat"] * 1009 + outer["fold"],
        )
        inner = list(inner_cv.split(np.zeros(len(train)), y[train], groups[train]))
        if any(set(groups[train[a]]) & set(groups[train[b]]) for a, b in inner):
            raise AssertionError("inner grouped validation split crosses source/duplicate groups")

        baseline_c, baseline_validation = choose_baseline_c(
            original_x[train], y[train], groups[train], inner
        )
        baseline = make_model(baseline_c).fit(original_x[train], y[train])
        baseline_pred = baseline.predict(original_x[valid]).astype(str)
        baseline_score = evaluate(y[valid], baseline_pred)

        synth_count, aug_c, aug_validation = choose_augmented_config(
            original_x[train], y[train], groups[train], synth, synth_x, inner,
            SEED + outer["repeat"] * 1009 + outer["fold"],
        )
        augmented_x, augmented_y, used_synthetic = make_augmented_training(
            train, y, original_x, synth, synth_x, synth_count,
            SEED + outer["repeat"] * 1009 + outer["fold"] + 700001,
        )
        if not used_synthetic.is_synthetic.all() or not used_synthetic.training_only.all():
            raise AssertionError("non-synthetic or non-training-only row entered augmentation")
        augmented = make_model(aug_c).fit(augmented_x, augmented_y)
        augmented_pred = augmented.predict(original_x[valid]).astype(str)
        augmented_score = evaluate(y[valid], augmented_pred)
        base = {"repeat": outer["repeat"], "fold": outer["fold"]}
        baseline_rows.append({**base, "selected_C": baseline_c, **baseline_score,
                              "y": y[valid].tolist(), "pred": baseline_pred.tolist()})
        augmented_rows.append({**base, "selected_C": aug_c, "synthetic_per_minority_class": synth_count,
                               "synthetic_training_rows": int(len(used_synthetic)), **augmented_score,
                               "y": y[valid].tolist(), "pred": augmented_pred.tolist()})
        config_selections.append({**base, "baseline_C": baseline_c,
                                  "baseline_inner_validation": baseline_validation,
                                  "synthetic_per_class": synth_count, "augmented_C": aug_c,
                                  "augmented_inner_validation": aug_validation})
        for idx, target, baseline_label, aug_label in zip(valid, y[valid], baseline_pred, augmented_pred):
            prediction_rows.append({"repeat": outer["repeat"], "fold": outer["fold"],
                                    "record_id": str(eligible.iloc[idx].pilot_id),
                                    "leakage_group": str(groups[idx]),
                                    "reference_stance": str(target),
                                    "baseline_prediction": str(baseline_label),
                                    "synthetic_augmented_prediction": str(aug_label),
                                    "is_synthetic_evaluation_row": False})
        for idx in np.concatenate((train, valid)):
            assignment_rows.append({"repeat": outer["repeat"], "fold": outer["fold"],
                                    "record_id": str(eligible.iloc[idx].pilot_id),
                                    "leakage_group": str(groups[idx]),
                                    "split": "train" if idx in set(train) else "validation",
                                    "is_synthetic": False})

    # Final fit configuration is selected by grouped validation over all 399 evaluable originals.
    final_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    final_splits = list(final_cv.split(np.zeros(len(y)), y, groups))
    final_count, final_c, final_validation = choose_augmented_config(
        original_x, y, groups, synth, synth_x, final_splits, SEED + 300001,
    )
    all_original_x = np.asarray([build_context_feature(r.question_wording, r.question_context, r.original_text)
                                 for r in all_originals.itertuples()], dtype=str)
    all_original_y = all_originals.gold_stance.astype(str).to_numpy()
    all_idx = np.arange(len(all_originals))
    final_x, final_y, final_synthetic = make_augmented_training(
        all_idx, all_original_y, all_original_x, synth, synth_x, final_count, SEED + 991337,
    )
    final_model = make_model(final_c).fit(final_x, final_y)
    if set(final_model.classes_) != set(STANCE_LABELS):
        raise AssertionError(f"candidate does not learn every schema label: {final_model.classes_}")

    artifact_path = CANDIDATE / "pipeline_five_class_stance.artifact"
    joblib.dump(final_model, artifact_path, compress=3)
    original_manifest = all_originals[[c for c in ("pilot_id", "source_dataset", "source_record_id",
        "source_row_number", "gold_stance") if c in all_originals]].copy()
    original_manifest = original_manifest.rename(columns={"pilot_id": "record_id", "gold_stance": "reference_stance"})
    original_manifest["is_synthetic"] = False
    original_manifest["training_source"] = "original_reference_record"
    selected_synthetic = final_synthetic[["synthetic_id", "reference_stance", "is_synthetic",
        "generation_method", "template_id", "template_family", "synthetic_reason", "random_seed"]].copy()
    selected_synthetic = selected_synthetic.rename(columns={"synthetic_id": "record_id"})
    selected_synthetic["training_source"] = "synthetic_training_example"
    training_manifest = pd.concat([original_manifest, selected_synthetic], ignore_index=True)
    train_manifest_path = CANDIDATE / "training_manifest.jsonl"
    training_manifest.to_json(train_manifest_path, orient="records", lines=True, force_ascii=False)

    prediction_path = OUT / "grouped_cv_predictions.jsonl"
    assignments_path = OUT / "fold_assignments.jsonl"
    pd.DataFrame(prediction_rows).to_json(prediction_path, orient="records", lines=True)
    pd.DataFrame(assignment_rows).to_json(assignments_path, orient="records", lines=True)
    prod_path = PRODUCTION / "pipeline_linsvm.joblib"
    manifest = {
        "candidate_id": "stance_context_wordchar_5class_synth_v1",
        "status": "staging_only_not_production", "deployed": False,
        "feature_builder": "models.context_stance_features.build_context_feature",
        "feature_order": ["question_context", "question_wording", "original_text"],
        "feature_allowlist": ["question_context", "question_wording", "original_text"],
        "forbidden_feature_families": ["labels", "reviewer decisions/notes", "label provenance",
                                        "source identifiers", "sentiment", "synthetic metadata"],
        "labels": list(STANCE_LABELS),
        "configuration": {"kind": "word_char_linsvm", "word_ngram_range": [1, 2],
                           "word_max_features": 20000, "char_ngram_range": [3, 5],
                           "char_max_features": 30000, "char_min_df": 2,
                           "class_weight": "balanced", "C": final_c,
                           "synthetic_per_class": final_count, "seed": SEED},
        "training": {"original_records": int(len(all_originals)),
                     "original_class_counts": all_originals.gold_stance.value_counts().to_dict(),
                     "synthetic_rows": int(len(final_synthetic)),
                     "synthetic_class_counts": final_synthetic.reference_stance.value_counts().to_dict(),
                     "training_manifest_sha256": sha(train_manifest_path),
                     "reference_labels_sha256": sha(REFERENCE_LABELS_PATH),
                     "synthetic_rows_sha256": sha(SYNTHETIC_PATH),
                     "synthetic_generation_manifest_sha256": sha(SYNTHETIC_DIR / "generation_manifest.json")},
        "configuration_validation": {"method": "5-fold StratifiedGroupKFold on original evaluable records; synthetic rows are appended only to each fold training partition",
                                     "selection_metric": "macro-F1 on original observed classes, then accuracy, fewer synthetic rows, lower C",
                                     "results": final_validation},
        "outer_evaluation_folds": {"repeats": 3, "folds_per_repeat": 5,
                                   "fold_manifest_sha256": sha(ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl")},
        "artifact_sha256": sha(artifact_path),
        "production_artifact_sha256": sha(prod_path),
        "limitations": ["Oppose has one original reference row and Mixed has none; neither class has original grouped evaluation support.",
                        "Metrics on existing original folds cannot validate Oppose or Mixed generalization.",
                        "Existing reference labels were previously exposed during development and are not human-verified."],
        "rollback": "No production artifact or route was changed. Stop using /api/inference/stance/five-class-staging and keep production loading data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (CANDIDATE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_metrics = {
        "evaluation_status": "saved source_duplicate_grouped_outer_folds; hyperparameters and synthetic counts selected only in grouped validation; original-only evaluation rows",
        "labels_order": list(STANCE_LABELS), "original_evaluation_labels_with_support": list(ORIGINAL_EVALUABLE_LABELS),
        "original_reference_n": len(eligible), "oof_repeated_n": int(sum(row["n"] for row in baseline_rows)),
        "original_counts": eligible.gold_stance.value_counts().reindex(STANCE_LABELS, fill_value=0).astype(int).to_dict(),
        "baseline": {"fold_metrics": baseline_rows, "summary": aggregate(baseline_rows)},
        "synthetic_augmented": {"fold_metrics": augmented_rows, "summary": aggregate(augmented_rows)},
        "outer_configuration_selections": config_selections,
        "synthetic_source": {"rows": int(len(synth)), "class_counts": synth.reference_stance.value_counts().to_dict(),
                             "generation_manifest_sha256": sha(SYNTHETIC_DIR / "generation_manifest.json"),
                             "training_rows_sha256": sha(SYNTHETIC_PATH),
                             "evaluation_synthetic_rows": 0},
        "fold_manifest_sha256": sha(saved_manifest),
        "predictions_sha256": sha(prediction_path), "fold_assignments_sha256": sha(assignments_path),
        "candidate_artifact_sha256": manifest["artifact_sha256"],
        "production_artifact_sha256": manifest["production_artifact_sha256"],
    }
    (OUT / "metrics.json").write_text(json.dumps(report_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(render_report(report_metrics, manifest), encoding="utf-8")
    print(json.dumps({"candidate": str(CANDIDATE), "report": str(OUT),
                      "baseline": report_metrics["baseline"]["summary"],
                      "synthetic_augmented": report_metrics["synthetic_augmented"]["summary"],
                      "selected": manifest["configuration"]}, ensure_ascii=False))


def render_report(metrics, manifest):
    lines = ["# Five-class stance synthetic staging evaluation", "",
             "This is a staging-only synthetic-data experiment. Original records and the production artifact remain unchanged. Synthetic data is marked training-only and never appears in an evaluation fold. Scores measure agreement with previously exposed, unverified repository reference labels.", "",
             "Original reference counts are Support 140, Neutral 219, Oppose 1, Mixed 0, Unclear 40. The saved shared grouped folds contain only the 399 Support/Neutral/Unclear records. Thus original evaluation support is zero for Oppose and Mixed; their real-world precision/recall/generalization cannot be established by this dataset.", "",
             "Both conditions use the same saved 3x5 source/duplicate-grouped folds and context-question-comment feature constructor. Each fold fits all vectorizers on training text only. The baseline sees original fold training rows only. The augmented model appends synthetic Oppose/Mixed/Unclear samples to each training split only; count and C are selected using grouped validation on original rows. The test labels are never used for selection.", "",
             "## Overall metrics", "",
             "Macro-F1 below is the requested five-class macro with zero-support Oppose/Mixed included as 0; observed-class macro-F1 is also shown to make three-class comparison interpretable.", "",
             "| Model | Repeated OOF rows | Accuracy | Five-class macro-F1 | Observed-class macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "synthetic_augmented"):
        sm = metrics[name]["summary"]
        lines.append(f"| {name} | {sm['n']} | {sm['accuracy']:.4f} | {sm['macro_f1']:.4f} | {sm['observed_class_macro_f1']:.4f} | {sm['weighted_f1']:.4f} |")
    for name in ("baseline", "synthetic_augmented"):
        sm = metrics[name]["summary"]
        lines += ["", f"## {name}: per-class precision, recall, F1 and support", "",
                  "Support is repeated OOF reference count, not synthetic training count.", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label in STANCE_LABELS:
            d = sm["per_class"][label]
            lines.append(f"| {label} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['support']} |")
        lines += ["", f"Confusion matrix (rows reference, columns predicted; order {', '.join(STANCE_LABELS)}): `{sm['confusion_matrix']}`."]
    lines += ["", f"Final training configuration selected in grouped validation: C={manifest['configuration']['C']}, synthetic count/class={manifest['configuration']['synthetic_per_class']} for Oppose/Mixed/Unclear.", "",
              f"Candidate artifact SHA-256 `{manifest['artifact_sha256']}`. Training manifest SHA-256 `{manifest['training']['training_manifest_sha256']}`. Production artifact SHA-256 `{manifest['production_artifact_sha256']}`.", "",
              "No 90% accuracy claim is supported. Oppose and Mixed need independent original examples before their generalization can be measured. Synthetic training agreement is not evidence of real-world accuracy.", "",
              "Reproduce: `python scripts/generate_stance_five_class_synthetic_v1.py` then `python scripts/train_stance_five_class_staging_v1.py`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
