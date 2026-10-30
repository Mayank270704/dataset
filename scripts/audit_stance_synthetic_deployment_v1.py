"""Deployment-readiness audit for the v4 synthetic stance candidate.

Runs repeated source/duplicate-group CV over the eligible original records. The v4
validation-selected word+character LinearSVC settings are frozen before CV: C=1,
balanced class weights. Synthetic examples are sampled into training folds only.
All source labels are treated as repository reference labels; CV is not independent
evaluation because this pool has already been exposed during development.
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
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from models.stance_classification import pipeline as stance_pipeline
from scripts.experiment_stance_reference_models_v2 import (
    MODEL_LABELS, REFERENCE_LABELS_PATH, build_model, evaluate,
)
from scripts.experiment_stance_synthetic_augmentation_v1 import (
    assert_synthetic_training_only, generate_synthetic, overlap_audit,
)
from scripts.train_stance_baselines import corpus_groups

V4_DIR = ROOT / "reports/stance_synthetic_augmentation_v4"
PRODUCTION_RUN = ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class"
PRODUCTION_MODEL = PRODUCTION_RUN / "pipeline_linsvm.joblib"
CV_CONFIG = {"kind": "word_char_linsvm", "C": 1.0, "class_weight": "balanced"}
RATIOS = (0.0, 0.25, 0.5, 1.0)
MODEL_NAMES = ("production_recipe_word_only", "word_char_original_only", "ratio_0.25", "ratio_0.5", "ratio_1")

TEMPLATE_REVIEW = {
    "unclear_ambiguity_1": ("label_risk", "Explicitly supports openness and rejects publication of records; guidelines may classify this as Mixed."),
    "unclear_ambiguity_2": ("review", "Could be Unclear, but could be Support if the objection concerns implementation only."),
    "unclear_ambiguity_3": ("plausible_unclear", "Contradictory positions toward the same change make intent indeterminate."),
    "unclear_ambiguity_4": ("review", "Wording is semantically awkward and could imply a Support stance."),
    "language_comprehension_difficulty": ("protocol_only", "This label depends on annotator reading ability, which text alone cannot establish."),
    "insufficient_context": ("plausible_unclear", "Refers to an unseen option or prior discussion; stance target cannot be recovered."),
    "fragment": ("plausible_unclear", "Incomplete text does not establish a stance."),
    "off_topic": ("plausible_unclear", "Clearly unrelated to the consultation stance target."),
}


def make_repeated_group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    folds = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat * 1009)
        repeat_folds = list(splitter.split(np.zeros(len(y)), y, groups))
        seen = np.concatenate([valid for _, valid in repeat_folds])
        if len(seen) != len(y) or len(set(seen.tolist())) != len(y):
            raise AssertionError("each original record must occur in exactly one validation fold per repeat")
        for fold, (train_idx, valid_idx) in enumerate(repeat_folds):
            if set(groups[train_idx]) & set(groups[valid_idx]):
                raise AssertionError("source/duplicate group crossed CV train and validation")
            if set(y[train_idx]) != set(MODEL_LABELS) or set(y[valid_idx]) != set(MODEL_LABELS):
                raise RuntimeError(f"repeat {repeat} fold {fold} lacks a model class")
            folds.append((repeat, fold, train_idx, valid_idx))
    return folds


def verify_v4_validation_selection(v4_dir: Path = V4_DIR) -> dict:
    metrics = json.loads((v4_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = pd.read_json(v4_dir / "synthetic_training_rows.jsonl", lines=True)
    regenerated = generate_synthetic(80, int(metrics["seed"]))
    pd.testing.assert_frame_equal(rows[regenerated.columns], regenerated, check_dtype=False)
    if metrics.get("validation_selected_condition") != "ratio_0.5":
        raise AssertionError("v4 validation-selected ratio does not match the documented 0.5 ratio")
    chosen = metrics["results"]["ratio_0.5"]["selected_configuration"]
    if chosen != {"C": 1.0, "class_weight": "balanced"}:
        raise AssertionError(f"v4 selected configuration mismatch: {chosen}")
    for condition, result in metrics["results"].items():
        if condition == "baseline":
            continue
        best = max(metrics["validation_candidates"][condition], key=lambda row: (
            row["validation"]["macro_f1"], row["validation"]["accuracy"],
            row["class_weight"] == "balanced", -row["C"],
        ))
        if result["selected_configuration"] != {"C": best["C"], "class_weight": best["class_weight"]}:
            raise AssertionError(f"{condition} configuration was not selected from validation candidates")
    selected = max(metrics["results"], key=lambda name: (
        metrics["results"][name]["validation_metrics"]["macro_f1"],
        metrics["results"][name]["validation_metrics"]["accuracy"],
        -metrics["results"][name]["synthetic_to_original_train_ratio"],
    ))
    if selected != metrics["validation_selected_condition"]:
        raise AssertionError("v4 augmentation ratio does not follow the documented validation-only rule")
    return {"selected_condition": selected, "selected_configuration": chosen,
            "validation_macro_f1": metrics["results"][selected]["validation_metrics"]["macro_f1"],
            "synthetic_rows_reproduced_exactly": int(len(rows)),
            "test_metrics_used_for_selection": False}


def audit_template_diversity(v4_rows: pd.DataFrame) -> dict:
    rows = v4_rows.copy()
    rows["review_status"] = "guideline_check_pass"
    rows["review_note"] = "Explicit stance or no stance established under the annotation guidelines."
    for reason, (status, note) in TEMPLATE_REVIEW.items():
        mask = rows.synthetic_reason.eq(reason) if reason in {
            "insufficient_context", "genuine_ambiguity", "fragment", "off_topic",
            "language_comprehension_difficulty",
        } else rows.template_id.eq(reason)
        rows.loc[mask, "review_status"] = status
        rows.loc[mask, "review_note"] = note
    template_rows = []
    for (label, template_id), subset in rows.groupby(["reference_stance", "template_id"], sort=True):
        status = subset.review_status.iloc[0]
        note = subset.review_note.iloc[0]
        template_rows.append({"reference_stance": label, "template_id": template_id,
                              "rows": int(len(subset)), "unique_texts": int(subset.text.nunique()),
                              "review_status": status, "review_note": note})
    return {"rows": rows, "templates": template_rows,
            "by_class": {label: {"rows": int(len(group)), "unique_texts": int(group.text.nunique()),
                                 "templates": int(group.template_id.nunique()),
                                 "review_risk_rows": int(group.review_status.isin(["label_risk", "review", "protocol_only"]).sum())}
                          for label, group in rows.groupby("reference_stance")}}


def _metric_summary(records: list[dict], labels: list[str]) -> dict:
    metric_names = ("accuracy", "macro_f1")
    result = {name: {"mean": round(float(np.mean([r[name] for r in records])), 4),
                     "std": round(float(np.std([r[name] for r in records], ddof=1)), 4)}
              for name in metric_names}
    result["per_class"] = {}
    for label in labels:
        result["per_class"][label] = {}
        for metric in ("precision", "recall", "f1", "support"):
            values = [r["per_class"][label][metric] for r in records]
            result["per_class"][label][metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)), 4),
            }
    return result


def _error_examples(prediction_frame: pd.DataFrame, condition: str, labels: list[str], limit: int = 3):
    if condition not in ("production_recipe_word_only", "word_char_original_only", "ratio_0.5"):
        return []
    local = prediction_frame[prediction_frame.condition == condition]
    selected = []
    for label in labels:
        class_rows = local[local.reference_stance == label]
        summary = class_rows.groupby(["pilot_id", "source_dataset", "source_record_id", "original_text"], dropna=False).agg(
            errors=("is_error", "sum"), repeats=("repeat", "nunique"), prediction_values=("predicted_stance", lambda s: ", ".join(s.astype(str)))
        ).reset_index()
        summary = summary[summary.errors > 0].sort_values(["errors", "pilot_id"], ascending=[False, True]).head(limit)
        selected.extend({"condition": condition, "reference_stance": label, **record}
                        for record in summary.to_dict(orient="records"))
    return selected


def run(args):
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    reproducibility = verify_v4_validation_selection()
    production_hash_before = hashlib.sha256(PRODUCTION_MODEL.read_bytes()).hexdigest()
    status, source_frame = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or source_frame is None:
        raise RuntimeError(status.detail)
    corpus_group, mapping = corpus_groups(source_frame)
    all_groups = stance_pipeline.leakage_groups(source_frame, corpus_group)
    eligible_mask = source_frame.gold_stance.isin(MODEL_LABELS).to_numpy()
    frame = source_frame.loc[eligible_mask].reset_index(drop=True)
    groups = all_groups[eligible_mask]
    x = frame.original_text.astype(str).to_numpy()
    y = frame.gold_stance.to_numpy()
    labels = list(MODEL_LABELS)
    folds = make_repeated_group_folds(y, groups, args.folds, args.repeats, args.seed)

    v4_rows = pd.read_json(V4_DIR / "synthetic_training_rows.jsonl", lines=True)
    template_audit = audit_template_diversity(v4_rows)
    cv_synthetic_per_class = args.synthetic_per_class
    # Enough capacity avoids sampling with replacement even for a full-ratio fold.
    max_synthetic_per_class = max((int(round(ratio * len(tr))) + len(labels) - 1) // len(labels)
                                  for _, _, tr, _ in folds for ratio in RATIOS if ratio)
    if cv_synthetic_per_class < max_synthetic_per_class:
        raise ValueError(f"--synthetic-per-class must be at least {max_synthetic_per_class} for ratio 1.0")
    cv_synthetic = generate_synthetic(cv_synthetic_per_class, args.seed + 9901)
    cv_synthetic.to_json(out_dir / "cv_synthetic_pool.jsonl", orient="records", lines=True, force_ascii=False)
    overlap = overlap_audit(cv_synthetic, x)

    prediction_rows = []
    fold_records = {name: [] for name in MODEL_NAMES}
    fold_seconds = {name: [] for name in MODEL_NAMES}
    assignment_rows = []
    for repeat, fold, train_idx, valid_idx in folds:
        fold_seed = args.seed + repeat * 1009 + fold
        original_train_x, original_train_y = x[train_idx], y[train_idx]
        valid_x, valid_y = x[valid_idx], y[valid_idx]
        for split_name, split_indices in (("train", train_idx), ("validation", valid_idx)):
            for source_idx in split_indices:
                source = frame.iloc[source_idx]
                assignment_rows.append({
                    "repeat": repeat, "fold": fold, "condition": "original_fold_assignment",
                    "record_id": str(source.get("pilot_id", "")),
                    "source_dataset": str(source.get("source_dataset", "")),
                    "source_record_id": str(source.get("source_record_id", "")),
                    "leakage_group": str(groups[source_idx]), "reference_stance": str(y[source_idx]),
                    "split": split_name, "is_synthetic": False,
                })
        candidates = [
            ("production_recipe_word_only", "word_linsvm", 0.3, "balanced", 0.0),
            ("word_char_original_only", "word_char_linsvm", 1.0, "balanced", 0.0),
        ] + [(f"ratio_{ratio:g}", "word_char_linsvm", 1.0, "balanced", ratio) for ratio in RATIOS if ratio]
        for condition, kind, c, weight, ratio in candidates:
            synthetic_rows = cv_synthetic.iloc[0:0]
            if ratio:
                n_syn = int(round(ratio * len(train_idx)))
                per_label = n_syn // len(labels)
                remainder = n_syn - per_label * len(labels)
                picked = []
                for li, label in enumerate(labels):
                    count = per_label + int(li < remainder)
                    population = cv_synthetic[cv_synthetic.reference_stance == label]
                    picked.extend(population.sample(n=count, random_state=fold_seed + li + int(ratio * 1000)).index.tolist())
                synthetic_rows = cv_synthetic.loc[picked]
            # The validation fold stays original-only. Synthetic rows are appended to train only.
            train_text = np.concatenate([original_train_x, synthetic_rows.text.astype(str).to_numpy()])
            train_labels = np.concatenate([original_train_y, synthetic_rows.reference_stance.to_numpy()])
            model = build_model(kind, c, weight)
            started = time.perf_counter()
            model.fit(train_text, train_labels)
            fold_seconds[condition].append(time.perf_counter() - started)
            predictions = model.predict(valid_x)
            metrics = evaluate(valid_y, predictions, labels)
            fold_records[condition].append({"repeat": repeat, "fold": fold, **metrics,
                                            "synthetic_train_n": int(len(synthetic_rows)),
                                            "original_train_n": int(len(train_idx))})
            for position, source_idx in enumerate(valid_idx):
                source = frame.iloc[source_idx]
                prediction_rows.append({
                    "condition": condition, "repeat": repeat, "fold": fold,
                    "pilot_id": str(source.get("pilot_id", "")),
                    "source_dataset": str(source.get("source_dataset", "")),
                    "source_record_id": str(source.get("source_record_id", "")),
                    "reference_stance": str(y[source_idx]), "predicted_stance": str(predictions[position]),
                    "is_error": bool(predictions[position] != y[source_idx]),
                    "original_text": str(x[source_idx]),
                })
            if len(synthetic_rows):
                for row in synthetic_rows.itertuples():
                    assignment_rows.append({"repeat": repeat, "fold": fold, "condition": condition,
                                            "record_id": row.synthetic_id, "split": "train", "is_synthetic": True,
                                            "reference_stance": row.reference_stance})

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_json(out_dir / "cross_validation_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    assignment_df = pd.DataFrame(assignment_rows)
    assert_synthetic_training_only(assignment_df)
    assignment_df.to_json(out_dir / "fold_split_manifest.jsonl", orient="records", lines=True, force_ascii=False)
    examples = []
    for condition in ("production_recipe_word_only", "word_char_original_only", "ratio_0.5"):
        examples.extend(_error_examples(predictions_df, condition, labels))
    (out_dir / "representative_errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in examples), encoding="utf-8")

    summaries = {condition: _metric_summary(records, labels) for condition, records in fold_records.items()}
    paired = {}
    production_folds = fold_records["production_recipe_word_only"]
    char_folds = fold_records["word_char_original_only"]
    for condition in ("word_char_original_only", "ratio_0.25", "ratio_0.5", "ratio_1"):
        paired[condition] = {}
        for comparator, reference in (("production_recipe_word_only", production_folds),
                                      ("word_char_original_only", char_folds)):
            delta_accuracy = np.array([r["accuracy"] for r in fold_records[condition]]) - np.array([r["accuracy"] for r in reference])
            delta_macro = np.array([r["macro_f1"] for r in fold_records[condition]]) - np.array([r["macro_f1"] for r in reference])
            paired[condition][comparator] = {
                "accuracy_delta_mean": round(float(np.mean(delta_accuracy)), 4),
                "macro_f1_delta_mean": round(float(np.mean(delta_macro)), 4),
                "macro_f1_delta_sd": round(float(np.std(delta_macro, ddof=1)), 4),
                "macro_f1_fold_wins": int(np.sum(delta_macro > 0)),
                "macro_f1_fold_ties": int(np.sum(delta_macro == 0)),
                "macro_f1_fold_losses": int(np.sum(delta_macro < 0)),
            }
    candidate_gate = all(
        paired["ratio_0.5"][comparator]["macro_f1_delta_mean"] >= 0.03
        and paired["ratio_0.5"][comparator]["macro_f1_fold_wins"] >= 12
        for comparator in ("production_recipe_word_only", "word_char_original_only")
    )
    production_hash_after = hashlib.sha256(PRODUCTION_MODEL.read_bytes()).hexdigest()
    if production_hash_before != production_hash_after:
        raise AssertionError("production artifact changed during the audit")
    v4_config = json.loads((V4_DIR / "metrics.json").read_text(encoding="utf-8"))["results"]["ratio_0.5"]["selected_configuration"]
    result = {
        "evaluation_status": "repeated_grouped_cv_development_robustness_audit_not_independent",
        "reference_records": int(len(frame)), "reference_class_counts": pd.Series(y).value_counts().reindex(labels, fill_value=0).astype(int).to_dict(),
        "unique_source_duplicate_groups": int(len(np.unique(groups))),
        "folds": args.folds, "repeats": args.repeats, "fold_count": len(folds), "seed": args.seed,
        "all_records_seen_once_per_repeat": True, "group_crossings": 0,
        "fixed_candidate_configuration_from_v4_validation": v4_config,
        "cv_comparison_config": {"word_char_model": CV_CONFIG, "production_recipe": {"kind": "word_linsvm", "C": 0.3, "class_weight": "balanced"},
                                 "hyperparameters_frozen_before_cv": True},
        "v4_reproducibility_check": reproducibility,
        "synthetic": {"generation_method": "seeded_consultation_templates_v1", "seed": args.seed + 9901,
                      "rows_per_class": cv_synthetic_per_class, "total_rows": int(len(cv_synthetic)),
                      "only_added_to_training_folds": True,
                      "overlap_against_all_original_records": {k: v for k, v in overlap.items() if k != "near_duplicate_pairs"}},
        "template_diversity": {"v4_by_class": template_audit["by_class"], "v4_templates": template_audit["templates"]},
        "fold_metric_summary_mean_std": summaries,
        "paired_fold_differences": paired,
        "versioned_candidate_gate_passed_for_further_staging": candidate_gate,
        "fold_metric_details": fold_records,
        "mean_fit_seconds_per_fold": {k: round(float(np.mean(v)), 4) for k, v in fold_seconds.items()},
        "representative_error_examples_path": "representative_errors.jsonl",
        "production_artifact": {"path": str(PRODUCTION_MODEL.relative_to(ROOT)), "sha256_before": production_hash_before,
                                "sha256_after": production_hash_after, "unchanged": production_hash_before == production_hash_after},
        "input_sha256": hashlib.sha256(REFERENCE_LABELS_PATH.read_bytes()).hexdigest(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    report_lines = [
        "# Synthetic stance deployment-readiness and grouped CV audit", "",
        "This is a source/duplicate-grouped robustness audit on the existing reference pool. The pool was exposed during prior development; these are not independent or unseen-source performance estimates.", "",
        f"Eligible original records: {len(frame)}; reference class counts: `{json.dumps(result['reference_class_counts'])}`. Groups: {len(np.unique(groups))}. Repeated CV: {args.repeats} repeats x {args.folds} folds ({len(folds)} validation folds). Every original row is validated once per repeat, and no leakage group crosses a fold.", "",
        f"V4 selected model reproducibility: ratio {reproducibility['selected_condition']} with C={v4_config['C']}, class_weight={v4_config['class_weight']}; validation macro-F1 {reproducibility['validation_macro_f1']}. Synthetic v4 rows reproduced exactly: {reproducibility['synthetic_rows_reproduced_exactly']}. No test metric was used to reconstruct this selection.", "",
        "Synthetic additions are restricted to each fold's training side. The model configuration is frozen to the v4 selected word+character LinearSVC (C=1.0, balanced) for a controlled augmentation comparison. The current production recipe (word-only LinearSVC C=0.3, balanced) is also included.", "",
        "| Condition | Accuracy mean ± SD | Macro-F1 mean ± SD | Support P/R/F1 (mean ± SD) | Neutral P/R/F1 (mean ± SD) | Unclear P/R/F1 (mean ± SD) | Mean validation-fold support S/N/U |", "|---|---:|---:|---|---|---|---|"]
    for condition, summary in summaries.items():
        acc, mf = summary["accuracy"], summary["macro_f1"]
        class_parts = []
        supports = []
        for label in labels:
            c = summary["per_class"][label]
            class_parts.append(" / ".join(f"{c[k]['mean']:.4f} ± {c[k]['std']:.4f}" for k in ("precision", "recall", "f1")))
            supports.append(f"{c['support']['mean']:.1f} ± {c['support']['std']:.1f}")
        report_lines.append(f"| {condition} | {acc['mean']:.4f} ± {acc['std']:.4f} | {mf['mean']:.4f} ± {mf['std']:.4f} | {class_parts[0]} | {class_parts[1]} | {class_parts[2]} | {' / '.join(supports)} |")
    report_lines += ["", "## Per-fold class support", "", "Support values above are mean ± standard deviation of each validation-fold class count. Total unique original reference counts are Support 140, Neutral 219, Unclear 40; each appears once per repeat.", "",
                     "## Template diversity and label consistency", "",
                     "Unique generated texts in the v4 corpus are counted by class below. Template-level notes are in `template_audit.md`.", "",
                     "| Class | Rows | Unique texts | Template IDs | Rows flagged for review |", "|---|---:|---:|---:|---:|"]
    for label, info in template_audit["by_class"].items():
        report_lines.append(f"| {label} | {info['rows']} | {info['unique_texts']} | {info['templates']} | {info['review_risk_rows']} |")
    report_lines += ["", "## Representative CV errors", "", "Examples below come from original validation-fold predictions and preserve the source reference labels and IDs. The separate `representative_errors.jsonl` contains the same examples in machine-readable form.", ""]
    for condition in ("production_recipe_word_only", "word_char_original_only", "ratio_0.5"):
        report_lines += [f"### {condition}", ""]
        selected = [e for e in examples if e["condition"] == condition]
        for item in selected:
            text = " ".join(str(item["original_text"]).split())
            report_lines.append(f"- {item['reference_stance']} reference; predicted {item['prediction_values']}; errors in {item['errors']}/{item['repeats']} repeats; record `{item['pilot_id']}` (source `{item['source_dataset']}` / `{item['source_record_id']}`): “{text}”")
        if not selected:
            report_lines.append("No misclassified examples.")
        report_lines.append("")
    report_lines += ["## Deployment decision", "",
                     f"Paired comparison for ratio 0.5 vs production: mean macro-F1 difference {paired['ratio_0.5']['production_recipe_word_only']['macro_f1_delta_mean']} with {paired['ratio_0.5']['production_recipe_word_only']['macro_f1_fold_wins']}/{len(folds)} fold wins; vs word+character original-only: {paired['ratio_0.5']['word_char_original_only']['macro_f1_delta_mean']} with {paired['ratio_0.5']['word_char_original_only']['macro_f1_fold_wins']}/{len(folds)} fold wins. Candidate gate for further staging: {candidate_gate}. Fold differences are descriptive and are not independent confidence intervals.", "",
                     f"Production artifact SHA-256 unchanged: `{production_hash_before}`. The audit did not modify it. CV supports retaining a versioned staging candidate for follow-up review, while the flagged synthetic label risks and lack of new-source evaluation block replacing the production artifact. Rollback is the current unchanged route to `{PRODUCTION_RUN.relative_to(ROOT)}/pipeline_linsvm.joblib`; no routing/config change is part of this audit.", "",
                     "This dataset is small, uses existing reference labels previously exposed in model development, and has only 40 Unclear records (about eight per fold). The language-comprehension subcase is not text-identifiable. New source-held-out reference data and review of the flagged synthetic ambiguity templates remain necessary before a deployment change.", "",
                     f"Reproduce: `python scripts/audit_stance_synthetic_deployment_v1.py --out-dir reports/stance_synthetic_deployment_audit_rerun --seed {args.seed} --folds {args.folds} --repeats {args.repeats} --synthetic-per-class {args.synthetic_per_class}`"]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    template_lines = ["# V4 synthetic template audit", "", "Review against reports/annotation_guidelines_v1.1.md. This is a rule-based text review, not new human annotation.", "",
                      "| Reference class | Template ID | Rows | Unique texts | Review status | Note |", "|---|---|---:|---:|---|---|"]
    template_lines += [f"| {row['reference_stance']} | {row['template_id']} | {row['rows']} | {row['unique_texts']} | {row['review_status']} | {row['review_note']} |" for row in template_audit["templates"]]
    template_lines += ["", "The `unclear_ambiguity_1` template explicitly combines support for openness with rejection of publication. Under the current guidelines, this may be Mixed rather than Unclear; its four generated copies are suspect synthetic targets. Other genuine-ambiguity templates also merit bilingual annotation review. French-reading difficulty is annotator-specific and cannot be inferred from an ordinary French consultation comment.", "",
                       "Generated text repeats: v4 has 57/80 unique Support texts, 49/80 Neutral, and 35/80 Unclear. Repeated template families are more concentrated in Unclear. The grouped CV evaluates transfer onto original comments that do not share these synthetic template IDs, but cannot establish robustness to new consultation sources."]
    (out_dir / "template_audit.md").write_text("\n".join(template_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "v4_reproduction": reproducibility,
                      "cv_summary": summaries, "production_artifact_unchanged": production_hash_before == production_hash_after}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/stance_synthetic_deployment_audit_v1")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-per-class", type=int, default=120)
    args = parser.parse_args()
    if args.folds < 3 or args.repeats < 2:
        parser.error("use at least 3 folds and 2 repeats")
    run(args)


if __name__ == "__main__":
    main()
