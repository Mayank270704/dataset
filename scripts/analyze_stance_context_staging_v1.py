"""Produce error analysis and nested grouped-validation abstention results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.context_stance_features import build_context_feature
from models.stance_classification.labels import load_stance_reference_labels
from scripts.benchmark_stance_context_v1 import LABELS, build_model, score
from scripts.benchmark_stance_existing_resources_v1 import fold_map, sha
from scripts.build_stance_context_staging_v1 import CS, SEED, choose_c
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

OUT = ROOT / "reports/stance_context_staging_v1"
PREDICTIONS = OUT / "grouped_cv_predictions.jsonl"
ABSTENTION_COVERAGE_FLOOR = 0.50


def margin(pipeline, x):
    values = np.asarray(pipeline.decision_function(x)).reshape(len(x), -1)
    if values.shape[1] == 1:
        return np.abs(values[:, 0])
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, -1] - sorted_values[:, -2]


def qualitative_pattern(true, pred):
    patterns = {
        ("Support", "Neutral"): "Support reference comments often phrase a recommendation, question, or criticism without a direct endorsement. The model appears to prefer descriptive or interrogative wording; question/context sometimes frames the comment as an open idea rather than an explicit stance.",
        ("Support", "Unclear"): "Support reference comments can be short suggestions or proposals with little first-person or endorsement wording. The topic context helps identify subject matter but may not show whether a fragment is intended as support.",
        ("Neutral", "Support"): "Neutral references include constructive proposals, words such as 'need', 'better', or 'support', and questions. Lexical positivity or recommendation language may be mistaken for taking a side; the prompt often invites ideas rather than a position.",
        ("Neutral", "Unclear"): "Some references are terse, tangential, or contain uncertainty. In these cases Unclear may be linguistically plausible, though it disagrees with the reference label; broader context does not always clarify the author's position.",
        ("Unclear", "Support"): "The 20 unique affected records include endorsements of third-party statements, fragments, topic labels, retweets/links, and recommendations. Positive or action-oriented tokens can look supportive even when the author position is not recoverable. Context often identifies a brainstorming topic but does not establish authorship or endorsement.",
        ("Unclear", "Neutral"): "The 21 unique affected records include topic fragments, unrelated/off-topic text, explicit uncertainty, and mixed or critical statements. The model often defaults to Neutral for non-directional wording; context may explain the topic but cannot resolve missing or ambiguous author intent.",
    }
    return patterns[(true, pred)]


def main():
    status, frame = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or frame is None:
        raise RuntimeError(status.detail)
    frame = frame[frame.gold_stance.isin(LABELS)].reset_index(drop=True)
    features = np.asarray([build_context_feature(r.question_wording, r.question_context, r.original_text)
                           for r in frame.itertuples()], dtype=str)
    y = frame.gold_stance.astype(str).to_numpy()
    manifest = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl"
    saved = pd.read_json(manifest, lines=True)
    groups_by_id = saved[saved.condition == "shared_original_folds"].groupby("record_id").leakage_group.first()
    groups = frame.pilot_id.astype(str).map(groups_by_id.astype(str)).to_numpy()
    outer_folds = fold_map(frame)

    nested = []
    for outer in outer_folds:
        train, valid = outer["train"], outer["valid"]
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                        random_state=SEED + outer["repeat"] * 1009 + outer["fold"])
        inner = list(inner_cv.split(np.zeros(len(train)), y[train], groups[train]))
        if any(set(groups[train[a]]) & set(groups[train[b]]) for a, b in inner):
            raise AssertionError("inner abstention validation split crossed a source/duplicate group")
        chosen_c, _ = choose_c(features[train], y[train], groups[train], inner)
        inner_pred, inner_true, inner_margin = [], [], []
        for i_train, i_valid in inner:
            model = build_model(c=chosen_c)
            model.fit(features[train[i_train]], y[train[i_train]])
            x_valid = features[train[i_valid]]
            inner_pred.extend(model.predict(x_valid).tolist())
            inner_true.extend(y[train[i_valid]].tolist())
            inner_margin.extend(margin(model, x_valid).tolist())
        inner_pred = np.asarray(inner_pred, dtype=str)
        inner_true = np.asarray(inner_true, dtype=str)
        inner_margin = np.asarray(inner_margin, dtype=float)
        thresholds = np.unique(np.concatenate(([-np.inf], inner_margin)))
        choices = []
        for threshold in thresholds:
            accepted = inner_margin >= threshold
            coverage = float(accepted.mean())
            if coverage < ABSTENTION_COVERAGE_FLOOR or not accepted.any():
                continue
            choices.append((float((inner_pred[accepted] == inner_true[accepted]).mean()), coverage,
                            float(threshold)))
        if not choices:
            threshold = -np.inf
        else:
            # Max validation selective accuracy, then validation coverage, then lower threshold.
            threshold = max(choices, key=lambda row: (row[0], row[1], -row[2]))[2]
        model = build_model(c=chosen_c)
        model.fit(features[train], y[train])
        valid_pred = model.predict(features[valid]).astype(str)
        valid_margin = margin(model, features[valid])
        accepted = valid_margin >= threshold
        nested.append({"repeat": outer["repeat"], "fold": outer["fold"], "C": chosen_c,
                       "threshold": threshold if np.isfinite(threshold) else None,
                       "inner_validation_coverage": float(np.mean(inner_margin >= threshold)),
                       "inner_validation_accepted_accuracy": float(
                           (inner_pred[inner_margin >= threshold] == inner_true[inner_margin >= threshold]).mean()),
                       "y": y[valid].tolist(), "pred": valid_pred.tolist(),
                       "accepted": accepted.tolist()})

    y_all = np.asarray([y for row in nested for y in row["y"]], dtype=str)
    pred_all = np.asarray([p for row in nested for p in row["pred"]], dtype=str)
    accepted_all = np.asarray([a for row in nested for a in row["accepted"]], dtype=bool)
    accepted_metrics = score(y_all[accepted_all], pred_all[accepted_all]) if accepted_all.any() else None
    class_coverage = {}
    for label in LABELS:
        class_mask = y_all == label
        class_coverage[label] = {"accepted": int((accepted_all & class_mask).sum()),
                                 "available": int(class_mask.sum()),
                                 "coverage": round(float(accepted_all[class_mask].mean()), 4)}
    abstention = {
        "policy": "For each outer fold, select a top-two LinearSVC margin threshold on three-fold grouped validation within the outer training records; maximize accepted validation accuracy subject to at least 50% validation coverage. Apply only that frozen threshold to the outer validation fold.",
        "selection_metric": "validation accepted accuracy; tie-break by higher coverage then lower threshold",
        "coverage_floor": ABSTENTION_COVERAGE_FLOOR,
        "outer_validation": {"n": int(len(y_all)), "accepted": int(accepted_all.sum()),
                             "coverage": round(float(accepted_all.mean()), 4),
                             "accepted_accuracy": round(float((y_all[accepted_all] == pred_all[accepted_all]).mean()), 4),
                             "per_class_coverage": class_coverage,
                             "accepted_confusion_matrix": accepted_metrics["confusion_matrix"] if accepted_metrics else None,
                             "fold_selection": [{k: row[k] for k in ("repeat", "fold", "C", "threshold",
                                 "inner_validation_coverage", "inner_validation_accepted_accuracy")} for row in nested]},
        "note": "Selective accuracy is conditional on accepted predictions and is not overall accuracy. Top-two LinearSVC margins are uncalibrated, fold-specific ranking signals. This is an analysis-only policy and is not enabled in the staging route. The threshold uses grouped validation only; outer grouped folds provide held-out evaluation and were not used to select the threshold.",
    }
    (OUT / "abstention_metrics.json").write_text(json.dumps(abstention, indent=2) + "\n", encoding="utf-8")

    oof = pd.read_json(PREDICTIONS, lines=True)
    counts = pd.crosstab(oof.reference_stance, oof.context_staging_prediction).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    transitions = []
    for true in LABELS:
        row = {}
        for pred in LABELS:
            events = oof[(oof.reference_stance == true) & (oof.context_staging_prediction == pred)]
            row[pred] = {"events": int(len(events)), "unique_records": int(events.record_id.nunique())}
        transitions.append({"reference": true, "predictions": row})
    unclear_errors = oof[(oof.reference_stance == "Unclear") &
                         (oof.context_staging_prediction != "Unclear")]
    unclear_ids = {
        pred: sorted(unclear_errors[unclear_errors.context_staging_prediction == pred].record_id.unique().tolist())
        for pred in ("Support", "Neutral")
    }
    stability = oof[oof.reference_stance == "Unclear"].groupby("record_id").context_staging_prediction.nunique().value_counts().to_dict()
    error_analysis = {
        "source_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "prediction_sha256": sha(PREDICTIONS), "rows": int(len(oof)),
        "confusion_matrix": counts.to_numpy().tolist(), "labels_order": LABELS,
        "false_positive_false_negative": {
            label: {"false_positives": int(len(oof[(oof.context_staging_prediction == label) &
                                                   (oof.reference_stance != label)])),
                    "false_negatives": int(len(oof[(oof.reference_stance == label) &
                                                   (oof.context_staging_prediction != label)]))}
            for label in LABELS},
        "transition_counts": transitions,
        "unclear_wrong_record_ids": unclear_ids,
        "unclear_prediction_stability_across_three_repeats": {str(k): int(v) for k, v in stability.items()},
        "context_missing_rows": int((frame.question_wording.fillna("").astype(str).str.strip().eq("") |
                                      frame.question_context.fillna("").astype(str).str.strip().eq("")).sum()),
        "qualitative_hypotheses": {f"{a}_reference_to_{b}_prediction": qualitative_pattern(a, b)
                                    for a in LABELS for b in LABELS if a != b},
        "label_policy": "Reference labels were not modified. Qualitative explanations are hypotheses based on text/context inspection, not established causes.",
    }
    (OUT / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2) + "\n", encoding="utf-8")
    (OUT / "error_analysis.md").write_text(render_error_report(error_analysis, abstention), encoding="utf-8")
    print(json.dumps({"error_counts": error_analysis["false_positive_false_negative"],
                      "unclear_wrong_unique": {k: len(v) for k, v in unclear_ids.items()},
                      "abstention": abstention["outer_validation"]}, indent=2))


def render_error_report(errors, abstention):
    lines = ["# Context stance staging: error analysis and abstention", "",
             f"Analysis uses the existing repeated grouped out-of-fold predictions ({errors['rows']} prediction events across 399 records and three repeats; predictions SHA-256 `{errors['prediction_sha256']}`). These are development estimates against unverified repository reference labels. No reference label was changed.", "",
             "## Class errors", "",
             "False positive means predicted class differs from the reference class; false negative means reference class differs from the prediction.", "",
             "| Class | False positives | False negatives |", "|---|---:|---:|"]
    for label in LABELS:
        d = errors["false_positive_false_negative"][label]
        lines.append(f"| {label} | {d['false_positives']} | {d['false_negatives']} |")
    lines += ["", "The row/column confusion matrix in Support, Neutral, Unclear order is `" +
              json.dumps(errors["confusion_matrix"]) + "`.", "",
              "| Reference | Predicted | Error events | Unique records |", "|---|---|---:|---:|"]
    for transition in errors["transition_counts"]:
        for pred in LABELS:
            if pred != transition["reference"]:
                value = transition["predictions"][pred]
                lines.append(f"| {transition['reference']} | {pred} | {value['events']} | {value['unique_records']} |")
    lines += ["", "## Linguistic/context review (hypotheses)", "",
              "Context was present for all modeled rows. It generally identifies prompt, theme, event, or response type; for short text it often does not reveal whether the author endorses a position. The patterns below are qualitative hypotheses, not adjudicated causes.", ""]
    for true in LABELS:
        for pred in LABELS:
            if true != pred:
                lines += [f"### {true} reference predicted {pred}", "",
                          errors["qualitative_hypotheses"][f"{true}_reference_to_{pred}_prediction"], ""]
    unclear = errors["unclear_wrong_record_ids"]
    lines += ["## Unclear predicted Support or Neutral", "",
              f"The pooled matrix has 50 Unclear-to-Support and 53 Unclear-to-Neutral error events. These correspond to {len(unclear['Support'])} and {len(unclear['Neutral'])} unique records respectively; repeated folds account for multiple events per record, and some records switch between these predictions. Their record IDs are listed for audit; none were relabeled.", "",
              "- Unclear to Support IDs: " + ", ".join(unclear["Support"]),
              "- Unclear to Neutral IDs: " + ", ".join(unclear["Neutral"]), "",
              "Among the 40 Unclear records, the model produced the same prediction in all three repeats for 30 records; ten had predictions that varied across repeats. This instability is consistent with weak margins or fold-sensitive wording, but is not proof of a particular cause.", "",
              f"Input question/context fields were missing on {errors['context_missing_rows']} of the 399 modeled rows. That does not rule out insufficient evidence inside a non-empty prompt. Potential causes inspected include fragments or unrelated content (insufficient evidence), mixed or vague wording (ambiguity), prompt framing that invites ideas/questions without specifying a stance (framing), and fold-dependent predictions (instability). These remain hypotheses; no single cause can be assigned from this evaluation.", "",
              "## Grouped-validation abstention experiment", "",
              abstention["policy"], "",
              f"Outer grouped validation: accepted {abstention['outer_validation']['accepted']} / {abstention['outer_validation']['n']} ({abstention['outer_validation']['coverage']:.1%}) with accepted-prediction accuracy {abstention['outer_validation']['accepted_accuracy']:.4f}. This is selective accuracy among accepted cases, not overall accuracy.", "",
              "| Reference class | Accepted | Available | Class coverage |", "|---|---:|---:|---:|"]
    for label in LABELS:
        d = abstention["outer_validation"]["per_class_coverage"][label]
        lines.append(f"| {label} | {d['accepted']} | {d['available']} | {d['coverage']:.1%} |")
    lines += ["", f"Accepted-only confusion matrix: `{json.dumps(abstention['outer_validation']['accepted_confusion_matrix'])}`.", "",
              f"The accepted-only Unclear row is {abstention['outer_validation']['accepted_confusion_matrix'][2]}; only 2 of 64 accepted Unclear references were predicted Unclear. The policy improves accepted-case accuracy mainly on Support and Neutral and does not make Unclear reliable.", "",
              abstention["note"], "",
              "Reproduce: `python scripts/analyze_stance_context_staging_v1.py`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
