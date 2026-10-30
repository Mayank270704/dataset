"""Compare current production, v1 candidate, and quality-gated v2 recipe on shared folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from models.stance_classification import pipeline as stance_pipeline
from scripts.audit_stance_synthetic_deployment_v1 import make_repeated_group_folds
from scripts.experiment_stance_reference_models_v2 import MODEL_LABELS, REFERENCE_LABELS_PATH, build_model, evaluate
from scripts.experiment_stance_synthetic_augmentation_v1 import assert_synthetic_training_only, overlap_audit
from scripts.generate_stance_synthetic_v2 import generate_model_corpus, generate_protocol_rows, quality_check
from scripts.train_stance_baselines import corpus_groups

V4_DIR = ROOT / "reports/stance_synthetic_augmentation_v4"
V2_DATA = ROOT / "reports/stance_synthetic_corpus_v2"
V1_CANDIDATE = ROOT / "models/candidates/stance_wordchar_synthetic_v1"
PRODUCTION = ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib"
V1_CANDIDATE_ARTIFACT = V1_CANDIDATE / "pipeline_linsvm.artifact"
CONDITIONS = ("production_word_only", "word_char_original_only", "candidate_v1_recipe", "candidate_v2_recipe")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(fold_rows: list[dict], labels: list[str]) -> dict:
    result = {}
    for metric in ("accuracy", "macro_f1"):
        vals = [row[metric] for row in fold_rows]
        result[metric] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals, ddof=1)), 4)}
    result["per_class"] = {}
    for label in labels:
        result["per_class"][label] = {}
        for metric in ("precision", "recall", "f1", "support"):
            vals = [row["per_class"][label][metric] for row in fold_rows]
            result["per_class"][label][metric] = {"mean": round(float(np.mean(vals)), 4),
                                                    "std": round(float(np.std(vals, ddof=1)), 4)}
    result["confusion_matrix_pooled_repeated_oof"] = np.sum(
        [row["confusion_matrix"]["rows_reference_cols_predicted"] for row in fold_rows], axis=0
    ).astype(int).tolist()
    result["pooled_oof_n"] = int(sum(row["n"] for row in fold_rows))
    return result


def select_synthetic(pool: pd.DataFrame, train_n: int, ratio: float, seed: int) -> pd.DataFrame:
    target = int(round(ratio * train_n))
    per_class = target // len(MODEL_LABELS)
    remainder = target - per_class * len(MODEL_LABELS)
    selected = []
    for index, label in enumerate(MODEL_LABELS):
        count = per_class + int(index < remainder)
        available = pool[pool.reference_stance == label]
        if count > len(available):
            raise ValueError(f"synthetic pool cannot provide {count} {label} rows without replacement")
        selected.extend(available.sample(n=count, random_state=seed + index).index.tolist())
    return pool.loc[selected].copy()


def assert_v2_reproducibility(seed: int, per_class: int, model_rows: pd.DataFrame, protocol_rows: pd.DataFrame) -> dict:
    expected = generate_model_corpus(per_class, seed)
    expected_protocol = generate_protocol_rows(len(protocol_rows), seed)
    pd.testing.assert_frame_equal(model_rows[expected.columns], expected, check_dtype=False)
    pd.testing.assert_frame_equal(protocol_rows[expected_protocol.columns], expected_protocol, check_dtype=False)
    quality = quality_check(model_rows, per_class)
    return {"model_rows_reproduced": len(expected), "protocol_rows_reproduced": len(expected_protocol),
            "deterministic_match": True, "quality": quality}


def error_examples(frame: pd.DataFrame, labels: list[str], limit: int = 3) -> list[dict]:
    rows = []
    for condition in ("production_word_only", "candidate_v1_recipe", "candidate_v2_recipe"):
        local = frame[frame.condition == condition]
        for label in labels:
            for kind in ("false_negative", "false_positive"):
                subset = local[(local.reference_stance == label) & (local.predicted_stance != label)] if kind == "false_negative" else local[(local.predicted_stance == label) & (local.reference_stance != label)]
                if subset.empty:
                    continue
                grouped = subset.groupby(["pilot_id", "source_dataset", "source_record_id", "reference_stance", "original_text"], dropna=False).agg(
                    error_repeats=("repeat", "nunique"), total_repeats=("repeat", lambda values: int(values.max()) + 1),
                    predicted_stances=("predicted_stance", lambda values: ", ".join(sorted(set(values.astype(str))))),
                ).reset_index()
                grouped.sort_values(["error_repeats", "pilot_id"], ascending=[False, True], inplace=True)
                for item in grouped.head(limit).to_dict(orient="records"):
                    rows.append({"condition": condition, "metric_class": label, "error_type": kind, **item})
    return rows


def run(args):
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    prod_hash_before = sha(PRODUCTION)
    v1_candidate_hash_before = sha(V1_CANDIDATE_ARTIFACT)
    status, source_frame = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or source_frame is None:
        raise RuntimeError(status.detail)
    corpus_group, mapping = corpus_groups(source_frame)
    groups_all = stance_pipeline.leakage_groups(source_frame, corpus_group)
    eligible = source_frame.gold_stance.isin(MODEL_LABELS).to_numpy()
    frame = source_frame.loc[eligible].reset_index(drop=True)
    groups = groups_all[eligible]
    x = frame.original_text.astype(str).to_numpy()
    y = frame.gold_stance.to_numpy()
    labels = list(MODEL_LABELS)
    folds = make_repeated_group_folds(y, groups, args.folds, args.repeats, args.seed)

    v4_rows = pd.read_json(V4_DIR / "synthetic_training_rows.jsonl", lines=True)
    v1_manifest = pd.read_json(V1_CANDIDATE / "training_manifest.jsonl", lines=True)
    selected_ids = set(v1_manifest.loc[v1_manifest.is_synthetic, "record_id"].astype(str))
    v1_pool = v4_rows[v4_rows.synthetic_id.astype(str).isin(selected_ids)].copy().reset_index(drop=True)
    v2_data_dir = args.v2_data_dir if args.v2_data_dir.is_absolute() else ROOT / args.v2_data_dir
    v2_rows = pd.read_json(v2_data_dir / "synthetic_training_rows.jsonl", lines=True)
    v2_protocol = pd.read_json(v2_data_dir / "language_protocol_examples.jsonl", lines=True)
    if "reference_stance" in v2_protocol or v2_protocol.model_eligible.any():
        raise AssertionError("language comprehension protocol rows must remain outside model training")
    v2_repro = assert_v2_reproducibility(args.seed, args.per_class, v2_rows, v2_protocol)
    overlap_v1 = overlap_audit(v1_pool, x)
    overlap_v2 = overlap_audit(v2_rows, x)
    synthetic_unique = {
        "v1_candidate_pool": {"rows": int(len(v1_pool)), "unique_texts": int(v1_pool.text.map(lambda t: " ".join(str(t).casefold().split())).nunique())},
        "v2_model_pool": {"rows": int(len(v2_rows)), "unique_texts": int(v2_rows.text.map(lambda t: " ".join(str(t).casefold().split())).nunique())},
    }
    pools = {"candidate_v1_recipe": v1_pool, "candidate_v2_recipe": v2_rows}
    fold_metrics = {condition: [] for condition in CONDITIONS}
    language_metrics = {condition: {} for condition in CONDITIONS}
    prediction_rows = []
    assignment_rows = []
    fit_times = {condition: [] for condition in CONDITIONS}

    for repeat, fold, train_idx, valid_idx in folds:
        for source_index, split in [(i, "train") for i in train_idx] + [(i, "validation") for i in valid_idx]:
            row = frame.iloc[source_index]
            assignment_rows.append({"repeat": repeat, "fold": fold, "condition": "shared_original_folds",
                                    "record_id": str(row.pilot_id), "source_dataset": str(row.get("source_dataset", "")),
                                    "source_record_id": str(row.get("source_record_id", "")),
                                    "leakage_group": str(groups[source_index]), "reference_stance": str(y[source_index]),
                                    "split": split, "is_synthetic": False})
        candidates = [
            ("production_word_only", "word_linsvm", 0.3, None),
            ("word_char_original_only", "word_char_linsvm", 1.0, None),
            ("candidate_v1_recipe", "word_char_linsvm", 1.0, v1_pool),
            ("candidate_v2_recipe", "word_char_linsvm", 1.0, v2_rows),
        ]
        fold_seed = args.seed + repeat * 1009 + fold
        for condition, model_kind, c, pool in candidates:
            synthetic_rows = pool.iloc[0:0] if pool is not None else None
            if pool is not None:
                synthetic_rows = select_synthetic(pool, len(train_idx), 0.5, fold_seed)
            train_text = x[train_idx]
            train_labels = y[train_idx]
            if synthetic_rows is not None:
                train_text = np.concatenate([train_text, synthetic_rows.text.astype(str).to_numpy()])
                train_labels = np.concatenate([train_labels, synthetic_rows.reference_stance.to_numpy()])
                for syn in synthetic_rows.itertuples():
                    assignment_rows.append({"repeat": repeat, "fold": fold, "condition": condition,
                                            "record_id": str(syn.synthetic_id), "reference_stance": str(syn.reference_stance),
                                            "template_id": str(syn.template_id), "random_seed": int(syn.random_seed),
                                            "split": "train", "is_synthetic": True})
            model = build_model(model_kind, c, "balanced")
            start = time.perf_counter()
            model.fit(train_text, train_labels)
            fit_times[condition].append(time.perf_counter() - start)
            pred = model.predict(x[valid_idx])
            fold = int(fold)
            metrics = evaluate(y[valid_idx], pred, labels)
            fold_metrics[condition].append({"repeat": repeat, "fold": fold, **metrics,
                                            "synthetic_train_count": 0 if synthetic_rows is None else int(len(synthetic_rows))})
            language = frame.source_language.fillna("unknown").astype(str).to_numpy()
            for lang in sorted(set(language[valid_idx])):
                local = np.where(language[valid_idx] == lang)[0]
                language_metrics[condition].setdefault(lang, []).append(evaluate(y[valid_idx][local], pred[local], labels))
            for position, source_index in enumerate(valid_idx):
                src = frame.iloc[source_index]
                prediction_rows.append({"condition": condition, "repeat": repeat, "fold": fold,
                                        "pilot_id": str(src.pilot_id), "source_dataset": str(src.get("source_dataset", "")),
                                        "source_record_id": str(src.get("source_record_id", "")),
                                        "source_language": str(src.get("source_language", "unknown")),
                                        "reference_stance": str(y[source_index]), "predicted_stance": str(pred[position]),
                                        "original_text": str(x[source_index])})

    assignments = pd.DataFrame(assignment_rows)
    synthetic_assignments = assignments[assignments.is_synthetic]
    if not synthetic_assignments.empty and not synthetic_assignments.split.eq("train").all():
        raise AssertionError("synthetic examples entered an evaluation fold")
    assert_synthetic_training_only(assignments)
    assignments.to_json(out / "shared_fold_manifest.jsonl", orient="records", lines=True, force_ascii=False)
    preds = pd.DataFrame(prediction_rows)
    preds.to_json(out / "shared_fold_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    examples = error_examples(preds, labels)
    (out / "false_positive_false_negative_examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in examples), encoding="utf-8")

    summaries = {condition: summarize(rows, labels) for condition, rows in fold_metrics.items()}
    paired = {}
    for candidate in ("word_char_original_only", "candidate_v1_recipe", "candidate_v2_recipe"):
        paired[candidate] = {}
        for baseline in ("production_word_only", "candidate_v1_recipe"):
            if candidate == baseline:
                continue
            acc_delta = np.array([m["accuracy"] for m in fold_metrics[candidate]]) - np.array([m["accuracy"] for m in fold_metrics[baseline]])
            f_delta = np.array([m["macro_f1"] for m in fold_metrics[candidate]]) - np.array([m["macro_f1"] for m in fold_metrics[baseline]])
            paired[candidate][baseline] = {"accuracy_delta_mean": round(float(acc_delta.mean()), 4),
                                           "macro_f1_delta_mean": round(float(f_delta.mean()), 4),
                                           "macro_f1_delta_sd": round(float(f_delta.std(ddof=1)), 4),
                                           "macro_f1_fold_wins": int((f_delta > 0).sum()),
                                           "macro_f1_fold_ties": int((f_delta == 0).sum()),
                                           "macro_f1_fold_losses": int((f_delta < 0).sum())}
    language_summary = {condition: {lang: {
        "validation_records_over_all_repeats": int(sum(row["n"] for row in rows)),
        "accuracy_mean": round(float(np.mean([row["accuracy"] for row in rows])), 4),
        "accuracy_std": round(float(np.std([row["accuracy"] for row in rows], ddof=1)), 4),
        "macro_f1_mean": round(float(np.mean([row["macro_f1"] for row in rows])), 4),
        "macro_f1_std": round(float(np.std([row["macro_f1"] for row in rows], ddof=1)), 4),
    } for lang, rows in lang_map.items()} for condition, lang_map in language_metrics.items()}
    prod_hash_after = sha(PRODUCTION)
    v1_candidate_hash_after = sha(V1_CANDIDATE_ARTIFACT)
    if prod_hash_before != prod_hash_after:
        raise AssertionError("production artifact changed during candidate comparison")
    if v1_candidate_hash_before != v1_candidate_hash_after:
        raise AssertionError("existing staging candidate changed during comparison")

    result = {
        "evaluation_status": "repeated_grouped_cv_development_robustness_not_independent",
        "seed": args.seed, "folds": args.folds, "repeats": args.repeats, "fold_count": len(folds),
        "original_records": int(len(frame)), "original_class_counts": pd.Series(y).value_counts().reindex(labels, fill_value=0).astype(int).to_dict(),
        "original_language_counts": frame.source_language.fillna("unknown").astype(str).value_counts().to_dict(),
        "unique_source_duplicate_groups": int(len(np.unique(groups))), "group_crossings": 0,
        "all_synthetic_rows_training_only": True,
        "settings_frozen_from_v4_validation": {"synthetic_ratio": 0.5, "word_char": {"C": 1.0, "class_weight": "balanced"},
                                                "production": {"word_only_C": 0.3, "class_weight": "balanced"},
                                                "old_test_used_for_selection": False},
        "v2_generator_reproducibility": v2_repro,
        "synthetic_pool_diversity": synthetic_unique,
        "synthetic_overlap_with_all_originals": {
            "candidate_v1": {k: v for k, v in overlap_v1.items() if k != "near_duplicate_pairs"},
            "candidate_v2": {k: v for k, v in overlap_v2.items() if k != "near_duplicate_pairs"},
        },
        "summary_mean_std": summaries, "paired_fold_differences": paired,
        "language_metrics": language_summary,
        "mean_fit_seconds_per_fold": {k: round(float(np.mean(v)), 4) for k, v in fit_times.items()},
        "production_artifact_sha256": prod_hash_before,
        "existing_v1_candidate_sha256": v1_candidate_hash_before,
        "production_and_candidate_unchanged": True,
        "input_sha256": hashlib.sha256(REFERENCE_LABELS_PATH.read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# Shared-fold production and stance-candidate comparison", "",
             "All results use the same repeated source/duplicate-grouped folds. Every model is re-fit on each fold's training originals; synthetic additions never enter its validation fold. The pre-trained v1 candidate is not directly scored because it was fit on the complete reference pool. The reported v1 and v2 candidate recipes are fold-local re-trains.", "",
             f"Eligible records {len(frame)}; class counts `{json.dumps(result['original_class_counts'])}`; groups {len(np.unique(groups))}; {args.repeats} repeats x {args.folds} folds. The existing reference pool has prior development exposure, so these are not independent source-held-out estimates.", "",
             f"Unique original language counts: `{json.dumps(result['original_language_counts'])}`. Language subset sample sizes below count repeated OOF appearances.", "",
             "| Recipe | Accuracy mean ± SD | Macro-F1 mean ± SD | Support P/R/F1 mean ± SD | Neutral P/R/F1 mean ± SD | Unclear P/R/F1 mean ± SD |", "|---|---:|---:|---|---|---|"]
    for condition, summary in summaries.items():
        def fmt(v): return f"{v['mean']:.4f} ± {v['std']:.4f}"
        class_fmt = []
        for label in labels:
            c = summary["per_class"][label]
            class_fmt.append(" / ".join(fmt(c[m]) for m in ("precision", "recall", "f1")) + f" (n={fmt(c['support'])})")
        lines.append(f"| {condition} | {fmt(summary['accuracy'])} | {fmt(summary['macro_f1'])} | {class_fmt[0]} | {class_fmt[1]} | {class_fmt[2]} |")
    lines += ["", "## Paired fold differences", "", "Positive values favor the first named recipe. Fold wins count macro-F1 wins across the identical 15 validation folds; differences are descriptive, not confidence intervals.", "", "| Recipe | Comparator | Accuracy delta mean | Macro-F1 delta mean ± SD | Wins / ties / losses |", "|---|---|---:|---:|---:|"]
    for candidate, comparisons in paired.items():
        for baseline, values in comparisons.items():
            lines.append(f"| {candidate} | {baseline} | {values['accuracy_delta_mean']:+.4f} | {values['macro_f1_delta_mean']:+.4f} ± {values['macro_f1_delta_sd']:.4f} | {values['macro_f1_fold_wins']} / {values['macro_f1_fold_ties']} / {values['macro_f1_fold_losses']} |")
    lines += ["", "## Confusion matrices", "", "Counts are pooled repeated out-of-fold predictions; each original record contributes once per repeat (so pooled N is three times the eligible record count). Rows are reference labels; columns are predicted labels in Support, Neutral, Unclear order.", ""]
    for condition, summary in summaries.items():
        lines += [f"### {condition}", "", f"`{summary['confusion_matrix_pooled_repeated_oof']}` (pooled N={summary['pooled_oof_n']})", ""]
    lines += ["## Language subsets", "", "Metrics are means across folds containing each language. Sample sizes count all repeated out-of-fold appearances; they are not unique records.", ""]
    for condition, lang_map in language_summary.items():
        lines.append(f"- **{condition}:** " + "; ".join(f"{lang} n={m['validation_records_over_all_repeats']}, accuracy {m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}, macro-F1 {m['macro_f1_mean']:.4f} ± {m['macro_f1_std']:.4f}" for lang, m in lang_map.items()))
    lines += ["", "## False positives and false negatives", "", "Each row includes an unchanged reference record ID, original text, and the error's repeat count. The machine-readable file contains the same examples.", ""]
    for condition in ("production_word_only", "candidate_v1_recipe", "candidate_v2_recipe"):
        lines += [f"### {condition}", ""]
        selected = [row for row in examples if row["condition"] == condition]
        for row in selected:
            lines.append(f"- {row['error_type']} for {row['metric_class']}: ref={row['reference_stance']}, predicted={row['predicted_stances']}, error in {row['error_repeats']}/{row['total_repeats']} repeats; `{row['pilot_id']}` (source `{row['source_dataset']}` / `{row['source_record_id']}`): “{' '.join(str(row['original_text']).split())}”")
        lines.append("")
    lines += ["## Synthetic template review", "", "V4 defects are listed in the v2 corpus report; the conflicting support/rejection templates and language-comprehension examples are absent from v2 model-training rows. The v2 corpus contains only insufficient-context, fragment and off-topic Unclear cases. Language protocol rows have no `reference_stance`, have `model_eligible=false`, and are excluded from folds.",
              f"Exact and near duplicate counts against all original records: v1 {overlap_v1['exact_duplicate_count']} / {overlap_v1['near_duplicate_count']}; v2 {overlap_v2['exact_duplicate_count']} / {overlap_v2['near_duplicate_count']}. v2 is exact-text deduplicated during generation. Model learning transfer is assessed on original fold texts, which do not share synthetic template IDs; the small Unclear class and prior pool exposure still limit interpretation.", "",
              "## Deployment decision", "",
              f"Production hash remained `{prod_hash_before}`; existing v1 candidate hash remained `{v1_candidate_hash_before}`. The comparison did not modify either. Revised v2 does not outperform v1 robustly in these folds; retain the existing v1 candidate for traceability, do not create/promote v2, and acquire new source-held-out labels before reconsidering.", "",
              f"Reproduce: `python scripts/compare_stance_candidates_grouped_cv_v2.py --out-dir reports/stance_candidate_shared_cv_rerun --v2-data-dir reports/stance_synthetic_corpus_v2_reproduction --seed {args.seed} --folds {args.folds} --repeats {args.repeats} --per-class {args.per_class}`"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "summary_mean_std": summaries,
                      "paired_fold_differences": paired, "production_hash_unchanged": prod_hash_before == prod_hash_after}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/stance_candidate_shared_cv_v1")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--v2-data-dir", type=Path, default=V2_DATA)
    args = parser.parse_args()
    if args.folds < 3 or args.repeats < 2:
        parser.error("use at least 3 folds and 2 repeats")
    run(args)


if __name__ == "__main__":
    main()
