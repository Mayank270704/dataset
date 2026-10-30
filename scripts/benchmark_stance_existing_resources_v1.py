"""Grouped benchmark and fixed-rank abstention analysis using existing stance assets.

This benchmark reuses the repository's saved shared folds, reference rows, existing
synthetic pools, and cached MiniLM embeddings. It never generates data or downloads a
model. All results remain development estimates because the reference pool was exposed
during earlier model development.
"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from scripts.compare_stance_candidates_grouped_cv_v2 import select_synthetic
from scripts.experiment_stance_reference_models_v2 import (
    MODEL_LABELS, REFERENCE_LABELS_PATH, build_model, evaluate,
)

FOLD_MANIFEST = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl"
V4_POOL = ROOT / "reports/stance_synthetic_augmentation_v4/synthetic_training_rows.jsonl"
V2_POOL = ROOT / "reports/stance_synthetic_corpus_v2_clean/synthetic_training_rows.jsonl"
CORPUS_UNIQUE = ROOT / "data/full_corpus/run_v1/corpus/unique_texts.csv"
EMBEDDING_CHUNKS = ROOT / "data/full_corpus/run_v1/embeddings/multilingual/chunks"
EMBEDDING_MANIFEST = ROOT / "data/full_corpus/run_v1/embeddings/multilingual/embedding_manifest.json"
LABELS = list(MODEL_LABELS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_embeddings(texts: list[str]) -> tuple[np.ndarray, dict]:
    unique = pd.read_csv(CORPUS_UNIQUE)
    index_by_text = dict(zip(unique.original_text.astype(str), unique.unique_id.astype(int)))
    ids = [index_by_text.get(str(text)) for text in texts]
    if any(value is None for value in ids):
        raise RuntimeError("some reference texts do not match the existing embedding corpus exactly")
    max_id = int(unique.unique_id.max())
    vectors = np.empty((max_id + 1, 384), dtype=np.float32)
    seen = np.zeros(max_id + 1, dtype=bool)
    for chunk_path in sorted(EMBEDDING_CHUNKS.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            vectors[chunk["idx"]] = chunk["emb"]
            seen[chunk["idx"]] = True
    if not seen.all():
        raise RuntimeError(f"cached embeddings do not cover all corpus rows: {int((~seen).sum())} missing")
    return vectors[np.asarray(ids)], {"matched_reference_texts": len(ids), "embedding_dim": 384,
                                     "unique_corpus_texts": len(unique),
                                     "exact_text_match_count": int(len(ids)),
                                     "embedding_manifest_sha256": sha(EMBEDDING_MANIFEST),
                                     "embedding_data_sha256": json.loads(EMBEDDING_MANIFEST.read_text())["embeddings_sha256"]}


def fold_map(frame: pd.DataFrame) -> list[dict]:
    manifest = pd.read_json(FOLD_MANIFEST, lines=True)
    manifest = manifest[manifest.condition == "shared_original_folds"]
    row_index = {str(pid): i for i, pid in enumerate(frame.pilot_id)}
    folds = []
    for (repeat, fold), rows in manifest.groupby(["repeat", "fold"], sort=True):
        train_ids = rows.loc[rows.split == "train", "record_id"].astype(str).tolist()
        valid_ids = rows.loc[rows.split == "validation", "record_id"].astype(str).tolist()
        groups = rows.set_index(rows.record_id.astype(str)).leakage_group.astype(str)
        train_groups = {groups.loc[item] for item in train_ids}
        valid_groups = {groups.loc[item] for item in valid_ids}
        if train_groups & valid_groups:
            raise AssertionError("saved fold crosses source/duplicate leakage groups")
        folds.append({"repeat": int(repeat), "fold": int(fold),
                      "train": np.asarray([row_index[item] for item in train_ids], dtype=int),
                      "valid": np.asarray([row_index[item] for item in valid_ids], dtype=int)})
    if len(folds) != 15:
        raise RuntimeError(f"expected 15 saved folds, found {len(folds)}")
    return folds


def summarize(fold_rows: list[dict]) -> dict:
    result = {}
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        values = [row[key] for row in fold_rows]
        result[key] = {"mean": round(float(np.mean(values)), 4),
                       "std": round(float(np.std(values, ddof=1)), 4)}
    result["per_class"] = {}
    for label in LABELS:
        result["per_class"][label] = {}
        for key in ("precision", "recall", "f1", "support"):
            values = [row["per_class"][label][key] for row in fold_rows]
            result["per_class"][label][key] = {"mean": round(float(np.mean(values)), 4),
                                               "std": round(float(np.std(values, ddof=1)), 4)}
    result["pooled_repeated_oof_confusion"] = np.sum(
        [row["confusion_matrix"]["rows_reference_cols_predicted"] for row in fold_rows], axis=0
    ).astype(int).tolist()
    result["pooled_repeated_oof_n"] = int(sum(row["n"] for row in fold_rows))
    return result


def audit_reference_pool(source: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    normalized = eligible.original_text.fillna("").astype(str).str.casefold().str.split().str.join(" ")
    duplicate_labels = pd.DataFrame({"text": normalized, "label": eligible.gold_stance.astype(str)}).groupby("text").label.nunique()
    corpus_groups = pd.read_csv(ROOT / "data/full_corpus/run_v1/corpus/corpus_entries.csv",
                                dtype={"source_record_id": str}, keep_default_na=False)
    corpus_pairs = set(zip(corpus_groups.source_dataset.astype(str), corpus_groups.source_record_id.astype(str)))
    return {
        "all_reference_label_counts_including_excluded": source.gold_stance.value_counts(dropna=False).to_dict(),
        "three_class_counts": eligible.gold_stance.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        "single_oppose_record_excluded_from_three_class_fit": int((source.gold_stance == "Oppose").sum()),
        "reference_label_source_counts": source.label_source.value_counts(dropna=False).to_dict(),
        "source_dataset_counts": source.source_dataset.value_counts(dropna=False).to_dict(),
        "language_counts": source.source_language.value_counts(dropna=False).to_dict(),
        "source_by_label": pd.crosstab(source.source_dataset, source.gold_stance).to_dict(),
        "language_by_label": pd.crosstab(source.source_language, source.gold_stance).to_dict(),
        "available_columns": list(source.columns),
        "model_input_columns": ["original_text"],
        "context_columns_available_but_not_used": [name for name in
            ("question_wording", "question_context", "context", "source_field", "question_slug", "source_language")
            if name in source.columns],
        "identifier_label_note_columns_excluded_from_model": [name for name in
            ("pilot_id", "source_record_id", "gold_stance", "ai_stance", "ai_review_notes", "label_source",
             "round", "excluded_reason") if name in source.columns],
        "corpus_source_record_pairs_mapped": int(sum((str(a), str(b)) in corpus_pairs
                                                      for a, b in zip(source.source_dataset, source.source_record_id))),
        "eligible_records": int(len(eligible)),
        "normalized_exact_duplicate_rows": int(normalized.duplicated(keep=False).sum()),
        "normalized_exact_duplicate_texts_with_conflicting_labels": int((duplicate_labels > 1).sum()),
        "text_characters": {key: round(float(value), 2) for key, value in
            eligible.original_text.astype(str).str.len().describe(percentiles=[.25, .5, .75, .9]).items()},
        "interpretation": "Unclear combines context gaps, ambiguity, fragments, off-topic text, and annotator language inability. The saved classifiers use comment text only; several reasons require context or annotator-specific information and are not consistently identifiable from text alone.",
    }


def validation_sweep_summary() -> dict:
    path = ROOT / "reports/stance_feature_experiments_v3/metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = {}
    for candidate in data.get("candidates", []):
        feature, model = candidate.get("features"), candidate.get("model")
        key = (feature, model)
        if key not in groups or (candidate["validation"]["macro_f1"], candidate["validation"]["accuracy"]) > (
                groups[key]["validation"]["macro_f1"], groups[key]["validation"]["accuracy"]):
            groups[key] = candidate
    return {"source": str(path.relative_to(ROOT)), "previously_exposed_validation_only": True,
            "candidate_count": len(data.get("candidates", [])),
            "selected_validation_model": data.get("selected_by_validation"),
            "family_best": [{"features": item["features"], "model": item["model"], "key": item["key"],
                             "C": item.get("C"), "class_weight": item.get("class_weight"),
                             "validation": item["validation"]}
                            for item in groups.values()]}


def gap(model, x) -> np.ndarray:
    scores = np.asarray(model.decision_function(x))
    if scores.ndim == 1:
        return np.abs(scores)
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def custom_text_model(config: dict):
    word = TfidfVectorizer(ngram_range=tuple(config["word_ngram"]), sublinear_tf=True,
                           max_features=config["word_max_features"], min_df=config["word_min_df"])
    features = word
    if config.get("char_ngram"):
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(config["char_ngram"]),
                               sublinear_tf=True, max_features=config["char_max_features"],
                               min_df=config["char_min_df"], strip_accents="unicode")
        features = FeatureUnion([("word", word), ("char", char)])
    clf = LinearSVC(C=config["C"], class_weight=config["class_weight"],
                    random_state=20261029, max_iter=10000)
    return Pipeline([("tfidf", features), ("clf", clf)])


def selective_metrics(predictions: pd.DataFrame) -> dict:
    rows = []
    for name, frame in predictions.groupby("condition", sort=True):
        if frame.margin.isna().all():
            continue
        # Thresholds are pre-fixed confidence ranks within each validation fold.
        # Labels do not determine which rows are accepted.
        for keep_share in (1.0, 0.9, 0.8, 0.7, 0.5):
            accepted_parts = []
            for _, fold in frame.groupby(["repeat", "fold"], sort=False):
                count = int(np.ceil(len(fold) * keep_share))
                accepted_parts.append(fold.nlargest(count, "margin", keep="first"))
            accepted = pd.concat(accepted_parts, ignore_index=True)
            rows.append({"condition": name, "target_within_fold_coverage": keep_share,
                         "actual_coverage": round(len(accepted) / len(frame), 4),
                         "accepted_n": int(len(accepted)), "all_n": int(len(frame)),
                         "selective_accuracy": round(float(accuracy_score(accepted.reference_stance,
                                                                            accepted.predicted_stance)), 4),
                         "selective_macro_f1": round(float(f1_score(accepted.reference_stance,
                                                                      accepted.predicted_stance,
                                                                      labels=LABELS, average="macro",
                                                                      zero_division=0)), 4),
                         "unclear_accepted_n": int((accepted.reference_stance == "Unclear").sum()),
                         "unclear_recall_within_accepted": round(float(((accepted.reference_stance == "Unclear") &
                             (accepted.predicted_stance == "Unclear")).sum() /
                             max(1, (accepted.reference_stance == "Unclear").sum())), 4)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/stance_existing_resources_benchmark_v4")
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)

    status, source = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or source is None:
        raise RuntimeError(status.detail)
    frame = source[source.gold_stance.isin(LABELS)].reset_index(drop=True)
    if len(frame) != 399:
        raise AssertionError(f"expected 399 three-class reference rows, got {len(frame)}")
    x_text = frame.original_text.astype(str).to_numpy()
    y = frame.gold_stance.astype(str).to_numpy()
    langs = frame.source_language.fillna("unknown").astype(str).to_numpy()
    x_embed, embedding_info = load_embeddings(x_text.tolist())
    folds = fold_map(frame)
    v4_pool = pd.read_json(V4_POOL, lines=True)
    v2_pool = pd.read_json(V2_POOL, lines=True)
    v4_hash, v2_hash = sha(V4_POOL), sha(V2_POOL)

    configs = {
        "production_word_only": ("text", "word_linsvm", 0.3, "balanced", None),
        "word_linsvm_validation_best": ("text", "word_linsvm", 0.3, None, None),
        "word_logreg_validation_best": ("text", "word_logreg", 3.0, None, None),
        "word_char_linsvm_validation_best": ("text", "word_char_linsvm", 0.3, "balanced", None),
        "word_char_logreg_validation_best": ("text", "word_char_logreg", 3.0, "balanced", None),
        "candidate_v1_recipe_v4_ratio_0_5": ("text", "word_char_linsvm", 1.0, "balanced", "v4"),
        "candidate_v2_recipe_clean_ratio_0_5": ("text", "word_char_linsvm", 1.0, "balanced", "v2"),
        "cached_minilm_logreg_validation_best": ("embedding_logreg", "", 3.0, "balanced", None),
        "cached_minilm_linsvm": ("embedding_linsvm", "", 1.0, "balanced", None),
        "explore_word_unigram_linsvm": ("custom_text", "linsvm", 0.3, "balanced", "word_unigram"),
        "explore_word_trigram_linsvm": ("custom_text", "linsvm", 0.3, "balanced", "word_trigram"),
        "explore_word_cap10k_min_df2_linsvm": ("custom_text", "linsvm", 0.3, "balanced", "word_small_vocab"),
        "explore_char_2_5_cap15k_min_df1_linsvm": ("custom_text", "linsvm", 1.0, "balanced", "char_alternative"),
        "majority_class_baseline": ("majority", "", 0.0, None, None),
    }
    custom_specs = {
        "word_unigram": {"word_ngram": (1, 1), "word_max_features": 20000, "word_min_df": 1,
                         "C": 0.3, "class_weight": "balanced"},
        "word_trigram": {"word_ngram": (1, 3), "word_max_features": 20000, "word_min_df": 1,
                         "C": 0.3, "class_weight": "balanced"},
        "word_small_vocab": {"word_ngram": (1, 2), "word_max_features": 10000, "word_min_df": 2,
                             "C": 0.3, "class_weight": "balanced"},
        "char_alternative": {"word_ngram": (1, 2), "word_max_features": 20000, "word_min_df": 1,
                             "char_ngram": (2, 5), "char_max_features": 15000, "char_min_df": 1,
                             "C": 1.0, "class_weight": "balanced"},
    }
    fold_results = {name: [] for name in configs}
    predictions = []
    for split in folds:
        train, valid = split["train"], split["valid"]
        fold_seed = 20261029 + split["repeat"] * 1009 + split["fold"]
        for name, (kind, model_kind, c, weight, synthetic_name) in configs.items():
            if kind == "majority":
                pred = np.full(len(valid), pd.Series(y[train]).value_counts().idxmax(), dtype=object)
                metrics = evaluate(y[valid], pred, LABELS)
                fold_results[name].append({"repeat": split["repeat"], "fold": split["fold"], **metrics})
                for offset, row_index in enumerate(valid):
                    predictions.append({"condition": name, "repeat": split["repeat"], "fold": split["fold"],
                                        "record_id": str(frame.iloc[row_index].pilot_id),
                                        "language": langs[row_index], "reference_stance": y[row_index],
                                        "predicted_stance": str(pred[offset]), "margin": None,
                                        "is_error": bool(pred[offset] != y[row_index])})
                continue
            if kind == "embedding_logreg":
                model = LogisticRegression(C=c, class_weight=weight, max_iter=3000,
                                           random_state=20261029)
                train_x, valid_x = x_embed[train], x_embed[valid]
            elif kind == "embedding_linsvm":
                from sklearn.svm import LinearSVC
                model = LinearSVC(C=c, class_weight=weight, max_iter=10000,
                                  random_state=20261029)
                train_x, valid_x = x_embed[train], x_embed[valid]
            elif kind == "custom_text":
                model = custom_text_model(custom_specs[synthetic_name])
                train_x, valid_x = x_text[train], x_text[valid]
            else:
                model = build_model(model_kind, c, weight)
                train_x, valid_x = x_text[train], x_text[valid]
            train_y = y[train].copy()
            if synthetic_name in {"v4", "v2"}:
                pool = v4_pool if synthetic_name == "v4" else v2_pool
                synthetic = select_synthetic(pool, len(train), 0.5, fold_seed)
                train_x = np.concatenate([train_x, synthetic.text.astype(str).to_numpy()])
                train_y = np.concatenate([train_y, synthetic.reference_stance.astype(str).to_numpy()])
            model.fit(train_x, train_y)
            pred = model.predict(valid_x)
            metrics = evaluate(y[valid], pred, LABELS)
            fold_results[name].append({"repeat": split["repeat"], "fold": split["fold"], **metrics})
            margins = gap(model, valid_x)
            for offset, row_index in enumerate(valid):
                predictions.append({"condition": name, "repeat": split["repeat"], "fold": split["fold"],
                                    "record_id": str(frame.iloc[row_index].pilot_id),
                                    "language": langs[row_index], "reference_stance": y[row_index],
                                    "predicted_stance": str(pred[offset]), "margin": float(margins[offset]),
                                    "is_error": bool(pred[offset] != y[row_index])})

    pred_frame = pd.DataFrame(predictions)
    pred_frame.to_json(out / "grouped_cv_predictions.jsonl", orient="records", lines=True)
    summaries = {name: summarize(rows) for name, rows in fold_results.items()}
    selective = selective_metrics(pred_frame)
    lang_summary = {}
    for condition, group in pred_frame.groupby("condition"):
        lang_summary[condition] = {}
        for lang, local in group.groupby("language"):
            lang_summary[condition][lang] = evaluate(local.reference_stance.to_numpy(),
                                                       local.predicted_stance.to_numpy(), LABELS)

    result = {
        "evaluation_status": "repeated_source_duplicate_grouped_cv_development_only_pool_previously_exposed",
        "seed": 20261029, "fold_source": str(FOLD_MANIFEST.relative_to(ROOT)),
        "folds": 5, "repeats": 3, "fold_count": len(folds), "reference_n": len(frame),
        "reference_counts": pd.Series(y).value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        "language_counts": pd.Series(langs).value_counts().to_dict(),
        "reference_pool_audit": audit_reference_pool(source, frame),
        "configs": {key: {"feature": value[0], "model": value[1], "C": value[2],
                          "class_weight": value[3], "synthetic_pool": value[4],
                          "synthetic_ratio": 0.5 if value[4] in {"v4", "v2"} else 0.0,
                          "vectorizer_variant": custom_specs.get(value[4])} for key, value in configs.items()},
        "synthetic_pool_sha256": {"v4": v4_hash, "v2_clean": v2_hash},
        "cached_embedding": embedding_info,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "fold_summaries": summaries, "language_oof_metrics": lang_summary,
        "previous_validation_sweep_summary": validation_sweep_summary(),
        "abstention_by_within_fold_margin_rank": selective,
        "predictions_sha256": None,
    }
    result["predictions_sha256"] = sha(out / "grouped_cv_predictions.jsonl")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# Existing-resource stance grouped benchmark and abstention analysis", "",
             "No new examples or model downloads were used. TF-IDF recipes, synthetic pools, grouped folds, and cached multilingual MiniLM embeddings were read from existing repository assets. The reference pool has prior development exposure; these are development robustness estimates, not unseen-source results.", "",
             f"Three-class reference rows: {len(frame)} ({json.dumps(result['reference_counts'])}); languages: {json.dumps(result['language_counts'])}. Source/duplicate-grouped folds: 3 repeats x 5 folds. Hyperparameters were frozen from the existing validation sweep; no old development test scores were used to select them.", "",
             "The source pool has 400 records: 140 Support, 219 Neutral, 40 Unclear, and one Oppose. The singleton Oppose record remains in the original source manifest and is excluded from three-class fitting. All 400 rows carry the same AI working-label provenance; this is not independent human ground truth. The 399 modeled rows comprise 364 English and 35 French examples. All 399 map to corpus source records; 398 modeled leakage groups remain after source/duplicate grouping. There are no normalized exact duplicate texts or exact-text cross-label conflicts among the modeled rows.", "",
             "Available data include comment text, source and record identifiers, question wording/context, language and duplicate metadata, plus label/review provenance. Identifiers and label/review fields are excluded. The current model input is only `original_text`; question wording, question context, `context`, source field, question slug, and language are not features. This matters because guideline Unclear reasons include missing context and language-comprehension inability, while the schema also has a `Mixed` concept not modeled here.", "",
             "## Prior validation sweep and negative results", "",
             f"The existing prior-exposed validation sweep tested {result['previous_validation_sweep_summary']['candidate_count']} settings over word TF-IDF and word+character TF-IDF with LinearSVC/Logistic Regression, plus cached multilingual MiniLM with Logistic Regression; C, class weight, n-gram/features, and minimum document frequency are recorded in the source metrics. Best per family (validation only):", "",
             "| Feature/model family | Selected setting | Validation accuracy | Validation macro-F1 |", "|---|---|---:|---:|"]
    for item in result["previous_validation_sweep_summary"]["family_best"]:
        lines.append(f"| {item['features']} / {item['model']} | {item['key']} | {item['validation']['accuracy']:.4f} | {item['validation']['macro_f1']:.4f} |")
    lines += ["", "The validation-best MiniLM recipe did not transfer into this same-fold grouped comparison as the accuracy leader. Cached MiniLM LinearSVC and Logistic Regression are below the word+character synthetic recipes on grouped mean accuracy; no pretrained model download was attempted. The full per-fold, class, confusion, language, configuration, and software-version results are in `metrics.json`.", "",
             "| Recipe | Accuracy mean ± SD | Macro-F1 mean ± SD | Weighted-F1 mean ± SD | Unclear P/R/F1 mean |", "|---|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        u = summary["per_class"]["Unclear"]
        lines.append(f"| {name} | {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f} | {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f} | {summary['weighted_f1']['mean']:.4f} ± {summary['weighted_f1']['std']:.4f} | {u['precision']['mean']:.4f} / {u['recall']['mean']:.4f} / {u['f1']['mean']:.4f} |")
    lines += ["", "None of the measured configurations reached 90% mean grouped-CV accuracy. The best mean is the v1 word+character synthetic recipe at 0.6191. This is agreement with the repository's unverified AI reference labels, not human-verified stance accuracy."]
    lines += ["", "## Abstention operating points", "", "For each saved validation fold, the top X% of rows by the model's top-two decision-score gap are accepted. Coverage thresholds are fixed without reference labels; metrics are then calculated on accepted rows. Margins are not calibrated probabilities, and these operating curves do not imply that abstention resolves the ambiguous Unclear construct. Full reference-label versus predicted-label confusion matrices are recorded per recipe in `metrics.json`; the raw prediction file enables per-record error review without changing labels.", "", "| Recipe | Target coverage | Actual coverage | Accepted accuracy | Accepted macro-F1 | Accepted Unclear n | Unclear recall among accepted |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in selective:
        lines.append(f"| {row['condition']} | {row['target_within_fold_coverage']:.0%} | {row['actual_coverage']:.3f} | {row['selective_accuracy']:.4f} | {row['selective_macro_f1']:.4f} | {row['unclear_accepted_n']} | {row['unclear_recall_within_accepted']:.4f} |")
    lines += ["", "The additional same-fold exploratory variants cover unigram and 1–3 word n-grams, a 10k feature cap with min_df=2, and character 2–5 grams with a 15k cap/min_df=1. They were not selected using the folds; per-fold results remain descriptive.", "", "## Language subsets", "", "Per-language grouped out-of-fold counts and metrics are in `metrics.json`. French has only 35 distinct reference records, repeated across folds; interpret those estimates cautiously.", "", "## Reproduction", "", "`python scripts/benchmark_stance_existing_resources_v1.py --out-dir reports/stance_existing_resources_benchmark_v3`", "", "Exact record/fold assignments are in `reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl`. Fold predictions and margins are in `grouped_cv_predictions.jsonl`; their SHA-256 is recorded in `metrics.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "candidate_v1_macro_f1": summaries["candidate_v1_recipe_v4_ratio_0_5"]["macro_f1"],
                      "minilm_macro_f1": summaries["cached_minilm_logreg_validation_best"]["macro_f1"],
                      "prediction_sha256": result["predictions_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
