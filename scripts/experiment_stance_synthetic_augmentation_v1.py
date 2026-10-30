"""Controlled, training-only synthetic stance augmentation experiment.

The repository reference pool has already been exposed in development, so all test
results are development estimates. Synthetic rows are generated independently and
may only be appended to the original training fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from models.stance_classification import pipeline as stance_pipeline
from scripts.experiment_stance_reference_models_v2 import (
    MODEL_LABELS, REFERENCE_LABELS_PATH, SEED, build_model, evaluate, grouped_three_way_split,
)
from scripts.train_stance_baselines import corpus_groups

DEFAULT_OUT = ROOT / "reports/stance_synthetic_augmentation_v1"
UNCLEAR_TEMPLATES = {
    "insufficient_context": [
        ("unclear_context_1", "en", "That should happen the way we discussed earlier about {topic}."),
        ("unclear_context_2", "en", "I agree with the second option for {topic}."),
        ("unclear_context_3", "en", "Please do the same thing as last time with {records}."),
        ("unclear_context_4", "en", "I'll go with that, as mentioned above, for {topic}."),
    ],
    "genuine_ambiguity": [
        ("unclear_ambiguity_1", "en", "I support more openness, but publishing {records} would go too far."),
        ("unclear_ambiguity_2", "en", "The proposal sounds right, although I cannot accept it as written."),
        ("unclear_ambiguity_3", "en", "I can accept the change to {topic}, but I do not support it."),
        ("unclear_ambiguity_4", "en", "More information is welcome, except for the information that should stay public."),
    ],
    "fragment": [
        ("unclear_fragment_1", "en", "Because the {records} were not."),
        ("unclear_fragment_2", "en", "If they would only publish the {topic}..."),
        ("unclear_fragment_3", "en", "The reason I chose that option was"),
        ("unclear_fragment_4", "en", "After the meeting, it was still not clear whether."),
    ],
    "off_topic": [
        ("unclear_off_topic_1", "en", "Happy birthday to my cousin; we are having cake on Saturday."),
        ("unclear_off_topic_2", "en", "The tomatoes in my garden finally turned red this week."),
        ("unclear_off_topic_3", "en", "I watched a great film about mountain climbing last night."),
        ("unclear_off_topic_4", "en", "My favourite team won the hockey game by two goals."),
    ],
    "language_comprehension_difficulty": [
        ("unclear_language_fr_1", "fr", "Je ne peux pas lire ce commentaire en français."),
        ("unclear_language_fr_2", "fr", "Je ne comprends pas le texte en français."),
        ("unclear_language_fr_3", "fr", "Je ne peux pas évaluer cette réponse écrite en français."),
        ("unclear_language_fr_4", "fr", "Je ne sais pas lire les commentaires en français."),
    ],
}
UNCLEAR_REASONS = tuple(UNCLEAR_TEMPLATES)
SYNTHETIC_METHOD = "seeded_consultation_templates_v1"

# Templates follow annotation_guidelines_v1.1: stance concerns the OG initiative,
# not sentiment or satisfaction. French-language difficulty examples are explicitly
# marked as protocol scenarios; they must not teach that French itself means Unclear.
TEMPLATES = {
    "Support": [
        ("support_publish", "en", "The ministry should publish {records} by default so residents can review them."),
        ("support_access", "en", "I support making {records} available to everyone, not only by request."),
        ("support_participation", "en", "Residents need a real say in {process}; please keep the consultation open."),
        ("support_accountability", "en", "Open reporting on {topic} would make government more accountable."),
        ("support_default", "en", "Please put {records} online as a standard part of the department's openness policy."),
        ("support_request", "en", "People should not need to file a request to see {records}."),
        ("support_review", "en", "Publishing {records} would let the public check how decisions are made."),
        ("support_accessible", "en", "I want {records} released in an accessible format for public use."),
        ("support_fr", "fr", "Le gouvernement devrait publier {records_fr} pour que tout le monde puisse les consulter."),
        ("support_fr_participation", "fr", "Je soutiens une participation publique plus large aux décisions sur {topic_fr}."),
        ("support_fr_default", "fr", "Je souhaite que {records_fr} soient accessibles au public sans demande particulière."),
        ("support_fr_accountability", "fr", "Rendre {records_fr} accessibles aiderait le public à demander des comptes au gouvernement."),
    ],
    "Neutral": [
        ("neutral_question", "en", "When will the department release the next update about {topic}?"),
        ("neutral_experience", "en", "I attended the {event}; it started at nine and ended before lunch."),
        ("neutral_process", "en", "The consultation form took about {number} minutes to complete."),
        ("neutral_service", "en", "My request about {records} was answered after {number} days."),
        ("neutral_schedule", "en", "Where can I find the current schedule for {topic}?"),
        ("neutral_report", "en", "The department posted its {topic} report on Monday."),
        ("neutral_location", "en", "The public session about {topic} was held at the library."),
        ("neutral_question_report", "en", "Is the latest report about {topic} available online?"),
        ("neutral_fr", "fr", "La séance sur {topic_fr} a commencé à neuf heures et a duré une heure."),
        ("neutral_fr_question", "fr", "Quand le ministère publiera-t-il la prochaine mise à jour sur {topic_fr}?"),
        ("neutral_fr_schedule", "fr", "Où puis-je consulter le calendrier actuel concernant {topic_fr}?"),
        ("neutral_fr_report", "fr", "Le ministère a publié son rapport sur {topic_fr} lundi."),
    ],
    "Unclear": [
        # Detailed Unclear templates are selected by reason in generate_synthetic().
    ],
}
SLOTS = {
    "records": ["contracts", "meeting minutes", "program spending", "inspection reports", "service data", "policy drafts"],
    "process": ["the budget review", "the data policy", "the service redesign", "local planning", "the records program"],
    "topic": ["public data", "government records", "the open data plan", "the policy review", "public participation", "the portal"],
    "event": ["public meeting", "information session", "online workshop", "community forum"],
    "number": ["five", "ten", "twelve", "twenty", "three"],
    "records_fr": ["les contrats", "les comptes rendus", "les dépenses du programme", "les rapports d'inspection", "les données des services", "les projets de politique"],
    "topic_fr": ["les données publiques", "les archives gouvernementales", "le plan de données ouvertes", "l'examen de la politique", "la participation du public", "le portail"],
}


def generate_synthetic(per_class: int, seed: int) -> pd.DataFrame:
    """Return a deterministic, explicitly provenance-tagged synthetic corpus."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    rows = []
    slot_keys = list(SLOTS)
    ordinal = 0
    for label in MODEL_LABELS:
        bank = TEMPLATES[label]
        for label_index in range(per_class):
            if label == "Unclear":
                reason = UNCLEAR_REASONS[label_index % len(UNCLEAR_REASONS)]
                reason_index = label_index // len(UNCLEAR_REASONS)
                template_id, language, template = UNCLEAR_TEMPLATES[reason][reason_index % len(UNCLEAR_TEMPLATES[reason])]
            else:
                template_id, language, template = bank[label_index % len(bank)]
                reason = "stance_explicit"
            values = {key: rng.choice(SLOTS[key]) for key in slot_keys}
            text = template.format(**values)
            ordinal += 1
            rows.append({
                "synthetic_id": f"synthetic-{seed}-{ordinal:05d}",
                "text": text,
                "reference_stance": label,
                "is_synthetic": True,
                "generation_method": SYNTHETIC_METHOD,
                "template_id": template_id,
                "language": language,
                "random_seed": int(seed),
                "generation_index": ordinal,
                "synthetic_reason": reason,
            })
    return pd.DataFrame(rows)


def normalize_text(value: str) -> str:
    return re.sub(r"\W+", " ", str(value).casefold(), flags=re.UNICODE).strip()


def overlap_audit(synthetic: pd.DataFrame, heldout_texts: np.ndarray) -> dict:
    """Check normalized exact matches and character n-gram cosine/sequence near matches."""
    synthetic_norm = [normalize_text(t) for t in synthetic.text]
    heldout_norm = [normalize_text(t) for t in heldout_texts]
    exact_pairs = [(i, j) for i, s in enumerate(synthetic_norm) for j, h in enumerate(heldout_norm) if s and s == h]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2")
    matrix = vectorizer.fit_transform(synthetic_norm + heldout_norm)
    sims = cosine_similarity(matrix[:len(synthetic_norm)], matrix[len(synthetic_norm):])
    near_pairs = []
    for i, row in enumerate(sims):
        for j, score in enumerate(row):
            seq = SequenceMatcher(None, synthetic_norm[i], heldout_norm[j]).ratio()
            if score >= 0.72 or seq >= 0.82:
                near_pairs.append({"synthetic_id": synthetic.iloc[i].synthetic_id,
                                   "heldout_index": int(j), "char_cosine": round(float(score), 4),
                                   "sequence_ratio": round(float(seq), 4)})
    return {"exact_duplicate_count": len(exact_pairs), "near_duplicate_thresholds": {"char_cosine": 0.72, "sequence_ratio": 0.82},
            "near_duplicate_count": len(near_pairs), "near_duplicate_pairs": near_pairs}


def assert_synthetic_training_only(rows: pd.DataFrame) -> None:
    """Fail closed if any synthetic row is assigned to validation, test, or no split."""
    synthetic_rows = rows[rows.is_synthetic.astype(bool)]
    invalid = synthetic_rows[synthetic_rows.split != "train"]
    if not invalid.empty:
        raise AssertionError(f"synthetic rows must be training-only; invalid rows: {invalid.index.tolist()[:10]}")


def fit_validation_select(x_train, y_train, x_val, y_val, labels, seed):
    candidates = []
    for weight in (None, "balanced"):
        for c in (0.03, 0.1, 0.3, 1.0, 3.0):
            model = build_model("word_char_linsvm", c, weight)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            elapsed = time.perf_counter() - started
            metrics = evaluate(y_val, model.predict(x_val), labels)
            candidates.append({"model": model, "class_weight": weight, "C": c,
                               "validation": metrics, "fit_seconds": round(elapsed, 4)})
    return max(candidates, key=lambda r: (r["validation"]["macro_f1"], r["validation"]["accuracy"],
                                          r["class_weight"] == "balanced", -r["C"])), candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--synthetic-per-class", type=int, default=80,
                        help="size of generated corpus per class; defaults to 80 rows per class")
    parser.add_argument("--ratios", default="0,0.25,0.5,1.0",
                        help="synthetic/original-training ratios; 0 is the no-augmentation control")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    ratios = [float(value) for value in args.ratios.split(",")]
    if any(value < 0 for value in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("ratios must be unique non-negative values")

    status, source_frame = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or source_frame is None:
        raise RuntimeError(status.detail)
    corpus_group, mapping = corpus_groups(source_frame)
    all_groups = stance_pipeline.leakage_groups(source_frame, corpus_group)
    eligible = source_frame.gold_stance.isin(MODEL_LABELS).to_numpy()
    frame = source_frame.loc[eligible].reset_index(drop=True)
    groups = all_groups[eligible]
    y = frame.gold_stance.to_numpy()
    x = frame.original_text.astype(str).to_numpy()
    labels = list(MODEL_LABELS)
    train, validation, test = grouped_three_way_split(y, groups, args.seed)
    for name, ix in (("train", train), ("validation", validation), ("test", test)):
        if set(y[ix]) != set(labels):
            raise RuntimeError(f"{name} is missing a class; choose a different split seed")

    synthetic = generate_synthetic(args.synthetic_per_class, args.seed)
    synthetic_path = out_dir / "synthetic_training_rows.jsonl"
    synthetic.to_json(synthetic_path, orient="records", lines=True, force_ascii=False)
    heldout = np.concatenate([x[validation], x[test]])
    overlap = overlap_audit(synthetic, heldout)

    # Ratios scale the eligible original TRAIN fold only. Each class is sampled equally,
    # and exactly the same validation/test records are reused for every condition.
    conditions = []
    original_train_n = len(train)
    for ratio in ratios:
        target_total = int(round(ratio * original_train_n))
        per_label = target_total // len(labels)
        remainder = target_total - per_label * len(labels)
        counts = {label: per_label + (i < remainder) for i, label in enumerate(labels)}
        selected_ids = []
        for label in labels:
            subset = synthetic[synthetic.reference_stance == label]
            if counts[label] > len(subset):
                raise ValueError(f"synthetic-per-class corpus too small for ratio {ratio}: need {counts[label]}")
            selected_ids.extend(subset.sample(n=counts[label], random_state=args.seed + int(ratio * 1000) + labels.index(label)).synthetic_id.tolist())
        conditions.append((ratio, counts, synthetic[synthetic.synthetic_id.isin(selected_ids)].copy()))

    prepared = {}
    candidate_logs = {}
    for ratio, synthetic_counts, synthetic_rows in conditions:
        condition = "baseline" if ratio == 0 else f"ratio_{ratio:g}"
        syn_x = synthetic_rows.text.astype(str).to_numpy()
        syn_y = synthetic_rows.reference_stance.to_numpy()
        # Synthetic examples enter only the train fold. Validation is used for selection;
        # final comparison refits on original train+validation plus those same train-only rows.
        train_x = np.concatenate([x[train], syn_x])
        train_y = np.concatenate([y[train], syn_y])
        best, candidates = fit_validation_select(train_x, train_y, x[validation], y[validation], labels, args.seed)
        candidate_logs[condition] = [{k: v for k, v in row.items() if k != "model"} for row in candidates]
        prepared[condition] = {"ratio": ratio, "synthetic_rows": synthetic_rows, "best": best}

    # Freeze the augmentation ratio using validation macro-F1 before accessing test labels.
    selected_condition = max(prepared, key=lambda name: (
        prepared[name]["best"]["validation"]["macro_f1"],
        prepared[name]["best"]["validation"]["accuracy"],
        -prepared[name]["ratio"],
    ))

    results = {}
    for condition, selection in prepared.items():
        ratio = selection["ratio"]
        synthetic_rows = selection["synthetic_rows"]
        best = selection["best"]
        syn_x = synthetic_rows.text.astype(str).to_numpy()
        syn_y = synthetic_rows.reference_stance.to_numpy()
        final_model = build_model("word_char_linsvm", best["C"], best["class_weight"])
        final_x = np.concatenate([x[np.concatenate([train, validation])], syn_x])
        final_y = np.concatenate([y[np.concatenate([train, validation])], syn_y])
        final_model.fit(final_x, final_y)
        test_pred = final_model.predict(x[test])
        test_metrics = evaluate(y[test], test_pred, labels)
        language_metrics = {}
        langs = frame.source_language.fillna("unknown").astype(str).to_numpy()
        for lang in sorted(set(langs[test])):
            idx = np.where(langs[test] == lang)[0]
            language_metrics[lang] = evaluate(y[test][idx], test_pred[idx], labels)
        train_counts = pd.Series(y[train]).value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
        synth_counts = synthetic_rows.reference_stance.value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
        results[condition] = {
            "synthetic_to_original_train_ratio": ratio,
            "selected_configuration": {"C": best["C"], "class_weight": best["class_weight"]},
            "validation_metrics": best["validation"], "test_metrics": test_metrics,
            "language_test_metrics": language_metrics,
            "training_counts": {"original_train_total": len(train), "original_train_by_class": train_counts,
                                "synthetic_train_total": len(synthetic_rows), "synthetic_train_by_class": synth_counts,
                                "final_fit_total_including_original_validation": len(final_y)},
            "fit_seconds_selected_validation_candidate": best["fit_seconds"],
        }

    # This manifest makes the training-only boundary machine-checkable.
    manifest = source_frame.copy()
    manifest["split"] = "excluded_outside_three_class_model"
    manifest["model_eligible"] = eligible
    manifest["leakage_group"] = all_groups
    split_names = np.full(len(frame), "", dtype=object)
    split_names[train], split_names[validation], split_names[test] = "train", "validation", "test"
    split_map = dict(zip(frame.pilot_id, split_names))
    manifest.loc[manifest.pilot_id.isin(split_map), "split"] = manifest.loc[manifest.pilot_id.isin(split_map), "pilot_id"].map(split_map)
    manifest["is_synthetic"] = False
    manifest_cols = [c for c in ("pilot_id", "source_dataset", "source_record_id", "source_row_number", "gold_stance", "split", "is_synthetic") if c in manifest]
    original_manifest = manifest[manifest_cols].rename(columns={"pilot_id": "record_id", "gold_stance": "reference_stance"})
    synthetic_manifest = synthetic[["synthetic_id", "reference_stance", "is_synthetic", "generation_method", "template_id", "language", "random_seed"]].copy()
    synthetic_manifest["split"] = "train"
    synthetic_manifest.rename(columns={"synthetic_id": "record_id"}, inplace=True)
    full_manifest = pd.concat([original_manifest, synthetic_manifest], ignore_index=True)
    assert_synthetic_training_only(full_manifest)
    full_manifest.to_json(out_dir / "split_manifest.jsonl", orient="records", lines=True, force_ascii=False)

    metrics = {
        "label_role": "repository_reference_labels_for_original_records; synthetic_labels_are_experiment_targets",
        "evaluation_status": "development_estimate_reference_pool_previously_exposed",
        "limitation": "The full reference pool has been used in earlier development. Test results are development estimates, and synthetic agreement is not real-world accuracy.",
        "seed": args.seed, "synthetic_per_class_generated": args.synthetic_per_class,
        "synthetic_generation_method": SYNTHETIC_METHOD,
        "synthetic_total_generated": len(synthetic),
        "synthetic_provenance_fields": ["is_synthetic", "generation_method", "template_id", "language", "random_seed"],
        "original_split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "original_class_counts": {part: pd.Series(y[ix]).value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
                                   for part, ix in (("train", train), ("validation", validation), ("test", test))},
        "test_language_counts": frame.iloc[test].source_language.fillna("unknown").astype(str).value_counts().to_dict(),
        "leakage_groups": len(np.unique(groups)), "corpus_group_mapping": mapping,
        "validation_test_synthetic_overlap_audit": overlap,
        "results": results, "validation_candidates": candidate_logs,
        "validation_selected_condition": selected_condition,
        "validation_selected_ratio": prepared[selected_condition]["ratio"],
        "ratio_selection_rule": "validation macro-F1, then validation accuracy, then lower synthetic/original-train ratio; test is evaluated only after all settings and the ratio are frozen",
        "reference_input_sha256": hashlib.sha256(REFERENCE_LABELS_PATH.read_bytes()).hexdigest(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Controlled synthetic stance augmentation", "",
             "Only Support, Neutral, and Unclear are modeled. Original reference records are unchanged.",
             "The full reference pool has prior development exposure, so test values below are development estimates. Synthetic examples are training data and are not evidence of real-world accuracy.", "",
             f"Split seed {args.seed}: {len(train)} original train / {len(validation)} validation / {len(test)} test records; {len(np.unique(groups))} grouped source/duplicate units.",
             f"Generated {len(synthetic)} rows ({args.synthetic_per_class} per class) at `{synthetic_path.name}` with seed {args.seed}.",
             f"Validation selected `{selected_condition}` (ratio {prepared[selected_condition]['ratio']:g}) before test evaluation; all ratios are reported as a controlled comparison.",
             f"Held-out exact matches: {overlap['exact_duplicate_count']}; near matches at char cosine >=0.72 or sequence ratio >=0.82: {overlap['near_duplicate_count']}.",
             "Synthetic rows appear as `train` only in `split_manifest.jsonl`; validation and test rows are original records.", "",
             "| Condition | Synthetic/original train | Synthetic rows | Selected C / weight | Accuracy | Macro-F1 | Weighted-F1 | Test N |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        met = row["test_metrics"]
        lines.append(f"| {name} | {row['synthetic_to_original_train_ratio']:g} | {row['training_counts']['synthetic_train_total']} | {row['selected_configuration']['C']} / {row['selected_configuration']['class_weight']} | {met['accuracy']} | {met['macro_f1']} | {met['weighted_f1']} | {met['n']} |")
    for name, row in results.items():
        met = row["test_metrics"]
        lines += ["", f"## {name}", "", "| Class | Precision | Recall | F1 | Test count |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {label} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for label, v in met["per_class"].items()]
        lines.append(f"Confusion matrix, reference rows and predicted columns ({', '.join(labels)}): `{met['confusion_matrix']['rows_reference_cols_predicted']}`")
        lines.append("Training counts: " + json.dumps(row["training_counts"]))
        lines.append("Language test subsets: " + json.dumps({k: v["n"] for k, v in row["language_test_metrics"].items()}))
    lines += ["", "## Limitations", "", "The seed-20261029 grouped holdout was created after prior work exposed the full reference pool. It is not an untouched evaluation of unseen consultations. The Unclear class combines guideline-specific cases, and language-comprehension difficulty depends on annotator ability rather than only comment text; synthetic examples for that case are protocol scenarios and should not be interpreted as a learnable text signal.",
              "", f"Reproduce: `python scripts/experiment_stance_synthetic_augmentation_v1.py --out-dir reports/stance_synthetic_augmentation_rerun1 --seed {args.seed} --synthetic-per-class {args.synthetic_per_class} --ratios {args.ratios}`"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "results": {k: v["test_metrics"] for k, v in results.items()},
                      "overlap_audit": {k: v for k, v in overlap.items() if k != "near_duplicate_pairs"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
