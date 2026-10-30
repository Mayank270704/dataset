"""Build a read-only annotation queue from original reference records and shared OOF predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification.labels import load_stance_reference_labels
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

CV_PATH = ROOT / "reports/stance_candidate_shared_cv_v2/shared_fold_predictions.jsonl"
OUTPUT_DIR = ROOT / "reports/stance_annotation_review_queue_v1"
SCHEMA = ["record_id", "source_id", "language", "text", "current_reference_label",
          "production_prediction", "candidate_prediction", "review_reason", "proposed_reason_code",
          "reviewer_1_label", "reviewer_2_label", "adjudicated_label", "reviewer_notes"]


def reason_code(note: str) -> str:
    """Surface existing documented Unclear reason hints without assigning a new label."""
    s = (note or "").casefold()
    tags = []
    if "non-english" in s or "cannot reliably read french" in s or "cannot read french" in s:
        tags.append("language_comprehension_protocol")
    if "off-topic" in s or "off topic" in s:
        tags.append("off_topic")
    if "fragment" in s:
        tags.append("fragment")
    if "insufficient context" in s or "context missing" in s or "unseen statement" in s:
        tags.append("insufficient_context")
    if "ambiguous" in s or "ambiguity" in s:
        tags.append("genuine_ambiguity")
    if "no content" in s or "unintelligible" in s:
        tags.append("unintelligible_or_no_content")
    return ";".join(dict.fromkeys(tags)) or "review_reason_unrecorded"


def build_queue(originals: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = originals[originals.gold_stance.isin(["Support", "Neutral", "Unclear"])].copy()
    key = "pilot_id"
    predictions = predictions[predictions.condition.isin(["production_word_only", "candidate_v1_recipe"])].copy()
    if predictions.empty:
        raise ValueError("No production/candidate OOF predictions found")
    # These are fold-held-out predictions; aggregate repeat votes without fitting on the row.
    pred_by = {}
    for condition, short in (("production_word_only", "production"), ("candidate_v1_recipe", "candidate")):
        part = predictions[predictions.condition.eq(condition)]
        pred_by[short] = part.groupby(key).predicted_stance.agg(
            lambda values: sorted(values.astype(str).value_counts().items(), key=lambda p: (-p[1], p[0]))[0][0])
    errors = predictions[predictions.reference_stance.ne(predictions.predicted_stance)]
    error_counts = errors.groupby(["condition", key]).size()

    rows = []
    for r in eligible.itertuples(index=False):
        rid = str(getattr(r, key))
        if rid not in pred_by["production"].index or rid not in pred_by["candidate"].index:
            continue
        prod, cand = str(pred_by["production"][rid]), str(pred_by["candidate"][rid])
        ref = str(r.gold_stance)
        lang = str(r.source_language or "Unknown")
        note_value = getattr(r, "ai_review_notes", "")
        note = "" if pd.isna(note_value) else str(note_value)
        reasons = []
        proposed = ""
        if ref == "Unclear":
            reasons.append("reference_unclear_requires_reason_review")
            proposed = reason_code(note)
        if prod != cand:
            reasons.append("production_candidate_oof_disagreement")
        prod_err = int(error_counts.get(("production_word_only", rid), 0))
        cand_err = int(error_counts.get(("candidate_v1_recipe", rid), 0))
        if ref in ("Support", "Neutral") and (prod_err or cand_err):
            reasons.append("support_neutral_oof_error" if ref in ("Support", "Neutral") else "")
        if lang.casefold().startswith("french"):
            reasons.append("underrepresented_french_review")
        if not reasons:
            continue
        rows.append({
            "record_id": rid,
            "source_id": f"{r.source_dataset}#{r.source_record_id}",
            "language": lang,
            "text": str(r.original_text),
            "current_reference_label": ref,
            "production_prediction": prod,
            "candidate_prediction": cand,
            "review_reason": ";".join(reasons),
            "proposed_reason_code": proposed,
            "reviewer_1_label": "",
            "reviewer_2_label": "",
            "adjudicated_label": "",
            "reviewer_notes": "",
            "_unclear": int(ref == "Unclear"),
            "_disagree": int(prod != cand),
            "_error_count": prod_err + cand_err,
            "_french": int(lang.casefold().startswith("french")),
        })
    queue = pd.DataFrame(rows)
    if queue.empty:
        return pd.DataFrame(columns=SCHEMA)
    queue.sort_values(["_unclear", "_disagree", "_error_count", "_french", "record_id"],
                      ascending=[False, False, False, False, True], inplace=True)
    return queue[SCHEMA].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=CV_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    status, originals = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or originals is None:
        raise RuntimeError(status.detail)
    predictions = pd.read_json(args.predictions, lines=True)
    queue = build_queue(originals, predictions)
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    csv_path = out / "review_queue.csv"
    queue.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "reference_input": str(REFERENCE_LABELS_PATH.relative_to(ROOT)),
        "prediction_input": str(args.predictions),
        "eligible_original_records": int(originals.gold_stance.isin(["Support", "Neutral", "Unclear"]).sum()),
        "eligible_reference_label_counts": originals[originals.gold_stance.isin(["Support", "Neutral", "Unclear"])].gold_stance.value_counts().to_dict(),
        "eligible_language_counts": originals[originals.gold_stance.isin(["Support", "Neutral", "Unclear"])].source_language.value_counts(dropna=False).to_dict(),
        "queue_records": int(len(queue)),
        "reference_label_counts_in_queue": queue.current_reference_label.value_counts().to_dict(),
        "language_counts_in_queue": queue.language.value_counts().to_dict(),
        "review_reason_counts": {reason: int(queue.review_reason.str.contains(reason, regex=False).sum())
                                 for reason in sorted({x for s in queue.review_reason for x in s.split(";")})},
        "reviewer_fields_blank": bool(queue[["reviewer_1_label", "reviewer_2_label", "adjudicated_label", "reviewer_notes"]].eq("").all().all()),
        "predictions_are_repeated_grouped_oof_votes": True,
        "confidence": "Not supplied: source grouped-CV artifact contains no calibrated confidence scores.",
        "labels_modified": False,
        "input_sha256": {
            "reference_labels": hashlib.sha256(REFERENCE_LABELS_PATH.read_bytes()).hexdigest(),
            "grouped_oof_predictions": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        },
    }
    (out / "queue_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    guide = ROOT / "reports/stance_annotation_review_queue_v1/annotation_guide.md"
    if guide.is_file():
        shutil.copyfile(guide, out / "annotation_guide.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
