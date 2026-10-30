"""Generate provenance-tagged five-class stance training examples reproducibly.

The generated rows are synthetic training material, never original records or
evaluation examples. Language-inability scenarios are recorded separately without
a model label because that condition belongs to the annotator, not the text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification.schema import STANCE_LABELS
from models.stance_classification.labels import load_stance_reference_labels
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

DEFAULT_OUT = ROOT / "reports/stance_five_class_synthetic_v1"
GENERATION_METHOD = "seeded_guideline_aligned_five_class_templates_v1"
SEED_DEFAULT = 20261030

SLOTS = {
    "records": ["department spending records", "contract awards", "meeting minutes",
                "inspection reports", "program performance data", "policy drafts",
                "service statistics", "environmental monitoring data", "public grant totals",
                "travel expense summaries", "inspection schedules", "research datasets",
                "procurement decisions", "audit findings", "service wait-time data",
                "ministerial briefing summaries", "project completion reports",
                "public consultation submissions", "regulatory decisions",
                "infrastructure spending totals", "departmental plans"],
    "personal_records": ["named patient files", "individual benefit case records",
                          "personal addresses in a registry", "unredacted applicant files",
                          "case-level health information", "individual tax records"],
    "initiative": ["open-data policy", "public-records release policy", "open-government plan",
                   "default-publication proposal", "transparency program", "records-access policy",
                   "public disclosure initiative", "government data-sharing proposal",
                   "open-by-default rule", "information release plan"],
    "topic": ["budget decisions", "public procurement", "government performance",
              "access to public information", "participation in policy decisions"],
    "prior_option": ["the second option", "the earlier approach", "that recommendation",
                     "the alternative discussed before", "the previous proposal"],
    "offtopic": ["community garden", "hockey team", "northern trail", "local bakery",
                 "school concert", "train journey", "winter market", "family reunion",
                 "neighbourhood pool", "weekend fishing trip", "old bicycle", "music festival"],
    "condition": ["privacy safeguards are explained", "the costs are made clear",
                  "the records are properly reviewed", "the department provides more detail",
                  "the scope is narrowed", "the public is consulted"],
}

TEMPLATES = {
    "Oppose": [
        ("oppose_direct", "I oppose the proposal to publish {records} as a default government policy."),
        ("oppose_disagree", "I disagree with making {records} publicly available under this open-government plan."),
        ("oppose_reject", "Please reject the proposal to release {records} to everyone; I do not support that policy."),
        ("oppose_cancel", "The plan to publish {records} should be withdrawn. I oppose that open-data measure."),
        ("oppose_goal", "Government should not pursue the goal of making {records} public. I reject that proposal."),
        ("oppose_do_not", "Do not make {records} public by default. I oppose this proposed transparency policy."),
        ("oppose_not_support", "I do not support expanding the open-government policy to publish {records}."),
        ("oppose_against", "I am against the proposal to release {records} through the public data portal."),
        ("oppose_refuse", "I reject public release of {records} as a government policy, even if the files can be prepared."),
        ("oppose_should_not", "The department should not publish {records} under this open-government initiative."),
        ("oppose_disapprove", "I disapprove of the proposed policy to make {records} freely accessible to the public."),
        ("oppose_no_release", "Keep {records} closed. I oppose the proposal to release them as open government data."),
    ],
    "Mixed": [
        ("mixed_records_privacy", "I support publishing {records}, but oppose releasing {personal_records}."),
        ("mixed_access_casefiles", "Public access to {records} is worthwhile; however, I reject publishing {personal_records}."),
        ("mixed_open_restrict", "Expand open government for {records}, but do not make {personal_records} public."),
        ("mixed_support_reject", "I favour releasing {records} for accountability, while opposing release of {personal_records}."),
        ("mixed_yes_no", "Yes to public access for {records}; no to publishing {personal_records}."),
        ("mixed_minutes_identifiers", "Meeting minutes should be open, but I oppose including {personal_records} in public files."),
        ("mixed_budget_health", "I support open reporting about {topic}, but reject public release of {personal_records}."),
        ("mixed_participation_records", "I support more public participation in {topic}, but I oppose publishing {personal_records}."),
        ("mixed_release_limits", "Release {records} for public review; keep {personal_records} out of the open-data program."),
        ("mixed_accountability_privacy", "Transparency about {topic} is important, yet I do not support making {personal_records} public."),
        ("mixed_support_scope", "I support the open-government plan for {records}, but oppose extending it to {personal_records}."),
        ("mixed_publish_not_publish", "I support publishing {records} to improve accountability, but do not publish {personal_records}."),
    ],
    "Unclear": [
        ("unclear_missing_referent_1", "I prefer {prior_option} mentioned earlier."),
        ("unclear_missing_referent_2", "Please use {prior_option} from the previous discussion."),
        ("unclear_missing_referent_3", "That is not what I meant by {prior_option}."),
        ("unclear_fragment_1", "I would support the proposal if {condition}..."),
        ("unclear_fragment_2", "After reviewing the information, I think the government should"),
        ("unclear_fragment_3", "The records should be released, unless {condition}..."),
        ("unclear_fragment_4", "My view on the proposal is that it depends on {condition}"),
        ("unclear_offtopic_1", "The {offtopic} had a wonderful day this summer."),
        ("unclear_offtopic_2", "Our {offtopic} was enjoyable over the weekend."),
        ("unclear_offtopic_3", "I usually think about the {offtopic} when I have free time."),
        ("unclear_offtopic_4", "The {offtopic} near my home is a favourite place to visit."),
        ("unclear_ambiguity_1", "The release sounds helpful, but I cannot tell which {records} or conditions the proposal means."),
        ("unclear_ambiguity_2", "I might agree with the change, although the comment does not say what {initiative} includes."),
        ("unclear_ambiguity_3", "Making it available could be acceptable, depending on what 'it' refers to here."),
        ("unclear_ambiguity_4", "The proposal may be right or wrong; the available wording does not explain its scope for {topic}."),
        ("unclear_ambiguity_5", "I cannot tell whether this comment is asking for more public access or referring to {topic}."),
    ],
}

UNCLEAR_REASONS = {
    "unclear_missing_referent_": "insufficient_context",
    "unclear_fragment_": "fragment",
    "unclear_offtopic_": "off_topic",
    "unclear_ambiguity_": "genuine_ambiguity",
}

QUESTION = "What is your position on the open-government proposal?"
CONTEXT = "Consultation topic: transparency, access to government information, and public participation."
PROTOCOL_ROWS = [
    ("language_ability_fr_1", "Je ne peux pas évaluer ce commentaire en français."),
    ("language_ability_fr_2", "Je ne comprends pas le texte soumis en français."),
    ("language_ability_fr_3", "Je ne peux pas lire de façon fiable cette réponse en français."),
]


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(count_per_class: int, seed: int, original_texts: set[str] | None = None) -> pd.DataFrame:
    if count_per_class < 1:
        raise ValueError("count-per-class must be positive")
    rng = random.Random(seed)
    known = set(original_texts or ())
    rows, seen = [], set()
    target_counts = {"Oppose": count_per_class, "Mixed": count_per_class,
                     "Unclear": count_per_class}
    attempts_limit = count_per_class * 500
    for label, target in target_counts.items():
        templates = TEMPLATES[label]
        added, attempts = 0, 0
        while added < target and attempts < attempts_limit:
            attempts += 1
            template_id, template = rng.choice(templates)
            values = {key: rng.choice(options) for key, options in SLOTS.items()}
            comment = template.format(**values)
            normalized = normalize(comment)
            if not normalized or normalized in seen or normalized in known:
                continue
            seen.add(normalized)
            if label == "Unclear":
                reason = next(reason for prefix, reason in UNCLEAR_REASONS.items()
                              if template_id.startswith(prefix))
            else:
                reason = "explicit_policy_position" if label == "Oppose" else "explicit_support_and_opposition"
            index = len(rows) + 1
            rows.append({
                "synthetic_id": f"stance5-syn-{seed}-{index:06d}",
                "original_record_id": None,
                "question_wording": QUESTION,
                "question_context": CONTEXT,
                "original_text": comment,
                "reference_stance": label,
                "is_synthetic": True,
                "model_eligible": True,
                "training_only": True,
                "label_source": "synthetic_guideline_template",
                "generation_method": GENERATION_METHOD,
                "template_id": template_id,
                "template_family": template_id.rsplit("_", 1)[0],
                "synthetic_reason": reason,
                "language": "English",
                "random_seed": int(seed),
                "generation_index": index,
            })
            added += 1
        if added != target:
            raise RuntimeError(f"generated {added}/{target} rows for {label}")
    return pd.DataFrame(rows)


def quality_check(rows: pd.DataFrame, count_per_class: int, original_texts: set[str]) -> dict:
    required = {"synthetic_id", "original_record_id", "question_wording", "question_context",
                "original_text", "reference_stance", "is_synthetic", "model_eligible", "training_only",
                "label_source", "generation_method", "template_id", "synthetic_reason",
                "language", "random_seed", "generation_index"}
    missing = required - set(rows.columns)
    if missing:
        raise AssertionError(f"synthetic provenance missing fields: {sorted(missing)}")
    if rows.synthetic_id.duplicated().any() or rows.original_text.map(normalize).duplicated().any():
        raise AssertionError("synthetic IDs/text must be unique")
    if rows.original_text.map(normalize).isin(original_texts).any():
        raise AssertionError("synthetic text duplicates an original reference record")
    if not rows.is_synthetic.all() or not rows.model_eligible.all() or not rows.training_only.all():
        raise AssertionError("synthetic corpus contains a row not marked synthetic training-only")
    if rows.original_record_id.notna().any():
        raise AssertionError("synthetic rows cannot claim original record IDs")
    if set(rows.reference_stance) != {"Oppose", "Mixed", "Unclear"}:
        raise AssertionError("unexpected synthetic labels")
    counts = rows.reference_stance.value_counts().reindex(STANCE_LABELS, fill_value=0).to_dict()
    expected = {"Support": 0, "Neutral": 0, "Oppose": count_per_class,
                "Mixed": count_per_class, "Unclear": count_per_class}
    if counts != expected:
        raise AssertionError(f"synthetic counts mismatch: {counts}")
    unclear_counts = rows[rows.reference_stance == "Unclear"].synthetic_reason.value_counts().to_dict()
    if any(reason in unclear_counts for reason in ("language_inability", "protocol_only")):
        raise AssertionError("annotator language ability cannot be an observable training target")
    for row in rows.itertuples():
        if row.reference_stance == "Oppose" and not re.search(
            r"\b(oppose|disagree|reject|disapprove|do not support|should not|against)\b", row.original_text, re.I
        ):
            raise AssertionError(f"Oppose template lacks explicit rejection: {row.template_id}")
        if row.reference_stance == "Mixed":
            if not re.search(r"\b(support|favour|worthwhile|should be open|should be public|expand|yes to|important|release .* for public review)\b", row.original_text, re.I):
                raise AssertionError(f"Mixed template lacks explicit support: {row.template_id}")
            if not re.search(r"\b(oppose|opposing|reject|do not|no to|keep .* out)\b", row.original_text, re.I):
                raise AssertionError(f"Mixed template lacks explicit opposition: {row.template_id}")
    return {"rows": int(len(rows)), "class_counts": counts,
            "unclear_reason_counts": unclear_counts,
            "template_counts": {f"{label}:{template}": int(count) for (label, template), count
                                in rows.groupby(["reference_stance", "template_id"]).size().items()},
            "normalized_exact_overlap_with_originals": 0,
            "language_inability_training_rows": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--count-per-class", type=int, default=120)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    status, originals = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or originals is None:
        raise RuntimeError(status.detail)
    original_texts = set(originals.original_text.astype(str).map(normalize))
    rows = generate(args.count_per_class, args.seed, original_texts)
    quality = quality_check(rows, args.count_per_class, original_texts)
    out.mkdir(parents=True, exist_ok=False)
    rows_path = out / "synthetic_training_rows.jsonl"
    rows.to_json(rows_path, orient="records", lines=True, force_ascii=False)
    protocol = pd.DataFrame([{
        "synthetic_id": f"stance5-protocol-{args.seed}-{i:03d}", "scenario_id": template_id,
        "text": text, "is_synthetic": True, "model_eligible": False,
        "training_only": False, "stance_label": None,
        "reason": "annotator_language_ability_is_not_inferable_from_comment_text",
        "generation_method": GENERATION_METHOD, "random_seed": args.seed,
    } for i, (template_id, text) in enumerate(PROTOCOL_ROWS, 1)])
    protocol.to_json(out / "language_ability_protocol_only.jsonl", orient="records", lines=True, force_ascii=False)
    meta = {
        "generation_method": GENERATION_METHOD, "seed": args.seed,
        "count_per_class": args.count_per_class,
        "reference_input_sha256": sha(REFERENCE_LABELS_PATH),
        "original_texts_used_for_overlap_exclusion": len(original_texts),
        "training_rows_sha256": sha(rows_path), "quality_checks": quality,
        "language_ability_protocol_rows": len(protocol),
        "language_ability_protocol_is_model_eligible": False,
        "schema": list(STANCE_LABELS),
        "provenance_note": "Synthetic rows are separate JSONL records. No original data file was modified.",
    }
    (out / "generation_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "rows": len(rows), "counts": quality["class_counts"],
                      "rows_sha256": meta["training_rows_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
