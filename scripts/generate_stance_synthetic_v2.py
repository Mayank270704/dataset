"""Generate quality-gated synthetic stance examples for training-only experiments.

This version disables ambiguous/mixed-risk Unclear templates and moves French
annotator-comprehension scenarios to a separate protocol-only file with no stance
label. It never edits the already generated v4 corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_stance_synthetic_augmentation_v1 import SLOTS, TEMPLATES

DEFAULT_OUT = ROOT / "reports/stance_synthetic_corpus_v2"
GENERATION_METHOD = "seeded_consultation_templates_v2_quality_gated"
LABELS = ("Support", "Neutral", "Unclear")

UNCLEAR_TEMPLATES = {
    "insufficient_context": [
        ("unclear_context_v2_1", "en", "I agree with the second option about {topic}."),
        ("unclear_context_v2_2", "en", "No, the other one is better for {topic}."),
        ("unclear_context_v2_3", "en", "Please use the same approach as before for {topic}."),
        ("unclear_context_v2_4", "en", "That is what we discussed earlier about {topic}."),
        ("unclear_context_v2_5", "en", "I prefer that one over the first option on {topic}."),
        ("unclear_context_v2_6", "en", "They should repeat the earlier arrangement for {records}."),
        ("unclear_context_v2_7", "en", "That is exactly what I was referring to in the earlier discussion about {topic}."),
        ("unclear_context_v2_8", "en", "The previous option would be better than this one for {topic}."),
    ],
    "fragment": [
        ("unclear_fragment_v2_1", "en", "Because the {subject} was still..."),
        ("unclear_fragment_v2_2", "en", "After they reviewed {topic}, the department decided to"),
        ("unclear_fragment_v2_3", "en", "If only we had been told whether {topic}"),
        ("unclear_fragment_v2_4", "en", "The person who attended the {event} said that"),
        ("unclear_fragment_v2_5", "en", "To be considered after the next review of {topic}."),
        ("unclear_fragment_v2_6", "en", "When the department received the response about {topic}, it"),
        ("unclear_fragment_v2_7", "en", "The main reason for the decision about {topic} was"),
        ("unclear_fragment_v2_8", "en", "At the end of the {event}, they were going to"),
    ],
    "off_topic": [
        ("unclear_off_topic_v2_1", "en", "The tomatoes in my garden finally turned {colour} this week."),
        ("unclear_off_topic_v2_2", "en", "My favourite {sport} team won the game by {number} goals."),
        ("unclear_off_topic_v2_3", "en", "I watched a film about {hobby} last night."),
        ("unclear_off_topic_v2_4", "en", "We are having {food} with my family on {weekday}."),
        ("unclear_off_topic_v2_5", "en", "The new {household_item} works much better than the old one."),
        ("unclear_off_topic_v2_6", "en", "I am planning a {hobby} trip for the summer."),
        ("unclear_off_topic_v2_7", "en", "Our neighbourhood garden club meets every {weekday}."),
        ("unclear_off_topic_v2_8", "en", "The recipe needs more {food}; otherwise it tastes fine."),
    ],
}
PROTOCOL_TEMPLATES = [
    ("protocol_french_ordinary_support", "Le ministère devrait publier les contrats par défaut."),
    ("protocol_french_ordinary_neutral", "La séance sur les données publiques a commencé à neuf heures."),
    ("protocol_french_ordinary_question", "Quand le ministère publiera-t-il la prochaine mise à jour?"),
    ("protocol_french_ordinary_feedback", "Le formulaire de consultation était facile à remplir."),
    ("protocol_french_ordinary_duration", "La consultation publique a duré deux semaines."),
    ("protocol_french_ordinary_update", "Les données du portail sont mises à jour chaque mois."),
    ("protocol_french_ordinary_report", "Le rapport annuel a été publié en avril."),
    ("protocol_french_ordinary_meeting", "La réunion aura lieu à la bibliothèque municipale."),
]
PROTOCOL_CONDITION = "Unclear applies only if the annotator cannot reliably read French; no text-only model target is assigned."
DISABLED_V4_TEMPLATES = {
    "unclear_ambiguity_1": "Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.",
    "unclear_ambiguity_2": "Could be Support if the caveat is about implementation rather than the stance target.",
    "unclear_ambiguity_3": "Directly contradictory support and opposition; no stable intended label is evident.",
    "unclear_ambiguity_4": "Semantically awkward wording can imply Support rather than Unclear.",
    "unclear_language_fr_1": "Annotator reading ability is not visible in the comment text.",
    "unclear_language_fr_2": "Annotator reading ability is not visible in the comment text.",
    "unclear_language_fr_3": "Annotator reading ability is not visible in the comment text.",
    "unclear_language_fr_4": "Annotator reading ability is not visible in the comment text.",
}
SLOT_EXTENSIONS = {
    "subject": ["response", "document", "meeting", "question", "schedule", "report", "decision", "consultation"],
    "colour": ["red", "ripe", "green", "bright", "orange", "yellow"],
    "sport": ["hockey", "football", "basketball", "baseball", "rugby", "tennis"],
    "hobby": ["mountain climbing", "birdwatching", "painting", "kayaking", "woodworking", "photography"],
    "food": ["bread", "pasta", "cheese", "soup", "rice", "vegetables"],
    "weekday": ["Tuesday", "Wednesday", "Friday", "Saturday", "Sunday"],
    "household_item": ["kettle", "lamp", "desk chair", "vacuum", "coffee maker", "bookshelf"],
}
SUPPORT_SUFFIXES = {
    "en": [
        "That would improve public access.", "This would make government more accountable.",
        "Residents should be able to find the information easily.", "That is a practical step toward open government.",
        "This would help people participate in public decisions.", "People should not need special permission to see it.",
        "That would make public information more useful.", "I support making this information available.",
    ],
    "fr": [
        "Cela faciliterait l'accès du public.", "Le gouvernement serait ainsi plus responsable.",
        "Les résidents devraient trouver ces renseignements facilement.", "C'est une mesure concrète pour un gouvernement ouvert.",
        "Cela aiderait le public à participer aux décisions.", "Il ne faudrait pas demander une permission spéciale pour les consulter.",
        "Ces renseignements seraient plus utiles au public.", "Je soutiens l'accès public à ces renseignements.",
    ],
}
NEUTRAL_SUFFIXES = {
    "en": [
        "I am asking only for the current details.", "That is the information I received.",
        "I am describing what happened at the session.", "This is a question about the schedule.",
        "The response included those details.", "I am noting the date for reference.",
        "The form displayed that message.", "Those are the details I was given.",
    ],
    "fr": [
        "Je demande seulement les renseignements actuels.", "C'est l'information qui m'a été communiquée.",
        "Je décris ce qui s'est passé pendant la séance.", "Ma question porte sur le calendrier.",
        "La réponse contenait ces renseignements.", "Je note la date à titre de référence.",
        "Le formulaire affichait ce message.", "Ce sont les détails qui m'ont été fournis.",
    ],
}


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def quality_check(rows: pd.DataFrame, expected_per_class: int) -> dict:
    required = {"synthetic_id", "text", "reference_stance", "is_synthetic", "generation_method",
                "template_id", "language", "random_seed", "synthetic_reason", "model_eligible"}
    missing = required - set(rows.columns)
    if missing:
        raise AssertionError(f"synthetic corpus missing provenance: {sorted(missing)}")
    if len(rows) != expected_per_class * len(LABELS):
        raise AssertionError("unexpected generated row count")
    if rows["synthetic_id"].duplicated().any() or rows["text"].map(normalize).duplicated().any():
        raise AssertionError("duplicate synthetic IDs or normalized texts")
    if not rows.is_synthetic.all() or not rows.model_eligible.all():
        raise AssertionError("model corpus contains non-synthetic or ineligible rows")
    if set(rows.reference_stance) != set(LABELS):
        raise AssertionError("model corpus contains labels outside the three-class schema")
    if not rows.generation_method.eq(GENERATION_METHOD).all() or rows.random_seed.isna().any():
        raise AssertionError("generation provenance is incomplete")
    if rows[rows.reference_stance == "Unclear"].synthetic_reason.isin(
        ["genuine_ambiguity", "language_comprehension_difficulty"]
    ).any():
        raise AssertionError("ambiguous or protocol-only examples entered model training corpus")
    # Guard against the defective v4 pattern: a positive and a direct rejection in one text.
    positive_terms = {"support", "welcome", "agree", "endorse", "favour"}
    negative_terms = {"oppose", "against", "reject", "cannot", "disagree", "unacceptable"}
    risky = rows[rows.reference_stance == "Unclear"].text.map(
        lambda text: bool(set(re.findall(r"[a-z']+", str(text).casefold())) & positive_terms)
        and bool(set(re.findall(r"[a-z']+", str(text).casefold())) & negative_terms)
    )
    if risky.any():
        raise AssertionError(f"Unclear rows contain explicit conflicting positions: {rows.loc[risky.index[risky], 'synthetic_id'].tolist()}")
    counts = rows.reference_stance.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    if any(count != expected_per_class for count in counts.values()):
        raise AssertionError(f"class counts are not balanced: {counts}")
    return {"rows": int(len(rows)), "class_counts": counts,
            "unique_texts": int(rows.text.map(normalize).nunique()), "duplicate_texts": 0,
            "unclear_reasons": rows[rows.reference_stance == "Unclear"].synthetic_reason.value_counts().to_dict(),
            "disabled_v4_templates": DISABLED_V4_TEMPLATES}


def generate_model_corpus(per_class: int, seed: int) -> pd.DataFrame:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    slots = {**SLOTS, **SLOT_EXTENSIONS}
    banks = {"Support": TEMPLATES["Support"], "Neutral": TEMPLATES["Neutral"],
             "Unclear": [item for group in UNCLEAR_TEMPLATES.values() for item in group]}
    reason_by_prefix = {
        "unclear_context_v2_": "insufficient_context",
        "unclear_fragment_v2_": "fragment",
        "unclear_off_topic_v2_": "off_topic",
    }
    rows, used = [], set()
    for label in LABELS:
        accepted = 0
        attempts = 0
        bank = banks[label]
        while accepted < per_class and attempts < per_class * 500:
            attempts += 1
            template_id, language, template = rng.choice(bank)
            text = template.format(**{key: rng.choice(values) for key, values in slots.items()})
            if label == "Support":
                text = f"{text} {rng.choice(SUPPORT_SUFFIXES[language])}"
            elif label == "Neutral":
                text = f"{text} {rng.choice(NEUTRAL_SUFFIXES[language])}"
            normalized = normalize(text)
            if not normalized or normalized in used:
                continue
            used.add(normalized)
            accepted += 1
            reason = "explicit_stance" if label == "Support" else "no_stance_established" if label == "Neutral" else next(
                value for prefix, value in reason_by_prefix.items() if template_id.startswith(prefix)
            )
            rows.append({
                "synthetic_id": f"synthetic-v2-{seed}-{len(rows) + 1:06d}",
                "text": text,
                "reference_stance": label,
                "is_synthetic": True,
                "model_eligible": True,
                "generation_method": GENERATION_METHOD,
                "template_id": template_id,
                "language": language,
                "random_seed": int(seed),
                "synthetic_reason": reason,
            })
        if accepted != per_class:
            raise RuntimeError(f"Could not generate {per_class} unique examples for {label}; generated {accepted}")
    result = pd.DataFrame(rows)
    quality_check(result, per_class)
    return result


def generate_protocol_rows(count: int, seed: int) -> pd.DataFrame:
    if count > len(PROTOCOL_TEMPLATES):
        raise ValueError(f"protocol-count must be at most {len(PROTOCOL_TEMPLATES)} to avoid duplicate protocol rows")
    rows = []
    for index in range(count):
        template_id, text = PROTOCOL_TEMPLATES[index]
        rows.append({
            "synthetic_id": f"protocol-v2-{seed}-{index + 1:05d}",
            "text": text,
            "is_synthetic": True,
            "model_eligible": False,
            "generation_method": GENERATION_METHOD,
            "template_id": template_id,
            "language": "fr",
            "random_seed": int(seed),
            "scenario": "annotator_unable_to_reliably_read_French",
            "protocol_condition": PROTOCOL_CONDITION,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--protocol-count", type=int, default=8)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    corpus = generate_model_corpus(args.per_class, args.seed)
    protocol = generate_protocol_rows(args.protocol_count, args.seed)
    quality = quality_check(corpus, args.per_class)
    v4_rows = pd.read_json(ROOT / "reports/stance_synthetic_augmentation_v4/synthetic_training_rows.jsonl", lines=True)
    flagged = v4_rows[v4_rows.template_id.isin(DISABLED_V4_TEMPLATES)].copy()
    flagged["review_note"] = flagged.template_id.map(DISABLED_V4_TEMPLATES)
    flagged["training_action"] = "disabled_from_v2_model_training"
    flagged_template_ids = {"unclear_ambiguity_1", "unclear_ambiguity_2", "unclear_ambiguity_4",
                            "unclear_language_fr_1", "unclear_language_fr_2", "unclear_language_fr_3", "unclear_language_fr_4"}
    flagged["audit_status"] = flagged.template_id.map(
        lambda template_id: "flagged_in_v4_audit" if template_id in flagged_template_ids else "precautionarily_disabled"
    )
    if int(flagged.audit_status.eq("flagged_in_v4_audit").sum()) != 28:
        raise AssertionError("the new review must include every one of the 28 v4 flagged rows")
    corpus.to_json(out_dir / "synthetic_training_rows.jsonl", orient="records", lines=True, force_ascii=False)
    protocol.to_json(out_dir / "language_protocol_examples.jsonl", orient="records", lines=True, force_ascii=False)
    flagged.to_json(out_dir / "flagged_v4_rows_review.jsonl", orient="records", lines=True, force_ascii=False)
    (out_dir / "quality_checks.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# Synthetic stance corpus v2: quality-gated generation", "",
        "The v4 corpus remains unchanged. This version disables unclear templates with Mixed-label risk and excludes annotator-language-comprehension examples from model training.", "",
        f"Generated {len(corpus)} model-eligible rows ({args.per_class} each for Support, Neutral and Unclear) with seed {args.seed}. Exact normalized duplicate count: 0.",
        f"Generated {len(protocol)} separate French protocol scenarios in `language_protocol_examples.jsonl`. They have no `reference_stance`, `model_eligible=false`, and are never included in model training.", "",
        "Unclear training rows cover only insufficient context, fragments, and off-topic text. Genuine ambiguity templates are disabled until examples can be supplied with explicit, guideline-grounded readings; language-comprehension cases are conditional on annotator ability.", "",
        "| Class | Rows | Unique texts | Subtypes |", "|---|---:|---:|---|",
    ]
    for label, group in corpus.groupby("reference_stance"):
        report.append(f"| {label} | {len(group)} | {group.text.nunique()} | {', '.join(sorted(group.synthetic_reason.unique()))} |")
    report += ["", "## Disabled v4 templates", ""]
    report += [f"- `{template_id}`: {reason}" for template_id, reason in DISABLED_V4_TEMPLATES.items()]
    report += ["", "## Row-by-row review of all disabled v4 examples", ""]
    report += [f"- `{row.synthetic_id}` / `{row.template_id}` / {row.reference_stance} ({row.synthetic_reason}; {row.audit_status}): “{row.text}” — {row.review_note}"
               for row in flagged.itertuples()]
    report += ["", "Automated quality checks require complete provenance, the three-class schema, balanced configured counts, unique normalized text, and no explicit Support/Oppose contradiction among Unclear rows. Original reference labels and source mappings were not read or modified by corpus generation."]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "quality": quality,
                      "protocol_only_rows": len(protocol), "protocol_has_stance_labels": "reference_stance" in protocol}, ensure_ascii=False))


if __name__ == "__main__":
    main()
