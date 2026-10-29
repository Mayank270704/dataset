"""
scripts/create_annotation_pilot.py
====================================
Generates data/annotation/pilot_annotation_50.csv — a small, stratified random sample
of policy-relevant free-text responses suitable for manual annotation.

Sampling rules (see reports/annotation_guidelines_v1.md Section 5 for full rationale):
  1. Exclude confirmed all-blank records (row_rang_e empty for individual_comments).
  2. Exclude records with a non-empty exclusion_reason.
  3. Require original_text length >= 20 characters after strip.
  4. Exclude generic-filler normalized text.
  5. When a duplicate_text_group_id is present, keep only the first occurrence
     by source_row_number (ascending) — prevents leakage of identical text.
  6. Avoid selecting the same underlying record more than once.
  7. Stratify by question_slug (individual/questionnaire) or theme (compilation).
  8. Draw random.seed(42) samples without replacement: 20 / 20 / 10 records per dataset.

Source files are NOT modified. Blank human annotation columns are appended.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ANNOTATION = ROOT / "data" / "annotation"
ANNOTATION.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ANNOTATION / "pilot_annotation_50.csv"
LOG_PATH = ANNOTATION / "pilot_sampling_log.md"

# ── Sampling parameters ────────────────────────────────────────────────────────
RANDOM_SEED = 42
MIN_TEXT_LEN = 20
TARGETS = {
    "individual_comments_2016_18": 20,
    "consultation_compilation_2017_18": 20,
    "get_involved_questionnaire_2017_18": 10,
}

GENERIC_FILLERS = frozenset({
    "none", "n/a", "na", ".", "thanks", "thank you",
    "no comment", "all of the above", "same as above",
    "non", "aucun", "aucune",
})

# ── Question metadata ──────────────────────────────────────────────────────────
INDIVIDUAL_FIELDS = [
    (
        "public_engagement_question",
        "public_engagement_question_also_in_compilation_question_sur_la_mobilisation_du",
        "Public Engagement Question (Also in Compilation) / Question sur la mobilisation du",
    ),
    (
        "practical_idea",
        "1_practical_idea_also_in_compilation_une_id_e_pratique_toujours_dans_compilation",
        "1 Practical idea (Also in Compilation) / Une idée pratique (Toujours dans Compilation)",
    ),
    (
        "og_is_important_because",
        "og_is_important_because_il_est_important_d_avoir_un_gouvernement_ouvert_parce_que",
        "OG is important because / Il est important d'avoir un gouvernement ouvert parce que",
    ),
    (
        "improvements",
        "improvements_am_liorations",
        "Improvements? / Améliorations",
    ),
    (
        "other_comments",
        "other_comments_autres_commentaires",
        "Other comments / Autres commentaires",
    ),
]

COMPILATION_TEXT_COL = (
    "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part"
)
COMPILATION_CTX_COL = "context_prompt_contexte_message_guide"
COMPILATION_THEME_COL = "theme_th_me"

QUESTIONNAIRE_FIELDS = [
    (
        "q2r_other",
        "q2r_other_please_specify",
        "Q2R. Other (please specify) — What are your three most important government transparency topics?",
    ),
    (
        "q4r_other",
        "q4r_other_please_specify_6",
        "Q4R. Other (please specify) — What are your three most important government accountability needs?",
    ),
    (
        "q6r_other",
        "q6r_other_please_specify",
        "Q6R. Other (please specify) — What are the three most important ways to participate and interact with government?",
    ),
    (
        "q8",
        "q8_did_we_miss_anything_if_the_previous_questions_did_not_speak_to_you_please_tell_us_about_the_important_aspects_of_open_government_that_drive_your_interest_if_you_have_more_than_one_additional_item_please_list_them_in_priority",
        "Q8. Did we miss anything? If the previous questions did not speak to you, please tell us about the important aspects of open government that drive your interest.",
    ),
    (
        "q14",
        "q14_please_tell_us_why",
        "Q14. Please tell us why (regarding satisfaction with open.canada.ca):",
    ),
    (
        "q18",
        "q18_if_you_have_knowledge_or_experience_with_a_open_government_topic_that_you_would_be_willing_to_share_please_tell_us_about_it",
        "Q18. If you have knowledge or experience with an open government topic that you would be willing to share, please tell us about it:",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def is_generic_filler(text: str) -> bool:
    return normalize_text(text) in GENERIC_FILLERS


def is_eligible(text: str, exclusion_reason: str) -> bool:
    """Return True when a text response passes all eligibility filters."""
    t = text.strip()
    if not t:
        return False
    if len(t) < MIN_TEXT_LEN:
        return False
    if exclusion_reason.strip():
        return False
    if is_generic_filler(t):
        return False
    return True


def deduplicate_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first occurrence (by source_row_number) per duplicate group."""
    df = df.copy()
    df["source_row_number"] = pd.to_numeric(df["source_row_number"], errors="coerce")
    no_group = df[df["duplicate_text_group_id"].str.strip() == ""]
    has_group = df[df["duplicate_text_group_id"].str.strip() != ""]
    if not has_group.empty:
        first_per_group = (
            has_group.sort_values("source_row_number")
            .groupby("duplicate_text_group_id", sort=False)
            .first()
            .reset_index()
        )
        df = pd.concat([no_group, first_per_group], ignore_index=True)
    return df


def stratified_sample_unique_records(
    records: list[dict],
    stratum_key: str,
    n: int,
    rng: random.Random,
    record_id_key: str = "_source_row_number",
) -> list[dict]:
    """
    Draw at most `n` records using stratified sampling on `stratum_key`,
    guaranteeing that no record_id_key (underlying source record) is selected more than once.
    """
    if not records:
        return []
    strata: dict[str, list[dict]] = {}
    for rec in records:
        k = rec.get(stratum_key, "other") or "other"
        strata.setdefault(k, []).append(rec)

    for pool in strata.values():
        rng.shuffle(pool)

    n_strata = len(strata)
    quota = max(1, n // n_strata)
    selected: list[dict] = []
    used_ids: set[str] = set()

    # Pass 1: pick up to quota from each stratum
    for stratum, pool in strata.items():
        picked_for_stratum = 0
        for rec in pool:
            rid = str(rec[record_id_key])
            if rid not in used_ids:
                used_ids.add(rid)
                selected.append(rec)
                picked_for_stratum += 1
                if picked_for_stratum >= quota:
                    break

    # Pass 2: fill remaining budget from any stratum
    remaining = n - len(selected)
    if remaining > 0:
        leftover_pool: list[dict] = []
        for pool in strata.values():
            for rec in pool:
                if str(rec[record_id_key]) not in used_ids:
                    leftover_pool.append(rec)
        rng.shuffle(leftover_pool)
        for rec in leftover_pool:
            rid = str(rec[record_id_key])
            if rid not in used_ids:
                used_ids.add(rid)
                selected.append(rec)
                if len(selected) >= n:
                    break

    return selected[:n]


# ── Dataset-specific samplers ──────────────────────────────────────────────────

def sample_individual_comments(
    target: int, rng: random.Random, log: list[str]
) -> list[dict]:
    df = pd.read_csv(
        PROCESSED / "individual_comments_2016_18_source_clean.csv",
        dtype=str,
        keep_default_na=False,
    )
    # Exclude confirmed blank rows (trailing blank rows where row_rang_e is empty)
    df = df[df["row_rang_e"].str.strip() != ""].copy()
    df = deduplicate_by_group(df)

    log.append("### `individual_comments_2016_18`")
    log.append(f"- Non-blank rows after excluding trailing blanks: {len(df)}")

    candidates: list[dict] = []
    for slug, col, wording in INDIVIDUAL_FIELDS:
        if col not in df.columns:
            log.append(f"  - Column not found: {col}")
            continue
        for _, row in df.iterrows():
            text = row.get(col, "")
            excl = row.get("exclusion_reason", "")
            if not is_eligible(text, excl):
                continue
            candidates.append({
                "_stratum": slug,
                "_source_row_number": str(row.get("source_row_number", "")),
                "source_dataset": "individual_comments_2016_18",
                "source_file": row.get("source_file", ""),
                "source_row_number": str(row.get("source_row_number", "")),
                "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                "question_slug": slug,
                "question_wording": wording,
                "context": f"Event: {row.get('event', '')} | Sector: {row.get('sector', '')}",
                "original_text": text.strip(),
            })

    log.append(f"- Eligible candidates across all 5 fields: {len(candidates)}")
    sampled = stratified_sample_unique_records(candidates, "_stratum", target, rng)
    log.append(f"- Sampled: {len(sampled)} (distinct source records: {len(set(r['_source_row_number'] for r in sampled))})")
    for slug in [s for s, _, _ in INDIVIDUAL_FIELDS]:
        n = sum(1 for r in sampled if r["question_slug"] == slug)
        log.append(f"  - `{slug}`: {n}")
    return sampled


def sample_compilation(
    target: int, rng: random.Random, log: list[str]
) -> list[dict]:
    df = pd.read_csv(
        PROCESSED / "consultation_compilation_2017_18_source_clean.csv",
        dtype=str,
        keep_default_na=False,
    )
    df = deduplicate_by_group(df)

    log.append("### `consultation_compilation_2017_18`")
    log.append(f"- Rows after deduplication: {len(df)}")

    candidates: list[dict] = []
    # Exclude 'Not Applicable' and blank theme rows for stratification purposes
    eligible_themes = df[COMPILATION_THEME_COL].map(
        lambda t: t not in ("Not Applicable", "Blank", "")
    )

    for _, row in df[eligible_themes].iterrows():
        text = row.get(COMPILATION_TEXT_COL, "")
        excl = row.get("exclusion_reason", "")
        if not is_eligible(text, excl):
            continue
        candidates.append({
            "_stratum": row.get(COMPILATION_THEME_COL, "Other"),
            "_source_row_number": str(row.get("source_row_number", "")),
            "source_dataset": "consultation_compilation_2017_18",
            "source_file": row.get("source_file", ""),
            "source_row_number": str(row.get("source_row_number", "")),
            "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
            "question_slug": "comment_question_or_idea",
            "question_wording": (
                "Comment, Question or Idea (In source language) / "
                "Commentaire, question ou idée (dans la langue de départ)"
            ),
            "context": (
                f"Event: {row.get('event_v_nement', '')} | "
                f"Source: {row.get('source', '')} | "
                f"Prompt: {row.get(COMPILATION_CTX_COL, '')} | "
                f"Theme: {row.get(COMPILATION_THEME_COL, '')}"
            ),
            "original_text": text.strip(),
        })

    log.append(f"- Eligible candidates (themed, non-generic, len>=20): {len(candidates)}")
    sampled = stratified_sample_unique_records(candidates, "_stratum", target, rng)
    log.append(f"- Sampled: {len(sampled)} (distinct source records: {len(set(r['_source_row_number'] for r in sampled))})")
    theme_dist: dict[str, int] = {}
    for r in sampled:
        theme_dist[r["_stratum"]] = theme_dist.get(r["_stratum"], 0) + 1
    for theme, count in sorted(theme_dist.items()):
        log.append(f"  - {theme}: {count}")
    return sampled


def sample_questionnaire(
    target: int, rng: random.Random, log: list[str]
) -> list[dict]:
    df = pd.read_csv(
        PROCESSED / "get_involved_questionnaire_2017_18_source_clean.csv",
        dtype=str,
        keep_default_na=False,
    )
    df = deduplicate_by_group(df)

    log.append("### `get_involved_questionnaire_2017_18`")
    log.append(f"- Rows after deduplication: {len(df)}")

    selected_fields = [f for f in QUESTIONNAIRE_FIELDS if f[0] in ("q8", "q14", "q18")]

    candidates: list[dict] = []
    for slug, col, wording in selected_fields:
        if col not in df.columns:
            log.append(f"  - Column not found: {col}")
            continue
        for _, row in df.iterrows():
            text = row.get(col, "")
            excl = row.get("exclusion_reason", "")
            if not is_eligible(text, excl):
                continue
            candidates.append({
                "_stratum": slug,
                "_source_row_number": str(row.get("source_row_number", "")),
                "source_dataset": "get_involved_questionnaire_2017_18",
                "source_file": row.get("source_file", ""),
                "source_row_number": str(row.get("source_row_number", "")),
                "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                "question_slug": slug,
                "question_wording": wording,
                "context": (
                    f"Language: {row.get('language', '')} | "
                    f"Sector: {row.get('q19_the_sector_or_group_that_i_identify_most_closely_with_is', '')}"
                ),
                "original_text": text.strip(),
            })

    log.append(f"- Eligible candidates across Q8, Q14, Q18: {len(candidates)}")
    sampled = stratified_sample_unique_records(candidates, "_stratum", target, rng)
    log.append(f"- Sampled: {len(sampled)} (distinct source records: {len(set(r['_source_row_number'] for r in sampled))})")
    for slug in [s for s, _, _ in selected_fields]:
        n = sum(1 for r in sampled if r["question_slug"] == slug)
        log.append(f"  - `{slug}`: {n}")
    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RANDOM_SEED)
    log_lines: list[str] = [
        "# Pilot Sampling Log",
        "",
        f"Random seed: `{RANDOM_SEED}`",
        f"Min text length: `{MIN_TEXT_LEN}`",
        f"Target per dataset: {TARGETS}",
        "",
        "## Sampling Details",
        "",
    ]

    # Collect samples from each dataset
    individual_records = sample_individual_comments(TARGETS["individual_comments_2016_18"], rng, log_lines)
    compilation_records = sample_compilation(TARGETS["consultation_compilation_2017_18"], rng, log_lines)
    questionnaire_records = sample_questionnaire(TARGETS["get_involved_questionnaire_2017_18"], rng, log_lines)

    all_records = individual_records + compilation_records + questionnaire_records

    # Assign pilot IDs (sort by dataset then source row number for deterministic ordering)
    all_records.sort(key=lambda r: (r["source_dataset"], int(r["source_row_number"]) if r["source_row_number"].isdigit() else r["source_row_number"]))

    ANNOTATION_COLUMNS = [
        "stance",
        "concern",
        "requested_action",
        "evidence_span",
        "review_notes",
        "annotator_id",
    ]

    rows = []
    for i, rec in enumerate(all_records, start=1):
        output_row = {
            "pilot_id": f"pilot_{i:03d}",
            "source_dataset": rec["source_dataset"],
            "source_record_id": rec["source_row_number"],
            "source_file": rec["source_file"],
            "source_row_number": rec["source_row_number"],
            "source_field": rec["question_slug"],
            "question_slug": rec["question_slug"],
            "question_wording": rec["question_wording"],
            "question_context": rec["context"],
            "context": rec["context"],
            "duplicate_text_group_id": rec["duplicate_text_group_id"],
            "original_text": rec["original_text"],
        }
        for col in ANNOTATION_COLUMNS:
            output_row[col] = ""
        rows.append(output_row)

    pilot_df = pd.DataFrame(rows)
    pilot_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {len(pilot_df)} records to {OUTPUT_PATH.relative_to(ROOT).as_posix()}")

    # Dataset breakdown summary
    log_lines.append("")
    log_lines.append("## Final Pilot CSV Summary")
    log_lines.append("")
    log_lines.append(f"Total records: **{len(pilot_df)}**")
    log_lines.append("")
    log_lines.append("| source_dataset | n |")
    log_lines.append("| --- | --- |")
    for ds, grp in pilot_df.groupby("source_dataset"):
        log_lines.append(f"| {ds} | {len(grp)} |")
    log_lines.append("")
    log_lines.append("| question_slug | n |")
    log_lines.append("| --- | --- |")
    for slug, grp in pilot_df.groupby("question_slug"):
        log_lines.append(f"| {slug} | {len(grp)} |")
    log_lines.append("")
    log_lines.append(
        "All annotation columns (`stance`, `concern`, `requested_action`, "
        "`evidence_span`, `review_notes`, `annotator_id`) are blank strings."
    )
    log_lines.append("")
    log_lines.append("## Column Provenance Mapping")
    log_lines.append("")
    log_lines.append("| Pilot Column | Standard Provenance Name | Description |")
    log_lines.append("| --- | --- | --- |")
    log_lines.append("| `pilot_id` | `pilot_id` | Unique pilot identifier (`pilot_001` .. `pilot_050`) |")
    log_lines.append("| `source_dataset` | `source_dataset` | Processed dataset identifier |")
    log_lines.append("| `source_record_id` | `source_record_id` | Source row identifier (alias of `source_row_number`) |")
    log_lines.append("| `source_file` | `source_file` | Original source filename |")
    log_lines.append("| `source_row_number` | `source_record_id` | 1-indexed source row number in clean dataset |")
    log_lines.append("| `source_field` | `source_field` | Field/question slug identifier (alias of `question_slug`) |")
    log_lines.append("| `question_slug` | `source_field` | Question/field identifier |")
    log_lines.append("| `question_wording` | `question_wording` | Original question text as presented to participants |")
    log_lines.append("| `question_context` | `question_context` | Metadata context: event, sector, prompt, theme |")
    log_lines.append("| `context` | `question_context` | Context string (alias of `question_context`) |")
    log_lines.append("| `duplicate_text_group_id` | `duplicate_group` | Duplicate review group identifier (if any) |")
    log_lines.append("| `original_text` | `original_text` | Verbatim response text without normalization |")
    log_lines.append("")
    log_lines.append("## Exclusion Rationale & Research Integrity")
    log_lines.append("")
    log_lines.append(
        "- 14 trailing all-blank rows from `individual_comments_2016_18` "
        "were excluded (confirmed by duplicate-review notebook). "
        "They remain in `data/processed/individual_comments_2016_18_source_clean.csv` "
        "as audit artefacts and were NOT deleted."
    )
    log_lines.append(
        "- Generic filler responses (e.g. `none`, `n/a`, `.`, `thanks`) were excluded "
        "from sampling eligibility. They may appear in the pilot only if they survive "
        "the >= 20 character filter, which in practice they do not."
    )
    log_lines.append(
        "- When a `duplicate_text_group_id` was present, only the first occurrence "
        "(lowest `source_row_number`) was eligible to prevent label leakage."
    )
    log_lines.append(
        "- Underlying source record deduplication: within each source dataset, "
        "each `source_row_number` was selected at most once. No respondent record "
        "is sampled multiple times across different question fields."
    )
    log_lines.append(
        "- Selected Source Row Numbers:"
    )
    for ds, grp in pilot_df.groupby("source_dataset"):
        rows_str = ", ".join(grp["source_row_number"].tolist())
        log_lines.append(f"  - `{ds}` ({len(grp)} records): {rows_str}")
    log_lines.append("")
    log_lines.append("## Provenance & Licensing Note")
    log_lines.append("")
    log_lines.append(
        "Source datasets are Canadian Open Government consultation datasets (2016–2018) "
        "published by the Treasury Board of Canada Secretariat / Government of Canada under the "
        "Open Government Licence – Canada. Unresolved questions regarding specific third-party submissions "
        "or external social media extracts (e.g., Reddit, Twitter) are logged for manual review."
    )

    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote sampling log to {LOG_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
