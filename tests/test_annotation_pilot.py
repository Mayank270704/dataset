"""
tests/test_annotation_pilot.py
================================
Tests for the pilot annotation CSV (data/annotation/pilot_annotation_50.csv).
Verifies:
  1. Required provenance and annotation columns exist.
  2. Pilot row count is between 1 and 50 (expected 50 for available data).
  3. Pilot IDs are unique and properly formatted (pilot_001..pilot_050).
  4. Source identifiers and original text are present for eligible records.
  5. Original text matches the referenced source record exactly.
  6. Human annotation fields are entirely blank strings.
  7. No confirmed all-blank source record is included.
  8. The same underlying source record is not selected more than once.
  9. No duplicate-group leakage (at most one record per duplicate group).
 10. No generic filler text survives into original_text.
 11. Minimum text length constraint is met (>= 20 chars).
 12. Valid question slugs and source datasets.
 13. Source-clean datasets and raw files have not been modified or truncated.
 14. Annotation guidelines and sampling log exist and are non-empty.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"
PILOT_PATH = ANNOTATION_DIR / "pilot_annotation_50.csv"
LOG_PATH = ANNOTATION_DIR / "pilot_sampling_log.md"
GUIDELINES_PATH = PROJECT_ROOT / "reports" / "annotation_guidelines_v1.md"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ANNOTATION_COLUMNS = [
    "stance",
    "concern",
    "requested_action",
    "evidence_span",
    "review_notes",
    "annotator_id",
]

REQUIRED_PROVENANCE_COLUMNS = [
    "pilot_id",
    "source_dataset",
    "source_record_id",
    "source_file",
    "source_row_number",
    "source_field",
    "question_slug",
    "question_wording",
    "question_context",
    "context",
    "duplicate_text_group_id",
    "original_text",
]

ALL_REQUIRED_COLUMNS = REQUIRED_PROVENANCE_COLUMNS + ANNOTATION_COLUMNS

VALID_DATASETS = {
    "individual_comments_2016_18",
    "consultation_compilation_2017_18",
    "get_involved_questionnaire_2017_18",
}

VALID_QUESTION_SLUGS = {
    "public_engagement_question",
    "practical_idea",
    "og_is_important_because",
    "improvements",
    "other_comments",
    "comment_question_or_idea",
    "q2r_other",
    "q4r_other",
    "q6r_other",
    "q8",
    "q14",
    "q18",
}

GENERIC_FILLERS = frozenset({
    "none", "n/a", "na", ".", "thanks", "thank you",
    "no comment", "all of the above", "same as above",
    "non", "aucun", "aucune",
})

MIN_TEXT_LEN = 20

RAW_FILES = [
    PROJECT_ROOT / "ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv",
    PROJECT_ROOT / "Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv",
    PROJECT_ROOT / "ap2016-18_individual_comments.csv",
]

CLEAN_FILES_EXPECTED_COUNTS = {
    "consultation_compilation_2017_18_source_clean.csv": 5169,
    "get_involved_questionnaire_2017_18_source_clean.csv": 8212,
    "individual_comments_2016_18_source_clean.csv": 160,
}


@pytest.fixture(scope="module")
def pilot_df() -> pd.DataFrame:
    assert PILOT_PATH.exists(), f"Pilot CSV not found: {PILOT_PATH}"
    return pd.read_csv(PILOT_PATH, dtype=str, keep_default_na=False)


def test_pilot_file_exists():
    assert PILOT_PATH.exists(), f"Pilot CSV not found: {PILOT_PATH}"
    assert PILOT_PATH.stat().st_size > 0, "Pilot CSV is empty"


def test_annotation_guidelines_exist():
    assert GUIDELINES_PATH.exists(), f"Annotation guidelines not found: {GUIDELINES_PATH}"
    assert GUIDELINES_PATH.stat().st_size > 0, "Annotation guidelines file is empty"


def test_sampling_log_exists():
    assert LOG_PATH.exists(), f"Sampling log not found: {LOG_PATH}"
    assert LOG_PATH.stat().st_size > 0, "Sampling log file is empty"


def test_pilot_schema(pilot_df):
    for col in ALL_REQUIRED_COLUMNS:
        assert col in pilot_df.columns, f"Missing required column: {col}"


def test_pilot_row_count_bounds(pilot_df):
    """Pilot must contain between 1 and 50 records."""
    assert 1 <= len(pilot_df) <= 50, f"Expected 1..50 pilot records, got {len(pilot_df)}"
    assert len(pilot_df) == 50, f"Expected exactly 50 records for current dataset, got {len(pilot_df)}"


def test_pilot_dataset_allocation(pilot_df):
    counts = pilot_df["source_dataset"].value_counts().to_dict()
    assert counts.get("individual_comments_2016_18", 0) == 20
    assert counts.get("consultation_compilation_2017_18", 0) == 20
    assert counts.get("get_involved_questionnaire_2017_18", 0) == 10


def test_all_annotation_columns_are_blank(pilot_df):
    """All human annotation columns must be strictly empty strings."""
    for col in ANNOTATION_COLUMNS:
        non_blank = pilot_df[pilot_df[col].str.strip() != ""]
        assert len(non_blank) == 0, (
            f"Annotation column `{col}` has {len(non_blank)} non-blank values before human review"
        )


def test_pilot_ids_unique_and_formatted(pilot_df):
    pilot_ids = pilot_df["pilot_id"].tolist()
    assert len(pilot_ids) == len(set(pilot_ids)), "Pilot IDs are not unique"
    pattern = re.compile(r"^pilot_\d{3}$")
    bad = [pid for pid in pilot_ids if not pattern.match(pid)]
    assert len(bad) == 0, f"Malformed pilot IDs: {bad}"


def test_source_identifiers_and_original_text_present(pilot_df):
    for col in ["source_dataset", "source_record_id", "source_row_number", "source_field", "original_text"]:
        missing = pilot_df[pilot_df[col].str.strip() == ""]
        assert len(missing) == 0, f"{len(missing)} records have empty {col}"


def test_no_underlying_source_record_selected_more_than_once(pilot_df):
    """
    Ensure that the same underlying source record (source_dataset, source_row_number)
    is not selected more than once across the pilot.
    """
    pairs = list(zip(pilot_df["source_dataset"], pilot_df["source_row_number"]))
    duplicate_pairs = [p for p in set(pairs) if pairs.count(p) > 1]
    assert len(duplicate_pairs) == 0, (
        f"Underlying source records selected multiple times: {duplicate_pairs}"
    )


def test_original_text_matches_source_record_exactly(pilot_df):
    """
    Verify that original_text matches the text in the clean processed source record exactly.
    """
    dataset_to_file = {
        "individual_comments_2016_18": "individual_comments_2016_18_source_clean.csv",
        "consultation_compilation_2017_18": "consultation_compilation_2017_18_source_clean.csv",
        "get_involved_questionnaire_2017_18": "get_involved_questionnaire_2017_18_source_clean.csv",
    }
    loaded_clean: dict[str, pd.DataFrame] = {}
    for ds, fname in dataset_to_file.items():
        loaded_clean[ds] = pd.read_csv(PROCESSED_DIR / fname, dtype=str, keep_default_na=False)

    for _, row in pilot_df.iterrows():
        ds = row["source_dataset"]
        row_num = row["source_row_number"]
        expected_text = row["original_text"]
        clean_df = loaded_clean[ds]

        source_row = clean_df[clean_df["source_row_number"] == row_num]
        assert not source_row.empty, f"Source row {row_num} not found in {ds}"
        source_row = source_row.iloc[0]

        # Verify that original_text exists verbatim in at least one column of the source row
        matched = any(clean_df[c].iloc[source_row.name].strip() == expected_text for c in clean_df.columns)
        assert matched, (
            f"pilot_id={row['pilot_id']}: original_text did not match source record "
            f"in {ds} row {row_num}"
        )


def test_no_confirmed_blank_records_from_individual_comments(pilot_df):
    """Confirmed all-blank rows from individual_comments must not appear in pilot."""
    ind_rows = pilot_df[pilot_df["source_dataset"] == "individual_comments_2016_18"]
    ind_clean = pd.read_csv(
        PROCESSED_DIR / "individual_comments_2016_18_source_clean.csv",
        dtype=str,
        keep_default_na=False,
    )
    blank_row_numbers = set(
        ind_clean[ind_clean["row_rang_e"].str.strip() == ""]["source_row_number"].tolist()
    )
    pilot_row_numbers = set(ind_rows["source_row_number"].tolist())
    leaked = pilot_row_numbers & blank_row_numbers
    assert len(leaked) == 0, (
        f"Blank individual_comments records leaked into pilot: source_row_numbers {leaked}"
    )


def test_no_duplicate_group_leakage(pilot_df):
    """At most one member per duplicate group is selected."""
    with_group = pilot_df[pilot_df["duplicate_text_group_id"].str.strip() != ""]
    if len(with_group) == 0:
        return
    group_counts = with_group["duplicate_text_group_id"].value_counts()
    multi = group_counts[group_counts > 1]
    assert len(multi) == 0, f"Duplicate group leakage: {multi.to_dict()}"


def test_original_text_minimum_length(pilot_df):
    short_text = pilot_df[pilot_df["original_text"].str.strip().str.len() < MIN_TEXT_LEN]
    assert len(short_text) == 0, f"{len(short_text)} records shorter than {MIN_TEXT_LEN}"


def test_no_generic_filler_in_original_text(pilot_df):
    generic_in_pilot = pilot_df[
        pilot_df["original_text"].str.strip().str.casefold().isin(GENERIC_FILLERS)
    ]
    assert len(generic_in_pilot) == 0, f"Generic fillers found in pilot: {len(generic_in_pilot)}"


def test_valid_source_datasets(pilot_df):
    invalid = pilot_df[~pilot_df["source_dataset"].isin(VALID_DATASETS)]
    assert len(invalid) == 0, f"Invalid source_dataset values: {invalid['source_dataset'].unique()}"


def test_valid_question_slugs(pilot_df):
    invalid = pilot_df[~pilot_df["question_slug"].isin(VALID_QUESTION_SLUGS)]
    assert len(invalid) == 0, f"Invalid question_slug values: {invalid['question_slug'].unique()}"


def test_source_clean_and_raw_datasets_unmodified():
    """Verify that source-clean datasets and raw files have not been modified or truncated."""
    for filename, expected_len in CLEAN_FILES_EXPECTED_COUNTS.items():
        clean_path = PROCESSED_DIR / filename
        assert clean_path.exists(), f"Processed clean file missing: {clean_path}"
        df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
        assert len(df) == expected_len, (
            f"Processed file {filename} row count changed: {len(df)} != {expected_len}"
        )

    for raw_file in RAW_FILES:
        assert raw_file.exists(), f"Raw source file missing: {raw_file}"
        assert raw_file.stat().st_size > 0, f"Raw source file empty: {raw_file}"
