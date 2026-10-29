import csv
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_ROOT / "data" / "interim" / "review"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_review_files_exist():
    expected_files = [
        REVIEW_DIR / "exact_duplicate_text_review.csv",
        REVIEW_DIR / "duplicate_record_review.csv",
        REVIEW_DIR / "near_duplicate_review.csv",
        REVIEW_DIR / "cross_dataset_row_id_overlap_review.csv",
        REPORTS_DIR / "duplicate_overlap_review_report.md",
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"File does not exist: {file_path}"
        assert file_path.stat().st_size > 0, f"File is empty: {file_path}"


def test_review_files_row_counts_and_blank_decision_fields():
    expected_counts = {
        "exact_duplicate_text_review.csv": 596,
        "duplicate_record_review.csv": 14,
        "near_duplicate_review.csv": 19,
        "cross_dataset_row_id_overlap_review.csv": 145,
    }
    review_columns = ["review_status", "review_decision", "review_notes", "reviewer"]

    for filename, expected_count in expected_counts.items():
        file_path = REVIEW_DIR / filename
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        assert len(df) == expected_count, f"{filename} row count mismatch: {len(df)} != {expected_count}"

        for col in review_columns:
            assert col in df.columns, f"{filename} missing review column: {col}"
            non_blank = df[df[col].str.strip() != ""]
            assert len(non_blank) == 0, f"{filename} has non-blank values in {col}: {len(non_blank)} rows"


def test_cross_dataset_overlap_evidence_categories():
    file_path = REVIEW_DIR / "cross_dataset_row_id_overlap_review.csv"
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    # 144 unrelated, 1 ambiguous
    category_counts = df["evidence_category"].value_counts().to_dict()
    assert category_counts.get("unrelated_different_event_and_text", 0) == 144
    assert category_counts.get("ambiguous_or_partial_similarity", 0) == 1
    assert category_counts.get("confirmed_same_record", 0) == 0

    # Ensure 0 pairs match on event or collection source
    assert (df["event_same"] == "True").sum() == 0
    assert (df["source_same"] == "True").sum() == 0

    # Ensure max text similarity does not exceed 0.55
    similarities = df["best_text_similarity"].astype(float)
    assert similarities.max() < 0.55
    assert similarities.min() >= 0.0


def test_duplicate_record_review_content():
    file_path = REVIEW_DIR / "duplicate_record_review.csv"
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    assert len(df) == 14
    assert (df["source_dataset"] == "individual_comments_2016_18").all()
    assert (df["duplicate_type"] == "exact_duplicate_record").all()
    # All 14 rows are trailing blanks preserved from the raw spreadsheet
    assert df["row_rang_e"].str.strip().eq("").all()


def test_exact_duplicate_text_original_text_preserved():
    file_path = REVIEW_DIR / "exact_duplicate_text_review.csv"
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    assert len(df) == 596
    # Original text must not be empty for any entry
    empty_original = df[df["original_text"].str.strip() == ""]
    assert len(empty_original) == 0, f"Found {len(empty_original)} empty original_text values in exact review"


def test_processed_cleaned_datasets_preserved():
    # Verify that clean datasets exist and have not been modified or truncated
    clean_files = {
        "consultation_compilation_2017_18_source_clean.csv": 5169,
        "get_involved_questionnaire_2017_18_source_clean.csv": 8212,
        "individual_comments_2016_18_source_clean.csv": 160,
    }
    for filename, expected_len in clean_files.items():
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Missing processed file: {path}"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert len(df) == expected_len, f"{filename} row count mismatch: {len(df)} != {expected_len}"
