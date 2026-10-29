"""
tests/test_analyze_concern_review_agreement.py
===============================================
Unit tests for scripts/analyze_concern_review_agreement.py.

Covers:
  - No completed files present -> status no_completed_files
  - Valid full inputs -> status success, correct kappa/agreement
  - Invalid concern_relation labels -> validation_failed
  - Invalid confidence labels -> validation_failed
  - Missing pair IDs in one reviewer file -> validation_failed
  - Extra/unexpected pair IDs -> validation_failed
  - Duplicate pair IDs -> validation_failed
  - U ratings excluded from numeric kappa, counted separately
  - Blank ratings excluded and reported
  - Mixed U + numeric: kappa runs on numeric subset only
  - Output directory already exists -> output_exists (no overwrite)
  - compute_weighted_kappa: perfect agreement
  - compute_weighted_kappa: zero variance -> zero_variance status
  - compute_weighted_kappa: insufficient data (N<5) -> insufficient_data
  - compute_exact_agreement counts
  - validate_pair_ids: missing, extra, duplicate
  - validate_judgment_fields: invalid concern_relation and confidence
  - merge_reviewer_judgments: correct suffix assignment
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_concern_review_agreement import (
    MODELS,
    REQUIRED_COMPLETED_COLUMNS,
    VALID_CONCERN_RELATION,
    VALID_CONFIDENCE,
    build_stratum_summary,
    compute_exact_agreement,
    compute_weighted_kappa,
    merge_reviewer_judgments,
    run_analysis,
    validate_judgment_fields,
    validate_pair_ids,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _make_template(
    pair_ids: list[str],
    concern_relations: list[str],
    confidences: list[str] | None = None,
    reviewer_id: str = "R1",
) -> pd.DataFrame:
    """Return a minimal reviewer-template-style DataFrame."""
    n = len(pair_ids)
    conf = confidences if confidences is not None else ["high"] * n
    return pd.DataFrame(
        {
            "pair_id": pair_ids,
            "a_corpus_id": [f"A{i}" for i in range(n)],
            "a_language": ["English"] * n,
            "a_original_text": [f"comment_a_{i}" for i in range(n)],
            "a_source_dataset": ["ds"] * n,
            "a_source_file": ["f.csv"] * n,
            "a_source_record_id": [str(i) for i in range(n)],
            "a_source_row_number": [str(i) for i in range(n)],
            "a_source_field": ["text"] * n,
            "a_question_context": ["ctx"] * n,
            "b_corpus_id": [f"B{i}" for i in range(n)],
            "b_language": ["English"] * n,
            "b_original_text": [f"comment_b_{i}" for i in range(n)],
            "b_source_dataset": ["ds"] * n,
            "b_source_file": ["f.csv"] * n,
            "b_source_record_id": [str(i + n) for i in range(n)],
            "b_source_row_number": [str(i + n) for i in range(n)],
            "b_source_field": ["text"] * n,
            "b_question_context": ["ctx"] * n,
            "reviewer_id": [reviewer_id] * n,
            "concern_relation": concern_relations,
            "confidence": conf,
            "shared_concern_note": [""] * n,
            "evidence_note": [""] * n,
        }
    )


def _make_catalog(pair_ids: list[str]) -> pd.DataFrame:
    n = len(pair_ids)
    rows = {
        "pair_id": pair_ids,
        "a_corpus_id": [f"A{i}" for i in range(n)],
        "a_language": ["English"] * n,
        "a_original_text": [f"comment_a_{i}" for i in range(n)],
        "a_source_dataset": ["ds"] * n,
        "a_source_file": ["f.csv"] * n,
        "a_source_record_id": [str(i) for i in range(n)],
        "a_source_row_number": [str(i) for i in range(n)],
        "a_source_field": ["text"] * n,
        "a_question_context": ["ctx"] * n,
        "b_corpus_id": [f"B{i}" for i in range(n)],
        "b_language": ["English"] * n,
        "b_original_text": [f"comment_b_{i}" for i in range(n)],
        "b_source_dataset": ["ds"] * n,
        "b_source_file": ["f.csv"] * n,
        "b_source_record_id": [str(i + n) for i in range(n)],
        "b_source_row_number": [str(i + n) for i in range(n)],
        "b_source_field": ["text"] * n,
        "b_question_context": ["ctx"] * n,
    }
    for model in MODELS:
        rows[f"{model}_cluster_a"] = ["0"] * n
        rows[f"{model}_cluster_b"] = ["0"] * n
        rows[f"{model}_cluster_relation"] = ["within_cluster"] * n
        rows[f"{model}_selected"] = [True] * n
        rows[f"{model}_selection_relationship"] = ["within_cluster"] * n
        rows[f"{model}_selection_language_pair"] = ["English|English"] * n
        rows[f"{model}_selection_tier"] = ["likely_similar"] * n
        rows[f"{model}_selection_cosine"] = [0.9] * n
        rows[f"{model}_selection_rank_in_tier"] = [1] * n
        rows[f"{model}_selection_candidate_count"] = [10] * n
        rows[f"{model}_selection_rationale"] = ["test"] * n
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _setup_run(
    tmp_path: Path,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> dict:
    """Write files and call run_analysis with isolated tmp paths."""
    completed = tmp_path / "completed"
    completed.mkdir()
    _write_csv(r1_df, completed / "reviewer_1_completed.csv")
    _write_csv(r2_df, completed / "reviewer_2_completed.csv")
    catalog_path = tmp_path / "pair_catalog.csv"
    _write_csv(catalog_df, catalog_path)
    output_base = tmp_path / "agreement_analysis"
    return run_analysis(
        completed_dir=completed,
        catalog_path=catalog_path,
        output_base=output_base,
    )


# ── No completed files ────────────────────────────────────────────────────────


def test_no_completed_dir_returns_no_completed_files(tmp_path):
    result = run_analysis(
        completed_dir=tmp_path / "completed",
        catalog_path=tmp_path / "pair_catalog.csv",
        output_base=tmp_path / "out",
    )
    assert result["status"] == "no_completed_files"
    assert any("completed" in m for m in result["missing"])


def test_missing_reviewer_2_returns_no_completed_files(tmp_path):
    completed = tmp_path / "completed"
    completed.mkdir()
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(pair_ids, ["1"] * 6)
    _write_csv(r1, completed / "reviewer_1_completed.csv")
    result = run_analysis(
        completed_dir=completed,
        catalog_path=tmp_path / "pair_catalog.csv",
        output_base=tmp_path / "out",
    )
    assert result["status"] == "no_completed_files"


# ── Valid inputs ──────────────────────────────────────────────────────────────


def test_valid_inputs_return_success(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    r1 = _make_template(pair_ids, ["3", "2", "1", "0", "3", "2", "1", "0"])
    r2 = _make_template(pair_ids, ["3", "2", "1", "0", "2", "1", "0", "1"], reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    assert result["exact_agreement"]["total_pairs"] == 8
    assert result["exact_agreement"]["jointly_numeric"] == 8
    assert result["weighted_kappa"]["status"] == "computed"
    assert result["weighted_kappa"]["kappa"] is not None


def test_output_files_are_written(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    r1 = _make_template(pair_ids, ["1"] * 8)
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    out = Path(result["output_dir"])
    assert (out / "merged_judgments.csv").exists()
    assert (out / "agreement_report.md").exists()
    assert (out / "agreement_metrics.json").exists()
    assert (out / "stratum_summary.csv").exists()


def test_merged_judgments_has_correct_columns(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(pair_ids, ["1"] * 6)
    r2 = _make_template(pair_ids, ["2"] * 6, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    merged = pd.read_csv(Path(result["output_dir"]) / "merged_judgments.csv")
    assert "concern_relation_r1" in merged.columns
    assert "concern_relation_r2" in merged.columns
    assert "reviewer_id_r1" in merged.columns
    assert "reviewer_id_r2" in merged.columns
    assert merged["pair_id"].nunique() == 6


# ── Invalid concern_relation labels ──────────────────────────────────────────


def test_invalid_concern_relation_fails_validation(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(pair_ids, ["1", "X", "2", "3", "0", "1"])  # "X" is invalid
    r2 = _make_template(pair_ids, ["1"] * 6, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "validation_failed"
    assert any("invalid" in e.lower() for e in result["errors"])


def test_empty_concern_relation_alone_does_not_raise_invalid_label(tmp_path):
    """Blank ratings are allowed (excluded from kappa) — not a validation error."""
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    r1 = _make_template(pair_ids, ["1", "", "2", "3", "0", "1", "2", "3"])
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"


# ── Invalid confidence labels ─────────────────────────────────────────────────


def test_invalid_confidence_fails_validation(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(
        pair_ids, ["1"] * 6,
        confidences=["high", "medium", "CERTAIN", "low", "high", "medium"],
    )
    r2 = _make_template(pair_ids, ["1"] * 6, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "validation_failed"
    assert any("confidence" in e.lower() for e in result["errors"])


def test_blank_confidence_does_not_fail_validation(tmp_path):
    """Blank confidence is allowed (treated as unfilled, not invalid)."""
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    r1 = _make_template(pair_ids, ["1"] * 8, confidences=[""] * 8)
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"


# ── Pair ID errors ────────────────────────────────────────────────────────────


def test_missing_pair_ids_fails_validation(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    cat = _make_catalog(pair_ids)
    # r1 missing last pair
    r1 = _make_template(pair_ids[:-1], ["1"] * 7)
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "validation_failed"
    assert any("missing pair_ids" in e for e in result["errors"])


def test_extra_pair_ids_fails_validation(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    cat = _make_catalog(pair_ids)
    # r2 has an extra pair not in catalog
    r1 = _make_template(pair_ids, ["1"] * 8)
    r2 = _make_template(pair_ids + ["P9999"], ["1"] * 9, reviewer_id="R2")
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "validation_failed"
    assert any("unexpected pair_ids" in e for e in result["errors"])


def test_duplicate_pair_ids_fails_validation(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    cat = _make_catalog(pair_ids)
    # r1 has a duplicate
    dup_ids = pair_ids[:7] + [pair_ids[0]]
    r1 = _make_template(dup_ids, ["1"] * 8)
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "validation_failed"
    assert any("duplicate pair_ids" in e for e in result["errors"])


# ── U ratings ─────────────────────────────────────────────────────────────────


def test_u_ratings_excluded_from_kappa_counted_separately(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    # 2 pairs have U from r1, 1 pair has U from r2 (different pair), 5 numeric
    cr1 = ["U", "1", "2", "3", "0", "U", "2", "1"]
    cr2 = ["1", "U", "2", "3", "0", "1", "2", "1"]
    r1 = _make_template(pair_ids, cr1)
    r2 = _make_template(pair_ids, cr2, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    ea = result["exact_agreement"]
    # Pairs where BOTH are numeric: indices 2,3,4,6,7 = 5 pairs
    assert ea["jointly_numeric"] == 5
    assert ea["one_U"] == 3   # pairs 0 (r1=U, r2=1), 1 (r1=1, r2=U), 5 (r1=U, r2=1)
    assert ea["both_U"] == 0
    wk = result["weighted_kappa"]
    assert wk["status"] == "computed"
    assert wk["n"] == 5


def test_both_u_counted_in_both_u_field(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    cr1 = ["U", "1", "2", "3", "0", "1", "2", "1"]
    cr2 = ["U", "1", "2", "3", "0", "1", "2", "1"]
    r1 = _make_template(pair_ids, cr1)
    r2 = _make_template(pair_ids, cr2, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    assert result["exact_agreement"]["both_U"] == 1
    assert result["exact_agreement"]["jointly_numeric"] == 7


def test_all_u_ratings_triggers_insufficient_data(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(pair_ids, ["U"] * 6)
    r2 = _make_template(pair_ids, ["U"] * 6, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    assert result["weighted_kappa"]["status"] == "insufficient_data"
    assert result["weighted_kappa"]["kappa"] is None


# ── Missing/blank ratings ─────────────────────────────────────────────────────


def test_blank_ratings_excluded_and_reported(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    cr1 = ["", "1", "2", "3", "0", "1", "2", "1"]
    cr2 = ["1", "1", "2", "3", "0", "1", "2", "1"]
    r1 = _make_template(pair_ids, cr1)
    r2 = _make_template(pair_ids, cr2, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    result = _setup_run(tmp_path, r1, r2, cat)
    assert result["status"] == "success"
    ea = result["exact_agreement"]
    assert ea["either_blank"] == 1
    assert ea["jointly_numeric"] == 7


# ── Output exists guard ───────────────────────────────────────────────────────


def test_existing_output_dir_prevents_overwrite(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 9)]
    r1 = _make_template(pair_ids, ["1"] * 8)
    r2 = _make_template(pair_ids, ["1"] * 8, reviewer_id="R2")
    cat = _make_catalog(pair_ids)
    completed = tmp_path / "completed"
    completed.mkdir()
    _write_csv(r1, completed / "reviewer_1_completed.csv")
    _write_csv(r2, completed / "reviewer_2_completed.csv")
    catalog_path = tmp_path / "pair_catalog.csv"
    _write_csv(cat, catalog_path)
    output_base = tmp_path / "agreement_analysis"
    output_base.mkdir()  # pre-create

    result = run_analysis(
        completed_dir=completed,
        catalog_path=catalog_path,
        output_base=output_base,
    )
    assert result["status"] == "output_exists"


# ── compute_weighted_kappa unit tests ─────────────────────────────────────────


def _merged_from_ratings(
    r1_ratings: list[str],
    r2_ratings: list[str],
) -> pd.DataFrame:
    n = len(r1_ratings)
    return pd.DataFrame(
        {
            "pair_id": [f"P{i:04d}" for i in range(n)],
            "concern_relation_r1": r1_ratings,
            "concern_relation_r2": r2_ratings,
        }
    )


def test_compute_weighted_kappa_perfect_agreement():
    ratings = ["0", "1", "2", "3", "0", "1", "2"]
    merged = _merged_from_ratings(ratings, ratings)
    result = compute_weighted_kappa(merged)
    assert result["status"] == "computed"
    assert result["kappa"] == pytest.approx(1.0, abs=1e-6)


def test_compute_weighted_kappa_insufficient_data():
    merged = _merged_from_ratings(["1", "2", "3"], ["1", "2", "3"])
    result = compute_weighted_kappa(merged)
    assert result["status"] == "insufficient_data"
    assert result["kappa"] is None
    assert result["n"] == 3


def test_compute_weighted_kappa_excludes_u_and_blank():
    r1 = ["1", "U", "", "2", "3", "0", "2"]
    r2 = ["1", "2", "1", "2", "3", "0", "2"]
    merged = _merged_from_ratings(r1, r2)
    result = compute_weighted_kappa(merged)
    assert result["status"] == "computed"
    assert result["n"] == 5  # excludes U at [1] and blank at [2]


def test_compute_weighted_kappa_zero_variance():
    # All same category -> zero_variance
    merged = _merged_from_ratings(["3"] * 8, ["3"] * 8)
    result = compute_weighted_kappa(merged)
    assert result["status"] == "zero_variance"
    assert result["kappa"] is None


def test_compute_weighted_kappa_linear_weight_known_value():
    """
    Manual check: r1=[0,3] r2=[3,0] — maximum disagreement.
    weights[0,3] = 1 - 3/3 = 0, so po_w should equal 0.
    """
    merged = _merged_from_ratings(
        ["0", "3", "1", "2", "0", "3", "1"],
        ["3", "0", "2", "1", "3", "0", "2"],
    )
    result = compute_weighted_kappa(merged)
    assert result["status"] == "computed"
    # po_weighted for total disagreement should be < pe_weighted
    assert result["po_weighted"] <= result["pe_weighted"]


# ── compute_exact_agreement unit tests ────────────────────────────────────────


def test_exact_agreement_counts_correctly():
    merged = _merged_from_ratings(
        ["1", "2", "U", "", "3", "3"],
        ["1", "3", "U", "2", "3", "2"],
    )
    result = compute_exact_agreement(merged)
    # Pairs where BOTH ratings are in {0,1,2,3}:
    # idx 0: "1","1" -> yes; idx 1: "2","3" -> yes; idx 4: "3","3" -> yes; idx 5: "3","2" -> yes = 4
    # idx 2: both U -> excluded; idx 3: r1 blank -> excluded
    assert result["total_pairs"] == 6
    assert result["jointly_numeric"] == 4
    assert result["exact_agree"] == 2   # idx 0 ("1"=="1") and idx 4 ("3"=="3") agree
    assert result["both_U"] == 1        # idx 2
    assert result["either_blank"] == 1  # idx 3


# ── validate_pair_ids unit tests ─────────────────────────────────────────────


def test_validate_pair_ids_no_errors():
    ids = ["P0001", "P0002", "P0003"]
    df = pd.DataFrame({"pair_id": ids})
    errors = validate_pair_ids(df, set(ids), "Test")
    assert errors == []


def test_validate_pair_ids_missing():
    ids = ["P0001", "P0002", "P0003"]
    df = pd.DataFrame({"pair_id": ["P0001", "P0002"]})
    errors = validate_pair_ids(df, set(ids), "Test")
    assert any("missing" in e for e in errors)


def test_validate_pair_ids_extra():
    ids = ["P0001", "P0002"]
    df = pd.DataFrame({"pair_id": ["P0001", "P0002", "P0099"]})
    errors = validate_pair_ids(df, set(ids), "Test")
    assert any("unexpected" in e for e in errors)


def test_validate_pair_ids_duplicate():
    ids = ["P0001", "P0002", "P0003"]
    df = pd.DataFrame({"pair_id": ["P0001", "P0001", "P0002"]})  # dup + missing
    errors = validate_pair_ids(df, set(ids), "Test")
    assert any("duplicate" in e for e in errors)


# ── validate_judgment_fields unit tests ──────────────────────────────────────


def test_validate_judgment_fields_valid():
    df = pd.DataFrame(
        {
            "concern_relation": ["0", "1", "U", "3"],
            "confidence": ["high", "medium", "low", "high"],
            "shared_concern_note": ["note"] * 4,
            "evidence_note": ["ev"] * 4,
        }
    )
    errors, stats = validate_judgment_fields(df, "Test")
    assert errors == []
    assert stats["concern_relation_U"] == 1
    assert stats["concern_relation_numeric"] == 3


def test_validate_judgment_fields_invalid_concern_relation():
    df = pd.DataFrame(
        {
            "concern_relation": ["0", "YES", "U"],
            "confidence": ["high", "high", "high"],
            "shared_concern_note": [""] * 3,
            "evidence_note": [""] * 3,
        }
    )
    errors, _ = validate_judgment_fields(df, "Test")
    assert any("invalid" in e.lower() for e in errors)
    assert any("YES" in e for e in errors)


def test_validate_judgment_fields_invalid_confidence():
    df = pd.DataFrame(
        {
            "concern_relation": ["1", "2", "3"],
            "confidence": ["high", "MAYBE", "low"],
            "shared_concern_note": [""] * 3,
            "evidence_note": [""] * 3,
        }
    )
    errors, _ = validate_judgment_fields(df, "Test")
    assert any("confidence" in e.lower() for e in errors)
    # The script lowercases confidence before validation; the error message contains 'maybe'
    assert any("maybe" in e.lower() for e in errors)


# ── merge_reviewer_judgments ──────────────────────────────────────────────────


def test_merge_reviewer_judgments_suffix_assignment():
    ids = ["P0001", "P0002", "P0003"]
    r1 = _make_template(ids, ["1", "2", "3"])
    r2 = _make_template(ids, ["0", "1", "2"], reviewer_id="R2")
    merged = merge_reviewer_judgments(r1, r2)
    assert "concern_relation_r1" in merged.columns
    assert "concern_relation_r2" in merged.columns
    assert list(merged["concern_relation_r1"]) == ["1", "2", "3"]
    assert list(merged["concern_relation_r2"]) == ["0", "1", "2"]
    assert merged["pair_id"].tolist() == ids


def test_merge_reviewer_judgments_row_count():
    ids = [f"P{i:04d}" for i in range(1, 11)]
    r1 = _make_template(ids, ["1"] * 10)
    r2 = _make_template(ids, ["2"] * 10, reviewer_id="R2")
    merged = merge_reviewer_judgments(r1, r2)
    assert len(merged) == 10


# ── Missing required columns ──────────────────────────────────────────────────


def test_missing_required_columns_fails(tmp_path):
    pair_ids = [f"P{i:04d}" for i in range(1, 7)]
    r1 = _make_template(pair_ids, ["1"] * 6)
    # Remove concern_relation from r2
    r2 = _make_template(pair_ids, ["1"] * 6, reviewer_id="R2").drop(
        columns=["concern_relation"]
    )
    cat = _make_catalog(pair_ids)
    completed = tmp_path / "completed"
    completed.mkdir()
    _write_csv(r1, completed / "reviewer_1_completed.csv")
    _write_csv(r2, completed / "reviewer_2_completed.csv")
    catalog_path = tmp_path / "pair_catalog.csv"
    _write_csv(cat, catalog_path)
    result = run_analysis(
        completed_dir=completed,
        catalog_path=catalog_path,
        output_base=tmp_path / "out",
    )
    assert result["status"] == "validation_failed"
    assert any("concern_relation" in e for e in result["errors"])
