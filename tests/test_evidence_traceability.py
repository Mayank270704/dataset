"""
tests/test_evidence_traceability.py
===================================
Unit and integration tests for PolicyLens 4-tier source-evidence traceability
and lineage verification.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from models.concern_discovery.traceability import (
    RAW_SPECS,
    RawDatasetContainer,
    load_clean_datasets,
    load_raw_containers,
    slugify_column,
    trace_evidence_item,
    verify_quote_against_texts,
)

ROOT = Path(__file__).resolve().parent.parent
PRIOR_V1 = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"
RUN_V1_CORPUS = ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv"


def test_raw_dataset_containers_load():
    """All 3 raw consultation CSVs must load with verified encodings and non-empty data."""
    containers = load_raw_containers(ROOT)
    assert len(containers) == 3
    assert set(containers.keys()) == set(RAW_SPECS.keys())

    for key, container in containers.items():
        assert container.path.is_file(), f"Raw file missing: {container.path}"
        assert len(container.data) > 0, f"Raw data empty for {key}"
        assert len(container.raw_cols) > 0, f"No columns parsed for {key}"
        assert len(container.slug_to_raw_col) == len(container.raw_cols)


def test_quote_verification_logic():
    """Quotes must be correctly categorized as verbatim, normalized, or ungrounded."""
    raw = "  This is an important citizen submission about transparency. \r\n"
    clean = "This is an important citizen submission about transparency."
    corpus = "This is an important citizen submission about transparency."

    # 1. Exact raw substring
    q1 = "citizen submission about transparency"
    in_c, in_cl, in_raw, status, diff = verify_quote_against_texts(q1, raw, clean, corpus)
    assert in_raw and in_cl and in_c
    assert status == "VERBATIM_RAW"

    # 2. Match after whitespace normalization
    raw_multiline = "Line one\n\nand line two."
    q_norm = "Line one and line two."
    in_c, in_cl, in_raw, status, diff = verify_quote_against_texts(q_norm, raw_multiline, clean, corpus)
    assert status in {"NORM_WHITESPACE_RAW", "VERBATIM_RAW"}

    # 3. Completely ungrounded/hallucinated quote must FAIL
    hallucinated = "The government should immediately cancel all open data programs."
    in_c, in_cl, in_raw, status, diff = verify_quote_against_texts(hallucinated, raw, clean, corpus)
    assert not in_raw and not in_cl and not in_c
    assert status == "FAILED"


def test_trace_actual_ranking_quote():
    """Traces an actual quote from ranking_evidence.csv through all 4 tiers."""
    rank_ev_path = PRIOR_V1 / "ranking_evidence.csv"
    assert rank_ev_path.is_file(), f"ranking_evidence.csv missing: {rank_ev_path}"

    df_rank = pd.read_csv(rank_ev_path, dtype=str, keep_default_na=False)
    assert len(df_rank) == 275

    first_item = df_rank.iloc[0].to_dict()
    raw_containers = load_raw_containers(ROOT)
    clean_dfs = load_clean_datasets(ROOT)

    corpus_df = pd.read_csv(RUN_V1_CORPUS, dtype=str, keep_default_na=False)
    corpus_map = {
        int(r.corpus_id): {
            "corpus_id": int(r.corpus_id),
            "unique_id": int(getattr(r, "unique_id", -1)),
            "dup_group_id": getattr(r, "duplicate_text_group_id", getattr(r, "dup_group_id", "")),
            "source_dataset": str(r.source_dataset),
            "source_record_id": str(r.source_record_id),
            "source_row_number": int(r.source_row_number),
            "source_field": str(r.source_field),
            "original_text": str(r.original_text),
            "language": str(getattr(r, "language", "")),
        }
        for r in corpus_df.itertuples()
    }

    result = trace_evidence_item(first_item, raw_containers, clean_dfs, corpus_map, evidence_type="ranked_concern")

    assert result.corpus_id == int(first_item["corpus_id"])
    assert result.source_dataset == first_item["source_dataset"]
    assert result.is_verbatim_in_corpus
    assert result.is_verbatim_in_clean
    assert result.match_status in {"VERBATIM_RAW", "VERBATIM_CLEAN"}
    assert result.raw_filename in RAW_SPECS[first_item["source_dataset"]].filename


def test_trace_ungrounded_quote_fails_validation():
    """An altered quote must be rejected as ungrounded."""
    raw_containers = load_raw_containers(ROOT)
    clean_dfs = load_clean_datasets(ROOT)
    corpus_df = pd.read_csv(RUN_V1_CORPUS, dtype=str, keep_default_na=False)
    corpus_map = {
        int(r.corpus_id): {
            "corpus_id": int(r.corpus_id),
            "unique_id": int(getattr(r, "unique_id", -1)),
            "dup_group_id": getattr(r, "duplicate_text_group_id", getattr(r, "dup_group_id", "")),
            "source_dataset": str(r.source_dataset),
            "source_record_id": str(r.source_record_id),
            "source_row_number": int(r.source_row_number),
            "source_field": str(r.source_field),
            "original_text": str(r.original_text),
            "language": str(getattr(r, "language", "")),
        }
        for r in corpus_df.itertuples()
    }

    fake_item = {
        "corpus_id": 0,
        "cluster_id": 1,
        "rank_multi": 1,
        "quote": "This sentence was hallucinated and never appeared in any citizen submission.",
        "full_text": "This sentence was hallucinated and never appeared in any citizen submission.",
    }

    result = trace_evidence_item(fake_item, raw_containers, clean_dfs, corpus_map, evidence_type="ranked_concern")
    assert result.match_status == "FAILED"
    assert not result.is_verbatim_in_raw
    assert not result.is_verbatim_in_clean
    assert not result.is_verbatim_in_corpus


def test_evidence_traceability_manifest_and_report_exist():
    """The generated evidence traceability manifest and report must exist and verify 100% grounding."""
    manifest_path = PRIOR_V1 / "evidence_traceability_manifest.json"
    report_path = ROOT / "reports" / "source_evidence_traceability_report_v1.md"
    ledger_path = PRIOR_V1 / "evidence_traceability_ledger.csv"

    assert manifest_path.is_file(), f"Manifest missing: {manifest_path}"
    assert report_path.is_file(), f"Report missing: {report_path}"
    assert ledger_path.is_file(), f"Ledger missing: {ledger_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASSED"
    summary = manifest["summary"]
    assert summary["total_evidence_quotes_audited"] == 655
    assert summary["failed_ungrounded_count"] == 0
    assert summary["pass_rate_grounded"] == 100.0

    ledger_df = pd.read_csv(ledger_path, dtype=str)
    assert len(ledger_df) == 655
    assert {"quote", "raw_filename", "raw_col_header", "match_status", "corpus_id"} <= set(ledger_df.columns)
    assert (ledger_df["match_status"] != "FAILED").all()


def test_cli_check_only_returns_success():
    """CLI script with --check-only must return exit code 0."""
    from scripts.verify_evidence_traceability import main as verify_main
    exit_code = verify_main(["--check-only"])
    assert exit_code == 0
