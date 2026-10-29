"""
tests/test_demand_generation.py
================================
Tests for the deterministic extractive collective-demand candidate generator.
Verifies research integrity, schema compliance, evidence traceability,
disclaimers, eligibility filtering, and JSON serialisation.
"""
import pytest
from app.demand_generation import (
    generate_candidates,
    get_candidates,
    get_candidate_by_id,
    DemandCandidate,
    DemandEvidence,
    MIN_INDEPENDENT_TEXTS,
    MIN_ENTRIES,
    CANDIDATE_TIERS,
    GENERATION_METHOD,
)


def test_candidates_generation_returns_valid_list():
    candidates = generate_candidates()
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all(isinstance(c, DemandCandidate) for c in candidates)


def test_candidate_tiers_strictly_enforced():
    candidates = generate_candidates()
    for c in candidates:
        assert c.candidate_id.startswith("cand-")
        assert isinstance(c.cluster_id, int)
        assert c.entries >= MIN_ENTRIES


def test_candidate_evidence_traceability():
    candidates = generate_candidates()
    for c in candidates:
        assert c.evidence, f"Candidate {c.candidate_id} must have evidence"
        assert len(c.supporting_corpus_ids) == len(c.evidence)
        for ev in c.evidence:
            assert isinstance(ev, DemandEvidence)
            assert ev.corpus_id > 0
            assert ev.source_dataset
            assert ev.quote, "Evidence quote must be non-empty"
            assert ev.corpus_id in c.supporting_corpus_ids


def test_candidate_metadata_and_disclaimer():
    candidates = generate_candidates()
    for c in candidates:
        assert c.state == "generated_extractive"
        assert c.human_review_status == "pending"
        assert c.generation_method == GENERATION_METHOD
        assert "extractive summaries" in c.disclaimer.lower()
        assert "canadian federal open government" in c.scope.lower()
        assert "not evidence about indian" in c.scope.lower()


def test_candidate_serialization():
    candidates = generate_candidates()
    sample = candidates[0]
    d = sample.as_dict()
    assert isinstance(d, dict)
    assert d["candidate_id"] == sample.candidate_id
    assert d["cluster_id"] == sample.cluster_id
    assert isinstance(d["evidence"], list)
    assert len(d["evidence"]) == len(sample.evidence)
    assert d["evidence"][0]["corpus_id"] == sample.evidence[0].corpus_id


def test_get_candidate_by_id():
    candidates = get_candidates()
    first_id = candidates[0].candidate_id
    retrieved = get_candidate_by_id(first_id)
    assert retrieved is not None
    assert retrieved.candidate_id == first_id

    # Non-existent candidate
    assert get_candidate_by_id("cand-999999") is None
    assert get_candidate_by_id("invalid-format") is None
