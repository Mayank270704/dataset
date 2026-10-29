"""
tests/test_cluster_visualization.py
===================================
Automated tests for PolicyLens Interactive Cluster Visualization & Diagnostics.

Validates:
1. Source traceability: strict validation against source consultation records (catches tampered IDs & quotes).
2. Missing data handling: graceful reporting of unavailable 2D coordinates, missing embeddings, and invalid paths.
3. Projection reproducibility: deterministic 2D output with fixed random seed.
4. HTML generation: self-contained offline document with exploratory disclaimers and visual separation warnings.
5. End-to-end package generation and manifest SHA-256 integrity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.concern_discovery.visualization import (
    DISCLAIMER_HEADER,
    VISUAL_SEPARATION_WARNING,
    MissingDataError,
    TraceabilityError,
    build_visualization_package,
    check_coordinates_availability,
    compute_reproducible_projection,
    generate_cluster_inspection_html,
    load_cluster_discovery_artifacts,
    sha256_file,
    verify_evidence_traceability,
)

ROOT = Path(__file__).resolve().parent.parent
DISCOVERY_V1_DIR = ROOT / "data" / "full_corpus" / "concern_discovery_v1"
RUN_V1_DIR = ROOT / "data" / "full_corpus" / "run_v1"
EXP_V2_DIR = ROOT / "data" / "full_corpus" / "experiments_v2"


# ==============================================================================
# 1. Coordinate Availability and Projection Tests
# ==============================================================================

def test_check_coordinates_availability(tmp_path: Path):
    """Test coordinate file inspection and shape validation."""
    # None or non-existent
    avail, msg = check_coordinates_availability(None)
    assert not avail
    assert "unavailable" in msg

    avail, msg = check_coordinates_availability(tmp_path / "non_existent.npy")
    assert not avail
    assert "unavailable" in msg

    # Invalid shape (3D array)
    invalid_npy = tmp_path / "invalid_3d.npy"
    np.save(invalid_npy, np.zeros((10, 3, 2), dtype=np.float32))
    avail, msg = check_coordinates_availability(invalid_npy)
    assert not avail

    # Invalid shape (5D array, like HDBSCAN input)
    umap_5d = tmp_path / "umap_5d.npy"
    np.save(umap_5d, np.zeros((10, 5), dtype=np.float32))
    avail, msg = check_coordinates_availability(umap_5d)
    assert not avail

    # Valid 2D numpy array
    valid_npy = tmp_path / "valid_2d.npy"
    np.save(valid_npy, np.zeros((50, 2), dtype=np.float32))
    avail, msg = check_coordinates_availability(valid_npy)
    assert avail
    assert "Valid 2D array" in msg

    # Valid 2D CSV
    valid_csv = tmp_path / "valid_2d.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(valid_csv, index=False)
    avail, msg = check_coordinates_availability(valid_csv)
    assert avail
    assert "Valid 2D CSV" in msg

    # Invalid CSV (missing x/y)
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(invalid_csv, index=False)
    avail, msg = check_coordinates_availability(invalid_csv)
    assert not avail


def test_compute_reproducible_projection_determinism():
    """Deterministic projection must yield bit-for-bit identical coordinates across runs with same seed."""
    rng = np.random.default_rng(123)
    synth_emb = rng.standard_normal((100, 32)).astype(np.float32)

    proj1 = compute_reproducible_projection(synth_emb, method="pca", random_state=42)
    proj2 = compute_reproducible_projection(synth_emb, method="pca", random_state=42)

    assert proj1.shape == (100, 2)
    assert np.array_equal(proj1, proj2), "Identical seed must produce identical coordinates"

    # Input validation
    with pytest.raises(ValueError, match="Embeddings must be 2D array"):
        compute_reproducible_projection(synth_emb.flatten(), method="pca")

    with pytest.raises(ValueError, match="NaN or infinite"):
        bad_emb = synth_emb.copy()
        bad_emb[0, 0] = np.nan
        compute_reproducible_projection(bad_emb, method="pca")

    with pytest.raises(ValueError, match="Unknown projection method"):
        compute_reproducible_projection(synth_emb, method="invalid_method")


# ==============================================================================
# 2. Source Traceability & Grounding Verification Tests
# ==============================================================================

def test_verify_evidence_traceability_passing():
    """Valid representative quotes grounded in corpus must pass traceability audit."""
    corpus_df = pd.DataFrame([
        {
            "corpus_id": 10,
            "unique_id": 10,
            "source_dataset": "dataset_alpha",
            "source_record_id": "rec_001",
            "original_text": "Transparency in open data is critical for citizen trust in government.",
        },
        {
            "corpus_id": 20,
            "unique_id": 20,
            "source_dataset": "dataset_beta",
            "source_record_id": "rec_002",
            "original_text": "We need improved access to procurement documents and digital archives.",
        },
    ])

    reps_df = pd.DataFrame([
        {
            "cluster_id": 1,
            "rank": 1,
            "corpus_id": 10,
            "unique_id": 10,
            "source_dataset": "dataset_alpha",
            "source_record_id": "rec_001",
            "quote": "Transparency in open data",
            "cosine_to_centroid": 0.85,
        },
        {
            "cluster_id": 1,
            "rank": 2,
            "corpus_id": 20,
            "unique_id": 20,
            "source_dataset": "dataset_beta",
            "source_record_id": "rec_002",
            "quote": "procurement documents and digital archives",
            "cosine_to_centroid": 0.81,
        },
    ])

    report = verify_evidence_traceability(reps_df, corpus_df)
    assert report["is_valid"] is True
    assert report["total_checked"] == 2
    assert report["matched"] == 2
    assert len(report["discrepancies"]) == 0


def test_verify_evidence_traceability_catches_tampering():
    """Traceability audit must flag tampered corpus IDs, modified datasets, and fabricated quotes."""
    corpus_df = pd.DataFrame([
        {
            "corpus_id": 10,
            "unique_id": 10,
            "source_dataset": "dataset_alpha",
            "source_record_id": "rec_001",
            "original_text": "Real authentic citizen comment.",
        }
    ])

    # Case 1: Invented corpus_id
    reps_invented_id = pd.DataFrame([{
        "cluster_id": 1, "rank": 1, "corpus_id": 9999, "unique_id": 9999,
        "source_dataset": "dataset_alpha", "source_record_id": "rec_001",
        "quote": "Real authentic citizen comment.",
    }])
    rep1 = verify_evidence_traceability(reps_invented_id, corpus_df)
    assert rep1["is_valid"] is False
    assert "missing from corpus_entries.csv" in rep1["discrepancies"][0]["reason"]

    # Case 2: Mismatched source record ID
    reps_mismatched_rec = pd.DataFrame([{
        "cluster_id": 1, "rank": 1, "corpus_id": 10, "unique_id": 10,
        "source_dataset": "dataset_alpha", "source_record_id": "tampered_row_id",
        "quote": "Real authentic citizen comment.",
    }])
    rep2 = verify_evidence_traceability(reps_mismatched_rec, corpus_df)
    assert rep2["is_valid"] is False
    assert "source_record_id mismatch" in rep2["discrepancies"][0]["reason"]

    # Case 3: Fabricated quote not in source text
    reps_fabricated_quote = pd.DataFrame([{
        "cluster_id": 1, "rank": 1, "corpus_id": 10, "unique_id": 10,
        "source_dataset": "dataset_alpha", "source_record_id": "rec_001",
        "quote": "Hallucinated quote never uttered by citizen.",
    }])
    rep3 = verify_evidence_traceability(reps_fabricated_quote, corpus_df)
    assert rep3["is_valid"] is False
    assert "not a substring" in rep3["discrepancies"][0]["reason"]


# ==============================================================================
# 3. Missing Data and Graceful Fallback Handling Tests
# ==============================================================================

def test_load_cluster_discovery_artifacts_missing_inputs(tmp_path: Path):
    """Missing input files must raise MissingDataError with informative path details."""
    with pytest.raises(MissingDataError, match="Input directory does not exist"):
        load_cluster_discovery_artifacts(tmp_path / "non_existent_folder")

    # Directory exists but missing concern_groups.csv
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(MissingDataError, match="concern_groups.csv missing"):
        load_cluster_discovery_artifacts(empty_dir)

    # Missing corpus_entries.csv
    sel_dir = empty_dir / "selected"
    sel_dir.mkdir()
    pd.DataFrame({"cluster_id": [0]}).to_csv(sel_dir / "concern_groups.csv", index=False)
    pd.DataFrame({"unique_id": [0]}).to_csv(sel_dir / "entry_assignments.csv", index=False)
    cfg_dir = empty_dir / "configurations" / "strip__umap_m25s5"
    cfg_dir.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [0], "corpus_id": [0]}).to_csv(cfg_dir / "representative_examples.csv", index=False)

    with pytest.raises(MissingDataError, match="corpus_entries.csv missing"):
        load_cluster_discovery_artifacts(empty_dir, run_v1_dir=tmp_path / "non_existent_run_v1")


# ==============================================================================
# 4. HTML Report Generation & Content Tests
# ==============================================================================

def test_generate_cluster_inspection_html():
    """HTML report must contain mandatory scientific disclaimers and offline components."""
    # Synthetic mock data
    groups_df = pd.DataFrame([
        {
            "cluster_id": 0,
            "tier": "candidate",
            "n_entries": 150,
            "n_unique_texts": 145,
            "share_of_eligible_entries": 0.015,
            "english": 0.9,
            "french": 0.1,
            "flags": "",
            "short_label": "Digital Open Data",
            "description": "Citizen concern on open data APIs",
            "cohesion_mean": 0.812,
            "margin_mean": 0.054,
            "negative_margin_share": 0.02,
            "stability_jaccard_mean": 0.88,
        }
    ])

    assignments_df = pd.DataFrame([
        {
            "corpus_id": 0,
            "unique_id": 0,
            "cluster_id": 0,
            "tier": "candidate",
            "short_label": "Digital Open Data",
        }
    ])

    reps_df = pd.DataFrame([
        {
            "cluster_id": 0,
            "rank": 1,
            "corpus_id": 0,
            "unique_id": 0,
            "source_dataset": "dataset_2018",
            "source_record_id": "row_42",
            "language": "en",
            "cosine_to_centroid": 0.84,
            "quote": "Digital open data is essential",
            "full_text": "Digital open data is essential for transparency.",
            "selection_reason": "MMR Representative",
        }
    ])

    coords_2d = np.array([[1.5, -2.3]], dtype=np.float32)
    centroids = {0: {"x": 1.5, "y": -2.3}}
    proj_info = {"method": "pca", "random_state": 42}

    html = generate_cluster_inspection_html({
        "groups": groups_df,
        "assignments": assignments_df,
        "representatives": reps_df,
        "coords_2d": coords_2d,
        "centroids": centroids,
        "projection_info": proj_info,
    })

    assert isinstance(html, str)
    assert len(html) > 1000

    # Mandatory disclaimers
    assert "EXPLORATORY & UNSUPERVISED" in html
    assert "NOT HUMAN VALIDATED" in html
    assert "Visual separation in 2D projections" in html
    assert "DOES NOT constitute evidence of cluster validity" in html

    # Cluster data rendered
    assert "Digital Open Data" in html
    assert "Digital open data is essential" in html
    assert "dataset_2018" in html
    assert "row_42" in html

    # Fully self-contained: no external CDN scripts
    assert "<script src=" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "unpkg.com" not in html
    assert "cdnjs.cloudflare.com" not in html


# ==============================================================================
# 5. End-to-End Deliverable Generation Against Real Corpus Artifacts
# ==============================================================================

@pytest.mark.skipif(not DISCOVERY_V1_DIR.exists(), reason="Real dataset artifacts not found")
def test_build_visualization_package_on_real_artifacts(tmp_path: Path):
    """End-to-end integration test against the real PolicyLens full-corpus artifacts."""
    out_dir = tmp_path / "test_cluster_vis"

    result = build_visualization_package(
        input_dir=DISCOVERY_V1_DIR,
        output_dir=out_dir,
        run_v1_dir=RUN_V1_DIR,
        exp_v2_dir=EXP_V2_DIR,
        projection_method="pca",
        random_state=42,
        verify_traceability=True,
    )

    assert result["status"] == "success"

    # Verify all expected artifacts exist on disk
    html_file = out_dir / "cluster_inspection_report.html"
    coords_file = out_dir / "projection_coordinates_2d.csv"
    summary_file = out_dir / "cluster_summary.csv"
    manifest_file = out_dir / "visualization_manifest.json"

    assert html_file.exists()
    assert coords_file.exists()
    assert summary_file.exists()
    assert manifest_file.exists()

    # Validate coordinates CSV
    coords_df = pd.read_csv(coords_file)
    assert len(coords_df) == 10043, "Must have coordinates for all 10,043 unique texts"
    assert set(["unique_id", "x", "y", "cluster_id", "tier", "short_label"]).issubset(coords_df.columns)

    # Validate manifest
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["human_validated"] is False
    assert manifest["counts"]["total_entries"] == 10108
    assert manifest["counts"]["n_clusters"] == 74
    assert manifest["traceability_audit"]["is_valid"] is True
    assert manifest["traceability_audit"]["total_checked"] == 370
    assert manifest["traceability_audit"]["matched"] == 370

    # Verify hashes in manifest match physical files
    assert manifest["outputs_sha256"]["cluster_inspection_report.html"] == sha256_file(html_file)
    assert manifest["outputs_sha256"]["projection_coordinates_2d.csv"] == sha256_file(coords_file)
    assert manifest["outputs_sha256"]["cluster_summary.csv"] == sha256_file(summary_file)
