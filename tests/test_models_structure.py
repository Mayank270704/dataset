"""
tests/test_models_structure.py
==============================
Validates the models/ directory organization, package integrity, and backward-compatible wrappers.

Environment
-----------
Set ``POLICYLENS_SKIP_UMAP_TESTS=1`` to skip tests that import models.concern_clustering or
models.concern_discovery.  These tests require a working UMAP installation and may hang in
sandbox environments where the umap-learn package initialisation blocks indefinitely.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Guard: skip any test that needs UMAP when the env-var is set, or when
# running inside the Antigravity sandbox (which blocks UMAP init).
# ---------------------------------------------------------------------------
_SKIP_UMAP = os.environ.get("POLICYLENS_SKIP_UMAP_TESTS", "").strip() not in {"", "0", "false", "False"}
_skip_umap_reason = (
    "Skipped: POLICYLENS_SKIP_UMAP_TESTS=1 — "
    "umap-learn hangs in this environment. "
    "Run outside the sandbox to execute these tests."
)

@pytest.mark.skipif(_SKIP_UMAP, reason=_skip_umap_reason)
def test_models_package_imports():
    import models
    import models.common
    import models.semantic_similarity
    import models.concern_clustering
    import models.concern_discovery
    import models.stance_classification
    import models.multilingual

    assert hasattr(models, "semantic_similarity")
    assert hasattr(models, "concern_clustering")
    assert hasattr(models, "concern_discovery")
    assert hasattr(models, "stance_classification")
    assert hasattr(models, "multilingual")
    assert hasattr(models, "common")


def test_models_common_paths():
    from models.common import (
        PROJECT_ROOT,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        EMBEDDINGS_DIR,
        CLUSTERING_DIR,
        ANNOTATION_DIR,
        MODELS_DIR,
    )

    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert EMBEDDINGS_DIR.exists()
    assert CLUSTERING_DIR.exists()
    assert ANNOTATION_DIR.exists()
    assert MODELS_DIR.exists()


def test_semantic_similarity_exports_match():
    import models.semantic_similarity as m_sem
    import scripts.semantic_similarity as s_sem

    symbols = [
        "MODEL_ID_DEFAULT",
        "DEFAULT_MAX_LENGTH",
        "DEFAULT_BATCH_SIZE",
        "MIN_TEXT_LEN",
        "GENERIC_FILLERS",
        "is_eligible",
        "mean_pool_and_normalize",
        "build_corpus",
        "compute_corpus_hash",
        "embed_and_save_corpus",
        "retrieve_similar",
    ]

    for sym in symbols:
        assert hasattr(m_sem, sym), f"models.semantic_similarity missing {sym}"
        assert hasattr(s_sem, sym), f"scripts.semantic_similarity missing {sym}"
        assert getattr(m_sem, sym) is getattr(s_sem, sym), f"Symbol {sym} does not match between models and scripts"


@pytest.mark.skipif(_SKIP_UMAP, reason=_skip_umap_reason)
def test_concern_clustering_exports_match():
    import models.concern_clustering as m_clus
    import scripts.concern_clustering as s_clus

    symbols = [
        "NOISE_LABEL",
        "ClusteringResult",
        "load_embeddings_and_metadata",
        "reduce_with_umap",
        "cluster_with_hdbscan",
        "_ctfidf_matrix",
        "extract_cluster_topics",
        "compute_quality_metrics",
        "save_clustering_artefacts",
        "run_clustering",
    ]

    for sym in symbols:
        assert hasattr(m_clus, sym), f"models.concern_clustering missing {sym}"
        assert hasattr(s_clus, sym), f"scripts.concern_clustering missing {sym}"
        assert getattr(m_clus, sym) is getattr(s_clus, sym), f"Symbol {sym} does not match between models and scripts"


def test_documentation_files_exist():
    from models.common import MODELS_DIR

    readme = MODELS_DIR / "README.md"
    registry = MODELS_DIR / "MODEL_REGISTRY.md"

    assert readme.exists(), "models/README.md does not exist"
    assert registry.exists(), "models/MODEL_REGISTRY.md does not exist"
    assert readme.stat().st_size > 500, "models/README.md is too short"
    assert registry.stat().st_size > 500, "models/MODEL_REGISTRY.md is too short"


def test_concern_discovery_all_exports_are_importable():
    """Every symbol listed in models.concern_discovery.__all__ must be importable and resolve."""
    import models.concern_discovery as cd

    assert hasattr(cd, "__all__"), "models.concern_discovery must define __all__"
    for sym in cd.__all__:
        assert hasattr(cd, sym), f"models.concern_discovery.__all__ lists '{sym}' but it is not accessible"
        assert getattr(cd, sym) is not None, f"models.concern_discovery.{sym} resolves to None"


def test_concern_discovery_submodule_apis():
    """Key public APIs from each concern_discovery submodule must be callable or be data objects."""
    from models.concern_discovery import diagnostics as dg
    from models.concern_discovery import duplicates as dup
    from models.concern_discovery import prioritization as pr
    from models.concern_discovery import review as rv
    from models.concern_discovery import schema as sc
    from models.concern_discovery import summaries as sm
    from models.concern_discovery import traceability as tr

    # diagnostics
    for name in ("assess_clusters", "cluster_geometry", "cluster_stability", "confound_report",
                 "cramers_v_corrected", "keyword_support", "partition_summary", "association",
                 "THRESHOLDS", "TIER_DESCRIPTIONS", "TIER_ORDER"):
        assert hasattr(dg, name), f"diagnostics missing {name}"

    # duplicates
    for name in ("near_duplicate_groups", "strip_social", "summarise_groups", "template_key"):
        assert hasattr(dup, name), f"duplicates missing {name}"

    # prioritization
    for name in ("DEFAULT_WEIGHTS", "SIGNALS", "DEFAULT_TIER_QUALITY", "RANKED_TIERS",
                 "validate_config", "compute_signals", "scaled_signals", "score",
                 "explain", "compare_baseline", "sensitivity",
                 "verify_evidence", "collective_demand_candidates", "validate_outputs"):
        assert hasattr(pr, name), f"prioritization missing {name}"
    assert set(pr.DEFAULT_WEIGHTS) == set(pr.SIGNALS), "DEFAULT_WEIGHTS keys must equal SIGNALS"
    assert abs(sum(pr.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, "DEFAULT_WEIGHTS must sum to 1"

    # review
    for name in ("build_review_items", "write_package", "analyze_reviews", "check_no_leakage", "REVIEWERS"):
        assert hasattr(rv, name), f"review missing {name}"

    # schema
    for name in ("validate_corpus", "validate_labels", "CheckLog", "CorpusIntegrityError",
                 "sha256_file", "sha256_text"):
        assert hasattr(sc, name), f"schema missing {name}"

    # summaries
    for name in ("select_representatives", "describe_cluster", "verify_grounding",
                 "quotable_span", "GroundingError"):
        assert hasattr(sm, name), f"summaries missing {name}"

    # traceability
    for name in ("RAW_SPECS", "RawDatasetContainer", "load_raw_containers", "load_clean_datasets",
                 "trace_evidence_item", "verify_quote_against_texts",
                 "audit_full_evidence_lineage", "slugify_column"):
        assert hasattr(tr, name), f"traceability missing {name}"

    # visualization
    from models.concern_discovery import visualization as vz
    for name in ("check_coordinates_availability", "compute_reproducible_projection",
                 "verify_evidence_traceability", "generate_cluster_inspection_html",
                 "build_visualization_package", "load_cluster_discovery_artifacts",
                 "MissingDataError", "TraceabilityError",
                 "DISCLAIMER_HEADER", "VISUAL_SEPARATION_WARNING"):
        assert hasattr(vz, name), f"visualization missing {name}"



def test_concern_discovery_weights_and_tiers_are_consistent():
    """Prioritization module invariants: weight keys, tier map, and NOT_RANKED_REASON coverage."""
    from models.concern_discovery import prioritization as pr

    w = pr.validate_config(pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    assert set(w) == set(pr.SIGNALS)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    # every ranked tier must have a tier_quality value in [0, 1]
    for tier, q in pr.DEFAULT_TIER_QUALITY.items():
        assert 0.0 <= q <= 1.0, f"tier_quality[{tier!r}] = {q} outside [0, 1]"
    # not-ranked tiers must not appear in tier_quality
    for tier in pr.NOT_RANKED_REASON:
        assert tier not in pr.DEFAULT_TIER_QUALITY, f"{tier!r} should not be in DEFAULT_TIER_QUALITY"
