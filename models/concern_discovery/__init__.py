"""
models/concern_discovery
========================
Evidence-backed concern discovery on top of the existing unsupervised clusterings.

  - schema      : input-schema and integrity checks (corpus, embeddings, label arrays)
  - duplicates  : exact / near-duplicate / repeated-template grouping
  - diagnostics : cohesion, separation, seed stability, metadata confounds, cluster tiers
  - summaries   : MMR representative selection, deterministic extractive descriptions, grounding checks
  - review      : blinded human-review package and analysis of completed reviews only

Clusters are unsupervised groupings, not ground-truth labels, and no output is human-validated
until completed reviewer files are analysed.
"""
from models.concern_discovery.diagnostics import (
    THRESHOLDS,
    TIER_DESCRIPTIONS,
    TIER_ORDER,
    assess_clusters,
    association,
    cluster_geometry,
    cluster_stability,
    confound_report,
    cramers_v_corrected,
    keyword_support,
    partition_summary,
)
from models.concern_discovery.duplicates import near_duplicate_groups, strip_social, summarise_groups, template_key
from models.concern_discovery.review import analyze_reviews, build_review_items, check_no_leakage, write_package
from models.concern_discovery.schema import CheckLog, CorpusIntegrityError, validate_corpus, validate_labels
from models.concern_discovery.summaries import (
    GroundingError,
    describe_cluster,
    quotable_span,
    select_representatives,
    verify_grounding,
)
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
    verify_evidence_traceability,
)

__all__ = [
    "THRESHOLDS", "TIER_DESCRIPTIONS", "TIER_ORDER", "assess_clusters", "association", "cluster_geometry",
    "cluster_stability", "confound_report", "cramers_v_corrected", "keyword_support", "partition_summary",
    "near_duplicate_groups", "strip_social", "summarise_groups", "template_key",
    "analyze_reviews", "build_review_items", "check_no_leakage", "write_package",
    "CheckLog", "CorpusIntegrityError", "validate_corpus", "validate_labels",
    "GroundingError", "describe_cluster", "quotable_span", "select_representatives", "verify_grounding",
    "DISCLAIMER_HEADER", "VISUAL_SEPARATION_WARNING", "MissingDataError", "TraceabilityError",
    "build_visualization_package", "check_coordinates_availability", "compute_reproducible_projection",
    "generate_cluster_inspection_html", "load_cluster_discovery_artifacts", "verify_evidence_traceability",
]

