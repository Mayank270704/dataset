"""
scripts/concern_clustering.py
=============================
Compatibility wrapper and CLI runner for PolicyLens concern clustering and topic discovery.

The core implementation has been migrated to `models.concern_clustering`.
This script maintains full backward compatibility for existing scripts, tests,
and notebooks importing from `scripts.concern_clustering`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.concern_clustering import (
    CTFIDF_MAX_FEATURES,
    CTFIDF_NGRAM_RANGE,
    CTFIDF_TOP_N_WORDS,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    NOISE_LABEL,
    UMAP_METRIC,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_MIN_DIST_VIZ,
    UMAP_N_COMPONENTS_CLUSTER,
    UMAP_N_COMPONENTS_VIZ,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
    ClusteringResult,
    _ctfidf_matrix,
    _FRENCH_STOPS,
    cluster_with_hdbscan,
    compute_quality_metrics,
    extract_cluster_topics,
    load_embeddings_and_metadata,
    reduce_with_umap,
    run_clustering,
    save_clustering_artefacts,
)

__all__ = [
    "UMAP_N_COMPONENTS_CLUSTER",
    "UMAP_N_COMPONENTS_VIZ",
    "UMAP_N_NEIGHBORS",
    "UMAP_MIN_DIST_CLUSTER",
    "UMAP_MIN_DIST_VIZ",
    "UMAP_METRIC",
    "UMAP_RANDOM_STATE",
    "HDBSCAN_MIN_CLUSTER_SIZE",
    "HDBSCAN_MIN_SAMPLES",
    "HDBSCAN_METRIC",
    "HDBSCAN_CLUSTER_SELECTION_METHOD",
    "CTFIDF_MAX_FEATURES",
    "CTFIDF_TOP_N_WORDS",
    "CTFIDF_NGRAM_RANGE",
    "_FRENCH_STOPS",
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

if __name__ == "__main__":
    from models.common.config import CLUSTERING_DIR, EMBEDDINGS_DIR

    print(f"Running PolicyLens concern clustering pipeline...")
    labels_df, topics_df, metrics = run_clustering(
        embeddings_dir=EMBEDDINGS_DIR,
        output_dir=CLUSTERING_DIR,
    )
    print(f"Clustering complete. {metrics.get('n_clusters')} clusters discovered.")
    print("Top topics:")
    print(topics_df[["cluster_id", "size", "top_keywords"]])
