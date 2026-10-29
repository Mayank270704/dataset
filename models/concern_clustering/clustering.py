"""
models/concern_clustering/clustering.py
=======================================
PolicyLens — Concern Clustering and Topic Discovery.

Pipeline
--------
1. Load pre-computed MiniLM embeddings and corpus metadata.
2. Reduce to 5-D with UMAP for clustering (cosine metric).
3. Cluster with HDBSCAN (density-based, no assumed cluster count).
4. Build per-cluster topic keywords via c-TF-IDF (using sklearn CountVectorizer).
5. Compute cluster quality metrics: silhouette score, persistence summary.
6. Save artefacts to data/clustering/:
   - cluster_labels.csv   — per-comment cluster assignment + metadata
   - cluster_topics.csv   — top keywords + size per cluster
   - umap_2d.npy          — 2-D UMAP coordinates for scatter plots
   - clustering_config.json — hyperparameters + quality metrics
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Default hyperparameters (all overridable by callers / the notebook)
# ---------------------------------------------------------------------------
UMAP_N_COMPONENTS_CLUSTER = 5      # dimensionality for HDBSCAN input
UMAP_N_COMPONENTS_VIZ = 2          # dimensionality for 2-D scatter
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST_CLUSTER = 0.0        # tight clusters for HDBSCAN
UMAP_MIN_DIST_VIZ = 0.1            # more spread for visualisation
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_METRIC = "euclidean"       # on UMAP-reduced vectors (UMAP already handles cosine)
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"  # excess-of-mass

CTFIDF_MAX_FEATURES = 5_000
CTFIDF_TOP_N_WORDS = 10            # keywords per cluster
CTFIDF_NGRAM_RANGE = (1, 2)

# Additional French stop-words to complement sklearn's English list
_FRENCH_STOPS = [
    "de", "du", "des", "la", "le", "les", "un", "une", "en", "et", "est",
    "qui", "que", "qu", "je", "il", "elle", "nous", "vous", "ils", "elles",
    "se", "sa", "son", "ses", "ce", "cet", "cette", "ces", "au", "aux",
    "par", "sur", "pour", "avec", "pas", "plus", "ne", "ou", "mais", "car",
    "si", "dans", "à", "a", "on", "y", "tout", "très", "bien", "faire",
    "être", "avoir", "même", "comme", "dont", "où", "aussi",
]

NOISE_LABEL = -1                   # HDBSCAN convention


class ClusteringResult(NamedTuple):
    """Return bundle from ``run_clustering``."""
    labels: np.ndarray            # shape (N,), int; -1 = noise
    probabilities: np.ndarray     # shape (N,), float; soft assignment confidence
    umap_5d: np.ndarray           # shape (N, 5) for quality metrics
    umap_2d: np.ndarray           # shape (N, 2) for visualisation
    clusterer: Any                # fitted hdbscan.HDBSCAN object


def load_embeddings_and_metadata(
    embeddings_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load pre-computed embeddings and aligned corpus metadata.

    Returns
    -------
    embeddings : np.ndarray, shape (N, 384), float32, L2-normalised
    meta_df    : pd.DataFrame with N rows; ``corpus_id`` matches embedding row
    """
    emb_file = embeddings_dir / "minilm_embeddings.npy"
    meta_file = embeddings_dir / "corpus_metadata.csv"

    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_file}")

    embeddings = np.load(emb_file)
    meta_df = pd.read_csv(meta_file, dtype=str, keep_default_na=False)
    meta_df["corpus_id"] = meta_df["corpus_id"].astype(int)

    if embeddings.shape[0] != len(meta_df):
        raise ValueError(
            f"Alignment mismatch: {embeddings.shape[0]} embeddings vs "
            f"{len(meta_df)} metadata rows."
        )
    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384-D MiniLM embeddings, got shape {embeddings.shape}."
        )

    return embeddings, meta_df


def reduce_with_umap(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS_CLUSTER,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST_CLUSTER,
    metric: str = UMAP_METRIC,
    random_state: int = UMAP_RANDOM_STATE,
) -> tuple[np.ndarray, Any]:
    """
    Reduce embedding dimensionality with UMAP.

    Returns
    -------
    reduced : np.ndarray, shape (N, n_components)
    reducer : fitted umap.UMAP object
    """
    n_samples = embeddings.shape[0]
    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
    if effective_neighbors < n_neighbors:
        warnings.warn(
            f"Dataset has {n_samples} samples; clamped n_neighbors from "
            f"{n_neighbors} to {effective_neighbors}."
        )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )
    reduced = reducer.fit_transform(embeddings).astype(np.float32)
    return reduced, reducer


def cluster_with_hdbscan(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
) -> ClusteringResult:
    """
    Density-based clustering with HDBSCAN.

    Returns
    -------
    ClusteringResult bundle with labels, probabilities, umap_5d, umap_2d, clusterer.
    """
    n_samples = reduced_embeddings.shape[0]
    effective_min_size = max(2, min(min_cluster_size, n_samples // 2))
    effective_min_samples = max(1, min(min_samples, effective_min_size))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_size,
        min_samples=effective_min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_.astype(np.float32)

    # Compute 2-D UMAP for visualization if not already 2-D
    if reduced_embeddings.shape[1] == 2:
        umap_2d = reduced_embeddings.copy()
    else:
        umap_2d, _ = reduce_with_umap(
            reduced_embeddings,
            n_components=UMAP_N_COMPONENTS_VIZ,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST_VIZ,
            metric="euclidean",
            random_state=UMAP_RANDOM_STATE,
        )

    return ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        umap_5d=reduced_embeddings,
        umap_2d=umap_2d,
        clusterer=clusterer,
    )


def _ctfidf_matrix(
    cluster_docs: list[str],
    max_features: int = CTFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = CTFIDF_NGRAM_RANGE,
    extra_stop_words: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the class-based TF-IDF (c-TF-IDF) matrix for pseudo-documents.

    c-TF-IDF formula:
        W_{t, c} = tf_{t, c} * log(1 + A / tf_t)
    where:
        tf_{t, c} = frequency of term t in cluster c
        tf_t      = total frequency of term t across all clusters
        A         = average number of words per cluster
    """
    if extra_stop_words is None:
        extra_stop_words = _FRENCH_STOPS
    combined_stops = list(ENGLISH_STOP_WORDS.union(set(extra_stop_words)))

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=combined_stops,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{2,}\b",
    )

    X = vectorizer.fit_transform(cluster_docs).toarray().astype(np.float64)
    vocab = vectorizer.get_feature_names_out().tolist()

    if X.shape[1] == 0:
        return X, vocab

    tf_c = X
    tf_t = X.sum(axis=0, keepdims=True)
    words_per_cluster = X.sum(axis=1)
    A = words_per_cluster.mean() if words_per_cluster.size > 0 else 1.0

    idf = np.log(1.0 + (A / np.maximum(tf_t, 1e-9)))
    ctfidf = tf_c * idf
    ctfidf = normalize(ctfidf, norm="l2", axis=1)

    return ctfidf, vocab


def extract_cluster_topics(
    meta_df: pd.DataFrame,
    labels: np.ndarray,
    top_n: int = CTFIDF_TOP_N_WORDS,
    max_features: int = CTFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = CTFIDF_NGRAM_RANGE,
    extra_stop_words: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract top keywords for each discovered cluster using c-TF-IDF.

    Returns
    -------
    pd.DataFrame with columns:
      - cluster_id (int)
      - size (int)
      - top_keywords (comma-separated string, or empty for noise)
      - language_distribution (JSON string)
      - source_distribution (JSON string)
    """
    unique_clusters = sorted(np.unique(labels))
    cluster_docs: list[str] = []
    valid_clusters: list[int] = []

    for cid in unique_clusters:
        if cid == NOISE_LABEL:
            continue
        mask = labels == cid
        texts = meta_df.loc[mask, "original_text"].tolist()
        cluster_docs.append(" ".join(texts))
        valid_clusters.append(cid)

    if cluster_docs:
        ctfidf, vocab = _ctfidf_matrix(
            cluster_docs,
            max_features=max_features,
            ngram_range=ngram_range,
            extra_stop_words=extra_stop_words,
        )
    else:
        ctfidf = np.empty((0, 0))
        vocab = []

    rows: list[dict[str, Any]] = []

    for idx, cid in enumerate(valid_clusters):
        mask = labels == cid
        cluster_meta = meta_df.loc[mask]
        size = int(mask.sum())

        if ctfidf.shape[1] > 0:
            top_indices = np.argsort(ctfidf[idx])[::-1][:top_n]
            keywords = [vocab[i] for i in top_indices if ctfidf[idx, i] > 0]
        else:
            keywords = []

        lang_dist = cluster_meta["language"].value_counts().to_dict() if "language" in cluster_meta.columns else {}
        src_dist = cluster_meta["source_dataset"].value_counts().to_dict() if "source_dataset" in cluster_meta.columns else {}

        rows.append({
            "cluster_id": int(cid),
            "size": size,
            "top_keywords": ", ".join(keywords),
            "language_distribution": json.dumps(lang_dist, ensure_ascii=False),
            "source_distribution": json.dumps(src_dist, ensure_ascii=False),
        })

    # Noise cluster
    if NOISE_LABEL in unique_clusters:
        mask = labels == NOISE_LABEL
        cluster_meta = meta_df.loc[mask]
        size = int(mask.sum())
        lang_dist = cluster_meta["language"].value_counts().to_dict() if "language" in cluster_meta.columns else {}
        src_dist = cluster_meta["source_dataset"].value_counts().to_dict() if "source_dataset" in cluster_meta.columns else {}

        rows.insert(0, {
            "cluster_id": int(NOISE_LABEL),
            "size": size,
            "top_keywords": "",
            "language_distribution": json.dumps(lang_dist, ensure_ascii=False),
            "source_distribution": json.dumps(src_dist, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def compute_quality_metrics(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    clusterer: Any,
) -> dict[str, Any]:
    """
    Compute clustering quality indicators:
      - n_clusters (excluding noise)
      - noise_ratio (% unclustered)
      - silhouette_score (on non-noise points if >= 2 clusters)
      - persistence summary (mean, max cluster lambda persistence)
    """
    unique_clusters = set(labels) - {NOISE_LABEL}
    n_clusters = len(unique_clusters)
    n_samples = len(labels)
    noise_count = int((labels == NOISE_LABEL).sum())
    noise_ratio = float(noise_count / n_samples) if n_samples > 0 else 0.0

    non_noise_mask = labels != NOISE_LABEL
    non_noise_labels = labels[non_noise_mask]
    non_noise_points = reduced_embeddings[non_noise_mask]

    sil_score = None
    if n_clusters >= 2 and len(non_noise_labels) > n_clusters:
        try:
            sil_score = float(silhouette_score(non_noise_points, non_noise_labels, metric="euclidean"))
        except Exception:
            sil_score = None

    persistence_summary = {}
    if hasattr(clusterer, "cluster_persistence_") and clusterer.cluster_persistence_ is not None:
        p = clusterer.cluster_persistence_
        if len(p) > 0:
            persistence_summary = {
                "mean_persistence": float(np.mean(p)),
                "max_persistence": float(np.max(p)),
                "min_persistence": float(np.min(p)),
            }

    return {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "n_noise": noise_count,
        "noise_count": noise_count,
        "noise_fraction": round(noise_ratio, 4),
        "noise_ratio": round(noise_ratio, 4),
        "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
        "persistence": persistence_summary,
    }


def save_clustering_artefacts(
    meta_df: pd.DataFrame,
    result: ClusteringResult,
    topics_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Save all clustering outputs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # cluster_labels.csv
    labels_df = meta_df.copy()
    labels_df["cluster_id"] = result.labels.astype(int)
    labels_df["cluster_probability"] = result.probabilities.astype(float).round(4)
    labels_df["umap_x"] = result.umap_2d[:, 0].round(5)
    labels_df["umap_y"] = result.umap_2d[:, 1].round(5)
    labels_df.to_csv(output_dir / "cluster_labels.csv", index=False, encoding="utf-8")

    # cluster_topics.csv
    topics_df.to_csv(output_dir / "cluster_topics.csv", index=False, encoding="utf-8")

    # umap_2d.npy
    np.save(output_dir / "umap_2d.npy", result.umap_2d)

    # clustering_config.json
    full_config = {**config, "quality_metrics": metrics}
    (output_dir / "clustering_config.json").write_text(
        json.dumps(full_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_clustering(
    embeddings_dir: Path,
    output_dir: Path,
    *,
    umap_n_components_cluster: int = UMAP_N_COMPONENTS_CLUSTER,
    umap_n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_min_dist_cluster: float = UMAP_MIN_DIST_CLUSTER,
    umap_metric: str = UMAP_METRIC,
    umap_random_state: int = UMAP_RANDOM_STATE,
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES,
    hdbscan_metric: str = HDBSCAN_METRIC,
    hdbscan_cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
    ctfidf_top_n: int = CTFIDF_TOP_N_WORDS,
    ctfidf_max_features: int = CTFIDF_MAX_FEATURES,
    ctfidf_ngram_range: tuple[int, int] = CTFIDF_NGRAM_RANGE,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """End-to-end clustering pipeline."""
    config = {
        "umap_n_components_cluster": umap_n_components_cluster,
        "umap_n_neighbors": umap_n_neighbors,
        "umap_min_dist_cluster": umap_min_dist_cluster,
        "umap_metric": umap_metric,
        "umap_random_state": umap_random_state,
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
        "hdbscan_metric": hdbscan_metric,
        "hdbscan_cluster_selection_method": hdbscan_cluster_selection_method,
        "ctfidf_top_n": ctfidf_top_n,
        "ctfidf_max_features": ctfidf_max_features,
        "ctfidf_ngram_range": list(ctfidf_ngram_range),
    }

    labels_file = output_dir / "cluster_labels.csv"
    topics_file = output_dir / "cluster_topics.csv"
    cfg_file = output_dir / "clustering_config.json"

    if not force_recompute and labels_file.exists() and topics_file.exists() and cfg_file.exists():
        saved_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        pipeline_cfg = {k: v for k, v in saved_cfg.items() if k != "quality_metrics"}
        if pipeline_cfg == config:
            labels_df = pd.read_csv(labels_file, dtype=str, keep_default_na=False)
            labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
            labels_df["corpus_id"] = labels_df["corpus_id"].astype(int)
            topics_df = pd.read_csv(topics_file, dtype=str, keep_default_na=False)
            topics_df["cluster_id"] = topics_df["cluster_id"].astype(int)
            metrics = saved_cfg.get("quality_metrics", {})
            return labels_df, topics_df, metrics

    embeddings, meta_df = load_embeddings_and_metadata(embeddings_dir)

    umap_5d, _ = reduce_with_umap(
        embeddings,
        n_components=umap_n_components_cluster,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist_cluster,
        metric=umap_metric,
        random_state=umap_random_state,
    )

    result = cluster_with_hdbscan(
        umap_5d,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
    )

    topics_df = extract_cluster_topics(
        meta_df,
        result.labels,
        top_n=ctfidf_top_n,
        max_features=ctfidf_max_features,
        ngram_range=ctfidf_ngram_range,
    )

    metrics = compute_quality_metrics(umap_5d, result.labels, result.clusterer)

    save_clustering_artefacts(
        meta_df=meta_df,
        result=result,
        topics_df=topics_df,
        metrics=metrics,
        output_dir=output_dir,
        config=config,
    )

    labels_df = pd.read_csv(output_dir / "cluster_labels.csv", dtype=str, keep_default_na=False)
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    labels_df["corpus_id"] = labels_df["corpus_id"].astype(int)

    return labels_df, topics_df, metrics
