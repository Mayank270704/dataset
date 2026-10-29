"""
tests/test_concern_clustering.py
=================================
Unit and integration tests for scripts/concern_clustering.py.

Tests do not require the full 450-row corpus embeddings; they use
deterministic synthetic fixtures to stay fast and offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.concern_clustering import (
    NOISE_LABEL,
    ClusteringResult,
    _ctfidf_matrix,
    cluster_with_hdbscan,
    compute_quality_metrics,
    extract_cluster_topics,
    load_embeddings_and_metadata,
    reduce_with_umap,
    run_clustering,
    save_clustering_artefacts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DOCS = 60   # enough for HDBSCAN to find real clusters
DIM = 384
N_REAL_CLUSTERS = 3


@pytest.fixture(scope="module")
def synthetic_embeddings() -> np.ndarray:
    """
    L2-normalised synthetic embeddings with 3 clear clusters (20 docs each).
    """
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((N_REAL_CLUSTERS, DIM)).astype(np.float32)
    # L2-normalise centres
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    parts = []
    for c in centres:
        noise = rng.standard_normal((N_DOCS // N_REAL_CLUSTERS, DIM)).astype(np.float32) * 0.05
        block = c + noise
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        parts.append(block)
    return np.vstack(parts)  # (60, 384)


@pytest.fixture(scope="module")
def synthetic_meta(synthetic_embeddings) -> pd.DataFrame:
    n = len(synthetic_embeddings)
    texts = (
        ["open data transparency government"] * 20
        + ["privacy security personal information"] * 20
        + ["participation engagement citizens democracy"] * 20
    )
    return pd.DataFrame({
        "corpus_id": list(range(n)),
        "source_dataset": ["individual_comments_2016_18"] * n,
        "source_record_id": [str(i) for i in range(n)],
        "source_file": ["test.csv"] * n,
        "source_row_number": [str(i) for i in range(n)],
        "source_field": ["q8"] * n,
        "question_slug": ["q8"] * n,
        "question_wording": ["Test question"] * n,
        "question_context": [""] * n,
        "duplicate_text_group_id": [""] * n,
        "language": (["English"] * 10 + ["French"] * 10) * 3,
        "stakeholder_group": [""] * n,
        "original_text": texts,
    })


@pytest.fixture(scope="module")
def tmp_embeddings_dir(tmp_path_factory, synthetic_embeddings, synthetic_meta):
    """Temporary directory with embeddings + metadata files."""
    d = tmp_path_factory.mktemp("embeddings")
    np.save(d / "minilm_embeddings.npy", synthetic_embeddings)
    synthetic_meta.to_csv(d / "corpus_metadata.csv", index=False)
    return d


@pytest.fixture(scope="module")
def tmp_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("clustering_output")


# ---------------------------------------------------------------------------
# load_embeddings_and_metadata
# ---------------------------------------------------------------------------


class TestLoadEmbeddingsAndMetadata:
    def test_returns_correct_shapes(self, tmp_embeddings_dir, synthetic_embeddings, synthetic_meta):
        emb, meta = load_embeddings_and_metadata(tmp_embeddings_dir)
        assert emb.shape == (N_DOCS, DIM)
        assert len(meta) == N_DOCS

    def test_corpus_id_aligned(self, tmp_embeddings_dir):
        emb, meta = load_embeddings_and_metadata(tmp_embeddings_dir)
        assert list(meta["corpus_id"]) == list(range(N_DOCS))

    def test_missing_embeddings_file_raises(self, tmp_path):
        meta_only = tmp_path / "emb_dir"
        meta_only.mkdir()
        (meta_only / "corpus_metadata.csv").write_text("corpus_id\n0\n")
        with pytest.raises(FileNotFoundError, match="Embeddings not found"):
            load_embeddings_and_metadata(meta_only)

    def test_missing_metadata_file_raises(self, tmp_path):
        emb_only = tmp_path / "emb_dir2"
        emb_only.mkdir()
        np.save(emb_only / "minilm_embeddings.npy", np.zeros((5, 384)))
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            load_embeddings_and_metadata(emb_only)

    def test_alignment_mismatch_raises(self, tmp_path):
        d = tmp_path / "mismatch"
        d.mkdir()
        np.save(d / "minilm_embeddings.npy", np.zeros((10, 384)))
        pd.DataFrame({"corpus_id": [0, 1]}).to_csv(d / "corpus_metadata.csv", index=False)
        with pytest.raises(ValueError, match="Alignment mismatch"):
            load_embeddings_and_metadata(d)


# ---------------------------------------------------------------------------
# reduce_with_umap
# ---------------------------------------------------------------------------


class TestReduceWithUmap:
    def test_output_shape_5d(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        assert reduced.shape == (N_DOCS, 5)

    def test_output_shape_2d(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=2)
        assert reduced.shape == (N_DOCS, 2)

    def test_output_dtype_float32(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        assert reduced.dtype == np.float32

    def test_finite_output(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        assert np.isfinite(reduced).all()

    def test_reducer_object_returned(self, synthetic_embeddings):
        _, reducer = reduce_with_umap(synthetic_embeddings, n_components=5)
        import umap as umap_lib
        assert isinstance(reducer, umap_lib.UMAP)


# ---------------------------------------------------------------------------
# cluster_with_hdbscan
# ---------------------------------------------------------------------------


class TestClusterWithHdbscan:
    @pytest.fixture(scope="class")
    def umap5d(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        return reduced

    def test_returns_clustering_result(self, umap5d):
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        assert isinstance(result, ClusteringResult)

    def test_labels_length(self, umap5d):
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        assert len(result.labels) == N_DOCS

    def test_probabilities_in_0_1(self, umap5d):
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        assert (result.probabilities >= 0).all()
        assert (result.probabilities <= 1).all()

    def test_umap_2d_shape(self, umap5d):
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        assert result.umap_2d.shape == (N_DOCS, 2)

    def test_finds_real_clusters_in_well_separated_data(self, umap5d):
        """With very tight clusters in the fixture, HDBSCAN should find ≥1 cluster."""
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        n_real = len(set(result.labels)) - (1 if NOISE_LABEL in result.labels else 0)
        assert n_real >= 1, f"Expected ≥1 cluster, got {n_real}"

    def test_labels_dtype_int(self, umap5d):
        result = cluster_with_hdbscan(umap5d, min_cluster_size=5, min_samples=3)
        assert result.labels.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# _ctfidf_matrix
# ---------------------------------------------------------------------------


class TestCtfidfMatrix:
    def test_output_shapes(self):
        docs = ["open data government", "privacy security"]
        mat, vocab = _ctfidf_matrix(docs)
        assert mat.shape[0] == 2
        assert mat.shape[1] == len(vocab)

    def test_no_nan_inf(self):
        docs = ["alpha beta gamma", "delta epsilon"]
        mat, _ = _ctfidf_matrix(docs)
        assert np.isfinite(mat).all()

    def test_empty_doc_handled(self):
        docs = ["alpha beta", " "]
        mat, vocab = _ctfidf_matrix(docs)
        assert mat.shape[0] == 2

    def test_single_class(self):
        docs = ["only one class here"]
        mat, vocab = _ctfidf_matrix(docs)
        assert mat.shape[0] == 1

    def test_class_specific_term_scores_highest(self):
        """'open' appears only in class 0 — it should score 0 in class 1."""
        docs = ["open government data transparency", "privacy security personal"]
        mat, vocab = _ctfidf_matrix(docs)
        vocab_list = list(vocab)
        if "open" in vocab_list:
            idx = vocab_list.index("open")
            assert mat[0, idx] > 0
            assert mat[1, idx] == 0.0


# ---------------------------------------------------------------------------
# extract_cluster_topics
# ---------------------------------------------------------------------------


class TestExtractClusterTopics:
    @pytest.fixture(scope="class")
    def dummy_labels_and_meta(self):
        meta = pd.DataFrame({
            "corpus_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "source_dataset": ["ds1"] * 10,
            "language": ["English"] * 6 + ["French"] * 4,
            "original_text": [
                "open government data", "open data transparency",
                "open data portal",
                "privacy security", "security personal data", "privacy rights",
                "participation citizens", "citizen engagement", "democracy vote",
                "gouvernement ouvert",
            ],
        })
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, NOISE_LABEL])
        return labels, meta

    def test_returns_dataframe(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        assert isinstance(topics, pd.DataFrame)

    def test_has_correct_columns(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        for col in ["cluster_id", "size", "top_keywords", "language_distribution"]:
            assert col in topics.columns, f"Missing column: {col}"

    def test_noise_row_present(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        assert NOISE_LABEL in topics["cluster_id"].values

    def test_noise_row_size_correct(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        noise_row = topics[topics["cluster_id"] == NOISE_LABEL].iloc[0]
        assert int(noise_row["size"]) == 1

    def test_real_cluster_sizes_correct(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        for cid in [0, 1, 2]:
            row = topics[topics["cluster_id"] == cid].iloc[0]
            assert int(row["size"]) == 3

    def test_top_keywords_not_empty_for_real_clusters(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        for cid in [0, 1, 2]:
            row = topics[topics["cluster_id"] == cid].iloc[0]
            assert len(str(row["top_keywords"])) > 0

    def test_all_noise_no_real_clusters(self):
        """When all docs are noise, output still has a noise row."""
        meta = pd.DataFrame({
            "corpus_id": [0, 1],
            "source_dataset": ["ds"] * 2,
            "language": ["English"] * 2,
            "original_text": ["alpha", "beta"],
        })
        labels = np.array([NOISE_LABEL, NOISE_LABEL])
        topics = extract_cluster_topics(meta, labels)
        assert NOISE_LABEL in topics["cluster_id"].values
        assert len(topics[topics["cluster_id"] != NOISE_LABEL]) == 0

    def test_language_distribution_json_valid(self, dummy_labels_and_meta):
        labels, meta = dummy_labels_and_meta
        topics = extract_cluster_topics(meta, labels)
        for _, row in topics.iterrows():
            parsed = json.loads(row["language_distribution"])
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# compute_quality_metrics
# ---------------------------------------------------------------------------


class TestComputeQualityMetrics:
    @pytest.fixture(scope="class")
    def cluster_result(self, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        return cluster_with_hdbscan(reduced, min_cluster_size=5, min_samples=3), reduced

    def test_returns_dict(self, cluster_result):
        result, umap5d = cluster_result
        metrics = compute_quality_metrics(umap5d, result.labels, result.clusterer)
        assert isinstance(metrics, dict)

    def test_required_keys(self, cluster_result):
        result, umap5d = cluster_result
        metrics = compute_quality_metrics(umap5d, result.labels, result.clusterer)
        for key in ["n_clusters", "n_noise", "noise_fraction"]:
            assert key in metrics, f"Missing key: {key}"

    def test_noise_fraction_in_0_1(self, cluster_result):
        result, umap5d = cluster_result
        metrics = compute_quality_metrics(umap5d, result.labels, result.clusterer)
        assert 0.0 <= metrics["noise_fraction"] <= 1.0

    def test_silhouette_present_when_multiple_clusters(self, cluster_result):
        result, umap5d = cluster_result
        metrics = compute_quality_metrics(umap5d, result.labels, result.clusterer)
        if metrics["n_clusters"] >= 2:
            assert "silhouette_score" in metrics

    def test_all_noise_no_silhouette(self):
        """When all labels are noise, silhouette_score should be None."""
        umap5d = np.random.default_rng(0).standard_normal((10, 5)).astype(np.float32)
        labels = np.full(10, NOISE_LABEL)

        class FakeClusterer:
            cluster_persistence_ = []

        metrics = compute_quality_metrics(umap5d, labels, FakeClusterer())
        assert metrics["silhouette_score"] is None


# ---------------------------------------------------------------------------
# save_clustering_artefacts
# ---------------------------------------------------------------------------


class TestSaveClusteringArtefacts:
    def test_files_written(self, tmp_path, synthetic_meta, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        result = cluster_with_hdbscan(reduced, min_cluster_size=5, min_samples=3)
        topics_df = extract_cluster_topics(synthetic_meta, result.labels)
        metrics = compute_quality_metrics(reduced, result.labels, result.clusterer)
        config = {"test": True}

        save_clustering_artefacts(
            meta_df=synthetic_meta,
            result=result,
            topics_df=topics_df,
            metrics=metrics,
            output_dir=tmp_path,
            config=config,
        )

        assert (tmp_path / "cluster_labels.csv").exists()
        assert (tmp_path / "cluster_topics.csv").exists()
        assert (tmp_path / "umap_2d.npy").exists()
        assert (tmp_path / "clustering_config.json").exists()

    def test_cluster_labels_has_expected_columns(self, tmp_path, synthetic_meta, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        result = cluster_with_hdbscan(reduced, min_cluster_size=5, min_samples=3)
        topics_df = extract_cluster_topics(synthetic_meta, result.labels)
        metrics = compute_quality_metrics(reduced, result.labels, result.clusterer)

        save_clustering_artefacts(
            meta_df=synthetic_meta,
            result=result,
            topics_df=topics_df,
            metrics=metrics,
            output_dir=tmp_path,
            config={},
        )

        labels_df = pd.read_csv(tmp_path / "cluster_labels.csv")
        for col in ["corpus_id", "cluster_id", "cluster_probability", "umap_x", "umap_y"]:
            assert col in labels_df.columns, f"Missing column: {col}"

    def test_umap_2d_shape_matches(self, tmp_path, synthetic_meta, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        result = cluster_with_hdbscan(reduced, min_cluster_size=5, min_samples=3)
        topics_df = extract_cluster_topics(synthetic_meta, result.labels)
        metrics = compute_quality_metrics(reduced, result.labels, result.clusterer)

        save_clustering_artefacts(
            meta_df=synthetic_meta,
            result=result,
            topics_df=topics_df,
            metrics=metrics,
            output_dir=tmp_path,
            config={},
        )

        arr = np.load(tmp_path / "umap_2d.npy")
        assert arr.shape == (N_DOCS, 2)

    def test_config_json_contains_metrics(self, tmp_path, synthetic_meta, synthetic_embeddings):
        reduced, _ = reduce_with_umap(synthetic_embeddings, n_components=5)
        result = cluster_with_hdbscan(reduced, min_cluster_size=5, min_samples=3)
        topics_df = extract_cluster_topics(synthetic_meta, result.labels)
        metrics = compute_quality_metrics(reduced, result.labels, result.clusterer)

        save_clustering_artefacts(
            meta_df=synthetic_meta,
            result=result,
            topics_df=topics_df,
            metrics=metrics,
            output_dir=tmp_path,
            config={"param": 42},
        )

        cfg = json.loads((tmp_path / "clustering_config.json").read_text())
        assert "quality_metrics" in cfg
        assert cfg["param"] == 42


# ---------------------------------------------------------------------------
# run_clustering (end-to-end)
# ---------------------------------------------------------------------------


class TestRunClustering:
    def test_end_to_end_returns_three_values(self, tmp_embeddings_dir, tmp_path):
        labels_df, topics_df, metrics = run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )
        assert isinstance(labels_df, pd.DataFrame)
        assert isinstance(topics_df, pd.DataFrame)
        assert isinstance(metrics, dict)

    def test_labels_df_row_count_matches_corpus(self, tmp_embeddings_dir, tmp_path):
        labels_df, _, _ = run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )
        assert len(labels_df) == N_DOCS

    def test_cache_hit_returns_same_results(self, tmp_embeddings_dir, tmp_path):
        """Second call with same hyperparams should load from cache."""
        kwargs = dict(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
        )
        labels_df1, topics_df1, _ = run_clustering(**kwargs, force_recompute=True)
        labels_df2, topics_df2, _ = run_clustering(**kwargs, force_recompute=False)

        assert list(labels_df1["cluster_id"]) == list(labels_df2["cluster_id"])

    def test_force_recompute_overwrites_cache(self, tmp_embeddings_dir, tmp_path):
        """force_recompute=True should rerun the pipeline without errors."""
        run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )
        # Should not raise
        run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )

    def test_output_files_exist(self, tmp_embeddings_dir, tmp_path):
        run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )
        assert (tmp_path / "cluster_labels.csv").exists()
        assert (tmp_path / "cluster_topics.csv").exists()
        assert (tmp_path / "umap_2d.npy").exists()
        assert (tmp_path / "clustering_config.json").exists()

    def test_cluster_id_column_present_in_labels_df(self, tmp_embeddings_dir, tmp_path):
        labels_df, _, _ = run_clustering(
            embeddings_dir=tmp_embeddings_dir,
            output_dir=tmp_path,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            force_recompute=True,
        )
        assert "cluster_id" in labels_df.columns
