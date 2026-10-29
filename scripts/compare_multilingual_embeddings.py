"""Compare MiniLM and multilingual MiniLM on the saved PolicyLens 450 sample."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "data" / "comparisons" / "multilingual_embedding_450"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CLUSTERING_DIR = PROJECT_ROOT / "data" / "clustering"
BASE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH = 256
BATCH_SIZE = 64
DEVICE = "cpu"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.concern_clustering.clustering import (  # noqa: E402
    CTFIDF_MAX_FEATURES,
    CTFIDF_NGRAM_RANGE,
    CTFIDF_TOP_N_WORDS,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_METRIC,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_N_COMPONENTS_CLUSTER,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
    cluster_with_hdbscan,
    compute_quality_metrics,
    extract_cluster_topics,
    reduce_with_umap,
    save_clustering_artefacts,
)
from models.semantic_similarity.minilm import mean_pool_and_normalize  # noqa: E402


def package_versions() -> dict[str, str]:
    """Return versions needed to reproduce embeddings and clustering."""
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "pandas": "pandas",
        "umap-learn": "umap-learn",
        "hdbscan": "hdbscan",
        "scikit-learn": "scikit-learn",
    }
    return {
        label: importlib.metadata.version(distribution)
        for label, distribution in packages.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_sample() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and validate the existing sample, baseline vectors, and assignments."""
    metadata_path = EMBEDDINGS_DIR / "corpus_metadata.csv"
    embeddings_path = EMBEDDINGS_DIR / "minilm_embeddings.npy"
    labels_path = CLUSTERING_DIR / "cluster_labels.csv"

    for path in (metadata_path, embeddings_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved comparison input is missing: {path}")

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    baseline_embeddings = np.load(embeddings_path)
    reference_labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)

    required_columns = {
        "corpus_id",
        "source_dataset",
        "source_record_id",
        "source_field",
        "language",
        "duplicate_text_group_id",
        "original_text",
    }
    missing = sorted(required_columns - set(metadata.columns))
    if missing:
        raise ValueError(f"Sample metadata is missing required columns: {missing}")
    if len(metadata) != 450:
        raise ValueError(f"Expected the existing 450-record sample, found {len(metadata)}.")
    if baseline_embeddings.shape != (len(metadata), 384):
        raise ValueError(
            "Baseline embeddings must align with the saved 450 records and have 384 "
            f"dimensions; got {baseline_embeddings.shape}."
        )
    if not np.isfinite(baseline_embeddings).all():
        raise ValueError("Baseline embeddings contain non-finite values.")

    corpus_ids = metadata["corpus_id"].astype(int).tolist()
    if corpus_ids != list(range(len(metadata))):
        raise ValueError("Sample corpus_id values are not the ordered sequence 0..449.")
    if reference_labels["corpus_id"].astype(int).tolist() != corpus_ids:
        raise ValueError("Existing clustering labels do not align with sample metadata.")

    for column in ("source_dataset", "source_record_id", "source_field", "original_text"):
        if reference_labels[column].tolist() != metadata[column].tolist():
            raise ValueError(f"Existing clustering labels differ from metadata in {column}.")

    return metadata, baseline_embeddings.astype(np.float32, copy=False), reference_labels


def cluster_model(
    model_id: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    corpus_hash: str,
    versions: dict[str, str],
    embedding_source: str,
) -> dict[str, Any]:
    """Apply the shared UMAP/HDBSCAN/c-TF-IDF procedure and persist outputs."""
    reduced, _ = reduce_with_umap(
        embeddings,
        n_components=UMAP_N_COMPONENTS_CLUSTER,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST_CLUSTER,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    result = cluster_with_hdbscan(
        reduced,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    topics = extract_cluster_topics(
        metadata,
        result.labels,
        top_n=CTFIDF_TOP_N_WORDS,
        max_features=CTFIDF_MAX_FEATURES,
        ngram_range=CTFIDF_NGRAM_RANGE,
    )
    metrics = compute_quality_metrics(reduced, result.labels, result.clusterer)

    config: dict[str, Any] = {
        "model_id": model_id,
        "embedding_source": embedding_source,
        "embedding_dimension": int(embeddings.shape[1]),
        "sample_size": int(len(metadata)),
        "corpus_metadata_sha256": corpus_hash,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "random_seed": UMAP_RANDOM_STATE,
        "packages": versions,
        "clustering": {
            "umap_n_components_cluster": UMAP_N_COMPONENTS_CLUSTER,
            "umap_n_neighbors": UMAP_N_NEIGHBORS,
            "umap_min_dist_cluster": UMAP_MIN_DIST_CLUSTER,
            "umap_metric": UMAP_METRIC,
            "umap_random_state": UMAP_RANDOM_STATE,
            "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            "hdbscan_metric": HDBSCAN_METRIC,
            "hdbscan_cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
            "ctfidf_top_n": CTFIDF_TOP_N_WORDS,
            "ctfidf_max_features": CTFIDF_MAX_FEATURES,
            "ctfidf_ngram_range": list(CTFIDF_NGRAM_RANGE),
        },
        "quality_metrics": metrics,
    }
    save_clustering_artefacts(
        meta_df=metadata,
        result=result,
        topics_df=topics,
        metrics=metrics,
        output_dir=output_dir,
        config=config["clustering"],
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / "embeddings.npy", embeddings)

    labels = pd.read_csv(output_dir / "cluster_labels.csv", dtype=str, keep_default_na=False)
    cluster_summaries: list[dict[str, Any]] = []
    for cluster_id in sorted({int(value) for value in labels["cluster_id"]}):
        cluster_rows = labels[labels["cluster_id"].astype(int) == cluster_id]
        topic_row = topics[topics["cluster_id"] == cluster_id]
        representatives = (
            cluster_rows.assign(
                _probability=pd.to_numeric(cluster_rows["cluster_probability"])
            )
            .sort_values(["_probability", "corpus_id"], ascending=[False, True])
            .head(5)
        )
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(cluster_rows)),
                "language_distribution": {
                    str(key): int(value)
                    for key, value in cluster_rows["language"].value_counts().items()
                },
                "top_keywords": (
                    str(topic_row.iloc[0]["top_keywords"]) if not topic_row.empty else ""
                ),
                "representative_comments": [
                    {
                        "corpus_id": int(row["corpus_id"]),
                        "language": row["language"],
                        "source_dataset": row["source_dataset"],
                        "cluster_probability": float(row["cluster_probability"]),
                        "original_text": row["original_text"],
                    }
                    for _, row in representatives.iterrows()
                ],
            }
        )

    return {
        "model_id": model_id,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_source": embedding_source,
        "embedding_sha256": sha256_file(output_dir / "embeddings.npy"),
        "quality_metrics": metrics,
        "clusters": cluster_summaries,
    }


def run_comparison(output_dir: Path = COMPARISON_DIR) -> dict[str, Any]:
    """Run both embedding models on exactly the saved sample in its saved order."""
    if output_dir.exists():
        raise FileExistsError(
            f"Comparison output already exists; refusing to overwrite: {output_dir}"
        )

    metadata, baseline_embeddings, _ = load_reference_sample()
    versions = package_versions()
    corpus_hash = sha256_file(EMBEDDINGS_DIR / "corpus_metadata.csv")
    baseline_hash = sha256_file(EMBEDDINGS_DIR / "minilm_embeddings.npy")
    duplicate_ids = metadata["duplicate_text_group_id"].str.strip()
    duplicate_info = {
        "nonblank_records": int(duplicate_ids.ne("").sum()),
        "distinct_nonblank_group_ids": int(duplicate_ids[duplicate_ids.ne("")].nunique()),
        "repeated_nonblank_group_ids": int(
            (duplicate_ids[duplicate_ids.ne("")].value_counts() > 1).sum()
        ),
    }

    # Load and encode the second model before creating any comparison output.
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_MODEL_ID)
    model = AutoModel.from_pretrained(MULTILINGUAL_MODEL_ID)
    multilingual_embeddings = mean_pool_and_normalize(
        metadata["original_text"].tolist(),
        tokenizer,
        model,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    if multilingual_embeddings.shape != baseline_embeddings.shape:
        raise ValueError(
            "The two models produced different embedding shapes: "
            f"{baseline_embeddings.shape} vs {multilingual_embeddings.shape}."
        )

    output_dir.mkdir(parents=True)
    baseline_result = cluster_model(
        BASE_MODEL_ID,
        baseline_embeddings,
        metadata,
        output_dir / "all_minilm_l6_v2",
        corpus_hash,
        versions,
        f"Reused saved artifact ({baseline_hash})",
    )
    multilingual_result = cluster_model(
        MULTILINGUAL_MODEL_ID,
        multilingual_embeddings,
        metadata,
        output_dir / "paraphrase_multilingual_minilm_l12_v2",
        corpus_hash,
        versions,
        "Encoded from saved corpus_metadata.csv using mean_pool_and_normalize",
    )
    comparison = {
        "sample": {
            "record_count": int(len(metadata)),
            "corpus_id_first": int(metadata["corpus_id"].iloc[0]),
            "corpus_id_last": int(metadata["corpus_id"].iloc[-1]),
            "corpus_metadata_sha256": corpus_hash,
            "source_embedding_sha256": baseline_hash,
            "duplicate_group_ids": duplicate_info,
        },
        "reproducibility": {
            "packages": versions,
            "device": DEVICE,
            "embedding_max_length": MAX_LENGTH,
            "embedding_batch_size": BATCH_SIZE,
            "random_seed": UMAP_RANDOM_STATE,
        },
        "models": {
            "all_minilm_l6_v2": baseline_result,
            "paraphrase_multilingual_minilm_l12_v2": multilingual_result,
        },
        "interpretation_note": (
            "Cluster count, silhouette, and language mixing are descriptive and do not "
            "establish policy-concern coherence. Review representative source comments."
        ),
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARISON_DIR,
        help="New output directory; an existing directory is never overwritten.",
    )
    args = parser.parse_args()
    result = run_comparison(args.output_dir)
    print(f"Comparison artifacts saved to {args.output_dir}")
    for name, model_result in result["models"].items():
        metrics = model_result["quality_metrics"]
        print(
            f"{name}: clusters={metrics['n_clusters']}, "
            f"noise_fraction={metrics['noise_fraction']}"
        )


if __name__ == "__main__":
    main()
