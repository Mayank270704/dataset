"""Input and reproducibility safeguards for the isolated embedding comparison."""
from pathlib import Path

from scripts.compare_multilingual_embeddings import (
    BATCH_SIZE,
    MAX_LENGTH,
    UMAP_RANDOM_STATE,
    load_reference_sample,
)


def test_comparison_uses_the_saved_450_record_sample_in_order():
    metadata, embeddings, reference_labels = load_reference_sample()

    assert len(metadata) == 450
    assert embeddings.shape == (450, 384)
    assert metadata["corpus_id"].astype(int).tolist() == list(range(450))
    assert reference_labels["corpus_id"].astype(int).tolist() == list(range(450))
    assert metadata["original_text"].tolist() == reference_labels["original_text"].tolist()


def test_sample_does_not_claim_duplicate_aware_group_evaluation():
    metadata, _, _ = load_reference_sample()

    assert metadata["duplicate_text_group_id"].str.strip().eq("").all()


def test_comparison_reuses_fixed_embedding_and_clustering_parameters():
    assert MAX_LENGTH == 256
    assert BATCH_SIZE == 64
    assert UMAP_RANDOM_STATE == 42
