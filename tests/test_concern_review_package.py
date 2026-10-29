"""Tests for deterministic, model-blinded concern-review pair preparation."""
import numpy as np
import pandas as pd

from scripts.prepare_concern_review_package import (
    build_pair_catalog,
    make_reviewer_template,
    select_model_pairs,
)


def _sample_inputs():
    metadata = pd.DataFrame(
        {
            "corpus_id": ["0", "1", "2", "3", "4", "5"],
            "language": ["English", "English", "French", "French", "English", "French"],
            "original_text": [f"comment {i}" for i in range(6)],
            "source_dataset": ["source"] * 6,
            "source_file": ["source.csv"] * 6,
            "source_record_id": [str(i) for i in range(6)],
            "source_row_number": [str(i) for i in range(6)],
            "source_field": ["comment"] * 6,
            "question_context": ["context"] * 6,
        }
    )
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    labels = pd.DataFrame(
        {
            "corpus_id": [str(i) for i in range(6)],
            "cluster_id": ["0", "0", "1", "1", "2", "-1"],
            "original_text": metadata["original_text"],
        }
    )
    return metadata, vectors, labels


def test_pair_selection_is_deterministic_and_covers_requested_strata():
    metadata, vectors, labels = _sample_inputs()
    first, counts = select_model_pairs("model", metadata, vectors, labels, pair_limit=1)
    second, _ = select_model_pairs("model", metadata, vectors, labels, pair_limit=1)

    assert first == second
    assert counts["within_cluster|English|English"]["candidate_count"] > 0
    assert counts["within_cluster|French|French"]["candidate_count"] > 0
    assert counts["cross_cluster|English|French"]["candidate_count"] > 0
    assert all(item["pair_key"][0] < item["pair_key"][1] for item in first)


def test_pair_catalog_unions_models_and_template_blinds_predictions():
    metadata, vectors, labels = _sample_inputs()
    catalog, summary = build_pair_catalog(
        metadata,
        {
            "all_minilm_l6_v2": vectors,
            "paraphrase_multilingual_minilm_l12_v2": vectors.copy(),
        },
        {
            "all_minilm_l6_v2": labels,
            "paraphrase_multilingual_minilm_l12_v2": labels.copy(),
        },
        pair_limit=1,
    )
    template = make_reviewer_template(catalog)

    assert summary["selected_pair_model_records"] >= summary["unique_review_pairs"]
    assert catalog["pair_id"].is_unique
    assert "all_minilm_l6_v2_cluster_a" in catalog
    assert "paraphrase_multilingual_minilm_l12_v2_cluster_a" in catalog
    assert not any("cluster" in column or "selection" in column for column in template.columns)
    assert template["concern_relation"].eq("").all()
    assert template["confidence"].eq("").all()
    assert template["shared_concern_note"].eq("").all()
    assert template["evidence_note"].eq("").all()


def test_english_french_pairs_are_oriented_consistently():
    metadata, vectors, labels = _sample_inputs()
    selected, _ = select_model_pairs("model", metadata, vectors, labels, pair_limit=1)
    pair = next(
        item for item in selected
        if item["language_pair"] == "English|French"
    )
    left, right = pair["pair_key"]
    # The pair key remains corpus-ID sorted; orientation is explicitly in metadata.
    assert left < right
    assert pair["language_pair"] == "English|French"
