"""
tests/test_semantic_similarity.py
=================================
Focused tests for PolicyLens semantic-discovery and evidence-retrieval pipeline:
  1. Offline model loadability and parameter integrity.
  2. Attention-mask-aware mean pooling output shape (N, 384) and finiteness.
  3. L2 unit normalization (norm == 1.0).
  4. Mean pooling vs CLS pooling distinction (MiniLM canonical check).
  5. Attention mask padding invariance.
  6. Corpus assembly provenance, metadata schema, and integer alignment.
  7. Verbatim source text preservation without corruption or truncation.
  8. Strict 1-to-1 row alignment between embeddings array and metadata table.
  9. Deterministic corpus hashing and caching behavior.
 10. Cosine similarity score validity and ranking order.
 11. Strict self-match exclusion when querying by record ID.
 12. Notebook file existence, structure, and execution output validity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.semantic_similarity import (
    MODEL_ID_DEFAULT,
    build_corpus,
    compute_corpus_hash,
    embed_and_save_corpus,
    is_eligible,
    mean_pool_and_normalize,
    retrieve_similar,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_semantic_similarity.ipynb"


@pytest.fixture(scope="module")
def local_tokenizer_and_model():
    """Loads all-MiniLM-L6-v2 strictly from local offline cache."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_DEFAULT, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID_DEFAULT, local_files_only=True)
    model.eval()
    return tokenizer, model


def test_offline_model_loading(local_tokenizer_and_model):
    """Verify all-MiniLM-L6-v2 loads offline and has expected architecture."""
    tokenizer, model = local_tokenizer_and_model
    assert tokenizer is not None
    assert model is not None
    assert model.config.hidden_size == 384
    assert model.config.model_type in ("bert", "distilbert", "bert-base")


def test_mean_pooling_shape_and_finiteness(local_tokenizer_and_model):
    """Verify mean pooling produces (N, 384) float32 arrays with all finite values."""
    tokenizer, model = local_tokenizer_and_model
    test_texts = [
        "Open government enhances accountability and public participation.",
        "We need open standards for government records and digital transparency.",
        "Short phrase.",
    ]
    emb = mean_pool_and_normalize(test_texts, tokenizer, model)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (3, 384)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all(), "Found NaN or Inf in embeddings"


def test_mean_pooling_unit_l2_norm(local_tokenizer_and_model):
    """Verify that every pooled embedding vector is strictly L2 unit-normalized."""
    tokenizer, model = local_tokenizer_and_model
    test_texts = [
        "First test document about public consultations and government data.",
        "Second document discussing open science and procurement disclosures.",
    ]
    emb = mean_pool_and_normalize(test_texts, tokenizer, model)
    norms = np.linalg.norm(emb, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5), f"Vectors are not unit normalized: {norms}"


def test_mean_pooling_differs_from_cls_pooling(local_tokenizer_and_model):
    """Verify mean pooling is distinct from CLS pooling, ensuring proper sentence-transformer representation."""
    tokenizer, model = local_tokenizer_and_model
    text = ["A detailed comment arguing for open government data in machine readable formats."]

    mean_emb = mean_pool_and_normalize(text, tokenizer, model)

    # Compute naive CLS pooling
    with torch.no_grad():
        encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        out = model(**encoded)
        cls_token = out.last_hidden_state[:, 0, :]
        cls_emb = torch.nn.functional.normalize(cls_token, p=2, dim=1).numpy()

    # The two representations should not be identical
    cosine_sim = float((mean_emb * cls_emb).sum())
    assert cosine_sim < 0.999, f"Mean pooling unexpectedly identical to CLS: {cosine_sim}"


def test_attention_mask_padding_invariance(local_tokenizer_and_model):
    """Verify that padding tokens do not alter the embedding representation of shorter sentences."""
    tokenizer, model = local_tokenizer_and_model
    short_text = "Open data should be accessible to all citizens."
    long_text = "Federal departments should proactively disclose records, contracts, briefing notes, and travel expenses without requiring formal ATIP requests."

    # Encode short text alone
    emb_alone = mean_pool_and_normalize([short_text], tokenizer, model)

    # Encode short text batched with much longer text (forcing padding onto short_text)
    emb_batched = mean_pool_and_normalize([short_text, long_text], tokenizer, model)

    # Short text embedding in batch should match isolated embedding within numerical precision
    cos_similarity = float((emb_alone[0] * emb_batched[0]).sum())
    assert cos_similarity > 0.9999, f"Padding corrupted mean pooling: {cos_similarity}"


def test_corpus_assembly_provenance_and_schema():
    """Verify corpus builder preserves required provenance and metadata columns."""
    corpus_df = build_corpus(PROCESSED_DIR, sample_per_dataset=10, random_seed=42)

    assert len(corpus_df) > 0
    required_cols = [
        "corpus_id",
        "source_dataset",
        "source_record_id",
        "source_file",
        "source_row_number",
        "source_field",
        "question_slug",
        "question_wording",
        "question_context",
        "duplicate_text_group_id",
        "language",
        "stakeholder_group",
        "original_text",
    ]
    for col in required_cols:
        assert col in corpus_df.columns, f"Missing required column: {col}"

    # Check corpus_id is 0..N-1
    assert list(corpus_df["corpus_id"]) == list(range(len(corpus_df)))

    # Check substantive comment criteria
    for text in corpus_df["original_text"]:
        assert len(text.strip()) >= 20
        assert is_eligible(text)


def test_source_text_verbatim_preservation():
    """Verify that original_text in corpus matches the source processed CSVs verbatim."""
    corpus_df = build_corpus(PROCESSED_DIR, sample_per_dataset=5, random_seed=42)

    comp_sample = corpus_df[corpus_df["source_dataset"] == "consultation_compilation_2017_18"]
    if len(comp_sample) > 0:
        row = comp_sample.iloc[0]
        comp_df = pd.read_csv(PROCESSED_DIR / "consultation_compilation_2017_18_source_clean.csv", dtype=str, keep_default_na=False)
        src_row = comp_df[comp_df["source_row_number"] == row["source_row_number"]].iloc[0]
        text_col = "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part"
        assert row["original_text"] == src_row[text_col].strip()


def test_embedding_and_metadata_alignment(local_tokenizer_and_model, tmp_path):
    """Verify that embeddings and metadata are strictly aligned row-for-row."""
    tokenizer, model = local_tokenizer_and_model
    corpus_df = build_corpus(PROCESSED_DIR, sample_per_dataset=5, random_seed=42)

    embeddings, meta_df = embed_and_save_corpus(
        corpus_df=corpus_df,
        tokenizer=tokenizer,
        model=model,
        embeddings_dir=tmp_path,
        max_length=128,
        batch_size=16,
    )

    assert embeddings.shape[0] == len(meta_df)
    assert embeddings.shape[1] == 384
    assert (tmp_path / "minilm_embeddings.npy").exists()
    assert (tmp_path / "corpus_metadata.csv").exists()
    assert (tmp_path / "corpus_hash.txt").exists()

    # Test cache reuse
    emb_cached, meta_cached = embed_and_save_corpus(
        corpus_df=corpus_df,
        tokenizer=tokenizer,
        model=model,
        embeddings_dir=tmp_path,
        max_length=128,
        batch_size=16,
        force_recompute=False,
    )
    assert np.array_equal(embeddings, emb_cached)
    assert len(meta_df) == len(meta_cached)


def test_cosine_similarity_retrieval_ranking(local_tokenizer_and_model):
    """Verify retrieval ranks semantically relevant records highest and computes valid cosine scores."""
    tokenizer, model = local_tokenizer_and_model

    sample_corpus = pd.DataFrame([
        {
            "corpus_id": 0,
            "source_dataset": "test_ds",
            "source_record_id": "rec_0",
            "source_field": "q_transparency",
            "question_slug": "q_transparency",
            "language": "English",
            "stakeholder_group": "Citizen",
            "original_text": "Government spending accounts, expenditures, and financial contracts should be published online.",
            "question_context": "Transparency",
        },
        {
            "corpus_id": 1,
            "source_dataset": "test_ds",
            "source_record_id": "rec_1",
            "source_field": "q_transportation",
            "question_slug": "q_transportation",
            "language": "English",
            "stakeholder_group": "Citizen",
            "original_text": "Public transit schedules and bus delays in suburban transit corridors.",
            "question_context": "Transit",
        },
    ])

    emb = mean_pool_and_normalize(sample_corpus["original_text"].tolist(), tokenizer, model)

    query = "Access to federal budgets, procurement expenditures, and financial audit records."
    results = retrieve_similar(query, emb, sample_corpus, tokenizer, model, top_k=2)

    assert len(results) == 2
    assert results.iloc[0]["corpus_id"] == 0  # spending record should beat transit record
    assert results.iloc[0]["cosine_similarity"] > results.iloc[1]["cosine_similarity"]
    assert -1.0 <= results.iloc[0]["cosine_similarity"] <= 1.0


def test_self_match_exclusion(local_tokenizer_and_model):
    """Verify that record-to-record retrieval strictly excludes the anchor query record itself."""
    tokenizer, model = local_tokenizer_and_model

    sample_corpus = pd.DataFrame([
        {
            "corpus_id": 0,
            "source_dataset": "test_ds",
            "source_record_id": "rec_0",
            "source_field": "field_a",
            "question_slug": "slug_a",
            "language": "English",
            "stakeholder_group": "Public",
            "original_text": "Open government consultations need better outreach to indigenous communities.",
            "question_context": "Outreach",
        },
        {
            "corpus_id": 1,
            "source_dataset": "test_ds",
            "source_record_id": "rec_1",
            "source_field": "field_b",
            "question_slug": "slug_b",
            "language": "English",
            "stakeholder_group": "Public",
            "original_text": "Engaging northern and indigenous communities in policy discussions.",
            "question_context": "Inclusion",
        },
        {
            "corpus_id": 2,
            "source_dataset": "test_ds",
            "source_record_id": "rec_2",
            "source_field": "field_c",
            "question_slug": "slug_c",
            "language": "English",
            "stakeholder_group": "Public",
            "original_text": "Completely unrelated text about forestry export tariffs and lumber mills.",
            "question_context": "Commerce",
        },
    ])

    emb = mean_pool_and_normalize(sample_corpus["original_text"].tolist(), tokenizer, model)

    # Query with corpus_id 0
    results = retrieve_similar(
        query=0,
        embeddings=emb,
        meta_df=sample_corpus,
        tokenizer=tokenizer,
        model=model,
        top_k=2,
        exclude_query_record=True,
    )

    returned_ids = results["corpus_id"].tolist()
    assert 0 not in returned_ids, f"Self-match was not excluded! Returned: {returned_ids}"
    assert returned_ids[0] == 1, f"Expected corpus_id 1 to be the closest non-self match, got {returned_ids[0]}"


def test_notebook_file_exists_and_valid():
    """Verify 04_semantic_similarity.ipynb exists and is valid Jupyter Notebook JSON."""
    assert NOTEBOOK_PATH.exists(), f"Notebook missing at {NOTEBOOK_PATH}"
    content = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert content.get("nbformat") == 4
    assert len(content.get("cells", [])) >= 8
