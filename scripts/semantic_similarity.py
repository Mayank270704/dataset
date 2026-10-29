"""
scripts/semantic_similarity.py
==============================
Compatibility wrapper for PolicyLens semantic-similarity and evidence-retrieval.

The core implementation has been migrated to `models.semantic_similarity`.
This script maintains full backward compatibility for existing scripts, tests,
and notebooks importing from `scripts.semantic_similarity`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.semantic_similarity import (
    COMPILATION_CTX_COL,
    COMPILATION_TEXT_COL,
    COMPILATION_THEME_COL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    GENERIC_FILLERS,
    INDIVIDUAL_FIELDS,
    MIN_TEXT_LEN,
    MODEL_ID_DEFAULT,
    QUESTIONNAIRE_FIELDS,
    build_corpus,
    compute_corpus_hash,
    embed_and_save_corpus,
    is_eligible,
    mean_pool_and_normalize,
    retrieve_similar,
)

__all__ = [
    "MODEL_ID_DEFAULT",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "MIN_TEXT_LEN",
    "GENERIC_FILLERS",
    "INDIVIDUAL_FIELDS",
    "COMPILATION_TEXT_COL",
    "COMPILATION_CTX_COL",
    "COMPILATION_THEME_COL",
    "QUESTIONNAIRE_FIELDS",
    "is_eligible",
    "mean_pool_and_normalize",
    "build_corpus",
    "compute_corpus_hash",
    "embed_and_save_corpus",
    "retrieve_similar",
]

if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    from models.common.config import EMBEDDINGS_DIR, PROCESSED_DATA_DIR

    print(f"Loading tokenizer & model: {MODEL_ID_DEFAULT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_DEFAULT)
    model = AutoModel.from_pretrained(MODEL_ID_DEFAULT)

    print("Building substantive consultation corpus...")
    corpus = build_corpus(PROCESSED_DATA_DIR, sample_per_dataset=150)
    print(f"Assembled {len(corpus)} records across 3 datasets.")

    embeddings, meta = embed_and_save_corpus(corpus, tokenizer, model, EMBEDDINGS_DIR)
    print(f"Embeddings shape: {embeddings.shape}")
    print("Semantic similarity pipeline ready.")
