"""
models/semantic_similarity
==========================
PolicyLens semantic-discovery, sentence embeddings, and evidence retrieval.
"""
from models.semantic_similarity.minilm import (
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
