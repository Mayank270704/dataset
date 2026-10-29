"""
models/common
=============
Shared constants, paths, and utilities across PolicyLens models.
"""
from models.common.config import (
    ANNOTATION_DIR,
    CLUSTERING_DIR,
    DATA_DIR,
    DEFAULT_MULTILINGUAL_MODEL,
    DEFAULT_SENTENCE_MODEL,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SCRIPTS_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EMBEDDINGS_DIR",
    "CLUSTERING_DIR",
    "ANNOTATION_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "NOTEBOOKS_DIR",
    "SCRIPTS_DIR",
    "DEFAULT_SENTENCE_MODEL",
    "DEFAULT_MULTILINGUAL_MODEL",
]
