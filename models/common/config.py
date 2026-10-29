"""
models/common/config.py
=======================
Centralized filesystem paths and runtime configurations for PolicyLens models.
"""
from pathlib import Path

# Resolve project root from models/common/config.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"
ANNOTATION_DIR = DATA_DIR / "annotation"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Default pretrained model identifiers
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MULTILINGUAL_MODEL = "google/muril-base-cased"
