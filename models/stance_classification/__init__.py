"""
models/stance_classification
============================
PolicyLens Stance Classification Subpackage.

Implements reproducible baseline pipelines and interfaces:
  - TfidfStanceClassifier (Logistic Regression & Linear SVM)
  - Leakage prevention utilities (group-aware splitting, fit-on-train-only)
  - Strict human label validation guards
"""
from models.stance_classification.baselines import (
    MIN_VERIFIED_SAMPLES_FOR_TRAINING,
    VALID_STANCES,
    TfidfStanceClassifier,
    train_test_split_without_leakage,
    validate_labeled_data,
)

__all__ = [
    "VALID_STANCES",
    "MIN_VERIFIED_SAMPLES_FOR_TRAINING",
    "TfidfStanceClassifier",
    "validate_labeled_data",
    "train_test_split_without_leakage",
]
