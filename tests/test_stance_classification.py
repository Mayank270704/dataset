"""
tests/test_stance_classification.py
===================================
Tests for PolicyLens stance classification baseline models and data leakage guardrails.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from models.stance_classification.baselines import (
    VALID_STANCES,
    MIN_VERIFIED_SAMPLES_FOR_TRAINING,
    TfidfStanceClassifier,
    train_test_split_without_leakage,
    validate_labeled_data,
)

ROOT = Path(__file__).resolve().parent.parent
PILOT_PATH = ROOT / "data" / "annotation" / "pilot_annotation_50.csv"
EXPANSION_POOL_PATH = ROOT / "data" / "annotation" / "expansion_pool_400.csv"


def test_validate_labeled_data_rejects_unlabeled_pilot():
    """Unlabeled pilot dataset must be rejected by the validation guardrail."""
    pilot_df = pd.read_csv(PILOT_PATH, dtype=str, keep_default_na=False)
    with pytest.raises(ValueError, match="0 valid labels found"):
        validate_labeled_data(pilot_df, label_col="stance")


def test_validate_labeled_data_rejects_insufficient_samples():
    """Datasets with fewer than MIN_VERIFIED_SAMPLES_FOR_TRAINING must be rejected."""
    small_df = pd.DataFrame({
        "original_text": [f"text {i}" for i in range(5)],
        "stance": ["Support", "Oppose", "Support", "Oppose", "Support"],
    })
    with pytest.raises(ValueError, match="at least 10 are required"):
        validate_labeled_data(small_df, label_col="stance")


def test_validate_labeled_data_rejects_single_class():
    """Datasets with only 1 class must be rejected."""
    mono_df = pd.DataFrame({
        "original_text": [f"text {i}" for i in range(15)],
        "stance": ["Support"] * 15,
    })
    with pytest.raises(ValueError, match="found only 1 unique stance class"):
        validate_labeled_data(mono_df, label_col="stance")


def test_leakage_prevention_duplicate_groups_not_split():
    """Duplicate-text groups must stay strictly within either train or test, never both."""
    df = pd.DataFrame({
        "original_text": [f"comment {i}" for i in range(20)],
        "stance": ["Support", "Oppose"] * 10,
        "duplicate_text_group_id": ["group_A", "group_A", "group_B", "group_B"] + [""] * 16,
    })

    X_train, X_test, y_train, y_test = train_test_split_without_leakage(
        df,
        text_col="original_text",
        label_col="stance",
        group_col="duplicate_text_group_id",
        test_size=0.3,
        random_state=42,
    )

    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    # Check that group_A indices (0, 1) are not split across train and test
    group_a_train = 0 in train_indices or 1 in train_indices
    group_a_test = 0 in test_indices or 1 in test_indices
    assert not (group_a_train and group_a_test), "group_A was split across train and test!"

    # Check that group_B indices (2, 3) are not split across train and test
    group_b_train = 2 in train_indices or 3 in train_indices
    group_b_test = 2 in test_indices or 3 in test_indices
    assert not (group_b_train and group_b_test), "group_B was split across train and test!"


def test_tfidf_vectorizer_fits_train_only():
    """Vocabulary must be learned exclusively from training split."""
    train_texts = ["open data is essential for transparent government"]
    test_texts = ["completely unique words that never appeared during training"]

    clf = TfidfStanceClassifier(classifier_type="logistic_regression")
    df = pd.DataFrame({
        "original_text": ["open government data"] * 10 + ["closed confidential records"] * 10,
        "stance": ["Support"] * 10 + ["Oppose"] * 10,
        "duplicate_text_group_id": [""] * 20,
    })

    metrics = clf.fit_and_evaluate(df, test_size=0.3)
    vocab = set(clf.vectorizer.vocabulary_.keys())

    # Verify vectorizer is fitted
    assert len(vocab) > 0
    assert metrics["vocabulary_size"] == len(vocab)
    assert metrics["macro_f1"] >= 0.0
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics


def test_tfidf_linear_svm_runs_and_evaluates():
    """Linear SVM baseline must compute metrics and confusion matrix correctly."""
    clf = TfidfStanceClassifier(classifier_type="linear_svm")
    df = pd.DataFrame({
        "original_text": [
            "we strongly support open government and data access",
            "open government initiatives are crucial and wonderful",
            "we fully endorse publishing all public records",
            "great proposal for open dialogue and transparency",
            "transparency and open data are very helpful",
            "we disagree with the consultation and oppose this plan",
            "this policy is wasteful, ineffective, and unacceptable",
            "we strongly reject government surveillance and secrecy",
            "this open government initiative is flawed and counterproductive",
            "we do not want this consultation policy implemented",
        ] * 2,
        "stance": (["Support"] * 5 + ["Oppose"] * 5) * 2,
        "duplicate_text_group_id": [""] * 20,
    })

    metrics = clf.fit_and_evaluate(df, test_size=0.3)
    assert metrics["model_type"] == "linear_svm"
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["classes"]) == 2

    # Predict on new unseen text
    preds = clf.predict(["we support open data", "we oppose this policy"])
    assert len(preds) == 2
    assert set(preds).issubset({"Support", "Oppose"})


def test_expansion_pool_integrity():
    """Expansion pool must contain 400 records and leave annotation fields blank."""
    assert EXPANSION_POOL_PATH.exists(), f"Expansion pool missing: {EXPANSION_POOL_PATH}"
    df = pd.read_csv(EXPANSION_POOL_PATH, dtype=str, keep_default_na=False)
    assert len(df) == 400

    # Human annotation fields must be entirely blank strings
    for col in ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]:
        non_blank = df[df[col].str.strip() != ""]
        assert len(non_blank) == 0, f"Expansion column {col} has non-blank entries"
