"""
tests/test_sentiment140_baseline.py
===================================
Automated tests for Sentiment140 streaming baseline pipeline.

Validates:
1. Incremental partial_fit and prediction contracts.
2. Serialization roundtrip and inference fidelity.
3. Evaluation metric calculation and strict neutral separation.
4. Memory pressure safety error handling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.sentiment_analysis.streaming_baseline import (
    LABEL_MAPPING,
    MemoryPressureError,
    Sentiment140StreamingClassifier,
    evaluate_sentiment_model,
)

ROOT = Path(__file__).resolve().parent.parent


def test_classifier_incremental_fitting_and_prediction():
    """Verify partial_fit, binary predictions, and probability calibration."""
    clf = Sentiment140StreamingClassifier(n_features=2**12, random_state=42)
    assert not clf.is_fitted_

    texts_batch_1 = [
        "I love this amazing product!",
        "This is the worst experience ever.",
        "Great quality and fast delivery.",
        "Completely broken and terrible.",
    ]
    labels_batch_1 = [1, 0, 1, 0]

    clf.partial_fit(texts_batch_1, labels_batch_1)
    assert clf.is_fitted_

    # Second batch
    texts_batch_2 = [
        "Delightful and wonderful support.",
        "Horrible service, total waste of money.",
    ]
    labels_batch_2 = [1, 0]
    clf.partial_fit(texts_batch_2, labels_batch_2)

    # Test predictions with in-vocabulary sentiment tokens
    preds = clf.predict([
        "I love this great product",
        "This was a terrible and horrible experience",
    ])
    assert len(preds) == 2
    assert preds[0] == "positive"
    assert preds[1] == "negative"


    # Probabilities
    probs = clf.predict_proba(["I love it", "I hate it"])
    assert probs.shape == (2, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_classifier_serialization_roundtrip(tmp_path: Path):
    """Verify model save and load produces bit-for-bit identical predictions."""
    clf = Sentiment140StreamingClassifier(n_features=2**12, random_state=42)
    clf.partial_fit(["Good text", "Bad text"], [1, 0])

    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    assert model_file.is_file()

    loaded = Sentiment140StreamingClassifier.load(model_file)
    assert loaded.is_fitted_
    assert loaded.n_features == clf.n_features

    test_inputs = ["Good text", "Bad text", "Unknown text"]
    orig_probs = clf.predict_proba(test_inputs)
    loaded_probs = loaded.predict_proba(test_inputs)

    assert np.allclose(orig_probs, loaded_probs)


def test_evaluate_sentiment_model_neutral_separation(tmp_path: Path):
    """Verify that evaluate_sentiment_model isolates binary classes from neutral class."""
    clf = Sentiment140StreamingClassifier(n_features=2**12, random_state=42)
    clf.partial_fit(["happy day", "sad day"], [1, 0])

    # Create synthetic test file with 0, 2, and 4
    test_csv = tmp_path / "test_sample.csv"
    test_data = [
        ("0", "1", "date", "query", "user", "I am very sad and disappointed"),
        ("4", "2", "date", "query", "user", "I am so happy and delighted"),
        ("2", "3", "date", "query", "user", "Just sitting here at the station"),
        ("2", "4", "date", "query", "user", "Reading a book about history"),
    ]
    df = pd.DataFrame(test_data)
    df.to_csv(test_csv, header=False, index=False)

    results = evaluate_sentiment_model(clf, test_csv)

    assert results["total_test_rows"] == 4
    assert results["test_label_distribution"]["negative_0"] == 1
    assert results["test_label_distribution"]["positive_4"] == 1
    assert results["test_label_distribution"]["neutral_2"] == 2

    # Binary evaluation should only cover the 2 negative/positive examples
    bin_eval = results["binary_evaluation"]
    assert bin_eval["test_size"] == 2
    assert "macro_f1" in bin_eval
    assert "confusion_matrix" in bin_eval

    # Neutral evaluation should cover the 2 neutral examples
    neut_eval = results["neutral_evaluation"]
    assert neut_eval["total_neutral_rows"] == 2
    assert "predicted_as_negative" in neut_eval
    assert "predicted_as_positive" in neut_eval
    assert neut_eval["predicted_as_negative"] + neut_eval["predicted_as_positive"] == 2


def test_memory_pressure_error_handling(tmp_path: Path):
    """Verify that MemoryPressureError is raised if available RAM is simulated below threshold."""
    from models.sentiment_analysis.streaming_baseline import train_streaming_sentiment140

    train_csv = tmp_path / "tiny_train.csv"
    pd.DataFrame([
        ("0", "1", "date", "query", "user", "bad"),
        ("4", "2", "date", "query", "user", "good"),
    ]).to_csv(train_csv, header=False, index=False)

    test_csv = tmp_path / "tiny_test.csv"
    pd.DataFrame([
        ("0", "1", "date", "query", "user", "bad"),
    ]).to_csv(test_csv, header=False, index=False)

    out_dir = tmp_path / "out"

    # Set min_free_ram_mb to a ridiculously huge number (1,000,000 MB) to force the safety abort
    with pytest.raises(MemoryPressureError, match="Aborting training: available RAM"):
        train_streaming_sentiment140(
            train_path=train_csv,
            test_path=test_csv,
            output_dir=out_dir,
            batch_size=1,
            min_free_ram_mb=1000000.0,
        )
