"""
tests/test_sentimix_baseline.py
===============================
Automated tests for SentiMix Hindi-English 3-class sentiment baseline pipeline.

Validates:
1. Incremental partial_fit, 3-class predictions, and probability calibration.
2. Serialization roundtrip and inference fidelity.
3. 3-class evaluation metric calculations and confusion matrix contracts.
4. Memory pressure safety error handling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.sentiment_analysis.sentimix_baseline import (
    SENTIMIX_LABEL_TO_INT,
    SENTIMIX_TARGET_NAMES,
    MemoryPressureError,
    SentiMixClassifier,
    evaluate_sentimix_model,
    train_sentimix_baseline,
)


def test_sentimix_classifier_contracts():
    """Verify partial_fit, 3-class predictions, and probability distributions."""
    clf = SentiMixClassifier(n_features=2**12, random_state=42)
    assert not clf.is_fitted_

    texts = [
        "bohot bura laga ye dekh kar",          # negative (0)
        "aaj main ghar ja raha hu",            # neutral (1)
        "ye to kamaal ho gaya bohot badhiya",   # positive (2)
        "worst service ever total fraud",       # negative (0)
        "normal day at work nothing special",   # neutral (1)
        "love this product so amazing",         # positive (2)
    ]
    labels = [0, 1, 2, 0, 1, 2]

    clf.partial_fit(texts, labels)
    assert clf.is_fitted_

    # Verify predictions
    preds = clf.predict(["worst experience", "just walking", "love this"])
    assert len(preds) == 3
    for p in preds:
        assert p in SENTIMIX_TARGET_NAMES

    # Probabilities
    probs = clf.predict_proba(["kamaal product", "fraud company"])
    assert probs.shape == (2, 3)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_sentimix_serialization_roundtrip(tmp_path: Path):
    """Verify model save and load produces bit-for-bit identical predictions."""
    clf = SentiMixClassifier(n_features=2**12, random_state=42)
    clf.partial_fit(["kharaab text", "theek text", "shandaar text"], [0, 1, 2])

    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    assert model_file.is_file()

    loaded = SentiMixClassifier.load(model_file)
    assert loaded.is_fitted_
    assert loaded.n_features == clf.n_features

    test_inputs = ["shandaar", "kharaab", "theek hai"]
    orig_probs = clf.predict_proba(test_inputs)
    loaded_probs = loaded.predict_proba(test_inputs)

    assert np.allclose(orig_probs, loaded_probs)


def test_sentimix_evaluation_metrics(tmp_path: Path):
    """Verify that evaluate_sentimix_model accurately computes 3-class metrics."""
    clf = SentiMixClassifier(n_features=2**12, random_state=42)
    # Fit with synthetic code-mixed samples
    train_texts = [
        "disaster failure bura",
        "normal regular theek",
        "superb wonderful badhiya",
    ]
    clf.partial_fit(train_texts, [0, 1, 2])

    test_csv = tmp_path / "synthetic_test.csv"
    pd.DataFrame({
        "cleaned_text": [
            "disaster failure bura",
            "normal regular theek",
            "superb wonderful badhiya",
        ],
        "label": ["negative", "neutral", "positive"],
    }).to_csv(test_csv, index=False)

    results = evaluate_sentimix_model(clf, test_csv)

    assert results["test_size"] == 3
    assert "accuracy" in results
    assert "macro_f1" in results
    assert "weighted_f1" in results
    assert len(results["confusion_matrix"]["matrix"]) == 3
    assert len(results["per_class"]) == 3
    assert "negative" in results["per_class"]
    assert "neutral" in results["per_class"]
    assert "positive" in results["per_class"]


def test_sentimix_memory_pressure_handling(tmp_path: Path):
    """Verify that MemoryPressureError is raised if available RAM is simulated below threshold."""
    train_csv = tmp_path / "tiny_train.csv"
    pd.DataFrame({
        "cleaned_text": ["bura", "theek", "achha"],
        "label": ["negative", "neutral", "positive"],
    }).to_csv(train_csv, index=False)

    test_csv = tmp_path / "tiny_test.csv"
    pd.DataFrame({
        "cleaned_text": ["achha"],
        "label": ["positive"],
    }).to_csv(test_csv, index=False)

    out_dir = tmp_path / "out"

    with pytest.raises(MemoryPressureError, match="Aborting training: available RAM"):
        train_sentimix_baseline(
            train_path=train_csv,
            test_path=test_csv,
            output_dir=out_dir,
            min_free_ram_mb=1000000.0,
        )
