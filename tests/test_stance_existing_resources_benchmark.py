from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.benchmark_stance_existing_resources_v1 import (
    LABELS,
    fold_map,
    load_embeddings,
    selective_metrics,
)
from models.stance_classification.labels import load_stance_reference_labels
from scripts.experiment_stance_reference_models_v2 import MODEL_LABELS, REFERENCE_LABELS_PATH


def _reference_frame():
    status, frame = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    assert status.state == "ok" and frame is not None
    return frame[frame.gold_stance.isin(MODEL_LABELS)].reset_index(drop=True)


def test_saved_grouped_folds_cover_each_reference_once_per_repeat_without_crossings():
    frame = _reference_frame()
    folds = fold_map(frame)
    assert len(folds) == 15
    for repeat in range(3):
        valid = np.concatenate([fold["valid"] for fold in folds if fold["repeat"] == repeat])
        assert sorted(valid.tolist()) == list(range(len(frame)))
        for fold in [row for row in folds if row["repeat"] == repeat]:
            assert not (set(fold["train"]) & set(fold["valid"]))


def test_cached_embeddings_map_all_reference_text_exactly():
    frame = _reference_frame()
    embeddings, metadata = load_embeddings(frame.original_text.astype(str).tolist())
    assert embeddings.shape == (399, 384)
    assert np.isfinite(embeddings).all()
    assert metadata["exact_text_match_count"] == 399


def test_fixed_rank_abstention_reports_full_coverage_without_hiding_rows():
    rows = pd.DataFrame([
        {"condition": "demo", "repeat": 0, "fold": 0, "margin": 0.1,
         "reference_stance": "Neutral", "predicted_stance": "Neutral"},
        {"condition": "demo", "repeat": 0, "fold": 0, "margin": 0.2,
         "reference_stance": "Unclear", "predicted_stance": "Neutral"},
        {"condition": "demo", "repeat": 0, "fold": 0, "margin": 0.9,
         "reference_stance": "Support", "predicted_stance": "Support"},
        {"condition": "demo", "repeat": 0, "fold": 0, "margin": 0.3,
         "reference_stance": "Neutral", "predicted_stance": "Support"},
    ])
    metrics = selective_metrics(rows)
    all_rows = next(row for row in metrics if row["target_within_fold_coverage"] == 1.0)
    half = next(row for row in metrics if row["target_within_fold_coverage"] == 0.5)
    assert all_rows["actual_coverage"] == 1.0
    assert all_rows["accepted_n"] == 4
    assert all_rows["selective_accuracy"] == 0.5
    assert half["actual_coverage"] == 0.5
    assert set(LABELS) == {"Support", "Neutral", "Unclear"}
