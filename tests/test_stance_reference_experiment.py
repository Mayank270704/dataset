"""Checks grouped split integrity and reference-label metric accounting."""
import numpy as np
import pandas as pd

from scripts.experiment_stance_reference_models_v2 import evaluate, grouped_three_way_split


def test_grouped_three_way_split_keeps_source_duplicate_groups_intact():
    reference_labels = np.repeat(["Support", "Neutral", "Unclear"], 30)
    leakage_groups = np.repeat(np.arange(45), 2)
    train, validation, test = grouped_three_way_split(reference_labels, leakage_groups, 20261029)

    assert not (set(leakage_groups[train]) & set(leakage_groups[validation]))
    assert not (set(leakage_groups[train]) & set(leakage_groups[test]))
    assert not (set(leakage_groups[validation]) & set(leakage_groups[test]))
    for indices in (train, validation, test):
        assert set(reference_labels[indices]) == {"Support", "Neutral", "Unclear"}


def test_evaluation_reports_exact_reference_counts_and_per_class_scores():
    reference_labels = np.array(["Support", "Neutral", "Unclear", "Unclear"])
    predicted_labels = np.array(["Support", "Unclear", "Neutral", "Unclear"])
    result = evaluate(reference_labels, predicted_labels, ["Support", "Neutral", "Unclear"])

    assert result["n"] == 4
    assert result["accuracy"] == 0.5
    assert result["per_class"]["Unclear"]["support"] == 2
    assert result["confusion_matrix"]["rows_reference_cols_predicted"] == [
        [1, 0, 0], [0, 0, 1], [0, 1, 1]
    ]


def test_reference_label_loader_is_a_read_only_alias():
    from models.stance_classification import labels
    from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

    status, reference_frame = labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    legacy_status, legacy_frame = labels.load_ai_working_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    assert status.state == legacy_status.state == "ok"
    pd.testing.assert_frame_equal(reference_frame, legacy_frame)
