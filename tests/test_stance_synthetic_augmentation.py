import pandas as pd
import pytest

from scripts.experiment_stance_synthetic_augmentation_v1 import (
    assert_synthetic_training_only,
    generate_synthetic,
    overlap_audit,
)


def test_generator_is_reproducible_balanced_and_has_required_provenance():
    first = generate_synthetic(per_class=20, seed=1234)
    second = generate_synthetic(per_class=20, seed=1234)
    pd.testing.assert_frame_equal(first, second)
    assert first.reference_stance.value_counts().to_dict() == {
        "Support": 20, "Neutral": 20, "Unclear": 20
    }
    assert first.is_synthetic.all()
    for column in ("generation_method", "template_id", "language", "random_seed"):
        assert first[column].notna().all()
    assert set(first.loc[first.reference_stance == "Unclear", "synthetic_reason"]) == {
        "insufficient_context", "genuine_ambiguity", "fragment", "off_topic",
        "language_comprehension_difficulty",
    }


def test_synthetic_rows_are_rejected_from_validation_and_test():
    valid = pd.DataFrame({"is_synthetic": [True, False, False],
                          "split": ["train", "validation", "test"]})
    assert_synthetic_training_only(valid)
    invalid = valid.copy()
    invalid.loc[0, "split"] = "test"
    with pytest.raises(AssertionError, match="training-only"):
        assert_synthetic_training_only(invalid)


def test_overlap_audit_detects_exact_and_near_matches():
    rows = generate_synthetic(per_class=2, seed=99)
    result = overlap_audit(rows, [rows.iloc[0].text, "A different unrelated consultation comment."])
    assert result["exact_duplicate_count"] >= 1
    assert result["near_duplicate_count"] >= result["exact_duplicate_count"]


def test_v4_selected_ratio_and_hyperparameters_reproduce_from_validation():
    from scripts.audit_stance_synthetic_deployment_v1 import verify_v4_validation_selection

    result = verify_v4_validation_selection()
    assert result["selected_condition"] == "ratio_0.5"
    assert result["selected_configuration"] == {"C": 1.0, "class_weight": "balanced"}
    assert result["test_metrics_used_for_selection"] is False


def test_repeated_group_folds_keep_each_group_whole_and_each_record_once():
    import numpy as np
    from scripts.audit_stance_synthetic_deployment_v1 import make_repeated_group_folds

    y = np.repeat(["Support", "Neutral", "Unclear"], 40)
    groups = np.repeat(np.arange(60), 2)
    folds = make_repeated_group_folds(y, groups, n_splits=4, n_repeats=2, seed=20261029)
    assert len(folds) == 8
    for repeat in range(2):
        per_repeat = [f for f in folds if f[0] == repeat]
        validation = np.concatenate([valid for _, _, _, valid in per_repeat])
        assert len(set(validation.tolist())) == len(y)
        for _, _, train, valid in per_repeat:
            assert not (set(groups[train]) & set(groups[valid]))


def test_v4_template_audit_flags_clear_mixed_stance_risk():
    import pandas as pd
    from scripts.audit_stance_synthetic_deployment_v1 import audit_template_diversity, V4_DIR

    rows = pd.read_json(V4_DIR / "synthetic_training_rows.jsonl", lines=True)
    audit = audit_template_diversity(rows)
    flagged = [row for row in audit["templates"] if row["template_id"] == "unclear_ambiguity_1"]
    assert len(flagged) == 1
    assert flagged[0]["review_status"] == "label_risk"
