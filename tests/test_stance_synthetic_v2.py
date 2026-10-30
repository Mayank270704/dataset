import pandas as pd
import pytest

from scripts.generate_stance_synthetic_v2 import (
    generate_model_corpus,
    generate_protocol_rows,
    quality_check,
)


def test_v2_generation_is_reproducible_balanced_and_provenance_complete():
    first = generate_model_corpus(per_class=30, seed=311)
    second = generate_model_corpus(per_class=30, seed=311)
    pd.testing.assert_frame_equal(first, second)
    assert first.reference_stance.value_counts().to_dict() == {"Support": 30, "Neutral": 30, "Unclear": 30}
    assert first.is_synthetic.all() and first.model_eligible.all()
    assert first.text.map(lambda t: " ".join(t.casefold().split())).nunique() == len(first)
    for field in ("generation_method", "template_id", "language", "random_seed", "synthetic_reason"):
        assert first[field].notna().all()
    assert set(first[first.reference_stance == "Unclear"].synthetic_reason) == {
        "insufficient_context", "fragment", "off_topic"
    }


def test_v2_quality_gate_rejects_duplicate_and_contradictory_unclear_rows():
    rows = generate_model_corpus(per_class=12, seed=312)
    duplicate = rows.copy()
    duplicate.loc[1, "text"] = duplicate.loc[0, "text"]
    with pytest.raises(AssertionError, match="duplicate synthetic"):
        quality_check(duplicate, expected_per_class=12)

    contradictory = rows.copy()
    unclear_index = contradictory.index[contradictory.reference_stance == "Unclear"][0]
    contradictory.loc[unclear_index, "text"] = "I support openness but oppose publishing the records."
    with pytest.raises(AssertionError, match="conflicting positions"):
        quality_check(contradictory, expected_per_class=12)


def test_language_comprehension_protocol_rows_have_no_model_label():
    rows = generate_protocol_rows(count=8, seed=313)
    assert rows.is_synthetic.all()
    assert not rows.model_eligible.any()
    assert "reference_stance" not in rows
    assert rows.text.nunique() == len(rows)
    assert rows.protocol_condition.notna().all()
    with pytest.raises(ValueError, match="avoid duplicate protocol rows"):
        generate_protocol_rows(count=9, seed=313)


def test_every_v4_flagged_row_is_retained_in_review_and_disabled():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    rows = pd.read_json(root / "reports/stance_synthetic_corpus_v2_clean/flagged_v4_rows_review.jsonl", lines=True)
    assert len(rows) == 32
    assert rows.audit_status.eq("flagged_in_v4_audit").sum() == 28
    assert rows.audit_status.eq("precautionarily_disabled").sum() == 4
    assert rows.training_action.eq("disabled_from_v2_model_training").all()


def test_shared_candidate_cv_folds_keep_synthetic_rows_training_only():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_json(root / "reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl", lines=True)
    synthetic = manifest[manifest.is_synthetic]
    assert not synthetic.empty
    assert synthetic.split.eq("train").all()
    originals = manifest[(~manifest.is_synthetic) & (manifest.condition == "shared_original_folds")]
    for _, fold in originals.groupby(["repeat", "fold"]):
        train = set(fold.loc[fold.split == "train", "leakage_group"])
        validation = set(fold.loc[fold.split == "validation", "leakage_group"])
        assert not train & validation
