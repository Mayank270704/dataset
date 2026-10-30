import numpy as np
import pandas as pd

from scripts.benchmark_stance_context_v1 import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    feature_input,
    fold_map,
)
from scripts.experiment_stance_reference_models_v2 import (
    MODEL_LABELS,
    REFERENCE_LABELS_PATH,
)
from models.stance_classification.labels import load_stance_reference_labels


def test_feature_allowlists_exclude_label_review_and_provenance_fields():
    assert all(not (set(columns) & FORBIDDEN_COLUMNS) for columns in FEATURE_COLUMNS.values())
    row = {
        "original_text": "The proposal should proceed.",
        "question_wording": "Do you support the proposal?",
        "question_context": "A consultation asks for views.",
        "source_language": "English",
        "language_basis": "source column `source_language`",
        "gold_stance": "Support",
        "ai_stance": "Oppose",
        "ai_review_notes": "Should be revised",
        "label_source": "review",
    }
    frame = pd.DataFrame([row])
    expected = {mode: feature_input(frame, mode).iloc[0] for mode in FEATURE_COLUMNS}
    frame.loc[0, ["gold_stance", "ai_stance", "ai_review_notes", "label_source"]] = [
        "Neutral", "Support", "different reviewer text", "other provenance",
    ]
    assert {mode: feature_input(frame, mode).iloc[0] for mode in FEATURE_COLUMNS} == expected


def test_missing_context_is_marked_and_comment_is_retained():
    frame = pd.DataFrame([{
        "original_text": "I have no view.", "question_wording": "", "question_context": None,
        "context": "", "source_language": "", "language_basis": "questionnaire language",
    }])
    value = feature_input(frame, "context_question_comment").iloc[0]
    assert "[NO_QUESTION]" in value
    assert "[NO_CONTEXT]" in value
    assert "I have no view." in value
    language = feature_input(frame, "verified_language_comment").iloc[0]
    assert "unknown" in language
    assert "I have no view." in language


def test_language_marker_uses_only_verified_comment_language():
    frame = pd.DataFrame([
        {"original_text": "A", "source_language": "French", "language_basis": "source column source_language"},
        {"original_text": "B", "source_language": "French", "language_basis": "questionnaire language"},
    ])
    output = feature_input(frame, "verified_language_comment").tolist()
    assert "french" in output[0]
    assert "unknown" in output[1]


def test_saved_source_grouped_folds_are_reproducible_and_cover_reference_rows():
    status, source = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    assert status.state == "ok" and source is not None
    frame = source[source.gold_stance.isin(list(MODEL_LABELS))].reset_index(drop=True)
    first = fold_map(frame)
    second = fold_map(frame)
    assert len(first) == 15
    assert all(np.array_equal(a["valid"], b["valid"]) for a, b in zip(first, second))
    for fold in first:
        assert not (set(fold["train"]) & set(fold["valid"]))
        assert set(frame.iloc[fold["valid"]].gold_stance) == set(MODEL_LABELS)
