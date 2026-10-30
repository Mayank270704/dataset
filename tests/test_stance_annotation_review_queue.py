import pandas as pd

from scripts.build_stance_annotation_review_queue_v1 import SCHEMA, build_queue, reason_code


def _source(record_id="r1", label="Unclear", note="AI v1: fragment"):
    return pd.DataFrame([{
        "pilot_id": record_id,
        "source_dataset": "source_a",
        "source_record_id": "17",
        "source_language": "English",
        "original_text": "An unchanged original comment.",
        "gold_stance": label,
        "ai_review_notes": note,
    }])


def _predictions(record_id="r1", reference="Unclear", prod="Neutral", candidate="Unclear"):
    rows = []
    for condition, prediction in (("production_word_only", prod), ("candidate_v1_recipe", candidate)):
        for repeat in range(3):
            rows.append({"condition": condition, "pilot_id": record_id, "repeat": repeat,
                         "reference_stance": reference, "predicted_stance": prediction})
    return pd.DataFrame(rows)


def test_review_queue_preserves_record_and_leaves_review_fields_blank():
    q = build_queue(_source(), _predictions())
    assert list(q.columns) == SCHEMA
    row = q.iloc[0]
    assert row.record_id == "r1"
    assert row.source_id == "source_a#17"
    assert row.text == "An unchanged original comment."
    assert row.current_reference_label == "Unclear"
    assert row.proposed_reason_code == "fragment"
    assert row.reviewer_1_label == row.reviewer_2_label == row.adjudicated_label == row.reviewer_notes == ""


def test_unclear_reason_codes_are_distinct_and_language_requires_protocol_note():
    assert reason_code("AI v1: fragment; ambiguous target") == "fragment;genuine_ambiguity"
    assert reason_code("AI v1: off-topic") == "off_topic"
    assert reason_code("non-English: French") == "language_comprehension_protocol"
    assert reason_code("French-language human review required") == "review_reason_unrecorded"


def test_review_queue_uses_oof_disagreement_and_not_confidence_probability():
    source = _source(label="Support", note="")
    preds = _predictions(reference="Support", prod="Support", candidate="Neutral")
    q = build_queue(source, preds)
    assert "production_candidate_oof_disagreement" in q.iloc[0].review_reason
    assert "confidence" not in q.columns


def test_french_records_are_prioritized_without_auto_assigning_language_reason():
    source = _source(label="Neutral", note="")
    source.loc[0, "source_language"] = "French"
    q = build_queue(source, _predictions(reference="Neutral", prod="Neutral", candidate="Neutral"))
    assert q.iloc[0].review_reason == "underrepresented_french_review"
    assert q.iloc[0].proposed_reason_code == ""
