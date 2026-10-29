import pandas as pd

from scripts.policylens_government_consultation_audit import (
    clean_cell,
    norm_key,
    redact_text,
    slugify_column,
    text_column_assessment,
)


def test_clean_cell_standardizes_blank_and_whitespace_only_values():
    assert clean_cell("  hello\r\n   world  ") == "hello\nworld"
    assert clean_cell(" n/a ") is None
    assert clean_cell("   ") is None


def test_slugify_column_deduplicates_names():
    seen = {}
    from collections import Counter

    seen_counter = Counter(seen)
    assert slugify_column("Q8. Did we miss anything?", seen_counter) == "q8_did_we_miss_anything"
    assert slugify_column("Q8. Did we miss anything?", seen_counter) == "q8_did_we_miss_anything_2"


def test_redact_text_masks_email_and_phone_like_values():
    text = redact_text("Contact me at person@example.com or +1 613 555 1212 about open data.")
    assert "[EMAIL]" in text
    assert "[PHONE_OR_ID]" in text


def test_text_column_assessment_flags_long_question_response_as_free_text():
    df = pd.DataFrame(
        {
            "Q14. Please tell us why:": [
                "The service should publish clearer records and explain decisions.",
                "",
                "More context is needed for people to participate.",
            ]
        }
    ).map(clean_cell)

    result = text_column_assessment(df, "Q14. Please tell us why:")

    assert result["field_type"] == "free_text_candidate"
    assert result["non_empty_responses"] == 2
    assert result["policy_relevance_recommendation"] == "review_for_policy_concerns_suggestions_or_requested_actions"


def test_norm_key_is_case_and_space_insensitive():
    assert norm_key(" Open   Data ") == norm_key("open data")
