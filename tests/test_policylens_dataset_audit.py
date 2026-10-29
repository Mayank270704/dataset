from pathlib import Path

import pandas as pd

from scripts.policylens_dataset_audit import (
    DATASETS,
    assign_duplicate_groups,
    canonicalize,
    clean_text,
    normalized_duplicate_key,
    read_sentimix_conll,
)


def test_clean_text_preserves_meaningful_punctuation_and_numbers():
    assert clean_text("  Policy 123!!!\r\nNeeds   review.  ") == "Policy 123!!!\nNeeds review."


def test_normalized_duplicate_key_is_case_and_space_insensitive():
    assert normalized_duplicate_key("  Same   TEXT ") == normalized_duplicate_key("same text")


def test_duplicate_groups_include_first_and_later_occurrences():
    groups = assign_duplicate_groups(pd.Series(["a", "b", "a", None, "b", "c"]))

    assert groups == ["dup_000001", "dup_000002", "dup_000001", None, "dup_000002", None]


def test_sentimix_parser_reads_meta_blocks(tmp_path: Path):
    conll = tmp_path / "sample.txt"
    conll.write_text("meta\t1\tpositive\nHello\tEng\nजी\tHin\n\nmeta\t2\tnegative\nNo\tEng\n", encoding="utf-8")

    df = read_sentimix_conll(conll)

    assert list(df["uid"]) == ["1", "2"]
    assert list(df["text"]) == ["Hello जी", "No"]
    assert list(df["sentiment"]) == ["positive", "negative"]
    assert list(df["token_count"]) == [2, 1]


def test_sentiment140_canonical_maps_numeric_targets():
    spec = next(item for item in DATASETS if item.name == "sentiment140_test")
    df = pd.DataFrame(
        {
            "target": [0, 2, 4],
            "ids": [1, 2, 3],
            "date": ["d", "d", "d"],
            "flag": ["f", "f", "f"],
            "user": ["u", "u", "u"],
            "text": ["bad", "okay", "good"],
        }
    )

    canonical = canonicalize(spec, df)

    assert list(canonical["label"]) == ["negative", "neutral", "positive"]
    assert list(canonical["cleaned_text"]) == ["bad", "okay", "good"]
