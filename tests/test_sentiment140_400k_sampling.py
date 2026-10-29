"""
tests/test_sentiment140_400k_sampling.py
===========================================
Unit tests for the pure helper functions in scripts/train_stance_sentiment140_400k_v1.py:
stratified_sample_exact, check_pairwise_text_overlap, dedup_by_normalised_text.
No real model artifacts are touched; a tiny synthetic pseudo_labels.csv fixture is used for the
sampling function.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_stance_sentiment140_400k_v1 import (  # noqa: E402
    check_pairwise_text_overlap, dedup_by_normalised_text, make_class_targets, stratified_sample_exact,
)

COLS = ["record_id", "source_dataset", "source_record_id", "cleaned_text", "pseudo_label", "rule",
        "sentiment_prediction", "sentiment_margin", "stance_raw", "stance_margin",
        "duplicate_group_id", "eligible_for_training"]


def _make_pseudo_csv(tmp_path, counts: dict[str, int]) -> Path:
    rows = []
    i = 0
    for label, n in counts.items():
        for _ in range(n):
            rows.append({
                "record_id": f"r{i}", "source_dataset": "sentiment140_train", "source_record_id": str(i),
                "cleaned_text": f"tweet number {i} about {label}", "pseudo_label": label, "rule": "test_rule",
                "sentiment_prediction": "positive", "sentiment_margin": "0.5", "stance_raw": "Support",
                "stance_margin": "0.5", "duplicate_group_id": str(i), "eligible_for_training": "True",
            })
            i += 1
    df = pd.DataFrame(rows, columns=COLS)
    p = tmp_path / "pseudo_labels.csv"
    df.to_csv(p, index=False)
    return p


class TestMakeClassTargets:
    def test_400000_matches_original_hardcoded_targets(self):
        assert make_class_targets(400_000, ("Support", "Neutral", "Oppose")) == {
            "Support": 133_334, "Neutral": 133_333, "Oppose": 133_333,
        }

    def test_800000_splits_evenly_with_remainder_distributed(self):
        targets = make_class_targets(800_000, ("Support", "Neutral", "Oppose"))
        assert sum(targets.values()) == 800_000
        assert targets == {"Support": 266_667, "Neutral": 266_667, "Oppose": 266_666}

    def test_exact_multiple_has_no_remainder(self):
        targets = make_class_targets(300, ("Support", "Neutral", "Oppose"))
        assert targets == {"Support": 100, "Neutral": 100, "Oppose": 100}


class TestStratifiedSampleExact:
    def test_exact_counts_when_enough_available(self, tmp_path):
        p = _make_pseudo_csv(tmp_path, {"Support": 50, "Neutral": 50, "Oppose": 50})
        sampled, report = stratified_sample_exact(p, {"Support": 10, "Neutral": 10, "Oppose": 10}, seed=42, overshoot_margin=5)
        assert len(sampled) == 30
        assert sampled["pseudo_label"].value_counts().to_dict() == {"Support": 10, "Neutral": 10, "Oppose": 10}
        assert report["shortfall"] == {}
        assert report["final_total"] == 30

    def test_reproducible_with_same_seed(self, tmp_path):
        p = _make_pseudo_csv(tmp_path, {"Support": 50, "Neutral": 50, "Oppose": 50})
        s1, _ = stratified_sample_exact(p, {"Support": 10, "Neutral": 10, "Oppose": 10}, seed=7, overshoot_margin=5)
        s2, _ = stratified_sample_exact(p, {"Support": 10, "Neutral": 10, "Oppose": 10}, seed=7, overshoot_margin=5)
        assert sorted(s1["record_id"].tolist()) == sorted(s2["record_id"].tolist())

    def test_shortfall_reported_not_silently_truncated(self, tmp_path):
        p = _make_pseudo_csv(tmp_path, {"Support": 5, "Neutral": 50, "Oppose": 50})
        sampled, report = stratified_sample_exact(p, {"Support": 10, "Neutral": 10, "Oppose": 10}, seed=42, overshoot_margin=5)
        assert "Support" in report["shortfall"]
        assert report["shortfall"]["Support"]["available"] == 5
        assert report["shortfall"]["Support"]["target"] == 10
        # classes that DO have enough are still sampled correctly
        assert sampled["pseudo_label"].value_counts().to_dict().get("Support", 0) == 0


class TestCheckPairwiseTextOverlap:
    def test_no_overlap(self):
        out = check_pairwise_text_overlap({
            "a": pd.Series(["alpha text", "beta text"]), "b": pd.Series(["gamma text"]),
        })
        assert out == {"a|b": 0}

    def test_overlap_counted_by_normalised_text(self):
        out = check_pairwise_text_overlap({
            "a": pd.Series(["Hello World", "unique a"]), "b": pd.Series(["hello   world", "unique b"]),
        })
        assert out["a|b"] == 1

    def test_three_way_pairs(self):
        out = check_pairwise_text_overlap({
            "a": pd.Series(["x"]), "b": pd.Series(["x"]), "c": pd.Series(["y"]),
        })
        assert out["a|b"] == 1
        assert out["a|c"] == 0
        assert out["b|c"] == 0


class TestDedupByNormalisedText:
    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "src": ["s1", "s2", "s3"]})
        out, report = dedup_by_normalised_text(df, "text", ["s1", "s2", "s3"], "src")
        assert len(out) == 3
        assert report["rows_dropped"] == 0

    def test_priority_order_keeps_highest_priority_source(self):
        df = pd.DataFrame({"text": ["Dup Text", "dup   text", "unique"], "src": ["low", "high", "low"]})
        out, report = dedup_by_normalised_text(df, "text", ["high", "low"], "src")
        assert len(out) == 2
        assert report["rows_dropped"] == 1
        kept_dup_row = out[out["text"].str.lower().str.contains("dup")]
        assert kept_dup_row.iloc[0]["src"] == "high"

    def test_dropped_by_source_reported(self):
        df = pd.DataFrame({"text": ["x", "x", "x"], "src": ["a", "a", "b"]})
        out, report = dedup_by_normalised_text(df, "text", ["a", "b"], "src")
        assert len(out) == 1
        assert report["rows_dropped"] == 2
        assert report["dropped_by_source"] == {"a": 1, "b": 1}
