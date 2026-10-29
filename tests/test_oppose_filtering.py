"""
tests/test_oppose_filtering.py
================================
Unit tests for the pure filtering logic in scripts/train_stance_oppose_filtering_v1.py
(filter_high_confidence). No model artifacts or large files are touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_stance_oppose_filtering_v1 import filter_high_confidence  # noqa: E402


def _df(rows):
    return pd.DataFrame(rows)


class TestFilterHighConfidence:
    def test_zero_percentile_keeps_everything(self):
        df = _df([
            {"gold_stance": "Support", "stance_margin_pseudo": 0.1, "sentiment_margin_pseudo": 0.1},
            {"gold_stance": "Oppose", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.9},
        ])
        out, cutoffs = filter_high_confidence(df, 0)
        assert len(out) == 2

    def test_oppose_needs_both_margins_above_cutoff(self):
        # 4 rows so the 50th percentile cutoff sits strictly between low and high values
        df = _df([
            {"gold_stance": "Oppose", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.1},   # high stance, low sentiment
            {"gold_stance": "Oppose", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.9},   # high stance, high sentiment
            {"gold_stance": "Support", "stance_margin_pseudo": 0.1, "sentiment_margin_pseudo": 0.1},
            {"gold_stance": "Support", "stance_margin_pseudo": 0.1, "sentiment_margin_pseudo": 0.9},
        ])
        out, cutoffs = filter_high_confidence(df, 50)
        kept_oppose = out[out.gold_stance == "Oppose"]
        # the low-sentiment-margin Oppose row must be dropped even though its stance margin is high
        assert len(kept_oppose) == 1
        assert kept_oppose.iloc[0]["sentiment_margin_pseudo"] >= cutoffs["sentiment_margin_cutoff_oppose_only"]

    def test_support_and_neutral_never_need_sentiment_margin(self):
        df = _df([
            {"gold_stance": "Support", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.0},
            {"gold_stance": "Neutral", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.0},
            {"gold_stance": "Oppose", "stance_margin_pseudo": 0.1, "sentiment_margin_pseudo": 0.0},
        ])
        out, cutoffs = filter_high_confidence(df, 50)
        # Support/Neutral rows with high stance margin survive regardless of sentiment margin
        assert set(out[out.gold_stance != "Oppose"]["gold_stance"]) == {"Support", "Neutral"}

    def test_low_stance_margin_always_dropped_regardless_of_class(self):
        df = _df([
            {"gold_stance": "Support", "stance_margin_pseudo": 0.01, "sentiment_margin_pseudo": 0.9},
            {"gold_stance": "Support", "stance_margin_pseudo": 0.9, "sentiment_margin_pseudo": 0.9},
        ])
        out, cutoffs = filter_high_confidence(df, 50)
        assert len(out) == 1
        assert out.iloc[0]["stance_margin_pseudo"] == 0.9

    def test_cutoffs_reported(self):
        df = _df([
            {"gold_stance": "Support", "stance_margin_pseudo": 0.1, "sentiment_margin_pseudo": 0.2},
            {"gold_stance": "Oppose", "stance_margin_pseudo": 0.5, "sentiment_margin_pseudo": 0.6},
        ])
        out, cutoffs = filter_high_confidence(df, 75)
        assert "stance_margin_cutoff" in cutoffs
        assert "sentiment_margin_cutoff_oppose_only" in cutoffs
        assert cutoffs["percentile"] == 75
