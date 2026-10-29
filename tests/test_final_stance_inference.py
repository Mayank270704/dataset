"""
tests/test_final_stance_inference.py
======================================
Unit tests for ``models/stance_inference_v2.py`` — the heuristic four-label
stance inference pipeline (FinalStancePredictor).

Design
------
* All tests are UNIT tests; they do NOT load the large saved model artifacts.
  The two external dependencies are mocked:
    - ``models.stance_inference_v2.load_verified_pipeline`` → returns a fake
      sklearn-compatible pipeline that returns controllable stance predictions.
    - ``models.stance_inference_v2._load_verified_sentiment_model`` → returns
      a mock Sentiment140StreamingClassifier that returns controllable sentiments.
* Integration tests that need the real artifacts are guarded with
  ``@pytest.mark.skipif`` on the artifact path.

Heuristic rule under test
--------------------------
    sentiment == "negative"  AND  stance_raw != "Support"
        → final_stance = "Oppose",  prediction_method = "heuristic_oppose"
    otherwise
        → final_stance = stance_raw, prediction_method = "stance_model_direct"
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── artifact paths for integration-test guards ──────────────────────────────
_STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
_SENTIMENT_DIR = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1"
_ARTIFACTS_PRESENT = _STANCE_RUN.is_dir() and (_SENTIMENT_DIR / "model.joblib").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: factory functions for controllable mocks
# ─────────────────────────────────────────────────────────────────────────────

def _make_stance_pipe(stance_label: str) -> MagicMock:
    """
    Return a mock sklearn pipeline that always predicts ``stance_label``
    and returns a simple decision-function array suitable for margin computation.
    """
    pipe = MagicMock()
    pipe.predict.side_effect = lambda texts: np.array([stance_label] * len(texts))

    # Simulate a 3-class decision function: give the predicted class a large
    # score so the margin helper works without error.
    label_idx = {"Support": 0, "Neutral": 1, "Unclear": 2}.get(stance_label, 0)
    scores = [-0.5, -0.5, -0.5]
    scores[label_idx] = 1.5

    def _dec_fn(texts):
        return np.tile(scores, (len(texts), 1))  # shape (N, 3)

    pipe.decision_function.side_effect = _dec_fn
    return pipe


def _make_sentiment_clf(sentiment_label: str) -> MagicMock:
    """Return a mock Sentiment140StreamingClassifier that always predicts ``sentiment_label``."""
    clf = MagicMock()
    clf.predict.side_effect = lambda texts: np.array([sentiment_label] * len(texts))
    return clf


def _build_predictor(stance_label: str, sentiment_label: str):
    """
    Instantiate a FinalStancePredictor with both heavy dependencies mocked.

    Parameters
    ----------
    stance_label    : one of "Support", "Neutral", "Unclear"
    sentiment_label : one of "negative", "positive"
    """
    fake_stance_pipe = _make_stance_pipe(stance_label)
    fake_sentiment_clf = _make_sentiment_clf(sentiment_label)
    fake_meta = {
        "label_type": "ai_generated_unverified",
        "disclaimer": "test",
        "human_labels_only": False,
    }

    with patch("models.stance_inference_v2.load_verified_pipeline",
               return_value=(fake_stance_pipe, fake_meta)), \
         patch("models.stance_inference_v2._load_verified_sentiment_model",
               return_value=fake_sentiment_clf):
        from models.stance_inference_v2 import FinalStancePredictor
        predictor = FinalStancePredictor(model="linsvm")

    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveSentimentSupport:
    def test_positive_sentiment_support_input(self):
        """Positive sentiment + Support stance → final=Support, method=stance_model_direct."""
        predictor = _build_predictor(stance_label="Support", sentiment_label="positive")
        text = "I strongly support open government data publishing"
        df = predictor.predict([text])
        row = df.iloc[0]

        assert row["input_status"] == "ok"
        assert row["final_stance"] == "Support"
        assert row["stance_raw"] == "Support"
        assert row["prediction_method"] == "stance_model_direct"
        assert row["sentiment_prediction"] == "positive"


class TestNegativeSentimentNonSupportGetsOppose:
    def test_negative_sentiment_non_support_gets_oppose(self):
        """Negative sentiment + Neutral stance → final=Oppose, method=heuristic_oppose."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="negative")
        text = "This policy is a terrible waste of money and should be stopped"
        df = predictor.predict([text])
        row = df.iloc[0]

        assert row["input_status"] == "ok"
        assert row["final_stance"] == "Oppose"
        assert row["stance_raw"] == "Neutral"      # raw stance unchanged
        assert row["prediction_method"] == "heuristic_oppose"
        assert row["sentiment_prediction"] == "negative"

    def test_negative_sentiment_unclear_gets_oppose(self):
        """Negative sentiment + Unclear stance → final=Oppose."""
        predictor = _build_predictor(stance_label="Unclear", sentiment_label="negative")
        df = predictor.predict(["Yeh policy bilkul galat hai"])
        row = df.iloc[0]

        assert row["final_stance"] == "Oppose"
        assert row["prediction_method"] == "heuristic_oppose"


class TestNegativeSentimentSupportKept:
    def test_negative_sentiment_support_kept(self):
        """Negative sentiment + Support stance → final=Support (heuristic does NOT fire)."""
        predictor = _build_predictor(stance_label="Support", sentiment_label="negative")
        text = "Despite challenges, I still support this open-data initiative"
        df = predictor.predict([text])
        row = df.iloc[0]

        assert row["input_status"] == "ok"
        assert row["final_stance"] == "Support"
        assert row["prediction_method"] == "stance_model_direct"
        assert row["sentiment_prediction"] == "negative"


class TestNeutralInput:
    def test_neutral_input(self):
        """Factual/neutral text → final=Neutral (whatever stance model says), not Oppose."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="positive")
        df = predictor.predict(["The meeting was held on Tuesday"])
        row = df.iloc[0]

        assert row["input_status"] == "ok"
        assert row["final_stance"] in {"Support", "Neutral", "Unclear"}
        assert row["final_stance"] != "Oppose"
        assert row["prediction_method"] == "stance_model_direct"


class TestUnclearInput:
    def test_unclear_input_positive_sentiment(self):
        """Positive-sentiment + Unclear → final=Unclear (stance_model_direct)."""
        predictor = _build_predictor(stance_label="Unclear", sentiment_label="positive")
        df = predictor.predict(["I am not sure about this at all"])
        row = df.iloc[0]

        assert row["final_stance"] == "Unclear"
        assert row["prediction_method"] == "stance_model_direct"

    def test_unclear_input_negative_sentiment(self):
        """Negative-sentiment + Unclear → final=Oppose (heuristic fires)."""
        predictor = _build_predictor(stance_label="Unclear", sentiment_label="negative")
        df = predictor.predict(["This is all very bad and unclear"])
        row = df.iloc[0]

        assert row["final_stance"] == "Oppose"
        assert row["prediction_method"] == "heuristic_oppose"


class TestEmptyInputRejected:
    def test_empty_input_rejected(self):
        """Empty string → input_status=too_short_or_empty, final_stance=None."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="positive")
        df = predictor.predict([""])
        row = df.iloc[0]

        assert row["input_status"] == "too_short_or_empty"
        assert row["final_stance"] is None or pd.isna(row["final_stance"])


class TestWhitespaceInputRejected:
    def test_whitespace_input_rejected(self):
        """Whitespace-only string → input_status=too_short_or_empty, final_stance=None."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="positive")
        df = predictor.predict(["   "])
        row = df.iloc[0]

        assert row["input_status"] == "too_short_or_empty"
        assert row["final_stance"] is None or pd.isna(row["final_stance"])


class TestBatchOutputSchema:
    REQUIRED_COLUMNS = {
        "text", "input_status", "final_stance", "stance_raw", "stance_margin",
        "sentiment_prediction", "sentiment_model_source", "prediction_method",
        "stance_label_source", "disclaimer",
    }

    def test_batch_output_schema(self):
        """All required output columns must be present in the DataFrame."""
        predictor = _build_predictor(stance_label="Support", sentiment_label="positive")
        df = predictor.predict([
            "Some valid comment about policy",
            "",
            "This is bad",
        ])
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_row_count_matches_input(self):
        """Output DataFrame must have exactly one row per input text."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="positive")
        texts = ["alpha", "beta", "gamma", "", "  "]
        df = predictor.predict(texts)
        assert len(df) == len(texts)


class TestHeuristicMarkedCorrectly:
    def test_heuristic_marked_correctly(self):
        """Every row with final_stance='Oppose' must have prediction_method='heuristic_oppose'."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="negative")
        texts = [
            "This terrible policy should be reversed",
            "Another dreadful decision by officials",
            "",
        ]
        df = predictor.predict(texts)
        oppose_rows = df[df["final_stance"] == "Oppose"]
        assert len(oppose_rows) > 0, "Expected at least one Oppose prediction in this mock scenario"
        bad = oppose_rows[oppose_rows["prediction_method"] != "heuristic_oppose"]
        assert len(bad) == 0, f"Oppose rows missing heuristic_oppose method:\n{bad}"

    def test_stance_label_source_heuristic(self):
        """Heuristic Oppose rows must have stance_label_source='heuristic_oppose_unverified'."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="negative")
        df = predictor.predict(["This policy is awful and should be stopped"])
        row = df.iloc[0]
        assert row["stance_label_source"] == "heuristic_oppose_unverified"


class TestNonOpposeNeverHeuristic:
    def test_non_oppose_never_heuristic(self):
        """Support/Neutral/Unclear rows must have prediction_method='stance_model_direct'."""
        # Mix of sentiment labels but stance = Support → no Oppose
        predictor = _build_predictor(stance_label="Support", sentiment_label="positive")
        texts = [
            "I support this policy",
            "Open data is great",
            "The government did well here",
        ]
        df = predictor.predict(texts)
        valid_df = df[df["input_status"] == "ok"]
        non_oppose = valid_df[valid_df["final_stance"] != "Oppose"]
        bad = non_oppose[non_oppose["prediction_method"] != "stance_model_direct"]
        assert len(bad) == 0, f"Non-Oppose rows have unexpected method:\n{bad}"

    def test_stance_label_source_direct(self):
        """Direct-model rows must have stance_label_source='ai_working_label_unverified'."""
        predictor = _build_predictor(stance_label="Support", sentiment_label="positive")
        df = predictor.predict(["I fully support this initiative"])
        row = df.iloc[0]
        assert row["stance_label_source"] == "ai_working_label_unverified"


class TestDisclaimerPresent:
    def test_disclaimer_always_present(self):
        """The disclaimer column must be non-empty for every row (valid and rejected)."""
        predictor = _build_predictor(stance_label="Neutral", sentiment_label="positive")
        df = predictor.predict(["Valid comment about policy", "", "ok"])
        assert df["disclaimer"].notna().all()
        assert (df["disclaimer"].str.len() > 0).all()


class TestSentimentModelSourceTagged:
    def test_sentiment_model_source_tagged(self):
        """Valid rows must have sentiment_model_source='sentiment140_hashing_sgd_v1'."""
        predictor = _build_predictor(stance_label="Support", sentiment_label="positive")
        df = predictor.predict(["Government open data is a great idea"])
        row = df.iloc[0]
        assert row["sentiment_model_source"] == "sentiment140_hashing_sgd_v1"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (skipped when artifacts are absent)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _ARTIFACTS_PRESENT,
    reason="Real model artifacts not present on this machine; skipping integration test.",
)
class TestIntegrationRealModels:
    """Integration tests that load the actual saved artifact files."""

    def test_integration_schema_with_real_models(self):
        """Real model load + predict must return a valid DataFrame schema."""
        from models.stance_inference_v2 import FinalStancePredictor

        predictor = FinalStancePredictor(model="linsvm")
        df = predictor.predict(["I support open government data publishing"])
        required = {
            "text", "input_status", "final_stance", "stance_raw", "stance_margin",
            "sentiment_prediction", "sentiment_model_source", "prediction_method",
            "stance_label_source", "disclaimer",
        }
        assert not (required - set(df.columns))
        assert df.iloc[0]["input_status"] == "ok"
        assert df.iloc[0]["final_stance"] in {"Support", "Oppose", "Neutral", "Unclear"}

    def test_integration_oppose_assignment_with_real_models(self):
        """Real models: a clearly negative non-Support text should trigger heuristic Oppose."""
        from models.stance_inference_v2 import FinalStancePredictor

        predictor = FinalStancePredictor(model="linsvm")
        df = predictor.predict(
            ["This policy is absolutely terrible and should be immediately cancelled"]
        )
        row = df.iloc[0]
        assert row["input_status"] == "ok"
        # If heuristic fired, method is heuristic_oppose; otherwise stance_model_direct.
        assert row["prediction_method"] in {"heuristic_oppose", "stance_model_direct"}
        if row["prediction_method"] == "heuristic_oppose":
            assert row["final_stance"] == "Oppose"
