"""Regression tests: stance inference emits only trained classes and never infers from sentiment."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def predictor(labels=("Support", "Neutral", "Unclear"), pred="Neutral"):
    pipe = MagicMock()
    pipe.classes_ = np.asarray(labels)
    pipe.predict.side_effect = lambda xs: np.asarray([pred] * len(xs))
    pipe.decision_function.side_effect = lambda xs: np.tile([1.0, 0.3, -0.1], (len(xs), 1))
    with patch("models.stance_inference_v2.load_verified_pipeline", return_value=(pipe, {"test": True})):
        from models.stance_inference_v2 import FinalStancePredictor
        obj = FinalStancePredictor()
    return obj, pipe


def test_negative_tone_does_not_create_oppose():
    obj, pipe = predictor(pred="Neutral")
    rows = obj.predict(["This terrible policy should be stopped"])
    assert rows.iloc[0].final_stance == "Neutral"
    assert "Oppose" not in set(rows.final_stance.dropna())
    pipe.predict.assert_called_once()


def test_output_preserves_schema_and_compatibility_fields():
    obj, _ = predictor(pred="Unclear")
    rows = obj.predict(["fragment", "", "  "])
    required = {"text", "input_status", "final_stance", "stance_raw", "stance_margin",
                "sentiment_prediction", "sentiment_model_source", "prediction_method",
                "stance_label_source", "disclaimer"}
    assert required.issubset(rows.columns)
    assert len(rows) == 3
    assert rows.iloc[0].final_stance == "Unclear"
    assert rows.iloc[0].sentiment_prediction is None
    assert rows.iloc[1].input_status == "too_short_or_empty"
    assert pd.isna(rows.iloc[1].final_stance)


def test_artifact_with_untrained_label_is_rejected():
    pipe = MagicMock(); pipe.classes_ = np.asarray(["Support", "Oppose"])
    with patch("models.stance_inference_v2.load_verified_pipeline", return_value=(pipe, {})):
        from models.stance_inference_v2 import FinalStancePredictor
        with pytest.raises(ValueError, match="Unsupported stance labels"):
            FinalStancePredictor()


def test_real_artifact_never_outputs_untrained_classes():
    from models.stance_inference_v2 import FinalStancePredictor
    rows = FinalStancePredictor().predict(["This policy is bad and should be stopped", "Je ne sais pas"])
    assert set(rows.final_stance.dropna()).issubset({"Support", "Neutral", "Unclear"})
