"""Direct stance inference using only classes present in the saved stance artifact.

The current artifact was trained against the repository reference labels for Support, Neutral and Unclear.
It cannot predict Oppose. Sentiment is not consulted to infer stance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from models.new_comment_inference import ArtifactError, load_verified_pipeline

ROOT = Path(__file__).resolve().parents[1]
STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
SENTIMENT_MODEL_DIR = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1"
SENTIMENT_MODEL_PATH = SENTIMENT_MODEL_DIR / "model.joblib"
SENTIMENT_MANIFEST_PATH = SENTIMENT_MODEL_DIR / "manifest.json"
DEFAULT_MODEL = "linsvm"
MIN_CHARS = 3
LABEL_SOURCE = "ai_working_label_unverified"
DISCLAIMER = ("Provisional suggestion trained against the repository's stance reference labels. "
              "Only Support, Neutral and Unclear are available; performance on external sources is not established.")


def _load_verified_sentiment_model(
    model_path: Path = SENTIMENT_MODEL_PATH,
    manifest_path: Path = SENTIMENT_MANIFEST_PATH,
):
    """Legacy independent sentiment loader used by pseudo-labeling jobs; never used to infer stance."""
    from models.sentiment_analysis.streaming_baseline import Sentiment140StreamingClassifier
    if not manifest_path.is_file() or not model_path.is_file():
        raise ArtifactError("Sentiment140 model or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (manifest.get("outputs_sha256") or {}).get("model.joblib")
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ArtifactError("Sentiment140 model SHA-256 does not match its manifest")
    return Sentiment140StreamingClassifier.load(model_path)


@dataclass
class FinalStancePredictor:
    """Compatibility wrapper around the verified three-class stance pipeline."""
    model: str = DEFAULT_MODEL
    stance_run: Path = field(default_factory=lambda: STANCE_RUN)

    def __post_init__(self) -> None:
        self.stance_pipe, self.stance_meta = load_verified_pipeline(self.stance_run, self.model)
        self.labels = list(self.stance_pipe.classes_)
        if not self.labels or not set(self.labels).issubset({"Support", "Neutral", "Unclear"}):
            raise ValueError(f"Unsupported stance labels in artifact: {self.labels}")

    @staticmethod
    def _margin(pipe, texts: list[str]) -> np.ndarray:
        dec = np.asarray(pipe.decision_function(texts)).reshape(len(texts), -1)
        if dec.shape[1] == 1:
            return np.abs(dec[:, 0])
        s = np.sort(dec, axis=1)
        return s[:, -1] - s[:, -2]

    def predict(self, texts: list[str]) -> pd.DataFrame:
        texts = ["" if t is None else str(t) for t in texts]
        valid = np.array([len(t.strip()) >= MIN_CHARS for t in texts])
        out = pd.DataFrame({"text": texts, "input_status": np.where(valid, "ok", "too_short_or_empty")})
        for col in ("final_stance", "stance_raw", "stance_margin", "sentiment_prediction",
                    "sentiment_model_source", "prediction_method", "stance_label_source"):
            out[col] = None
        out["disclaimer"] = DISCLAIMER
        valid_texts = [t for t, ok in zip(texts, valid) if ok]
        if valid_texts:
            preds = self.stance_pipe.predict(valid_texts)
            margins = np.round(self._margin(self.stance_pipe, valid_texts), 4)
            out.loc[valid, "final_stance"] = preds
            out.loc[valid, "stance_raw"] = preds
            out.loc[valid, "stance_margin"] = margins
            out.loc[valid, "prediction_method"] = "stance_model_direct"
            out.loc[valid, "stance_label_source"] = LABEL_SOURCE
        return out
