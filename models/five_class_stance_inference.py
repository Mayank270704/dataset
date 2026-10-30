"""Five-class synthetic candidate inference, isolated from production predictors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.context_stance_features import ContextPredictionInput, build_context_feature
from models.stance_classification.schema import STANCE_LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "models/candidates/stance_context_wordchar_5class_synth_v1"
ARTIFACT_NAME = "pipeline_five_class_stance.artifact"
CANDIDATE_ID = "stance_context_wordchar_5class_synth_v1"


class FiveClassArtifactError(RuntimeError):
    pass


class FiveClassStancePredictor:
    def __init__(self, candidate_dir: Path = DEFAULT_CANDIDATE):
        self.candidate_dir = Path(candidate_dir)
        try:
            self.manifest = json.loads((self.candidate_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FiveClassArtifactError("five-class candidate manifest is missing or malformed") from exc
        artifact = self.candidate_dir / ARTIFACT_NAME
        expected = self.manifest.get("artifact_sha256")
        if (self.manifest.get("candidate_id") != CANDIDATE_ID
                or self.manifest.get("status") != "staging_only_not_production"
                or self.manifest.get("labels") != list(STANCE_LABELS)
                or self.manifest.get("feature_builder") != "models.context_stance_features.build_context_feature"):
            raise FiveClassArtifactError("five-class candidate manifest has an incompatible schema/feature contract")
        if not artifact.is_file() or not expected:
            raise FiveClassArtifactError("five-class artifact is missing or unhashed")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            raise FiveClassArtifactError("five-class artifact SHA-256 does not match manifest")
        self.pipeline = joblib.load(artifact)
        self.labels = tuple(str(value) for value in self.pipeline.classes_)
        if set(self.labels) != set(STANCE_LABELS):
            raise FiveClassArtifactError(f"candidate output classes differ from five-class schema: {self.labels}")

    @staticmethod
    def margin(pipeline, features):
        values = np.asarray(pipeline.decision_function(features)).reshape(len(features), -1)
        if values.shape[1] == 1:
            return np.abs(values[:, 0])
        ordered = np.sort(values, axis=1)
        return ordered[:, -1] - ordered[:, -2]

    def predict(self, items: list[ContextPredictionInput]) -> pd.DataFrame:
        for item in items:
            item.validate()
        valid = [item.comment_is_usable for item in items]
        features = [item.feature_text() for item, ok in zip(items, valid) if ok]
        predictions = list(self.pipeline.predict(features)) if features else []
        margins = np.round(self.margin(self.pipeline, features), 4).tolist() if features else []
        prediction_iter, margin_iter = iter(predictions), iter(margins)
        rows = []
        for index, (item, ok) in enumerate(zip(items, valid)):
            rows.append({
                "item_index": index,
                "input_status": "ok" if ok else "too_short_or_empty",
                "final_stance": next(prediction_iter) if ok else None,
                "stance_margin": next(margin_iter) if ok else None,
                "question_missing": item.question is None or not item.question.strip(),
                "context_missing": item.context is None or not item.context.strip(),
                "disclaimer": "Five-class synthetic staging candidate. Outputs are provisional and not human-validated.",
            })
        result = pd.DataFrame(rows)
        for column in ("final_stance", "stance_margin"):
            result[column] = result[column].astype(object).where(result[column].notna(), None)
        return result
