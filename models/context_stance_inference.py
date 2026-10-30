"""Staging-only inference for the context-aware three-class stance prototype."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.context_stance_features import ContextPredictionInput, SUPPORTED_LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "models/candidates/stance_context_wordchar_v1"


class ContextArtifactError(RuntimeError):
    pass


class ContextStancePredictor:
    def __init__(self, candidate_dir: Path = DEFAULT_CANDIDATE):
        self.candidate_dir = Path(candidate_dir)
        manifest_path = self.candidate_dir / "candidate_manifest.json"
        artifact_path = self.candidate_dir / "pipeline_context_wordchar.artifact"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContextArtifactError("context staging manifest missing or malformed") from exc
        if (manifest.get("candidate_id") != "stance_context_wordchar_v1"
                or manifest.get("status") != "staging_only_not_production"
                or manifest.get("feature_builder") != "models.context_stance_features.build_context_feature"):
            raise ContextArtifactError("context staging manifest has an incompatible candidate or feature contract")
        expected = manifest.get("artifact_sha256")
        if not artifact_path.is_file() or not expected:
            raise ContextArtifactError("context staging artifact missing or unhashed")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ContextArtifactError("context staging artifact SHA-256 mismatch")
        self.pipeline = joblib.load(artifact_path)
        self.manifest = manifest
        self.labels = tuple(str(label) for label in self.pipeline.classes_)
        if set(self.labels) != set(SUPPORTED_LABELS):
            raise ContextArtifactError(f"unexpected classes in context artifact: {self.labels}")

    @staticmethod
    def _margin(pipeline, features: list[str]) -> np.ndarray:
        decisions = np.asarray(pipeline.decision_function(features)).reshape(len(features), -1)
        if decisions.shape[1] == 1:
            return np.abs(decisions[:, 0])
        ordered = np.sort(decisions, axis=1)
        return ordered[:, -1] - ordered[:, -2]

    def predict(self, items: list[ContextPredictionInput]) -> pd.DataFrame:
        for item in items:
            item.validate()
        valid = [item.comment_is_usable for item in items]
        usable = [item.feature_text() for item, ok in zip(items, valid) if ok]
        predictions = list(self.pipeline.predict(usable)) if usable else []
        margins = np.round(self._margin(self.pipeline, usable), 4).tolist() if usable else []
        pred_iter = iter(predictions)
        margin_iter = iter(margins)
        rows = []
        for index, (item, ok) in enumerate(zip(items, valid)):
            context_missing = item.context is None or not item.context.strip()
            question_missing = item.question is None or not item.question.strip()
            prediction = next(pred_iter) if ok else None
            margin = next(margin_iter) if ok else None
            rows.append({
                "item_index": index,
                "input_status": "ok" if ok else "too_short_or_empty",
                "final_stance": prediction,
                "stance_margin": margin,
                "question_missing": question_missing,
                "context_missing": context_missing,
                "language_status": "ignored_not_a_model_feature" if item.language else "not_provided",
                "disclaimer": "Staging prototype only. Provisional agreement with repository reference labels; external performance is not established.",
            })
        result = pd.DataFrame(rows)
        for column in ("final_stance", "stance_margin"):
            result[column] = result[column].astype(object).where(result[column].notna(), None)
        return result
