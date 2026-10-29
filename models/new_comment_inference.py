"""
models/new_comment_inference.py
===============================
Inference for NEW consultation comments using only components with saved, hash-verified artifacts.

Supported
    stance     TF-IDF + Linear SVM / Logistic Regression trained on AI-generated working labels
               (data/stance_training_ai_working_labels/run_pool400_ai_v1_3class). Output is a provisional machine
               suggestion; the training labels are not human-verified.
    sentiment  TF-IDF + Linear SVM / Logistic Regression trained on AI-generated sentiment working labels
               (data/sentiment_training/consultation_ai_labels_tfidf_v1). Tone only; never stance.

Not supported (returned as explicit status, never guessed)
    concern group   no validated assignment method: nearest-centroid assignment failed the pre-set acceptance
                    criteria (data/full_corpus/concern_audit_v1/audit_results.json, part 3) and HDBSCAN/UMAP models were
                    not saved. New comments are NOT assigned to existing groups.
    priority        rankings describe existing groups only; a single comment has no priority score.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
SENTIMENT_RUN = ROOT / "data" / "sentiment_training" / "consultation_ai_labels_tfidf_v1"
CONCERN_AUDIT = ROOT / "data" / "full_corpus" / "concern_audit_v1" / "audit_results.json"
DEFAULT_MODEL = "linsvm"
MIN_CHARS = 3
PROVENANCE_NOTE = ("Stance and sentiment suggestions come from TF-IDF baselines trained on AI-generated working labels "
                   "(not human-verified). Held-out agreement with those labels is modest; treat outputs as provisional.")


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, incomplete or fails its SHA-256 check."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_verified_pipeline(run_dir: Path, model: str):
    """Verify the full run, source inputs, task label map and model hash before unpickling."""
    import joblib
    meta_p = run_dir / "metadata.json"
    if not meta_p.is_file():
        raise ArtifactError(f"{run_dir} has no metadata.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("status") != "trained_and_evaluated":
        raise ArtifactError(f"{run_dir} is not a completed training run")
    if meta.get("label_type") != "ai_generated_unverified" or meta.get("human_labels_only") is not False:
        raise ArtifactError(f"{run_dir} does not declare the expected unverified AI label provenance")
    declared_models = meta.get("models") or [
        name.removeprefix("pipeline_").removesuffix(".joblib")
        for name in meta.get("artifacts_sha256", {})
        if name.startswith("pipeline_") and name.endswith(".joblib")
    ]
    if model not in declared_models:
        raise ArtifactError(f"{model} is not listed in {run_dir}/metadata.json")
    for name, want in meta.get("artifacts_sha256", {}).items():
        artifact = run_dir / name
        if not artifact.is_file() or _sha(artifact) != want:
            raise ArtifactError(f"{name} is missing or does not match its recorded SHA-256")
    if not meta.get("artifacts_sha256"):
        raise ArtifactError(f"{run_dir} has no artifact checksums")
    for rel, want in meta.get("inputs_sha256", {}).items():
        source = ROOT / rel
        if not source.is_file() or _sha(source) != want:
            raise ArtifactError(f"training input {rel} is missing or does not match its recorded SHA-256")
    if not meta.get("inputs_sha256"):
        raise ArtifactError(f"{run_dir} has no recorded training input checksums")
    metrics_p = run_dir / "metrics.json"
    labels_p = run_dir / "label_mapping.json"
    config_p = run_dir / ("preprocessing_config.json" if (run_dir / "preprocessing_config.json").is_file() else "config.json")
    try:
        metrics = json.loads(metrics_p.read_text(encoding="utf-8"))
        label_map = json.loads(labels_p.read_text(encoding="utf-8"))
        config = json.loads(config_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{run_dir} has malformed evaluation/configuration artifacts") from exc
    if metrics.get("label_type") != meta["label_type"] or metrics.get("labels") != label_map.get("labels"):
        raise ArtifactError(f"{run_dir} metrics and label mapping are inconsistent")
    if not metrics.get("disclaimer") or not meta.get("disclaimer"):
        raise ArtifactError(f"{run_dir} does not preserve its label caveat")
    final_cfg = config.get("final_per_model", {}).get(model)
    construct = str(label_map.get("construct", "")).lower()
    expected_component = "sentiment" if construct.startswith("sentiment") else "stance"
    if expected_component == "stance" and "stance" not in construct:
        raise ArtifactError(f"{run_dir} does not identify its output as stance")
    if expected_component == "sentiment" and "not stance" not in construct:
        raise ArtifactError(f"{run_dir} sentiment mapping does not distinguish sentiment from stance")
    if not final_cfg or "tfidf" not in final_cfg or model not in final_cfg:
        raise ArtifactError(f"{run_dir} has no saved TF-IDF/model configuration for {model}")
    p = run_dir / f"pipeline_{model}.joblib"
    want = meta.get("artifacts_sha256", {}).get(p.name)
    if not p.is_file() or want is None:
        raise ArtifactError(f"{p.name} is missing from the verified run; refusing to unpickle")
    return joblib.load(p), meta


def cluster_assignment_status(audit_path: Path = CONCERN_AUDIT) -> dict[str, Any]:
    if not audit_path.is_file():
        return {"supported": False, "reason": "concern audit not found; no validated assignment method"}
    p3 = json.loads(audit_path.read_text(encoding="utf-8"))["part3_nearest_centroid_assignment"]
    if p3.get("valid_for_new_comment_assignment"):
        return {"supported": True, "reason": "nearest-centroid passed the pre-set criteria"}
    return {"supported": False,
            "reason": (f"nearest-centroid assignment failed pre-set criteria (agreement among accepted "
                       f"{p3['agreement_among_accepted']}, noise acceptance {p3['noise_acceptance_share']} > "
                       f"{p3['criteria_fixed_before_run']['noise_acceptance_max']}); no saved UMAP/HDBSCAN model")}


@dataclass
class NewCommentPredictor:
    model: str = DEFAULT_MODEL
    stance_run: Path = STANCE_RUN
    sentiment_run: Path = SENTIMENT_RUN

    def __post_init__(self) -> None:
        self.stance, self.stance_meta = load_verified_pipeline(self.stance_run, self.model)
        self.sentiment, self.sentiment_meta = load_verified_pipeline(self.sentiment_run, self.model)
        self.cluster = cluster_assignment_status()

    @staticmethod
    def _margin(pipe, texts: list[str]) -> np.ndarray:
        dec = pipe.decision_function(texts)
        dec = dec.reshape(len(texts), -1)
        if dec.shape[1] == 1:                       # binary: distance from boundary
            return np.abs(dec[:, 0])
        s = np.sort(dec, axis=1)
        return s[:, -1] - s[:, -2]

    def predict(self, texts: list[str]) -> pd.DataFrame:
        texts = ["" if t is None else str(t) for t in texts]
        valid = np.array([len(t.strip()) >= MIN_CHARS for t in texts])
        out = pd.DataFrame({"text": texts, "input_status": np.where(valid, "ok", "too_short_or_empty")})
        for col in ("stance_suggestion", "stance_margin", "sentiment_suggestion", "sentiment_margin"):
            out[col] = None
        v = [t for t, ok in zip(texts, valid) if ok]
        if v:
            out.loc[valid, "stance_suggestion"] = self.stance.predict(v)
            out.loc[valid, "stance_margin"] = np.round(self._margin(self.stance, v), 4)
            out.loc[valid, "sentiment_suggestion"] = self.sentiment.predict(v)
            out.loc[valid, "sentiment_margin"] = np.round(self._margin(self.sentiment, v), 4)
        out["stance_label_source"] = "ai_working_label_unverified"
        out["concern_group"] = None
        out["concern_group_status"] = "not_assigned: " + self.cluster["reason"] if not self.cluster["supported"] else "supported"
        out["priority"] = None
        out["priority_status"] = "not_applicable: rankings describe existing groups, not single comments"
        return out
