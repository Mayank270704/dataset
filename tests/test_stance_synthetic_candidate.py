import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "models/candidates/stance_wordchar_synthetic_v1"


def test_versioned_candidate_is_integrity_checked_but_not_deployed():
    metadata = json.loads((CANDIDATE / "candidate_metadata.json").read_text(encoding="utf-8"))
    artifact = CANDIDATE / "pipeline_linsvm.artifact"
    assert metadata["status"] == "versioned_candidate_not_deployed"
    assert metadata["deployed"] is False
    assert metadata["production_route_unchanged"] is True
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert hashlib.sha256((ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib").read_bytes()).hexdigest() == metadata["production_artifact_sha256"]
    model = joblib.load(artifact)
    assert set(model.classes_) == {"Support", "Neutral", "Unclear"}
    predictions = model.predict(["The department should publish these records for residents.", "The update is available on Monday."])
    assert set(predictions).issubset({"Support", "Neutral", "Unclear"})


def test_candidate_manifest_keeps_synthetic_rows_separate_and_tagged():
    manifest = pd.read_json(CANDIDATE / "training_manifest.jsonl", lines=True)
    synthetic = manifest[manifest.is_synthetic]
    original = manifest[~manifest.is_synthetic]
    assert len(original) == 399
    assert len(synthetic) == 200
    assert synthetic.generation_method.notna().all()
    assert synthetic.template_id.notna().all()
    assert synthetic.random_seed.notna().all()
    assert set(original.training_source) == {"original_reference_record"}
    assert set(synthetic.training_source) == {"synthetic_training_example"}
