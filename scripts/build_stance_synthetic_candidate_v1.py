"""Build, but do not deploy, the versioned v4-selected stance candidate.

Requires the grouped-CV staging gate to pass. The existing production model and its
route remain untouched. This candidate uses all available original records plus a
0.5 synthetic/original ratio, so its training report is not an independent evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as stance_labels
from scripts.experiment_stance_reference_models_v2 import (
    MODEL_LABELS, REFERENCE_LABELS_PATH, build_model,
)

V4_DIR = ROOT / "reports/stance_synthetic_augmentation_v4"
PRODUCTION_MODEL = ROOT / "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib"
RATIO = 0.5
CANDIDATE_CONFIG = {"kind": "word_char_linsvm", "C": 1.0, "class_weight": "balanced"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "reports/stance_synthetic_deployment_audit_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models/candidates/stance_wordchar_synthetic_v1")
    args = parser.parse_args()
    audit_dir = args.audit_dir if args.audit_dir.is_absolute() else ROOT / args.audit_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"write-once candidate directory already exists: {out_dir}")
    audit = json.loads((audit_dir / "metrics.json").read_text(encoding="utf-8"))
    if not audit.get("versioned_candidate_gate_passed_for_further_staging"):
        raise RuntimeError("cross-validation evidence did not pass the predeclared candidate staging gate")
    if audit["v4_reproducibility_check"]["selected_condition"] != "ratio_0.5":
        raise RuntimeError("v4 validation-selected ratio is not 0.5; refusing to change selection")
    if audit["v4_reproducibility_check"]["selected_configuration"] != {"C": 1.0, "class_weight": "balanced"}:
        raise RuntimeError("v4 validation-selected model configuration mismatch")

    status, frame = stance_labels.load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    if status.state != "ok" or frame is None:
        raise RuntimeError(status.detail)
    eligible = frame[frame.gold_stance.isin(MODEL_LABELS)].reset_index(drop=True)
    synthetic = pd.read_json(V4_DIR / "synthetic_training_rows.jsonl", lines=True)
    target_n = int(round(RATIO * len(eligible)))
    per_class = {label: target_n // len(MODEL_LABELS) for label in MODEL_LABELS}
    for label in MODEL_LABELS[:target_n % len(MODEL_LABELS)]:
        per_class[label] += 1
    selected_ids = []
    seed = int(audit["seed"])
    for label in MODEL_LABELS:
        available = synthetic[synthetic.reference_stance == label]
        count = per_class[label]
        if count > len(available):
            raise RuntimeError(f"v4 synthetic pool cannot supply {count} examples for {label}")
        selected_ids.extend(available.sample(n=count, random_state=seed + 500 + MODEL_LABELS.index(label)).synthetic_id.tolist())
    selected = synthetic[synthetic.synthetic_id.isin(selected_ids)].copy()
    train_texts = np.concatenate([eligible.original_text.astype(str).to_numpy(), selected.text.astype(str).to_numpy()])
    train_labels = np.concatenate([eligible.gold_stance.to_numpy(), selected.reference_stance.to_numpy()])
    model = build_model(CANDIDATE_CONFIG["kind"], CANDIDATE_CONFIG["C"], CANDIDATE_CONFIG["class_weight"])
    model.fit(train_texts, train_labels)

    out_dir.mkdir(parents=True)
    model_path = out_dir / "pipeline_linsvm.artifact"
    joblib.dump(model, model_path, compress=3)
    original_cols = [c for c in ("pilot_id", "source_dataset", "source_record_id", "source_row_number", "source_language", "gold_stance") if c in eligible]
    original_manifest = eligible[original_cols].copy().rename(columns={"pilot_id": "record_id", "gold_stance": "reference_stance"})
    original_manifest["is_synthetic"] = False
    original_manifest["training_source"] = "original_reference_record"
    synthetic_manifest = selected[["synthetic_id", "reference_stance", "is_synthetic", "generation_method", "template_id", "language", "random_seed", "synthetic_reason"]].copy()
    synthetic_manifest.rename(columns={"synthetic_id": "record_id"}, inplace=True)
    synthetic_manifest["training_source"] = "synthetic_training_example"
    manifest = pd.concat([original_manifest, synthetic_manifest], ignore_index=True)
    manifest.to_json(out_dir / "training_manifest.jsonl", orient="records", lines=True, force_ascii=False)

    prod_hash = sha(PRODUCTION_MODEL)
    metadata = {
        "candidate_version": "stance_wordchar_synthetic_v1",
        "status": "versioned_candidate_not_deployed",
        "deployed": False,
        "production_route_unchanged": True,
        "production_artifact_sha256": prod_hash,
        "selection_basis": "v4 validation selected ratio_0.5 and C=1.0 balanced; no development-test metric used to select settings",
        "cv_staging_gate": audit["paired_fold_differences"]["ratio_0.5"],
        "cv_is_independent_evaluation": False,
        "model_configuration": CANDIDATE_CONFIG,
        "labels": list(MODEL_LABELS),
        "training_counts": {
            "original_reference_records": int(len(eligible)),
            "original_by_class": eligible.gold_stance.value_counts().reindex(MODEL_LABELS, fill_value=0).astype(int).to_dict(),
            "synthetic_total": int(len(selected)), "synthetic_by_class": per_class,
            "synthetic_to_original_ratio": len(selected) / len(eligible),
        },
        "provenance": "Synthetic examples are separately tagged in training_manifest.jsonl. Original labels and source mappings were read-only.",
        "evaluation_limitation": "All available reference records were used to fit this candidate and the pool was previously exposed during development. No independent or unseen-source test result is claimed.",
        "rollback": "No rollout occurred. The application still loads data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib. Rollback requires no action; keep that route unchanged unless a later reviewed deployment explicitly switches it.",
        "inputs_sha256": {
            "reference_labels": sha(REFERENCE_LABELS_PATH),
            "v4_synthetic_rows": sha(V4_DIR / "synthetic_training_rows.jsonl"),
        },
        "artifact_sha256": sha(model_path),
        "artifact_bytes": model_path.stat().st_size,
    }
    (out_dir / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Stance word+character synthetic candidate v1\n\n"
        "This versioned artifact is a staging candidate only. It is not wired into API or UI inference. "
        "The production artifact and its SHA-256 are recorded in `candidate_metadata.json`.\n\n"
        "Training used all eligible original reference records plus a 0.5 synthetic/original ratio, with the v4 validation-selected word+character LinearSVC configuration (C=1.0, balanced). The grouped CV staging gate supports keeping this candidate for further review, but the pool has prior development exposure and several Unclear synthetic templates carry label risk. Do not promote without reviewing those templates and obtaining source-held-out reference evaluation.\n\n"
        "Rollback: no production change occurred; leave the current route pointing to `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate": str(out_dir), "artifact_sha256": metadata["artifact_sha256"],
                      "bytes": metadata["artifact_bytes"], "production_artifact_sha256_unchanged": prod_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
