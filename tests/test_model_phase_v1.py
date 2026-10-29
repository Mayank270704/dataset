"""Tests for the model training and evaluation phase v1 (AI working labels, sentiment check, audits, inference)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1"
ST3 = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
ST2 = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_support_vs_neutral"
SEN = ROOT / "data" / "sentiment_training" / "consultation_ai_labels_tfidf_v1"
SDC = ROOT / "data" / "sentiment_evaluation" / "consultation_domain_check_v1"
CA = ROOT / "data" / "full_corpus" / "concern_audit_v1"
PA = ROOT / "data" / "full_corpus" / "prioritization_audit_v1"
INF = ROOT / "data" / "inference" / "new_comment_inference_v1"
POOL = ROOT / "data" / "annotation" / "expansion_pool_400_v2.csv"

pytestmark = pytest.mark.skipif(not (AI / "manifest.json").is_file(), reason="model phase v1 outputs not generated")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ── AI labels ──

def test_ai_label_provenance_is_explicitly_unverified():
    prov = rj(AI / "provenance.json")
    assert "NOT human-verified" in prov["status"]
    assert prov["records"] == 400
    w = pd.read_csv(AI / "ai_working_labels_pool400.csv", dtype=str, keep_default_na=False)
    assert set(w.annotator_id) == {"ai_working_label_unverified"}
    assert set(w.stance) <= {"Support", "Oppose", "Mixed", "Neutral", "Unclear"}


def test_ai_label_manifest_matches_and_official_pool_unchanged():
    man = rj(AI / "manifest.json")
    for rel, h in man["outputs_sha256"].items():
        assert sha(ROOT / rel) == h, rel
    assert sha(POOL) == man["inputs_sha256"]["data/annotation/expansion_pool_400_v2.csv"]


def test_ai_labels_source_columns_identical_to_pool():
    w = pd.read_csv(AI / "ai_working_labels_pool400.csv", dtype=str, keep_default_na=False)
    p = pd.read_csv(POOL, dtype=str, keep_default_na=False)
    fixed = [c for c in p.columns if c not in ("stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id")]
    assert w[fixed].equals(p[fixed])


def test_sentiment_labels_kept_separate_from_stance_file():
    w = pd.read_csv(AI / "ai_working_labels_pool400.csv", dtype=str, keep_default_na=False)
    s = pd.read_csv(AI / "ai_sentiment_labels_pool400.csv", dtype=str, keep_default_na=False)
    assert "ai_sentiment" not in w.columns and "stance" not in s.columns
    assert set(s.ai_sentiment) <= {"Positive", "Negative", "Neutral", "Mixed"}


# ── stance and sentiment runs ──

@pytest.mark.parametrize("run", [ST3, ST2, SEN])
def test_run_artifacts_hash_verified_and_flagged_ai(run):
    meta = rj(run / "metadata.json")
    assert meta["status"] == "trained_and_evaluated" and meta["human_labels_only"] is False
    assert meta["label_type"] == "ai_generated_unverified"
    for name, h in meta["artifacts_sha256"].items():
        assert sha(run / name) == h, name


@pytest.mark.parametrize("run,label_col", [(ST3, "ai_stance"), (ST2, "ai_stance")])
def test_no_group_or_normalised_text_crosses_split(run, label_col):
    from models.concern_discovery.schema import normalise_text
    d = pd.read_csv(run / "split_assignments.csv", dtype=str, keep_default_na=False)
    assert set(d.split) == {"train", "validation", "test"}
    assert (d.groupby("leakage_group").split.nunique() == 1).all()
    norm = d.original_text.map(normalise_text)
    assert (d.groupby(norm).split.nunique() == 1).all()


def test_sentiment_split_groups_do_not_cross():
    d = pd.read_csv(SEN / "split_assignments.csv", dtype=str, keep_default_na=False)
    assert (d.groupby("leakage_group").split.nunique() == 1).all()


def test_stance_metrics_report_baselines_and_drop_untrainable_classes():
    m = rj(ST3 / "metrics.json")
    assert m["dropped"]["below_min_class_count"] == {"Oppose": 1}
    assert "Oppose" not in m["labels"] and "Mixed" not in m["labels"]
    assert {"majority_class", "stratified_random"} <= set(m["baselines"])
    for v in m["models"].values():
        assert set(v["test"]["per_class"]) == set(m["labels"])


# ── sentiment domain check ──

def test_sentiment_domain_check_never_presents_tweet_scores_as_consultation_accuracy():
    r = rj(SDC / "results.json")
    assert r["construct"].startswith("sentiment")
    for m in r["models"].values():
        assert "NOT consultation accuracy" in m["in_domain_note"]
    assert "NOT human-verified" in r["B_pool400_vs_ai_labels"]["label_status"]


# ── concern audit / inference ──

def test_cluster_assignment_refused_when_validation_fails():
    r = rj(CA / "audit_results.json")["part3_nearest_centroid_assignment"]
    assert r["valid_for_new_comment_assignment"] is (r["agreement_among_accepted"] >= 0.8 and r["noise_acceptance_share"] <= 0.5)
    if not r["valid_for_new_comment_assignment"]:
        assert not (CA / "nearest_centroid_model.npz").exists()


def test_predictor_outputs_and_refusals():
    from models.new_comment_inference import NewCommentPredictor
    res = NewCommentPredictor().predict(["", "Please publish all contracts as open data.", "Le gouvernement doit être transparent."])
    assert res.loc[0, "input_status"] == "too_short_or_empty" and res.loc[0, "stance_suggestion"] is None
    assert res.loc[1, "stance_suggestion"] in {"Support", "Neutral", "Unclear"}
    assert res.loc[1, "sentiment_suggestion"] in {"Positive", "Negative", "Neutral"}
    assert res.concern_group.isna().all() and res.priority.isna().all()
    assert (res.stance_label_source == "ai_working_label_unverified").all()


def test_tampered_pipeline_is_refused(tmp_path):
    from models.new_comment_inference import ArtifactError, load_verified_pipeline
    run = tmp_path / "run"
    shutil.copytree(ST3, run)
    p = run / "pipeline_linsvm.joblib"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ArtifactError):
        load_verified_pipeline(run, "linsvm")


# ── prioritization audit ──

def test_prioritization_audit_reproduces_ranking_and_evidence_passes():
    r = rj(PA / "audit_results.json")
    assert r["reproducibility"]["same_order"] and r["reproducibility"]["max_abs_score_diff"] == 0
    assert r["reproducibility"]["ranked"] == 55 and r["reproducibility"]["not_ranked"] == 19
    ev = r["evidence_verification"]
    for k in ("ranking_evidence", "representative_examples"):
        assert all(v == 0 for c, v in ev[k].items() if c != "rows")
    assert ev["raw_lineage_check_only_exit_code"] == 0


@pytest.mark.parametrize("d", [SDC, CA, PA, INF, ROOT / "data" / "full_corpus" / "experiments_field_centered_v1"])
def test_output_manifests_match(d):
    man = rj(d / "manifest.json")
    for name, h in man["outputs_sha256"].items():
        assert sha(d / name) == h, f"{d.name}/{name}"


def test_inference_smoke_test_measured_no_accuracy_and_assigned_no_groups():
    s = rj(INF / "summary.json")
    assert s["accuracy_measured"] is False and s["concern_groups_assigned"] == 0
