"""
tests/test_registry_and_traceability.py
=======================================
Experiment-registry consistency with the repository, and the read-only integrity check of the evidence
traceability ledger in scripts/run_research_evaluation.py.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_research_evaluation as ev  # noqa: E402

REGISTRY = ROOT / "models" / "MODEL_REGISTRY.md"


def registry_paths() -> list[str]:
    text = REGISTRY.read_text(encoding="utf-8")
    found = set(re.findall(r"`((?:data|reports|scripts|models)/[^`{}*<> ]+)`", text))
    return sorted(p.rstrip("/") for p in found)


# Locations the registry names as destinations for future human work; they must NOT exist yet.
EXPECTED_ABSENT = {"data/annotation/completed"}


def test_every_registry_path_exists():
    missing = [p for p in registry_paths() if not (ROOT / p).exists() and p not in EXPECTED_ABSENT]
    assert registry_paths() and missing == []
    if (ROOT / "data/annotation/completed").exists():
        pytest.skip("completed human annotation now exists; update EXPECTED_ABSENT and the stance registry rows")


def test_registry_rows_and_labels():
    text = REGISTRY.read_text(encoding="utf-8")
    for row in ("Per-Language Sub-Cluster Keywords v1", "AI-assisted working labels (exploratory)",
                "Sentiment140 Hashing + SGD Baseline v1", "SentiMix Hinglish Sentiment Baseline v1",
                "Exploratory v1 implemented (not validated)"):
        assert row in text, row
    ai_row = next(line for line in text.splitlines() if "AI-assisted working labels (exploratory)" in line)
    assert "NOT human-verified" in ai_row and "Not stance accuracy" in ai_row
    for line in text.splitlines():
        if line.startswith("| **TF-IDF + Logistic Regression**") or line.startswith("| **TF-IDF + Linear SVM**"):
            assert "Pending Human Labels" in line                       # human baselines stay blocked
    assert "Planned; blocked" in text and "Not run." in text


def test_registry_metrics_match_artifacts():
    text = REGISTRY.read_text(encoding="utf-8")
    s140 = json.loads((ROOT / "data/sentiment_training/sentiment140_hashing_sgd_v1/manifest.json").read_text(encoding="utf-8"))
    smx = json.loads((ROOT / "data/sentiment_training/sentimix_hinglish_baseline_v1/manifest.json").read_text(encoding="utf-8"))
    assert f"Macro-F1 {s140['evaluation_summary']['binary_macro_f1']}" in text
    assert f"Macro-F1 {smx['evaluation_summary']['macro_f1']}" in text
    met = json.loads((ROOT / "data/stance_training_ai_working_labels/run_pilot_primary_draft_v1/metrics.json").read_text(encoding="utf-8"))
    assert f"{met['models']['logreg']['test']['macro_f1']:.2f} (LR)" in text
    meta = json.loads((ROOT / "data/stance_training_ai_working_labels/run_pilot_primary_draft_v1/metadata.json").read_text(encoding="utf-8"))
    assert meta["human_labels_only"] is False and meta["label_type"] == "ai_generated_unverified"


def test_traceability_check_passes_on_project():
    res = ev.check_evidence_traceability()
    assert [r["status"] for r in res] == ["PASSED", "PASSED", "PASSED"]


def _copy_tree(tmp: Path) -> Path:
    m = json.loads(ev.TRACEABILITY_MANIFEST.read_text(encoding="utf-8"))
    for rel in list(m["outputs_sha256"]) + list(ev.TRACEABILITY_INPUT_PATHS.values()) + [
            ev.TRACEABILITY_MANIFEST.relative_to(ROOT).as_posix()]:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp / rel)
    return tmp / ev.TRACEABILITY_MANIFEST.relative_to(ROOT)


def test_traceability_check_detects_changes(tmp_path):
    mp = _copy_tree(tmp_path)
    assert [r["status"] for r in ev.check_evidence_traceability(mp, tmp_path)] == ["PASSED"] * 3
    ledger = tmp_path / "data/full_corpus/concern_prioritization_v1/evidence_traceability_ledger.csv"
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert ev.check_evidence_traceability(mp, tmp_path)[0]["status"] == "FAILED"


def test_traceability_check_detects_changed_inputs_and_missing_manifest(tmp_path):
    mp = _copy_tree(tmp_path)
    inp = tmp_path / ev.TRACEABILITY_INPUT_PATHS["ranking_evidence.csv"]
    inp.write_text(inp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    res = {r["name"]: r["status"] for r in ev.check_evidence_traceability(mp, tmp_path)}
    assert res["ledger inputs unchanged since the audit"] == "FAILED"
    assert ev.check_evidence_traceability(tmp_path / "none.json", tmp_path)[0]["status"] == "NOT RUN"


def test_traceability_check_is_read_only(tmp_path):
    mp = _copy_tree(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    ev.check_evidence_traceability(mp, tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
