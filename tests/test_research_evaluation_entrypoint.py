"""
tests/test_research_evaluation_entrypoint.py
============================================
Tests for scripts/run_research_evaluation.py (consolidated, non-destructive evaluation).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_research_evaluation as ev  # noqa: E402

ALLOWED = {"PASSED", "FAILED", "WARN", "BLOCKED", "NOT RUN"}


@pytest.fixture(scope="module")
def results():
    return ev.run_all()


def test_every_item_has_an_allowed_status_and_a_section(results):
    assert results
    for r in results:
        assert r["status"] in ALLOWED and r["section"] and r["name"]


def test_no_check_fails_on_the_current_project(results):
    failed = [r for r in results if r["status"] == "FAILED"]
    assert not failed, failed


def test_human_dependent_work_is_blocked_not_passed(results):
    by = {r["name"]: r["status"] for r in results}
    assert by["pilot agreement analysis"] == "BLOCKED"
    assert by["expansion_v2 agreement analysis"] == "BLOCKED"
    assert by["44-pair concern-coherence review"] == "BLOCKED"
    assert by["supervised stance baselines (TF-IDF + LR/SVM)"] == "BLOCKED"
    assert by["concern-priority scoring"] == "NOT RUN"
    assert "full-corpus (13,541) embedding and clustering" not in by          # superseded by section 6
    full = [i for i in results if i["section"].startswith("6. Full-corpus")]
    assert full and all(i["status"] in ("PASSED", "NOT RUN") for i in full)
    assert all("NOT human-validated" in i["section"] for i in full)
    exp = [i for i in results if i["section"].startswith("7. Clustering experiments v2")]
    assert exp and all(i["status"] in ("PASSED", "NOT RUN") for i in exp)
    assert all("NOT human-validated" in i["section"] and "proxies only" in i["section"] for i in exp)


def test_unsupervised_results_are_labelled_unvalidated(results):
    sections = {r["section"] for r in results}
    assert any("UNSUPERVISED" in s and "not human-validated" in s for s in sections)
    stored = [r for r in results if r["name"].startswith("stored comparison")]
    assert stored and all("not concern validity" in r["detail"] for r in stored)


def test_rendered_report_has_interpretation_rules_and_separates_constructs(results):
    text = ev.render(results)
    for fragment in ("NOT human-validated", "Sentiment (tone), stance", "design weights", "never that two comments came from the same respondent"):
        assert fragment in text
    assert len(ev.INTERPRETATION_RULES) >= 5
    assert "BLOCKED" in text and "NOT RUN" in text


def test_module_never_imports_modelling_or_embedding_libraries():
    src = (ROOT / "scripts" / "run_research_evaluation.py").read_text(encoding="utf-8")
    for banned in ("torch", "transformers", "sentence_transformers", "umap", "hdbscan", "sklearn"):
        assert not re.search(rf"^\s*(import|from)\s+{banned}\b", src, re.M), banned
    assert ".fit(" not in src and "to_csv(" not in src


# ── readiness mapping (agreement.analyze stubbed) ─────────────────────────────

@pytest.mark.parametrize("status,expected", [
    ("missing_files", "BLOCKED"), ("pending_no_annotations", "BLOCKED"), ("invalid", "FAILED"), ("ok", "PASSED"),
])
def test_agreement_status_maps_to_evaluation_status(monkeypatch, status, expected):
    fake = {"status": status, "errors": ["e"], "warnings": [],
            "segments": {"expansion": {"sample_sizes": {"paired_valid": 7, "rows_in_segment": 9},
                                       "stance_agreement": {"status": "computed", "kappa": 0.5}}}}
    monkeypatch.setattr(ev.agreement, "analyze", lambda *_a, **_k: fake)
    items = {i["name"]: i["status"] for i in ev.check_annotation_readiness()}
    assert items["pilot agreement analysis"] == expected
    assert items["expansion_v2 agreement analysis"] == expected


def test_completed_analysis_item_states_stratified_not_population(monkeypatch):
    fake = {"status": "ok", "errors": [], "warnings": [],
            "segments": {"expansion": {"sample_sizes": {"paired_valid": 7, "rows_in_segment": 9},
                                       "stance_agreement": {"status": "computed", "kappa": 0.5}}}}
    monkeypatch.setattr(ev.agreement, "analyze", lambda *_a, **_k: fake)
    item = next(i for i in ev.check_annotation_readiness() if i["name"] == "expansion_v2 agreement analysis")
    assert "stratified sample" in item["detail"] and "not population estimates" in item["detail"]


# ── protected baseline ────────────────────────────────────────────────────────

def test_baseline_init_refuses_overwrite_and_detects_changes(tmp_path):
    root = tmp_path / "proj"
    (root / "data" / "annotation").mkdir(parents=True)
    target = root / "data" / "annotation" / "pilot_annotation_50.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    data = ev.init_baseline(root, baseline)
    assert "data/annotation/pilot_annotation_50.csv" in data["files"]
    with pytest.raises(FileExistsError):
        ev.init_baseline(root, baseline)
    assert ev.check_protected(root, baseline)[0]["status"] == "PASSED"
    target.write_text("a,b\n1,3\n", encoding="utf-8")
    changed = ev.check_protected(root, baseline)[0]
    assert changed["status"] == "FAILED" and "pilot_annotation_50.csv" in changed["detail"]
    target.unlink()
    assert ev.check_protected(root, baseline)[0]["status"] == "FAILED"


def test_missing_baseline_is_a_warning_not_a_failure(tmp_path):
    out = ev.check_protected(ROOT, tmp_path / "none.json")
    assert out[0]["status"] == "WARN"


@pytest.mark.skipif(not ev.BASELINE_PATH.exists(), reason="baseline not initialised")
def test_real_baseline_covers_protected_artifacts_without_completed_outputs():
    files = json.loads(ev.BASELINE_PATH.read_text(encoding="utf-8"))["files"]
    for must in ("data/annotation/pilot_annotation_50.csv", "data/annotation/expansion_pool_400.csv",
                 "data/annotation/expansion_annotator_1.csv", "data/annotation/expansion_annotator_2.csv",
                 "data/clustering/cluster_labels.csv", "data/embeddings/minilm_embeddings.npy",
                 "data/comparisons/multilingual_embedding_450/human_review/reviewer_1_template.csv",
                 "data/comparisons/multilingual_embedding_450/human_review/reviewer_2_template.csv"):
        assert must in files, must
    assert not any("/completed/" in f or "agreement_analysis" in f for f in files)


# ── report writing ────────────────────────────────────────────────────────────

def test_main_writes_new_report_folder_and_refuses_existing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ev, "OUTPUT_BASE", tmp_path / "evals")
    monkeypatch.setattr(ev, "run_all", lambda: [ev.item("S", "x", "PASSED", "ok"), ev.item("S", "y", "BLOCKED", "waiting")])
    assert ev.main(["--tag", "t1"]) == 0
    folder = tmp_path / "evals" / "run_t1"
    assert (folder / "evaluation_report.md").exists() and (folder / "evaluation_results.json").exists()
    before = (folder / "evaluation_report.md").read_text(encoding="utf-8")
    assert ev.main(["--tag", "t1"]) == 1
    assert (folder / "evaluation_report.md").read_text(encoding="utf-8") == before


def test_main_exit_code_is_nonzero_only_for_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "OUTPUT_BASE", tmp_path / "evals")
    monkeypatch.setattr(ev, "run_all", lambda: [ev.item("S", "x", "BLOCKED", "waiting"), ev.item("S", "z", "NOT RUN", "by design")])
    assert ev.main(["--no-write"]) == 0
    monkeypatch.setattr(ev, "run_all", lambda: [ev.item("S", "bad", "FAILED", "boom")])
    assert ev.main(["--no-write"]) == 1
    assert not (tmp_path / "evals").exists()
