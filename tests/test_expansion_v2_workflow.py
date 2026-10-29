"""
tests/test_expansion_v2_workflow.py
===================================
Annotator templates, v2 verification module and workflow documents.
Mutation tests run on COPIES of the v2 artifacts in temp directories.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import create_annotation_expansion_pool_v2 as builder  # noqa: E402
import create_expansion_v2_annotator_templates as templates  # noqa: E402
import verify_expansion_pool_v2 as verify  # noqa: E402
import analyze_annotation_agreement as agreement  # noqa: E402

ANN = ROOT / "data" / "annotation"
POOL, LEDGER, MANIFEST = builder.POOL_PATH, builder.LEDGER_PATH, builder.MANIFEST_PATH
T1, T2, TMAN = templates.TEMPLATE_1, templates.TEMPLATE_2, templates.TEMPLATES_MANIFEST
needs_v2 = pytest.mark.skipif(not (POOL.exists() and MANIFEST.exists()), reason="v2 pool not built")
needs_templates = pytest.mark.skipif(not (T1.exists() and T2.exists() and TMAN.exists()), reason="templates not created")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── Templates (real) ──────────────────────────────────────────────────────────

@needs_v2
@needs_templates
def test_real_templates_are_blank_identical_copies_of_the_pool():
    pool = pd.read_csv(POOL, dtype=str, keep_default_na=False)
    for t in (T1, T2):
        assert t.read_bytes() == POOL.read_bytes()
        df = pd.read_csv(t, dtype=str, keep_default_na=False)
        assert len(df) == 400 and list(df.columns) == list(pool.columns)
        assert (df[builder.ANNOTATION_COLUMNS] == "").all().all()
    info = json.loads(TMAN.read_text(encoding="utf-8"))
    assert info["source_pool_sha256"] == sha(POOL)
    assert info["pilot_segment_rows"] == 50 and info["expansion_segment_rows"] == 350
    assert info["labels_created"] == 0
    assert "not part of the 350" in info["segment_rule"]


@needs_v2
@needs_templates
def test_template_segments_are_explicit_in_row_order():
    df = pd.read_csv(T1, dtype=str, keep_default_na=False)
    assert df.pilot_id.iloc[:50].str.startswith("pilot_").all()
    assert df.pilot_id.iloc[50:].str.startswith("expansion_v2_").all()
    assert df.pilot_id.iloc[50:].nunique() == 350


@needs_v2
@needs_templates
def test_templates_are_not_the_original_expansion_artifacts_and_originals_unchanged():
    assert T1.name != "expansion_annotator_1.csv"
    for name in ("expansion_pool_400.csv", "expansion_annotator_1.csv", "expansion_annotator_2.csv"):
        assert sha(ANN / name) == "f399885e3f4f406d6fcc5150de962347469ec16fce56e8361c42127e0d570cad"
    assert len(pd.read_csv(ANN / "expansion_pool_400.csv")) == 400


# ── Template creation (fixtures) ──────────────────────────────────────────────

def _mini_pool(tmp_path, with_label=False, bad_hash=False):
    cols = ["pilot_id", "source_dataset", "source_row_number", "original_text", *builder.ANNOTATION_COLUMNS,
            "source_language", "language_basis"]
    rows = [[f"pilot_{i:03d}" if i <= 2 else f"expansion_v2_{i - 2:03d}", "d", str(i), f"text {i}", "", "", "", "", "", "",
             "English", "basis"] for i in range(1, 6)]
    df = pd.DataFrame(rows, columns=cols)
    if with_label:
        df.loc[0, "stance"] = "Support"
    pool = tmp_path / "pool.csv"
    df.to_csv(pool, index=False, lineterminator="\n")
    manifest = {"outputs_sha256": {"pool.csv": "0" * 64 if bad_hash else sha(pool)},
                "stats": {"pool_rows": 5, "pilot_rows": 2}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return pool, mpath


def _create(tmp_path, pool, mpath):
    return templates.create_templates(pool, mpath, tmp_path / "t1.csv", tmp_path / "t2.csv", tmp_path / "tm.json")


def test_create_templates_makes_two_identical_blank_copies_and_refuses_overwrite(tmp_path):
    pool, mpath = _mini_pool(tmp_path)
    info = _create(tmp_path, pool, mpath)
    assert (tmp_path / "t1.csv").read_bytes() == pool.read_bytes() == (tmp_path / "t2.csv").read_bytes()
    assert info["pilot_segment_rows"] == 2 and info["expansion_segment_rows"] == 3
    before = (tmp_path / "t1.csv").read_bytes()
    with pytest.raises(FileExistsError):
        _create(tmp_path, pool, mpath)
    assert (tmp_path / "t1.csv").read_bytes() == before


def test_create_templates_refuses_modified_pool_or_labelled_pool(tmp_path):
    pool, mpath = _mini_pool(tmp_path, bad_hash=True)
    with pytest.raises(ValueError, match="SHA-256"):
        _create(tmp_path, pool, mpath)
    assert not (tmp_path / "t1.csv").exists()
    sub = tmp_path / "labelled"
    sub.mkdir()
    pool2, m2 = _mini_pool(sub, with_label=True)
    with pytest.raises(ValueError, match="annotation values"):
        templates.create_templates(pool2, m2, sub / "t1.csv", sub / "t2.csv", sub / "tm.json")


def test_template_script_never_writes_labels():
    src = (ROOT / "scripts" / "create_expansion_v2_annotator_templates.py").read_text(encoding="utf-8")
    assert "stance" not in src.replace("builder.ANNOTATION_COLUMNS", "")


# ── Verification module ───────────────────────────────────────────────────────

def _copy_v2(tmp_path):
    for p in (POOL, LEDGER, MANIFEST, builder.PILOT_PATH):
        shutil.copy2(p, tmp_path / p.name)
    return (tmp_path / POOL.name, tmp_path / LEDGER.name, tmp_path / MANIFEST.name, tmp_path / builder.PILOT_PATH.name)


def _run(paths, templates_=()):
    pool, ledger, manifest, pilot = paths
    return {r["check"]: r["status"] for r in verify.run_checks(pool, ledger, manifest, pilot, template_paths=tuple(templates_))}


@needs_v2
def test_real_v2_pool_passes_all_checks():
    results = verify.run_checks()
    failed = [r for r in results if r["status"] == "FAIL"]
    assert not failed, failed
    assert len(results) >= 19


@needs_v2
def test_verification_detects_text_tampering_and_hash_mismatch(tmp_path):
    paths = _copy_v2(tmp_path)
    pool = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    pool.loc[100, "original_text"] = pool.loc[100, "original_text"] + " tampered"
    pool.to_csv(paths[0], index=False, lineterminator="\n")
    status = _run(paths)
    assert status["manifest_output_hashes"] == "FAIL"
    assert status["source_text_provenance_verbatim"] == "FAIL"


@needs_v2
def test_verification_detects_duplicate_record_and_filled_labels(tmp_path):
    paths = _copy_v2(tmp_path)
    pool = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    pool.loc[120, ["source_dataset", "source_row_number"]] = pool.loc[121, ["source_dataset", "source_row_number"]].values
    pool.loc[130, "stance"] = "Support"
    pool.to_csv(paths[0], index=False, lineterminator="\n")
    status = _run(paths)
    assert status["source_record_uniqueness"] == "FAIL"
    assert status["annotation_columns_blank"] == "FAIL"


@needs_v2
def test_verification_detects_pilot_changes_and_ledger_damage(tmp_path):
    paths = _copy_v2(tmp_path)
    pool = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    pool.loc[3, "context"] = "changed"
    pool.to_csv(paths[0], index=False, lineterminator="\n")
    ledger = pd.read_csv(paths[1], dtype=str, keep_default_na=False).iloc[:-5]
    ledger.to_csv(paths[1], index=False, lineterminator="\n")
    status = _run(paths)
    assert status["pilot_rows_verbatim_and_first"] == "FAIL"
    assert status["ledger_completeness"] == "FAIL" and status["row_counts"] == "FAIL"


@needs_v2
def test_verification_reports_missing_artifacts_cleanly(tmp_path):
    results = verify.run_checks(tmp_path / "x.csv", tmp_path / "y.csv", tmp_path / "z.json", tmp_path / "p.csv")
    assert results[0]["status"] == "FAIL" and "missing" in results[0]["detail"]


# ── Documents ─────────────────────────────────────────────────────────────────

def test_annotator_readme_states_pilot_rows_are_not_additional_and_uses_tool_filenames():
    text = (ANN / "README_expansion_v2_annotators.md").read_text(encoding="utf-8")
    assert "not additional records" in text and "Never add them to the 350" in text
    for key in ("completed_a", "completed_b"):
        name = agreement.ROUNDS["expansion_v2"][key].name
        assert f"data/annotation/completed/{name}" in text
    for t in (T1, T2):
        assert t.name in text
    assert "survey language" in text and "annotator_A" in text and "annotator_B" in text


def test_workflow_document_commands_reference_existing_scripts_and_files():
    text = (ROOT / "reports" / "annotation_and_evaluation_workflow.md").read_text(encoding="utf-8")
    for script in ("create_expansion_v2_annotator_templates.py", "analyze_annotation_agreement.py",
                   "run_research_evaluation.py", "verify_expansion_pool_v2.py"):
        assert script in text and (ROOT / "scripts" / script).exists()
    for name in (agreement.ROUNDS["pilot"]["completed_a"].name, agreement.ROUNDS["expansion_v2"]["completed_b"].name):
        assert name in text
    assert "never part of the 350" in text


def test_methodology_notes_do_not_equate_duplicate_groups_with_respondents():
    text = (ROOT / "reports" / "expansion_pool_v2_methodology_notes.md").read_text(encoding="utf-8")
    assert "not a respondent identifier" in text
    assert "does not show that two comments came from the same person" in text
    assert "Not representative without weights" in text
    assert "Inclusion fraction" in text and "administrative tags" in text


def test_compilation_quota_rule_matches_documentation():
    q = builder.allocate_quotas(160, {"NA": 1344, "Blank": 12, **{f"T{i}": 100 + i for i in range(13)}},
                                capped=frozenset({"NA", "Blank"}))
    assert sum(q.values()) == 160 and q["NA"] == 10 and q["Blank"] == 10
    assert sorted(q[f"T{i}"] for i in range(13)).count(11) == 10
