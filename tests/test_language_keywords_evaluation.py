"""
tests/test_language_keywords_evaluation.py
==========================================
Read-only research-evaluation check of data/full_corpus/language_keywords_v1 (section 12 of
scripts/run_research_evaluation.py). Failure cases run on temporary copies; the real outputs are never touched.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_research_evaluation as ev  # noqa: E402

pytestmark = pytest.mark.skipif(not (ev.LANGUAGE_KEYWORDS_DIR / "run_manifest.json").exists(),
                                reason="language_keywords_v1 not present")


def statuses(res):
    return {r["name"]: r["status"] for r in res}


@pytest.fixture()
def copy(tmp_path):
    """Copy of the experiment folder and its recorded inputs, laid out like data/full_corpus/."""
    full = tmp_path / "full_corpus"
    out = full / "language_keywords_v1"
    shutil.copytree(ev.LANGUAGE_KEYWORDS_DIR, out)
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    for rel in m["inputs_sha256"]:
        (full / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ev.LANGUAGE_KEYWORDS_DIR.parent / rel, full / rel)
    return out


def _rewrite_csv(path: Path, edit) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    fields = list(rows[0])
    edit(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _rehash(out: Path, name: str) -> None:
    """Simulate an output that was edited AND re-recorded, so that only the semantic invariant can catch it."""
    mp = out / "run_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["outputs_sha256"][name] = ev.sha256_file(out / name)
    mp.write_text(json.dumps(m), encoding="utf-8")


def test_project_passes_and_check_is_read_only():
    before = {p: p.read_bytes() for p in ev.LANGUAGE_KEYWORDS_DIR.iterdir()}
    res = ev.check_language_keywords()
    assert [r["status"] for r in res] == ["PASSED"] * 5
    assert {p: p.read_bytes() for p in ev.LANGUAGE_KEYWORDS_DIR.iterdir()} == before


def test_missing_manifest_is_not_run(tmp_path):
    assert ev.check_language_keywords(tmp_path)[0]["status"] == "NOT RUN"


def test_changed_output_fails(copy):
    (copy / "summary.json").write_text("{}", encoding="utf-8")
    assert statuses(ev.check_language_keywords(copy))["keyword outputs match run_manifest.json"] == "FAILED"


def test_changed_cluster_assignments_fail(copy):
    lab = next(copy.parent.glob("experiments_v2/*/labels_seed42.npy"))
    lab.write_bytes(lab.read_bytes() + b"\0")
    assert statuses(ev.check_language_keywords(copy))["cluster assignments and other inputs unchanged"] == "FAILED"


def test_rule_violations_fail(copy):
    def bump(rows):
        r = next(r for r in rows if r["eligible"] == "False" and r["language"] == "French" and r["independent_texts"] != "0")
        r["eligible"], r["exclusion_reason"] = "True", ""                  # below-minimum subgroup marked eligible
        r2 = next(r for r in rows if r["eligible"] == "False")
        r2["exclusion_reason"] = ""                                          # exclusion without reason
    _rewrite_csv(copy / "subgroup_inventory.csv", bump)
    _rehash(copy, "subgroup_inventory.csv")
    res = statuses(ev.check_language_keywords(copy))
    assert res["minimum of 15 independent texts and recorded exclusion reasons"] == "FAILED"
    assert res["keyword outputs match run_manifest.json"] == "PASSED"      # only the semantic check catches it


def test_keywords_for_excluded_subgroup_fail(copy):
    def add(rows):
        rows.append({**rows[0], "cluster_id": "999", "language": "French", "rank": "1"})
    _rewrite_csv(copy / "language_keywords.csv", add)
    _rehash(copy, "language_keywords.csv")
    assert statuses(ev.check_language_keywords(copy))["keywords only for eligible subgroups"] == "FAILED"


def test_removed_disclaimer_fails(copy):
    rp = copy / "report.md"
    rp.write_text(rp.read_text(encoding="utf-8").replace("not evidence of public importance", "important"), encoding="utf-8")
    _rehash(copy, "report.md")
    assert statuses(ev.check_language_keywords(copy))["descriptive-only limitations retained"] == "FAILED"


def test_changed_minimum_in_config_fails(copy):
    cp = copy / "config.json"
    cfg = json.loads(cp.read_text(encoding="utf-8"))
    cfg["min_independent_texts"] = 10
    cp.write_text(json.dumps(cfg), encoding="utf-8")
    _rehash(copy, "config.json")
    assert statuses(ev.check_language_keywords(copy))["minimum of 15 independent texts and recorded exclusion reasons"] == "FAILED"
