"""
tests/test_results_access.py
============================
Read-only, manifest-verified results-access layer (app/results_access.py): integrity verification, missing files,
checksum mismatches, staleness, malformed data, and evidence/source-id consistency. Failure cases run on temporary
copies of a synthetic project tree built with the existing pipelines; the real artifacts are only read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from app.results_access import ArtifactPaths, ArtifactUnavailable, ResultsStore  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Synthetic project laid out like data/full_corpus/, produced by the existing pipeline scripts."""
    import run_concern_discovery as rcd
    import run_concern_prioritization as rcp
    import run_language_keywords as rlk
    import test_concern_discovery as tcd
    root = tmp_path_factory.mktemp("proj")
    fc = root / "data" / "full_corpus"
    run_v1, exp = tcd._build_tree(fc)
    rcd.stage_run(fc / "concern_discovery_v1", run_v1, exp)
    rcp.stage_run(fc / "concern_prioritization_v1", None, fc / "concern_discovery_v1", run_v1)
    rlk.run(fc / "language_keywords_v1", fc / "concern_discovery_v1", exp, run_v1)
    return root


@pytest.fixture()
def proj(built, tmp_path):
    dst = tmp_path / "p"
    shutil.copytree(built, dst)
    return dst


def store(root: Path) -> ResultsStore:
    return ResultsStore(ArtifactPaths(root=root))


def fc(root: Path) -> Path:
    return root / "data" / "full_corpus"


def rehash(manifest: Path, key: str, file: Path) -> None:
    """Simulate an edited file whose checksum was also re-recorded, so only schema/evidence checks can catch it."""
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["outputs_sha256"][key] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(m), encoding="utf-8")


def edit_csv(path: Path, fn) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    rows, fields = fn(rows, fields)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def statuses(s: ResultsStore) -> dict[str, str]:
    return {c: v["status"] for c, v in s.verification_report()["components"].items()}


# ── happy path ────────────────────────────────────────────────────────────────

def test_verified_components_and_payload_contract(proj):
    s = store(proj)
    st = statuses(s)
    assert st == {"corpus": "VERIFIED", "concern_discovery": "VERIFIED", "prioritization": "VERIFIED",
                  "language_keywords": "VERIFIED", "traceability": "MISSING"}       # no ledger in the synthetic tree
    p = s.prioritized_concerns()
    assert p["kind"] == "exploratory" and p["human_validated"] is False
    assert "not evidence about Indian citizens" in p["scope"] and any("public importance" in c for c in p["caveats"])
    assert p["provenance"]["manifest_sha256"] and p["weights"]
    for r in p["data"]:
        assert abs(sum(r["contributions"].values()) - r["exploratory_score"]) < 1e-5
    assert [r["rank_multi_signal"] for r in p["data"]] == sorted(r["rank_multi_signal"] for r in p["data"])
    fb = s.frequency_baseline()["data"]
    assert [r["rank_frequency"] for r in fb] == list(range(1, len(fb) + 1))
    assert s.verify("traceability")["status"] == "MISSING"
    assert s.verification_report()["all_verified"] is False


def test_ids_and_evidence_match_the_original_corpus(proj):
    s = store(proj)
    with open(fc(proj) / "run_v1/corpus/corpus_entries.csv", encoding="utf-8", newline="") as f:
        corpus = {int(r["corpus_id"]): r for r in csv.DictReader(f)}
    with open(fc(proj) / "concern_prioritization_v1/ranking.csv", encoding="utf-8", newline="") as f:
        ranked_ids = {int(r["cluster_id"]) for r in csv.DictReader(f)}
    assert {r["cluster_id"] for r in s.prioritized_concerns()["data"]} == ranked_ids      # original cluster ids
    for g in s.concern_groups()["data"]:
        for e in s.concern_group(g["cluster_id"])["data"]["evidence"]:
            src = corpus[int(e["corpus_id"])]
            assert (src["source_dataset"], src["source_record_id"]) == (e["source_dataset"], str(e["source_record_id"]))
            assert e["quote"] in src["original_text"] and e["full_text"] == src["original_text"]
    cid = next(iter(ranked_ids))
    for e in s.ranking_evidence(cid)["data"]:
        assert e["quote"] in corpus[e["corpus_id"]]["original_text"]
    inv = s.subgroup_eligibility()["data"]
    kw = s.language_keywords()["data"]
    eligible = {(r["cluster_id"], r["language"]) for r in inv if r["eligible"]}
    assert {(k["cluster_id"], k["language"]) for k in kw} <= eligible
    assert all(r["exclusion_reason"] for r in inv if not r["eligible"])


def test_unknown_component_and_cluster(proj):
    s = store(proj)
    with pytest.raises(KeyError):
        s.verify("embeddings")
    with pytest.raises(KeyError):
        s.concern_group(9999)
    with pytest.raises(KeyError):
        s.ranking_evidence(9999)


# ── failure handling ──────────────────────────────────────────────────────────

def test_missing_component_is_isolated(proj):
    (fc(proj) / "language_keywords_v1" / "run_manifest.json").unlink()
    s = store(proj)
    assert statuses(s)["language_keywords"] == "MISSING"
    with pytest.raises(ArtifactUnavailable) as e:
        s.language_keywords()
    assert e.value.status == "MISSING"
    assert s.concern_groups()["data"]                                        # other components still served
    assert "unavailable (MISSING)" in s.limitations()["data"]["language_keywords"]


def test_checksum_mismatch(proj):
    p = fc(proj) / "concern_prioritization_v1" / "ranking.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    s = store(proj)
    assert statuses(s)["prioritization"] == "MISMATCH"
    with pytest.raises(ArtifactUnavailable):
        s.prioritized_concerns()


def test_stale_artifact_is_reverified(proj):
    s = store(proj)
    assert s.concern_groups()["data"]
    p = fc(proj) / "concern_discovery_v1" / "selected" / "concern_groups.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("candidate", "candidat", 1), encoding="utf-8")
    with pytest.raises(ArtifactUnavailable) as e:
        s.concern_groups()
    assert e.value.status == "MISMATCH"


def test_malformed_file(proj):
    d = fc(proj) / "concern_prioritization_v1"
    edit_csv(d / "not_ranked.csv", lambda rows, f: (rows, [c for c in f if c != "not_ranked_reason"]))
    rehash(d / "run_manifest.json", "not_ranked.csv", d / "not_ranked.csv")
    s = store(proj)
    v = s.verify("prioritization")
    assert v["status"] == "MALFORMED" and "not_ranked_reason" in v["problems"][0]
    (fc(proj) / "language_keywords_v1" / "config.json").write_text("{not json", encoding="utf-8")
    assert s.verify("language_keywords")["status"] in ("MISMATCH", "MALFORMED")


def test_inconsistent_ranking_evidence(proj):
    d = fc(proj) / "concern_prioritization_v1"

    def wrong_source(rows, f):
        rows[0]["source_record_id"] = "999999"
        return rows, f
    edit_csv(d / "ranking_evidence.csv", wrong_source)
    rehash(d / "run_manifest.json", "ranking_evidence.csv", d / "ranking_evidence.csv")
    v = store(proj).verify("prioritization")
    assert v["status"] == "INCONSISTENT" and "source ids differ" in v["problems"][0]


def test_inconsistent_concern_evidence(proj):
    d = fc(proj) / "concern_discovery_v1"
    sel = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))["selected_configuration"]
    sp = d / "configurations" / sel / "concern_summaries.json"
    summ = json.loads(sp.read_text(encoding="utf-8"))
    summ["clusters"][0]["evidence"][0]["quote"] = "An invented quotation"
    sp.write_text(json.dumps(summ), encoding="utf-8")
    rehash(d / "run_manifest.json", f"configurations/{sel}/concern_summaries.json", sp)
    v = store(proj).verify("concern_discovery")
    assert v["status"] == "INCONSISTENT" and "not verbatim" in v["problems"][0]


def test_changed_corpus_blocks_evidence(proj):
    c = fc(proj) / "run_v1" / "corpus" / "corpus_entries.csv"
    c.write_text(c.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    st = statuses(store(proj))
    assert st["corpus"] == "MISMATCH" and st["concern_discovery"] != "VERIFIED"


def test_access_is_read_only(proj):
    before = {p: p.read_bytes() for p in proj.rglob("*") if p.is_file()}
    s = store(proj)
    s.verification_report()
    s.concern_groups(); s.prioritized_concerns(); s.frequency_baseline()
    s.language_keywords(); s.subgroup_eligibility(); s.limitations()
    for g in s.concern_groups()["data"]:
        s.concern_group(g["cluster_id"])
    assert {p: p.read_bytes() for p in proj.rglob("*") if p.is_file()} == before


def test_status_and_limitations(proj):
    s = store(proj)
    fake = [{"section": "S", "name": "a", "status": "BLOCKED", "detail": "x"},
            {"section": "S", "name": "b", "status": "PASSED", "detail": "y"}]
    ev = s.evaluation_status(run=lambda: fake)
    assert ev["summary"] == {"BLOCKED": 1, "PASSED": 1} and len(ev["blocked"]) == 1 and ev["human_validated"] is False
    lim = s.limitations()["data"]
    assert "not human-validated" in lim["stance"] and "not evidence about Indian citizens" in lim["scope"]


@pytest.mark.skipif(not (ROOT / "data/full_corpus/concern_prioritization_v1/run_manifest.json").exists(),
                    reason="real artifacts not present")
def test_real_project_verifies():
    s = ResultsStore()
    assert s.verification_report()["all_verified"]
    g = s.concern_group(30)["data"]
    assert g["evidence"] and g["cluster_id"] == 30
