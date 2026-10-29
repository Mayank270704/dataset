"""
tests/test_human_review_handoff.py
==================================
Reviewer handoff, intake validation, stage gating and leak audits (scripts/human_review_handoff.py),
plus the completed-file checks of the three analysis paths.

All "completed" files here are fixtures written to a temporary copy of the real templates. They are
never written into project folders and are not human judgments.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import human_review_handoff as hh  # noqa: E402

REAL_PRESENT = all((ROOT / t.templates[n]).exists() for t in hh.TRACKS.values() for n in (1, 2))
pytestmark = pytest.mark.skipif(not REAL_PRESENT, reason="real review templates not present")


@pytest.fixture()
def root(tmp_path):
    """Temporary project tree: real templates and reviewer docs, the guide, and a decoy answer key."""
    files = {hh.GUIDE}
    for t in hh.TRACKS.values():
        files |= set(t.templates.values()) | {src for src, _ in t.docs}
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    key = tmp_path / hh.CL / "_key_do_not_share" / "task_a_key.csv"
    key.parent.mkdir(parents=True)
    key.write_text("item_id,cluster_id,intruder_position\nA-000000,1,3\n", encoding="utf-8")
    return tmp_path


def filled(root: Path, track: str, person: int, **edit) -> pd.DataFrame:
    """A syntactically valid completed file (fixture values only)."""
    t = hh.TRACKS[track]
    df = hh.read_csv(root / t.templates[person])
    rid = f"reviewer_{person}"
    if t.kind == "pair44":
        df["reviewer_id"], df["concern_relation"], df["confidence"] = rid, "1", "medium"
    elif t.kind == "cluster_a":
        df["reviewer_id"], df["intruder_position"], df["coherence_of_others"], df["confidence"] = rid, "U", "1", "low"
    elif t.kind == "cluster_b":
        df["reviewer_id"], df["summary_support"], df["misleading_claim"], df["confidence"] = rid, "U", "no", "low"
    else:
        req = df["pilot_id"].str.startswith(t.rows_required) if t.rows_required else pd.Series(True, index=df.index)
        df.loc[req, "stance"] = "Unclear"
        df.loc[req, "annotator_id"] = t.expected_ids[person]
    for (row, col), value in edit.get("cells", {}).items():
        df.loc[row, col] = value
    return df


def save(root: Path, track: str, person: int, df: pd.DataFrame, encoding: str = "utf-8-sig") -> Path:
    p = root / hh.TRACKS[track].completed[person]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding=encoding)
    return p


def state(root: Path, track: str, person: int) -> hh.Check:
    return {(c.track, c.person): c for c in hh.status(root)}[(track, person)]


# ── intake validation ─────────────────────────────────────────────────────────

def test_fresh_project_is_blocked_everywhere(root):
    checks = hh.status(root)
    assert {c.state for c in checks} == {"MISSING"} and len(checks) == 10
    assert all(r.startswith("BLOCKED") for r in hh.track_readiness(checks).values())


@pytest.mark.parametrize("track", sorted(hh.TRACKS))
def test_valid_completed_files_pass_for_every_track(root, track):
    for n in (1, 2):
        save(root, track, n, filled(root, track, n))
    assert state(root, track, 1).state == "VALID" and state(root, track, 2).state == "VALID"
    assert hh.track_readiness(hh.status(root))[track].startswith("READY")


def test_blank_judgment_is_returned_to_reviewer(root):
    df = filled(root, "pair_review_44", 1, cells={(3, "concern_relation"): ""})
    save(root, "pair_review_44", 1, df)
    c = state(root, "pair_review_44", 1)
    assert c.state == "INVALID" and any("blank" in p for p in c.problems)


def test_spreadsheet_formula_corruption_is_detected(root):
    df = filled(root, "cluster_task_a", 1, cells={(0, "comment_2"): "#NAME?"})
    save(root, "cluster_task_a", 1, df)
    c = state(root, "cluster_task_a", 1)
    assert c.state == "INVALID" and any("formula" in p for p in c.problems)


def test_crlf_line_breaks_and_case_are_tolerated(root):
    df = filled(root, "cluster_task_b", 2)
    df = df.map(lambda v: v.replace("\n", "\r\n") if isinstance(v, str) else v)
    df["confidence"] = "High"
    save(root, "cluster_task_b", 2, df)
    assert state(root, "cluster_task_b", 2).state == "VALID"


def test_invalid_values_and_reviewer_id(root):
    df = filled(root, "cluster_task_a", 2, cells={(0, "intruder_position"): "7", (1, "coherence_of_others"): "4"})
    df.loc[2, "reviewer_id"] = "someone_else"
    save(root, "cluster_task_a", 2, df)
    probs = " ".join(state(root, "cluster_task_a", 2).problems)
    assert "intruder_position" in probs and "coherence_of_others" in probs and "several values" in probs


def test_rows_and_columns_must_match_template(root):
    save(root, "pair_review_44", 2, filled(root, "pair_review_44", 2).iloc[:-1])
    assert any("rows differ" in p for p in state(root, "pair_review_44", 2).problems)
    save(root, "cluster_task_b", 1, filled(root, "cluster_task_b", 1).drop(columns=["note"]))
    assert any("columns differ" in p for p in state(root, "cluster_task_b", 1).problems)


def test_stance_rules(root):
    df = filled(root, "stance_pilot", 1)
    df.loc[0, "annotator_id"] = "annotator_B"
    df.loc[1, "evidence_span"] = "words that are not in the comment at all"
    save(root, "stance_pilot", 1, df)
    probs = " ".join(state(root, "stance_pilot", 1).problems)
    assert "annotator_A" in probs and "verbatim" in probs
    save(root, "stance_pilot", 2, filled(root, "stance_pilot", 2).iloc[::-1])
    assert any("order" in p for p in state(root, "stance_pilot", 2).problems)
    v2 = filled(root, "stance_v2", 1)
    v2.loc[0, ["stance", "annotator_id"]] = ["Support", "annotator_A"]          # optional pilot_* row
    save(root, "stance_v2", 1, v2)
    c = state(root, "stance_v2", 1)
    assert c.state == "VALID" and any("never counted among the 350" in w for w in c.warnings)


def test_non_utf8_file_is_rejected(root):
    df = filled(root, "pair_review_44", 1)
    df.loc[0, "shared_concern_note"] = "réponse évidente"
    p = root / hh.TRACKS["pair_review_44"].completed[1]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="cp1252", errors="replace")
    c = state(root, "pair_review_44", 1)
    assert c.state == "INVALID"


def test_same_reviewer_id_in_both_files_is_flagged(root):
    for n in (1, 2):
        df = filled(root, "pair_review_44", n)
        df["reviewer_id"] = "R"                                   # neutral id used by both people -> warning
        save(root, "pair_review_44", n, df)
    assert any("two different people" in w for w in state(root, "pair_review_44", 2).warnings)
    assert any("identical" in w for w in state(root, "pair_review_44", 2).warnings)
    df = filled(root, "pair_review_44", 2)
    df["reviewer_id"] = "reviewer_1"                              # names the other person -> rejected
    save(root, "pair_review_44", 2, df)
    assert any("names the other reviewer" in p for p in state(root, "pair_review_44", 2).problems)


# ── accept and receipts ───────────────────────────────────────────────────────

def test_accept_stores_records_receipt_and_never_overwrites(root, tmp_path):
    ret = tmp_path / "returned.csv"
    filled(root, "cluster_task_a", 1).to_csv(ret, index=False, encoding="utf-8-sig")
    ck = hh.accept(root, "cluster_task_a", 1, ret)
    assert ck.state == "VALID" and (root / hh.TRACKS["cluster_task_a"].completed[1]).exists()
    rec = [json.loads(line) for line in (root / hh.RECEIPTS).read_text(encoding="utf-8").splitlines()]
    assert rec[0]["sha256"] == ck.sha256 and rec[0]["track"] == "cluster_task_a"
    again = hh.accept(root, "cluster_task_a", 1, ret)
    assert again.state == "INVALID" and "never replaced" in again.problems[0]
    # modifying an accepted file afterwards is detected
    p = root / hh.TRACKS["cluster_task_a"].completed[1]
    df = hh.read_csv(p)
    df.loc[0, "intruder_position"] = "2"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    assert any("changed after it was accepted" in x for x in state(root, "cluster_task_a", 1).problems)


def test_invalid_returned_file_is_not_stored(root, tmp_path):
    ret = tmp_path / "bad.csv"
    filled(root, "cluster_task_a", 2, cells={(0, "intruder_position"): ""}).to_csv(ret, index=False)
    assert hh.accept(root, "cluster_task_a", 2, ret).state == "INVALID"
    assert not (root / hh.TRACKS["cluster_task_a"].completed[2]).exists() and not (root / hh.RECEIPTS).exists()


# ── packets and leak audit ────────────────────────────────────────────────────

def test_stage1_packets_contain_only_allow_listed_files(root):
    res = hh.build_packets(root, 1)
    assert res["status"] == "BUILT" and res["leak_audit"] == "PASSED"
    for n in (1, 2):
        d = root / hh.PACKETS / "stage_1" / f"person_{n}"
        names = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        assert names == sorted([
            "00_REVIEWER_HANDOFF_GUIDE.md",
            f"stance_pilot/annotator_{n}.csv", "stance_pilot/annotation_guidelines_v1.1.md", "stance_pilot/README_annotators.md",
            f"pair_review_44/reviewer_{n}_template.csv", "pair_review_44/README_reviewers.md", "pair_review_44/review_guide.md",
            f"cluster_task_a/reviewer_{n}_task_a.csv", "cluster_task_a/cluster_review_guide.md"])
        assert not any("key" in x.lower() or "catalog" in x or "task_b" in x for x in names)
    assert not (root / hh.PACKETS / "stage_1" / "person_1" / "cluster_task_a" / "reviewer_2_task_a.csv").exists()
    with pytest.raises(FileExistsError):
        hh.build_packets(root, 1)
    assert hh.audit(root) == []


def test_stage2_requires_accepted_task_a(root, tmp_path):
    assert hh.build_packets(root, 2, 1)["status"] == "BLOCKED"
    ret = tmp_path / "a1.csv"
    filled(root, "cluster_task_a", 1).to_csv(ret, index=False, encoding="utf-8-sig")
    save(root, "cluster_task_a", 2, filled(root, "cluster_task_a", 2))          # present but never accepted
    hh.accept(root, "cluster_task_a", 1, ret)
    assert hh.build_packets(root, 2, 2)["status"] == "BLOCKED"
    res = hh.build_packets(root, 2, 1)
    assert res["status"] == "BUILT" and res["folders"] == ["data/reviewer_handoff/packets/stage_2/person_1"]
    d = root / hh.PACKETS / "stage_2" / "person_1" / "cluster_task_b"
    assert sorted(p.name for p in d.iterdir()) == ["cluster_review_guide.md", "reviewer_1_task_b.csv"]


def test_stage3_requires_all_concern_files_and_pilot_analysis(root):
    res = hh.build_packets(root, 3)
    assert res["status"] == "BLOCKED" and any("pilot agreement analysis" in u for u in res["unmet"])
    for t in ("stance_pilot", "pair_review_44", "cluster_task_a", "cluster_task_b"):
        for n in (1, 2):
            save(root, t, n, filled(root, t, n))
    res = hh.build_packets(root, 3)
    assert res["status"] == "BLOCKED" and len(res["unmet"]) == 1
    (root / hh.PILOT_ANALYSIS).mkdir(parents=True)
    assert hh.build_packets(root, 3)["status"] == "BUILT"


def test_audit_catches_leaks(root):
    hh.build_packets(root, 1)
    d = root / hh.PACKETS / "stage_1" / "person_1"
    shutil.copyfile(root / hh.CL / "_key_do_not_share" / "task_a_key.csv", d / "task_a_key.csv")
    (d / "notes.md").write_text("selected as likely_similar", encoding="utf-8")
    probs = " ".join(hh.audit(root))
    assert "forbidden file task_a_key.csv" in probs and "likely_similar" in probs
    t = root / hh.TRACKS["cluster_task_a"].templates[1]
    df = hh.read_csv(t)
    df.loc[0, "intruder_position"] = "3"
    df.to_csv(t, index=False, encoding="utf-8-sig")
    assert any("pre-filled" in p for p in hh.audit(root))


def test_handoff_never_reads_answer_keys(root):
    shutil.rmtree(root / hh.CL / "_key_do_not_share")                          # works without any key present
    assert len(hh.status(root)) == 10 and hh.build_packets(root, 1)["status"] == "BUILT"
    src = (ROOT / "scripts" / "human_review_handoff.py").read_text(encoding="utf-8")
    assert src.count("_key_do_not_share") == 1                                  # only in the forbidden-name list
    readers = [ln for ln in src.splitlines() if ("read_csv(" in ln or "open(" in ln or "read_text(" in ln)]
    assert not any("catalog" in ln or "key" in ln.lower() for ln in readers)


# ── analysis paths: completed-file requirements and missing reviewers ─────────

def test_real_templates_pass_audit_and_guide_is_key_free():
    packets = hh.PACKETS.as_posix()
    assert [p for p in hh.audit(ROOT) if not p.startswith(packets)] == []   # packet state is reported by run_research_evaluation
    guide = (ROOT / hh.GUIDE).read_text(encoding="utf-8")
    assert not any(t in guide for t in hh.FORBIDDEN_TOKENS)
    for phrase in ("Never leave a judgment cell blank", "`U`", "Unclear", "Format quoted field as text", "_key_do_not_share"):
        assert phrase in guide


def test_cluster_analysis_blocked_with_one_reviewer_and_never_opens_key(root):
    from models.concern_discovery import review as rv
    shutil.copyfile(ROOT / hh.CL / "package_manifest.json", root / hh.CL / "package_manifest.json")
    for t in ("cluster_task_a", "cluster_task_b"):
        save(root, t, 1, filled(root, t, 1))
    shutil.rmtree(root / hh.CL / "_key_do_not_share")
    res = rv.analyze_reviews(root / hh.CL, root / "out")
    assert res["status"] == "BLOCKED" and sorted(res["missing"]) == ["reviewer_2_task_a.csv", "reviewer_2_task_b.csv"]
    assert not (root / "out").exists()


def test_cluster_analysis_rejects_edited_text_but_tolerates_crlf():
    from models.concern_discovery import review as rv
    tmpl = hh.read_csv(ROOT / hh.TRACKS["cluster_task_a"].templates[1])
    comp = tmpl.copy()
    comp["reviewer_id"], comp["intruder_position"], comp["coherence_of_others"], comp["confidence"] = "r1", "U", "U", "low"
    valid = {"intruder_position": rv.VALID_A_INTRUDER, "coherence_of_others": rv.VALID_COHERENCE, "confidence": rv.VALID_CONFIDENCE}
    crlf = comp.map(lambda v: v.replace("\n", "\r\n"))
    assert rv.validate_completed(crlf, tmpl, rv.A_FILL, valid, "x") == []
    comp.loc[0, "comment_1"] = "#NAME?"
    assert any("modified" in p for p in rv.validate_completed(comp, tmpl, rv.A_FILL, valid, "x"))


def test_pair44_analysis_rejects_edited_text(tmp_path):
    import analyze_concern_review_agreement as acr
    p44 = ROOT / hh.P44
    cat = pd.read_csv(p44 / "pair_catalog.csv", dtype=str, keep_default_na=False)
    (tmp_path / "completed").mkdir()
    for n in (1, 2):
        df = hh.read_csv(p44 / f"reviewer_{n}_template.csv")
        df["reviewer_id"], df["concern_relation"], df["confidence"] = f"reviewer_{n}", "U", "low"
        if n == 2:
            df.loc[0, "b_original_text"] = "#NAME?"
        df.to_csv(tmp_path / "completed" / f"reviewer_{n}_completed.csv", index=False, encoding="utf-8-sig")
    res = acr.run_analysis(completed_dir=tmp_path / "completed", catalog_path=p44 / "pair_catalog.csv", output_base=tmp_path / "out")
    assert res["status"] == "validation_failed" and "modified" in res["errors"][0]
    assert not (tmp_path / "out").exists() and len(cat) == 44


def test_missing_reviewer_handling_of_every_analysis():
    import analyze_annotation_agreement as aaa
    import analyze_concern_review_agreement as acr
    for rnd in ("pilot", "expansion_v2"):
        assert aaa.analyze(rnd)["status"] == "missing_files"
    if not (ROOT / hh.P44 / "completed").exists():
        assert acr.run_analysis()["status"] == "no_completed_files"


# ── regressions found in the operational audit ────────────────────────────────

def test_other_reviewers_file_is_rejected_with_hint(root, tmp_path):
    ret = tmp_path / "a2.csv"
    filled(root, "cluster_task_a", 2).to_csv(ret, index=False, encoding="utf-8-sig")
    ck = hh.accept(root, "cluster_task_a", 1, ret)
    assert ck.state == "INVALID" and any("person 2's file" in p for p in ck.problems)
    assert not (root / hh.TRACKS["cluster_task_a"].completed[1]).exists()
    resorted = tmp_path / "sorted.csv"
    filled(root, "cluster_task_a", 1).sort_values("item_id").to_csv(resorted, index=False, encoding="utf-8-sig")
    assert any("must not be sorted" in p for p in hh.accept(root, "cluster_task_a", 1, resorted).problems)


def test_same_submission_cannot_count_for_both_people(root, tmp_path):
    ret = tmp_path / "s.csv"
    df = filled(root, "pair_review_44", 1)
    df["reviewer_id"] = "R"
    df.to_csv(ret, index=False, encoding="utf-8-sig")
    assert hh.accept(root, "pair_review_44", 1, ret).state == "VALID"
    ck = hh.accept(root, "pair_review_44", 2, ret)
    assert ck.state == "INVALID" and "byte-identical" in ck.problems[0]
    assert not (root / hh.TRACKS["pair_review_44"].completed[2]).exists()


def test_deleted_accepted_file_is_flagged_and_cannot_be_replaced(root, tmp_path):
    ret = tmp_path / "a1.csv"
    filled(root, "cluster_task_a", 1).to_csv(ret, index=False, encoding="utf-8-sig")
    hh.accept(root, "cluster_task_a", 1, ret)
    (root / hh.TRACKS["cluster_task_a"].completed[1]).unlink()
    c = state(root, "cluster_task_a", 1)
    assert c.state == "INVALID" and "accepted file is missing" in c.problems[0]
    other = tmp_path / "a1b.csv"
    filled(root, "cluster_task_a", 1, cells={(0, "confidence"): "high"}).to_csv(other, index=False, encoding="utf-8-sig")
    ck = hh.accept(root, "cluster_task_a", 1, other)
    assert ck.state == "INVALID" and "already accepted" in ck.problems[0]
    assert hh.build_packets(root, 2, 1)["status"] == "BLOCKED"


def test_file_saved_directly_in_completed_can_be_accepted_in_place(root):
    p = save(root, "cluster_task_a", 2, filled(root, "cluster_task_a", 2))
    c = state(root, "cluster_task_a", 2)
    assert c.state == "VALID" and any("not accepted" in w for w in c.warnings)
    assert hh.build_packets(root, 2, 2)["status"] == "BLOCKED"
    ck = hh.accept(root, "cluster_task_a", 2, p)
    assert ck.state == "VALID" and hh.sha256(p) == ck.sha256
    assert hh.build_packets(root, 2, 2)["status"] == "BUILT"
    assert hh.accept(root, "cluster_task_a", 2, p).state == "INVALID"          # second receipt refused


def test_cli_entry_points(root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hh, "ROOT", root)
    save(root, "cluster_task_a", 1, filled(root, "cluster_task_a", 1, cells={(0, "intruder_position"): "9"}))
    assert hh.main(["status"]) == 1 and "INVALID" in capsys.readouterr().out          # proves the sandbox root is used
    (root / hh.TRACKS["cluster_task_a"].completed[1]).unlink()
    assert hh.main(["status"]) == 0 and "BLOCKED" in capsys.readouterr().out
    assert hh.main(["audit"]) == 0
    assert hh.main(["build-packets", "--stage", "2", "--person", "1"]) == 2
    bad = tmp_path / "bad.csv"
    hh.read_csv(root / hh.TRACKS["cluster_task_b"].templates[1]).to_csv(bad, index=False)
    assert hh.main(["accept", "--track", "cluster_task_b", "--person", "1", "--file", str(bad)]) == 1
    assert hh.main(["build-packets", "--stage", "1"]) == 0 and hh.main(["build-packets", "--stage", "1"]) == 1


# ── packet integrity against the build manifest ───────────────────────────────

def _packet_problems(root):
    return [p for p in hh.audit(root) if p.startswith(hh.PACKETS.as_posix())]


def test_built_packets_match_manifest(root):
    hh.build_packets(root, 1)
    assert _packet_problems(root) == []


def test_extra_prefilled_file_in_packet_is_flagged(root):
    hh.build_packets(root, 1)
    d = root / hh.PACKETS / "stage_1" / "person_1" / "stance_pilot"
    draft = hh.read_csv(d / "annotator_1.csv")
    draft["stance"] = "Neutral"                                    # fixture: any pre-filled judgment
    draft.to_csv(d / "draft.csv", index=False)
    probs = " ".join(_packet_problems(root))
    assert "not in the packet manifest" in probs and "pre-filled judgment columns ['stance']" in probs
    assert (d / "draft.csv").exists()                              # audit never removes files


def test_modified_or_missing_packet_files_are_flagged(root):
    hh.build_packets(root, 1)
    d = root / hh.PACKETS / "stage_1" / "person_2"
    (d / "00_REVIEWER_HANDOFF_GUIDE.md").write_text("edited", encoding="utf-8")
    (d / "pair_review_44" / "review_guide.md").unlink()
    probs = " ".join(_packet_problems(root))
    assert "changed after the build: 00_REVIEWER_HANDOFF_GUIDE.md" in probs
    assert "missing: pair_review_44/review_guide.md" in probs


def test_packet_zip_is_verified_byte_for_byte(root):
    import zipfile
    hh.build_packets(root, 1)
    stage = root / hh.PACKETS / "stage_1"
    files = sorted(p for p in (stage / "person_1").rglob("*") if p.is_file())
    with zipfile.ZipFile(stage / "person_1.zip", "w") as z:
        for p in files:
            z.write(p, p.relative_to(stage).as_posix())
    assert _packet_problems(root) == []
    with zipfile.ZipFile(stage / "person_2.zip", "w") as z:
        for p in sorted(q for q in (stage / "person_2").rglob("*") if q.is_file()):
            data = p.read_bytes() if p.name != "review_guide.md" else b"altered"
            z.writestr(p.relative_to(stage).as_posix(), data)
        z.writestr("person_2/extra_notes.csv", "pilot_id,stance\npilot_001,Support\n")
    probs = " ".join(_packet_problems(root))
    assert "differs from the built file: person_2/pair_review_44/review_guide.md" in probs
    assert "not in the packet manifest (do not send): person_2/extra_notes.csv" in probs
    assert "pre-filled judgment columns ['stance'] in archive member" in probs
