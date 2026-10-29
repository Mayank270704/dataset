"""
tests/test_annotation_agreement_tool.py
=======================================
Tests for scripts/analyze_annotation_agreement.py (pilot and v2 expansion rounds).
All annotation data here is SYNTHETIC and lives in pytest temp directories; nothing is written to
the repository and no real annotation exists.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("agreement_tool", ROOT / "scripts" / "analyze_annotation_agreement.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

COLS = ["pilot_id", "source_dataset", "source_record_id", "source_file", "source_row_number", "source_field",
        "question_slug", "question_wording", "question_context", "context", "duplicate_text_group_id",
        "original_text", "stance", "concern", "requested_action", "evidence_span", "review_notes",
        "annotator_id", "source_language", "language_basis"]
ANN = tool.ANNOTATION_COLUMNS


def make_reference(n_pilot=3, n_exp=8, extra_cols=True) -> pd.DataFrame:
    rows = []
    ids = [f"pilot_{i:03d}" for i in range(1, n_pilot + 1)] + [f"expansion_v2_{i:03d}" for i in range(1, n_exp + 1)]
    for i, pid in enumerate(ids, start=1):
        row = {c: "" for c in COLS}
        row.update({
            "pilot_id": pid, "source_dataset": "synthetic", "source_record_id": str(i), "source_file": "f.csv",
            "source_row_number": str(i), "source_field": "q8", "question_slug": "q8", "question_wording": "Q",
            "question_context": "ctx", "context": "ctx",
            "original_text": f"Synthetic annotation text {i}: publish datasets openly and explain delays",
            "source_language": "English", "language_basis": "source column `language`",
        })
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)
    return df if extra_cols else df.drop(columns=["source_language", "language_basis"])


def annotate(ref, annotator_id, stances_by_prefix, **overrides):
    df = ref.copy()
    for prefix, stances in stances_by_prefix.items():
        idx = df.index[df.pilot_id.str.startswith(prefix)]
        assert len(idx) == len(stances), (prefix, len(idx), len(stances))
        df.loc[idx, "stance"] = stances
        df.loc[idx, "annotator_id"] = annotator_id
    for col, value in overrides.items():
        df[col] = value
    return df


EXP_A = ["Support", "Support", "Oppose", "Neutral", "Mixed", "Unclear", "Support", "Neutral"]
EXP_B = ["Support", "Oppose", "Oppose", "Neutral", "Mixed", "Neutral", "Support", "Neutral"]


@pytest.fixture
def env(tmp_path):
    ref = make_reference()
    ref_path = tmp_path / "reference.csv"
    ref.to_csv(ref_path, index=False, encoding="utf-8")
    template = tmp_path / "template.csv"
    ref.to_csv(template, index=False, encoding="utf-8")
    return tmp_path, ref, ref_path, template


def write(path, df):
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def run(env, df_a, df_b, round_name="expansion_v2"):
    tmp, ref, ref_path, template = env
    a = write(tmp / "a.csv", df_a)
    b = write(tmp / "b.csv", df_b)
    return tool.analyze(round_name, a, b, ref_path, templates=[template])


def good_pair(ref):
    return (annotate(ref, "annotator_A", {"expansion_v2_": EXP_A}),
            annotate(ref, "annotator_B", {"expansion_v2_": EXP_B}))


# ── missing / blank / template inputs ─────────────────────────────────────────

def test_missing_completed_files_block_with_instruction_and_no_template_fallback(env):
    tmp, ref, ref_path, template = env
    res = tool.analyze("expansion_v2", tmp / "no_a.csv", tmp / "no_b.csv", ref_path, templates=[template])
    assert res["status"] == "missing_files"
    assert "Finish annotation first" in res["message"] and "never falls back to blank templates" in res["message"]
    assert res["segments"] == {}


def test_blank_template_passed_as_completed_is_refused(env):
    tmp, ref, ref_path, template = env
    res = tool.analyze("expansion_v2", template, template, ref_path, templates=[template])
    assert res["status"] == "invalid"
    assert any("blank template" in e for e in res["errors"])


def test_blank_completed_files_report_pending_and_compute_nothing(env):
    tmp, ref, ref_path, template = env
    blank_a = ref.copy()
    blank_a["annotator_id"] = "annotator_A"
    blank_b = ref.copy()
    blank_b["annotator_id"] = "annotator_B"
    res = run(env, blank_a, blank_b)
    assert res["status"] == "pending_no_annotations"
    for m in res["segments"].values():
        assert m["stance_agreement"]["status"] == "insufficient_data"
        assert m["sample_sizes"]["paired_valid"] == 0
    with pytest.raises(ValueError):
        tool.write_outputs(res, tmp / "out_pending")
    assert not (tmp / "out_pending").exists()


# ── segments, sample sizes, kappa ─────────────────────────────────────────────

def test_expansion_and_pilot_segments_are_separate_and_never_pooled(env):
    tmp, ref, *_ = env
    a, b = good_pair(ref)
    res = run(env, a, b)
    assert res["status"] == "ok"
    assert set(res["segments"]) == {"expansion", "pilot_segment"}
    exp, pil = res["segments"]["expansion"], res["segments"]["pilot_segment"]
    assert exp["sample_sizes"]["rows_in_segment"] == 8 and exp["sample_sizes"]["paired_valid"] == 8
    assert pil["sample_sizes"]["rows_in_segment"] == 3 and pil["sample_sizes"]["paired_valid"] == 0
    assert pil["required"] is False and exp["required"] is True
    assert "all" not in res["segments"] and "combined" not in res["segments"]


def test_kappa_matches_formula_and_reports_sample_size(env):
    a, b = good_pair(env[1])
    res = run(env, a, b)
    k = res["segments"]["expansion"]["stance_agreement"]
    n = 8
    agree = sum(x == y for x, y in zip(EXP_A, EXP_B))
    po = agree / n
    cats = tool.PERMITTED_STANCES
    pe = sum((EXP_A.count(c) / n) * (EXP_B.count(c) / n) for c in cats)
    assert k["status"] == "computed" and k["n"] == n
    assert np.isclose(k["po"], po) and np.isclose(k["kappa"], (po - pe) / (1 - pe))
    cm = res["segments"]["expansion"]["confusion_matrix_rows_A_cols_B"]
    assert cm["Support"]["Support"] == 2 and cm["Support"]["Oppose"] == 1 and sum(sum(r.values()) for r in cm.values()) == n


def test_kappa_guards_for_small_n_and_zero_variance():
    assert tool.cohen_kappa(["Support"] * 3, ["Support"] * 3)["status"] == "insufficient_data"
    zv = tool.cohen_kappa(["Support"] * 10, ["Support"] * 10)
    assert zv["status"] == "zero_variance" and zv["kappa"] is None
    y1 = ["Support"] * 12 + ["Oppose"] * 6 + ["Neutral"] * 2
    y2 = ["Support"] * 10 + ["Oppose"] * 2 + ["Oppose"] * 4 + ["Support"] * 2 + ["Neutral"] * 2
    k = tool.cohen_kappa(y1, y2)
    assert np.isclose(k["po"], 0.8) and 0.60 <= k["kappa"] <= 0.80 and k["target_met"] is True


def test_invalid_and_blank_labels_reported_and_excluded_not_blocking(env):
    ref = env[1]
    a = annotate(ref, "annotator_A", {"expansion_v2_": ["Support", "Opposed", "Oppose", "Neutral", "", "Unclear", "Support", "Neutral"]})
    b = annotate(ref, "annotator_B", {"expansion_v2_": EXP_B})
    res = run(env, a, b)
    assert res["status"] == "ok" and not res["errors"]
    s = res["segments"]["expansion"]
    assert s["annotator_A_labels"]["invalid"] == 1 and s["annotator_A_labels"]["invalid_values"] == ["Opposed"]
    assert s["annotator_A_labels"]["blank"] == 1
    assert s["sample_sizes"]["paired_valid"] == 6 and s["sample_sizes"]["excluded_total"] == 2
    assert s["sample_sizes"]["excluded_a_invalid"] == 1 and s["sample_sizes"]["excluded_a_blank"] == 1


# ── validation ────────────────────────────────────────────────────────────────

def test_annotator_ids_must_be_correct_and_distinct(env):
    ref = env[1]
    a = annotate(ref, "annotator_A", {"expansion_v2_": EXP_A})
    b_same = annotate(ref, "annotator_A", {"expansion_v2_": EXP_B})
    res = run(env, a, b_same)
    assert res["status"] == "invalid"
    assert any("annotator_B" in e for e in res["errors"]) and any("must be distinct" in e for e in res["errors"])


def test_blank_annotator_id_on_required_rows_blocks_but_optional_rows_may_be_blank(env):
    ref = env[1]
    a = annotate(ref, "annotator_A", {"expansion_v2_": EXP_A})
    b = annotate(ref, "annotator_B", {"expansion_v2_": EXP_B})
    ok = run(env, a, b)                       # pilot_ rows are blank and carry no id: allowed (optional segment)
    assert ok["status"] == "ok"
    b.loc[b.pilot_id == "expansion_v2_003", "annotator_id"] = ""
    bad = run(env, a, b)
    assert bad["status"] == "invalid" and any("annotator_id blank" in e for e in bad["errors"])


def test_optional_pilot_segment_rows_with_labels_need_id_and_are_reported_separately(env):
    ref = env[1]
    a = annotate(ref, "annotator_A", {"expansion_v2_": EXP_A, "pilot_": ["Support", "Neutral", "Oppose"]})
    b = annotate(ref, "annotator_B", {"expansion_v2_": EXP_B, "pilot_": ["Support", "Neutral", "Mixed"]})
    res = run(env, a, b)
    assert res["status"] == "ok"
    assert res["segments"]["pilot_segment"]["sample_sizes"]["paired_valid"] == 3
    assert res["segments"]["expansion"]["sample_sizes"]["paired_valid"] == 8      # not 11
    a.loc[a.pilot_id == "pilot_001", "annotator_id"] = ""
    assert run(env, a, b)["status"] == "invalid"


def test_evidence_span_must_be_verbatim_and_within_200_chars(env):
    ref = env[1]
    a, b = good_pair(ref)
    a.loc[a.pilot_id == "expansion_v2_001", "evidence_span"] = "publish datasets openly"        # verbatim
    assert run(env, a, b)["status"] == "ok"
    a.loc[a.pilot_id == "expansion_v2_002", "evidence_span"] = "publish datasets in the open"   # paraphrase
    res = run(env, a, b)
    assert res["status"] == "invalid"
    assert any("expansion_v2_002" in e and "verbatim" in e for e in res["errors"])


def test_evidence_span_over_200_characters_blocks(tmp_path):
    ref = make_reference()
    ref.loc[ref.pilot_id == "expansion_v2_001", "original_text"] = "x" * 250
    ref_path = write(tmp_path / "ref.csv", ref)
    a, b = good_pair(ref)
    a.loc[a.pilot_id == "expansion_v2_001", "evidence_span"] = "x" * 201
    pa, pb = write(tmp_path / "a.csv", a), write(tmp_path / "b.csv", b)
    res = tool.analyze("expansion_v2", pa, pb, ref_path, templates=[])
    assert res["status"] == "invalid" and any("201 characters" in e for e in res["errors"])


def test_word_limits_are_warnings_only(env):
    ref = env[1]
    a, b = good_pair(ref)
    a.loc[a.pilot_id == "expansion_v2_001", "concern"] = " ".join(["w"] * 15)
    a.loc[a.pilot_id == "expansion_v2_002", "concern"] = " ".join(["w"] * 16)
    a.loc[a.pilot_id == "expansion_v2_003", "requested_action"] = " ".join(["w"] * 21)
    res = run(env, a, b)
    assert res["status"] == "ok" and not res["errors"]
    text = " ".join(res["warnings"])
    assert "expansion_v2_002 concern has 16 words" in text and "expansion_v2_003 requested_action has 21 words" in text
    assert "expansion_v2_001" not in text


def test_schema_order_ids_and_provenance_columns_are_enforced(env):
    ref = env[1]
    a, b = good_pair(ref)
    res = run(env, a.drop(columns=["context"]), b)
    assert res["status"] == "invalid" and any("columns differ" in e for e in res["errors"])
    reordered = b[["stance"] + [c for c in b.columns if c != "stance"]]
    assert run(env, a, reordered)["status"] == "invalid"
    shuffled = a.iloc[::-1].reset_index(drop=True)
    assert any("row IDs/order" in e for e in run(env, shuffled, b)["errors"])
    edited = a.copy()
    edited.loc[0, "original_text"] += " (edited)"
    res = run(env, edited, b)
    assert res["status"] == "invalid" and any("source/provenance columns modified" in e for e in res["errors"])


def test_pilot_round_uses_one_required_segment(tmp_path):
    ref = make_reference(n_pilot=6, n_exp=0)
    ref18 = ref.drop(columns=["source_language", "language_basis"])
    ref_path = write(tmp_path / "pilot_ref.csv", ref18)
    a = annotate(ref18, "annotator_A", {"pilot_": ["Support", "Oppose", "Neutral", "Mixed", "Unclear", "Support"]})
    b = annotate(ref18, "annotator_B", {"pilot_": ["Support", "Oppose", "Neutral", "Neutral", "Unclear", "Oppose"]})
    pa, pb = write(tmp_path / "pa.csv", a), write(tmp_path / "pb.csv", b)
    res = tool.analyze("pilot", pa, pb, ref_path, templates=[])
    assert res["status"] == "ok" and list(res["segments"]) == ["pilot"]
    assert res["segments"]["pilot"]["sample_sizes"]["paired_valid"] == 6
    assert "PILOT" in res["label"] and "EXPANSION" not in res["label"]


# ── outputs ───────────────────────────────────────────────────────────────────

def test_outputs_written_to_new_directory_and_never_overwritten(env):
    tmp, ref, *_ = env
    a, b = good_pair(ref)
    res = run(env, a, b)
    out = tmp / "agreement_out"
    paths = tool.write_outputs(res, out)
    assert {p.name for p in paths} == {"agreement_report.md", "agreement_metrics.json", "disagreements.csv", "validation_issues.csv"}
    with pytest.raises(FileExistsError):
        tool.write_outputs(res, out)
    dis = pd.read_csv(out / "disagreements.csv", dtype=str, keep_default_na=False)
    assert (dis["adjudicated_stance"] == "").all() and (dis["adjudication_rationale"] == "").all()
    assert set(dis["segment"]) <= {"expansion", "pilot_segment"}
    assert len(dis[dis.segment == "expansion"]) == sum(x != y for x, y in zip(EXP_A, EXP_B))
    report = (out / "agreement_report.md").read_text(encoding="utf-8")
    assert "stratified sample" in report and "never pooled" in report and "not population prevalence" in report
    metrics = json.loads((out / "agreement_metrics.json").read_text(encoding="utf-8"))
    assert "_disagreements" not in json.dumps(metrics)


def test_cli_exit_codes_and_check_only(env, capsys):
    tmp, ref, ref_path, template = env
    a, b = good_pair(ref)
    pa, pb = write(tmp / "ca.csv", a), write(tmp / "cb.csv", b)
    base = ["--round", "expansion_v2", "--reference", str(ref_path)]
    assert tool.main(base + ["--completed-a", str(tmp / "none.csv"), "--completed-b", str(pb)]) == 2
    assert tool.main(base + ["--completed-a", str(pa), "--completed-b", str(pb), "--check-only"]) == 0
    assert not (tmp / "agreement_analysis").exists()
    out = tmp / "cli_out"
    assert tool.main(base + ["--completed-a", str(pa), "--completed-b", str(pb), "--output-dir", str(out)]) == 0
    assert (out / "agreement_report.md").exists()
    assert tool.main(base + ["--completed-a", str(pa), "--completed-b", str(pb), "--output-dir", str(out)]) == 1


def test_default_completed_paths_follow_the_documented_names():
    assert tool.ROUNDS["pilot"]["completed_a"].name == "pilot_annotator_A_completed.csv"
    assert tool.ROUNDS["pilot"]["completed_b"].name == "pilot_annotator_B_completed.csv"
    assert tool.ROUNDS["expansion_v2"]["completed_a"].name == "expansion_v2_annotator_A_completed.csv"
    assert tool.ROUNDS["expansion_v2"]["completed_b"].name == "expansion_v2_annotator_B_completed.csv"
    assert all(r["completed_a"].parent.name == "completed" for r in tool.ROUNDS.values())
    assert not any(r["completed_a"] in r["templates"] for r in tool.ROUNDS.values())


# ── parity with the existing pilot notebook ───────────────────────────────────

def test_pilot_results_match_the_existing_notebook(tmp_path):
    """The notebook (unchanged) and the reusable tool must agree on the same synthetic pilot files."""
    nb_path = ROOT / "notebooks" / "03_annotation_agreement_analysis.ipynb"
    ref = make_reference(n_pilot=6, n_exp=0).drop(columns=["source_language", "language_basis"])
    proj = tmp_path / "proj"
    (proj / "notebooks").mkdir(parents=True)
    ann = proj / "data" / "annotation"
    (ann / "completed").mkdir(parents=True)
    write(ann / "pilot_annotation_50.csv", ref)
    sa = ["Support", "Oppose", "Neutral", "Mixed", "Unclear", "Support"]
    sb = ["Support", "Oppose", "Neutral", "Neutral", "Unclear", "Oppose"]
    a, b = annotate(ref, "annotator_A", {"pilot_": sa}), annotate(ref, "annotator_B", {"pilot_": sb})
    write(ann / "completed" / "pilot_annotator_A_completed.csv", a)
    write(ann / "completed" / "pilot_annotator_B_completed.csv", b)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "nb"}
    prev = os.getcwd()
    os.chdir(proj / "notebooks")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    exec(compile("".join(c["source"]), "<nb>", "exec"), ns)
    finally:
        os.chdir(prev)

    res = tool.analyze("pilot", ann / "completed" / "pilot_annotator_A_completed.csv",
                       ann / "completed" / "pilot_annotator_B_completed.csv",
                       ann / "pilot_annotation_50.csv", templates=[])
    k = res["segments"]["pilot"]["stance_agreement"]
    assert k["status"] == ns["res"]["status"] == "computed"
    assert np.isclose(k["kappa"], ns["res"]["kappa"]) and np.isclose(k["po"], ns["res"]["po"])
    assert k["n"] == ns["res"]["n"] == len(ns["included_pairs"])
