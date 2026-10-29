"""
tests/test_stance_ai_working_labels.py
======================================
Separate AI-working-label input mode of scripts/train_stance_baselines.py, and proof that the human-annotation path,
thresholds and evaluation rules are unchanged. Synthetic drafts are written to temporary folders only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402

PRIMARY = ROOT / "data" / "reviewer_handoff" / "ai_assisted_drafts_review_only" / "person_1" / "policylens_annotator_1_AI_assisted_draft.csv"
REF = ROOT / "data" / "annotation" / "pilot_annotation_50.csv"


def draft(tmp_path: Path, labels: list[str] | None = None, edit=None) -> Path:
    d = pd.read_csv(REF, dtype=str, keep_default_na=False)
    d["stance"] = labels if labels is not None else (["Support", "Neutral", "Unclear", "Support", "Neutral"] * 10)
    if edit:
        d = edit(d)
    p = tmp_path / "draft.csv"
    d.to_csv(p, index=False)
    return p


def test_thresholds_and_rules_unchanged():
    c = sp.DEFAULT_CONFIG
    assert (c["min_gold"], c["min_class_count"], c["min_test_records"], c["test_size"]) == (40, 5, 10, 0.25)
    assert lb.SCHEMA_LABELS == ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]


@pytest.mark.skipif(not PRIMARY.exists(), reason="primary AI draft not present")
def test_primary_draft_loads_unchanged():
    st, d = lb.load_ai_working_labels(PRIMARY)
    raw = pd.read_csv(PRIMARY, dtype=str, keep_default_na=False)
    assert st.state == "ok" and len(d) == 50
    assert d["gold_stance"].tolist() == raw["stance"].str.strip().tolist()          # labels preserved, no mapping
    assert d["pilot_id"].tolist() == raw["pilot_id"].tolist()
    assert set(d["label_source"]) == {lb.AI_LABEL_SOURCE}


def test_invalid_drafts_are_rejected(tmp_path):
    (tmp_path / "a").mkdir()
    bad_label = draft(tmp_path / "a", ["Positive"] + ["Support"] * 49)          # sentiment word, not a stance label
    assert lb.load_ai_working_labels(bad_label)[0].state == "invalid"
    (tmp_path / "b").mkdir()
    edited = draft(tmp_path / "b", edit=lambda d: d.assign(original_text=d["original_text"].where(d.index != 0, "changed")))
    assert "text differ" in lb.load_ai_working_labels(edited)[0].detail
    (tmp_path / "c").mkdir()
    reordered = draft(tmp_path / "c", edit=lambda d: d.iloc[::-1])
    assert lb.load_ai_working_labels(reordered)[0].state == "invalid"
    assert lb.load_ai_working_labels(tmp_path / "missing.csv")[0].state == "missing"


def test_human_annotation_folders_are_refused():
    for d in lb.PROTECTED_LABEL_DIRS:
        st, out = lb.load_ai_working_labels(d / "anything.csv")
        assert st.state == "invalid" and out is None and "human annotation folders" in st.detail


def test_language_inability_and_blank_rows_excluded(tmp_path):
    def edit(d):
        d.loc[0, "review_notes"] = "non-English: French"
        d.loc[1, "stance"] = ""
        return d
    st, d = lb.load_ai_working_labels(draft(tmp_path, edit=edit))
    r = d.set_index("pilot_id")["excluded_reason"]
    assert r["pilot_001"] == "language_inability" and r["pilot_002"] == "missing_annotation"
    assert (d["gold_stance"] != "").sum() == 48


def test_insufficient_ai_labels_block_without_writing(tmp_path, capsys):
    import train_stance_baselines as tsb
    p = draft(tmp_path, ["Support"] * 25 + [""] * 25)                    # 25 usable, one class
    assert tsb.main(["--ai-working-labels", str(p), "--output-root", str(tmp_path / "runs")]) == 3
    assert not (tmp_path / "runs").exists() and "[BLOCKED]" in capsys.readouterr().out


def test_ai_run_is_separate_and_labelled(tmp_path, monkeypatch):
    import train_stance_baselines as tsb
    monkeypatch.setattr(tsb, "AI_OUTPUT_ROOT", tmp_path / "ai_runs")
    p = draft(tmp_path)
    assert tsb.main(["--ai-working-labels", str(p), "--check-only"]) == 0
    assert not (tmp_path / "ai_runs").exists()
    assert tsb.main(["--ai-working-labels", str(p), "--tag", "t"]) == 0            # default root redirected to AI root
    run = tmp_path / "ai_runs" / "run_t"
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert meta["human_labels_only"] is False and meta["label_type"] == "ai_generated_unverified"
    assert meta["merged_or_voted"] is False and meta["label_sources"] == {lb.AI_LABEL_SOURCE: 50}
    rep = (run / "report.md").read_text(encoding="utf-8")
    assert "UNVERIFIED" in rep and "not human-verified stance accuracy" in rep
    assert not (ROOT / "data" / "stance_training" / "run_t").exists()
    sa = pd.read_csv(run / "split_assignments.csv")
    assert set(sa["label_source"]) == {lb.AI_LABEL_SOURCE} and sa["pilot_id"].is_unique


def test_human_path_unchanged_on_real_project():
    import run_research_evaluation as ev
    import train_stance_baselines as tsb
    if (ROOT / "data" / "annotation" / "completed").exists():
        pytest.skip("completed human annotation exists")
    assert tsb.main(["--check-only"]) == 3                                     # human mode still blocked
    item = {r["name"]: r for r in ev.check_supervised_blockers()}["supervised stance baselines (TF-IDF + LR/SVM)"]
    assert item["status"] == "BLOCKED"                                       # AI runs never count as human baselines
