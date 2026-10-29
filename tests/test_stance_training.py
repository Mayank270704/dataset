"""
tests/test_stance_training.py
=============================
Supervised stance pipeline: missing labels, invalid labels, gold rules, duplicate leakage, reproducible splits,
evaluation and serialization. All labelled data here are SYNTHETIC FIXTURES written to temporary folders; they are
never placed in data/annotation and are not human labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402

COLS = list(pd.read_csv(ROOT / "data" / "annotation" / "pilot_annotation_50.csv", nrows=0).columns)
CUE = {"Support": "strongly support publish welcome", "Oppose": "strongly oppose waste reject",
       "Neutral": "asks question about timeline", "Mixed": "support goal but oppose cost"}
FILLER = [f"w{i:03d}" for i in range(300)]


def synth_reference(n: int = 160, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic pool rows (fixture) + the stance each fixture text was written to express."""
    rng = np.random.default_rng(seed)
    classes = list(CUE)
    rows, truth = [], []
    for i in range(n):
        c = classes[i % len(classes)]
        text = f"{CUE[c]} " + " ".join(rng.choice(FILLER, 8, replace=False))
        rows.append({c_: "" for c_ in COLS} | {"pilot_id": f"pilot_{i:03d}", "source_dataset": "get_involved_questionnaire_2017_18",
                     "source_record_id": str(1000 + i), "source_file": "fixture.csv", "source_row_number": str(i),
                     "source_field": "q8", "question_slug": "q8", "original_text": text})
        truth.append(c)
    ref = pd.DataFrame(rows, columns=COLS)
    ref.loc[5, "original_text"] = ref.loc[1, "original_text"]           # exact duplicate text, other record
    truth[5] = truth[1]
    ref.loc[9, "source_record_id"] = ref.loc[2, "source_record_id"]      # same submission, second entry
    ref.loc[13, "original_text"] = ref.loc[3, "original_text"] + " ok"   # near-duplicate
    truth[13] = truth[3]
    return ref, truth


def write_round(tmp: Path, ref: pd.DataFrame, a_labels: list[str], b_labels: list[str], notes_b: dict[int, str] | None = None) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    ref.to_csv(tmp / "ref.csv", index=False)
    for who, labs, notes in (("A", a_labels, {}), ("B", b_labels, notes_b or {})):
        d = ref.copy()
        d["stance"] = labs
        d["annotator_id"] = f"annotator_{who}"
        for i, n in notes.items():
            d.loc[i, "review_notes"] = n
        d.to_csv(tmp / f"{who}.csv", index=False)
    return {"pilot": {"reference": tmp / "ref.csv", "completed_a": tmp / "A.csv", "completed_b": tmp / "B.csv",
                      "segments": [{"name": "pilot", "prefix": "", "required": True}]}}


def gold_for(tmp, a, b, ref=None, notes_b=None, adj: pd.DataFrame | None = None):
    ref = ref if ref is not None else synth_reference()[0]
    cfg = write_round(tmp, ref, a, b, notes_b)
    adir = tmp / "adj"
    adir.mkdir(exist_ok=True)
    if adj is not None:
        adj.to_csv(adir / "pilot_adjudicated.csv", index=False)
    return lb.build_gold(("pilot",), cfg, adir), cfg, adir


# ── missing / invalid labels ──────────────────────────────────────────────────

def test_real_project_is_blocked_and_cli_writes_nothing(tmp_path, capsys):
    import train_stance_baselines as tsb
    if (ROOT / "data" / "annotation" / "completed").exists():
        pytest.skip("completed annotation files exist in this workspace")
    assert lb.readiness()["status"] == "BLOCKED"
    assert tsb.main(["--output-root", str(tmp_path / "runs")]) == 3
    assert "No genuine stance labels" in capsys.readouterr().out and not (tmp_path / "runs").exists()
    assert tsb.main(["--check-only"]) == 3


def test_invalid_labels_are_rejected(tmp_path):
    ref, truth = synth_reference(40)
    bad = list(truth)
    bad[0] = "Positive"                                                   # sentiment word, not a stance label
    (statuses, gold), cfg, adir = gold_for(tmp_path, bad, truth, ref)
    assert statuses[0].state == "invalid" and gold.empty
    assert lb.readiness(("pilot",), cfg, adir)["status"] == "INVALID"
    blank = [""] * 40
    (statuses, _), _, _ = gold_for(tmp_path / "b", blank, blank, ref)
    assert statuses[0].state == "invalid" and any("no stance labels" in e for e in statuses[0].errors)


def test_invalid_adjudication_is_rejected(tmp_path):
    ref, truth = synth_reference(40)
    b = list(truth)
    b[0] = "Neutral" if truth[0] != "Neutral" else "Support"
    for adj in (pd.DataFrame({"pilot_id": ["pilot_000"], "adjudicated_stance": ["Agree"], "adjudication_rationale": ["x"]}),
                pd.DataFrame({"pilot_id": ["pilot_000"], "adjudicated_stance": ["Support"], "adjudication_rationale": [""]}),
                pd.DataFrame({"pilot_id": ["nope"], "adjudicated_stance": ["Support"], "adjudication_rationale": ["x"]})):
        (statuses, _), _, _ = gold_for(tmp_path / str(len(list(tmp_path.iterdir()))), truth, b, ref, adj=adj)
        assert statuses[0].state == "invalid"


def test_gold_rules(tmp_path):
    ref, truth = synth_reference(40)
    a, b = list(truth), list(truth)
    b[0] = "Neutral" if truth[0] != "Neutral" else "Support"              # disagreement, unresolved
    b[1] = "Neutral" if truth[1] != "Neutral" else "Support"              # disagreement, adjudicated
    a[2] = b[2] = "Unclear"                                               # agreement but language inability
    a[3], b[3] = "", ""                                                    # not annotated
    adj = pd.DataFrame({"pilot_id": ["pilot_001"], "adjudicated_stance": [truth[1]], "adjudication_rationale": ["reviewed"]})
    (statuses, gold), _, _ = gold_for(tmp_path, a, b, ref, notes_b={2: "non-English: French"}, adj=adj)
    g = gold.set_index("pilot_id")
    assert g.loc["pilot_000", "excluded_reason"] == "unresolved_disagreement"
    assert (g.loc["pilot_001", "gold_stance"], g.loc["pilot_001", "label_source"]) == (truth[1], "adjudicated")
    assert g.loc["pilot_002", "excluded_reason"] == "language_inability"
    assert g.loc["pilot_003", "excluded_reason"] == "missing_annotation"
    assert g.loc["pilot_004", "label_source"] == "agreement"
    row = pd.Series({"round": "expansion_v2", "pilot_id": "pilot_007", "stance_A": "Support", "stance_B": "Support",
                     "adjudicated_stance": "", "notes_A": "", "notes_B": ""})
    assert lb.gold_label(row)[2] == "pilot_record_in_v2_file"


# ── leakage and splits ────────────────────────────────────────────────────────

def _prepared(tmp_path, seed=42, n=160):
    ref, truth = synth_reference(n)
    (_, gold), _, _ = gold_for(tmp_path, truth, truth, ref)
    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG)) | {"seed": seed, "bootstrap": 200}
    return sp.prepare(gold, cfg), cfg


def test_duplicates_and_same_record_stay_together(tmp_path):
    prep, _ = _prepared(tmp_path)
    d = prep["data"].set_index("pilot_id")
    for x, y in (("pilot_001", "pilot_005"), ("pilot_002", "pilot_009"), ("pilot_003", "pilot_013")):
        assert d.loc[x, "leakage_group"] == d.loc[y, "leakage_group"] and d.loc[x, "split"] == d.loc[y, "split"]
    tr, te = prep["data"][prep["data"]["split"] == "train"], prep["data"][prep["data"]["split"] == "test"]
    assert not set(tr["leakage_group"]) & set(te["leakage_group"])
    norm = lambda s: set(s.str.casefold().str.strip())  # noqa: E731
    assert not norm(tr["original_text"]) & norm(te["original_text"])


def test_split_is_reproducible(tmp_path):
    a, _ = _prepared(tmp_path / "a", seed=7)
    b, _ = _prepared(tmp_path / "b", seed=7)
    c, _ = _prepared(tmp_path / "c", seed=8)
    assert a["data"]["split"].tolist() == b["data"]["split"].tolist()
    assert a["data"]["split"].tolist() != c["data"]["split"].tolist()


def test_insufficient_labels_block_before_fitting(tmp_path):
    ref, truth = synth_reference(30)
    (_, gold), _, _ = gold_for(tmp_path, truth, truth, ref)
    with pytest.raises(sp.TrainingBlocked, match="insufficient"):
        sp.prepare(gold, sp.DEFAULT_CONFIG)
    ref, truth = synth_reference(80)
    truth = ["Support"] * 76 + ["Oppose"] * 4                              # Oppose below min_class_count -> one class left
    (_, gold), _, _ = gold_for(tmp_path / "r", truth, truth, ref)
    with pytest.raises(sp.TrainingBlocked):
        sp.prepare(gold, sp.DEFAULT_CONFIG)


def test_rare_class_and_excluded_labels_are_reported(tmp_path):
    ref, truth = synth_reference(160)
    truth = list(truth)
    for i in range(160):
        if truth[i] == "Mixed" and i > 10 and i != 13:
            truth[i] = "Neutral"                                           # leaves 3 Mixed (indices 3, 7, 11 ... < 20)
    (_, gold), _, _ = gold_for(tmp_path, truth, truth, ref)
    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG)) | {"exclude_labels": ["Unclear"]}
    prep = sp.prepare(gold, cfg)
    assert "Mixed" in prep["dropped"]["below_min_class_count"] and "Mixed" not in prep["labels"]


# ── training, evaluation, serialization ───────────────────────────────────────

def test_train_evaluate_and_tfidf_fit_on_train_only(tmp_path):
    prep, cfg = _prepared(tmp_path)
    res = sp.train_and_evaluate(prep, cfg)
    for m in sp.MODELS:
        t = res["models"][m]["test"]
        assert set(t["per_class"]) == set(prep["labels"])
        assert np.array(t["confusion_matrix"]["rows_true_cols_pred"]).shape == (len(prep["labels"]),) * 2
        assert t["macro_f1"] > res["majority_class_baseline"]["macro_f1"]
        lo, hi = t["macro_f1_ci95_group_bootstrap"]
        assert lo <= t["macro_f1"] <= hi or lo <= hi
        assert res["models"][m]["grouped_cv_on_train"]["status"] == "ok"
    d = res["class_distribution"]
    assert sum(d["train"].values()) == res["n_train"] and sum(d["test"].values()) == res["n_test"]
    vocab = res["pipelines"]["logreg"].named_steps["tfidf"].vocabulary_
    train_tokens = set(" ".join(prep["data"].loc[prep["data"]["split"] == "train", "original_text"]).split())
    test_only = set(" ".join(prep["data"].loc[prep["data"]["split"] == "test", "original_text"]).split()) - train_tokens
    assert not test_only & set(vocab)


def test_serialization_roundtrip_and_tamper_refusal(tmp_path):
    prep, cfg = _prepared(tmp_path)
    res = sp.train_and_evaluate(prep, cfg)
    run = tmp_path / "run"
    meta = sp.save_run(run, prep, res, cfg, {"rounds": ["pilot"]})
    assert meta["status"] == "trained_and_evaluated" and meta["human_labels_only"]
    for name in ("pipeline_logreg.joblib", "pipeline_linsvm.joblib", "label_mapping.json", "preprocessing_config.json",
                 "metrics.json", "split_assignments.csv"):
        assert name in meta["artifacts_sha256"] and (run / name).exists()
    texts = prep["data"]["original_text"].tolist()[:10]
    for m in sp.MODELS:
        assert list(sp.load_trained_pipeline(run, m).predict(texts)) == list(res["pipelines"][m].predict(texts))
    sa = pd.read_csv(run / "split_assignments.csv")
    assert {"pilot_id", "source_record_id", "split", "leakage_group", "label_source"} <= set(sa.columns)
    with pytest.raises(FileExistsError):
        sp.save_run(run, prep, res, cfg, {})
    with open(run / "pipeline_linsvm.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        sp.load_trained_pipeline(run, "linsvm")
    unfitted = {"models": {}, "pipelines": {}}
    with pytest.raises(ValueError, match="refusing"):
        sp.save_run(tmp_path / "run2", prep, unfitted, cfg, {})
    assert not (tmp_path / "run2").exists()


def test_cli_trains_and_saves_only_with_labels(tmp_path, monkeypatch, capsys):
    import train_stance_baselines as tsb
    ref, truth = synth_reference(160)
    cfg = write_round(tmp_path / "ann", ref, truth, truth)
    monkeypatch.setitem(lb.aaa.ROUNDS, "pilot", cfg["pilot"])
    assert tsb.main(["--rounds", "pilot", "--check-only", "--output-root", str(tmp_path / "runs")]) == 0
    assert not (tmp_path / "runs").exists()
    assert tsb.main(["--rounds", "pilot", "--tag", "t1", "--output-root", str(tmp_path / "runs")]) == 0
    run = tmp_path / "runs" / "run_t1"
    assert (run / "report.md").exists() and json.loads((run / "metadata.json").read_text())["status"] == "trained_and_evaluated"
    assert tsb.main(["--rounds", "pilot", "--tag", "t1", "--output-root", str(tmp_path / "runs")]) == 2
    assert "[TRAINED]" in capsys.readouterr().out


def test_pipeline_objects_and_construct():
    p = sp.build_pipeline("logreg")
    assert [n for n, _ in p.steps] == ["tfidf", "clf"]
    assert type(sp.build_pipeline("linsvm").named_steps["clf"]).__name__ == "LinearSVC"
    with pytest.raises(ValueError):
        sp.build_pipeline("bert")
    assert lb.SCHEMA_LABELS == ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]


def test_research_evaluation_reports_blocked():
    import run_research_evaluation as ev
    by = {r["name"]: r for r in ev.check_supervised_blockers()}
    item = by["supervised stance baselines (TF-IDF + LR/SVM)"]
    if not (ROOT / "data" / "annotation" / "completed").exists():
        assert item["status"] == "BLOCKED" and "train_stance_baselines.py" in item["detail"]
