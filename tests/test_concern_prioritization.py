"""
tests/test_concern_prioritization.py
====================================
Signals, scoring, baseline comparison, evidence verification, collective-demand screening and the end-to-end
pipeline of models/concern_discovery/prioritization.py and scripts/run_concern_prioritization.py.
Synthetic data except one read-only check of the real inputs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from models.concern_discovery import prioritization as pr  # noqa: E402

NON_SUB = re.compile(r"don'?t know|thanks", re.I)


def make_entries():
    """Cluster 1: 30 entries, mostly one repeated template (campaign). Cluster 2: 20 independent texts in 3 datasets.
    Cluster 3: 25 independent texts, one dataset. Cluster 4: non-concern tier. Two noise entries near cluster 2."""
    rows, cid = [], 0

    def add(cluster, uid, dup, ds, field, state="clustered", near=-1, lang="English"):
        nonlocal cid
        rows.append({"corpus_id": cid, "unique_id": uid, "source_dataset": ds, "source_record_id": str(cid),
                     "source_row_number": str(cid), "source_field": field, "question_slug": field, "language": lang,
                     "cluster_id": cluster, "state": state, "dup_group_id": dup, "nearest_cluster_if_noise": near})
        cid += 1

    for i in range(26):
        add(1, 1000, 1000, "consultation_compilation_2017_18", "comment")          # one text repeated 26 times
    for i in range(4):
        add(1, 1001 + i, 1001 + i, "consultation_compilation_2017_18", "comment")
    dss = ["consultation_compilation_2017_18", "get_involved_questionnaire_2017_18", "individual_comments_2016_18"]
    for i in range(20):
        add(2, 2000 + i, 2000 + i, dss[i % 3], ["q8", "q14", "practical_idea", "comment"][i % 4], lang="French" if i % 5 == 0 else "English")
    for i in range(25):
        add(3, 3000 + i, 3000 + i, "get_involved_questionnaire_2017_18", "q14")
    for i in range(20):
        add(4, 4000 + i, 4000 + i, "get_involved_questionnaire_2017_18", "q14")
    add(-1, 5000, 5000, "get_involved_questionnaire_2017_18", "q8", state="noise", near=2)
    add(-1, 5001, 5001, "get_involved_questionnaire_2017_18", "q8", state="noise", near=2)
    return pd.DataFrame(rows)


GROUPS = pd.DataFrame({"cluster_id": [1, 2, 3, 4], "tier": ["campaign_or_template", "candidate", "needs_review", "non_concern_pattern"],
                       "stability_jaccard_mean": [0.95, 0.9, 0.6, 0.99], "short_label": ["registry", "privacy", "data", "site"],
                       "flags": ["template_driven", "", "moderately_unstable", "no_opinion_pattern"]})
EVID = {1: 2, 2: 5, 3: 3, 4: 5}


def test_validate_config():
    w = pr.validate_config({k: 2.0 for k in pr.SIGNALS}, pr.DEFAULT_TIER_QUALITY)
    assert abs(sum(w.values()) - 1) < 1e-12 and len(set(w.values())) == 1
    for bad in ({**pr.DEFAULT_WEIGHTS, "urgency": 0.1}, {k: v for k, v in pr.DEFAULT_WEIGHTS.items() if k != "evidence"},
                {**pr.DEFAULT_WEIGHTS, "evidence": -0.1}, {k: 0.0 for k in pr.SIGNALS}, {**pr.DEFAULT_WEIGHTS, "evidence": float("nan")}):
        with pytest.raises(pr.PrioritizationInputError):
            pr.validate_config(bad, pr.DEFAULT_TIER_QUALITY)
    with pytest.raises(pr.PrioritizationInputError):
        pr.validate_config(pr.DEFAULT_WEIGHTS, {"candidate": 1.0})


def test_entropy():
    assert pr.normalised_entropy(pd.Series([10]), 3) == 0.0
    assert pr.normalised_entropy(pd.Series([5, 5, 5]), 3) == pytest.approx(1.0)
    assert 0 < pr.normalised_entropy(pd.Series([9, 1]), 3) < pr.normalised_entropy(pd.Series([5, 5]), 3)


def test_signals_keep_raw_and_independent_counts_separate():
    s = pr.compute_signals(make_entries(), GROUPS, EVID).set_index("cluster_id")
    assert (s.loc[1, "entries"], s.loc[1, "unique_texts"], s.loc[1, "independent_texts"]) == (30, 5, 5)
    assert s.loc[1, "duplicate_influence"] == pytest.approx(1 - 5 / 30, abs=1e-4)
    assert s.loc[1, "largest_repeat_share"] == pytest.approx(26 / 30, abs=1e-4)
    assert s.loc[2, "datasets_present"] == 3 and s.loc[3, "dataset_entropy"] == 0.0
    assert s.loc[2, "nearby_noise_entries"] == 2 and s.loc[2, "french"] == 4
    assert (s["independent_texts"] <= s["unique_texts"]).all() and (s["unique_texts"] <= s["entries"]).all()
    assert not s.loc[4, "ranked"] and "not a policy concern" in s.loc[4, "not_ranked_reason"]
    s2 = pr.compute_signals(make_entries(), GROUPS, {1: 2, 2: 0, 3: 3}).set_index("cluster_id")
    assert not s2.loc[2, "ranked"] and "no traceable" in s2.loc[2, "not_ranked_reason"]


def test_score_contributions_and_campaign_not_counted_as_independent():
    sig = pr.compute_signals(make_entries(), GROUPS, EVID)
    r = pr.score(sig, pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY).set_index("cluster_id")
    contrib = r[[f"contrib_{k}" for k in pr.SIGNALS]].sum(axis=1)
    assert np.allclose(contrib, r["score"], atol=1e-6)
    assert r.loc[1, "rank_frequency"] == 1                       # 30 raw entries: first by frequency
    assert r.loc[1, "rank_multi"] == 3                           # 5 independent texts: last by multi-signal score
    assert r.loc[2, "rank_multi"] == 1 and r.loc[2, "rank_change_vs_frequency"] == 2
    assert 4 not in r.index
    w = pr.validate_config(pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    assert r.loc[2, "contrib_reliability"] == pytest.approx(w["reliability"] * 0.9, abs=1e-6)
    expl = pr.explain(r.reset_index())
    assert any(e.startswith("up ") for e in expl) and any(e.startswith("down ") for e in expl)


def test_weights_change_ranking_transparently():
    sig = pr.compute_signals(make_entries(), GROUPS, EVID)
    only_volume = {k: (1.0 if k == "independent_volume" else 0.0) for k in pr.SIGNALS}
    r = pr.score(sig, only_volume, pr.DEFAULT_TIER_QUALITY).set_index("cluster_id")
    assert r.loc[3, "rank_multi"] == 1                           # 25 independent texts is the largest
    assert (r[[f"contrib_{k}" for k in pr.SIGNALS if k != "independent_volume"]] == 0).all().all()


def test_baseline_comparison_and_sensitivity():
    sig = pr.compute_signals(make_entries(), GROUPS, EVID)
    r = pr.score(sig, pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    c = pr.compare_baseline(r, ks=(1, 2))
    assert c["n_ranked"] == 3 and c["top1_overlap"]["entered"] == [2] and c["top1_overlap"]["left"] == [1]
    s = pr.sensitivity(sig, pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    assert len(s["kendall_tau_vs_default"]) == 3 * len(pr.SIGNALS)
    assert all(lo <= hi for lo, hi in s["rank_range"].values())


def test_evidence_verification():
    e = make_entries()
    text = {i: f"comment number {i} about something" for i in e["corpus_id"]}
    ex = pd.DataFrame({"cluster_id": [2, 2, 3, 2], "corpus_id": [40, 41, 40, 42],
                       "quote": [text[40][:10], "invented quotation", text[40][:10], text[42]],
                       "full_text": [text[40], text[41], text[40], text[42] + " edited"]})
    ok, probs = pr.verify_evidence(ex, e, text)
    assert ok["corpus_id"].tolist() == [40]
    assert len(probs) == 3 and any("not a member" in p for p in probs)


def test_collective_demand_screening():
    e = make_entries().query("cluster_id in (1, 2)").copy()
    text = {i: f"independent comment {i} about privacy" for i in e["corpus_id"]}
    for i in e.loc[e["unique_id"] == 1000, "corpus_id"]:
        text[i] = "We want a public registry of beneficial owners #OpenGovCan https://t.co/x"
    rows = [{"unique_id": u, "dup_group_id": u, "group_entries": int((e["unique_id"] == u).sum()), "group_unique_texts": 1}
            for u in e["unique_id"].unique()]
    extra = make_entries().iloc[:0]
    dups = pd.DataFrame(rows)
    # add a markup-only group and a no-opinion group of 5 entries each
    for k, t in ((9000, "@OpenGovCan https://t.co/abc"), (9001, "I don't know about it")):
        add = pd.DataFrame([{**e.iloc[0].to_dict(), "corpus_id": 900 + 10 * (k - 9000) + j, "unique_id": k, "dup_group_id": k,
                             "state": "noise", "cluster_id": -1} for j in range(5)])
        e = pd.concat([e, add], ignore_index=True)
        for c in add["corpus_id"]:
            text[c] = t
        dups = pd.concat([dups, pd.DataFrame([{"unique_id": k, "dup_group_id": k, "group_entries": 5, "group_unique_texts": 1}])])
    _ = extra
    cand, ev = pr.collective_demand_candidates(e, dups, text, NON_SUB)
    dec = dict(zip(cand["dup_group_id"], cand["decision"]))
    assert dec[1000] == "included" and dec[9000].startswith("excluded: no substantive") and dec[9001].startswith("excluded: no-opinion")
    assert set(ev["dup_group_id"]) == {1000} and len(ev) == 26
    assert all(text[c] == t for c, t in zip(ev["corpus_id"], ev["text"]))
    assert cand.set_index("dup_group_id").loc[1000, "source_records"] == 26


def test_output_validation_catches_errors():
    sig = pr.compute_signals(make_entries(), GROUPS, EVID)
    r = pr.score(sig, pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    w = pr.validate_config(pr.DEFAULT_WEIGHTS, pr.DEFAULT_TIER_QUALITY)
    ev = pd.DataFrame({"cluster_id": [1, 2, 3]})
    assert pr.validate_outputs(r, ev, w) == []
    bad = r.copy()
    bad.loc[0, "score"] += 0.1
    assert any("sum" in p for p in pr.validate_outputs(bad, ev, w))
    assert any("without verified evidence" in p for p in pr.validate_outputs(r, ev.iloc[:2], w))
    bad = r.copy()
    bad.loc[0, "rank_multi"] = 1 if bad.loc[1, "rank_multi"] != 1 else 2
    bad.loc[1, "rank_multi"] = bad.loc[0, "rank_multi"]
    assert any("permutation" in p for p in pr.validate_outputs(bad, ev, w))


# ── end to end on the synthetic concern-discovery tree ───────────────────────

@pytest.fixture()
def discovery(tmp_path):
    import run_concern_discovery as rcd
    import test_concern_discovery as tcd
    run_v1, exp = tcd._build_tree(tmp_path)
    cd = tmp_path / "cd"
    rcd.stage_run(cd, run_v1, exp)
    return cd, run_v1


def test_end_to_end(discovery, tmp_path):
    import run_concern_prioritization as rcp
    cd, run_v1 = discovery
    out = tmp_path / "prio"
    res = rcp.stage_run(out, None, cd, run_v1)
    assert res["ranked"] >= 1
    ranked = pd.read_csv(out / "ranking.csv")
    ev = pd.read_csv(out / "ranking_evidence.csv", keep_default_na=False)
    corpus = pd.read_csv(run_v1 / "corpus" / "corpus_entries.csv", keep_default_na=False).set_index("corpus_id")
    assert set(ranked["cluster_id"]) <= set(ev["cluster_id"])
    assert all(q in corpus.loc[c, "original_text"] for q, c in zip(ev["quote"], ev["corpus_id"]))
    assert all(corpus.loc[c, "source_record_id"] == str(r) or str(corpus.loc[c, "source_record_id"]) == str(r)
               for c, r in zip(ev["corpus_id"], ev["source_record_id"]))
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["human_validated"] is False and man["output_validation"] == "PASSED"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["fitted_to_human_judgments"] is False
    with pytest.raises(FileExistsError):
        rcp.stage_run(out, None, cd, run_v1)
    txt = rcp.stage_report(out, tmp_path / "report.md")
    for phrase in ("Not human-validated", "None of them is a count of respondents", "Kendall", "corpus_id"):
        assert phrase in txt


def test_custom_weights_and_bad_weights(discovery, tmp_path):
    import run_concern_prioritization as rcp
    cd, run_v1 = discovery
    wf = tmp_path / "w.json"
    wf.write_text(json.dumps({"weights": {k: (1.0 if k == "independent_volume" else 0.0) for k in pr.SIGNALS}}), encoding="utf-8")
    rcp.stage_run(tmp_path / "p1", wf, cd, run_v1)
    cfg = json.loads((tmp_path / "p1" / "config.json").read_text(encoding="utf-8"))
    assert cfg["weights_normalised"]["independent_volume"] == 1.0
    wf.write_text(json.dumps({"weights": {"popularity": 1.0}}), encoding="utf-8")
    with pytest.raises(pr.PrioritizationInputError):
        rcp.stage_run(tmp_path / "p2", wf, cd, run_v1)


def test_tampered_inputs_rejected(discovery, tmp_path):
    import run_concern_prioritization as rcp
    cd, run_v1 = discovery
    p = cd / "selected" / "entry_assignments.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(rcp.InputError, match="manifest"):
        rcp.stage_run(tmp_path / "p", None, cd, run_v1)


@pytest.mark.skipif(not (ROOT / "data" / "full_corpus" / "concern_discovery_v1" / "run_manifest.json").exists(),
                    reason="real concern-discovery outputs not present")
def test_real_inputs_pass_integrity_checks():
    import run_concern_prioritization as rcp
    d = rcp.load_inputs()
    assert len(d["entries"]) == 10108 and all(c["status"] == "PASSED" for c in d["checks"])
    ok, probs = pr.verify_evidence(d["examples"], d["entries"], d["corpus_text"])
    assert probs == [] and len(ok) == len(d["examples"])


def test_research_evaluation_section(discovery, tmp_path):
    import run_concern_prioritization as rcp
    import run_research_evaluation as ev
    assert ev.check_prioritization(tmp_path / "none")[0]["status"] == "NOT RUN"
    cd, run_v1 = discovery
    out = tmp_path / "prio"
    rcp.stage_run(out, None, cd, run_v1)
    assert {r["status"] for r in ev.check_prioritization(out)} == {"PASSED"}
    (out / "ranking.csv").write_text("tampered", encoding="utf-8")
    assert ev.check_prioritization(out)[0]["status"] == "FAILED"
