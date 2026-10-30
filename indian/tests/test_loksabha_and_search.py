"""Tests for indian/retrieval/loksabha.py, the Lok Sabha pilot artifacts and indian/retrieval/policy_search.py.

    python -m pytest indian/tests/test_loksabha_and_search.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

IND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IND))
from retrieval import loksabha as ls  # noqa: E402

LS_DIR = IND / "processed" / "loksabha" / "retrieval_pilot_v1"


def test_clean_question_removes_names_number_and_boilerplate():
    q = ("## 4980. SHRI P.R. SUNDARAM SHRI PR. SENTHIL NATHAN:\n\nWill the Minister of FINANCE be pleased to state: -\n\n"
         "(a) whether the Government has received loans from IDA;")
    c = ls.clean_question(q, ["P.R. SUNDARAM"])
    assert c.startswith("(a) whether the Government has received loans")
    assert "SUNDARAM" not in c and "Minister of FINANCE" not in c and "4980" not in c


def test_clean_question_cuts_embedded_answer():
    q = "714. SHRI X: Will the Minister of IB be pleased to state: (a) vacant posts? **ANSWER** **THE MINISTER** (a): The information is being collected"
    c = ls.clean_question(q, [])
    assert "information is being collected" not in c and "vacant posts" in c


def test_leakage_flags():
    f = ls.leakage_flags("Q text **ANSWER** the minister said yes", "the minister said yes")
    assert f["q_has_answer_marker"] and f["q_contains_answer_head"]
    g = ls.leakage_flags("(a) whether roads are built in the state of kerala this year", "Roads: 10 km were built in Kerala.")
    assert not g["q_has_answer_marker"] and not g["q_contains_answer_head"] and not g["q_shares_8gram_with_answer"]


def test_answer_key_normalises_whitespace_and_case():
    assert ls.answer_key("A  statement\nis laid") == ls.answer_key("a statement is LAID")


needs_pilot = pytest.mark.skipif(not (LS_DIR / "sample_manifest.json").is_file(), reason="pilot not built")


@needs_pilot
def test_pilot_sample_is_eligible_and_leak_free():
    c = pd.read_csv(LS_DIR / "corpus.csv", dtype={"qa_id": str}, keep_default_na=False)
    q = pd.read_csv(LS_DIR / "queries.csv", dtype={"qa_id": str}, keep_default_na=False)
    assert len(c) == 10000 and c.qa_id.is_unique and set(q.qa_id) <= set(c.qa_id)
    used = q[q.excluded_reason == ""]
    cc = c.set_index("qa_id")
    assert not cc.loc[used.qa_id, "q_has_answer_marker"].astype(str).eq("True").any()
    assert not cc.loc[used.qa_id, "q_contains_answer_head"].astype(str).eq("True").any()
    assert set(c.licence_status) == {ls.LICENCE_STATUS}


@pytest.mark.skipif(not (LS_DIR / "evaluation" / "metrics.json").is_file(), reason="evaluation not run")
def test_pilot_metrics_are_bounded_and_complete():
    m = json.loads((LS_DIR / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    for qt, d in m["metrics"].items():
        assert "bm25" in d
        for method, subs in d.items():
            x = subs["all"]
            assert x["queries"] == m["queries_used"] and 0 <= x["recall@1"] <= x["recall@5"] <= x["recall@10"] <= 1


@pytest.mark.skipif(not (LS_DIR / "corpus.csv").is_file() or not (IND / "processed" / "retrieval" / "civis_v1" / "chunks.csv").is_file(),
                    reason="indexes not built")
def test_policy_search_keeps_sources_separate_and_labels_provenance():
    from retrieval import policy_search as ps
    res = ps.search("motor vehicles registration rules", top=3)
    assert set(res) == {"civis", "loksabha"}
    assert res["civis"] and all(r.source == "CIVIS" for r in res["civis"])
    assert all(r.source == "Lok Sabha Q&A" for r in res["loksabha"])
    assert any("motor vehicles" in r.title_or_question.lower() for r in res["civis"])
    for rs in res.values():
        assert [r.rank for r in rs] == list(range(1, len(rs) + 1))
        for r in rs:
            assert r.licence_status.startswith("unverified") and r.record_id and r.provenance
            assert "mygov" not in r.provenance.lower() and "tweet" not in r.provenance.lower()
    assert "not verified answers" in ps.DISCLAIMER


def test_policy_search_cli_json(capsys):
    if not (LS_DIR / "corpus.csv").is_file():
        pytest.skip("indexes not built")
    from retrieval import policy_search as ps
    assert ps.main(["open data licence", "--source", "civis", "--top", "2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["results_by_source"]) == ["civis"] and "UNVERIFIED" in out["disclaimer"]
