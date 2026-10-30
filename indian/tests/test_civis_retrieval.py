"""Tests for indian/retrieval/civis.py and the CIVIS retrieval v1 artifacts.

    python -m pytest indian/tests/test_civis_retrieval.py -q -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

IND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IND))
from retrieval import civis as cv  # noqa: E402

OUT = IND / "processed" / "retrieval" / "civis_v1"
SRC = IND / "processed" / "civis_policy_documents_clean.csv"


def toy_df():
    return pd.DataFrame({
        "civis_record_id": ["train_000", "train_001", "train_002", "test_003", "train_004"],
        "split": ["train", "train", "train", "test", "train"],
        "title": ["Water Policy", "Water Policy Final", "Empty One", "Telecom Rules", "Forest Act"],
        "document_text": ["river water allocation rules", "river water allocation rules", "  ", "telecom spectrum licence fees", "forest land diversion"],
        "reference_summary": ["s1", "s2", "s3", "s4", "s5"],
    })


# ── unit tests ──

def test_canonical_excludes_empty_and_merges_identical_text():
    docs, mapping, excluded = cv.canonical_documents(toy_df())
    assert excluded == ["train_002"]
    assert len(docs) == 3
    water = next(d for d in docs if "train_000" in d.member_ids)
    assert water.member_ids == ["train_000", "train_001"] and water.titles == ["Water Policy", "Water Policy Final"]
    assert mapping["train_000"] == mapping["train_001"] == "train_000"
    assert "train_002" not in mapping


@pytest.mark.parametrize("n", [0, 5, 999, 1000, 1001, 5432])
def test_chunks_cover_text_with_overlap(n):
    text = ("word " * (n // 5 + 1))[:n]
    spans = cv.chunk_text(text, 1000, 100)
    if n == 0:
        assert spans == []
        return
    assert spans[0][0] == 0 and spans[-1][1] == n
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 < b1 and a2 > a1                       # overlap and progress
    assert all(b - a <= 1000 for a, b in spans)


def test_bm25_ranks_exact_term_document_first():
    bm = cv.BM25().fit(["river water allocation", "telecom spectrum fees", "forest land diversion"])
    s = bm.score("spectrum licence")
    assert int(np.argmax(s)) == 1 and s[0] == 0 and s[2] == 0
    assert np.all(bm.score("zzzz unseen") == 0)


def test_doc_scores_take_max_chunk_and_rank_ties_deterministic():
    s = cv.doc_scores_from_chunks(np.array([0.1, 0.9, 0.5, 0.5]), np.array([0, 0, 1, 2]), 3)
    assert s.tolist() == [0.9, 0.5, 0.5]
    assert cv.rank_order(s).tolist() == [0, 1, 2]


def test_rrf_and_minmax_fusion():
    r = cv.rrf([np.array([0, 1, 2]), np.array([2, 1, 0])], 3, k=60)
    assert r[1] == pytest.approx(2 / 62) and r[0] == pytest.approx(r[2])
    f = cv.minmax_fusion(np.array([0.0, 5.0, 10.0]), np.array([1.0, 0.0, 0.5]), 0.5)
    assert f.tolist() == pytest.approx([0.5, 0.25, 0.75])


def test_metrics_from_ranks():
    m = cv.metrics_from_ranks(np.array([1, 2, 11]), n_boot=50)
    ap = lambda x: pytest.approx(x, abs=1e-4)          # metrics are rounded to 4 decimals
    assert m["recall@1"] == ap(1 / 3) and m["recall@5"] == ap(2 / 3) and m["recall@10"] == ap(2 / 3)
    assert m["mrr@10"] == ap((1 + 0.5 + 0) / 3) and m["mrr"] == ap((1 + 0.5 + 1 / 11) / 3)
    assert m["ndcg@10"] == ap((1 + 1 / np.log2(3)) / 3)
    assert cv.first_relevant_rank(np.array([4, 2, 9]), {9}) == 3


def test_first_sentence():
    s = cv.first_sentence("The Ministry released the draft rules on 5th May 2021. It seeks comments. More text.")
    assert s == "The Ministry released the draft rules on 5th May 2021."


# ── artifact integrity (run after the experiment) ──

needs_index = pytest.mark.skipif(not (OUT / "index_manifest.json").is_file(), reason="index not built")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@needs_index
def test_index_matches_source_and_excludes_empty():
    man = json.loads((OUT / "index_manifest.json").read_text(encoding="utf-8"))
    assert man["source_sha256"] == sha(SRC)
    assert sorted(man["excluded_empty_records"]) == ["test_075", "train_079", "train_127", "train_133"]
    docs = pd.read_csv(OUT / "documents.csv", dtype=str, keep_default_na=False)
    members = [m for x in docs.member_record_ids for m in x.split(";")]
    assert not set(members) & set(man["excluded_empty_records"])
    assert len(members) == len(set(members)) == man["records_covered_by_index"] == 166
    assert set(docs.licence_status) == {cv.LICENCE_STATUS}


@needs_index
def test_chunk_offsets_reproduce_hashes_from_source():
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    text = dict(zip(df.civis_record_id, df.document_text))
    ch = pd.read_csv(OUT / "chunks.csv", keep_default_na=False).sample(200, random_state=0)
    for k, a, b, h in zip(ch.doc_key, ch.start, ch.end, ch.chunk_sha256):
        assert cv.sha_text(text[k][a:b]) == h
    assert "text" not in pd.read_csv(OUT / "chunks.csv", nrows=1).columns      # chunk text is not copied


@needs_index
@pytest.mark.parametrize("name", list(cv.ENCODERS))
def test_embeddings_aligned_and_hashed(name):
    p, m = OUT / f"chunk_embeddings_{name}.npy", OUT / f"chunk_embeddings_{name}.json"
    if not p.exists():
        pytest.skip(f"{name} not embedded")
    meta = json.loads(m.read_text(encoding="utf-8"))
    E = np.load(p)
    assert E.shape == (meta["rows"], cv.ENCODERS[name]["dim"]) and meta["embeddings_sha256"] == sha(p)
    norms = np.linalg.norm(E[:500].astype(np.float32), axis=1)
    assert np.allclose(norms, 1, atol=1e-2)


@pytest.mark.skipif(not (OUT / "evaluation" / "metrics.json").is_file(), reason="evaluation not run")
def test_evaluation_targets_and_identical_pairs():
    q = pd.read_csv(OUT / "evaluation" / "queries.csv", dtype=str, keep_default_na=False)
    assert q.groupby("query_type").size().to_dict() == {"summary_first_sentence": 166, "summary_full": 166, "title": 166}
    t = q[q.query_type == "title"].set_index("record_id").target_doc_key
    assert t["train_132"] == t["train_084"] and t["train_124"] == t["train_096"]
    assert not set(q.record_id) & {"test_075", "train_079", "train_127", "train_133"}
