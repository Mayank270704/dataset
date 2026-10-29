"""
tests/test_full_corpus_pipeline.py
==================================
Tests for scripts/run_full_corpus_clustering.py (unsupervised full-corpus pipeline).
Part A uses synthetic data in temp directories; Part B checks the real run_v1 artifacts when present.
No human labels exist or are used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import run_full_corpus_clustering as fc  # noqa: E402


# ═══════════════ Part A: unit tests ═══════════════

def make_corpus(texts, datasets=None, langs=None):
    n = len(texts)
    return pd.DataFrame({
        "corpus_id": range(n),
        "source_dataset": datasets or ["consultation_compilation_2017_18"] * n,
        "source_record_id": [str(i + 2) for i in range(n)],
        "source_row_number": [str(i + 2) for i in range(n)],
        "source_field": ["f"] * n, "question_slug": ["q"] * n,
        "language": langs or ["English"] * n, "stakeholder_group": ["Twitter"] * n,
        "duplicate_text_group_id": [""] * n, "original_text": texts,
    })


def test_unique_table_collapses_exact_and_case_whitespace_variants_keeping_first_and_multiplicity():
    corpus = make_corpus(["Open data now please", "open  data NOW please", "Another text entirely", "Open data now please"])
    unique, entries = fc.build_unique_table(corpus)
    assert list(unique["unique_id"]) == [0, 1]
    assert unique.loc[0, "original_text"] == "Open data now please" and unique.loc[0, "corpus_id"] == 0
    assert unique.loc[0, "multiplicity"] == 3 and unique.loc[1, "multiplicity"] == 1
    assert entries["unique_id"].tolist() == [0, 0, 1, 0]
    assert len(entries) == len(corpus)                      # no entry is dropped


def test_reconciliation_accounts_for_every_extracted_entry(tmp_path):
    rows = []
    def add(ds, col, n):
        rows.extend({"dataset_name": ds, "source_row_index": i, "source_row_number": i, "column_name": col, "text_key": "x",
                     "text_sample_redacted": "x"} for i in range(n))
    add("consultation_compilation_2017_18", "Comment, Question or Idea (In source language)\nCommentaire", 10)
    add("consultation_compilation_2017_18", "Comment, Question or Idea translation if available\nTraduction", 3)
    add("get_involved_questionnaire_2017_18", "Q8. Did we miss anything?", 6)
    add("get_involved_questionnaire_2017_18", "Q2R. Other (please specify)", 4)
    path = tmp_path / "entries.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    eligible = make_corpus(["t"] * 8, ["consultation_compilation_2017_18"] * 4 + ["get_involved_questionnaire_2017_18"] * 4)
    rec = fc.reconcile_extracted_entries(path, eligible).set_index("dataset_name")
    c = rec.loc["consultation_compilation_2017_18"]
    assert c["total_extracted_entries"] == 13 and c["translation_column_not_designed"] == 3
    assert c["designed_field"] == 10 and c["designed_field_eligible_entries"] == 4 and c["designed_field_short_or_filler"] == 6
    q = rec.loc["get_involved_questionnaire_2017_18"]
    assert q["other_free_text_field_not_designed"] == 4 and q["designed_field"] == 6
    assert rec["total_extracted_entries"].sum() == len(rows)


def _unique_fixture(run_root, texts):
    corpus = make_corpus(texts)
    unique, entries = fc.build_unique_table(corpus)
    (run_root / "corpus").mkdir(parents=True)
    unique[["unique_id", "corpus_id", "multiplicity", "source_dataset", "source_record_id", "question_slug", "language",
            "original_text"]].to_csv(run_root / "corpus" / "unique_texts.csv", index=False)
    return unique


def fake_encoder(dim=8):
    def encode(batch):
        out = []
        for t in batch:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h[: dim * 4], dtype=np.uint8)[:dim].astype(np.float32) + 1.0
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)
    return encode


def test_embedding_stage_preserves_alignment_and_resumes_without_recomputation(tmp_path):
    texts = [f"distinct response number {i} about open data" + "x" * (i % 7) for i in range(50)]
    unique = _unique_fixture(tmp_path, texts)
    calls = []
    enc = fake_encoder()
    def counting(batch):
        calls.append(len(batch))
        return enc(batch)
    # simulate an interrupted run: compute only 2 of 4 chunks, then delete final artifacts
    m = fc.stage_embed("multilingual", tmp_path, chunk_size=15, encode=counting)
    assert m["chunks_total"] == 4 and m["chunks_computed_this_run"] == 4 and m["chunks_reused"] == 0
    emb = np.load(tmp_path / "embeddings" / "multilingual" / "embeddings.npy")
    for i, t in enumerate(unique["original_text"]):          # row i is the embedding of unique text i
        assert np.allclose(emb[i], enc([t])[0])
    (tmp_path / "embeddings" / "multilingual" / "embeddings.npy").unlink()
    (tmp_path / "embeddings" / "multilingual" / "embedding_manifest.json").unlink()
    chunks = sorted((tmp_path / "embeddings" / "multilingual" / "chunks").glob("chunk_*.npz"))
    chunks[-1].unlink()
    calls.clear()
    m2 = fc.stage_embed("multilingual", tmp_path, chunk_size=15, encode=counting)
    assert m2["chunks_reused"] == 3 and m2["chunks_computed_this_run"] == 1
    assert sum(calls) <= 15                                    # only the missing chunk was encoded
    assert np.allclose(np.load(tmp_path / "embeddings" / "multilingual" / "embeddings.npy"), emb)


def test_embedding_cache_for_different_texts_is_refused(tmp_path):
    _unique_fixture(tmp_path, [f"text number {i} is here for caching" for i in range(20)])
    fc.stage_embed("baseline", tmp_path, chunk_size=10, encode=fake_encoder())
    (tmp_path / "embeddings" / "baseline" / "embeddings.npy").unlink()
    (tmp_path / "embeddings" / "baseline" / "embedding_manifest.json").unlink()
    pd.DataFrame({"unique_id": [0], "corpus_id": [0], "multiplicity": [1], "source_dataset": ["d"], "source_record_id": ["1"],
                  "question_slug": ["q"], "language": ["English"], "original_text": ["a completely different text"]}
                 ).to_csv(tmp_path / "corpus" / "unique_texts.csv", index=False)
    with pytest.raises(RuntimeError, match="different texts"):
        fc.stage_embed("baseline", tmp_path, chunk_size=10, encode=fake_encoder())


def test_write_new_and_report_refuse_overwrite(tmp_path):
    p = tmp_path / "x.json"
    fc.write_json_new(p, {"a": 1})
    with pytest.raises(FileExistsError):
        fc.write_json_new(p, {"a": 2})
    assert json.loads(p.read_text())["a"] == 1
    existing = tmp_path / "report.md"
    existing.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fc.stage_report(tmp_path, existing)
    assert existing.read_text() == "old"


def test_npmi_coherence_extremes():
    docs = ["alpha beta gamma", "alpha beta delta", "gamma delta", "epsilon zeta", "epsilon zeta eta", "unrelated words only"]
    coh = fc.npmi_coherence(docs, {0: ["alpha", "beta"], 1: ["alpha", "epsilon"]})
    assert coh[0] > 0.5            # co-occur in 2 of 2 alpha-docs
    assert coh[1] == -1.0          # never co-occur


def test_representatives_are_verbatim_traceable_and_exclude_noise():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 6)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    unique = pd.DataFrame({
        "unique_id": [str(i) for i in range(30)], "corpus_id": [str(100 + i) for i in range(30)],
        "multiplicity": ["1"] * 30, "source_dataset": ["d"] * 30, "source_record_id": [str(i) for i in range(30)],
        "question_slug": ["q"] * 30, "language": ["English"] * 30, "original_text": [f"text {i}" for i in range(30)],
    })
    labels = np.array([0] * 12 + [1] * 12 + [-1] * 6)
    reps = fc.representatives(unique, emb, labels, n_central=3, n_edge=2)
    assert set(reps.cluster_id) == {0, 1}
    assert reps.groupby("cluster_id").size().eq(5).all()
    for _, r in reps.iterrows():
        row = unique.iloc[int(r.unique_id)]
        assert r.original_text == row.original_text and r.corpus_id == int(row.corpus_id)
        assert labels[int(r.unique_id)] == r.cluster_id
    central = reps[reps.kind == "central"].groupby("cluster_id").cosine_to_centroid.min()
    edge = reps[reps.kind == "peripheral"].groupby("cluster_id").cosine_to_centroid.max()
    assert (central.values >= edge.values).all()


def test_cluster_diagnostics_uses_entry_multiplicity_and_language_composition():
    entries = pd.DataFrame({
        "unique_id": [0, 0, 0, 1, 2, 3],
        "source_dataset": ["a", "a", "a", "a", "b", "b"],
        "language": ["English", "English", "English", "French", "French", "English"],
        "stakeholder_group": ["Twitter"] * 6, "text_key": ["x", "x", "x", "y", "z", "w"],
    })
    labels = np.array([0, 0, 1, -1])
    topics = pd.DataFrame()
    d = fc.cluster_diagnostics(entries, labels, topics, {0: 0.2, 1: None})
    assert d["entries"] == 6 and d["entries_clustered"] == 5 and abs(d["entry_noise_ratio"] - round(1 / 6, 4)) < 1e-9
    assert d["unique_noise_ratio"] == 0.25 and d["n_clusters"] == 2
    assert d["clusters_duplicate_heavy"] == 0 or d["clusters_duplicate_heavy"] == 1
    c0 = next(p for p in d["per_cluster"] if p["cluster_id"] == 0)
    assert c0["entries"] == 4 and c0["unique_texts"] == 2 and c0["top_duplicate_text_share"] == 0.75


def test_reporting_configuration_rule():
    summary = {
        "a": {"n_clusters": 30, "entry_noise_ratio": 0.2, "mean_npmi": 0.10, "hdbscan": {"min_cluster_size": 10}},
        "b": {"n_clusters": 60, "entry_noise_ratio": 0.3, "mean_npmi": 0.25, "hdbscan": {"min_cluster_size": 25}},
        "c": {"n_clusters": 5, "entry_noise_ratio": 0.1, "mean_npmi": 0.90, "hdbscan": {"min_cluster_size": 50}},   # too few
        "d": {"n_clusters": 40, "entry_noise_ratio": 0.5, "mean_npmi": 0.80, "hdbscan": {"min_cluster_size": 50}},  # too noisy
        "_runtime": {},
    }
    assert fc.select_reporting_config(summary) == "b"
    assert fc.select_reporting_config({"x": {"n_clusters": 2, "entry_noise_ratio": 0.9, "mean_npmi": 0.1,
                                             "hdbscan": {"min_cluster_size": 10}}}) is None


def test_size_stats():
    s = fc.size_stats([50, 30, 10, 5, 5])
    assert s["n"] == 5 and s["max"] == 50 and s["min"] == 5 and s["median"] == 10
    assert fc.size_stats([]) == {"n": 0}


def test_cluster_stage_end_to_end_on_synthetic_blobs(tmp_path):
    rng = np.random.default_rng(1)
    centers = np.eye(3, 16)[:, :16] * 4
    pts, texts = [], []
    for c in range(3):
        for i in range(40):
            v = centers[c] + rng.normal(scale=0.3, size=16)
            pts.append(v / np.linalg.norm(v))
            texts.append(f"topic{c} words about subject{c} item {i} open government data")
    emb = np.vstack(pts).astype(np.float32)
    unique = _unique_fixture(tmp_path, texts)
    corpus = make_corpus(texts)
    _, entries = fc.build_unique_table(corpus)
    entries.to_csv(tmp_path / "corpus" / "corpus_entries.csv", index=False)
    (tmp_path / "embeddings" / "multilingual").mkdir(parents=True)
    np.save(tmp_path / "embeddings" / "multilingual" / "embeddings.npy", emb)
    umap_cfg = {"u5": {"n_components": 3, "n_neighbors": 8, "min_dist": 0.0, "metric": "cosine"}}
    hdb_cfg = {"m10s3": {"min_cluster_size": 10, "min_samples": 3}}
    res = fc.stage_cluster("multilingual", tmp_path, umap_cfg, hdb_cfg, seed=42, do_seed_check=False)
    d = res["u5_m10s3"]
    assert d["n_clusters"] == 3 and d["entry_noise_ratio"] < 0.1
    cdir = tmp_path / "clustering" / "multilingual" / "u5_m10s3"
    ea = pd.read_csv(cdir / "entry_assignments.csv")
    assert len(ea) == len(entries) and ea["corpus_id"].tolist() == list(range(len(entries)))
    topics = pd.read_csv(cdir / "topics.csv")
    assert {"cluster_id", "top_keywords", "npmi"} <= set(topics.columns)
    reps = pd.read_csv(cdir / "representatives.csv")
    assert set(reps.original_text) <= set(texts)
    # each cluster maps to one blob (topic prefix) in this synthetic data
    for cid, g in reps.groupby("cluster_id"):
        assert g.original_text.str.extract(r"^(topic\d)")[0].nunique() == 1
    again = fc.stage_cluster("multilingual", tmp_path, umap_cfg, hdb_cfg, seed=42, do_seed_check=False)
    assert again["u5_m10s3"]["n_clusters"] == 3                 # cached summary returned, nothing overwritten


def test_exact_knn_matches_brute_force_and_puts_self_first():
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(60, 10)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx, dist = fc.exact_knn_cosine(emb, 7, chunk=16)       # chunk smaller than N exercises chunking
    assert idx.shape == (60, 7) and (idx[:, 0] == np.arange(60)).all() and (dist[:, 0] == 0).all()
    for q in (0, 17, 59):
        sims = emb @ emb[q]
        sims[q] = np.inf
        assert set(np.argsort(-sims)[:7]) == set(idx[q])
        assert np.all(np.diff(dist[q]) >= -1e-7)             # distances non-decreasing
    assert dist.min() >= 0.0 and dist.max() <= 2.0


def test_low_memory_loader_matches_standard_loading_on_a_tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertModel
    cfg = BertConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(tmp_path, safe_serialization=True)
    ref = BertModel.from_pretrained(tmp_path).eval()
    low = fc.load_model_from_dir(tmp_path)
    ids = torch.randint(0, 120, (3, 11))
    mask = torch.ones_like(ids)
    mask[0, 8:] = 0
    with torch.no_grad():
        a = ref(input_ids=ids, attention_mask=mask).last_hidden_state
        b = low(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(a, b, atol=1e-6)
    assert not any(p.is_meta for p in low.parameters()) and not any(x.is_meta for x in low.buffers())


# ═══════════════ Part B: real run artifacts ═══════════════

RUN = fc.RUN_ROOT
real = pytest.mark.skipif(not (RUN / "run_manifest.json").exists(), reason="full-corpus run_v1 not completed")


@real
def test_real_corpus_counts_and_reconciliation():
    m = json.loads((RUN / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert m["entries"] == 10108 and m["unique_texts"] == 10043 and m["reconciliation_total_extracted"] == 14135
    assert sum(m["by_dataset"].values()) == 10108
    rec = pd.read_csv(RUN / "corpus" / "reconciliation.csv")
    assert rec["total_extracted_entries"].sum() == 14135
    parts = (rec["designed_field"] + rec["translation_column_not_designed"] + rec["other_free_text_field_not_designed"])
    assert (parts == rec["total_extracted_entries"]).all()
    assert (rec["designed_field_eligible_entries"] + rec["designed_field_short_or_filler"] == rec["designed_field"]).all()
    assert rec["designed_field_eligible_entries"].sum() == 10108


@real
@pytest.mark.parametrize("key", ["multilingual", "baseline"])
def test_real_embeddings_are_aligned_normalised_and_hash_matches(key):
    emb = np.load(RUN / "embeddings" / key / "embeddings.npy")
    unique = pd.read_csv(RUN / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    m = json.loads((RUN / "embeddings" / key / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert emb.shape == (len(unique), 384) and np.isfinite(emb).all()
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    assert m["texts_sha256"] == fc.sha256_text_list(unique["original_text"].tolist())
    assert m["embeddings_sha256"] == fc.sha256_file(RUN / "embeddings" / key / "embeddings.npy")
    ref = fc.check_against_reference(key, RUN)
    assert ref["status"] == "compared" and ref["n_compared"] > 400 and ref["min_cosine"] > 0.999


@real
def test_real_assignments_cover_every_entry_exactly_once_and_stay_traceable():
    entries = pd.read_csv(RUN / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_validated"] is False
    for cfg_path in manifest["assignment_sha256"]:
        key, cfg = cfg_path.split("/")
        a = pd.read_csv(RUN / "clustering" / key / cfg / "entry_assignments.csv", dtype=str, keep_default_na=False)
        assert a["corpus_id"].tolist() == entries["corpus_id"].tolist()
        assert (a["source_record_id"] == entries["source_record_id"]).all()
        assert fc.sha256_file(RUN / "clustering" / key / cfg / "entry_assignments.csv") == manifest["assignment_sha256"][cfg_path]
    reps = pd.read_csv(RUN / "clustering" / "multilingual" / manifest["reporting_configuration"].split("/")[1] / "representatives.csv",
                       dtype=str, keep_default_na=False)
    by_id = entries.set_index("corpus_id")["original_text"]
    assert all(by_id[r.corpus_id] == r.original_text for r in reps.itertuples())


@real
def test_real_report_states_unvalidated_and_lists_limitations():
    text = fc.REPORT_PATH.read_text(encoding="utf-8")
    assert "NOT human-validated" in text and "not validated public concerns" in text
    assert "13,541" in text and "could not be reproduced" in text
    assert "Limitations" in text and "Reproducibility" in text
    assert "accuracy" not in text.lower().replace("accuracy figures were used or produced", "")
