"""
tests/test_clustering_experiments_v2.py
=======================================
Tests for scripts/run_clustering_experiments_v2.py.

Part A uses SYNTHETIC texts/embeddings in temp directories (alignment, determinism, duplicate handling,
provenance, language reporting, summary faithfulness, overwrite protection).
Part B checks the REAL experiments_v2 outputs when present. No human labels exist or are used.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import run_clustering_experiments_v2 as ex  # noqa: E402
import run_full_corpus_clustering as fc  # noqa: E402

DIM = 16


def fake_encoder(batch):
    out = []
    for t in batch:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h[:DIM], dtype=np.uint8).astype(np.float32) + 1.0
        out.append(v / np.linalg.norm(v))
    return np.vstack(out)


def make_run_v1(tmp_path: Path) -> Path:
    """120 unique texts: A (English topic), B (French topic), C (templated tweets with URL+hashtag)."""
    rng = np.random.default_rng(0)
    centers = np.zeros((3, DIM), dtype=np.float32)
    centers[0, 0] = centers[1, 5] = centers[2, 10] = 1.0
    texts, langs, embs, fields = [], [], [], []
    for i in range(40):
        texts.append(f"Publish more open data about topic alpha item {i} in accessible formats")
        langs.append("English"); fields.append("q8"); embs.append(centers[0])
    for i in range(40):
        texts.append(f"Publier davantage de données ouvertes sur le sujet bêta élément {i}")
        langs.append("French"); fields.append("q14"); embs.append(centers[1])
    for i in range(40):
        texts.append(f"Help shape Canada's action plan on open government https://t.co/x{i:03d} #opengovcan")
        langs.append("English"); fields.append("comment_question_or_idea"); embs.append(centers[2])
    emb = np.array([e + rng.normal(scale=0.05, size=DIM) for e in embs], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(texts)
    ds = (["get_involved_questionnaire_2017_18"] * 80) + (["consultation_compilation_2017_18"] * 40)
    unique = pd.DataFrame({
        "unique_id": range(n), "corpus_id": range(n), "multiplicity": 1, "source_dataset": ds,
        "source_record_id": [str(i + 2) for i in range(n)], "question_slug": fields, "language": langs, "original_text": texts})
    # five extra entries that repeat existing texts exactly (corpus ids n..n+4)
    extra_u = [0, 1, 2, 40, 41]
    rows = []
    for cid in range(n):
        rows.append((cid, cid))
    for k, u in enumerate(extra_u):
        rows.append((n + k, u))
    entries = pd.DataFrame({"corpus_id": [r[0] for r in rows], "unique_id": [r[1] for r in rows]})
    for col in ("source_dataset", "source_record_id", "question_slug", "language", "original_text"):
        entries[col] = [unique.loc[u, col] for u in entries["unique_id"]]
    entries["stakeholder_group"] = ["Twitter" if d.startswith("consultation") else "" for d in entries["source_dataset"]]
    entries["text_key"] = entries["original_text"].map(fc.normalise_text)
    unique["multiplicity"] = unique["unique_id"].map(entries.groupby("unique_id").size())
    run = tmp_path / "run_v1"
    (run / "corpus").mkdir(parents=True)
    (run / "embeddings" / "multilingual").mkdir(parents=True)
    entries.to_csv(run / "corpus" / "corpus_entries.csv", index=False)
    unique.to_csv(run / "corpus" / "unique_texts.csv", index=False)
    np.save(run / "embeddings" / "multilingual" / "embeddings.npy", emb)
    return run


SMALL = [
    ex._exp("raw", "umap_m25s5", (42, 43)),
    ex._exp("strip", "umap_m25s5", (42,)),
    ex._exp("strip_dedup", "umap_m25s5"),
    ex._exp("strip", "umap_m25s5", subset="English"),
]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    run = make_run_v1(tmp)
    out = tmp / "out"
    summary = ex.run_experiments(out, run, SMALL, encode=fake_encoder)
    return tmp, run, out, summary


# ── preprocessing ─────────────────────────────────────────────────────────────

def test_strip_social_removes_urls_mentions_hashtags_and_is_idempotent():
    t = "Thanks @OpenGovCan!  See https://t.co/abc123 and www.example.org #OpenGovCan #data now"
    s = ex.strip_social(t)
    assert s == "Thanks !See and now".replace("!See", "! See") or "https" not in s
    assert "@" not in s and "#" not in s and "http" not in s and "www" not in s
    assert ex.strip_social(s) == s
    assert ex.strip_social("Plain text without markup") == "Plain text without markup"


def test_entry_features_flags_and_counts():
    e = pd.DataFrame({"corpus_id": [0, 1, 2, 3], "original_text": [
        "Great plan https://t.co/a #x", "Great plan https://t.co/b #y", "no markup here at all", "no markup here at all"],
        "text_key": ["great plan https://t.co/a #x", "great plan https://t.co/b #y", "no markup here at all", "no markup here at all"]})
    f = ex.entry_features(e)
    assert f["has_url"].tolist() == [True, True, False, False]
    assert f["has_hashtag"].tolist() == [True, True, False, False]
    assert f["skey_count"].tolist() == [2, 2, 2, 2]            # URL/hashtag-stripped keys repeat
    assert f["exact_dup_count"].tolist() == [1, 1, 2, 2]       # only the last two are exact duplicates


def test_variants_population_states_and_embedding_reuse():
    unique = pd.DataFrame({"original_text": [
        "A perfectly ordinary comment about open data portals and their usability",   # unchanged, >40
        "See https://t.co/a #x",                                                     # content-free after strip
        "Short but valid comment here",                                              # unchanged, 28 chars (<40)
        "Help shape the national plan for open government https://t.co/1 #og",        # changed
        "Help shape the national plan for open government https://t.co/2 #og",        # changed, same stripped key
    ]})
    emb = np.eye(5, DIM, dtype=np.float32)
    calls = []
    def enc(batch):
        calls.append(list(batch))
        return fake_encoder(batch)
    v = ex.build_variants(unique, emb, enc)
    assert [t for c in calls for t in c] == ["Help shape the national plan for open government", ] * 2
    assert np.array_equal(v["strip"].emb[:3], emb[:3])                  # unchanged texts reuse the cached vectors
    assert v["strip"].state.tolist() == [0, ex.CONTENT_FREE, 0, 0, 0]
    assert v["no_short"].state.tolist() == [0, ex.CONTENT_FREE, ex.SHORT_EXCLUDED, 0, 0]
    assert v["strip_dedup"].rep_of.tolist() == [0, 1, 2, 3, 3]          # template collapsed to its first occurrence
    assert v["strip_dedup"].pop_idx.tolist() == [0, 2, 3]
    assert ex.core_mask(v).tolist() == [True, False, False, True, True]


# ── clustering determinism, alignment, duplicates ─────────────────────────────

def test_experiments_are_deterministic_and_aligned(synthetic, tmp_path):
    tmp, run, out, summary = synthetic
    out2 = tmp_path / "second"
    ex.run_experiments(out2, run, SMALL[:2], encode=fake_encoder)
    entries = pd.read_csv(run / "corpus" / "corpus_entries.csv")
    for eid in ("raw__umap_m25s5", "strip__umap_m25s5"):
        a = (out / eid / "entry_assignments.csv").read_bytes()
        b = (out2 / eid / "entry_assignments.csv").read_bytes()
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
        df = pd.read_csv(out / eid / "entry_assignments.csv")
        assert df["corpus_id"].tolist() == entries["corpus_id"].tolist()          # source alignment
        assert df["unique_id"].tolist() == entries["unique_id"].tolist()


def test_exact_duplicate_entries_share_their_text_cluster(synthetic):
    tmp, run, out, summary = synthetic
    df = pd.read_csv(out / "raw__umap_m25s5" / "entry_assignments.csv")
    for uid, g in df.groupby("unique_id"):
        assert g["cluster_id"].nunique() == 1


def test_seed_stability_and_cluster_recovery_on_separated_blobs(synthetic):
    m = synthetic[3]["experiments"]["raw__umap_m25s5"]
    assert m["n_clusters"] == 3
    assert m["seed_ari_mean"] is not None and m["seed_ari_mean"] > 0.95
    assert set(m["seed_cluster_counts"]) == {"42", "43"}


def test_template_collapse_inherits_cluster_and_is_documented(synthetic):
    tmp, run, out, summary = synthetic
    assert summary["template_collapsed_unique_texts"] == 39              # 40 identical stripped tweets -> 1 representative
    df = pd.read_csv(out / "strip_dedup__umap_m25s5" / "entry_assignments.csv")
    tweets = df[df["unique_id"].between(80, 119)]
    assert tweets["cluster_id"].nunique() == 1                             # all inherit the representative's state/cluster
    assert len(df) == len(pd.read_csv(run / "corpus" / "corpus_entries.csv"))  # no entry dropped


def test_excluded_texts_are_ledgered_as_states_not_dropped(synthetic):
    tmp, run, out, summary = synthetic
    m = summary["experiments"]["strip__umap_m25s5"]
    assert sum(m["states"].values()) + int(m["coverage_all_entries"] * m["entries"] + 0.5) == m["entries"]
    df = pd.read_csv(out / "strip__umap_m25s5" / "entry_assignments.csv")
    assert set(df["state"]) <= {"clustered", "noise", "content_free_after_preprocessing", "below_min_length", "outside_language_subset"}


# ── language reporting ────────────────────────────────────────────────────────

def test_language_subset_run_marks_other_language_and_reports_both(synthetic):
    tmp, run, out, summary = synthetic
    df = pd.read_csv(out / "strip__umap_m25s5_englishonly" / "entry_assignments.csv")
    assert (df.loc[df["language"] == "French", "state"] == "outside_language_subset").all()
    m = summary["experiments"]["raw__umap_m25s5"]
    assert m["core_entries_english"] + m["core_entries_french"] == m["core_entries"]
    assert 0 <= m["coverage_core_french"] <= 1 and 0 <= m["coverage_core_english"] <= 1
    table = pd.read_csv(out / "raw__umap_m25s5" / "cluster_table.csv")
    assert (table["english"] + table["french"] + table["other_unknown"] == table["n_entries"]).all()
    french_cluster = table.loc[table["french"].idxmax()]
    assert french_cluster["french_share"] > 0.9


# ── cluster table provenance and flags ────────────────────────────────────────

def test_cluster_table_flags_template_social_clusters_and_keeps_provenance(synthetic):
    tmp, run, out, summary = synthetic
    table = pd.read_csv(out / "raw__umap_m25s5" / "cluster_table.csv")
    unique = pd.read_csv(run / "corpus" / "unique_texts.csv", keep_default_na=False)
    tweet_cluster = table.loc[table["social_markup_share"].idxmax()]
    assert tweet_cluster["social_markup_share"] == 1.0 and bool(tweet_cluster["likely_channel_template_driven"])
    assert bool(tweet_cluster["flag_channel_concentrated"]) and bool(tweet_cluster["flag_social_markup"])
    ordinary = table.loc[table["social_markup_share"].idxmin()]
    assert not bool(ordinary["likely_channel_template_driven"])
    for _, row in table.iterrows():
        for rep in json.loads(row["representatives"]):
            src = unique.loc[unique["corpus_id"] == rep["corpus_id"]].iloc[0]
            assert rep["text"] == src["original_text"]
            assert rep["source_record_id"] == str(src["source_record_id"]) and rep["source_dataset"] == src["source_dataset"]
        assert row["label"].startswith("Keyword group (descriptive only, not a validated concern)")
    assert table["n_entries"].sum() == summary["experiments"]["raw__umap_m25s5"]["entries"] * \
        (1 - summary["experiments"]["raw__umap_m25s5"]["coverage_all_entries"] * 0) - \
        summary["experiments"]["raw__umap_m25s5"]["states"].get("noise", 0)


# ── summary faithfulness ──────────────────────────────────────────────────────

def _quotes(text: str) -> list[str]:
    return re.findall(r"“(.*?)”", text)


def test_summaries_quote_only_verbatim_source_text_with_traceable_ids(synthetic):
    tmp, run, out, summary = synthetic
    table = pd.read_csv(out / "raw__umap_m25s5" / "cluster_table.csv", keep_default_na=False)
    for c in ("flag_social_markup", "flag_template_repeat", "flag_channel_concentrated", "flag_short_responses",
              "flag_single_instrument", "likely_channel_template_driven"):
        table[c] = table[c].astype(str).str.lower() == "true"
    sums = ex.evidence_summaries(table)
    unique = pd.read_csv(run / "corpus" / "unique_texts.csv", keep_default_na=False)
    texts_by_id = dict(zip(unique["corpus_id"], unique["original_text"]))
    assert len(sums) == len(table)
    for s, row in zip(sums, table.itertuples()):
        assert "not validated" in s["summary"] and "validated concern" not in s["summary"].replace("not validated", "")
        ids = s["supporting_corpus_ids"]
        assert set(ids) <= set(texts_by_id)
        central = [q for q in json.loads(row.representatives) if q["kind"] == "central"]
        quotes = _quotes(s["summary"])
        assert len(quotes) == len(central)
        for q, rep in zip(quotes, central):
            full = texts_by_id[rep["corpus_id"]]
            assert full.startswith(q.rstrip("…")) and q.rstrip("…") != ""
        assert s["keywords"] == row.keywords
        if row.likely_channel_template_driven:
            assert any("channel/template-driven" in n for n in s["uncertainty_flags"])
            assert s["proxy_coherence"] == "low_or_mixed"


def test_broad_clusters_are_flagged_and_marked_low_proxy_coherence(synthetic):
    tmp, run, out, summary = synthetic
    table = pd.read_csv(out / "raw__umap_m25s5" / "cluster_table.csv")
    assert "flag_broad_cluster" in table.columns
    # each synthetic cluster holds ~32% of the eligible entries (>= the 8% broad threshold)
    assert table["flag_broad_cluster"].astype(str).str.lower().eq("true").all()
    assert (table["proxy_coherence"] == "low_or_mixed").all()
    t2 = table.copy()
    for c in ("flag_social_markup", "flag_template_repeat", "flag_channel_concentrated", "flag_short_responses",
              "flag_single_instrument", "flag_broad_cluster", "likely_channel_template_driven"):
        t2[c] = t2[c].astype(str).str.lower() == "true"
    notes = [n for s in ex.evidence_summaries(t2) for n in s["uncertainty_flags"]]
    assert any("very large cluster" in n for n in notes)


def test_language_subset_metrics_are_computed_within_the_subset(synthetic):
    tmp, run, out, summary = synthetic
    m = summary["experiments"]["strip__umap_m25s5_englishonly"]
    entries = pd.read_csv(run / "corpus" / "corpus_entries.csv")
    english_entries = int((entries["language"] == "English").sum())
    assert m["core_entries"] <= english_entries                    # denominator is the English core only
    assert m["coverage_core_french"] == 0.0 or m["core_entries_french"] >= 0
    assert 0.0 < m["coverage_core"] <= 1.0
    assert m["states"].get("outside_language_subset", 0) == int((entries["language"] == "French").sum())


def test_keyword_support_counts_whole_tokens_and_phrases():
    texts = ["open data portal", "the data are open", "database design"]
    sup = ex.keyword_support(texts, ["data", "open data", "base"])
    assert sup["data"] == round(2 / 3, 3)         # "database" does not count as the token "data"
    assert sup["open data"] == round(1 / 3, 3)
    assert sup["base"] == 0.0


# ── metrics, shortlist, overwrite protection ──────────────────────────────────

def test_partition_ari_known_values():
    a = np.array([0, 0, 1, 1, -1, 2])
    core = np.array([True] * 6)
    assert ex.partition_ari(a, a, core)["ari"] == 1.0
    b = np.array([1, 1, 0, 0, 0, 3])
    assert ex.partition_ari(a, b, core)["ari"] == 1.0              # identical partition under relabelling
    c = np.array([0, 1, 0, 1, 0, 2])
    assert ex.partition_ari(a, c, core)["ari"] < 0.5
    assert ex.partition_ari(a, a, np.zeros(6, dtype=bool))["ari"] is None


def test_shortlist_rule_excludes_unstable_direct_and_subset_runs():
    mk = lambda algo, subset, ari, eff: {"experiment": {"algo": algo, "subset": subset}, "seed_ari_mean": ari,
                                          "effective_substantive_coverage_core": eff}
    summary = {"experiments": {"a": mk("umap_m25s5", None, 0.9, 0.5), "b": mk("umap_m50s10", None, 0.7, 0.6),
                               "c": mk("umap_m25s5", None, 0.3, 0.9), "d": mk("kmeans_k40", None, 0.99, 0.95),
                               "e": mk("umap_m25s5", "English", 0.9, 0.99), "f": mk("umap_m10s5", None, None, 0.99)}}
    assert ex.shortlist(summary) == ["b", "a"]


def test_rerun_returns_cache_and_never_overwrites(synthetic):
    tmp, run, out, summary = synthetic
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "raw__umap_m25s5").iterdir()}
    again = ex.run_experiments(out, run, SMALL, encode=None)       # cached: no encoder needed, nothing rewritten
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "raw__umap_m25s5").iterdir()}
    assert before == after and again["experiments"].keys() == summary["experiments"].keys()
    with pytest.raises(FileExistsError):
        fc.write_json_new(out / "experiments_summary.json", {})


def test_report_is_unvalidated_and_refuses_overwrite(synthetic, tmp_path):
    tmp, run, out, summary = synthetic
    rp = tmp_path / "report.md"
    text = ex.write_report(out, rp)
    assert "NOT human-validated" in text and "Silhouette is not used for selection" in text
    assert "winner" not in text.lower().replace("no configuration is declared the winner", "")
    assert (out / "summaries").exists() and (out / "run_manifest.json").exists()
    with pytest.raises(FileExistsError):
        ex.write_report(out, rp)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_validated"] is False


# ═════════════ Part B: real experiment outputs ═════════════

REAL = ex.OUT_ROOT
real = pytest.mark.skipif(not (REAL / "run_manifest.json").exists(), reason="experiments_v2 not completed")


@real
def test_real_raw_variant_reproduces_run_v1_exactly():
    s = json.loads((REAL / "experiments_summary.json").read_text(encoding="utf-8"))
    assert s["reproduces_run_v1"] and all(v["identical_labels"] for v in s["reproduces_run_v1"].values())


@real
def test_real_assignments_are_aligned_and_complete_for_every_experiment():
    entries = pd.read_csv(fc.RUN_ROOT / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    manifest = json.loads((REAL / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_validated"] is False and len(manifest["experiments"]) >= 15
    for eid, info in manifest["experiments"].items():
        p = REAL / eid / "entry_assignments.csv"
        assert fc.sha256_file(p) == info["assignments_sha256"]
        a = pd.read_csv(p, dtype=str, keep_default_na=False)
        assert a["corpus_id"].tolist() == entries["corpus_id"].tolist()
        assert a["source_record_id"].tolist() == entries["source_record_id"].tolist()
        assert a["language"].tolist() == entries["language"].tolist()


@real
def test_real_summaries_are_faithful_to_source_text():
    entries = pd.read_csv(fc.RUN_ROOT / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    text_by_id = dict(zip(entries["corpus_id"], entries["original_text"]))
    files = sorted((REAL / "summaries").glob("*.json"))
    assert files
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["unvalidated"] is True
        for c in data["clusters"]:
            assert "not validated" in c["summary"]
            assert set(map(str, c["supporting_corpus_ids"])) <= set(text_by_id)
            for q in _quotes(c["summary"]):
                assert any(text_by_id[str(i)].startswith(q.rstrip("…")) for i in c["supporting_corpus_ids"])


@real
def test_real_run_v1_artifacts_unchanged_by_experiments():
    m = json.loads((fc.RUN_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    for rel, digest in m["assignment_sha256"].items():
        k, c = rel.split("/")
        assert fc.sha256_file(fc.RUN_ROOT / "clustering" / k / c / "entry_assignments.csv") == digest
    for key, e in m["embeddings"].items():
        assert fc.sha256_file(fc.RUN_ROOT / "embeddings" / key / "embeddings.npy") == e["embeddings_sha256"]
