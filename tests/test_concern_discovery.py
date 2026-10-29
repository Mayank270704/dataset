"""
tests/test_concern_discovery.py
===============================
Core logic, edge cases and failure handling of models/concern_discovery and
scripts/run_concern_discovery.py. Synthetic data only, except one read-only check of the real
run_v1 inputs (skipped when absent). Reviewer judgments created here are test fixtures, never results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.concern_discovery import diagnostics as dg  # noqa: E402
from models.concern_discovery import duplicates as dup  # noqa: E402
from models.concern_discovery import review as rv  # noqa: E402
from models.concern_discovery import summaries as sm  # noqa: E402
from models.concern_discovery.schema import CorpusIntegrityError, validate_corpus, validate_labels  # noqa: E402


# ── synthetic corpus helpers ──────────────────────────────────────────────────

def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_corpus(texts, langs=None, mult=None, ds="consultation_compilation_2017_18"):
    n = len(texts)
    langs = langs or ["English"] * n
    mult = mult or [1] * n
    unique = pd.DataFrame({"unique_id": range(n), "corpus_id": 0, "multiplicity": mult, "source_dataset": ds,
                           "source_record_id": [str(i + 1) for i in range(n)], "question_slug": "comment_question_or_idea",
                           "language": langs, "original_text": texts})
    rows, cid = [], 0
    for u in range(n):
        for _ in range(mult[u]):
            rows.append({"corpus_id": cid, "unique_id": u, "source_dataset": ds, "source_record_id": str(u + 1),
                         "source_row_number": str(u + 1), "source_field": "comment", "question_slug": "comment_question_or_idea",
                         "language": langs[u], "stakeholder_group": "Twitter" if "#" in texts[u] else "Slido Text",
                         "duplicate_text_group_id": "", "text_key": texts[u].casefold(), "original_text": texts[u]})
            cid += 1
    entries = pd.DataFrame(rows)
    unique["corpus_id"] = entries.groupby("unique_id")["corpus_id"].min().to_numpy()
    return entries, unique


def blob_embeddings(labels, dim=16, seed=0, spread=0.15):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(max(labels) + 2, 2), dim))
    return unit(np.array([centers[l if l >= 0 else -1] + spread * rng.normal(size=dim) for l in labels]))


# ── schema ────────────────────────────────────────────────────────────────────

class TestSchema:
    def setup_method(self):
        self.entries, self.unique = make_corpus(["alpha text one", "beta text two", "gamma three"], mult=[2, 1, 1])
        self.emb = unit(np.eye(3, 8))

    def test_valid_corpus_passes(self):
        log = validate_corpus(self.entries.astype(str), self.unique.astype(str), self.emb)
        assert all(c["status"] in ("PASSED", "WARN") for c in log.checks)

    def test_missing_column_fails(self):
        with pytest.raises(CorpusIntegrityError, match="schema"):
            validate_corpus(self.entries.drop(columns=["language"]).astype(str), self.unique.astype(str), self.emb)

    def test_misaligned_embeddings_fail(self):
        with pytest.raises(CorpusIntegrityError, match="alignment"):
            validate_corpus(self.entries.astype(str), self.unique.astype(str), self.emb[:2])

    def test_non_normalised_embeddings_fail(self):
        with pytest.raises(CorpusIntegrityError, match="normalisation"):
            validate_corpus(self.entries.astype(str), self.unique.astype(str), self.emb * 2)

    def test_multiplicity_mismatch_fails(self):
        u = self.unique.copy()
        u.loc[0, "multiplicity"] = 5
        with pytest.raises(CorpusIntegrityError, match="multiplicity"):
            validate_corpus(self.entries.astype(str), u.astype(str), self.emb)

    def test_text_substitution_fails(self):
        e = self.entries.copy()
        e.loc[1, "original_text"] = "something else entirely"
        with pytest.raises(CorpusIntegrityError, match="text identity"):
            validate_corpus(e.astype(str), self.unique.astype(str), self.emb)

    def test_duplicate_corpus_id_fails(self):
        e = self.entries.copy()
        e.loc[1, "corpus_id"] = 0
        with pytest.raises(CorpusIntegrityError, match="corpus_id"):
            validate_corpus(e.astype(str), self.unique.astype(str), self.emb)

    def test_manifest_count_change_fails(self):
        with pytest.raises(CorpusIntegrityError, match="manifest"):
            validate_corpus(self.entries.astype(str), self.unique.astype(str), self.emb, {"entries": 999, "unique_texts": 3})

    def test_labels_unknown_state_code_fails(self):
        with pytest.raises(CorpusIntegrityError):
            validate_labels(np.array([0, -7, 1]), 3, "x")
        with pytest.raises(CorpusIntegrityError):
            validate_labels(np.array([0, 1]), 3, "x")
        validate_labels(np.array([0, -1, -2]), 3, "x")


# ── duplicates ────────────────────────────────────────────────────────────────

class TestDuplicates:
    def test_template_key_ignores_markup_case_punctuation(self):
        a = dup.template_key("Support the REGISTRY! https://t.co/abc #opengov @TBS")
        b = dup.template_key("support the registry https://t.co/xyz")
        assert a == b == "support the registry"

    def test_strip_social_matches_experiments_v2(self):
        v2 = pytest.importorskip("run_clustering_experiments_v2")
        for t in ["Hi @a see https://x.y/z #tag now", "plain", "www.site.ca and #x"]:
            assert dup.strip_social(t) == v2.strip_social(t)

    def test_near_duplicates_require_lexical_overlap(self):
        texts = ["I did not know it existed", "I didn't know it existed", "Lack of transparency in spending",
                 "Not enough openness about how money is spent"]
        emb = unit([[1, 0, 0], [0.99, 0.05, 0], [0, 1, 0], [0, 0.995, 0.05]])
        g, edges = dup.near_duplicate_groups(texts, emb, cos_threshold=0.9)
        assert g.loc[0, "dup_group_id"] == g.loc[1, "dup_group_id"] == 0          # lexical + semantic -> grouped
        assert g.loc[2, "dup_group_id"] != g.loc[3, "dup_group_id"]               # paraphrase only -> separate
        assert set(edges["kind"]) == {"near_duplicate"}

    def test_repeated_template_counts_multiplicity(self):
        texts = ["Create a beneficial ownership registry now", "create a beneficial ownership registry now!", "other topic here"]
        emb = unit([[1, 0], [1, 0.01], [0, 1]])
        g, _ = dup.near_duplicate_groups(texts, emb, multiplicity=np.array([3, 2, 1]), campaign_min_entries=5)
        assert g.loc[0, "group_entries"] == 5 and bool(g.loc[1, "repeated_template_group"])
        assert not bool(g.loc[2, "repeated_template_group"])
        s = dup.summarise_groups(g)
        assert s["repeated_template_groups"] == 1 and s["entries_in_repeated_template_groups"] == 5

    def test_group_ids_deterministic_and_misalignment_rejected(self):
        texts = ["same text here", "Same text here.", "different"]
        emb = unit([[1, 0], [1, 0], [0, 1]])
        g1, _ = dup.near_duplicate_groups(texts, emb)
        g2, _ = dup.near_duplicate_groups(texts, emb)
        assert g1["dup_group_id"].tolist() == g2["dup_group_id"].tolist() == [0, 0, 2]
        with pytest.raises(ValueError):
            dup.near_duplicate_groups(texts, emb[:2])


# ── diagnostics ───────────────────────────────────────────────────────────────

class TestDiagnostics:
    def test_cramers_v_bounds(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 4, 2000)
        assert dg.cramers_v_corrected(a, a) > 0.99
        assert dg.cramers_v_corrected(a, rng.integers(0, 3, 2000)) < 0.05
        assert dg.cramers_v_corrected(a, np.zeros(2000)) == 0.0

    def test_association_permutation_null(self):
        rng = np.random.default_rng(1)
        lab = rng.integers(0, 5, 600)
        r = dg.association(lab, np.where(lab < 2, "French", "English"), n_perm=50)
        assert r["cramers_v"] > r["null_v_p95"] and r["perm_p"] <= 0.05
        r0 = dg.association(lab, rng.choice(["a", "b"], 600), n_perm=50)
        assert r0["perm_p"] > 0.05
        assert dg.association(lab, np.array(["a"] * 600))["cramers_v"] is None

    def test_stability_invariant_to_relabeling_and_ignores_excluded(self):
        ref = np.array([0, 0, 0, 1, 1, 1, -1, -2])
        alt = np.array([5, 5, 5, 3, 3, 3, -1, 3])               # -2 in ref: excluded, must not count
        s = dg.cluster_stability(ref, [alt])
        assert s["stability_jaccard_mean"].tolist() == [1.0, 1.0]
        split = np.array([0, 0, 1, 2, 2, 2, -1, -2])
        s2 = dg.cluster_stability(ref, [split])
        assert s2.set_index("cluster_id").loc[0, "stability_jaccard_mean"] == pytest.approx(2 / 3, abs=1e-3)

    def test_geometry_margin_detects_misplaced_point(self):
        labels = np.array([0] * 5 + [1] * 5)
        emb = blob_embeddings(labels.tolist(), spread=0.05)
        emb[4] = emb[7]                                          # member of 0 sitting inside cluster 1
        geom, pts = dg.cluster_geometry(emb, labels)
        assert pts.set_index("unique_id").loc[4, "margin"] < 0
        assert geom.set_index("cluster_id").loc[0, "negative_margin_share"] == pytest.approx(0.2)
        assert dg.simplified_silhouette(pts) is not None

    def _assess(self, texts, labels, alts=None, markup=True, mult=None):
        entries, unique = make_corpus(texts, mult=mult)
        emb = blob_embeddings(labels, seed=3, spread=0.08)
        # cos_threshold > 1 disables embedding links: these unit tests isolate tier logic from duplicate detection
        groups, _ = dup.near_duplicate_groups(texts, emb, unique["multiplicity"].to_numpy(), cos_threshold=1.1)
        kw = {c: ["policy", "data"] for c in set(labels) if c >= 0}
        return dg.assess_clusters("t", np.array(labels), entries, unique, emb, groups, kw,
                                  alts if alts is not None else [np.array(labels)], None, None, markup)

    def test_tiers(self):
        words = ["housing", "transit", "budget", "clinics", "schools", "rivers", "farms", "ports", "courts", "museums",
                 "parks", "roads", "grants", "audits", "pensions", "permits", "libraries", "bridges", "forests", "tariffs"]
        subst = [f"{w} policy data should be published with {words[(i + 3) % 20]} figures" for i, w in enumerate(words)]
        noop = [f"I haven't looked at the {w} pages, {words[(i + 7) % 20]} maybe later" for i, w in enumerate(words)]
        campaign = ["We want a public registry of beneficial owners now"] + [f"{w} tax policy data note" for w in words[:19]]
        tiny = [f"{w} policy data on cannabis" for w in words[:10]]
        texts = subst + noop + campaign + tiny
        labels = [0] * 20 + [1] * 20 + [2] * 20 + [3] * 10
        mult = [1] * 40 + [12] + [1] * 19 + [1] * 10
        t, _ = self._assess(texts, labels, mult=mult)
        tier = t.set_index("cluster_id")["tier"]
        assert tier[1] == "non_concern_pattern"
        assert tier[2] == "campaign_or_template"
        assert tier[3] == "unreliable"                           # 10 distinct texts < 15
        assert tier[0] in ("candidate", "needs_review")

    def test_unstable_cluster_is_unreliable(self):
        texts = [f"policy data comment number {i} on housing" for i in range(40)]
        labels = [0] * 20 + [1] * 20
        alt = np.array([0] * 5 + [1] * 35)                     # cluster 0: best Jaccard 15/40 = 0.375 -> unstable
        t, _ = self._assess(texts, labels, alts=[alt])
        t = t.set_index("cluster_id")
        assert t.loc[0, "flag_unstable"] and t.loc[0, "tier"] == "unreliable"
        assert t.loc[1, "stability_jaccard_mean"] == pytest.approx(20 / 35, abs=1e-3)
        assert t.loc[1, "flag_moderately_unstable"] and not t.loc[1, "flag_unstable"]

    def test_markup_driven_only_when_markup_was_embedded(self):
        texts = [f"Join #opengovcan discussion {i} on policy data https://t.co/x{i}" for i in range(20)]
        labels = [0] * 20
        t_raw, _ = self._assess(texts, labels, markup=True)
        t_strip, _ = self._assess(texts, labels, markup=False)
        assert t_raw.loc[0, "flag_markup_driven"] and t_raw.loc[0, "tier"] == "unreliable"
        assert not t_strip.loc[0, "flag_markup_driven"]

    def test_partition_summary_separates_noise_and_excluded(self):
        texts = [f"policy data text {i} about cities" for i in range(20)] + ["x" * 25, "y" * 25]
        labels = [0] * 20 + [-1, -2]
        entries, unique = make_corpus(texts)
        t, _ = self._assess(texts, labels)
        p = dg.partition_summary(t, np.array(labels), entries, np.ones(len(texts), bool))
        assert p["noise_entries"] == 1 and p["excluded_by_preprocessing"] == 1
        assert p["noise_share_of_population"] == pytest.approx(1 / 21, abs=1e-4)


# ── summaries ─────────────────────────────────────────────────────────────────

class TestSummaries:
    def setup_method(self):
        self.texts = ["Protect personal privacy in open data releases please",
                      "Protect personal privacy in open data releases please!",       # near-duplicate of 0
                      "Privacy of citizens must come first when sharing data",
                      "short privacy",
                      "La vie privée des citoyens doit être protégée avant tout partage",
                      "Les données personnelles doivent rester confidentielles et protégées",
                      "Les renseignements personnels méritent une protection renforcée",
                      "Security and privacy safeguards for government datasets are essential"]
        self.langs = ["English"] * 4 + ["French"] * 3 + ["English"]
        self.entries, self.unique = make_corpus(self.texts, self.langs)
        rng = np.random.default_rng(0)
        base = rng.normal(size=12)
        self.emb = unit(np.array([base + 0.3 * rng.normal(size=12) for _ in self.texts]))
        self.emb[1] = self.emb[0]
        g, _ = dup.near_duplicate_groups(self.texts, self.emb)
        self.groups = g["dup_group_id"].to_numpy()

    def test_representatives_dedup_length_language(self):
        reps = sm.select_representatives(np.arange(8), self.emb, self.groups, self.texts, self.langs, k=4)
        ids = [r["unique_id"] for r in reps]
        assert not ({0, 1} <= set(ids))                          # one per near-duplicate group
        assert 3 not in ids                                      # short text avoided
        assert any(self.langs[i] == "French" for i in ids)       # French holds >= 15%
        dep = np.zeros(8, bool)
        dep[[0, 1]] = True
        ids2 = [r["unique_id"] for r in sm.select_representatives(np.arange(8), self.emb, self.groups, self.texts, self.langs,
                                                                   k=3, deprioritise=dep)]
        assert not ({0, 1} & set(ids2))
        assert sm.select_representatives(np.array([], dtype=int), self.emb, self.groups, self.texts, self.langs) == []

    def test_quotable_span_is_verbatim(self):
        long = "First sentence about privacy. " * 20
        q = sm.quotable_span(long, 100)
        assert q in long and len(q) <= 100
        assert sm.quotable_span("short") == "short"
        nosplit = "word " * 100
        assert sm.quotable_span(nosplit, 50) in nosplit

    def _summary(self, tier="candidate"):
        row = {"config_id": "t", "cluster_id": 7, "tier": tier, "flags": [] if tier == "candidate" else ["unstable"],
               "n_entries": 8, "n_unique_texts": 8, "distinct_near_dup_groups": 7, "english": 5, "french": 3,
               "other_unknown": 0, "top_dup_group_share": 0.25, "keywords": ["privacy", "data", "personal", "protect"]}
        reps = sm.select_representatives(np.arange(8), self.emb, self.groups, self.texts, self.langs)
        base = sm.CorpusSupport(self.texts + ["unrelated text"] * 10)
        return sm.describe_cluster(row, self.texts, reps, self.unique, base)

    def test_description_grounded(self):
        s = self._summary()
        cids = set(self.entries["corpus_id"])
        by = dict(zip(self.entries["corpus_id"], self.entries["original_text"]))
        corpus = self.texts + ["unrelated text"] * 10
        assert sm.verify_grounding(s, cids, by, self.texts, corpus_texts=corpus) == []
        assert s["description"].startswith("Candidate concern (unvalidated).")
        assert "privacy" in s["term_support"] and s["human_validated"] is False

    def test_grounding_detects_fabrication(self):
        cids = set(self.entries["corpus_id"])
        by = dict(zip(self.entries["corpus_id"], self.entries["original_text"]))
        s = self._summary()
        s["evidence"][0]["quote"] = "A quotation nobody wrote"
        assert any("not verbatim" in p for p in sm.verify_grounding(s, cids, by, self.texts))
        s = self._summary()
        assert any("not a member" in p for p in sm.verify_grounding(s, cids - {s["evidence"][0]["corpus_id"]}, by, self.texts))
        s = self._summary()
        k = next(iter(s["term_support"]))
        s["term_support"][k] = 0.99
        assert any("support" in p for p in sm.verify_grounding(s, cids, by, self.texts))
        s = self._summary()
        s["description"] = s["description"].replace(s["evidence"][0]["quote"], "Invented words")
        assert any("not grounded" in p for p in sm.verify_grounding(s, cids, by, self.texts))

    def test_unreliable_cluster_not_summarised(self):
        s = self._summary("unreliable")
        assert s["description"] is None and s["term_support"] == {} and "unstable" in s["not_summarised_reason"]


# ── review package & analysis ─────────────────────────────────────────────────

def _review_fixture(seed=5):
    rng = np.random.default_rng(0)
    n_clusters, per = 10, 12
    labels = np.repeat(np.arange(n_clusters), per)
    texts = [f"comment {i} about topic {labels[i]} with details" for i in range(len(labels))]
    entries, unique = make_corpus(texts)
    emb = blob_embeddings(labels.tolist(), seed=1, spread=0.05)
    tiers = ["candidate"] * 6 + ["needs_review", "campaign_or_template", "non_concern_pattern", "unreliable"]
    table = pd.DataFrame({"cluster_id": range(n_clusters), "tier": tiers})
    summaries = {c: {"description_body": f"Member comments most often mention “topic” (100%). Most representative comment [corpus_id {c * per}]: “{texts[c * per]}”."}
                 for c in range(n_clusters) if tiers[c] != "unreliable"}
    groups = np.arange(len(labels))
    a, ak, b, bk = rv.build_review_items(table, labels, unique, emb, groups, summaries, seed, n_suitable=6, n_controls=2)
    _ = rng
    return a, ak, b, bk


class TestReview:
    def test_items_deterministic_and_blind(self):
        a1, ak1, b1, _ = _review_fixture()
        a2, ak2, _, _ = _review_fixture()
        pd.testing.assert_frame_equal(a1, a2)
        assert set(ak1["stratum"]) == {"suitable", "control"}
        assert (ak1["intruder_cluster_id"] != ak1["cluster_id"]).all()
        assert not b1["description"].str.contains("corpus_id").any()
        for r in ak1.itertuples():                              # intruder position points at the intruder text
            row = a1.set_index("item_id").loc[r.item_id]
            assert f"topic {r.intruder_cluster_id} " in row[f"comment_{r.intruder_position}"]

    def test_leakage_checks(self):
        a, ak, _, _ = _review_fixture()
        t = rv.reviewer_template(a, rv.A_FILL, 1)
        assert rv.check_no_leakage(t, ak, rv.A_FILL) == []
        assert any("forbidden" in p for p in rv.check_no_leakage(t.assign(cluster_id=1), ak, rv.A_FILL))
        t2 = t.copy()
        t2.loc[0, "intruder_position"] = "3"
        assert any("pre-filled" in p for p in rv.check_no_leakage(t2, ak, rv.A_FILL))
        t3 = t.copy()
        t3["item_id"] = [f"cl{i}" for i in range(len(t3))]
        assert rv.check_no_leakage(t3, ak, rv.A_FILL)

    def test_package_and_analysis(self, tmp_path):
        a, ak, b, bk = _review_fixture()
        pkg = tmp_path / "human_review"
        m = rv.write_package(pkg, a, ak, b, bk, 9, {"configuration": "test"})
        assert m["leakage_check"] == "PASSED"
        with pytest.raises(FileExistsError):
            rv.write_package(pkg, a, ak, b, bk, 9, {})
        t1 = pd.read_csv(pkg / "reviewer_1_task_a.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        t2 = pd.read_csv(pkg / "reviewer_2_task_a.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert t1["item_id"].tolist() != t2["item_id"].tolist() and set(t1["item_id"]) == set(t2["item_id"])

        out = tmp_path / "analysis"
        st = rv.analyze_reviews(pkg, out)
        assert st["status"] == "BLOCKED" and len(st["missing"]) == 4 and not out.exists()

        key = ak.set_index("item_id")
        (pkg / "completed").mkdir()
        for rid, correct in (("reviewer_1", True), ("reviewer_2", False)):
            ta = pd.read_csv(pkg / f"{rid}_task_a.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
            ta["reviewer_id"] = rid
            ta["intruder_position"] = [str(key.loc[i, "intruder_position"]) if correct else
                                       str(key.loc[i, "intruder_position"] % 6 + 1) for i in ta["item_id"]]
            ta["coherence_of_others"] = ["3" if key.loc[i, "stratum"] == "suitable" else "0" for i in ta["item_id"]]
            ta["confidence"] = "High"
            tb = pd.read_csv(pkg / f"{rid}_task_b.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
            tb["reviewer_id"], tb["summary_support"], tb["misleading_claim"], tb["confidence"] = rid, "2", "no", "low"
            ta.to_csv(pkg / "completed" / f"{rid}_task_a.csv", index=False, encoding="utf-8-sig")
            tb.to_csv(pkg / "completed" / f"{rid}_task_b.csv", index=False, encoding="utf-8-sig")
        res = rv.analyze_reviews(pkg, out)
        assert res["status"] == "COMPLETE"
        assert res["per_reviewer"]["reviewer_1"]["intrusion_accuracy"] == 1.0
        assert res["per_reviewer"]["reviewer_2"]["intrusion_accuracy"] == 0.0
        assert res["agreement_task_a"]["coherence_quadratic_kappa"] == pytest.approx(1.0)
        assert (out / "review_metrics.json").exists()
        with pytest.raises(FileExistsError):
            rv.analyze_reviews(pkg, out)

    def test_invalid_or_tampered_completed_files(self, tmp_path):
        a, ak, b, bk = _review_fixture()
        pkg = tmp_path / "hr"
        rv.write_package(pkg, a, ak, b, bk, 9, {})
        (pkg / "completed").mkdir()
        for rid in rv.REVIEWERS:
            ta = pd.read_csv(pkg / f"{rid}_task_a.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
            ta["reviewer_id"], ta["intruder_position"], ta["coherence_of_others"], ta["confidence"] = rid, "1", "2", "high"
            tb = pd.read_csv(pkg / f"{rid}_task_b.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
            tb["reviewer_id"], tb["summary_support"], tb["misleading_claim"], tb["confidence"] = rid, "1", "no", "high"
            if rid == "reviewer_2":
                ta.loc[0, "comment_1"] = "edited text"
                ta.loc[1, "coherence_of_others"] = ""                # blank judgment is never converted
            ta.to_csv(pkg / "completed" / f"{rid}_task_a.csv", index=False, encoding="utf-8-sig")
            tb.to_csv(pkg / "completed" / f"{rid}_task_b.csv", index=False, encoding="utf-8-sig")
        res = rv.analyze_reviews(pkg, tmp_path / "out")
        assert res["status"] == "INVALID"
        assert any("modified" in p for p in res["problems"]) and any("invalid values" in p for p in res["problems"])
        assert not (tmp_path / "out").exists()

    def test_wilson(self):
        lo, hi = rv.wilson(8, 10)
        assert 0 < lo < 0.8 < hi < 1
        assert rv.wilson(0, 0) == (None, None)


# ── end-to-end pipeline on a synthetic artifact tree ──────────────────────────

def _build_tree(root: Path):
    import hashlib
    rng = np.random.default_rng(0)
    topics = ["privacy of personal data", "electoral reform and voting", "open data standards",
              "indigenous reconciliation", "tax spending waste"]
    texts, labels = [], []
    subjects = ["Families", "Seniors", "Students", "Farmers", "Nurses", "Veterans", "Workers", "Parents", "Artists",
                "Doctors", "Cities", "Teachers", "Retirees", "Newcomers", "Youth", "Scientists", "Drivers", "Owners",
                "Tenants", "Volunteers"]
    verbs = ["worry about", "keep asking for", "want progress on", "are frustrated with", "demand clarity on"]
    for c, t in enumerate(topics):
        for i in range(20):
            texts.append(f"{subjects[i]} {verbs[(i + c) % 5]} {t}")
            labels.append(c)
    for i in range(10):
        texts.append(f"Random unrelated remark number {i} xyz{i}")
        labels.append(-1)
    texts.append("#opengovcan https://t.co/abc")          # content-free after stripping
    labels.append(-2)
    langs = ["French" if i % 9 == 0 else "English" for i in range(len(texts))]
    entries, unique = make_corpus(texts, langs)
    emb = blob_embeddings([l if l >= 0 else 5 for l in labels], seed=2, spread=0.4)   # within-topic cosine < 0.92
    run_v1, exp = root / "run_v1", root / "experiments_v2"
    (run_v1 / "corpus").mkdir(parents=True)
    (run_v1 / "embeddings" / "multilingual").mkdir(parents=True)
    entries.to_csv(run_v1 / "corpus" / "corpus_entries.csv", index=False)
    unique.to_csv(run_v1 / "corpus" / "unique_texts.csv", index=False)
    np.save(run_v1 / "embeddings" / "multilingual" / "embeddings.npy", emb)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    man = {"entries": len(entries), "unique_texts": len(unique),
           "by_dataset": {"consultation_compilation_2017_18": len(entries)},
           "corpus_entries_sha256": sha(run_v1 / "corpus" / "corpus_entries.csv"),
           "unique_texts_sha256": sha(run_v1 / "corpus" / "unique_texts.csv")}
    (run_v1 / "corpus" / "corpus_manifest.json").write_text(json.dumps(man))
    raw = unique["original_text"].tolist()
    changed = np.array([i for i, t in enumerate(raw) if dup.strip_social(t) != t and len(dup.strip_social(t)) >= 20], dtype=int)
    (exp / "preprocess").mkdir(parents=True)
    np.savez(exp / "preprocess" / "strip_embeddings.npz", idx=changed, emb=emb[changed])
    lab = np.array(labels)
    raw_lab = np.where(lab == -2, -1, lab)
    for cfg in ("raw__umap_m25s5", "raw__umap_m50s10", "strip__umap_m25s5", "strip__umap_m50s10"):
        d = exp / cfg
        d.mkdir()
        L = raw_lab if cfg.startswith("raw") else lab
        for s in (42, 43, 44):
            np.save(d / f"labels_seed{s}.npy", L)
        kw = {0: "privacy, personal, data", 1: "electoral, reform, voting", 2: "open, data, standards",
              3: "indigenous, reconciliation", 4: "tax, spending, waste"}
        pd.DataFrame({"cluster_id": list(kw), "keywords": list(kw.values())}).to_csv(d / "cluster_table.csv", index=False)
        (d / "metrics.json").write_text(json.dumps({"seed_ari_mean": 1.0 if "m25" in cfg else 0.99, "seed_ari_min": 1.0}))
    _ = rng
    return run_v1, exp


class TestPipeline:
    def test_end_to_end_synthetic(self, tmp_path):
        import run_concern_discovery as rcd
        run_v1, exp = _build_tree(tmp_path)
        out = tmp_path / "cd"
        r = rcd.stage_run(out, run_v1, exp)
        assert r["selected"] in rcd.CANDIDATE_CONFIGS
        man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert man["human_validated"] is False and "PASSED" in man["grounding_verification"]
        ea = pd.read_csv(out / "selected" / "entry_assignments.csv")
        assert len(ea) == 111 and set(ea["state"]) >= {"clustered", "noise"}
        for col in ("corpus_id", "source_dataset", "source_record_id", "source_field", "question_slug", "language"):
            assert col in ea.columns
        reps = pd.read_csv(out / "configurations" / r["selected"] / "representative_examples.csv")
        texts = pd.read_csv(run_v1 / "corpus" / "corpus_entries.csv").set_index("corpus_id")["original_text"]
        assert all(q in texts[c] for q, c in zip(reps["quote"], reps["corpus_id"]))
        with pytest.raises(FileExistsError):
            rcd.stage_run(out, run_v1, exp)
        tiers = pd.read_csv(out / "configurations" / r["selected"] / "cluster_diagnostics.csv")["tier"]
        assert (tiers != "unreliable").any()
        m = rcd.stage_review_package(out, run_v1, exp)
        assert m["leakage_check"] == "PASSED" and m["task_a_items"] > 0 and m["task_b_items"] > 0
        assert rcd.stage_analyze_reviews(out)["status"] == "BLOCKED"
        txt = rcd.stage_report(out, tmp_path / "report.md")
        assert "NOT human-validated" in txt and "BLOCKED" in txt and "MCA" in txt

    def test_tampered_corpus_is_rejected(self, tmp_path):
        import run_concern_discovery as rcd
        run_v1, exp = _build_tree(tmp_path)
        p = run_v1 / "corpus" / "corpus_entries.csv"
        p.write_text(p.read_text(encoding="utf-8").replace("Families", "Households"), encoding="utf-8")
        with pytest.raises(CorpusIntegrityError, match="hash"):
            rcd.stage_run(tmp_path / "cd", run_v1, exp)

    def test_stale_strip_cache_is_rejected(self, tmp_path):
        import run_concern_discovery as rcd
        run_v1, exp = _build_tree(tmp_path)
        z = np.load(exp / "preprocess" / "strip_embeddings.npz")
        np.savez(exp / "preprocess" / "strip_embeddings.npz", idx=np.array([0]), emb=z["emb"][:1] if len(z["emb"]) else np.zeros((1, 16)))
        with pytest.raises(CorpusIntegrityError, match="stripped-embedding"):
            rcd.stage_run(tmp_path / "cd", run_v1, exp)


@pytest.mark.skipif(not (ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_manifest.json").exists(),
                    reason="real run_v1 artifacts not present")
def test_real_inputs_pass_integrity_checks():
    import run_concern_discovery as rcd
    d = rcd.load_inputs()
    assert len(d["entries"]) == 10108 and len(d["unique"]) == 10043
    assert all(c["status"] != "FAILED" for c in d["log"].checks)


# ── research-evaluation integration ───────────────────────────────────────────

def test_research_evaluation_section(tmp_path):
    import run_concern_discovery as rcd
    import run_research_evaluation as ev
    assert ev.check_concern_discovery(tmp_path / "missing")[0]["status"] == "NOT RUN"
    run_v1, exp = _build_tree(tmp_path)
    out = tmp_path / "cd"
    rcd.stage_run(out, run_v1, exp)
    res = {r["name"]: r["status"] for r in ev.check_concern_discovery(out)}
    assert res["concern discovery outputs match run_manifest.json"] == "PASSED"
    assert res["concern-cluster human review"] == "BLOCKED"
    rcd.stage_review_package(out, run_v1, exp)
    res = {r["name"]: r["status"] for r in ev.check_concern_discovery(out)}
    assert res["blank reviewer templates unchanged"] == "PASSED" and res["concern-cluster human review"] == "BLOCKED"
    (out / "selected" / "concern_groups.csv").write_text("tampered", encoding="utf-8")
    res = {r["name"]: r["status"] for r in ev.check_concern_discovery(out)}
    assert res["concern discovery outputs match run_manifest.json"] == "FAILED"
