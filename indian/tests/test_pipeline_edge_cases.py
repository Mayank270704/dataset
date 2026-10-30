"""
Edge-case, error-handling, provenance and structural tests for the Indian pipeline.

Covers gaps in test_civis_retrieval.py and test_loksabha_and_search.py:
  - civis.py: BM25 empty/stop-word query, chunk_text size<=overlap error, metrics edge cases,
    minmax_fusion equal scores, rrf single ranker, first_sentence edge cases, sha_text
  - loksabha.py: clean_question empty/None/Devanagari, leakage_flags positive 8-gram,
    devanagari_share, answer_key None/empty
  - policy_search.py: _snippet edge cases, DISCLAIMER content
  - Artifact structural checks: annotation templates blank labels, weak labels provenance,
    report CSV columns, source separation invariant

    python -m pytest indian/tests/test_pipeline_edge_cases.py -q -p no:cacheprovider
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

IND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IND))
from retrieval import civis as cv  # noqa: E402
from retrieval import loksabha as ls  # noqa: E402

# ── civis.py: BM25 ──────────────────────────────────────────────────────────

def test_bm25_empty_query_returns_zeros():
    """A query whose terms are all English stop words produces an all-zero score vector."""
    bm = cv.BM25().fit(["river water allocation", "telecom spectrum fees"])
    # "the" and "a" are stop words for the CountVectorizer (stop_words="english")
    scores = bm.score("the a")
    assert scores.shape == (2,) and np.all(scores == 0)


def test_bm25_single_document_corpus():
    bm = cv.BM25().fit(["river water allocation"])
    s = bm.score("water")
    assert s.shape == (1,) and s[0] > 0
    assert bm.score("unseen_zzzz").shape == (1,) and bm.score("unseen_zzzz")[0] == 0


def test_bm25_all_empty_documents():
    """Empty strings produce an empty vocabulary. BM25 either raises ValueError (sklearn empty-vocab
    guard) or returns an all-zero score vector — both are acceptable; what must NOT happen is silent
    corruption (non-zero scores, wrong shape, or a non-ValueError exception)."""
    try:
        bm = cv.BM25().fit(["", "  "])
        s = bm.score("water")
        assert np.all(s == 0), "expected all-zero scores for empty-vocabulary BM25"
    except ValueError:
        pass  # sklearn empty-vocabulary guard fires — valid and expected


# ── civis.py: chunk_text ────────────────────────────────────────────────────

def test_chunk_text_raises_when_overlap_ge_size():
    with pytest.raises(ValueError):
        cv.chunk_text("some text", size=100, overlap=100)
    with pytest.raises(ValueError):
        cv.chunk_text("some text", size=50, overlap=60)


def test_chunk_text_exactly_size_chars_is_single_span():
    text = "x" * 100
    spans = cv.chunk_text(text, size=100, overlap=10)
    assert spans == [(0, 100)]


def test_chunk_text_two_chars_with_small_window():
    """Chunk text shorter than overlap-adjusted window still terminates."""
    spans = cv.chunk_text("ab", size=5, overlap=1)
    assert spans and spans[-1][1] == 2


# ── civis.py: doc_scores_from_chunks / rank_order ───────────────────────────

def test_doc_scores_single_chunk_per_doc():
    s = cv.doc_scores_from_chunks(np.array([0.3, 0.7]), np.array([0, 1]), 2)
    assert s.tolist() == [0.3, 0.7]


def test_rank_order_stable_on_ties():
    """Ties are broken by lower index first (lexsort secondary key = arange)."""
    order = cv.rank_order(np.array([1.0, 1.0, 1.0]))
    assert list(order) == [0, 1, 2]


# ── civis.py: minmax_fusion ─────────────────────────────────────────────────

def test_minmax_fusion_equal_a_scores_gives_zero_a_component():
    """When a is constant, mm(a)=0 everywhere; result is entirely from b."""
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([0.0, 0.5, 1.0])
    f = cv.minmax_fusion(a, b, alpha=0.5)
    assert f.tolist() == pytest.approx([0.0, 0.25, 0.5], abs=1e-6)


def test_minmax_fusion_both_equal():
    """Both constant: result is all zeros."""
    f = cv.minmax_fusion(np.array([3.0, 3.0]), np.array([7.0, 7.0]), alpha=0.5)
    assert np.all(f == 0)


# ── civis.py: rrf ───────────────────────────────────────────────────────────

def test_rrf_single_ranker():
    order = np.array([2, 0, 1])
    s = cv.rrf([order], n_docs=3, k=60)
    assert s[2] == pytest.approx(1 / 61) and s[0] == pytest.approx(1 / 62) and s[1] == pytest.approx(1 / 63)


# ── civis.py: metrics_from_ranks ────────────────────────────────────────────

def test_metrics_all_rank_one():
    m = cv.metrics_from_ranks(np.array([1, 1, 1]), n_boot=20)
    assert m["recall@1"] == 1.0 and m["recall@5"] == 1.0 and m["mrr@10"] == 1.0


def test_metrics_all_beyond_k():
    """Ranks all > 10: recall@10 = mrr@10 = ndcg@10 = 0."""
    m = cv.metrics_from_ranks(np.array([11, 15, 20]), n_boot=20)
    assert m["recall@10"] == 0.0 and m["mrr@10"] == 0.0 and m["ndcg@10"] == 0.0


def test_metrics_single_query():
    m = cv.metrics_from_ranks(np.array([3]), n_boot=10)
    assert m["queries"] == 1 and m["recall@1"] == 0.0 and m["recall@5"] == 1.0


# ── civis.py: first_sentence ────────────────────────────────────────────────

def test_first_sentence_short_text_no_boundary():
    """Text shorter than min_chars: returns the whole text (no sentence boundary found)."""
    t = "Short text."
    assert cv.first_sentence(t) == "Short text."


def test_first_sentence_respects_max_chars():
    long = "A" * 350 + ". Next sentence here."
    s = cv.first_sentence(long)
    assert len(s) <= 300


def test_first_sentence_no_punctuation():
    t = "This is a long text with no sentence ending at all and it keeps going and going here"
    s = cv.first_sentence(t)
    assert len(s) <= 300 and len(s) > 0


# ── civis.py: sha_text ──────────────────────────────────────────────────────

def test_sha_text_deterministic_and_distinct():
    h1 = cv.sha_text("hello")
    h2 = cv.sha_text("hello")
    h3 = cv.sha_text("Hello")
    assert h1 == h2 and h1 != h3 and len(h1) == 64


# ── civis.py: canonical_documents ───────────────────────────────────────────

def test_canonical_documents_all_empty():
    import pandas as pd
    df = pd.DataFrame({"civis_record_id": ["a", "b"], "split": ["train", "train"],
                       "title": ["T1", "T2"], "document_text": ["", "  "],
                       "reference_summary": ["s1", "s2"]})
    docs, mapping, excluded = cv.canonical_documents(df)
    assert docs == [] and mapping == {} and sorted(excluded) == ["a", "b"]


def test_canonical_documents_three_identical():
    import pandas as pd
    df = pd.DataFrame({"civis_record_id": ["a", "b", "c"], "split": ["train"] * 3,
                       "title": ["T1", "T2", "T3"], "document_text": ["same text"] * 3,
                       "reference_summary": ["s1", "s2", "s3"]})
    docs, mapping, excluded = cv.canonical_documents(df)
    assert len(docs) == 1 and excluded == [] and len(docs[0].member_ids) == 3
    assert mapping["a"] == mapping["b"] == mapping["c"] == "a"


# ── loksabha.py: clean_question ─────────────────────────────────────────────

def test_clean_question_empty_string():
    assert ls.clean_question("") == ""


def test_clean_question_none():
    assert ls.clean_question(None) == ""


def test_clean_question_devanagari_only_not_stripped():
    """Non-English text has no English boilerplate: cleaning must not destroy it."""
    q = "सरकार ने क्या उपाय किए हैं?"
    c = ls.clean_question(q, [])
    # must keep the Devanagari content
    assert "सरकार" in c


def test_clean_question_answer_marker_variants():
    """Both ## ANSWER and **ANSWER** (with surrounding text) must be cut."""
    for marker in ("## ANSWER", "**ANSWER**"):
        q = f"(a) what are the plans? {marker} the minister said this"
        c = ls.clean_question(q, [])
        assert "minister said this" not in c
        assert "plans" in c


def test_clean_question_removes_minister_line():
    q = "Will the Minister of Health be pleased to state: (a) whether hospitals are crowded?"
    c = ls.clean_question(q, [])
    assert "Minister of Health" not in c and "pleased to state" not in c
    assert "hospitals are crowded" in c


# ── loksabha.py: leakage_flags ──────────────────────────────────────────────

def test_leakage_flags_detects_8gram_overlap():
    """When question and answer share 8 consecutive words, the flag fires."""
    shared = "the government has allocated ten crore rupees for the scheme"
    q = "Will the minister explain? " + shared + " in the budget?"
    a = shared + " according to finance ministry records"
    f = ls.leakage_flags(q, a, n=8)
    assert f["q_shares_8gram_with_answer"]


def test_leakage_flags_no_overlap_short_answer():
    """An answer shorter than 8 words cannot produce an 8-gram overlap."""
    f = ls.leakage_flags("Will the government act?", "Yes.", n=8)
    assert not f["q_shares_8gram_with_answer"]


def test_leakage_flags_answer_head_in_question():
    """When the first 80 chars of the answer appear verbatim in the question, the flag fires.
    The check is exact string containment of the normalised answer head inside the question."""
    # answer head = first 80 chars after whitespace normalisation
    answer = "The Ministry of Finance has confirmed the allocation in the current year."
    head = answer[:80]  # "The Ministry of Finance has confirmed the allocation in the current year."
    question = "Will the minister state: " + head + " How many were affected?"
    f = ls.leakage_flags(question, answer)
    assert f["q_contains_answer_head"]


# ── loksabha.py: devanagari_share ───────────────────────────────────────────

def test_devanagari_share_latin_only():
    assert ls.devanagari_share("hello world") == 0.0


def test_devanagari_share_devanagari_only():
    # A string of Devanagari characters
    dev = "नमस्ते"
    share = ls.devanagari_share(dev)
    assert share == pytest.approx(1.0)


def test_devanagari_share_empty():
    assert ls.devanagari_share("") == 0.0


def test_devanagari_share_no_alpha():
    assert ls.devanagari_share("123 !!") == 0.0


# ── loksabha.py: answer_key ─────────────────────────────────────────────────

def test_answer_key_none_does_not_raise():
    h = ls.answer_key(None)
    assert isinstance(h, str) and len(h) == 64


def test_answer_key_empty():
    h = ls.answer_key("")
    assert isinstance(h, str) and len(h) == 64


def test_answer_key_consistent():
    assert ls.answer_key("  A B  ") == ls.answer_key("a b")


# ── policy_search.py: _snippet ──────────────────────────────────────────────

def test_snippet_empty_query_returns_start():
    from retrieval.policy_search import _snippet
    text = "This is a policy document about road safety rules."
    s = _snippet(text, "", width=30)
    assert s.startswith("This is a policy")


def test_snippet_query_match_shifts_window():
    from retrieval.policy_search import _snippet
    text = "a " * 50 + "motor vehicle registration" + " b " * 50
    s = _snippet(text, "motor vehicle", width=80)
    assert "motor vehicle" in s


def test_snippet_short_text_no_ellipsis():
    from retrieval.policy_search import _snippet
    text = "Short text."
    s = _snippet(text, "short", width=240)
    assert not s.endswith("…")


# ── policy_search.py: DISCLAIMER ────────────────────────────────────────────

def test_disclaimer_mentions_unverified_and_not_verified_answers():
    from retrieval.policy_search import DISCLAIMER
    assert "UNVERIFIED" in DISCLAIMER.upper() or "unverified" in DISCLAIMER.lower()
    assert "not verified" in DISCLAIMER.lower() or "candidate evidence" in DISCLAIMER.lower()


# ── artifact structure: annotation templates have blank label columns ────────

MYGOV_ANN = IND / "processed" / "mygov" / "annotation_v1" / "mygov_annotation_template_v1_annotator_A.csv"
LABEL_COLS_MYGOV = {"stance", "concern_type", "sentiment", "requested_action", "evidence_span"}

@pytest.mark.skipif(not MYGOV_ANN.is_file(), reason="mygov annotation not built")
def test_mygov_annotation_template_labels_are_blank():
    import pandas as pd
    df = pd.read_csv(MYGOV_ANN, dtype=str, keep_default_na=False)
    for col in LABEL_COLS_MYGOV:
        assert col in df.columns, f"column {col!r} missing from annotation template"
        non_blank = (df[col].str.strip() != "").sum()
        assert non_blank == 0, f"column {col!r} has {non_blank} pre-filled values (must be blank)"


TWEETS_ANN = IND / "processed" / "tweets" / "annotation_v1" / "tweets_pilot_annotation_template_v1_annotator_A.csv"
LABEL_COLS_TWEETS = {"sentiment", "sarcasm_or_irony", "target_entity"}

@pytest.mark.skipif(not TWEETS_ANN.is_file(), reason="tweets annotation not built")
def test_tweets_annotation_template_labels_are_blank():
    import pandas as pd
    df = pd.read_csv(TWEETS_ANN, dtype=str, keep_default_na=False)
    for col in LABEL_COLS_TWEETS:
        assert col in df.columns, f"column {col!r} missing"
        non_blank = (df[col].str.strip() != "").sum()
        assert non_blank == 0, f"column {col!r} has {non_blank} pre-filled values"


# ── artifact structure: weak labels carry provenance and are flagged ─────────

WEAK_PREDS = IND / "processed" / "tweets" / "weak_labels_sentiment140_v1" / "weak_predictions.csv"

@pytest.mark.skipif(not WEAK_PREDS.is_file(), reason="weak labels not built")
def test_weak_labels_have_provenance_column_and_not_ground_truth_flag():
    import pandas as pd
    df = pd.read_csv(WEAK_PREDS, dtype=str, keep_default_na=False)
    assert "label_source" in df.columns
    assert df["label_source"].str.contains("NOT_ground_truth", case=False).all(), \
        "Every weak-label row must carry a NOT_ground_truth marker in label_source"
    assert "weak_label" in df.columns and set(df["weak_label"]).issubset({"positive", "negative"})
    assert "confidence" in df.columns


@pytest.mark.skipif(not WEAK_PREDS.is_file(), reason="weak labels not built")
def test_weak_labels_do_not_use_reactions_as_labels():
    """Likes and retweets columns must not appear in the weak-label output."""
    import pandas as pd
    df = pd.read_csv(WEAK_PREDS, dtype=str, keep_default_na=False)
    for col in ("likes", "retweets", "Reactions"):
        assert col not in df.columns, f"engagement column {col!r} must not appear in weak labels"


# ── artifact structure: final report CSVs have expected columns ─────────────

QUALITY_CSV = IND / "reports" / "final_dataset_quality_summary.csv"
RESULTS_CSV = IND / "reports" / "final_experiment_results.csv"
QUALITY_REQUIRED = {"dataset", "records", "unique_ids", "labels", "licence_status"}
RESULTS_REQUIRED = {"experiment", "label_status"}


@pytest.mark.skipif(not QUALITY_CSV.is_file(), reason="quality report not generated")
def test_quality_report_has_required_columns_and_four_sources():
    import pandas as pd
    df = pd.read_csv(QUALITY_CSV, dtype=str, keep_default_na=False)
    for col in QUALITY_REQUIRED:
        assert col in df.columns, f"quality report missing column {col!r}"
    assert len(df) == 4, f"expected 4 source rows, got {len(df)}"
    # Licence status must say "unverified" for every row (no row may claim verified)
    assert df["licence_status"].str.contains("unverified", case=False).all(), \
        "Every source row must carry 'unverified' in licence_status"


@pytest.mark.skipif(not RESULTS_CSV.is_file(), reason="results report not generated")
def test_results_report_has_required_columns_and_status_rows():
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV, dtype=str, keep_default_na=False)
    for col in RESULTS_REQUIRED:
        assert col in df.columns
    # Supervised classifiers must be marked blocked, not completed
    classifiers = df[df["experiment"].str.contains("classifier", case=False, na=False)]
    assert len(classifiers) >= 2, "Expected at least 2 blocked classifier rows"
    assert classifiers["status"].str.contains("blocked|not run", case=False, na=False).all(), \
        "Classifier rows must be marked not run / blocked"


@pytest.mark.skipif(not RESULTS_CSV.is_file(), reason="results report not generated")
def test_results_report_no_supervised_metrics_for_blocked_classifiers():
    """Blocked classifier rows must not have filled recall@1 or mrr@10 values."""
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV, dtype=str, keep_default_na=False)
    classifiers = df[df["experiment"].str.contains("classifier", case=False, na=False)]
    if "recall@1" in df.columns:
        assert (classifiers["recall@1"].str.strip() == "").all() or classifiers["recall@1"].isna().all(), \
            "Blocked classifiers must have no recall@1 value"


# ── source separation: policy_search never returns mygov or tweets ───────────

LS_CORPUS = IND / "processed" / "loksabha" / "retrieval_pilot_v1" / "corpus.csv"
CIVIS_CHUNKS = IND / "processed" / "retrieval" / "civis_v1" / "chunks.csv"

@pytest.mark.skipif(not LS_CORPUS.is_file() or not CIVIS_CHUNKS.is_file(),
                    reason="indexes not built")
def test_policy_search_never_returns_mygov_or_tweets_source():
    from retrieval.policy_search import search
    results = search("data policy government india", top=5)
    # The only permitted sources are CIVIS and Lok Sabha Q&A
    allowed_sources = {"CIVIS", "Lok Sabha Q&A"}
    for src_key, rs in results.items():
        for r in rs:
            assert r.source in allowed_sources, \
                f"Unexpected source {r.source!r}; only CIVIS and Lok Sabha Q&A are indexed"
            assert "mygov" not in r.source.lower(), f"MyGov must not appear in search results: {r.source}"
            assert "tweet" not in r.source.lower(), f"Tweets must not appear in search results: {r.source}"


@pytest.mark.skipif(not LS_CORPUS.is_file() or not CIVIS_CHUNKS.is_file(),
                    reason="indexes not built")
def test_policy_search_result_ids_are_non_empty():
    from retrieval.policy_search import search
    results = search("motor vehicle licensing rules", top=3)
    for src, rs in results.items():
        for r in rs:
            assert r.record_id and r.record_id.strip(), f"{src} result has empty record_id"
            assert r.licence_status.startswith("unverified"), \
                f"{src} result must have unverified licence_status, got: {r.licence_status!r}"


# ── regression: clean_question ANSWER marker fix (search raw text, not stripped) ─

def test_regression_answer_marker_found_in_raw_text_not_stripped():
    """
    Regression for the bug where **ANSWER** was searched AFTER stripping '**', so the
    marker was missed. The fix searches the raw text. Verify the question is still cut.
    """
    q = "123. SHRI X:\n\nWill the Minister of Y be pleased to state: (a) whether rules apply? **ANSWER** **THE MINISTER** (a): The data shows growth."
    c = ls.clean_question(q, ["X"])
    assert "data shows growth" not in c, "Answer text leaked into the cleaned query"
    assert "rules apply" in c, "Question text was over-stripped"


def test_regression_answer_marker_hash_variant():
    """## ANSWER variant must also be caught in the raw text."""
    q = "500. SHRI Y:\n\nWill the Minister state: (a) how many permits? ## ANSWER THE MINISTER: 5,000 permits were issued."
    c = ls.clean_question(q, ["Y"])
    assert "5,000 permits" not in c
    assert "permits" in c or "how many" in c


# ── annotation pilot: fixed seed produces deterministic pilot set ────────────

MYGOV_MAN = IND / "processed" / "mygov" / "annotation_v1" / "manifest.json"

@pytest.mark.skipif(not MYGOV_MAN.is_file(), reason="mygov annotation not built")
def test_mygov_annotation_pilot_size_and_seed():
    man = json.loads(MYGOV_MAN.read_text(encoding="utf-8"))
    assert man["pilot_size"] == 50, f"pilot should be 50, got {man['pilot_size']}"
    assert man["seed"] == 42
    assert "source_sha256" in man


# ── paired statistical tests artifact ───────────────────────────────────────

PAIRED_TESTS = IND / "processed" / "loksabha" / "retrieval_pilot_v1" / "evaluation" / "paired_tests.json"

@pytest.mark.skipif(not PAIRED_TESTS.is_file(), reason="paired tests not run")
def test_paired_tests_hybrid_minmax_significantly_better_than_bm25():
    """hybrid_minmax must be significantly better than BM25 on Lok Sabha pilot by both tests."""
    d = json.loads(PAIRED_TESTS.read_text(encoding="utf-8"))
    mm = next(c for c in d["comparisons"] if c["method"] == "hybrid_minmax_all-MiniLM-L6-v2")
    assert mm["wilcoxon_mrr10"]["significant_alpha005"] == "yes", \
        "Wilcoxon test: hybrid_minmax must be significantly better than BM25 on MRR@10"
    assert mm["mcnemar_r1"]["significant_alpha005"] == "yes", \
        "McNemar test: hybrid_minmax R@1 advantage must be significant"
    # The improvement must go in the right direction
    assert mm["wilcoxon_mrr10"]["pairs_where_method_wins"] > mm["wilcoxon_mrr10"]["pairs_where_bm25_wins"], \
        "hybrid_minmax must win more discordant MRR@10 pairs than BM25"


CIVIS_PAIRED = IND / "processed" / "retrieval" / "civis_v1" / "evaluation" / "paired_tests.json"

@pytest.mark.skipif(not CIVIS_PAIRED.is_file(), reason="CIVIS paired tests not run")
def test_civis_paired_tests_no_method_dominating_on_recall1():
    """On CIVIS title queries, no method achieves a significant R@1 advantage over BM25 (McNemar)
    because n=166 lacks power. A marginal MRR@10 signal (Wilcoxon) for hybrid_minmax is expected."""
    d = json.loads(CIVIS_PAIRED.read_text(encoding="utf-8"))
    title_comps = [c for c in d["comparisons"] if c["query_type"] == "title"]
    for c in title_comps:
        # McNemar R@1: no method should claim strong significance
        assert c["mcnemar_r1"]["significant_alpha005"] == "no", \
            f"{c['method']} shows significant R@1 advantage on 166-query CIVIS — check this finding"


@pytest.mark.skipif(not PAIRED_TESTS.is_file(), reason="paired tests not run")
def test_paired_tests_dense_not_better_than_bm25():
    """Dense MiniLM is not significantly better than BM25 — it is significantly worse (truncation artefact)."""
    d = json.loads(PAIRED_TESTS.read_text(encoding="utf-8"))
    dense = next(c for c in d["comparisons"] if c["method"] == "dense_all-MiniLM-L6-v2")
    assert dense["wilcoxon_mrr10"]["significant_alpha005"] == "no", \
        "Dense MiniLM must NOT be significantly better than BM25 in MRR@10"
    # Dense wins fewer non-tied pairs than BM25 (truncation effect)
    assert dense["wilcoxon_mrr10"]["pairs_where_method_wins"] < dense["wilcoxon_mrr10"]["pairs_where_bm25_wins"], \
        "BM25 should win more discordant pairs than dense MiniLM (truncation artefact)"


TWEETS_MAN = IND / "processed" / "tweets" / "annotation_v1" / "manifest.json"

@pytest.mark.skipif(not TWEETS_MAN.is_file(), reason="tweets annotation not built")
def test_tweets_annotation_manifest_locked_test_and_pilot():
    man = json.loads(TWEETS_MAN.read_text(encoding="utf-8"))
    assert man["locked_test"] == 300, f"locked test set should be 300, got {man['locked_test']}"
    assert man["pilot"] == 200
    assert man["seed"] == 42
    # Confirm pilot and test are drawn from separate splits
    import pandas as pd
    plan = pd.read_csv(IND / "processed" / "tweets" / "annotation_v1" / "sampling_plan_1500.csv",
                       dtype=str, keep_default_na=False)
    test_ids = set(plan.loc[plan.study_split == "locked_test_300", "tweet_row_id"])
    pilot_ids = set(plan.loc[plan.annotation_batch == "pilot_200", "tweet_row_id"])
    assert test_ids.isdisjoint(pilot_ids), "Pilot and locked test set must not overlap"
