"""
models/concern_clustering/language_keywords.py
==============================================
Per-language sub-cluster keyword extraction with the EXISTING c-TF-IDF implementation
(models/concern_clustering/clustering.py: _ctfidf_matrix / extract_cluster_topics). No new algorithm.

Unit       one (cluster, language) subgroup: the unique texts of an existing concern cluster whose source-metadata
           language is English or French. Cluster assignments are read, never changed.
Rule       a subgroup is eligible if it has at least MIN_INDEPENDENT_TEXTS independent texts (distinct near-duplicate
           groups). The value 15 is the existing concern-discovery minimum for a whole cluster
           (models/concern_discovery/diagnostics.py THRESHOLDS["undersized_min_distinct_groups"]). Other/Unknown
           language texts are never eligible.
Method     for each language separately, every eligible subgroup is one c-TF-IDF class document (its unique texts
           joined), with the existing parameters. Terms are therefore distinctive relative to other subgroups of the
           SAME language. Selection is identical to extract_cluster_topics: descending c-TF-IDF weight, top N,
           positive weight only.
Meaning    keyword lists describe frequent distinctive wording. They are not validated policy concerns and not
           evidence of public importance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from models.concern_clustering.clustering import (
    _FRENCH_STOPS,
    CTFIDF_MAX_FEATURES,
    CTFIDF_NGRAM_RANGE,
    CTFIDF_TOP_N_WORDS,
    _ctfidf_matrix,
    extract_cluster_topics,
)
from models.concern_discovery.diagnostics import THRESHOLDS

LANGUAGES = ("English", "French")
MIN_INDEPENDENT_TEXTS = int(THRESHOLDS["undersized_min_distinct_groups"])     # existing project threshold (15)


def language_group(lang: str) -> str:
    return lang if lang in LANGUAGES else "Other/Unknown"


def subgroup_inventory(unique: pd.DataFrame, labels: np.ndarray, dup_group: np.ndarray,
                       min_independent: int = MIN_INDEPENDENT_TEXTS) -> pd.DataFrame:
    """One row per (cluster, language) subgroup present, plus English/French subgroups that are empty."""
    if len(unique) != len(labels) or len(unique) != len(dup_group):
        raise ValueError("unique texts, labels and near-duplicate groups are not aligned")
    d = pd.DataFrame({"cluster_id": np.asarray(labels, dtype=int), "language": unique["language"].map(language_group).to_numpy(),
                      "dup_group_id": np.asarray(dup_group), "multiplicity": unique["multiplicity"].astype(int).to_numpy()})
    d = d[d["cluster_id"] >= 0]
    inv = d.groupby(["cluster_id", "language"]).agg(unique_texts=("dup_group_id", "size"),
                                                     independent_texts=("dup_group_id", "nunique"),
                                                     entries=("multiplicity", "sum")).reset_index()
    clusters = sorted(d["cluster_id"].unique())
    full = pd.MultiIndex.from_product([clusters, list(LANGUAGES)], names=["cluster_id", "language"])
    inv = inv.set_index(["cluster_id", "language"])
    inv = inv.reindex(full.union(inv.index), fill_value=0).reset_index()
    inv["eligible"] = inv["language"].isin(LANGUAGES) & (inv["independent_texts"] >= min_independent)
    inv["exclusion_reason"] = np.where(
        inv["eligible"], "",
        np.where(~inv["language"].isin(LANGUAGES), "language Other/Unknown is never eligible",
                 np.where(inv["independent_texts"] == 0, "no texts in this language",
                          f"fewer than {min_independent} independent texts")))
    return inv.sort_values(["cluster_id", "language"]).reset_index(drop=True)


def ctfidf_analyzer(ngram_range: tuple[int, int] = CTFIDF_NGRAM_RANGE):
    """The term analyzer of _ctfidf_matrix (same stop words, token pattern and n-grams), applied to single texts.

    Stop words are removed BEFORE n-grams are formed, so a bigram such as "difficile trouver" can come from
    "difficile de trouver"; support must therefore be measured with this analyzer, not by substring search.
    """
    stops = list(ENGLISH_STOP_WORDS.union(set(_FRENCH_STOPS)))
    return CountVectorizer(ngram_range=ngram_range, stop_words=stops, token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{2,}\b").build_analyzer()


def _support(term_sets: list[set[str]], dup_ids: list[Any], term: str) -> tuple[float, int]:
    """Share of subgroup unique texts whose analysed terms include the term, and independent texts containing it."""
    hit = [term in ts for ts in term_sets]
    return round(float(np.mean(hit)) if hit else 0.0, 4), int(len({g for g, h in zip(dup_ids, hit) if h}))


def language_keywords(texts: list[str], unique: pd.DataFrame, labels: np.ndarray, dup_group: np.ndarray,
                      inventory: pd.DataFrame, top_n: int = CTFIDF_TOP_N_WORDS,
                      max_features: int = CTFIDF_MAX_FEATURES,
                      ngram_range: tuple[int, int] = CTFIDF_NGRAM_RANGE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ranked keywords per eligible subgroup, computed within each language. Returns (keywords, equivalence check)."""
    labels = np.asarray(labels, dtype=int)
    lang = unique["language"].map(language_group).to_numpy()
    analyze = ctfidf_analyzer(ngram_range)
    rows, check = [], {}
    for L in LANGUAGES:
        elig = sorted(inventory.loc[(inventory["language"] == L) & inventory["eligible"], "cluster_id"].astype(int))
        if not elig:
            check[L] = {"subgroups": 0, "identical_to_extract_cluster_topics": None}
            continue
        sub_labels = np.where((lang == L) & np.isin(labels, elig), labels, -1)
        docs = [" ".join(texts[i] for i in np.where(sub_labels == c)[0]) for c in elig]
        ctfidf, vocab = _ctfidf_matrix(docs, max_features=max_features, ngram_range=ngram_range)
        meta = pd.DataFrame({"original_text": texts, "language": lang})
        ref = extract_cluster_topics(meta, sub_labels, top_n=top_n, max_features=max_features, ngram_range=ngram_range)
        ref_kw = {int(r.cluster_id): [k for k in r.top_keywords.split(", ") if k] for r in ref.itertuples() if r.cluster_id != -1}
        same = True
        for k, c in enumerate(elig):
            idx = np.where(sub_labels == c)[0]
            sub_terms = [set(analyze(texts[i])) for i in idx]
            sub_dups = [dup_group[i] for i in idx]
            top = np.argsort(ctfidf[k])[::-1][:top_n] if ctfidf.shape[1] else []
            terms = [(vocab[i], float(ctfidf[k, i])) for i in top if ctfidf[k, i] > 0]
            same &= [t for t, _ in terms] == ref_kw.get(c, [])
            for rank, (t, w) in enumerate(terms, 1):
                share, n_indep = _support(sub_terms, sub_dups, t)
                rows.append({"cluster_id": c, "language": L, "rank": rank, "term": t, "ctfidf_weight": round(w, 6),
                             "support_share_unique_texts": share, "support_independent_texts": n_indep,
                             "single_text_term": n_indep == 1,
                             # occurs in no single text: an n-gram formed across two joined texts by the existing
                             # class-document construction (texts joined with a space); kept for equivalence, flagged
                             "spans_text_boundary": n_indep == 0})
        check[L] = {"subgroups": len(elig), "vocabulary_size": len(vocab), "identical_to_extract_cluster_topics": bool(same)}
    cols = ["cluster_id", "language", "rank", "term", "ctfidf_weight", "support_share_unique_texts",
            "support_independent_texts", "single_text_term", "spans_text_boundary"]
    return pd.DataFrame(rows, columns=cols), check


def compare_with_cluster_keywords(kw: pd.DataFrame, cluster_keywords: dict[int, list[str]]) -> pd.DataFrame:
    """Per cluster: overlap of each language's top terms with the existing all-language keywords, and EN/FR overlap."""
    out = []
    for c in sorted(set(kw["cluster_id"])):
        k = kw[kw["cluster_id"] == c]
        en = set(k.loc[k["language"] == "English", "term"])
        fr = set(k.loc[k["language"] == "French", "term"])
        base = set(cluster_keywords.get(int(c), []))
        out.append({"cluster_id": int(c), "english_terms": len(en), "french_terms": len(fr),
                    "english_overlap_with_cluster_keywords": len(en & base),
                    "french_overlap_with_cluster_keywords": len(fr & base) if fr else None,
                    "english_french_shared_terms": ", ".join(sorted(en & fr)) if fr else None})
    return pd.DataFrame(out)
