"""
models/concern_discovery/duplicates.py
======================================
Explicit handling of exact duplicates, near-duplicates and repeated campaign/template text.

Two unique texts are linked when EITHER
  * their template key is identical (URLs, @mentions, #hashtags, punctuation and case removed), OR
  * one is among the other's k nearest neighbours with embedding cosine >= ``cos_threshold`` AND
    character 5-gram Jaccard >= ``jaccard_threshold``. The lexical confirmation keeps translations
    and paraphrases with the same meaning OUT of duplicate groups.
Linked texts are merged by union-find into near-duplicate groups. A group whose entry count
(including exact repeats) is >= ``campaign_min_entries`` is a *repeated template group*.

Identical or near-identical text does not show that the texts came from the same person.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

DEFAULTS: dict[str, Any] = {"cos_threshold": 0.92, "jaccard_threshold": 0.5, "containment_threshold": 0.9,
                            "k_neighbours": 10, "shingle_n": 5, "campaign_min_entries": 5}


def strip_social(text: str) -> str:
    """Remove URLs, @mentions and #hashtags; collapse whitespace (same rule as experiments v2)."""
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def template_key(text: str) -> str:
    """Key under which texts count as the same template: social markup, punctuation, case, spacing removed."""
    t = unicodedata.normalize("NFC", strip_social(text)).casefold().replace("’", "'")
    t = _PUNCT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def char_shingles(text: str, n: int = 5) -> frozenset[str]:
    t = template_key(text)
    if len(t) <= n:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def containment(a: frozenset[str], b: frozenset[str], min_shingles: int = 36) -> float:
    """Share of the SHORTER text's shingles found in the longer one (quoted/extended copies of a template).

    Returns 0 when the shorter text has fewer than ``min_shingles`` shingles (about 40 characters), so that
    short generic phrases are not linked merely because a longer text contains them.
    """
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    if len(small) < min_shingles:
        return 0.0
    return len(small & big) / len(small)


def topk_cosine(emb: np.ndarray, k: int, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k neighbours by cosine (rows L2-normalised), excluding self; sorted by similarity."""
    n = emb.shape[0]
    k = max(1, min(k, n - 1))
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        s = emb[i0:i1] @ emb.T
        s[np.arange(i1 - i0), np.arange(i0, i1)] = -np.inf
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        idx[i0:i1] = np.take_along_axis(part, order, axis=1)
        sim[i0:i1] = np.take_along_axis(ps, order, axis=1)
    return idx, sim


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(texts: list[str], emb: np.ndarray, multiplicity: np.ndarray | None = None,
                          cos_threshold: float = DEFAULTS["cos_threshold"],
                          jaccard_threshold: float = DEFAULTS["jaccard_threshold"],
                          containment_threshold: float = DEFAULTS["containment_threshold"],
                          k_neighbours: int = DEFAULTS["k_neighbours"], shingle_n: int = DEFAULTS["shingle_n"],
                          campaign_min_entries: int = DEFAULTS["campaign_min_entries"]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-unique-text group table, edge table).

    Group ids are the smallest unique index in the group, so they are deterministic and order-independent
    of the neighbour search.
    """
    n = len(texts)
    if emb.shape[0] != n:
        raise ValueError("texts and embeddings are not aligned")
    mult = np.ones(n, dtype=int) if multiplicity is None else np.asarray(multiplicity, dtype=int)
    if len(mult) != n:
        raise ValueError("multiplicity is not aligned with texts")
    uf = _UnionFind(n)
    edges: list[dict[str, Any]] = []
    keys = [template_key(t) for t in texts]
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if not k:
            continue
        j = first.setdefault(k, i)
        if j != i:
            uf.union(i, j)
            edges.append({"a": j, "b": i, "kind": "same_template_key", "cosine": np.nan, "jaccard": 1.0})
    if n > 1:
        nn_idx, nn_sim = topk_cosine(np.ascontiguousarray(emb, dtype=np.float32), k_neighbours)
        shingles: dict[int, frozenset[str]] = {}
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            for j, s in zip(nn_idx[i], nn_sim[i]):
                if s < cos_threshold:
                    break
                j = int(j)
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen or (keys[a] and keys[a] == keys[b]):
                    continue
                seen.add((a, b))
                sa = shingles.setdefault(a, char_shingles(texts[a], shingle_n))
                sb = shingles.setdefault(b, char_shingles(texts[b], shingle_n))
                jac = jaccard(sa, sb)
                if jac >= jaccard_threshold or containment(sa, sb) >= containment_threshold:
                    uf.union(a, b)
                    edges.append({"a": a, "b": b, "kind": "near_duplicate", "cosine": round(float(s), 4),
                                  "jaccard": round(jac, 4), "containment": round(containment(sa, sb), 4)})
    group = np.array([uf.find(i) for i in range(n)], dtype=int)
    edge_df = pd.DataFrame(edges, columns=["a", "b", "kind", "cosine", "jaccard", "containment"])
    g = pd.DataFrame({"unique_id": np.arange(n), "dup_group_id": group, "multiplicity": mult, "template_key": keys})
    g["group_unique_texts"] = g.groupby("dup_group_id")["unique_id"].transform("size")
    g["group_entries"] = g.groupby("dup_group_id")["multiplicity"].transform("sum")
    g["is_exact_repeat"] = g["multiplicity"] > 1
    g["in_near_dup_group"] = g["group_unique_texts"] > 1
    g["repeated_template_group"] = g["group_entries"] >= campaign_min_entries
    return g, edge_df


def summarise_groups(groups: pd.DataFrame) -> dict[str, Any]:
    gg = groups.drop_duplicates("dup_group_id")
    return {
        "unique_texts": int(len(groups)),
        "groups": int(len(gg)),
        "unique_texts_in_multi_text_groups": int(groups["in_near_dup_group"].sum()),
        "multi_text_groups": int((gg["group_unique_texts"] > 1).sum()),
        "unique_texts_with_exact_repeats": int(groups["is_exact_repeat"].sum()),
        "repeated_template_groups": int(gg["repeated_template_group"].sum()),
        "entries_in_repeated_template_groups": int(groups.loc[groups["repeated_template_group"], "multiplicity"].sum()),
        "largest_group_entries": int(gg["group_entries"].max()) if len(gg) else 0,
    }
