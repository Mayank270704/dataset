"""
models/concern_discovery/summaries.py
=====================================
Deterministic, extractive, evidence-backed cluster descriptions.

No generative model is used. A free local summariser was looked for: the Hugging Face cache holds
only a ``config.json`` for ``facebook/bart-large-cnn`` (no weights), and an English news summariser
would in any case be unsuitable for short bilingual consultation comments. Every sentence of a
description is therefore assembled from computed fields:

* term support  = share of member comments containing the term (recomputed and verified);
* quotations    = verbatim substrings of a cited member comment (verified character for character);
* citations     = ``corpus_id`` values of cluster members, linked to source dataset, record id and field.

``verify_grounding`` re-checks every quotation, citation and support figure against the corpus and
raises on any violation, so a fabricated quote or a non-member citation cannot be written.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from models.concern_discovery.diagnostics import keyword_support, lang_group

MIN_TERM_SUPPORT = 0.10
MAX_QUOTE_CHARS = 220
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def select_representatives(members: np.ndarray, emb: np.ndarray, dup_group: np.ndarray, texts: list[str],
                           languages: list[str], k: int = 5, lam: float = 0.7, min_chars: int = 40,
                           min_lang_share: float = 0.15, deprioritise: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Maximal-marginal-relevance selection of representative members.

    relevance = cosine to the cluster centroid; redundancy = max cosine to already selected examples.
    At most one example per near-duplicate group (so a template is shown once), texts shorter than
    ``min_chars`` are used only if not enough longer texts exist, members flagged in ``deprioritise`` (unique-level
    boolean, e.g. repeated-template or markup-only texts) are likewise used only as a fallback, and a language
    holding >= ``min_lang_share`` of members is guaranteed one example when it would otherwise be missing.
    """
    members = np.asarray(members)
    if len(members) == 0:
        return []
    cen = emb[members].mean(axis=0)
    cen /= max(np.linalg.norm(cen), 1e-12)
    rel = emb[members] @ cen
    long_enough = np.array([len(texts[i].strip()) >= min_chars for i in members])
    if deprioritise is not None:
        long_enough &= ~np.asarray(deprioritise, dtype=bool)[members]
    pool = list(np.where(long_enough)[0]) if long_enough.sum() >= min(k, len(members)) else list(range(len(members)))
    chosen: list[int] = []
    used_groups: set[int] = set()
    reasons: dict[int, str] = {}

    def mmr_pick(cands: list[int]) -> int | None:
        best, best_score = None, -np.inf
        for c in cands:
            if c in chosen or int(dup_group[members[c]]) in used_groups:
                continue
            red = max((float(emb[members[c]] @ emb[members[s]]) for s in chosen), default=0.0)
            score = lam * float(rel[c]) - (1 - lam) * red
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and best is not None and members[c] < members[best]):
                best, best_score = c, score
        return best

    lang_arr = np.array([lang_group(languages[i]) for i in members])
    shares = pd.Series(lang_arr).value_counts(normalize=True)
    needed = [lg for lg, s in shares.items() if s >= min_lang_share and (lang_arr == lg).sum() >= 3]
    while len(chosen) < min(k, len(members)):
        missing = [lg for lg in needed if lg not in {lang_arr[c] for c in chosen}]
        slots_left = min(k, len(members)) - len(chosen)
        if missing and slots_left <= len(missing):
            c = mmr_pick([p for p in pool if lang_arr[p] == missing[0]])
            if c is None:
                c = mmr_pick([i for i in range(len(members)) if lang_arr[i] == missing[0]])
            if c is None:                                   # language only present in excluded duplicates
                needed.remove(missing[0])
                continue
            if c is not None:
                reasons[c] = f"language coverage ({missing[0]})"
        else:
            c = mmr_pick(pool)
            if c is None:                                   # pool exhausted: allow shorter texts
                c = mmr_pick(list(range(len(members))))
            if c is not None:
                reasons[c] = "most central" if not chosen else "central and non-redundant (MMR)"
        if c is None:
            break
        chosen.append(c)
        used_groups.add(int(dup_group[members[c]]))
    out = []
    for rank, c in enumerate(chosen, 1):
        red = max((float(emb[members[c]] @ emb[members[s]]) for s in chosen if s != c), default=None)
        out.append({"rank": rank, "unique_id": int(members[c]), "cosine_to_centroid": round(float(rel[c]), 4),
                    "max_cosine_to_other_examples": None if red is None else round(red, 4),
                    "selection_reason": reasons.get(c, "")})
    return out


def quotable_span(text: str, max_chars: int = MAX_QUOTE_CHARS) -> str:
    """A verbatim span: the whole text if short, else its first sentence(s) up to max_chars, else a word-boundary prefix."""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    span = ""
    for s in SENT_SPLIT.split(t):
        cand = (span + " " + s).strip() if span else s
        if len(cand) > max_chars:
            break
        span = cand
    if span and t.startswith(span):
        return span
    cut = t[:max_chars]
    return cut[:cut.rfind(" ")] if " " in cut else cut


class CorpusSupport:
    """Cached share of ALL clustered-variant texts containing a term (the corpus base rate)."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.cache: dict[str, float] = {}

    def __call__(self, term: str) -> float:
        if term not in self.cache:
            self.cache[term] = keyword_support(self.texts, [term]).get(term, 0.0)
        return self.cache[term]


def describe_cluster(row: dict[str, Any], member_texts: list[str], reps: list[dict[str, Any]],
                     unique: pd.DataFrame, corpus_support: CorpusSupport | None = None) -> dict[str, Any]:
    """Build the extractive description for one assessed cluster (``row`` from diagnostics.assess_clusters)."""
    tier = row["tier"]
    support = keyword_support(member_texts, list(row["keywords"])[:10])
    terms = [(k, s) for k, s in support.items() if s >= MIN_TERM_SUPPORT]
    # drop terms contained in an already selected longer phrase with similar support
    chosen_terms: list[tuple[str, float]] = []
    for k, s in sorted(terms, key=lambda x: -x[1]):
        if any(k in c or c in k for c, _ in chosen_terms):
            continue
        chosen_terms.append((k, s))
        if len(chosen_terms) == 4:
            break
    evidence = []
    for r in reps:
        u = unique.iloc[r["unique_id"]]
        evidence.append({**r, "corpus_id": int(u["corpus_id"]), "source_dataset": u["source_dataset"],
                         "source_record_id": u["source_record_id"], "question_slug": u["question_slug"],
                         "language": u["language"], "multiplicity": int(u["multiplicity"]),
                         "quote": quotable_span(u["original_text"]), "full_text": u["original_text"]})
    lead = evidence[0] if evidence else None
    body = None
    if tier == "unreliable":
        description = None
        note = "Not summarised as a concern: " + ", ".join(row["flags"]) + "."
    else:
        if chosen_terms:
            if corpus_support is not None:
                term_txt = ", ".join(f"“{k}” ({round(100 * s)}%; corpus {round(100 * corpus_support(k))}%)"
                                     for k, s in chosen_terms)
            else:
                term_txt = ", ".join(f"“{k}” ({round(100 * s)}%)" for k, s in chosen_terms)
            body = (f"Member comments most often mention {term_txt} (share of the {row['n_unique_texts']} distinct member "
                    "texts containing each term" + ("; corpus = share of all texts in the clustering population" if corpus_support is not None else "") + ").")
        else:
            body = (f"No single term occurs in at least {round(100 * MIN_TERM_SUPPORT)}% of the {row['n_unique_texts']} "
                    "distinct member texts; the shared theme is not lexically explicit.")
        if lead:
            body += f" Most representative comment [corpus_id {lead['corpus_id']}]: “{lead['quote']}”."
        prefix = {"candidate": "Candidate concern (unvalidated). ",
                  "needs_review": "Candidate concern with quality warnings (unvalidated). ",
                  "campaign_or_template": (f"Campaign/template-driven group (unvalidated): its largest repeated text accounts "
                                           f"for {round(100 * row['top_dup_group_share'])}% of entries and the group has only "
                                           f"{row['distinct_near_dup_groups']} distinct texts. "),
                  "non_concern_pattern": "Response pattern, not a policy concern (unvalidated). "}[tier]
        description = prefix + body
        note = None
    short_label = " / ".join(k for k, _ in chosen_terms[:2]) if chosen_terms and tier != "unreliable" else None
    return {
        "config_id": row["config_id"], "cluster_id": int(row["cluster_id"]), "tier": tier, "flags": list(row["flags"]),
        "n_entries": int(row["n_entries"]), "n_unique_texts": int(row["n_unique_texts"]),
        "distinct_near_dup_groups": int(row["distinct_near_dup_groups"]),
        "language_counts": {"English": int(row["english"]), "French": int(row["french"]), "Other/Unknown": int(row["other_unknown"])},
        "short_label": short_label, "description": description, "description_body": body,
        "not_summarised_reason": note,
        "term_support": {} if tier == "unreliable" else {k: s for k, s in chosen_terms}, "all_keyword_support": support,
        "term_corpus_support": {} if tier == "unreliable" or corpus_support is None else {k: corpus_support(k) for k, _ in chosen_terms},
        "summary_method": "deterministic_extractive_v1", "human_validated": False,
        "evidence": evidence,
    }


class GroundingError(AssertionError):
    pass


def verify_grounding(summary: dict[str, Any], member_corpus_ids: set[int], text_by_corpus_id: dict[int, str],
                     member_texts: list[str], tol: float = 1e-9, corpus_texts: list[str] | None = None) -> list[str]:
    """Return a list of violations (empty = grounded). Checks quotes, citations and term-support figures."""
    problems: list[str] = []
    for ev in summary["evidence"]:
        cid = ev["corpus_id"]
        if cid not in member_corpus_ids:
            problems.append(f"cluster {summary['cluster_id']}: cited corpus_id {cid} is not a member")
            continue
        src = text_by_corpus_id.get(cid)
        if src is None:
            problems.append(f"cluster {summary['cluster_id']}: corpus_id {cid} not found in corpus")
        elif ev["quote"] not in src or ev["full_text"] != src:
            problems.append(f"cluster {summary['cluster_id']}: quote for corpus_id {cid} is not verbatim")
    desc = summary.get("description") or ""
    for m in re.finditer(r"\[corpus_id (\d+)\]: “(.*?)”\.", desc, flags=re.S):
        cid, quote = int(m.group(1)), m.group(2)
        if cid not in member_corpus_ids or quote not in text_by_corpus_id.get(cid, ""):
            problems.append(f"cluster {summary['cluster_id']}: description quote [corpus_id {cid}] is not grounded")
    recomputed = keyword_support(member_texts, list(summary["term_support"]))
    for k, s in summary["term_support"].items():
        if abs(recomputed.get(k, -1) - s) > tol:
            problems.append(f"cluster {summary['cluster_id']}: support for '{k}' is {recomputed.get(k)} not {s}")
        if f"“{k}” ({round(100 * s)}%" not in desc:
            problems.append(f"cluster {summary['cluster_id']}: term '{k}' support not stated as computed")
    base = summary.get("term_corpus_support") or {}
    if base and corpus_texts is not None:
        rec = keyword_support(corpus_texts, list(base))
        for k, s in base.items():
            if abs(rec.get(k, -1) - s) > tol or f"corpus {round(100 * s)}%" not in desc:
                problems.append(f"cluster {summary['cluster_id']}: corpus share for '{k}' is not grounded")
    return problems
