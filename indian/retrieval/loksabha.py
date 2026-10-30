"""
indian/retrieval/loksabha.py
============================
Lok Sabha parliamentary Q&A helpers for retrieval: memory-safe Parquet access, eligibility and quality flags,
question cleaning for queries, and answer-leakage checks. Records are parliamentary questions and ministerial
answers, not consultation comments. Text is never rewritten; legacy-font Hindi is only flagged.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

LICENCE_STATUS = "unverified: no licence or dataset card present locally; local experiment only, do not redistribute"
META_COLS = ["qa_id", "lok_sabha", "session", "question_no", "question_type", "date", "subject", "ministry", "member_count",
             "pdf_url", "question_empty", "answer_empty", "legacy_hindi_font_artifacts"]
ANSWER_MARKER = re.compile(r"(?:\*\*|##|^|\n)\s*(?:A\s*N\s*S\s*W\s*E\s*R|उत्‍?तर)\b", re.I)
PLEASED = re.compile(r"be\s+pleased\s+to\s+state\s*[:\-–]*", re.I)
DEV = re.compile(r"[ऀ-ॿ]")


def read_metadata(parquet: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(parquet, columns=META_COLS).to_pandas()


def read_texts(parquet: Path, ids: set[str], cols=("qa_id", "members", "question_text", "answer_text")) -> pd.DataFrame:
    """Read text columns only for `ids`, one row group at a time (bounded memory)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import pyarrow as pa
    f = pq.ParquetFile(parquet)
    want = pa.array(sorted(ids))
    parts = []
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=list(cols))
        t = t.filter(pc.is_in(t["qa_id"], value_set=want))
        if t.num_rows:
            parts.append(t.to_pandas())
        del t
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cols))


def devanagari_share(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(bool(DEV.match(c)) for c in letters) / len(letters) if letters else 0.0


def clean_question(q: str, members: list[str] | None = None) -> str:
    """Query text: drop markdown, question number, member names and 'Will the Minister of X be pleased to state'."""
    t = q or ""
    m = ANSWER_MARKER.search(t)                               # search the RAW text: markers are written as **ANSWER** / ## ANSWER
    if m:
        t = t[: m.start()]                                    # never let an embedded answer reach the query
    t = re.sub(r"[#*]+", " ", t)
    p = PLEASED.search(t)
    if p:
        t = t[p.end():]
    else:
        t = re.sub(r"^\s*\d+\s*\.\s*[^:]{0,300}:", " ", t)    # leading 'NNN. MEMBER NAMES:'
    for name in members or []:
        if name and len(name) > 3:
            t = re.sub(re.escape(name), " ", t, flags=re.I)
    return re.sub(r"^[\s\-–:.]+", "", re.sub(r"\s+", " ", t)).strip()


def answer_key(a: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", (a or "")).strip().lower().encode("utf-8")).hexdigest()


def leakage_flags(question: str, answer: str, n: int = 8) -> dict:
    """Does the raw question text contain the answer, or share a long word n-gram with it?"""
    q, a = (question or ""), (answer or "")
    head = re.sub(r"\s+", " ", a).strip()[:80]
    marker = bool(ANSWER_MARKER.search(q))
    contains_head = bool(head) and head in re.sub(r"\s+", " ", q)
    qw, aw = re.findall(r"\w+", q.lower()), re.findall(r"\w+", a.lower())
    grams = {tuple(qw[i:i + n]) for i in range(len(qw) - n + 1)}
    shared = any(tuple(aw[i:i + n]) in grams for i in range(len(aw) - n + 1)) if grams else False
    return {"q_has_answer_marker": marker, "q_contains_answer_head": contains_head, f"q_shares_{n}gram_with_answer": shared}
