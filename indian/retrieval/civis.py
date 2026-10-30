"""
indian/retrieval/civis.py
=========================
Minimal CIVIS policy-document retrieval: canonical documents, character chunking, BM25 (scikit-learn sparse),
dense retrieval with a locally cached encoder, hybrid fusion, and known-item evaluation.

CIVIS records are policy/consultation DOCUMENTS with reference summaries of unverified origin. They are not
citizen comments, and summaries are never treated as submissions.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

LICENCE_STATUS = "unverified: no licence or dataset card present locally; local experiment only, do not redistribute"
SUMMARY_NOTE = "reference summary of unverified origin; derived description of the document, not a citizen submission"


# ── documents ─────────────────────────────────────────────────────────────────

@dataclass
class CanonicalDoc:
    doc_key: str                       # first member record id (sorted)
    member_ids: list[str]
    titles: list[str]
    splits: list[str]
    text: str
    summaries: list[str] = field(default_factory=list)


def canonical_documents(df: pd.DataFrame) -> tuple[list[CanonicalDoc], dict[str, str], list[str]]:
    """Exclude empty texts; collapse records with byte-identical text into one canonical document.

    Returns (docs, record_id -> doc_key for indexed records, excluded empty record ids).
    """
    empty = df["document_text"].fillna("").str.strip() == ""
    excluded = sorted(df.loc[empty, "civis_record_id"].tolist())
    groups: dict[str, list[pd.Series]] = {}
    for _, r in df[~empty].sort_values("civis_record_id").iterrows():
        groups.setdefault(r["document_text"], []).append(r)
    docs, mapping = [], {}
    for text, rows in groups.items():
        key = rows[0]["civis_record_id"]
        docs.append(CanonicalDoc(key, [r["civis_record_id"] for r in rows], [r["title"] for r in rows],
                                 [r["split"] for r in rows], text, [r["reference_summary"] for r in rows]))
        for r in rows:
            mapping[r["civis_record_id"]] = key
    docs.sort(key=lambda d: d.doc_key)
    return docs, mapping, excluded


def chunk_text(text: str, size: int = 1000, overlap: int = 100) -> list[tuple[int, int]]:
    """Character windows of about `size` chars, ending at whitespace where possible; consecutive windows overlap."""
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    spans, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            ws = text.rfind(" ", start + size // 2, end)
            if ws > start:
                end = ws
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25:
    """Okapi BM25 over a scikit-learn CountVectorizer term matrix (no extra dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, **vec_kwargs: Any):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words="english", **vec_kwargs)

    def fit(self, texts: list[str]) -> "BM25":
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 0.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        denom_row = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1e-9))
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + denom_row[tf.row])
        self.W = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).tocsc()
        return self

    def score(self, query: str) -> np.ndarray:
        q = self.vec.transform([query]).tocsr()
        cols = q.indices
        if len(cols) == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        weights = self.idf[cols] * q.data                     # repeated query terms count multiple times
        return np.asarray(self.W[:, cols] @ weights).ravel()


# ── dense encoder ─────────────────────────────────────────────────────────────

ENCODERS = {
    "e5-base-v2": {"model_id": "intfloat/e5-base-v2", "dim": 768, "query_prefix": "query: ", "passage_prefix": "passage: ", "licence": "MIT (cached model card)"},
    "all-MiniLM-L6-v2": {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "query_prefix": "", "passage_prefix": "", "licence": "Apache-2.0 (cached model card)"},
}


def cached_weights(model_id: str) -> Path | None:
    """Path of the complete local safetensors file for model_id, or None (incomplete downloads are ignored)."""
    from huggingface_hub import snapshot_download
    try:
        p = Path(snapshot_download(model_id, local_files_only=True)) / "model.safetensors"
    except Exception:  # noqa: BLE001
        return None
    return p if p.is_file() and p.stat().st_size > 1_000_000 else None


def encode(texts: list[str], tokenizer: Any, model: Any, max_length: int = 256, batch_size: int = 16,
           progress: Callable[[int], None] | None = None) -> tuple[np.ndarray, int]:
    """Attention-mask mean pooling + L2 normalisation (the pooling used by both e5 and MiniLM). Returns (float32 matrix, truncated count)."""
    import torch
    out, truncated = [], 0
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        full = tokenizer(batch, truncation=False, padding=False)["input_ids"]
        truncated += sum(len(x) > max_length for x in full)
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1).numpy().astype(np.float32)
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite embedding in batch starting at {i}")
        out.append(v)
        if progress:
            progress(i + len(batch))
    return np.vstack(out), truncated


# ── document-level ranking and fusion ─────────────────────────────────────────

def doc_scores_from_chunks(chunk_scores: np.ndarray, chunk_doc: np.ndarray, n_docs: int) -> np.ndarray:
    """Document score = maximum chunk score."""
    s = np.full(n_docs, -np.inf, dtype=np.float64)
    np.maximum.at(s, chunk_doc, chunk_scores.astype(np.float64))
    return s


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Document indices by descending score; ties broken by index (deterministic)."""
    return np.lexsort((np.arange(len(scores)), -scores))


def rrf(rankings: list[np.ndarray], n_docs: int, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion scores from several full rankings."""
    s = np.zeros(n_docs)
    for order in rankings:
        s[order] += 1.0 / (k + np.arange(1, len(order) + 1))
    return s


def minmax_fusion(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """alpha * minmax(a) + (1 - alpha) * minmax(b), per query. alpha is fixed, never tuned on evaluation queries."""
    def mm(x):
        x = np.where(np.isfinite(x), x, np.nan)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return np.nan_to_num((x - lo) / (hi - lo) if hi > lo else np.zeros_like(x))
    return alpha * mm(a) + (1 - alpha) * mm(b)


# ── evaluation ────────────────────────────────────────────────────────────────

def first_relevant_rank(order: np.ndarray, relevant: set[int]) -> int:
    """1-based rank of the first relevant document (each query has one target document)."""
    for pos, d in enumerate(order, 1):
        if d in relevant:
            return pos
    return len(order) + 1


def metrics_from_ranks(ranks: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    ranks = np.asarray(ranks)
    per = {
        "recall@1": (ranks <= 1).astype(float), "recall@5": (ranks <= 5).astype(float), "recall@10": (ranks <= 10).astype(float),
        "mrr@10": np.where(ranks <= 10, 1.0 / ranks, 0.0), "mrr": 1.0 / ranks,
        "ndcg@10": np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0),
    }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ranks), size=(n_boot, len(ranks)))
    out: dict[str, Any] = {"queries": int(len(ranks)), "median_rank": float(np.median(ranks)), "mean_rank": round(float(ranks.mean()), 2)}
    for k, v in per.items():
        b = v[idx].mean(axis=1)
        out[k] = round(float(v.mean()), 4)
        out[f"{k}_ci95"] = [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    return out


def first_sentence(text: str, min_chars: int = 40, max_chars: int = 300) -> str:
    """First sentence of a summary (sentence end = . ! ? followed by space + capital/digit, after min_chars)."""
    t = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(?<=[.!?])\s+(?=[A-Z0-9(])", t[min_chars:])
    s = t[: min_chars + m.start()] if m else t
    return s[:max_chars].strip()


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()
