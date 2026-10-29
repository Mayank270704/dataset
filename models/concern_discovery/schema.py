"""
models/concern_discovery/schema.py
==================================
Input-schema and data-integrity checks for the concern discovery pipeline.

The pipeline never redefines the corpus: it consumes the eligible corpus of
``data/full_corpus/run_v1`` and refuses to run if counts, alignment or text identity
differ from what that run recorded.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_ENTRY_COLUMNS = (
    "corpus_id", "unique_id", "source_dataset", "source_record_id", "source_row_number",
    "source_field", "question_slug", "language", "stakeholder_group", "text_key", "original_text",
)
REQUIRED_UNIQUE_COLUMNS = (
    "unique_id", "corpus_id", "multiplicity", "source_dataset", "source_record_id",
    "question_slug", "language", "original_text",
)
KNOWN_DATASETS = {
    "consultation_compilation_2017_18", "get_involved_questionnaire_2017_18", "individual_comments_2016_18",
}
# label state codes shared with scripts/run_clustering_experiments_v2.py
NOISE, CONTENT_FREE, SHORT_EXCLUDED, OUT_OF_SUBSET = -1, -2, -3, -4
VALID_STATE_CODES = {NOISE, CONTENT_FREE, SHORT_EXCLUDED, OUT_OF_SUBSET}


class CorpusIntegrityError(ValueError):
    """Raised when an input violates a hard integrity rule; the pipeline stops."""


@dataclass
class CheckLog:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": "PASSED", "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.checks.append({"check": name, "status": "FAILED", "detail": detail})
        raise CorpusIntegrityError(f"{name}: {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.checks.append({"check": name, "status": "WARN", "detail": detail})


def normalise_text(text: str) -> str:
    """Identical to scripts/run_full_corpus_clustering.normalise_text (NFC, trim, casefold, collapse spaces)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], name: str, log: CheckLog) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.fail(f"{name} schema", f"missing columns {missing}")
    log.ok(f"{name} schema", f"{len(cols)} required columns present")


def validate_corpus(entries: pd.DataFrame, unique: pd.DataFrame, emb: np.ndarray,
                    manifest: dict[str, Any] | None = None, log: CheckLog | None = None) -> CheckLog:
    """Validate entry/unique-text tables and their embedding matrix. Raises CorpusIntegrityError on failure."""
    log = log or CheckLog()
    _require_columns(entries, REQUIRED_ENTRY_COLUMNS, "corpus_entries", log)
    _require_columns(unique, REQUIRED_UNIQUE_COLUMNS, "unique_texts", log)
    if len(entries) == 0 or len(unique) == 0:
        log.fail("non-empty corpus", f"entries={len(entries)}, unique={len(unique)}")

    cid = pd.to_numeric(entries["corpus_id"], errors="coerce")
    if cid.isna().any() or cid.duplicated().any():
        log.fail("corpus_id unique integers", "missing, non-integer or duplicated corpus_id values")
    log.ok("corpus_id unique integers", f"{len(entries):,} entries")

    uid = pd.to_numeric(unique["unique_id"], errors="coerce")
    if uid.isna().any() or uid.astype(int).tolist() != list(range(len(unique))):
        log.fail("unique_id contiguous", "unique_texts.unique_id must be 0..n-1 in row order")
    e_uid = pd.to_numeric(entries["unique_id"], errors="coerce")
    if e_uid.isna().any() or e_uid.min() < 0 or e_uid.max() >= len(unique):
        log.fail("entry -> unique mapping", "entries.unique_id missing or out of range")
    e_uid = e_uid.astype(int)
    counts = e_uid.value_counts().reindex(range(len(unique)), fill_value=0)
    if (counts == 0).any():
        log.fail("entry -> unique mapping", f"{int((counts == 0).sum())} unique texts have no entry")
    mult = pd.to_numeric(unique["multiplicity"], errors="coerce").fillna(-1).astype(int)
    if not np.array_equal(mult.to_numpy(), counts.to_numpy()):
        log.fail("multiplicity", "unique_texts.multiplicity disagrees with entry counts")
    log.ok("entry -> unique mapping", f"{len(unique):,} unique texts; multiplicities consistent")

    # every entry's text must normalise to its unique text (no silent text substitution)
    u_norm = np.array([normalise_text(t) for t in unique["original_text"]], dtype=object)
    e_norm = np.array([normalise_text(t) for t in entries["original_text"]], dtype=object)
    bad = int((u_norm[e_uid.to_numpy()] != e_norm).sum())
    if bad:
        log.fail("entry text identity", f"{bad} entries whose normalised text differs from their unique text")
    log.ok("entry text identity", "every entry text normalises to its unique text")

    blank = int((entries["original_text"].astype(str).str.strip() == "").sum())
    if blank:
        log.fail("non-blank text", f"{blank} blank texts")
    unknown_ds = sorted(set(entries["source_dataset"]) - KNOWN_DATASETS)
    if unknown_ds:
        log.fail("source datasets", f"unknown datasets {unknown_ds}")
    log.ok("source datasets", ", ".join(f"{k}={v}" for k, v in entries["source_dataset"].value_counts().items()))
    for col in ("source_record_id", "question_slug", "language"):
        n_blank = int((entries[col].astype(str).str.strip() == "").sum())
        if n_blank:
            log.warn(f"{col} metadata", f"{n_blank} blank values")
        else:
            log.ok(f"{col} metadata", "no blank values")

    if emb.ndim != 2 or emb.shape[0] != len(unique):
        log.fail("embedding alignment", f"shape {emb.shape} vs {len(unique)} unique texts")
    if not np.isfinite(emb).all():
        log.fail("embedding finiteness", "non-finite values")
    norms = np.linalg.norm(emb, axis=1)
    if np.abs(norms - 1).max() > 1e-3:
        log.fail("embedding normalisation", f"row norms deviate from 1 by up to {np.abs(norms - 1).max():.2e}")
    log.ok("embeddings", f"{emb.shape[0]:,} x {emb.shape[1]}, finite, L2-normalised")

    if manifest is not None:
        if manifest.get("entries") != len(entries) or manifest.get("unique_texts") != len(unique):
            log.fail("corpus counts vs run_v1 manifest",
                     f"manifest {manifest.get('entries')}/{manifest.get('unique_texts')} vs {len(entries)}/{len(unique)}")
        by_ds = {k: int(v) for k, v in entries["source_dataset"].value_counts().items()}
        if manifest.get("by_dataset") and by_ds != manifest["by_dataset"]:
            log.fail("dataset counts vs run_v1 manifest", f"{by_ds} vs {manifest['by_dataset']}")
        log.ok("corpus counts vs run_v1 manifest", f"{len(entries):,} entries / {len(unique):,} unique texts unchanged")
    return log


def validate_labels(labels: np.ndarray, n_unique: int, name: str, log: CheckLog | None = None) -> CheckLog:
    """Unique-level label arrays: one integer per unique text; negatives must be known state codes."""
    log = log or CheckLog()
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n_unique:
        log.fail(f"labels {name}", f"shape {labels.shape} vs {n_unique} unique texts")
    if not np.issubdtype(labels.dtype, np.integer):
        log.fail(f"labels {name}", f"non-integer dtype {labels.dtype}")
    neg = set(np.unique(labels[labels < 0]).tolist())
    if neg - VALID_STATE_CODES:
        log.fail(f"labels {name}", f"unknown state codes {sorted(neg - VALID_STATE_CODES)}")
    log.ok(f"labels {name}", f"{len(set(labels[labels >= 0].tolist()))} clusters; states {sorted(neg)}")
    return log
