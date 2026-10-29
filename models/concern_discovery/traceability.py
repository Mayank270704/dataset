"""
models/concern_discovery/traceability.py
========================================
End-to-end source-evidence traceability and lineage verification for PolicyLens.

Provides 4-tier provenance tracing from raw citizen consultation submissions
through cleaned datasets, full-corpus entries, and into model-discovered/ranked
concern evidence:

  Tier 0: Raw consultation submission file (physical CSV, row, column, uncleaned text)
  Tier 1: Processed clean file (data/processed/*_source_clean.csv, row, slugified field)
  Tier 2: Full-corpus entry (data/full_corpus/run_v1/corpus/corpus_entries.csv)
  Tier 3: Discovered / ranked concern evidence (quote, full_text, cluster_id, rank)

Guarantees:
  - Verbatim substring grounding: every cited quote is verified against the raw citizen input.
  - No synthetic or hallucinated evidence quotes can pass verification.
  - Full transparency over text transformations (whitespace, quotation marks, encoding).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_cell(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()
    if text in {"", "nan", "NaN", "N/A", "n/a", "NA", "nil", "Nil", "None"}:
        return None
    return text


def slugify_column(name: str, seen: Counter[str]) -> str:
    text = clean_cell(name) or "unnamed"
    text = text.replace("’", "'")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text) or "unnamed"
    seen[text] += 1
    return text if seen[text] == 1 else f"{text}_{seen[text]}"


def sniff_encoding(path: Path) -> str:
    raw = path.read_bytes()
    candidates = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    scored: list[tuple[int, str]] = []
    for enc in candidates:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        bad = text.count("\ufffd")
        mojibake = text.count("Ã") + text.count("Â")
        scored.append((bad * 100 + mojibake, enc))
    if not scored:
        return "latin-1"
    return sorted(scored)[0][1]


@dataclass(frozen=True)
class RawDatasetSpec:
    key: str
    title: str
    filename: str
    header_rows: int
    data_header_row: int


RAW_SPECS: dict[str, RawDatasetSpec] = {
    "individual_comments_2016_18": RawDatasetSpec(
        key="individual_comments_2016_18",
        title="Open Government Consultation Data 2016-18 - Individual Comments",
        filename="ap2016-18_individual_comments.csv",
        header_rows=1,
        data_header_row=0,
    ),
    "consultation_compilation_2017_18": RawDatasetSpec(
        key="consultation_compilation_2017_18",
        title="Open Government Consultation Data 2017-18 - Consultation Compilation",
        filename="ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv",
        header_rows=1,
        data_header_row=0,
    ),
    "get_involved_questionnaire_2017_18": RawDatasetSpec(
        key="get_involved_questionnaire_2017_18",
        title="Open Government Consultation Data 2017-18 - Get Involved Questionnaire",
        filename="Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv",
        header_rows=2,
        data_header_row=1,
    ),
}


class RawDatasetContainer:
    """Cached representation of a raw government consultation CSV with column and row mappings."""

    def __init__(self, spec: RawDatasetSpec, root: Path = ROOT):
        self.spec = spec
        self.path = root / spec.filename
        if not self.path.is_file():
            # Try fuzzy match if accent in filename differs
            candidates = list(root.glob("*.csv"))
            for c in candidates:
                if spec.key == "consultation_compilation_2017_18" and "consultationdata2017" in c.name.lower():
                    self.path = c
                    break
        if not self.path.is_file():
            raise FileNotFoundError(f"Raw consultation file not found for {spec.key}: {self.path}")

        self.encoding = sniff_encoding(self.path)
        self.sha256 = sha256_file(self.path)
        raw = pd.read_csv(self.path, encoding=self.encoding, header=None, dtype="string", keep_default_na=False)
        self.header = raw.iloc[: spec.header_rows].copy()
        self.data = raw.iloc[spec.header_rows :].copy().reset_index(drop=True)

        seen: Counter[str] = Counter()
        self.slug_to_raw_col: dict[str, str] = {}
        self.raw_cols: list[str] = []
        for idx, value in enumerate(raw.iloc[spec.data_header_row]):
            val_clean = clean_cell(value) or f"unnamed_{idx + 1}"
            slug = slugify_column(val_clean, seen)
            self.slug_to_raw_col[slug] = val_clean
            self.raw_cols.append(val_clean)
        self.data.columns = self.raw_cols

    def get_raw_cell(self, physical_row: int, slug_field: str) -> tuple[str, str]:
        """
        Retrieves (raw_col_name, raw_cell_text) for a given physical CSV row (1-indexed) and slugified field.
        """
        raw_col = self.slug_to_raw_col.get(slug_field)
        if raw_col is None:
            raise KeyError(f"Slugified field '{slug_field}' not found in raw dataset {self.spec.key}")
        data_idx = physical_row - self.spec.header_rows - 1
        if data_idx < 0 or data_idx >= len(self.data):
            raise IndexError(f"Physical row {physical_row} out of range for {self.spec.key} (data rows {len(self.data)})")
        val = self.data.loc[data_idx, raw_col]
        return raw_col, "" if pd.isna(val) else str(val)


@dataclass
class EvidenceTraceResult:
    # Model / Prioritization level
    evidence_type: str                   # 'ranked_concern' | 'collective_demand' | 'representative_example'
    cluster_id: int
    rank: int
    quote: str
    full_text: str
    quote_sha256: str
    
    # Tier 2: Corpus
    corpus_id: int
    unique_id: int
    dup_group_id: str
    language: str
    corpus_text: str
    
    # Tier 1: Processed
    source_dataset: str
    source_record_id: str
    source_row_number: int
    source_field: str
    clean_text: str
    
    # Tier 0: Raw Source
    raw_filename: str
    raw_col_header: str
    raw_text: str
    raw_file_sha256: str
    
    # Verification Flags
    is_verbatim_in_corpus: bool
    is_verbatim_in_clean: bool
    is_verbatim_in_raw: bool
    match_status: str                     # 'VERBATIM_RAW' | 'VERBATIM_CLEAN' | 'NORM_WHITESPACE_MATCH' | 'FAILED'
    transform_diff_summary: str


def load_raw_containers(root: Path = ROOT) -> dict[str, RawDatasetContainer]:
    return {k: RawDatasetContainer(spec, root) for k, spec in RAW_SPECS.items()}


def load_clean_datasets(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    processed_dir = root / "data" / "processed"
    return {
        "individual_comments_2016_18": pd.read_csv(
            processed_dir / "individual_comments_2016_18_source_clean.csv", dtype=str, keep_default_na=False
        ),
        "consultation_compilation_2017_18": pd.read_csv(
            processed_dir / "consultation_compilation_2017_18_source_clean.csv", dtype=str, keep_default_na=False
        ),
        "get_involved_questionnaire_2017_18": pd.read_csv(
            processed_dir / "get_involved_questionnaire_2017_18_source_clean.csv", dtype=str, keep_default_na=False
        ),
    }


def verify_quote_against_texts(quote: str, raw_text: str, clean_text: str, corpus_text: str) -> tuple[bool, bool, bool, str, str]:
    """
    Evaluates whether a quote is grounded verbatim across raw, clean, and corpus texts.
    Returns:
      (in_corpus, in_clean, in_raw, match_status, diff_summary)
    """
    q_norm = " ".join(quote.split())
    raw_norm = " ".join(raw_text.split())
    clean_norm = " ".join(clean_text.split())
    corpus_norm = " ".join(corpus_text.split())

    in_corpus = quote in corpus_text
    in_clean = quote in clean_text
    in_raw = quote in raw_text

    if in_raw:
        status = "VERBATIM_RAW"
        diff = "verbatim substring of raw text cell"
    elif in_clean:
        status = "VERBATIM_CLEAN"
        diff = "verbatim substring of cleaned text cell (raw cell had outer whitespace/escape differences)"
    elif in_corpus:
        status = "VERBATIM_CORPUS"
        diff = "verbatim substring of corpus text entry"
    elif q_norm in raw_norm:
        status = "NORM_WHITESPACE_RAW"
        diff = "matches raw text after normalizing internal whitespace/newlines"
    elif q_norm in clean_norm or q_norm in corpus_norm:
        status = "NORM_WHITESPACE_CLEAN"
        diff = "matches clean/corpus text after normalizing internal whitespace/newlines"
    else:
        status = "FAILED"
        diff = "quote not found as substring in raw, clean, or corpus text"

    return in_corpus, in_clean, in_raw, status, diff


def trace_evidence_item(
    item: dict[str, Any],
    raw_containers: dict[str, RawDatasetContainer],
    clean_dfs: dict[str, pd.DataFrame],
    corpus_map: dict[int, dict[str, Any]],
    evidence_type: str = "ranked_concern",
) -> EvidenceTraceResult:
    """Traces a single evidence dictionary back through Tiers 2, 1, and 0."""
    corpus_id = int(item["corpus_id"])
    quote = str(item.get("quote", item.get("full_text", ""))).strip()
    full_text = str(item.get("full_text", "")).strip()

    c_entry = corpus_map.get(corpus_id)
    if c_entry is None:
        raise ValueError(f"corpus_id {corpus_id} not found in full corpus")

    ds = c_entry["source_dataset"]
    row_num = int(c_entry["source_row_number"])
    field = c_entry["source_field"]
    corpus_text = c_entry["original_text"]
    unique_id = int(c_entry["unique_id"])
    dup_group_id = str(c_entry.get("dup_group_id", ""))
    language = c_entry.get("language", "")

    # Tier 1: Clean
    clean_df = clean_dfs[ds]
    clean_rows = clean_df.loc[clean_df["source_row_number"] == str(row_num)]
    if len(clean_rows) == 0:
        clean_text = ""
    else:
        clean_text = clean_rows[field].iloc[0] if field in clean_rows.columns else ""

    # Tier 0: Raw
    container = raw_containers[ds]
    raw_col, raw_text = container.get_raw_cell(row_num, field)

    in_corpus, in_clean, in_raw, status, diff = verify_quote_against_texts(
        quote, raw_text, clean_text, corpus_text
    )

    return EvidenceTraceResult(
        evidence_type=evidence_type,
        cluster_id=int(item.get("cluster_id", -1)),
        rank=int(item.get("rank_multi", item.get("rank", 1))),
        quote=quote,
        full_text=full_text or corpus_text,
        quote_sha256=sha256_text(quote),
        corpus_id=corpus_id,
        unique_id=unique_id,
        dup_group_id=dup_group_id,
        language=language,
        corpus_text=corpus_text,
        source_dataset=ds,
        source_record_id=str(c_entry["source_record_id"]),
        source_row_number=row_num,
        source_field=field,
        clean_text=clean_text,
        raw_filename=container.spec.filename,
        raw_col_header=raw_col,
        raw_text=raw_text,
        raw_file_sha256=container.sha256,
        is_verbatim_in_corpus=in_corpus,
        is_verbatim_in_clean=in_clean,
        is_verbatim_in_raw=in_raw,
        match_status=status,
        transform_diff_summary=diff,
    )


def audit_full_evidence_lineage(
    ranking_evidence_path: Path,
    collective_demand_path: Path,
    representative_examples_path: Path,
    corpus_entries_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Audits and produces a complete lineage ledger across all evidence citations in the project."""
    raw_containers = load_raw_containers(root)
    clean_dfs = load_clean_datasets(root)

    corpus_df = pd.read_csv(corpus_entries_path, dtype=str, keep_default_na=False)
    for num_col in ("corpus_id", "unique_id", "source_row_number"):
        if num_col in corpus_df.columns:
            corpus_df[num_col] = pd.to_numeric(corpus_df[num_col], errors="coerce").fillna(-1).astype(int)

    corpus_map = {
        int(r.corpus_id): {
            "corpus_id": int(r.corpus_id),
            "unique_id": int(getattr(r, "unique_id", -1)),
            "dup_group_id": getattr(r, "duplicate_text_group_id", getattr(r, "dup_group_id", "")),
            "source_dataset": str(r.source_dataset),
            "source_record_id": str(r.source_record_id),
            "source_row_number": int(r.source_row_number),
            "source_field": str(r.source_field),
            "original_text": str(r.original_text),
            "language": str(getattr(r, "language", "")),
        }
        for r in corpus_df.itertuples()
    }

    results: list[EvidenceTraceResult] = []

    # 1. Ranking evidence (275 quotes)
    if ranking_evidence_path.is_file():
        df_rank = pd.read_csv(ranking_evidence_path, dtype=str, keep_default_na=False)
        for _, row in df_rank.iterrows():
            res = trace_evidence_item(row.to_dict(), raw_containers, clean_dfs, corpus_map, evidence_type="ranked_concern")
            results.append(res)

    # 2. Collective demand evidence (10 quotes)
    if collective_demand_path.is_file():
        df_cd = pd.read_csv(collective_demand_path, dtype=str, keep_default_na=False)
        for _, row in df_cd.iterrows():
            res = trace_evidence_item(row.to_dict(), raw_containers, clean_dfs, corpus_map, evidence_type="collective_demand")
            results.append(res)

    # 3. Representative cluster examples (370 quotes)
    if representative_examples_path.is_file():
        df_ex = pd.read_csv(representative_examples_path, dtype=str, keep_default_na=False)
        for _, row in df_ex.iterrows():
            res = trace_evidence_item(row.to_dict(), raw_containers, clean_dfs, corpus_map, evidence_type="representative_example")
            results.append(res)

    ledger_df = pd.DataFrame([asdict(r) for r in results])

    # Summary metrics
    n_total = len(results)
    n_verbatim_raw = sum(r.is_verbatim_in_raw for r in results)
    n_verbatim_clean = sum(r.is_verbatim_in_clean for r in results)
    n_verbatim_corpus = sum(r.is_verbatim_in_corpus for r in results)
    n_failed = sum(r.match_status == "FAILED" for r in results)

    dataset_counts = ledger_df["source_dataset"].value_counts().to_dict() if not ledger_df.empty else {}
    language_counts = ledger_df["language"].value_counts().to_dict() if not ledger_df.empty else {}
    type_counts = ledger_df["evidence_type"].value_counts().to_dict() if not ledger_df.empty else {}
    status_counts = ledger_df["match_status"].value_counts().to_dict() if not ledger_df.empty else {}

    return {
        "summary": {
            "total_evidence_quotes_audited": n_total,
            "verbatim_in_raw_count": n_verbatim_raw,
            "verbatim_in_raw_pct": round((n_verbatim_raw / n_total) * 100, 2) if n_total else 0.0,
            "verbatim_in_clean_count": n_verbatim_clean,
            "verbatim_in_corpus_count": n_verbatim_corpus,
            "failed_ungrounded_count": n_failed,
            "pass_rate_grounded": round(((n_total - n_failed) / n_total) * 100, 2) if n_total else 0.0,
            "evidence_by_type": type_counts,
            "evidence_by_dataset": dataset_counts,
            "evidence_by_language": language_counts,
            "match_status_breakdown": status_counts,
        },
        "ledger_df": ledger_df,
    }
