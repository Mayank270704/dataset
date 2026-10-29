"""Audit and first-stage cleaning for Open Government consultation datasets.

This script locates the three government consultation CSVs already present in
the workspace, preserves them unchanged, and writes measured profiles,
conservative cleaned outputs, text-field review files, and overlap reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


@dataclass(frozen=True)
class GovDataset:
    key: str
    title: str
    path: Path
    header_rows: int
    data_header_row: int
    source_url_or_record: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def sniff_dialect(path: Path, encoding: str) -> csv.Dialect:
    sample = path.read_text(encoding=encoding, errors="replace")[:50_000]
    return csv.Sniffer().sniff(sample, delimiters=",;\t|")


def clean_cell(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()
    if text in {"", "nan", "NaN", "N/A", "n/a", "NA", "nil", "Nil", "None"}:
        return None
    return text


def norm_key(value: Any) -> str | None:
    text = clean_cell(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text.casefold()).strip()


def slugify_column(name: str, seen: Counter[str]) -> str:
    text = clean_cell(name) or "unnamed"
    text = text.replace("’", "'")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text) or "unnamed"
    seen[text] += 1
    return text if seen[text] == 1 else f"{text}_{seen[text]}"


def redact_text(text: Any) -> str | None:
    value = clean_cell(text)
    if value is None:
        return None
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", value)
    value = re.sub(r"\b(?:\+?\d[\d .()/-]{7,}\d)\b", "[PHONE_OR_ID]", value)
    return value[:500]


def locate_government_files() -> list[GovDataset]:
    ignored_parts = {".pytest_cache", "data", "reports", "__pycache__"}
    files = [p for p in ROOT.rglob("*.csv") if not (set(part.lower() for part in p.relative_to(ROOT).parts) & ignored_parts)]
    found: dict[str, Path] = {}
    for path in files:
        lower = path.name.lower()
        if "get involved questionnaire" in lower:
            found["get_involved_questionnaire_2017_18"] = path
        elif "individual_comments" in lower or "individual comments" in lower:
            found["individual_comments_2016_18"] = path
        elif "consultationdata2017engagement" in lower or "compilation" in lower:
            found["consultation_compilation_2017_18"] = path

    required = {
        "get_involved_questionnaire_2017_18": (
            "Open Government Consultation Data 2017-18 - Get Involved Questionnaire",
            2,
            1,
        ),
        "individual_comments_2016_18": (
            "Open Government Consultation Data 2016-18 - Individual Comments",
            1,
            0,
        ),
        "consultation_compilation_2017_18": (
            "Open Government Consultation Data 2017-18 - Consultation Compilation",
            1,
            0,
        ),
    }
    missing = sorted(set(required) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing expected government dataset(s): {', '.join(missing)}")

    return [
        GovDataset(
            key=key,
            title=required[key][0],
            path=found[key],
            header_rows=required[key][1],
            data_header_row=required[key][2],
            source_url_or_record="Local file provided by user; original source URL not documented in workspace.",
        )
        for key in required
    ]


def raw_csv_stats(path: Path, encoding: str, dialect: csv.Dialect, header_rows: int) -> dict[str, Any]:
    widths: Counter[int] = Counter()
    blank_rows = 0
    rows = []
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f, dialect)
        for row in reader:
            rows.append(row)
            widths[len(row)] += 1
            if not any(clean_cell(cell) for cell in row):
                blank_rows += 1
    expected_width = len(rows[header_rows - 1]) if rows and header_rows else len(rows[0])
    malformed_rows = sum(count for width, count in widths.items() if width != expected_width)
    return {
        "physical_rows": len(rows),
        "blank_rows": blank_rows,
        "expected_columns": expected_width,
        "malformed_records": malformed_rows,
        "row_width_distribution": dict(sorted(widths.items())),
        "header_preview": rows[:header_rows],
    }


def read_government_dataset(spec: GovDataset, encoding: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(spec.path, encoding=encoding, header=None, dtype="string", keep_default_na=False)
    header = raw.iloc[: spec.header_rows].copy()
    data = raw.iloc[spec.header_rows :].copy()
    seen: Counter[str] = Counter()
    columns = []
    for idx, value in enumerate(raw.iloc[spec.data_header_row]):
        original = clean_cell(value) or f"unnamed_{idx + 1}"
        seen[original] += 1
        columns.append(original if seen[original] == 1 else f"{original}__duplicate_{seen[original]}")
    data.columns = columns
    data = data.reset_index(drop=True)
    data = data.map(clean_cell)
    return data, header


def exact_duplicate_rows(df: pd.DataFrame) -> int:
    filled = df.fillna("")
    return int(filled.duplicated().sum())


def missing_json(df: pd.DataFrame) -> str:
    return json.dumps(df.isna().sum().astype(int).to_dict(), ensure_ascii=False)


def dtypes_json(df: pd.DataFrame) -> str:
    return json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}, ensure_ascii=False)


def unique_ratio(series: pd.Series) -> float:
    nonempty = series.dropna()
    if len(nonempty) == 0:
        return 0.0
    return float(nonempty.nunique() / len(nonempty))


def looks_numeric_or_codes(series: pd.Series) -> bool:
    values = [str(v) for v in series.dropna().head(500)]
    if not values:
        return False
    numeric_like = sum(bool(re.fullmatch(r"[0-9,.; /-]+", v)) for v in values)
    short = sum(len(v) <= 5 for v in values)
    return numeric_like / len(values) > 0.75 or short / len(values) > 0.85


def text_column_assessment(df: pd.DataFrame, column: str) -> dict[str, Any]:
    series = df[column]
    nonempty = series.dropna()
    lengths = nonempty.astype(str).map(len)
    name = column.lower()
    likely_question = bool(re.search(r"\bq\d+|question|comment|idea|why|please|specify|miss|like|improve|concern|suggest|participat|important|prompt|subject", name))
    avg_len = float(lengths.mean()) if len(lengths) else 0.0
    max_len = int(lengths.max()) if len(lengths) else 0
    sample_values = [redact_text(v) for v in nonempty.astype(str).head(5)]
    is_free_text = bool(len(nonempty) and not looks_numeric_or_codes(series) and (avg_len >= 20 or max_len >= 80 or likely_question))
    likely_policy_relevant = bool(is_free_text and re.search(r"comment|idea|question|why|miss|concern|suggest|important|improve|prompt|subject|open government|participat|other|specify", name))
    if not len(nonempty):
        field_type = "empty_or_missing"
    elif is_free_text:
        field_type = "free_text_candidate"
    elif looks_numeric_or_codes(series):
        field_type = "numeric_checkbox_or_code"
    elif unique_ratio(series) < 0.2:
        field_type = "categorical_or_metadata"
    else:
        field_type = "short_text_or_metadata"
    return {
        "column_name": column,
        "non_empty_responses": int(len(nonempty)),
        "missing_responses": int(series.isna().sum()),
        "avg_length": round(avg_len, 2),
        "max_length": max_len,
        "unique_values": int(nonempty.nunique()),
        "field_type": field_type,
        "policy_relevance_recommendation": "review_for_policy_concerns_suggestions_or_requested_actions"
        if likely_policy_relevant
        else "do_not_treat_as_policy_comment_without_manual_review",
        "representative_samples_redacted": json.dumps(sample_values, ensure_ascii=False),
    }


def dataset_text_fields(spec: GovDataset, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        row = text_column_assessment(df, column)
        row["dataset_name"] = spec.key
        rows.append(row)
    columns = [
        "column_name",
        "non_empty_responses",
        "missing_responses",
        "avg_length",
        "max_length",
        "unique_values",
        "field_type",
        "policy_relevance_recommendation",
        "representative_samples_redacted",
        "dataset_name",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_cleaned_outputs(spec: GovDataset, df: pd.DataFrame, header: pd.DataFrame) -> tuple[Path, Path, Path]:
    seen: Counter[str] = Counter()
    original_columns = list(df.columns)
    normalized_columns = [slugify_column(col, seen) for col in original_columns]

    clean_df = df.copy()
    clean_df.columns = normalized_columns
    clean_df.insert(0, "source_dataset", spec.key)
    clean_df.insert(1, "source_file", str(spec.path.relative_to(ROOT)))
    clean_df.insert(2, "source_row_number", range(spec.header_rows + 1, spec.header_rows + 1 + len(clean_df)))

    mapping = pd.DataFrame(
        {
            "dataset_name": spec.key,
            "original_column_name": original_columns,
            "normalized_column_name": normalized_columns,
        }
    )
    header_path = INTERIM / f"{spec.key}_header_rows.csv"
    mapping_path = INTERIM / f"{spec.key}_column_mapping.csv"
    clean_path = PROCESSED / f"{spec.key}_source_clean.csv"
    header.to_csv(header_path, index=False, header=False, encoding="utf-8", lineterminator="\n")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8", lineterminator="\n")
    clean_df.to_csv(clean_path, index=False, encoding="utf-8", lineterminator="\n")
    return clean_path, mapping_path, header_path


def text_entries(spec: GovDataset, df: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    text_entry_pattern = re.compile(
        r"comment|question or idea|practical idea|public engagement question|og is important|improvements|did we miss|please tell us why|knowledge or experience|other|specify",
        re.IGNORECASE,
    )
    relevant = [
        row["column_name"]
        for _, row in fields.iterrows()
        if row["field_type"] == "free_text_candidate" and text_entry_pattern.search(str(row["column_name"]))
    ]
    rows = []
    for col in relevant:
        for idx, value in df[col].items():
            key = norm_key(value)
            if key:
                rows.append(
                    {
                        "dataset_name": spec.key,
                        "source_row_index": int(idx),
                        "source_row_number": int(spec.header_rows + 1 + idx),
                        "column_name": col,
                        "text_key": key,
                        "text_sample_redacted": redact_text(value),
                    }
                )
    return pd.DataFrame(rows)


def duplicate_text_count(entries: pd.DataFrame) -> int:
    if entries.empty:
        return 0
    return int(len(entries) - entries["text_key"].nunique())


def profile_one(spec: GovDataset) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoding = sniff_encoding(spec.path)
    dialect = sniff_dialect(spec.path, encoding)
    raw_stats = raw_csv_stats(spec.path, encoding, dialect, spec.header_rows)
    df, header = read_government_dataset(spec, encoding)
    fields = dataset_text_fields(spec, df)
    entries = text_entries(spec, df, fields)
    clean_path, mapping_path, header_path = write_cleaned_outputs(spec, df, header)
    fields.to_csv(INTERIM / f"{spec.key}_text_fields.csv", index=False, encoding="utf-8", lineterminator="\n")
    entries.to_csv(INTERIM / f"{spec.key}_text_entries.csv", index=False, encoding="utf-8", lineterminator="\n")

    profile = {
        "dataset_name": spec.key,
        "dataset_title": spec.title,
        "relative_path": str(spec.path.relative_to(ROOT)),
        "file_size_bytes": spec.path.stat().st_size,
        "source_url_or_record": spec.source_url_or_record,
        "encoding": encoding,
        "delimiter": dialect.delimiter,
        "header_structure": f"{spec.header_rows} header row(s); data column names taken from row {spec.data_header_row + 1}",
        "sha256": sha256(spec.path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": json.dumps(list(df.columns), ensure_ascii=False),
        "dtypes": dtypes_json(df),
        "missing_values": missing_json(df),
        "blank_rows": raw_stats["blank_rows"],
        "malformed_records": raw_stats["malformed_records"],
        "row_width_distribution": json.dumps(raw_stats["row_width_distribution"], ensure_ascii=False),
        "exact_duplicate_rows": exact_duplicate_rows(df),
        "duplicate_text_entries": duplicate_text_count(entries),
        "free_text_candidate_fields": int((fields["field_type"] == "free_text_candidate").sum()),
        "processed_output": str(clean_path.relative_to(ROOT)),
        "column_mapping_output": str(mapping_path.relative_to(ROOT)),
        "header_rows_output": str(header_path.relative_to(ROOT)),
        "license_provenance_status": "Unknown from local files; user-provided government open-data CSV, source URL/license not present in workspace.",
        "recommended_use": "Consultation audit and later policy-concern/requested-action annotation after overlap review.",
        "limitations": "Do not merge or train yet; bilingual headers/values and overlap with compilation require human review.",
    }
    return profile, df, fields, entries


def write_duplicate_review(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return entries
    dupes = entries[entries["text_key"].duplicated(keep=False)].copy()
    dupes = dupes.sort_values(["text_key", "dataset_name", "source_row_number", "column_name"])
    dupes.to_csv(INTERIM / "government_duplicate_text_entries.csv", index=False, encoding="utf-8", lineterminator="\n")
    return dupes


def schema_overlap(profiles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    cols_by_dataset = {p["dataset_name"]: json.loads(p["column_names"]) for p in profiles}
    for left_name, left_cols in cols_by_dataset.items():
        for right_name, right_cols in cols_by_dataset.items():
            if left_name >= right_name:
                continue
            left_norm = {norm_key(c) for c in left_cols}
            right_norm = {norm_key(c) for c in right_cols}
            common = sorted(c for c in left_norm & right_norm if c)
            rows.append(
                {
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "left_columns": len(left_cols),
                    "right_columns": len(right_cols),
                    "common_normalized_columns": len(common),
                    "common_column_names": json.dumps(common[:50], ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def id_overlap(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidate_names = ["Row\nRangée", "RowRangée", "Row/Ligne", "Unique Identifier", "ID"]
    rows = []
    ids: dict[tuple[str, str], set[str]] = {}
    for name, df in datasets.items():
        for col in df.columns:
            if col in candidate_names or "identifier" in col.lower() or col.lower() in {"id", "row/ligne"}:
                values = {str(v) for v in df[col].dropna() if clean_cell(v)}
                ids[(name, col)] = values
    keys = list(ids)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            common = sorted(ids[left] & ids[right])
            if common:
                rows.append(
                    {
                        "left_dataset": left[0],
                        "left_id_column": left[1],
                        "right_dataset": right[0],
                        "right_id_column": right[1],
                        "overlapping_ids": len(common),
                        "sample_ids": json.dumps(common[:20], ensure_ascii=False),
                    }
                )
    return pd.DataFrame(rows)


def near_duplicate_candidates(entries: pd.DataFrame, threshold: float = 0.92, max_pairs: int = 250) -> pd.DataFrame:
    if entries.empty:
        return entries.iloc[0:0].copy()
    candidates = []
    sampled = entries[entries["text_key"].str.len().between(30, 500)].copy()
    sampled["block_key"] = sampled["text_key"].str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().str[:24]
    sampled["length_bucket"] = (sampled["text_key"].str.len() // 50).astype(int)
    sampled = sampled.sort_values(["block_key", "length_bucket", "text_key"]).head(5000)
    for _, block in sampled.groupby(["block_key", "length_bucket"], dropna=True):
        records = block.head(120).to_dict("records")
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if abs(len(left["text_key"]) - len(right["text_key"])) > 80:
                    continue
                if left["dataset_name"] == right["dataset_name"] and left["source_row_number"] == right["source_row_number"]:
                    continue
                if left["text_key"] == right["text_key"]:
                    continue
                ratio = SequenceMatcher(None, left["text_key"], right["text_key"]).ratio()
                if ratio < threshold:
                    continue
                candidates.append(
                    {
                        "left_dataset": left["dataset_name"],
                        "left_row": left["source_row_number"],
                        "left_column": left["column_name"],
                        "right_dataset": right["dataset_name"],
                        "right_row": right["source_row_number"],
                        "right_column": right["column_name"],
                        "similarity": round(ratio, 3),
                        "left_sample_redacted": left["text_sample_redacted"],
                        "right_sample_redacted": right["text_sample_redacted"],
                    }
                )
                if len(candidates) >= max_pairs:
                    return pd.DataFrame(candidates)
    return pd.DataFrame(candidates)


def write_reports(profiles: list[dict[str, Any]], all_fields: pd.DataFrame, duplicate_entries: pd.DataFrame, schema_df: pd.DataFrame, id_df: pd.DataFrame, near_df: pd.DataFrame) -> None:
    inventory = pd.DataFrame(profiles)
    inventory.to_csv(REPORTS / "data_inventory.csv", index=False, encoding="utf-8", lineterminator="\n")
    all_fields.to_csv(INTERIM / "government_text_field_assessment.csv", index=False, encoding="utf-8", lineterminator="\n")
    schema_df.to_csv(INTERIM / "government_schema_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
    id_df.to_csv(INTERIM / "government_id_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
    near_df.to_csv(INTERIM / "government_near_duplicate_candidates.csv", index=False, encoding="utf-8", lineterminator="\n")

    quality = [
        "# PolicyLens Data Quality Report",
        "",
        "Generated by `python scripts/policylens_government_consultation_audit.py`.",
        "",
        "## Confirmed Files",
        "",
    ]
    for p in profiles:
        fields = all_fields[all_fields["dataset_name"] == p["dataset_name"]]
        useful = fields[fields["policy_relevance_recommendation"] == "review_for_policy_concerns_suggestions_or_requested_actions"]
        quality.extend(
            [
                f"### {p['dataset_title']}",
                f"- Dataset key: `{p['dataset_name']}`",
                f"- Path: `{p['relative_path']}`",
                f"- Size: {p['file_size_bytes']} bytes; encoding: `{p['encoding']}`; delimiter: `{p['delimiter']}`",
                f"- Header structure: {p['header_structure']}",
                f"- Rows/columns: {p['rows']} / {p['columns']}",
                f"- Blank rows: {p['blank_rows']}; malformed records: {p['malformed_records']}",
                f"- Exact duplicate rows: {p['exact_duplicate_rows']}; duplicate text entries: {p['duplicate_text_entries']}",
                f"- SHA-256: `{p['sha256']}`",
                f"- Clean output: `{p['processed_output']}`",
                f"- Useful text fields recommended for review: {len(useful)}",
                "",
            ]
        )
        for _, row in useful.head(12).iterrows():
            quality.append(f"  - `{row['column_name']}`: {row['non_empty_responses']} non-empty responses")
            try:
                samples = json.loads(row["representative_samples_redacted"])
            except (TypeError, json.JSONDecodeError):
                samples = []
            for sample in samples[:2]:
                if sample:
                    quality.append(f"    Sample: {sample}")
        quality.append("")
    quality.extend(
        [
            "## Unresolved Issues",
            "",
            "- Original source URLs and explicit license files were not present locally.",
            "- Near-duplicate candidates are heuristic review items, not automatic exclusions.",
            "- Compilation overlap requires human semantic review before any merge.",
            "",
        ]
    )
    (REPORTS / "data_quality_report.md").write_text("\n".join(quality), encoding="utf-8")

    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows produced."
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            values = [str(row[col]).replace("\n", " ") for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    overlap = [
        "# Dataset Overlap Report",
        "",
        "Generated by `python scripts/policylens_government_consultation_audit.py`.",
        "",
        "## Schema Overlap",
        "",
        markdown_table(schema_df),
        "",
        "## Identifier Overlap",
        "",
        markdown_table(id_df) if not id_df.empty else "No overlapping identifier values were found among detected ID-like columns.",
        "",
        "## Exact Duplicate Text Entries",
        "",
        f"Duplicate text-entry rows across the three datasets: {len(duplicate_entries)}.",
        f"Output: `data/interim/government_duplicate_text_entries.csv`.",
        "",
        "## Near-Duplicate Candidates",
        "",
        f"Heuristic near-duplicate candidate pairs written: {len(near_df)}.",
        f"Output: `data/interim/government_near_duplicate_candidates.csv`.",
        "",
        "## Assessment",
        "",
        "The compilation dataset shares question/prompt concepts and may overlap with both the questionnaire and individual-comments datasets, but it has a wider schema and should not be merged until duplicate text, row identifiers, and source semantics have been reviewed.",
    ]
    (REPORTS / "dataset_overlap_report.md").write_text("\n".join(overlap), encoding="utf-8")

    log = [
        "# Cleaning Log",
        "",
        "Raw government CSV files were not modified, renamed, moved, or deleted.",
        "",
        "Transformations applied:",
        "",
        "- Located the three government consultation CSVs by filename/content hints.",
        "- Detected encoding and delimiter for each file.",
        "- Loaded the questionnaire with two header rows and retained both header rows separately.",
        "- Loaded individual-comments and compilation datasets with single parsed CSV header rows.",
        "- Standardized empty values such as blank strings and `n/a`/`nil` to null in processed outputs.",
        "- Applied only conservative whitespace and line-ending normalization.",
        "- Wrote normalized column-name mappings without discarding original column labels.",
        "- Preserved source dataset, source file, and source row number in each cleaned output.",
        "- Detected exact duplicate rows, duplicate text entries, identifier overlap, schema overlap, and heuristic near-duplicates.",
        "",
        "Exclusions:",
        "",
        "- No rows were excluded from source-clean outputs.",
    ]
    (REPORTS / "cleaning_log.md").write_text("\n".join(log), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    specs = locate_government_files()
    profiles = []
    loaded: dict[str, pd.DataFrame] = {}
    fields = []
    entries = []
    checksum_rows = []

    for spec in specs:
        profile, df, field_df, entry_df = profile_one(spec)
        profiles.append(profile)
        loaded[spec.key] = df
        fields.append(field_df)
        entries.append(entry_df)
        checksum_rows.append({"dataset_name": spec.key, "relative_path": str(spec.path.relative_to(ROOT)), "sha256": profile["sha256"]})

    all_fields = pd.concat(fields, ignore_index=True)
    all_entries = pd.concat(entries, ignore_index=True)
    duplicate_entries = write_duplicate_review(all_entries)
    schema_df = schema_overlap(profiles)
    id_df = id_overlap(loaded)
    near_df = near_duplicate_candidates(all_entries)

    pd.DataFrame(checksum_rows).to_csv(INTERIM / "government_raw_file_checksums.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(
        [
            {
                "dataset_name": p["dataset_name"],
                "input_rows": p["rows"],
                "source_clean_rows": p["rows"],
                "excluded_rows": 0,
            }
            for p in profiles
        ]
    ).to_csv(INTERIM / "government_row_count_verification.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_reports(profiles, all_fields, duplicate_entries, schema_df, id_df, near_df)
    print(f"Profiled {len(profiles)} government consultation datasets.")
    print("Wrote reports/data_inventory.csv, reports/data_quality_report.md, reports/dataset_overlap_report.md, and reports/cleaning_log.md.")


if __name__ == "__main__":
    main()
