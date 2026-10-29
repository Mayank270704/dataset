"""Create the PolicyLens dataset audit notebook.

The generated notebook is self-contained: it performs discovery, auditing,
cleaning, reporting, and validation inside notebook cells.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_dataset_audit_and_cleaning.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # PolicyLens Dataset Audit and Conservative Cleaning

        This notebook audits the three Open Government consultation datasets already present in the project. It is designed for a B.Tech CSE (AI) research workflow: every result is calculated from local files, raw inputs are preserved unchanged, and cleaned outputs are kept separate from source data.

        **Important boundaries:** this notebook does not train models, does not download data, does not merge datasets, and does not invent sentiment/stance/concern labels.
        """
    ),
    md(
        """
        ## 1. Setup

        We import only standard scientific Python tools available in the environment. Paths are resolved relative to the project root so the notebook can be rerun from VS Code without hardcoded machine-specific absolute paths.
        """
    ),
    code(
        r"""
        import csv
        import hashlib
        import json
        import re
        from collections import Counter
        from dataclasses import dataclass
        from difflib import SequenceMatcher
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        pd.set_option("display.max_columns", 80)
        pd.set_option("display.max_colwidth", 140)
        pd.set_option("display.width", 160)
        plt.rcParams["figure.figsize"] = (10, 4)

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
        DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
        REPORTS_DIR = PROJECT_ROOT / "reports"
        NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

        for directory in [DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR, NOTEBOOKS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        PROJECT_ROOT
        """
    ),
    md(
        """
        ## Helper Functions

        These functions keep the notebook readable. They detect encodings and delimiters, normalize only whitespace/empty values, redact obvious contact details in displayed samples, and preserve traceability through source row numbers.
        """
    ),
    code(
        r"""
        @dataclass(frozen=True)
        class ConsultationDataset:
            key: str
            title: str
            path: Path
            header_rows: int
            data_header_row: int


        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def sniff_encoding(path: Path) -> str:
            raw = path.read_bytes()
            scores = []
            for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
                try:
                    text = raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
                replacement_chars = text.count("\ufffd")
                mojibake_markers = text.count("Ã") + text.count("Â")
                scores.append((replacement_chars * 100 + mojibake_markers, encoding))
            return sorted(scores)[0][1] if scores else "latin-1"


        def sniff_delimiter(path: Path, encoding: str) -> str:
            sample = path.read_text(encoding=encoding, errors="replace")[:50_000]
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter


        def clean_cell(value):
            if value is None or pd.isna(value):
                return None
            text = str(value).replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"[ \t\f\v]+", " ", text)
            text = re.sub(r" *\n *", "\n", text).strip()
            if text in {"", "nan", "NaN", "N/A", "n/a", "NA", "nil", "Nil", "None"}:
                return None
            return text


        def normalized_text_key(value):
            text = clean_cell(value)
            if text is None:
                return None
            return re.sub(r"\s+", " ", text.casefold()).strip()


        def redact_text(value, limit=500):
            text = clean_cell(value)
            if text is None:
                return None
            text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
            text = re.sub(r"\b(?:\+?\d[\d .()/-]{7,}\d)\b", "[PHONE_OR_ID]", text)
            return text[:limit]


        def slugify_column(name: str, seen: Counter) -> str:
            text = clean_cell(name) or "unnamed"
            text = text.replace("’", "'")
            text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
            text = re.sub(r"_+", "_", text) or "unnamed"
            seen[text] += 1
            return text if seen[text] == 1 else f"{text}_{seen[text]}"
        """
    ),
    md(
        """
        ## 2. Dataset Loading

        The files are located by searching the workspace and excluding generated folders such as `data/`, `reports/`, and cache directories. The questionnaire has two header rows; the English row is used for working column names while both header rows are preserved.
        """
    ),
    code(
        r"""
        def locate_consultation_files(root: Path) -> list[ConsultationDataset]:
            ignored_parts = {".pytest_cache", "data", "reports", "__pycache__", "notebooks"}
            csv_files = [
                path for path in root.rglob("*.csv")
                if not (set(part.lower() for part in path.relative_to(root).parts) & ignored_parts)
            ]
            found = {}
            for path in csv_files:
                lower_name = path.name.lower()
                if "get involved questionnaire" in lower_name:
                    found["get_involved_questionnaire_2017_18"] = path
                elif "individual_comments" in lower_name or "individual comments" in lower_name:
                    found["individual_comments_2016_18"] = path
                elif "consultationdata2017engagement" in lower_name or "compilation" in lower_name:
                    found["consultation_compilation_2017_18"] = path

            expected = {
                "get_involved_questionnaire_2017_18": ("Open Government Consultation Data 2017-18 - Get Involved Questionnaire", 2, 1),
                "individual_comments_2016_18": ("Open Government Consultation Data 2016-18 - Individual Comments", 1, 0),
                "consultation_compilation_2017_18": ("Open Government Consultation Data 2017-18 - Consultation Compilation", 1, 0),
            }
            missing = sorted(set(expected) - set(found))
            if missing:
                raise FileNotFoundError(f"Missing expected consultation dataset(s): {missing}")
            return [
                ConsultationDataset(key=key, title=expected[key][0], path=found[key], header_rows=expected[key][1], data_header_row=expected[key][2])
                for key in expected
            ]


        datasets = locate_consultation_files(PROJECT_ROOT)
        pd.DataFrame([
            {
                "dataset": item.key,
                "relative_path": item.path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": item.path.stat().st_size,
                "header_rows": item.header_rows,
            }
            for item in datasets
        ])
        """
    ),
    code(
        r"""
        def raw_csv_stats(path: Path, encoding: str, delimiter: str, expected_columns: int) -> dict:
            widths = Counter()
            blank_rows = 0
            with path.open("r", encoding=encoding, errors="replace", newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                for row in reader:
                    widths[len(row)] += 1
                    if not any(clean_cell(cell) for cell in row):
                        blank_rows += 1
            malformed_records = sum(count for width, count in widths.items() if width != expected_columns)
            return {
                "blank_rows": blank_rows,
                "malformed_records": malformed_records,
                "row_width_distribution": dict(sorted(widths.items())),
            }


        def load_consultation_dataset(spec: ConsultationDataset, encoding: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            raw = pd.read_csv(spec.path, encoding=encoding, header=None, dtype="string", keep_default_na=False)
            header_rows = raw.iloc[: spec.header_rows].copy()
            data = raw.iloc[spec.header_rows :].copy()
            seen = Counter()
            columns = []
            for index, value in enumerate(raw.iloc[spec.data_header_row]):
                original = clean_cell(value) or f"unnamed_{index + 1}"
                seen[original] += 1
                columns.append(original if seen[original] == 1 else f"{original}__duplicate_{seen[original]}")
            data.columns = columns
            data = data.reset_index(drop=True).map(clean_cell)
            return data, header_rows


        loaded_data = {}
        header_rows_by_dataset = {}
        file_metadata = []
        raw_checksums_before = {}

        for spec in datasets:
            encoding = sniff_encoding(spec.path)
            delimiter = sniff_delimiter(spec.path, encoding)
            data_frame, header_rows = load_consultation_dataset(spec, encoding)
            raw_stats = raw_csv_stats(spec.path, encoding, delimiter, data_frame.shape[1])
            checksum = sha256(spec.path)
            raw_checksums_before[spec.key] = checksum
            loaded_data[spec.key] = data_frame
            header_rows_by_dataset[spec.key] = header_rows
            file_metadata.append({
                "dataset_name": spec.key,
                "dataset_title": spec.title,
                "relative_path": spec.path.relative_to(PROJECT_ROOT).as_posix(),
                "file_size_bytes": spec.path.stat().st_size,
                "encoding": encoding,
                "delimiter": delimiter,
                "header_structure": f"{spec.header_rows} header row(s); data column names from row {spec.data_header_row + 1}",
                "sha256": checksum,
                "blank_rows": raw_stats["blank_rows"],
                "malformed_records": raw_stats["malformed_records"],
                "row_width_distribution": json.dumps(raw_stats["row_width_distribution"], ensure_ascii=False),
            })

        file_metadata_df = pd.DataFrame(file_metadata)
        file_metadata_df
        """
    ),
    code(
        r"""
        for spec in datasets:
            print(f"\n{spec.title}")
            display(loaded_data[spec.key].head(3))
            display(pd.DataFrame({
                "column_name": loaded_data[spec.key].columns,
                "dtype": loaded_data[spec.key].dtypes.astype(str).values,
                "missing": loaded_data[spec.key].isna().sum().values,
            }).head(25))
        """
    ),
    md(
        """
        ## 3. Data Quality Analysis

        This section calculates row/column counts, missingness, exact duplicate rows, empty strings, malformed row counts from CSV parsing, and duplicate text values in likely text columns.
        """
    ),
    code(
        r"""
        def exact_duplicate_rows(data_frame: pd.DataFrame) -> int:
            return int(data_frame.fillna("").duplicated().sum())


        def looks_numeric_or_codes(series: pd.Series) -> bool:
            values = [str(value) for value in series.dropna().head(500)]
            if not values:
                return False
            numeric_like = sum(bool(re.fullmatch(r"[0-9,.; /#-]+", value)) for value in values)
            short = sum(len(value) <= 5 for value in values)
            return numeric_like / len(values) > 0.75 or short / len(values) > 0.85


        def unique_ratio(series: pd.Series) -> float:
            nonempty = series.dropna()
            return float(nonempty.nunique() / len(nonempty)) if len(nonempty) else 0.0


        def assess_text_column(data_frame: pd.DataFrame, column_name: str) -> dict:
            series = data_frame[column_name]
            nonempty = series.dropna()
            lengths = nonempty.astype(str).map(len)
            lower_name = column_name.lower()
            question_like_name = bool(re.search(r"q\d+|question|comment|idea|why|please|specify|miss|like|improve|concern|suggest|participat|important|prompt|subject|other", lower_name))
            average_length = float(lengths.mean()) if len(lengths) else 0.0
            maximum_length = int(lengths.max()) if len(lengths) else 0
            is_free_text = bool(len(nonempty) and not looks_numeric_or_codes(series) and (average_length >= 20 or maximum_length >= 80 or question_like_name))
            is_policy_relevant = bool(
                is_free_text
                and re.search(r"comment|idea|question|why|miss|concern|suggest|important|improve|prompt|subject|open government|participat|other|specify", lower_name)
            )
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
                "column_name": column_name,
                "non_empty_responses": int(len(nonempty)),
                "missing_responses": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "avg_length": round(average_length, 2),
                "max_length": maximum_length,
                "unique_values": int(nonempty.nunique()),
                "field_type": field_type,
                "policy_relevance_recommendation": "review_for_policy_concerns_suggestions_or_requested_actions"
                if is_policy_relevant
                else "do_not_treat_as_policy_comment_without_manual_review",
                "representative_samples_redacted": json.dumps([redact_text(value) for value in nonempty.head(5)], ensure_ascii=False),
            }


        text_field_rows = []
        quality_rows = []

        for metadata in file_metadata:
            key = metadata["dataset_name"]
            df = loaded_data[key]
            field_assessments = []
            for column_name in df.columns:
                row = assess_text_column(df, column_name)
                row["dataset_name"] = key
                field_assessments.append(row)
                text_field_rows.append(row)

            fields_df = pd.DataFrame(field_assessments)
            response_like = fields_df[
                (fields_df["field_type"] == "free_text_candidate")
                & fields_df["column_name"].str.contains(
                    r"comment|question or idea|practical idea|public engagement question|og is important|improvements|did we miss|please tell us why|knowledge or experience|other|specify",
                    case=False,
                    regex=True,
                    na=False,
                )
            ]
            duplicate_text_entries = 0
            for column_name in response_like["column_name"]:
                keys = df[column_name].map(normalized_text_key).dropna()
                duplicate_text_entries += int(len(keys) - keys.nunique())

            quality_rows.append({
                "dataset_name": key,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "exact_duplicate_rows": exact_duplicate_rows(df),
                "duplicate_text_entries": duplicate_text_entries,
                "free_text_candidate_fields": int((fields_df["field_type"] == "free_text_candidate").sum()),
                "empty_or_missing_columns": int((fields_df["field_type"] == "empty_or_missing").sum()),
            })

        text_fields_df = pd.DataFrame(text_field_rows)
        quality_summary_df = pd.DataFrame(quality_rows)
        quality_summary_df
        """
    ),
    code(
        r"""
        for key, df in loaded_data.items():
            missing = df.isna().sum().sort_values(ascending=False)
            missing_pct = (df.isna().mean() * 100).round(2)
            display(pd.DataFrame({"missing": missing, "missing_pct": missing_pct[missing.index]}).head(20).rename_axis(f"{key} columns"))
        """
    ),
    code(
        r"""
        useful_fields = text_fields_df[
            text_fields_df["policy_relevance_recommendation"] == "review_for_policy_concerns_suggestions_or_requested_actions"
        ].copy()
        display(useful_fields[["dataset_name", "column_name", "non_empty_responses", "avg_length", "max_length", "field_type"]])

        plot_data = useful_fields.sort_values("non_empty_responses", ascending=True).tail(18)
        ax = plot_data.plot.barh(x="column_name", y="non_empty_responses", legend=False, figsize=(11, 8))
        ax.set_title("Non-empty responses in recommended review fields")
        ax.set_xlabel("Non-empty responses")
        ax.set_ylabel("Field")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        for _, row in useful_fields.head(12).iterrows():
            print(f"\n{row['dataset_name']} :: {row['column_name']}")
            samples = json.loads(row["representative_samples_redacted"])
            for sample in samples[:3]:
                print("-", sample)
        """
    ),
    md(
        """
        ## 4. Identify Policy-Relevant Text

        The recommended fields are candidates for later human annotation because they may contain concerns, suggestions, requested actions, or explanations. Checkbox, numeric, categorical, and metadata fields remain useful for context but are not treated as policy comments.
        """
    ),
    code(
        r"""
        field_recommendation_summary = (
            text_fields_df.groupby(["dataset_name", "field_type", "policy_relevance_recommendation"])
            .size()
            .reset_index(name="field_count")
            .sort_values(["dataset_name", "field_count"], ascending=[True, False])
        )
        field_recommendation_summary
        """
    ),
    md(
        """
        ## 5. Dataset Overlap Analysis

        We compare schemas, ID-like columns, exact duplicate response text, and near-duplicate candidates. These are review aids only; the datasets are not merged.
        """
    ),
    code(
        r"""
        def response_text_entries(key: str, df: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
            pattern = re.compile(
                r"comment|question or idea|practical idea|public engagement question|og is important|improvements|did we miss|please tell us why|knowledge or experience|other|specify",
                re.IGNORECASE,
            )
            relevant_columns = [
                row["column_name"]
                for _, row in fields[fields["dataset_name"] == key].iterrows()
                if row["field_type"] == "free_text_candidate" and pattern.search(str(row["column_name"]))
            ]
            rows = []
            for column_name in relevant_columns:
                for index, value in df[column_name].items():
                    text_key = normalized_text_key(value)
                    if text_key:
                        rows.append({
                            "dataset_name": key,
                            "source_row_index": int(index),
                            "source_row_number": int(index + 1),
                            "column_name": column_name,
                            "text_key": text_key,
                            "text_sample_redacted": redact_text(value),
                        })
            return pd.DataFrame(rows)


        all_text_entries_df = pd.concat(
            [response_text_entries(key, df, text_fields_df) for key, df in loaded_data.items()],
            ignore_index=True,
        )

        duplicate_text_entries_df = all_text_entries_df[
            all_text_entries_df["text_key"].duplicated(keep=False)
        ].sort_values(["text_key", "dataset_name", "source_row_number", "column_name"])

        def schema_overlap() -> pd.DataFrame:
            rows = []
            column_sets = {key: {normalized_text_key(col) for col in df.columns} for key, df in loaded_data.items()}
            keys = list(loaded_data)
            for left_index, left_key in enumerate(keys):
                for right_key in keys[left_index + 1:]:
                    common = sorted(value for value in column_sets[left_key] & column_sets[right_key] if value)
                    rows.append({
                        "left_dataset": left_key,
                        "right_dataset": right_key,
                        "left_columns": len(loaded_data[left_key].columns),
                        "right_columns": len(loaded_data[right_key].columns),
                        "common_normalized_columns": len(common),
                        "common_column_names": json.dumps(common[:50], ensure_ascii=False),
                    })
            return pd.DataFrame(rows)


        def identifier_overlap() -> pd.DataFrame:
            id_values = {}
            for key, df in loaded_data.items():
                for column_name in df.columns:
                    lower_name = column_name.lower()
                    if lower_name in {"id", "row/ligne"} or "identifier" in lower_name or "rang" in lower_name:
                        values = {str(value) for value in df[column_name].dropna() if clean_cell(value)}
                        id_values[(key, column_name)] = values
            rows = []
            keys = list(id_values)
            for left_index, left in enumerate(keys):
                for right in keys[left_index + 1:]:
                    common = sorted(id_values[left] & id_values[right])
                    if common:
                        rows.append({
                            "left_dataset": left[0],
                            "left_id_column": left[1],
                            "right_dataset": right[0],
                            "right_id_column": right[1],
                            "overlapping_ids": len(common),
                            "sample_ids": json.dumps(common[:20], ensure_ascii=False),
                        })
            return pd.DataFrame(rows)


        def near_duplicate_candidates(entries: pd.DataFrame, threshold=0.92, max_pairs=250) -> pd.DataFrame:
            candidates = []
            if entries.empty:
                return pd.DataFrame()
            sampled = entries[entries["text_key"].str.len().between(30, 500)].copy()
            sampled["block_key"] = sampled["text_key"].str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().str[:24]
            sampled["length_bucket"] = (sampled["text_key"].str.len() // 50).astype(int)
            sampled = sampled.sort_values(["block_key", "length_bucket", "text_key"]).head(5000)
            for _, block in sampled.groupby(["block_key", "length_bucket"], dropna=True):
                records = block.head(120).to_dict("records")
                for left_index, left in enumerate(records):
                    for right in records[left_index + 1:]:
                        if abs(len(left["text_key"]) - len(right["text_key"])) > 80:
                            continue
                        if left["dataset_name"] == right["dataset_name"] and left["source_row_number"] == right["source_row_number"]:
                            continue
                        if left["text_key"] == right["text_key"]:
                            continue
                        ratio = SequenceMatcher(None, left["text_key"], right["text_key"]).ratio()
                        if ratio >= threshold:
                            candidates.append({
                                "left_dataset": left["dataset_name"],
                                "left_row": left["source_row_number"],
                                "left_column": left["column_name"],
                                "right_dataset": right["dataset_name"],
                                "right_row": right["source_row_number"],
                                "right_column": right["column_name"],
                                "similarity": round(ratio, 3),
                                "left_sample_redacted": left["text_sample_redacted"],
                                "right_sample_redacted": right["text_sample_redacted"],
                            })
                            if len(candidates) >= max_pairs:
                                return pd.DataFrame(candidates)
            return pd.DataFrame(candidates)


        schema_overlap_df = schema_overlap()
        identifier_overlap_df = identifier_overlap()
        near_duplicates_df = near_duplicate_candidates(all_text_entries_df)

        display(schema_overlap_df)
        display(identifier_overlap_df)
        display(duplicate_text_entries_df.head(20))
        display(near_duplicates_df.head(20))
        """
    ),
    md(
        """
        ## 6. Conservative Cleaning

        The cleaned outputs keep each source separate. They add provenance columns, normalize column names in the processed copy, preserve original text values with conservative whitespace cleanup, and flag duplicate response text instead of deleting it.
        """
    ),
    code(
        r"""
        def assign_duplicate_groups(keys: pd.Series) -> list:
            counts = Counter(key for key in keys if key is not None)
            duplicate_keys = [key for key, count in counts.items() if count > 1]
            group_for_key = {key: f"dup_{index + 1:06d}" for index, key in enumerate(duplicate_keys)}
            return [group_for_key.get(key) for key in keys]


        cleaned_output_paths = []
        row_count_rows = []
        column_mapping_frames = []

        for spec in datasets:
            key = spec.key
            original_df = loaded_data[key]
            seen = Counter()
            normalized_columns = [slugify_column(column, seen) for column in original_df.columns]
            clean_df = original_df.copy()
            clean_df.columns = normalized_columns
            clean_df.insert(0, "source_dataset", key)
            clean_df.insert(1, "source_file", spec.path.relative_to(PROJECT_ROOT).as_posix())
            clean_df.insert(2, "source_row_number", range(spec.header_rows + 1, spec.header_rows + 1 + len(clean_df)))

            # Per-row duplicate flag based on all response-like text in that row.
            row_text_keys = []
            response_columns = [
                row["column_name"]
                for _, row in text_fields_df[text_fields_df["dataset_name"] == key].iterrows()
                if row["field_type"] == "free_text_candidate"
                and re.search(
                    r"comment|question or idea|practical idea|public engagement question|og is important|improvements|did we miss|please tell us why|knowledge or experience|other|specify",
                    str(row["column_name"]),
                    re.IGNORECASE,
                )
            ]
            for _, source_row in original_df.iterrows():
                combined = " | ".join([clean_cell(source_row.get(column)) or "" for column in response_columns]).strip()
                row_text_keys.append(normalized_text_key(combined) if combined else None)
            clean_df.insert(3, "duplicate_text_group_id", assign_duplicate_groups(pd.Series(row_text_keys)))
            clean_df.insert(4, "exclusion_reason", pd.NA)

            output_path = DATA_PROCESSED / f"{key}_source_clean.csv"
            clean_df.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
            cleaned_output_paths.append(output_path)

            mapping = pd.DataFrame({
                "dataset_name": key,
                "original_column_name": original_df.columns,
                "normalized_column_name": normalized_columns,
            })
            mapping_path = DATA_INTERIM / f"{key}_column_mapping.csv"
            mapping.to_csv(mapping_path, index=False, encoding="utf-8", lineterminator="\n")
            header_rows_by_dataset[key].to_csv(DATA_INTERIM / f"{key}_header_rows.csv", index=False, header=False, encoding="utf-8", lineterminator="\n")
            column_mapping_frames.append(mapping)

            row_count_rows.append({
                "dataset_name": key,
                "input_rows": len(original_df),
                "source_clean_rows": len(clean_df),
                "excluded_rows": int(clean_df["exclusion_reason"].notna().sum()),
            })

        row_count_verification_df = pd.DataFrame(row_count_rows)
        row_count_verification_df
        """
    ),
    md(
        """
        ## 7. Audit Trail and Reports

        The notebook writes machine-readable CSV summaries plus Markdown reports. These reports are regenerated from the current data each time the notebook is run.
        """
    ),
    code(
        r"""
        inventory_df = file_metadata_df.merge(quality_summary_df, on="dataset_name", how="left")
        inventory_df["source_url_or_record"] = "Local file provided by user; original source URL not documented in workspace."
        inventory_df["license_provenance_status"] = "Unknown from local files; verify original open-data source/license before redistribution."
        inventory_df["recommended_use"] = "Consultation audit and later human annotation for policy concerns, suggestions, or requested actions."
        inventory_df["limitations"] = "Do not merge or train yet; overlap, malformed row-width anomalies, and source semantics require review."

        inventory_df.to_csv(REPORTS_DIR / "data_inventory.csv", index=False, encoding="utf-8", lineterminator="\n")
        text_fields_df.to_csv(DATA_INTERIM / "government_text_field_assessment.csv", index=False, encoding="utf-8", lineterminator="\n")
        all_text_entries_df.to_csv(DATA_INTERIM / "government_response_text_entries.csv", index=False, encoding="utf-8", lineterminator="\n")
        duplicate_text_entries_df.to_csv(DATA_INTERIM / "government_duplicate_text_entries.csv", index=False, encoding="utf-8", lineterminator="\n")
        schema_overlap_df.to_csv(DATA_INTERIM / "government_schema_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
        identifier_overlap_df.to_csv(DATA_INTERIM / "government_id_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
        near_duplicates_df.to_csv(DATA_INTERIM / "government_near_duplicate_candidates.csv", index=False, encoding="utf-8", lineterminator="\n")
        row_count_verification_df.to_csv(DATA_INTERIM / "government_row_count_verification.csv", index=False, encoding="utf-8", lineterminator="\n")

        checksum_df = pd.DataFrame([
            {"dataset_name": spec.key, "relative_path": spec.path.relative_to(PROJECT_ROOT).as_posix(), "sha256": raw_checksums_before[spec.key]}
            for spec in datasets
        ])
        checksum_df.to_csv(DATA_INTERIM / "government_raw_file_checksums.csv", index=False, encoding="utf-8", lineterminator="\n")

        def markdown_table(df: pd.DataFrame) -> str:
            if df.empty:
                return "No rows produced."
            columns = list(df.columns)
            lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
            for _, row in df.iterrows():
                values = [str(row[column]).replace("\n", " ") for column in columns]
                lines.append("| " + " | ".join(values) + " |")
            return "\n".join(lines)

        quality_lines = [
            "# PolicyLens Data Quality Report",
            "",
            "Generated by `notebooks/01_dataset_audit_and_cleaning.ipynb`.",
            "",
            "## Confirmed Findings",
            "",
        ]
        for _, row in inventory_df.iterrows():
            useful = useful_fields[useful_fields["dataset_name"] == row["dataset_name"]]
            quality_lines.extend([
                f"### {row['dataset_title']}",
                f"- Path: `{row['relative_path']}`",
                f"- Size: {row['file_size_bytes']} bytes; encoding: `{row['encoding']}`; delimiter: `{row['delimiter']}`",
                f"- Header: {row['header_structure']}",
                f"- Rows/columns: {row['rows']} / {row['columns']}",
                f"- Blank rows: {row['blank_rows']}; malformed row-width records: {row['malformed_records']}",
                f"- Exact duplicate rows: {row['exact_duplicate_rows']}; duplicate response-text entries: {row['duplicate_text_entries']}",
                f"- SHA-256: `{row['sha256']}`",
                f"- Clean output: `data/processed/{row['dataset_name']}_source_clean.csv`",
                f"- Recommended review fields: {len(useful)}",
                "",
            ])
            for _, field in useful.head(12).iterrows():
                quality_lines.append(f"  - `{field['column_name']}`: {field['non_empty_responses']} non-empty responses")
                for sample in json.loads(field["representative_samples_redacted"])[:2]:
                    if sample:
                        quality_lines.append(f"    Sample: {sample}")
            quality_lines.append("")
        quality_lines.extend([
            "## Limitations",
            "",
            "- Source URLs and explicit license records are not present in this workspace.",
            "- Malformed row-width counts indicate records requiring manual CSV inspection.",
            "- Near-duplicates are heuristic review candidates, not automatic exclusions.",
            "- No model-training labels were created.",
        ])
        (REPORTS_DIR / "data_quality_report.md").write_text("\n".join(quality_lines), encoding="utf-8")

        overlap_lines = [
            "# Dataset Overlap Report",
            "",
            "Generated by `notebooks/01_dataset_audit_and_cleaning.ipynb`.",
            "",
            "## Schema Overlap",
            "",
            markdown_table(schema_overlap_df),
            "",
            "## Identifier Overlap",
            "",
            markdown_table(identifier_overlap_df) if not identifier_overlap_df.empty else "No overlapping identifier values were found among detected ID-like columns.",
            "",
            "## Exact Duplicate Response Text",
            "",
            f"Duplicate response-text rows for review: {len(duplicate_text_entries_df)}.",
            "Output: `data/interim/government_duplicate_text_entries.csv`.",
            "",
            "## Near-Duplicate Candidates",
            "",
            f"Heuristic near-duplicate candidate pairs: {len(near_duplicates_df)}.",
            "Output: `data/interim/government_near_duplicate_candidates.csv`.",
            "",
            "## Assessment",
            "",
            "The compilation dataset appears to overlap with the individual-comments dataset by row identifiers and should not be merged until source semantics and duplicate records have been reviewed.",
        ]
        (REPORTS_DIR / "dataset_overlap_report.md").write_text("\n".join(overlap_lines), encoding="utf-8")

        cleaning_log = [
            "# Cleaning Log",
            "",
            "Generated by `notebooks/01_dataset_audit_and_cleaning.ipynb`.",
            "",
            "Raw government CSV files were not modified, renamed, moved, or deleted.",
            "",
            "Transformations applied:",
            "",
            "- Located the three government consultation CSVs by filename/content hints.",
            "- Detected encoding and delimiter for each file.",
            "- Loaded the questionnaire with two header rows and preserved both header rows.",
            "- Standardized blank strings and common empty markers to null in processed copies.",
            "- Applied conservative whitespace and line-ending normalization only.",
            "- Added source dataset, source file, and source row number to cleaned outputs.",
            "- Normalized column names in processed outputs and wrote column mapping files.",
            "- Flagged duplicate response text groups; no duplicate records were deleted.",
            "",
            "Exclusions:",
            "",
            "- No rows were excluded from source-clean outputs.",
        ]
        (REPORTS_DIR / "cleaning_log.md").write_text("\n".join(cleaning_log), encoding="utf-8")

        inventory_df[["dataset_name", "rows", "columns", "exact_duplicate_rows", "duplicate_text_entries", "free_text_candidate_fields"]]
        """
    ),
    md(
        """
        ## 8. Validation

        These assertions are intentionally strict. If one fails, the notebook should stop so the issue can be reviewed instead of producing misleading outputs.
        """
    ),
    code(
        r"""
        # Raw files must remain byte-for-byte unchanged.
        for spec in datasets:
            assert sha256(spec.path) == raw_checksums_before[spec.key], f"Raw file changed: {spec.path}"

        # Cleaned outputs must preserve row counts.
        for _, row in row_count_verification_df.iterrows():
            assert int(row["input_rows"]) == int(row["source_clean_rows"]), f"Row-count mismatch for {row['dataset_name']}"
            assert int(row["excluded_rows"]) == 0, f"Unexpected exclusions for {row['dataset_name']}"

        # Required audit artifacts must exist.
        required_outputs = [
            REPORTS_DIR / "data_inventory.csv",
            REPORTS_DIR / "data_quality_report.md",
            REPORTS_DIR / "dataset_overlap_report.md",
            REPORTS_DIR / "cleaning_log.md",
            DATA_INTERIM / "government_raw_file_checksums.csv",
            DATA_INTERIM / "government_row_count_verification.csv",
        ] + cleaned_output_paths
        for path in required_outputs:
            assert path.exists() and path.stat().st_size > 0, f"Missing or empty output: {path}"

        print("Validation passed.")
        print(f"Datasets audited: {len(datasets)}")
        print(f"Cleaned outputs written: {len(cleaned_output_paths)}")
        """
    ),
    md(
        """
        ## Final Notebook Summary

        Use `reports/data_quality_report.md` for the readable audit narrative, `reports/dataset_overlap_report.md` for overlap findings, and the CSV files in `data/interim/` for manual review of text fields, duplicates, and near-duplicates. The next research step is to manually review overlap/duplicate candidates and define an annotation guide before creating any modeling dataset.
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
