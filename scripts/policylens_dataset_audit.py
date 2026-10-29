"""PolicyLens phase-1 dataset audit and conservative curation.

The script profiles datasets already present in this workspace, writes
traceable reports, and creates safe cleaned/canonical CSVs without modifying
the original files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

SENTIMENT140_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
TEXT_COLUMNS = {
    "policylens_sentiment": "sentence",
    "sentiment140_train": "text",
    "sentiment140_test": "text",
    "sentimix_train": "text",
    "sentimix_test": "text",
    "hinemo9_full": "text",
    "hinemo9_train": "text",
    "hinemo9_dev": "text",
    "hinemo9_test": "text",
    "hinemo9_gold": "text",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    rel_path: str
    format: str
    task_type: str
    documented_source: str
    license_status: str
    recommended_use: str
    limitations: str

    @property
    def path(self) -> Path:
        return ROOT / self.rel_path


DATASETS = [
    DatasetSpec(
        "policylens_sentiment",
        "PolicyLens_Hindi_English_Hinglish_Sentiment.csv",
        "csv",
        "sentiment_classification",
        "Unknown; no local README/source record found for this file.",
        "Unknown; not documented locally.",
        "Small local multilingual sentiment example set; usable for smoke tests or annotation-format experiments only.",
        "Not public-consultation data; provenance/license unknown; do not use as real-world test evidence.",
    ),
    DatasetSpec(
        "sentiment140_train",
        "trainingandtestdata/training.1600000.processed.noemoticon.csv",
        "csv_no_header",
        "sentiment_classification",
        "Sentiment140 training file name; original URL/license not documented in workspace.",
        "Unknown locally; verify before redistribution.",
        "Large English Twitter sentiment training resource for baseline experiments only.",
        "Twitter domain mismatch; labels are tweet polarity, not public consultation concerns or demands.",
    ),
    DatasetSpec(
        "sentiment140_test",
        "trainingandtestdata/testdata.manual.2009.06.14.csv",
        "csv_no_header",
        "sentiment_classification",
        "Sentiment140 manual test file name; original URL/license not documented in workspace.",
        "Unknown locally; verify before redistribution.",
        "English Twitter sentiment benchmark for polarity experiments only.",
        "Twitter domain mismatch; labels are polarity; includes entity-specific historical tweets.",
    ),
    DatasetSpec(
        "sentimix_train",
        "SentiMix-Hindi-English-main/SentiMix-Hindi-English-main/train_conll.txt",
        "conll",
        "sentiment_classification",
        "Local README: Kaggle Competition SentiMix-Hindi-English.",
        "Unknown locally; README does not state a license.",
        "Code-mixed Hindi-English sentiment training data for domain-transfer experiments.",
        "Tweet domain; token language tags are useful, but sentiment labels should not be treated as consultation stance.",
    ),
    DatasetSpec(
        "sentimix_test",
        "SentiMix-Hindi-English-main/SentiMix-Hindi-English-main/Hindi_test_unalbelled_conll_updated.txt",
        "conll_with_labels",
        "sentiment_classification",
        "Local README: Kaggle Competition SentiMix-Hindi-English.",
        "Unknown locally; README does not state a license.",
        "Code-mixed Hindi-English sentiment test data when joined with local test_labels_hinglish.txt.",
        "Tweet domain; labels are sentiment; label file provenance/license not documented.",
    ),
    DatasetSpec(
        "hinemo9_full",
        "hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/full.jsonl",
        "jsonl",
        "emotion_classification",
        "Local README/dataset card: public YouTube comments, HinEmo-9.",
        "Verified locally: CC BY 4.0 for annotations/taxonomy/splits/curation; underlying comments remain third-party/platform text.",
        "Emotion and language-variety research; evaluate on human_gold only.",
        "Emotion labels are not sentiment labels; 27,073 labels are model-derived; YouTube domain mismatch.",
    ),
    DatasetSpec(
        "hinemo9_train",
        "hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/train.jsonl",
        "jsonl",
        "emotion_classification",
        "Local README/dataset card: HinEmo-9 fixed train split.",
        "Verified locally: CC BY 4.0 for annotations/taxonomy/splits/curation; underlying comments remain third-party/platform text.",
        "Training split for emotion models; includes model-derived labels.",
        "Do not evaluate final systems on model-derived labels.",
    ),
    DatasetSpec(
        "hinemo9_dev",
        "hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/dev.jsonl",
        "jsonl",
        "emotion_classification",
        "Local README/dataset card: HinEmo-9 fixed dev split.",
        "Verified locally: CC BY 4.0 for annotations/taxonomy/splits/curation; underlying comments remain third-party/platform text.",
        "Gold-only development split for emotion models.",
        "Emotion domain only; over-samples scarce classes.",
    ),
    DatasetSpec(
        "hinemo9_test",
        "hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/test.jsonl",
        "jsonl",
        "emotion_classification",
        "Local README/dataset card: HinEmo-9 fixed test split.",
        "Verified locally: CC BY 4.0 for annotations/taxonomy/splits/curation; underlying comments remain third-party/platform text.",
        "Gold-only held-out emotion evaluation.",
        "Not a public-consultation test set; do not report as PolicyLens concern extraction performance.",
    ),
    DatasetSpec(
        "hinemo9_gold",
        "hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/gold.jsonl",
        "jsonl",
        "emotion_classification",
        "Local README/dataset card: HinEmo-9 human_gold subset.",
        "Verified locally: CC BY 4.0 for annotations/taxonomy/splits/curation; underlying comments remain third-party/platform text.",
        "Human-labeled evaluation/analysis subset for emotion studies.",
        "Prior-exposure caveat in dataset card; not consultation data.",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:200_000]
    candidates = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for enc in candidates:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.strip()
    return text or None


def normalized_duplicate_key(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    return re.sub(r"\s+", " ", text.casefold()).strip()


def read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def read_sentimix_conll(path: Path, labels_path: Path | None = None) -> pd.DataFrame:
    labels = {}
    if labels_path and labels_path.exists():
        label_df = pd.read_csv(labels_path)
        labels = dict(zip(label_df["Uid"].astype(str), label_df["Sentiment"].astype(str)))

    rows = []
    current_id: str | None = None
    current_label: str | None = None
    tokens: list[str] = []
    lang_tags: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_label, tokens, lang_tags
        if current_id is None:
            return
        rows.append(
            {
                "uid": current_id,
                "text": " ".join(tokens).strip(),
                "token_count": len(tokens),
                "token_language_tags": " ".join(lang_tags),
                "sentiment": current_label if current_label is not None else labels.get(current_id),
            }
        )
        current_id = None
        current_label = None
        tokens = []
        lang_tags = []

    with path.open("r", encoding=detect_encoding(path), errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if parts[0] == "meta":
                flush()
                current_id = parts[1] if len(parts) > 1 else None
                current_label = parts[2] if len(parts) > 2 else None
            elif len(parts) >= 2:
                tokens.append(parts[0])
                lang_tags.append(parts[1])
        flush()
    return pd.DataFrame(rows)


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    enc = detect_encoding(spec.path)
    if spec.format == "jsonl":
        return read_jsonl(spec.path), "utf-8"
    if spec.name.startswith("sentiment140"):
        for candidate in [enc, "utf-8", "cp1252", "latin-1"]:
            try:
                return (
                    pd.read_csv(spec.path, encoding=candidate, header=None, names=SENTIMENT140_COLUMNS),
                    candidate,
                )
            except UnicodeDecodeError:
                continue
        return (
            pd.read_csv(spec.path, encoding="latin-1", header=None, names=SENTIMENT140_COLUMNS),
            "latin-1",
        )
    if spec.name == "sentimix_train":
        return read_sentimix_conll(spec.path), enc
    if spec.name == "sentimix_test":
        labels_path = spec.path.parent / "test_labels_hinglish.txt"
        return read_sentimix_conll(spec.path, labels_path), enc
    for candidate in ["utf-8-sig", "utf-8", enc, "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(spec.path, encoding=candidate)
            df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
            if spec.name == "policylens_sentiment" and "sentence" not in df.columns:
                continue
            return df, candidate
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(spec.path, encoding="latin-1")
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    return df, "latin-1"


def dtype_summary(df: pd.DataFrame) -> str:
    return json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}, ensure_ascii=False)


def missing_summary(df: pd.DataFrame) -> str:
    return json.dumps(df.isna().sum().astype(int).to_dict(), ensure_ascii=False)


def distribution_summary(df: pd.DataFrame, columns: list[str]) -> str:
    output = {}
    for col in columns:
        if col in df.columns:
            output[col] = df[col].fillna("<NA>").astype(str).value_counts(dropna=False).head(25).to_dict()
    return json.dumps(output, ensure_ascii=False)


def exact_duplicate_rows(df: pd.DataFrame) -> int:
    stable_rows = df.map(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value)
    return int(stable_rows.duplicated().sum())


def assign_duplicate_groups(keys: pd.Series) -> list[str | None]:
    counts = Counter(key for key in keys if key is not None)
    duplicate_keys = [key for key, count in counts.items() if count > 1]
    group_for_key = {key: f"dup_{idx + 1:06d}" for idx, key in enumerate(duplicate_keys)}
    return [group_for_key.get(key) for key in keys]


def canonicalize(spec: DatasetSpec, df: pd.DataFrame) -> pd.DataFrame:
    text_col = TEXT_COLUMNS.get(spec.name)
    rows = pd.DataFrame()
    rows["record_id"] = [f"{spec.name}:{i + 1}" for i in range(len(df))]
    rows["source_dataset"] = spec.name
    source_id_col = "id" if "id" in df.columns else "ids" if "ids" in df.columns else "uid" if "uid" in df.columns else None
    rows["source_record_id"] = df[source_id_col].astype(str) if source_id_col else pd.NA
    rows["consultation_id"] = pd.NA
    rows["question_id"] = pd.NA
    rows["question_text"] = pd.NA
    rows["original_text"] = df[text_col] if text_col and text_col in df.columns else pd.NA
    rows["cleaned_text"] = rows["original_text"].map(clean_text)
    rows["language"] = df["language_label"] if "language_label" in df.columns else pd.NA
    rows["task_type"] = spec.task_type
    if "label" in df.columns:
        rows["label"] = df["label"]
    elif "sentiment" in df.columns:
        rows["label"] = df["sentiment"]
    elif "target" in df.columns:
        rows["label"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"}).fillna(df["target"])
    else:
        rows["label"] = pd.NA
    rows["is_synthetic"] = False
    keys = rows["cleaned_text"].map(normalized_duplicate_key)
    rows["duplicate_group_id"] = assign_duplicate_groups(keys)
    rows["exclusion_reason"] = rows["cleaned_text"].map(lambda x: "empty_text" if x is None else pd.NA)
    return rows


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def profile_dataset(spec: DatasetSpec, df: pd.DataFrame, encoding: str, checksum: str) -> dict[str, Any]:
    text_col = TEXT_COLUMNS.get(spec.name)
    duplicate_text_records = 0
    empty_text_records = 0
    if text_col and text_col in df.columns:
        keys = df[text_col].map(normalized_duplicate_key)
        duplicate_text_records = int(keys.notna().sum() - keys.dropna().nunique())
        empty_text_records = int(keys.isna().sum())
    return {
        "dataset_name": spec.name,
        "relative_path": spec.rel_path,
        "source_url_or_record": spec.documented_source,
        "format": spec.format,
        "encoding": encoding,
        "sha256": checksum,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": json.dumps(list(df.columns), ensure_ascii=False),
        "dtypes": dtype_summary(df),
        "missing_values": missing_summary(df),
        "exact_duplicate_rows": exact_duplicate_rows(df),
        "duplicate_text_records": duplicate_text_records,
        "empty_text_records": empty_text_records,
        "language_distribution": distribution_summary(df, ["language", "language_label", "lang", "token_language_tags"]),
        "label_distribution": distribution_summary(df, ["label", "sentiment", "target", "label_source"]),
        "license_provenance_status": spec.license_status,
        "recommended_use": spec.recommended_use,
        "limitations": spec.limitations,
    }


def write_duplicate_review(spec: DatasetSpec, canonical: pd.DataFrame) -> None:
    dupes = canonical[canonical["duplicate_group_id"].notna()].copy()
    if dupes.empty:
        dupes = canonical.iloc[0:0].copy()
    write_csv(dupes, INTERIM / f"{spec.name}_duplicate_text_review.csv")


def write_source_specific(spec: DatasetSpec, df: pd.DataFrame) -> None:
    out = df.copy()
    text_col = TEXT_COLUMNS.get(spec.name)
    if text_col and text_col in out.columns:
        out["cleaned_text"] = out[text_col].map(clean_text)
    write_csv(out, PROCESSED / f"{spec.name}_source_clean.csv")


def write_reports(inventory: list[dict[str, Any]], row_counts: list[dict[str, Any]]) -> None:
    inv_df = pd.DataFrame(inventory)
    write_csv(inv_df, REPORTS / "data_inventory.csv")
    write_csv(pd.DataFrame(row_counts), INTERIM / "row_count_verification.csv")

    lines = [
        "# PolicyLens Data Quality Report",
        "",
        "Generated by `python scripts/policylens_dataset_audit.py`.",
        "",
        "## Workspace Findings",
        "",
        "- No PolicyLens A-to-Z project report was found in this workspace.",
        "- No Open Government Consultation 2017-18 Get Involved Questionnaire file was found.",
        "- No Open Government Consultation 2016-18 Individual Comments file was found.",
        "- Existing local datasets are auxiliary sentiment/emotion datasets, not public consultation corpora.",
        "",
        "## Dataset Inventory Summary",
        "",
    ]
    for item in inventory:
        lines.extend(
            [
                f"### {item['dataset_name']}",
                f"- Path: `{item['relative_path']}`",
                f"- Rows/columns: {item['rows']} / {item['columns']}",
                f"- Encoding: {item['encoding']}; SHA-256: `{item['sha256']}`",
                f"- Exact duplicate rows: {item['exact_duplicate_rows']}; duplicate text records: {item['duplicate_text_records']}; empty text records: {item['empty_text_records']}",
                f"- Labels: `{item['label_distribution']}`",
                f"- Languages: `{item['language_distribution']}`",
                f"- License/provenance: {item['license_provenance_status']}",
                f"- Recommended use: {item['recommended_use']}",
                f"- Limitations: {item['limitations']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Policy-Relevant Free Text",
            "",
            "No verified public-consultation free-text fields were found locally. The detected text fields are auxiliary NLP text fields: `sentence` in the PolicyLens sentiment CSV, `text` in Sentiment140 and HinEmo-9, and reconstructed `text` from SentiMix token blocks.",
            "",
            "## Suitability Before Model Training",
            "",
            "- Do not train or evaluate PolicyLens concern extraction on these files as if they were consultation submissions.",
            "- Use HINEMO9 for emotion research only, with human_gold/dev/test for evaluation.",
            "- Use SentiMix and Sentiment140 only for sentiment/domain-transfer baselines after license verification.",
            "- Keep synthetic or locally authored examples separate from real-world consultation data.",
            "",
            "## Unresolved Questions",
            "",
            "- Locate and ingest the Open Government Consultation questionnaire and individual-comments datasets if they exist outside this folder.",
            "- Verify source URLs and redistribution licenses for SentiMix, Sentiment140, and `PolicyLens_Hindi_English_Hinglish_Sentiment.csv`.",
            "- Define consultation-specific annotation guidelines before model training.",
            "",
        ]
    )
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")

    log_lines = [
        "# Cleaning Log",
        "",
        "Raw input files were not modified, renamed, moved, or deleted.",
        "",
        "Transformations applied:",
        "",
        "- Detected readable encoding from utf-8-sig, utf-8, cp1252, latin-1 candidates.",
        "- Computed SHA-256 checksums for every profiled input file.",
        "- Reconstructed SentiMix sentence text from CoNLL token blocks and preserved token language tags.",
        "- Joined SentiMix test labels from `test_labels_hinglish.txt` by `Uid`.",
        "- Mapped Sentiment140 targets 0/2/4 to negative/neutral/positive in canonical outputs only.",
        "- Created `cleaned_text` with line-ending normalization, whitespace collapse, and trimming only.",
        "- Preserved `original_text` in canonical outputs.",
        "- Marked empty text records with `exclusion_reason=empty_text`; no rows were silently dropped.",
        "- Detected exact duplicate rows and normalized exact duplicate text; near-duplicates were not deleted.",
        "",
        "Exclusions:",
        "",
        "- No records were excluded from processed outputs.",
    ]
    (REPORTS / "cleaning_log.md").write_text("\n".join(log_lines), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    row_counts: list[dict[str, Any]] = []
    all_canonical = []
    checksums = []

    for spec in DATASETS:
        if not spec.path.exists():
            continue
        checksum = sha256(spec.path)
        df, encoding = load_dataset(spec)
        before_rows = len(df)
        profile = profile_dataset(spec, df, encoding, checksum)
        inventory.append(profile)
        checksums.append({"dataset_name": spec.name, "relative_path": spec.rel_path, "sha256": checksum})

        canonical = canonicalize(spec, df)
        write_source_specific(spec, df)
        write_csv(canonical, PROCESSED / f"{spec.name}_canonical.csv")
        write_duplicate_review(spec, canonical)
        all_canonical.append(canonical)

        row_counts.append(
            {
                "dataset_name": spec.name,
                "input_rows": before_rows,
                "source_clean_rows": before_rows,
                "canonical_rows": len(canonical),
                "excluded_rows": int(canonical["exclusion_reason"].notna().sum()),
            }
        )

    if all_canonical:
        write_csv(pd.concat(all_canonical, ignore_index=True), PROCESSED / "all_detected_datasets_canonical.csv")

    write_csv(pd.DataFrame(checksums), INTERIM / "raw_file_checksums.csv")
    (INTERIM / "dataset_profiles.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(inventory, row_counts)
    print(f"Profiled {len(inventory)} datasets.")
    print(f"Wrote reports to {REPORTS.relative_to(ROOT)} and outputs to {PROCESSED.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
