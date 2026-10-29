"""
scripts/analyze_concern_review_agreement.py
===========================================
Post-review merger and inter-annotator agreement analysis for the
PolicyLens concern-coherence human-review package.

Usage (once completed reviewer files exist):
    python scripts/analyze_concern_review_agreement.py

Expected inputs  (never modified):
    data/comparisons/multilingual_embedding_450/human_review/pair_catalog.csv
    data/comparisons/multilingual_embedding_450/human_review/completed/reviewer_1_completed.csv
    data/comparisons/multilingual_embedding_450/human_review/completed/reviewer_2_completed.csv

Outputs  (written to a new directory, never overwriting originals):
    data/comparisons/multilingual_embedding_450/agreement_analysis/
        merged_judgments.csv
        agreement_report.md
        agreement_metrics.json
        stratum_summary.csv

Safety rules:
  - Reads completed files ONLY from human_review/completed/.
  - Never modifies blank templates or pair_catalog.
  - Refuses to run if completed/ or either file is missing.
  - Refuses to overwrite an existing agreement_analysis/ output directory.
  - Does not fabricate labels or infer missing ratings.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REVIEW_DIR = (
    PROJECT_ROOT
    / "data"
    / "comparisons"
    / "multilingual_embedding_450"
    / "human_review"
)
CATALOG_PATH = REVIEW_DIR / "pair_catalog.csv"
COMPLETED_DIR = REVIEW_DIR / "completed"
OUTPUT_BASE = (
    PROJECT_ROOT
    / "data"
    / "comparisons"
    / "multilingual_embedding_450"
    / "agreement_analysis"
)

EXPECTED_PAIR_COUNT = 44
VALID_CONCERN_RELATION = {"0", "1", "2", "3", "U"}
VALID_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_COMPLETED_COLUMNS = [
    "pair_id",
    "reviewer_id",
    "concern_relation",
    "confidence",
    "shared_concern_note",
    "evidence_note",
]
MODELS = [
    "all_minilm_l6_v2",
    "paraphrase_multilingual_minilm_l12_v2",
]


# ── Validation helpers ────────────────────────────────────────────────────────


def _load_completed_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_pair_ids(
    df: pd.DataFrame,
    expected_pair_ids: set[str],
    reviewer_label: str,
) -> list[str]:
    errors: list[str] = []
    found_set = set(df["pair_id"].tolist())
    missing = expected_pair_ids - found_set
    extra = found_set - expected_pair_ids
    duplicates = df["pair_id"][df["pair_id"].duplicated()].tolist()
    if missing:
        errors.append(f"{reviewer_label}: missing pair_ids: {sorted(missing)}")
    if extra:
        errors.append(f"{reviewer_label}: unexpected pair_ids: {sorted(extra)}")
    if duplicates:
        errors.append(f"{reviewer_label}: duplicate pair_ids: {sorted(duplicates)}")
    return errors


def validate_judgment_fields(
    df: pd.DataFrame,
    reviewer_label: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    cr = df["concern_relation"].str.strip()
    blank_cr = int((cr == "").sum())
    valid_cr = int(cr.isin(VALID_CONCERN_RELATION).sum())
    invalid_cr_vals = cr[~cr.isin(VALID_CONCERN_RELATION) & (cr != "")].unique().tolist()
    u_count = int((cr == "U").sum())
    numeric_count = int(cr.isin({"0", "1", "2", "3"}).sum())
    if invalid_cr_vals:
        errors.append(
            f"{reviewer_label} concern_relation has invalid values: {invalid_cr_vals}. "
            "Permitted: 0, 1, 2, 3, U."
        )
    conf = df["confidence"].str.strip().str.lower()
    blank_conf = int((conf == "").sum())
    invalid_conf = conf[(conf != "") & ~conf.isin(VALID_CONFIDENCE)].unique().tolist()
    if invalid_conf:
        errors.append(
            f"{reviewer_label} confidence has invalid values: {invalid_conf}. "
            "Permitted: high, medium, low."
        )
    stats: dict[str, Any] = {
        "concern_relation_blank": blank_cr,
        "concern_relation_valid": valid_cr,
        "concern_relation_numeric": numeric_count,
        "concern_relation_U": u_count,
        "concern_relation_invalid": invalid_cr_vals,
        "confidence_blank": blank_conf,
        "confidence_invalid": invalid_conf,
        "shared_concern_note_filled": int((df["shared_concern_note"].str.strip() != "").sum()),
        "evidence_note_filled": int((df["evidence_note"].str.strip() != "").sum()),
    }
    return errors, stats


def _norm_cell(value: str) -> str:
    """Spreadsheet round-trips may turn in-cell line breaks into CRLF; nothing else is tolerated."""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def validate_source_text(df: pd.DataFrame, catalog: pd.DataFrame, reviewer_label: str) -> list[str]:
    """Pair text and source references must be unchanged relative to pair_catalog.csv.

    Catches edited comments and spreadsheet corruption (for example a comment starting with '@' that a
    spreadsheet turned into a formula error such as '#NAME?'), which would otherwise flow into the
    merged judgments unnoticed.
    """
    judgment = set(REQUIRED_COMPLETED_COLUMNS)
    cols = [c for c in df.columns if c not in judgment and c in catalog.columns]
    if not cols:
        return []
    a = df.set_index("pair_id")[cols].astype(str).map(_norm_cell)
    b = catalog.set_index("pair_id").reindex(a.index)[cols].fillna("").astype(str).map(_norm_cell)
    diff = (a != b)
    if not diff.values.any():
        return []
    cells = [f"{pid}/{col}" for pid, row in diff.iterrows() for col in cols if row[col]]
    return [f"{reviewer_label}: pair text or source references were modified in {len(cells)} cell(s), "
            f"e.g. {cells[:5]}. Restore them from the blank template (open the CSV with all columns as text)."]


# ── Merging ───────────────────────────────────────────────────────────────────


def merge_reviewer_judgments(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    judgment_cols = ["reviewer_id", "concern_relation", "confidence",
                     "shared_concern_note", "evidence_note"]
    text_cols = [c for c in r1.columns if c not in judgment_cols + ["pair_id"]]
    r1_j = r1[["pair_id"] + judgment_cols].rename(columns={c: f"{c}_r1" for c in judgment_cols})
    r2_j = r2[["pair_id"] + judgment_cols].rename(columns={c: f"{c}_r2" for c in judgment_cols})
    text = r1[["pair_id"] + text_cols]
    return text.merge(r1_j, on="pair_id").merge(r2_j, on="pair_id")


# ── Agreement metrics ─────────────────────────────────────────────────────────


def compute_exact_agreement(merged: pd.DataFrame) -> dict[str, Any]:
    cr1 = merged["concern_relation_r1"].str.strip()
    cr2 = merged["concern_relation_r2"].str.strip()
    is_num_r1 = cr1.isin({"0", "1", "2", "3"})
    is_num_r2 = cr2.isin({"0", "1", "2", "3"})
    both_numeric = is_num_r1 & is_num_r2
    both_u = (cr1 == "U") & (cr2 == "U")
    one_u = ((cr1 == "U") | (cr2 == "U")) & ~both_u
    either_blank = (cr1 == "") | (cr2 == "")
    jointly_numeric = int(both_numeric.sum())
    exact_agree = int((cr1[both_numeric] == cr2[both_numeric]).sum())
    return {
        "total_pairs": len(merged),
        "jointly_numeric": jointly_numeric,
        "exact_agree": exact_agree,
        "exact_agree_rate": (
            round(float(exact_agree) / jointly_numeric, 4)
            if jointly_numeric > 0 else None
        ),
        "both_U": int(both_u.sum()),
        "one_U": int(one_u.sum()),
        "either_blank": int(either_blank.sum()),
        "excluded_from_kappa": int(len(merged) - jointly_numeric),
    }


def compute_weighted_kappa(merged: pd.DataFrame) -> dict[str, Any]:
    cr1 = merged["concern_relation_r1"].str.strip()
    cr2 = merged["concern_relation_r2"].str.strip()
    numeric_mask = cr1.isin({"0", "1", "2", "3"}) & cr2.isin({"0", "1", "2", "3"})
    y1 = cr1[numeric_mask].astype(int).tolist()
    y2 = cr2[numeric_mask].astype(int).tolist()
    n = len(y1)
    if n < 5:
        return {
            "status": "insufficient_data",
            "n": n,
            "kappa": None,
            "po_weighted": None,
            "pe_weighted": None,
            "message": (
                f"Only {n} jointly numeric pair(s) — fewer than 5. "
                "Weighted kappa is not reported to avoid distortion."
            ),
        }
    categories = [0, 1, 2, 3]
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1
    weights = np.array(
        [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    )
    po_w = float(np.sum(weights * matrix) / n)
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    pe_w = float(np.sum(weights * expected) / n)
    if np.isclose(pe_w, 1.0) or np.isclose(1.0 - pe_w, 0.0):
        return {
            "status": "zero_variance",
            "n": n,
            "kappa": None,
            "po_weighted": round(po_w, 4),
            "pe_weighted": round(pe_w, 4),
            "message": (
                "Pe_weighted ≈ 1.0 (all ratings in one category). "
                "Weighted kappa is mathematically undefined."
            ),
            "confusion_matrix": matrix.astype(int).tolist(),
            "category_labels": categories,
        }
    kappa = float((po_w - pe_w) / (1.0 - pe_w))
    return {
        "status": "computed",
        "n": n,
        "kappa": round(kappa, 4),
        "po_weighted": round(po_w, 4),
        "pe_weighted": round(pe_w, 4),
        "message": None,
        "confusion_matrix": matrix.astype(int).tolist(),
        "category_labels": categories,
    }


# ── Stratum breakdown ─────────────────────────────────────────────────────────


def build_stratum_summary(
    merged_with_catalog: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    rows = []
    for model in models:
        selected_col = f"{model}_selected"
        if selected_col not in merged_with_catalog.columns:
            continue
        selected_mask = merged_with_catalog[selected_col].fillna(False).astype(bool)
        sub = merged_with_catalog[selected_mask].copy()
        rel_col = f"{model}_selection_relationship"
        lang_col = f"{model}_selection_language_pair"
        tier_col = f"{model}_selection_tier"
        missing_cols = [c for c in [rel_col, lang_col, tier_col] if c not in sub.columns]
        if missing_cols:
            continue
        for (rel, lang, tier), grp_idx in sub.groupby([rel_col, lang_col, tier_col]).groups.items():
            grp = sub.loc[grp_idx]
            cr1_g = grp["concern_relation_r1"].str.strip()
            cr2_g = grp["concern_relation_r2"].str.strip()
            both_g = cr1_g.isin({"0", "1", "2", "3"}) & cr2_g.isin({"0", "1", "2", "3"})
            n_jointly = int(both_g.sum())
            exact = int((cr1_g[both_g] == cr2_g[both_g]).sum()) if n_jointly > 0 else 0
            rows.append({
                "model": model,
                "relationship": rel,
                "language_pair": lang,
                "tier": tier,
                "n_pairs": len(grp),
                "n_jointly_numeric": n_jointly,
                "exact_agree_in_stratum": exact,
                "exact_agree_rate": (
                    round(float(exact) / n_jointly, 3) if n_jointly > 0 else None
                ),
                "mean_concern_r1": (
                    round(cr1_g[both_g].astype(int).mean(), 2) if n_jointly > 0 else None
                ),
                "mean_concern_r2": (
                    round(cr2_g[both_g].astype(int).mean(), 2) if n_jointly > 0 else None
                ),
            })
    return pd.DataFrame(rows)


# ── Report ────────────────────────────────────────────────────────────────────


def _kappa_interpretation(kappa: float) -> str:
    if kappa < 0:
        return "poor (less than chance)"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


def write_markdown_report(
    output_dir: Path,
    r1_stats: dict[str, Any],
    r2_stats: dict[str, Any],
    exact_stats: dict[str, Any],
    kappa_result: dict[str, Any],
    stratum_df: pd.DataFrame,
    r1_path: Path,
    r2_path: Path,
) -> Path:
    report_path = output_dir / "agreement_report.md"
    k = kappa_result.get("kappa")
    interp = _kappa_interpretation(k) if k is not None else "N/A"

    if kappa_result.get("confusion_matrix") is not None:
        labels = kappa_result["category_labels"]
        header = "| R1 \\ R2 | " + " | ".join(str(l) for l in labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        cm_rows = [
            "| **%s** | " % labels[i] + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(kappa_result["confusion_matrix"])
        ]
        cm_block = "\n".join([header, sep] + cm_rows)
    else:
        cm_block = "_Not available._"

    if not stratum_df.empty:
        strat_header = (
            "| Model | Relationship | Language pair | Tier | N pairs "
            "| N jointly rated | Exact agree rate | Mean R1 | Mean R2 |"
        )
        strat_sep = "|---|---|---|---|---:|---:|---:|---:|---:|"
        strat_rows = []
        for _, row in stratum_df.iterrows():
            strat_rows.append(
                "| %s | %s | %s | %s | %d | %d | %s | %s | %s |" % (
                    row["model"], row["relationship"], row["language_pair"], row["tier"],
                    row["n_pairs"], row["n_jointly_numeric"],
                    row["exact_agree_rate"] if row["exact_agree_rate"] is not None else "N/A",
                    row["mean_concern_r1"] if row["mean_concern_r1"] is not None else "N/A",
                    row["mean_concern_r2"] if row["mean_concern_r2"] is not None else "N/A",
                )
            )
        strat_block = "\n".join([strat_header, strat_sep] + strat_rows)
    else:
        strat_block = "_No stratum data available._"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# PolicyLens — Concern-Coherence Review: Inter-Annotator Agreement Report",
        "",
        f"_Generated: {now}_  ",
        f"_Reviewer 1 file: `{r1_path.name}`_  ",
        f"_Reviewer 2 file: `{r2_path.name}`_",
        "",
        "---",
        "",
        "## 1. Reviewer Completion Status",
        "",
        "| Field | Reviewer 1 | Reviewer 2 |",
        "|---|---:|---:|",
        f"| Total pairs | {r1_stats['total_pairs']} | {r2_stats['total_pairs']} |",
        f"| concern_relation filled (numeric + U) | {r1_stats['concern_relation_valid']} | {r2_stats['concern_relation_valid']} |",
        f"| concern_relation numeric (0-3) | {r1_stats['concern_relation_numeric']} | {r2_stats['concern_relation_numeric']} |",
        f"| concern_relation = U | {r1_stats['concern_relation_U']} | {r2_stats['concern_relation_U']} |",
        f"| concern_relation blank | {r1_stats['concern_relation_blank']} | {r2_stats['concern_relation_blank']} |",
        f"| confidence filled | {r1_stats['confidence_filled']} | {r2_stats['confidence_filled']} |",
        f"| shared_concern_note filled | {r1_stats['shared_concern_note_filled']} | {r2_stats['shared_concern_note_filled']} |",
        f"| evidence_note filled | {r1_stats['evidence_note_filled']} | {r2_stats['evidence_note_filled']} |",
        "",
        "---",
        "",
        "## 2. Pair Coverage",
        "",
        f"- **Total unique pairs:** {exact_stats['total_pairs']}",
        f"- **Jointly rated numeric (0-3):** {exact_stats['jointly_numeric']}",
        f"- **Excluded from numeric kappa:** {exact_stats['excluded_from_kappa']}",
        f"  - Both rated U: {exact_stats['both_U']}",
        f"  - One rated U: {exact_stats['one_U']}",
        f"  - Either blank: {exact_stats['either_blank']}",
        "",
        "---",
        "",
        "## 3. Exact Agreement (ordinal 0-3, jointly rated pairs only)",
        "",
        f"- **Exact matches:** {exact_stats['exact_agree']} / {exact_stats['jointly_numeric']}",
        f"- **Exact agreement rate:** {exact_stats['exact_agree_rate'] if exact_stats['exact_agree_rate'] is not None else 'N/A'}",
        "",
        "---",
        "",
        "## 4. Linearly Weighted Cohen's kappa",
        "",
        f"- **Status:** {kappa_result['status']}",
        f"- **N (jointly numeric pairs):** {kappa_result['n']}",
        f"- **kappa (linearly weighted):** {k if k is not None else 'N/A'}",
        f"- **Weighted Po:** {kappa_result.get('po_weighted', 'N/A')}",
        f"- **Weighted Pe:** {kappa_result.get('pe_weighted', 'N/A')}",
        f"- **Interpretation (Landis & Koch):** {interp}",
    ]
    if kappa_result.get("message"):
        lines += ["", f"> WARNING: {kappa_result['message']}"]
    lines += [
        "",
        "### 4x4 Confusion Matrix (rows = Reviewer 1, columns = Reviewer 2)",
        "",
        cm_block,
        "",
        "> Pairs where either reviewer gave U or left concern_relation blank are",
        "> excluded from the matrix and the kappa calculation.",
        "",
        "---",
        "",
        "## 5. Per-Model Stratum Breakdown (post-unblinding)",
        "",
        "> NOTE: Pairs selected by both models appear in both models' rows (4 overlap pairs).",
        "> Inter-reviewer agreement is the same human judgment regardless of which model",
        "> nominated the pair. Do not interpret stratum agreement rates as model accuracy.",
        "",
        strat_block,
        "",
        "---",
        "",
        "## 6. Limitations",
        "",
        "- This sample is a stratified diagnostic set (score extremes), not random.",
        "- Inter-reviewer agreement measures reviewer consistency, not model quality.",
        "- Stratum-level findings (especially French-French) are based on very few pairs.",
        "- Model comparison using these ratings requires a separate analysis step.",
        "",
        "---",
        "",
        "_Raw annotations are preserved unchanged in human_review/completed/._",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────


def run_analysis(
    completed_dir: Path | None = None,
    catalog_path: Path | None = None,
    output_base: Path | None = None,
) -> dict[str, Any]:
    completed_dir = completed_dir or COMPLETED_DIR
    catalog_path = catalog_path or CATALOG_PATH
    output_base = output_base or OUTPUT_BASE

    r1_path = completed_dir / "reviewer_1_completed.csv"
    r2_path = completed_dir / "reviewer_2_completed.csv"

    missing_files = []
    if not completed_dir.exists():
        missing_files.append(str(completed_dir))
    else:
        if not r1_path.exists():
            missing_files.append(str(r1_path))
        if not r2_path.exists():
            missing_files.append(str(r2_path))

    if missing_files:
        msg = (
            "\n[PolicyLens] Completed reviewer files not found.\n"
            f"Expected:\n  {r1_path}\n  {r2_path}\n\n"
            f"Place completed CSVs in: {completed_dir}\n\n"
            "Blank templates remain untouched at:\n"
            f"  {REVIEW_DIR / 'reviewer_1_template.csv'}\n"
            f"  {REVIEW_DIR / 'reviewer_2_template.csv'}\n"
        )
        print(msg, file=sys.stderr)
        return {"status": "no_completed_files", "missing": missing_files}

    r1 = _load_completed_file(r1_path)
    r2 = _load_completed_file(r2_path)

    all_errors: list[str] = []
    for label, df in [("Reviewer 1", r1), ("Reviewer 2", r2)]:
        missing_cols = [c for c in REQUIRED_COMPLETED_COLUMNS if c not in df.columns]
        if missing_cols:
            all_errors.append(f"{label}: missing columns {missing_cols}")
    if all_errors:
        for e in all_errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        return {"status": "validation_failed", "errors": all_errors}

    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    expected_pair_ids = set(catalog["pair_id"].tolist())

    id_errors: list[str] = []
    id_errors += validate_pair_ids(r1, expected_pair_ids, "Reviewer 1")
    id_errors += validate_pair_ids(r2, expected_pair_ids, "Reviewer 2")
    if id_errors:
        for e in id_errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        return {"status": "validation_failed", "errors": id_errors}

    field_errors_r1, r1_field_stats = validate_judgment_fields(r1, "Reviewer 1")
    field_errors_r2, r2_field_stats = validate_judgment_fields(r2, "Reviewer 2")
    field_errors = field_errors_r1 + field_errors_r2
    field_errors += validate_source_text(r1, catalog, "Reviewer 1") + validate_source_text(r2, catalog, "Reviewer 2")
    if field_errors:
        for e in field_errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        return {"status": "validation_failed", "errors": field_errors}

    r1_stats = {
        "total_pairs": len(r1),
        "confidence_filled": int((r1["confidence"].str.strip() != "").sum()),
        **r1_field_stats,
    }
    r2_stats = {
        "total_pairs": len(r2),
        "confidence_filled": int((r2["confidence"].str.strip() != "").sum()),
        **r2_field_stats,
    }

    merged = merge_reviewer_judgments(r1, r2)
    exact_stats = compute_exact_agreement(merged)
    kappa_result = compute_weighted_kappa(merged)

    catalog_typed = pd.read_csv(catalog_path)
    catalog_typed["pair_id"] = catalog_typed["pair_id"].astype(str)
    merged_with_catalog = merged.merge(
        catalog_typed,
        on="pair_id",
        how="left",
        suffixes=("", "_cat"),
    )
    stratum_df = build_stratum_summary(merged_with_catalog, MODELS)

    if output_base.exists():
        msg = (
            f"\n[PolicyLens] Output directory already exists: {output_base}\n"
            "Refusing to overwrite existing agreement analysis.\n"
            "Move or rename the existing directory before re-running.\n"
        )
        print(msg, file=sys.stderr)
        return {"status": "output_exists", "output_dir": str(output_base)}

    output_base.mkdir(parents=True)

    merged.to_csv(output_base / "merged_judgments.csv", index=False)
    stratum_df.to_csv(output_base / "stratum_summary.csv", index=False)

    report_path = write_markdown_report(
        output_base, r1_stats, r2_stats, exact_stats,
        kappa_result, stratum_df, r1_path, r2_path,
    )

    metrics: dict[str, Any] = {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_1_file": str(r1_path),
        "reviewer_2_file": str(r2_path),
        "reviewer_1_stats": r1_stats,
        "reviewer_2_stats": r2_stats,
        "exact_agreement": exact_stats,
        "weighted_kappa": kappa_result,
        "output_dir": str(output_base),
        "files_written": [
            str(output_base / "merged_judgments.csv"),
            str(output_base / "stratum_summary.csv"),
            str(report_path),
            str(output_base / "agreement_metrics.json"),
        ],
    }
    (output_base / "agreement_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print("\n[PolicyLens] Agreement analysis complete.")
    print(f"  Output directory : {output_base}")
    print(f"  Jointly rated    : {exact_stats['jointly_numeric']} / {exact_stats['total_pairs']} pairs")
    kv = kappa_result.get("kappa")
    if kv is not None:
        print(f"  Weighted kappa   : {kv:.4f}  ({_kappa_interpretation(kv)})")
    else:
        print(f"  Weighted kappa   : {kappa_result['status']} -- {kappa_result.get('message', '')}")

    return metrics


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):      # messages contain non-ASCII (e.g. "≈"); Windows consoles are cp1252
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    result = run_analysis()
    if result.get("status") not in {"success", None}:
        sys.exit(1)
