"""
scripts/create_agreement_analysis_notebook.py
=============================================
Generates notebooks/03_annotation_agreement_analysis.ipynb.
Follows all project guidelines:
  1. Reads ONLY the completed files data/annotation/completed/pilot_annotator_{A,B}_completed.csv;
     stops with an instruction if either is missing (never falls back to blank templates).
  2. Validates schema, distinct annotator IDs, verbatim evidence spans and word limits,
     then pilot IDs match exactly without duplication or omission.
  3. Validates permitted stance labels (Support, Oppose, Mixed, Neutral, Unclear).
  4. Computes Cohen's kappa only on valid paired records.
  5. Reports included vs. excluded records with exact reasons.
  6. Produces confusion matrix and record-level disagreement table.
  7. Avoids presenting misleading kappa if N < 5 or zero variance; explains limitation.
  8. Treats concern and requested_action as free-text (no categorical kappa).
  9. Preserves original annotations; never automatically resolves disagreements.
 10. Clearly separates preliminary raw annotations from adjudicated labels.
"""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "03_annotation_agreement_analysis.ipynb"


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7",
        },
    }

    cells = []

    # ── Cell 1: Header ────────────────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""# PolicyLens — 03. Annotation Agreement Analysis & Adjudication Protocol

**Project:** PolicyLens — Public Consultation NLP Analysis  
**Phase:** Human Annotation Pilot Agreement & Calibration  
**Dataset:** 50 Canadian Open-Government Consultation Free-Text Responses (2016–2018)  

### Purpose and Protocol Overview
This notebook evaluates inter-annotator agreement between two independent annotators (**`annotator_A`** and **`annotator_B`**) on the 50-record human annotation pilot, labelled under `reports/annotation_guidelines_v1.1.md`.

#### Key Principles:
1. **Independent, Blind Annotation:** `annotator_A` (`data/annotation/completed/pilot_annotator_A_completed.csv`) and `annotator_B` (`data/annotation/completed/pilot_annotator_B_completed.csv`) label records independently without consulting each other or accessing each other's submissions before round completion. The blank templates (`annotator_1.csv`, `annotator_2.csv`) are never read by this notebook.
2. **Quality & Validation:** All annotations are checked for valid stance categories (`Support`, `Oppose`, `Mixed`, `Neutral`, `Unclear`), pilot ID alignment, and missingness.
3. **Statistical Agreement Metric:** Cohen's Kappa ($\kappa$) is computed for categorical `stance` only on records with valid, paired annotations from both annotators.
4. **Conservative Metric Safeguards:** Kappa is never computed or presented if there are insufficient valid pairs ($N < 5$) or if zero variation exists in marginal distributions ($P_e = 1.0$), with explanatory limitations stated.
5. **Free-Text Fields Treated Separately:** `concern` and `requested_action` are open-ended free-text fields; categorical kappa is mathematically inappropriate and will not be applied without a closed coding taxonomy.
6. **No Automated Resolution:** Disagreements are preserved and compiled into an adjudication agenda for human review, strictly separating preliminary raw annotations from adjudicated gold labels."""))

    # ── Cell 2: Imports & Environment ─────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"
PILOT_CSV = ANNOTATION_DIR / "pilot_annotation_50.csv"
COMPLETED_DIR = ANNOTATION_DIR / "completed"
ANNOTATOR_A_CSV = COMPLETED_DIR / "pilot_annotator_A_completed.csv"
ANNOTATOR_B_CSV = COMPLETED_DIR / "pilot_annotator_B_completed.csv"
EXPECTED_ANNOTATOR_IDS = {"A": "annotator_A", "B": "annotator_B"}
EVIDENCE_SPAN_MAX_CHARS = 200
WORD_LIMITS = {"concern": 15, "requested_action": 20}

# Permitted Stance Taxonomy
PERMITTED_STANCES = ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]
ANNOTATION_COLUMNS = ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]

print(f"Project root: {PROJECT_ROOT}")
print(f"Annotator A completed file: {ANNOTATOR_A_CSV} (exists: {ANNOTATOR_A_CSV.exists()})")
print(f"Annotator B completed file: {ANNOTATOR_B_CSV} (exists: {ANNOTATOR_B_CSV.exists()})")
print(f"Pilot reference: {PILOT_CSV} (exists: {PILOT_CSV.exists()})")"""))

    # ── Cell 3: Section 1 Markdown ────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Ingestion & Completion Status Check
We require the two completed files in `data/annotation/completed/`. If either is missing, the notebook **stops with an instruction** to finish annotation first; it does not fall back to the blank templates or to any other file.
If the completed files exist but contain no stance labels yet, the notebook reports the pending status."""))

    # ── Cell 4: Section 1 Code ────────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def check_completion_status(df: pd.DataFrame, annotator_name: str) -> dict:
    total = len(df)
    completed_stance = df["stance"].str.strip().ne("").sum()
    completed_concern = df["concern"].str.strip().ne("").sum()
    completed_action = df["requested_action"].str.strip().ne("").sum()
    completed_evidence = df["evidence_span"].str.strip().ne("").sum()
    annotator_ids = df["annotator_id"].str.strip().unique()
    annotator_ids = [aid for aid in annotator_ids if aid]
    
    return {
        "annotator": annotator_name,
        "total_records": total,
        "stance_filled": int(completed_stance),
        "concern_filled": int(completed_concern),
        "action_filled": int(completed_action),
        "evidence_filled": int(completed_evidence),
        "annotator_id": annotator_ids[0] if annotator_ids else "(not set)",
        "is_complete": completed_stance == total,
    }

missing_files = [p for p in (ANNOTATOR_A_CSV, ANNOTATOR_B_CSV) if not p.exists()]
if missing_files:
    raise FileNotFoundError(
        "Completed annotation file(s) not found:\\n"
        + "\\n".join(f"  - {p}" for p in missing_files)
        + "\\n\\nFinish annotation first: each annotator must complete a copy of the template following "
        "data/annotation/README_annotators.md and reports/annotation_guidelines_v1.1.md, "
        "and save it under data/annotation/completed/ as "
        "pilot_annotator_A_completed.csv and pilot_annotator_B_completed.csv. "
        "This notebook does not read the blank templates (annotator_1.csv / annotator_2.csv) "
        "and will not fall back to them."
    )

# df_a1 = annotator_A's completed file, df_a2 = annotator_B's completed file
df_a1 = pd.read_csv(ANNOTATOR_A_CSV, dtype=str, keep_default_na=False)
df_a2 = pd.read_csv(ANNOTATOR_B_CSV, dtype=str, keep_default_na=False)

status_1 = check_completion_status(df_a1, "Annotator A")
status_2 = check_completion_status(df_a2, "Annotator B")

print("=== Annotation Ingestion Status ===")
print("Annotator A:", status_1)
print("Annotator B:", status_2)

both_completed = (
    status_1.get("stance_filled", 0) > 0 and
    status_2.get("stance_filled", 0) > 0
)
if not both_completed:
    print("\\n[NOTE] Annotation round is currently pending or in progress.")
    print("The completed files contain no stance labels yet. The notebook will compute")
    print("inter-annotator agreement once both files contain labels.")"""))

    # ── Cell 4b: File-level validation ────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 1b: Completed-File Validation
Before any agreement statistic is computed, each completed file is checked against the pilot template and the v1/v1.1 rules:

**Blocking errors (analysis stops):**
1. Column names and column order identical to `pilot_annotation_50.csv`.
2. `annotator_id` set on every row, equal to `annotator_A` (file A) or `annotator_B` (file B), and the two IDs distinct.
3. `evidence_span`, when non-blank, is a verbatim substring of `original_text` and at most 200 characters.

**Warnings (reported, not blocking):**
4. `concern` longer than 15 words or `requested_action` longer than 20 words (v1 states these as preferred limits).

Invalid or blank `stance` values are not blocking; they are reported in Section 3 and excluded from kappa in Section 4."""))

    # ── Cell 4c: File-level validation code ───────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def validate_completed_file(df: pd.DataFrame, ref: pd.DataFrame, expected_id: str, label: str):
    # Returns (errors, warnings) for one completed annotation file.
    errors, warnings = [], []

    if list(df.columns) != list(ref.columns):
        missing = [c for c in ref.columns if c not in df.columns]
        extra = [c for c in df.columns if c not in ref.columns]
        errors.append(
            f"{label}: columns differ from template (missing={missing}, extra={extra}, "
            f"same_order={[c for c in df.columns if c in ref.columns] == [c for c in ref.columns if c in df.columns]})"
        )
        return errors, warnings  # cannot validate row content without the expected schema

    ids = df["annotator_id"].str.strip()
    blank_ids = int((ids == "").sum())
    if blank_ids:
        errors.append(f"{label}: annotator_id is blank on {blank_ids} row(s); it must be '{expected_id}' on every row.")
    wrong_ids = sorted(set(ids[ids != ""]) - {expected_id})
    if wrong_ids:
        errors.append(f"{label}: annotator_id must be '{expected_id}', found {wrong_ids}.")

    for pid, span, text in zip(df["pilot_id"], df["evidence_span"], df["original_text"]):
        if span.strip() == "":
            continue
        if span.strip() not in text:
            errors.append(f"{label}: {pid} evidence_span is not a verbatim substring of original_text.")
        if len(span) > EVIDENCE_SPAN_MAX_CHARS:
            errors.append(f"{label}: {pid} evidence_span has {len(span)} characters (maximum {EVIDENCE_SPAN_MAX_CHARS}).")

    for col, limit in WORD_LIMITS.items():
        for pid, value in zip(df["pilot_id"], df[col]):
            n_words = len(value.split())
            if n_words > limit:
                warnings.append(f"{label}: {pid} {col} has {n_words} words (preferred maximum {limit}).")

    return errors, warnings


def validate_completed_files(df_a: pd.DataFrame, df_b: pd.DataFrame, ref: pd.DataFrame):
    errors, warnings = [], []
    for df, key in ((df_a, "A"), (df_b, "B")):
        e, w = validate_completed_file(df, ref, EXPECTED_ANNOTATOR_IDS[key], f"Annotator {key}")
        errors += e
        warnings += w
    id_a = set(df_a["annotator_id"].str.strip()) - {""} if "annotator_id" in df_a.columns else set()
    id_b = set(df_b["annotator_id"].str.strip()) - {""} if "annotator_id" in df_b.columns else set()
    if id_a and id_a == id_b:
        errors.append(f"Both files use the same annotator_id {sorted(id_a)}; the two annotators must have distinct IDs.")
    return errors, warnings


ref_df = pd.read_csv(PILOT_CSV, dtype=str, keep_default_na=False)
file_errors, file_warnings = validate_completed_files(df_a1, df_a2, ref_df)

print("=== Completed-File Validation ===")
print(f"Blocking errors: {len(file_errors)} | Warnings: {len(file_warnings)}")
for w in file_warnings[:10]:
    print("[WARN]", w)
if len(file_warnings) > 10:
    print(f"[WARN] ... {len(file_warnings) - 10} more warning(s)")
if file_errors:
    for err in file_errors:
        print("[ERROR]", err)
    raise ValueError(
        f"{len(file_errors)} blocking validation error(s) in the completed annotation files. "
        "Correct the files (never edit original_text or the template columns) and re-run."
    )"""))

    # ── Cell 5: Section 2 Markdown ────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Record Alignment & ID Integrity
Before analyzing agreement, we must verify that:
1. Both datasets contain exactly the expected number of records (50).
2. All `pilot_id` values match 1-to-1 in identical order.
3. No duplicate or missing `pilot_id` values exist.
4. The underlying `original_text` and source provenance match the reference pilot dataset."""))

    # ── Cell 6: Section 2 Code ────────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def validate_record_alignment(df1: pd.DataFrame, df2: pd.DataFrame, ref: pd.DataFrame) -> bool:
    errors = []
    if len(df1) != len(ref):
        errors.append(f"Annotator A record count mismatch: {len(df1)} != {len(ref)}")
    if len(df2) != len(ref):
        errors.append(f"Annotator B record count mismatch: {len(df2)} != {len(ref)}")
        
    if not (df1["pilot_id"] == ref["pilot_id"]).all():
        errors.append("Annotator A pilot_id sequence does not match reference pilot.")
    if not (df2["pilot_id"] == ref["pilot_id"]).all():
        errors.append("Annotator B pilot_id sequence does not match reference pilot.")
        
    if not (df1["original_text"] == ref["original_text"]).all():
        errors.append("Annotator A original_text was modified or desynchronized from reference.")
    if not (df2["original_text"] == ref["original_text"]).all():
        errors.append("Annotator B original_text was modified or desynchronized from reference.")
        
    if errors:
        print("[FAIL] Record Alignment Validation Errors:")
        for err in errors:
            print(" -", err)
        return False
    else:
        print(f"[PASS] Record Alignment Validated: All {len(ref)} pilot IDs, texts, and sequences match exactly.")
        return True

alignment_ok = validate_record_alignment(df_a1, df_a2, ref_df)"""))

    # ── Cell 7: Section 3 Markdown ────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Label Validation & Missingness Assessment
The annotation guideline defines a strict 5-class taxonomy for stance:
- `Support`
- `Oppose`
- `Mixed`
- `Neutral`
- `Unclear`

We validate all entered stance labels against this permitted taxonomy and report any invalid, misspelled, or missing entries."""))

    # ── Cell 8: Section 3 Code ────────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def audit_labels(df: pd.DataFrame, annotator_name: str) -> dict:
    stances = df["stance"].str.strip()
    blank_mask = stances.eq("")
    valid_mask = stances.isin(PERMITTED_STANCES)
    invalid_mask = (~valid_mask) & (~blank_mask)
    
    invalid_values = stances[invalid_mask].unique().tolist()
    
    return {
        "annotator": annotator_name,
        "total_records": len(df),
        "valid_count": int(valid_mask.sum()),
        "blank_count": int(blank_mask.sum()),
        "invalid_count": int(invalid_mask.sum()),
        "invalid_values": invalid_values,
        "valid_distribution": df.loc[valid_mask, "stance"].value_counts().to_dict(),
    }

audit_1 = audit_labels(df_a1, "Annotator A")
audit_2 = audit_labels(df_a2, "Annotator B")

print("=== Annotator A Label Audit ===")
print(audit_1)
print("\\n=== Annotator B Label Audit ===")
print(audit_2)"""))

    # ── Cell 9: Section 4 Markdown ────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Filtering Included vs. Excluded Records
To calculate Cohen's Kappa rigorously:
- **Included records ($N_{valid}$):** Records where **both** Annotator A and Annotator B provided a valid, non-missing stance label.
- **Excluded records ($N_{excluded}$):** Any record where either annotator left the stance field blank, or provided an unpermitted label.

We generate an explicit accounting table explaining why any record was excluded."""))

    # ── Cell 10: Section 4 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def filter_paired_records(df1: pd.DataFrame, df2: pd.DataFrame):
    s1 = df1["stance"].str.strip()
    s2 = df2["stance"].str.strip()
    
    v1 = s1.isin(PERMITTED_STANCES)
    v2 = s2.isin(PERMITTED_STANCES)
    
    both_valid = v1 & v2
    
    included_df = pd.DataFrame({
        "pilot_id": df1["pilot_id"][both_valid],
        "question_slug": df1["question_slug"][both_valid],
        "original_text": df1["original_text"][both_valid],
        "stance_annotator_A": s1[both_valid],
        "stance_annotator_B": s2[both_valid],
    })
    
    exclusion_records = []
    for i in range(len(df1)):
        pid = df1.iloc[i]["pilot_id"]
        val1 = s1.iloc[i]
        val2 = s2.iloc[i]
        
        reasons = []
        if val1 == "":
            reasons.append("Annotator A stance blank")
        elif val1 not in PERMITTED_STANCES:
            reasons.append(f"Annotator A invalid stance: '{val1}'")
            
        if val2 == "":
            reasons.append("Annotator B stance blank")
        elif val2 not in PERMITTED_STANCES:
            reasons.append(f"Annotator B invalid stance: '{val2}'")
            
        if reasons:
            exclusion_records.append({
                "pilot_id": pid,
                "annotator_A_stance": val1 if val1 else "(blank)",
                "annotator_B_stance": val2 if val2 else "(blank)",
                "exclusion_reason": "; ".join(reasons)
            })
            
    excluded_df = pd.DataFrame(exclusion_records)
    return included_df, excluded_df

included_pairs, excluded_pairs = filter_paired_records(df_a1, df_a2)

print(f"Total Pilot Records: {len(df_a1)}")
print(f"Included for Kappa Analysis: {len(included_pairs)}")
print(f"Excluded from Kappa Analysis: {len(excluded_pairs)}")

if not excluded_pairs.empty:
    print("\\nSample of Excluded Records:")
    print(excluded_pairs.head(10).to_string(index=False))"""))

    # ── Cell 11: Section 5 Markdown ───────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""## Section 5: Cohen's Kappa Calculation for Stance
Cohen's Kappa ($\kappa$) measures inter-annotator agreement for categorical items while accounting for the agreement expected by chance:

$$\\kappa = \\frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o$ is the observed proportional agreement:
  $$P_o = \\frac{\\sum_{k} n_{kk}}{N}$$
- $P_e$ is the expected agreement under statistical independence of annotators:
  $$P_e = \\sum_{k} \\left( \\frac{n_{1,k}}{N} \\times \\frac{n_{2,k}}{N} \\right)$$
- $N$ is the number of included paired records.

### Methodological Guardrails:
1. **Minimum Sample Threshold:** If $N < 5$, calculating $\kappa$ is statistically meaningless and prone to distortion. The notebook will report the limitation and withhold the metric.
2. **Degenerate Marginal Variation:** If all annotations fall into a single category, $P_e = 1.0$ and $1 - P_e = 0$, causing division by zero. The metric is undefined, and the limitation is explained.
3. **Target Calibration:** The PolicyLens calibration threshold is $\kappa \\ge 0.60$ (substantial agreement under Landis & Koch, 1977)."""))

    # ── Cell 12: Section 5 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def compute_cohen_kappa(y1: list[str], y2: list[str], categories: list[str] = PERMITTED_STANCES) -> dict:
    n = len(y1)
    if n != len(y2):
        raise ValueError("Annotator label lists must have identical length.")
        
    if n < 5:
        return {
            "status": "insufficient_data",
            "n": n,
            "message": f"Insufficient valid records ({n} < 5) to compute Cohen's kappa reliably.",
            "kappa": None,
            "po": None,
            "pe": None,
        }
        
    # Build contingency table across all categories
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    for a, b in zip(y1, y2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1
        
    po = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n ** 2))
    
    if np.isclose(pe, 1.0) or np.isclose(1.0 - pe, 0.0):
        return {
            "status": "zero_variance",
            "n": n,
            "message": "Expected agreement Pe = 1.0 (zero marginal variation). Kappa is mathematically undefined.",
            "kappa": None,
            "po": float(po),
            "pe": float(pe),
            "matrix": matrix,
        }
        
    kappa = float((po - pe) / (1.0 - pe))
    return {
        "status": "computed",
        "n": n,
        "kappa": kappa,
        "po": float(po),
        "pe": float(pe),
        "matrix": matrix,
        "target_met": kappa >= 0.60,
    }

# Execute calculation if data is available
if len(included_pairs) >= 5:
    res = compute_cohen_kappa(
        included_pairs["stance_annotator_A"].tolist(),
        included_pairs["stance_annotator_B"].tolist(),
        PERMITTED_STANCES,
    )
    print("=== Cohen's Kappa Results ===")
    print(f"Status: {res['status']}")
    print(f"Included records (N): {res['n']}")
    print(f"Observed Agreement (Po): {res['po']:.4f}")
    print(f"Expected Agreement (Pe): {res['pe']:.4f}")
    print(f"Cohen's Kappa (kappa): {res['kappa']:.4f}")
    print(f"Calibration Target Met (kappa >= 0.60): {res['target_met']}")
else:
    print("=== Cohen's Kappa Results ===")
    print(f"Status: Deferred (N_valid = {len(included_pairs)}).")
    print("Explanation: Annotators have not yet completed and submitted their labels.")
    print("Once both files have at least 5 completed labels, kappa will compute automatically upon rerun.")"""))

    # ── Cell 13: Section 6 Markdown ───────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 6: Confusion Matrix & Disagreement Diagnosis
The confusion matrix highlights specific disagreement patterns between Annotator A and Annotator B (e.g., whether `Neutral` is frequently confused with `Unclear`, or whether `Mixed` is confused with `Support`/`Oppose`)."""))

    # ── Cell 14: Section 6 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def build_confusion_matrix(y1: list[str], y2: list[str], categories: list[str] = PERMITTED_STANCES) -> pd.DataFrame:
    cm = pd.DataFrame(0, index=[f"A_{c}" for c in categories], columns=[f"B_{c}" for c in categories])
    for a, b in zip(y1, y2):
        cm.loc[f"A_{a}", f"B_{b}"] += 1
    return cm

if len(included_pairs) > 0:
    cm_df = build_confusion_matrix(
        included_pairs["stance_annotator_A"].tolist(),
        included_pairs["stance_annotator_B"].tolist(),
        PERMITTED_STANCES
    )
    print("=== Stance Confusion Matrix (Annotator A rows vs. Annotator B columns) ===")
    print(cm_df)
else:
    print("[NOTE] Confusion matrix empty: waiting for completed annotations.")"""))

    # ── Cell 15: Section 7 Markdown ───────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Record-Level Disagreement Table for Adjudication
Disagreements between Annotator A and Annotator B are compiled into a dedicated review table.
Under no circumstances are disagreements automatically resolved; human adjudicators must review each flagged record alongside original text and annotator notes."""))

    # ── Cell 16: Section 7 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def compile_disagreements(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    disagreements = []
    for i in range(len(df1)):
        pid = df1.iloc[i]["pilot_id"]
        s1 = df1.iloc[i]["stance"].strip()
        s2 = df2.iloc[i]["stance"].strip()
        
        # Flag if both non-empty and differ, or if one is missing while the other is set
        if s1 != s2:
            disagreements.append({
                "pilot_id": pid,
                "source_dataset": df1.iloc[i]["source_dataset"],
                "question_slug": df1.iloc[i]["question_slug"],
                "original_text": df1.iloc[i]["original_text"],
                "annotator_A_stance": s1 if s1 else "(blank)",
                "annotator_B_stance": s2 if s2 else "(blank)",
                "annotator_A_notes": df1.iloc[i].get("review_notes", ""),
                "annotator_B_notes": df2.iloc[i].get("review_notes", ""),
                "adjudicated_stance": "",  # To be filled during adjudication
                "adjudication_rationale": "",
            })
    return pd.DataFrame(disagreements)

disagreements_df = compile_disagreements(df_a1, df_a2)
print(f"Total Disagreements / Differences Found: {len(disagreements_df)}")
if not disagreements_df.empty:
    print("\\nSample Disagreements Table:")
    print(disagreements_df[["pilot_id", "annotator_A_stance", "annotator_B_stance", "original_text"]].head(5).to_string(index=False))"""))

    # ── Cell 17: Section 8 Markdown ───────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""## Section 8: Free-Text Fields Analysis (`concern` & `requested_action`)
The `concern` and `requested_action` fields capture open-ended natural language phrases rather than closed categorical classes:
- **`concern`:** Concise noun phrase ($\le$ 15 words) or `none` / `unclear`.
- **`requested_action`:** Concise verb phrase ($\le$ 20 words) or `none` / `unclear`.

### Important Methodological Rule:
**Do not compute categorical Cohen's kappa for free-text strings.**
Categorical kappa assumes discrete, mutually exclusive classes. Applying kappa to raw string equality would drastically underestimate agreement because semantically identical concerns phrased with slight vocabulary differences (e.g., *"lack of data standards"* vs. *"inconsistent data standards"*) would be falsely penalized as independent category errors.

Instead, we compute:
1. Exact string equality rate.
2. Token-level Jaccard similarity across non-empty entries.
3. Qualitative alignment inspection."""))

    # ── Cell 18: Section 8 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""def token_jaccard(str1: str, str2: str) -> float:
    t1 = set(re.findall(r"\\w+", str1.lower()))
    t2 = set(re.findall(r"\\w+", str2.lower()))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def analyze_free_text_fields(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    concerns_1 = df1["concern"].str.strip()
    concerns_2 = df2["concern"].str.strip()
    actions_1 = df1["requested_action"].str.strip()
    actions_2 = df2["requested_action"].str.strip()
    
    both_concern_filled = (concerns_1 != "") & (concerns_2 != "")
    both_action_filled = (actions_1 != "") & (actions_2 != "")
    
    concern_exact = (concerns_1[both_concern_filled] == concerns_2[both_concern_filled]).sum()
    action_exact = (actions_1[both_action_filled] == actions_2[both_action_filled]).sum()
    
    concern_jaccards = [
        token_jaccard(c1, c2)
        for c1, c2 in zip(concerns_1[both_concern_filled], concerns_2[both_concern_filled])
    ]
    action_jaccards = [
        token_jaccard(a1, a2)
        for a1, a2 in zip(actions_1[both_action_filled], actions_2[both_action_filled])
    ]
    
    return {
        "concern_pairs_evaluated": int(both_concern_filled.sum()),
        "concern_exact_matches": int(concern_exact),
        "concern_mean_jaccard": float(np.mean(concern_jaccards)) if concern_jaccards else 0.0,
        "action_pairs_evaluated": int(both_action_filled.sum()),
        "action_exact_matches": int(action_exact),
        "action_mean_jaccard": float(np.mean(action_jaccards)) if action_jaccards else 0.0,
    }

free_text_metrics = analyze_free_text_fields(df_a1, df_a2)
print("=== Free-Text Qualitative Metrics ===")
for k, v in free_text_metrics.items():
    print(f"{k}: {v}")"""))

    # ── Cell 19: Section 9 Markdown ───────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## Section 9: Human Adjudication Protocol & Output Separation
### Clear Separation of Artifacts:
1. **Raw Annotations:** `data/annotation/completed/pilot_annotator_A_completed.csv` and `pilot_annotator_B_completed.csv` are strictly preserved as immutable historical records of independent labeling. This notebook never writes to them.
2. **Adjudication Round:** The project lead and annotators convene to review `disagreements_df`.
3. **Adjudicated Dataset:** Final resolved labels are saved to a separate file, `data/annotation/pilot_adjudicated_50.csv`, with documented rationales."""))

    # ── Cell 20: Section 9 Code ───────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""ADJUDICATED_CSV = ANNOTATION_DIR / "pilot_adjudicated_50.csv"

print("=== Next Steps for Human Annotators ===")
print("1. annotator_A: complete a copy of the template and save data/annotation/completed/pilot_annotator_A_completed.csv independently.")
print("2. annotator_B: complete a copy of the template and save data/annotation/completed/pilot_annotator_B_completed.csv independently.")
print("3. Re-run this notebook to compute Cohen's kappa and generate the disagreement table.")
print("4. Adjudication meeting: resolve flagged disagreements and export to:", ADJUDICATED_CSV)"""))

    nb.cells = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook to {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_notebook()
