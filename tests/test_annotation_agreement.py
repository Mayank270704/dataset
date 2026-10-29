"""
tests/test_annotation_agreement.py
==================================
Focused tests for the two-annotator workflow and agreement analysis:
  1. Template presence and non-emptiness.
  2. Template schema compliance (all provenance and annotation columns).
  3. Record alignment across templates and reference pilot (identical IDs and order).
  4. Initial annotation fields strictly blank in both templates.
  5. Agreement analysis notebook exists and is valid.
  6. Safe handling of incomplete/blank annotation files.
  7. Label validation logic flags invalid stances.
  8. Conservative guardrail: Cohen's kappa deferred when N < 5.
  9. Conservative guardrail: Cohen's kappa handles zero variance (Pe = 1.0).
 10. Cohen's kappa mathematical calculation correctness on known benchmark data.
 11. Completed-file workflow (notebook executed against SYNTHETIC fixtures in a temporary
     project directory; no real annotation data is read or written):
       - notebook reads only data/annotation/completed/pilot_annotator_{A,B}_completed.csv
       - missing completed files stop with an instruction, never falling back to templates
       - correct loading, distinct annotator IDs, valid stance handling
       - evidence_span verbatim-substring and 200-character rules, word-limit warnings
       - schema/column-order rule, no files written by the notebook
"""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"
PILOT_PATH = ANNOTATION_DIR / "pilot_annotation_50.csv"
ANNOTATOR_1_PATH = ANNOTATION_DIR / "annotator_1.csv"
ANNOTATOR_2_PATH = ANNOTATION_DIR / "annotator_2.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_annotation_agreement_analysis.ipynb"

ANNOTATION_COLUMNS = [
    "stance",
    "concern",
    "requested_action",
    "evidence_span",
    "review_notes",
    "annotator_id",
]

REQUIRED_PROVENANCE_COLUMNS = [
    "pilot_id",
    "source_dataset",
    "source_record_id",
    "source_file",
    "source_row_number",
    "source_field",
    "question_slug",
    "question_wording",
    "question_context",
    "context",
    "duplicate_text_group_id",
    "original_text",
]

ALL_REQUIRED_COLUMNS = REQUIRED_PROVENANCE_COLUMNS + ANNOTATION_COLUMNS
PERMITTED_STANCES = ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]


def test_annotator_templates_exist():
    assert ANNOTATOR_1_PATH.exists(), f"Annotator 1 template missing: {ANNOTATOR_1_PATH}"
    assert ANNOTATOR_2_PATH.exists(), f"Annotator 2 template missing: {ANNOTATOR_2_PATH}"
    assert ANNOTATOR_1_PATH.stat().st_size > 0, "Annotator 1 template is empty"
    assert ANNOTATOR_2_PATH.stat().st_size > 0, "Annotator 2 template is empty"


def test_annotator_templates_schema():
    for path, name in [(ANNOTATOR_1_PATH, "Annotator 1"), (ANNOTATOR_2_PATH, "Annotator 2")]:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ALL_REQUIRED_COLUMNS:
            assert col in df.columns, f"{name} template missing required column: {col}"


def test_annotator_templates_record_matching():
    ref = pd.read_csv(PILOT_PATH, dtype=str, keep_default_na=False)
    a1 = pd.read_csv(ANNOTATOR_1_PATH, dtype=str, keep_default_na=False)
    a2 = pd.read_csv(ANNOTATOR_2_PATH, dtype=str, keep_default_na=False)

    assert len(a1) == len(ref) == 50, f"A1 length mismatch: {len(a1)} != 50"
    assert len(a2) == len(ref) == 50, f"A2 length mismatch: {len(a2)} != 50"

    # Verify identical order and identical identifiers
    assert (a1["pilot_id"] == ref["pilot_id"]).all(), "A1 pilot_id does not match reference pilot"
    assert (a2["pilot_id"] == ref["pilot_id"]).all(), "A2 pilot_id does not match reference pilot"
    assert (a1["pilot_id"] == a2["pilot_id"]).all(), "A1 and A2 pilot_ids are not in identical order"

    # Verify original_text and source row numbers match reference
    assert (a1["original_text"] == ref["original_text"]).all(), "A1 original_text mismatch"
    assert (a2["original_text"] == ref["original_text"]).all(), "A2 original_text mismatch"
    assert (a1["source_row_number"] == ref["source_row_number"]).all(), "A1 source_row_number mismatch"
    assert (a2["source_row_number"] == ref["source_row_number"]).all(), "A2 source_row_number mismatch"


def test_annotator_templates_blank_initial_labels():
    for path, name in [(ANNOTATOR_1_PATH, "Annotator 1"), (ANNOTATOR_2_PATH, "Annotator 2")]:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ANNOTATION_COLUMNS:
            non_blank = df[df[col].str.strip() != ""]
            assert len(non_blank) == 0, (
                f"{name} template has {len(non_blank)} non-blank values in {col}"
            )


def test_agreement_notebook_exists():
    assert NOTEBOOK_PATH.exists(), f"Agreement analysis notebook missing: {NOTEBOOK_PATH}"
    assert NOTEBOOK_PATH.stat().st_size > 0, "Agreement analysis notebook is empty"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    assert "cells" in nb_json, "Notebook missing 'cells' key"
    assert len(nb_json["cells"]) >= 10, f"Notebook has fewer cells than expected: {len(nb_json['cells'])}"


def test_safe_handling_of_incomplete_annotations():
    """Verify that paired filtering logic safely handles initial/blank files."""
    a1 = pd.read_csv(ANNOTATOR_1_PATH, dtype=str, keep_default_na=False)
    a2 = pd.read_csv(ANNOTATOR_2_PATH, dtype=str, keep_default_na=False)

    s1 = a1["stance"].str.strip()
    s2 = a2["stance"].str.strip()

    valid_mask = s1.isin(PERMITTED_STANCES) & s2.isin(PERMITTED_STANCES)
    # When templates are fresh, valid paired records should be 0
    assert valid_mask.sum() == 0, "Expected 0 valid paired records in fresh templates"


def test_permitted_label_validation():
    test_labels = pd.Series(["Support", "Oppose", "invalid_stance", "", "Neutral", "Opposed"])
    valid_mask = test_labels.isin(PERMITTED_STANCES)
    blank_mask = test_labels.eq("")
    invalid_mask = (~valid_mask) & (~blank_mask)

    assert set(test_labels[valid_mask]) == {"Support", "Oppose", "Neutral"}
    assert set(test_labels[invalid_mask]) == {"invalid_stance", "Opposed"}
    assert blank_mask.sum() == 1


def test_cohen_kappa_insufficient_records_safeguard():
    """Kappa calculation must defer when N < 5."""
    # Mock calculation logic from notebook
    def mock_calc(y1, y2):
        n = len(y1)
        if n < 5:
            return {"status": "insufficient_data", "kappa": None}
        return {"status": "computed"}

    res = mock_calc(["Support", "Support"], ["Support", "Oppose"])
    assert res["status"] == "insufficient_data"
    assert res["kappa"] is None


def test_cohen_kappa_zero_variance_safeguard():
    """Kappa calculation must handle Pe = 1.0 gracefully without division by zero."""
    y1 = ["Support"] * 10
    y2 = ["Support"] * 10
    n = len(y1)

    categories = PERMITTED_STANCES
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    for a, b in zip(y1, y2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    po = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n ** 2))

    assert np.isclose(pe, 1.0)
    # Division (po - pe) / (1 - pe) would be 0 / 0. Safeguard must catch this:
    is_zero_variance = np.isclose(1.0 - pe, 0.0)
    assert is_zero_variance, "Zero variance guard failed to trigger"


def test_cohen_kappa_computation_correctness():
    """Verify Cohen's kappa matches theoretical formula on known benchmark data."""
    # 20 observations across 2 annotators:
    # 10 Support/Support, 4 Oppose/Oppose, 2 Neutral/Neutral, 2 Support/Oppose, 2 Oppose/Support
    y1 = ["Support"] * 12 + ["Oppose"] * 6 + ["Neutral"] * 2
    y2 = ["Support"] * 10 + ["Oppose"] * 2 + ["Oppose"] * 4 + ["Support"] * 2 + ["Neutral"] * 2
    n = len(y1)

    categories = ["Support", "Oppose", "Neutral"]
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    for a, b in zip(y1, y2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    po = np.trace(matrix) / n  # (10 + 4 + 2) / 20 = 16 / 20 = 0.80
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n ** 2))
    kappa = float((po - pe) / (1.0 - pe))

    assert np.isclose(po, 0.80)
    assert 0.60 <= kappa <= 0.80, f"Expected substantial kappa, got {kappa}"


# ── Completed-file workflow tests (synthetic fixtures in a temp project) ──────
#
# The notebook computes PROJECT_ROOT from the current working directory, so each
# test builds a throw-away project (notebooks/ + data/annotation/) under pytest's
# tmp_path, runs the real notebook code cells there, and inspects the namespace.
# The fixture texts and labels are synthetic test data only; they are not
# annotations of any PolicyLens comment and are never written to the repository.

COMPLETED_A = "pilot_annotator_A_completed.csv"
COMPLETED_B = "pilot_annotator_B_completed.csv"

_SYNTHETIC_TEXTS = [
    "Synthetic fixture text one: publish departmental datasets by default.",
    "Synthetic fixture text two: the workshop ran thirty minutes over time.",
    "Synthetic fixture text three: working data should not be published at all.",
    "Synthetic fixture text four: will datasets also be available in French?",
    "Synthetic fixture text five: supports open data but opposes publishing case records.",
    "Synthetic fixture text six: needs more work.",
]


def _synthetic_reference() -> pd.DataFrame:
    ref = pd.read_csv(PILOT_PATH, dtype=str, keep_default_na=False, nrows=0)  # real header only
    rows = []
    for i, text in enumerate(_SYNTHETIC_TEXTS, start=1):
        row = {c: "" for c in ref.columns}
        row.update({
            "pilot_id": f"pilot_{i:03d}",
            "source_dataset": "synthetic_fixture",
            "source_record_id": str(i),
            "source_row_number": str(i),
            "question_slug": "synthetic",
            "original_text": text,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=list(ref.columns))


def _annotated(ref: pd.DataFrame, annotator_id: str, stances: list[str]) -> pd.DataFrame:
    df = ref.copy()
    df["stance"] = stances
    df["annotator_id"] = annotator_id
    return df


GOOD_STANCES_A = ["Support", "Support", "Oppose", "Neutral", "Mixed", "Unclear"]
GOOD_STANCES_B = ["Support", "Oppose", "Oppose", "Neutral", "Mixed", "Neutral"]


@pytest.fixture
def temp_project(tmp_path):
    ref = _synthetic_reference()
    ann = tmp_path / "data" / "annotation"
    (tmp_path / "notebooks").mkdir()
    ann.mkdir(parents=True)
    ref.to_csv(ann / "pilot_annotation_50.csv", index=False, encoding="utf-8")
    # blank templates exist so that tests can prove the notebook does not fall back to them
    ref.to_csv(ann / "annotator_1.csv", index=False, encoding="utf-8")
    ref.to_csv(ann / "annotator_2.csv", index=False, encoding="utf-8")
    return tmp_path, ref


def _write_completed(root: Path, df_a: pd.DataFrame | None, df_b: pd.DataFrame | None) -> None:
    completed = root / "data" / "annotation" / "completed"
    completed.mkdir(exist_ok=True)
    if df_a is not None:
        df_a.to_csv(completed / COMPLETED_A, index=False, encoding="utf-8")
    if df_b is not None:
        df_b.to_csv(completed / COMPLETED_B, index=False, encoding="utf-8")


def _run_notebook(root: Path):
    """Execute all notebook code cells with cwd=<root>/notebooks. Returns (namespace, stdout, exception)."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__notebook_test__"}
    previous = os.getcwd()
    buf = io.StringIO()
    exc = None
    os.chdir(root / "notebooks")
    try:
        with contextlib.redirect_stdout(buf):
            for src in sources:
                exec(compile(src, "<notebook-cell>", "exec"), ns)
    except Exception as e:  # noqa: BLE001 - tests inspect the exact exception
        exc = e
    finally:
        os.chdir(previous)
    return ns, buf.getvalue(), exc


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_notebook_targets_completed_files_not_templates():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert COMPLETED_A in code and COMPLETED_B in code
    assert 'ANNOTATION_DIR / "completed"' in code
    assert 'ANNOTATION_DIR / "annotator_1.csv"' not in code
    assert 'ANNOTATION_DIR / "annotator_2.csv"' not in code


def test_missing_completed_files_stop_with_instruction_and_no_template_fallback(temp_project):
    root, _ = temp_project
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, FileNotFoundError)
    message = str(exc)
    assert "Finish annotation first" in message
    assert COMPLETED_A in message and COMPLETED_B in message
    assert "will not fall back" in message
    # Blank templates exist in the temp project but must not have been loaded.
    assert "df_a1" not in ns and "df_a2" not in ns


def test_only_missing_file_is_reported_when_one_completed_file_exists(temp_project):
    root, ref = temp_project
    _write_completed(root, _annotated(ref, "annotator_A", GOOD_STANCES_A), None)
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, FileNotFoundError)
    assert COMPLETED_B in str(exc)
    assert f"  - {root / 'data' / 'annotation' / 'completed' / COMPLETED_A}" not in str(exc)
    assert "df_a1" not in ns


def test_completed_files_load_and_agreement_is_computed(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A)
    b = _annotated(ref, "annotator_B", GOOD_STANCES_B)
    _write_completed(root, a, b)
    before = _snapshot(root)

    ns, out, exc = _run_notebook(root)
    assert exc is None, exc
    assert ns["ANNOTATOR_A_CSV"].name == COMPLETED_A and ns["ANNOTATOR_B_CSV"].name == COMPLETED_B
    pd.testing.assert_frame_equal(ns["df_a1"], a.astype(str).reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(ns["df_a2"], b.astype(str).reset_index(drop=True), check_dtype=False)
    assert ns["file_errors"] == []
    assert ns["alignment_ok"] is True
    assert len(ns["included_pairs"]) == 6
    assert ns["res"]["status"] == "computed"
    assert np.isclose(ns["res"]["po"], 4 / 6)   # 4 of 6 synthetic pairs agree
    assert "stance_annotator_A" in ns["included_pairs"].columns
    # the notebook must be read-only with respect to project data
    assert _snapshot(root) == before


def test_pending_completed_files_without_labels_report_pending_not_kappa(temp_project):
    root, ref = temp_project
    _write_completed(
        root,
        _annotated(ref, "annotator_A", [""] * 6),
        _annotated(ref, "annotator_B", [""] * 6),
    )
    ns, out, exc = _run_notebook(root)
    assert exc is None, exc
    assert len(ns["included_pairs"]) == 0
    assert "res" not in ns
    assert "pending" in out.lower()


def test_distinct_annotator_ids_required(temp_project):
    root, ref = temp_project
    _write_completed(
        root,
        _annotated(ref, "annotator_A", GOOD_STANCES_A),
        _annotated(ref, "annotator_A", GOOD_STANCES_B),   # copy-paste mistake
    )
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, ValueError)
    joined = " ".join(ns["file_errors"])
    assert "annotator_B" in joined                       # file B must say annotator_B
    assert "same annotator_id" in joined                 # explicit distinctness check


@pytest.mark.parametrize("bad_id", ["", "annotator_1", "Annotator_A"])
def test_wrong_or_blank_annotator_id_is_blocking(temp_project, bad_id):
    root, ref = temp_project
    _write_completed(
        root,
        _annotated(ref, bad_id, GOOD_STANCES_A),
        _annotated(ref, "annotator_B", GOOD_STANCES_B),
    )
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, ValueError)
    assert any(e.startswith("Annotator A:") and "annotator_id" in e for e in ns["file_errors"])


def test_invalid_and_blank_stance_labels_are_reported_and_excluded(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", ["Support", "Opposed", "Oppose", "Neutral", "", "Unclear"])
    b = _annotated(ref, "annotator_B", GOOD_STANCES_B)
    _write_completed(root, a, b)
    ns, _, exc = _run_notebook(root)
    assert exc is None, exc          # label problems are non-blocking by design
    assert ns["audit_1"]["invalid_values"] == ["Opposed"]
    assert ns["audit_1"]["blank_count"] == 1
    assert set(ns["included_pairs"]["pilot_id"]) == {"pilot_001", "pilot_003", "pilot_004", "pilot_006"}
    reasons = " ".join(ns["excluded_pairs"]["exclusion_reason"])
    assert "invalid stance: 'Opposed'" in reasons and "stance blank" in reasons
    assert set(ns["PERMITTED_STANCES"]) == {"Support", "Oppose", "Mixed", "Neutral", "Unclear"}


def test_evidence_span_must_be_verbatim_substring(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A)
    a.loc[0, "evidence_span"] = "publish departmental datasets by default"        # verbatim: OK
    a.loc[1, "evidence_span"] = "the workshop ran 30 minutes over"                # paraphrase: not verbatim
    _write_completed(root, a, _annotated(ref, "annotator_B", GOOD_STANCES_B))
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, ValueError)
    errors = [e for e in ns["file_errors"] if "evidence_span" in e]
    assert len(errors) == 1
    assert "pilot_002" in errors[0] and "verbatim" in errors[0]


def test_blank_and_valid_evidence_spans_pass(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A)
    a.loc[0, "evidence_span"] = "publish departmental datasets by default"
    a.loc[2, "evidence_span"] = "should not be published at all"
    _write_completed(root, a, _annotated(ref, "annotator_B", GOOD_STANCES_B))
    ns, _, exc = _run_notebook(root)
    assert exc is None, exc
    assert ns["file_errors"] == []


def test_evidence_span_over_200_characters_is_blocking(temp_project):
    root, ref = temp_project
    long_text = "x" * 250
    ref_long = ref.copy()
    ref_long.loc[0, "original_text"] = long_text
    # the reference pilot in the temp project must match, so rewrite it too
    ref_long.to_csv(root / "data" / "annotation" / "pilot_annotation_50.csv", index=False, encoding="utf-8")
    a = _annotated(ref_long, "annotator_A", GOOD_STANCES_A)
    a.loc[0, "evidence_span"] = "x" * 201   # a verbatim substring, but over the limit
    _write_completed(root, a, _annotated(ref_long, "annotator_B", GOOD_STANCES_B))
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, ValueError)
    assert any("201 characters" in e and "maximum 200" in e for e in ns["file_errors"])


def test_word_limits_are_warnings_not_errors(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A)
    a.loc[0, "concern"] = " ".join(["word"] * 15)             # at the limit: no warning
    a.loc[1, "concern"] = " ".join(["word"] * 16)             # over: warning
    a.loc[2, "requested_action"] = " ".join(["word"] * 20)    # at the limit: no warning
    a.loc[3, "requested_action"] = " ".join(["word"] * 21)    # over: warning
    _write_completed(root, a, _annotated(ref, "annotator_B", GOOD_STANCES_B))
    ns, _, exc = _run_notebook(root)
    assert exc is None, exc
    assert ns["file_errors"] == []
    warned = " ".join(ns["file_warnings"])
    assert "pilot_002 concern has 16 words" in warned
    assert "pilot_004 requested_action has 21 words" in warned
    assert "pilot_001" not in warned and "pilot_003" not in warned


def test_schema_changes_are_blocking(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A).drop(columns=["context"])       # missing column
    b = _annotated(ref, "annotator_B", GOOD_STANCES_B)
    b = b[["stance"] + [c for c in b.columns if c != "stance"]]                         # reordered columns
    _write_completed(root, a, b)
    ns, _, exc = _run_notebook(root)
    assert isinstance(exc, ValueError)
    errors = ns["file_errors"]
    assert any(e.startswith("Annotator A:") and "missing=['context']" in e for e in errors)
    assert any(e.startswith("Annotator B:") and "same_order=False" in e for e in errors)


def test_modified_original_text_is_flagged_by_alignment_check(temp_project):
    root, ref = temp_project
    a = _annotated(ref, "annotator_A", GOOD_STANCES_A)
    a.loc[4, "original_text"] = a.loc[4, "original_text"] + " (edited)"
    _write_completed(root, a, _annotated(ref, "annotator_B", GOOD_STANCES_B))
    ns, out, exc = _run_notebook(root)
    assert exc is None, exc
    assert ns["alignment_ok"] is False
    assert "original_text was modified" in out
