"""
Unit tests for scripts/comprehensive_data_cleaning_audit.py.
"""
from pathlib import Path
import pandas as pd
import pytest

from scripts.comprehensive_data_cleaning_audit import (
    classify_role,
    detect_file_encoding,
    audit_tabular,
    determine_suitability,
    scan_workspace,
    PROVENANCE_MAP,
)

ROOT = Path(__file__).resolve().parents[1]

def test_classify_role():
    assert classify_role(ROOT / "data" / "annotation" / "annotator_1.csv") == "annotated"
    assert classify_role(ROOT / "data" / "clustering" / "cluster_labels.csv") == "model_generated_artifact"
    assert classify_role(ROOT / "data" / "processed" / "hinemo9_dev_source_clean.csv") == "processed_cleaned"
    assert classify_role(ROOT / "data" / "processed" / "hinemo9_dev_canonical.csv") == "processed_canonical"
    assert classify_role(ROOT / "data" / "interim" / "government_id_overlap.csv") == "interim_audit_artifact"
    assert classify_role(ROOT / "hinemo9-v1.0.1" / "im-rakeshsahani-hinemo9-764fb36" / "reference" / "paper_tables__appendix_b.csv") == "reference_benchmark"


def test_detect_file_encoding(tmp_path):
    f_utf8 = tmp_path / "test_utf8.csv"
    f_utf8.write_text("col1,col2\nval1,val2", encoding="utf-8")
    assert detect_file_encoding(f_utf8) in ["utf-8", "utf-8-sig"]

    f_cp = tmp_path / "test_cp1252.csv"
    f_cp.write_bytes("col1,col2\nval1,Données".encode("cp1252"))
    assert detect_file_encoding(f_cp) in ["cp1252", "latin-1"]


def test_audit_tabular_metrics(tmp_path):
    csv_file = tmp_path / "sample_dataset.csv"
    csv_file.write_text("id,comment,stance\n1,hello,pro\n2, ,con\n3,hello,pro\n4,,pro", encoding="utf-8")
    
    res = audit_tabular(csv_file)
    assert res["row_count"] == 4
    assert res["column_count"] == 3
    assert "comment" in res["column_names"]
    assert res["missing_values"]["comment"] == 1  # 4th row is null
    assert res["blank_text_cells"] == 1  # 2nd row is whitespace
    assert res["exact_duplicate_rows"] == 0  # id is unique
    assert res["duplicate_text_groups"]["comment"] == 1  # 'hello' duplicated


def test_provenance_map_integrity():
    for filename, meta in PROVENANCE_MAP.items():
        assert meta["verified"] is True
        assert len(meta["verified_in"]) > 0
        if meta["cleaned_path"]:
            cleaned_file = ROOT / meta["cleaned_path"]
            assert cleaned_file.exists(), f"Cleaned file {cleaned_file} does not exist!"


def test_determine_suitability():
    res = {
        "filename": "pilot_annotation_50.csv",
        "path": "data/annotation/pilot_annotation_50.csv",
        "role": "annotated",
    }
    determine_suitability(res)
    assert "supervised_stance_training" in res["suitability"]

    res2 = {
        "filename": "get_involved_questionnaire_2017_18_source_clean.csv",
        "path": "data/processed/get_involved_questionnaire_2017_18_source_clean.csv",
        "role": "processed_cleaned",
    }
    determine_suitability(res2)
    assert "unsupervised_clustering" in res2["suitability"]
