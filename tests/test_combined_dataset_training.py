"""
tests/test_combined_dataset_training.py
==========================================
Unit tests for the pure dataset-verification and leakage-check helpers in
scripts/train_stance_combined_dataset_v1.py. No model artifacts or large files are touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_stance_combined_dataset_v1 import (  # noqa: E402
    REQUIRED_COLUMNS, find_train_test_text_leakage, verify_combined_dataset,
)


def _write_csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


class TestVerifyCombinedDataset:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            verify_combined_dataset(tmp_path / "does_not_exist.csv", REQUIRED_COLUMNS)

    def test_missing_required_column_raises(self, tmp_path):
        df = pd.DataFrame({c: ["x"] for c in REQUIRED_COLUMNS if c != "gold_stance"})
        p = _write_csv(tmp_path, "bad.csv", df)
        with pytest.raises(ValueError, match="missing required columns"):
            verify_combined_dataset(p, REQUIRED_COLUMNS)

    def test_empty_file_raises(self, tmp_path):
        df = pd.DataFrame({c: [] for c in REQUIRED_COLUMNS})
        p = _write_csv(tmp_path, "empty.csv", df)
        with pytest.raises(ValueError, match="zero rows"):
            verify_combined_dataset(p, REQUIRED_COLUMNS)

    def test_valid_file_returns_correct_summary(self, tmp_path):
        df = pd.DataFrame({
            "original_text": ["a", "b", "c"], "gold_stance": ["Support", "Neutral", "Oppose"],
            "label_source": ["src1", "src1", "src2"], "source_dataset": ["d1", "d1", "d2"],
            "source_record_id": ["1", "2", "3"], "rule": ["", "", ""],
            "sentiment_prediction": ["", "", ""], "stance_margin_pseudo": ["0.1", "0.2", "0.3"],
            "sentiment_margin_pseudo": ["0.1", "0.2", "0.3"], "split": ["train_or_validation"] * 3,
        })
        p = _write_csv(tmp_path, "good.csv", df)
        out_df, summary = verify_combined_dataset(p, REQUIRED_COLUMNS)
        assert summary["n_rows"] == 3
        assert summary["class_distribution"] == {"Support": 1, "Neutral": 1, "Oppose": 1}
        assert len(out_df) == 3
        assert "sha256" in summary and len(summary["sha256"]) == 64


class TestFindTrainTestTextLeakage:
    def test_no_overlap(self):
        train = pd.Series(["one thing", "another thing"])
        test = pd.Series(["something else entirely"])
        out = find_train_test_text_leakage(train, test)
        assert out["n_train_rows_overlapping_test_text"] == 0
        assert out["overlap_mask"].tolist() == [False, False]

    def test_exact_overlap_detected(self):
        train = pd.Series(["Hello World", "unrelated text"])
        test = pd.Series(["hello   world"])  # case + whitespace differences only
        out = find_train_test_text_leakage(train, test)
        assert out["n_train_rows_overlapping_test_text"] == 1
        assert out["overlap_mask"].tolist() == [True, False]

    def test_multiple_train_rows_can_match_one_test_text(self):
        train = pd.Series(["dup text", "dup text", "unique text"])
        test = pd.Series(["dup text"])
        out = find_train_test_text_leakage(train, test)
        assert out["n_train_rows_overlapping_test_text"] == 2

    def test_n_test_texts_counts_unique_normalised_texts(self):
        train = pd.Series(["a"])
        test = pd.Series(["Alpha", "alpha", "Beta"])
        out = find_train_test_text_leakage(train, test)
        assert out["n_test_texts"] == 2
