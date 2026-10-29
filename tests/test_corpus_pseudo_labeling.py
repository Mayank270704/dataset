"""
tests/test_corpus_pseudo_labeling.py
=====================================
Unit tests for scripts/run_corpus_pseudo_labeling_v1.py, mirroring the structure of
tests/test_sentiment140_pseudo_labeling.py. Covers: pool400-exclusion, too-short rejection,
duplicate-group handling, Unclear separation, resume, write-once protection.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_corpus_pseudo_labeling_v1 as rcpl  # noqa: E402

_STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
_SENTIMENT_MODEL = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1" / "model.joblib"
_ARTIFACTS_PRESENT = _STANCE_RUN.is_dir() and _SENTIMENT_MODEL.is_file() and rcpl.POOL400_FILE.is_file()


def _mock_stance_pipe(labels_by_text: dict[str, str]):
    pipe = MagicMock()
    pipe.predict.side_effect = lambda texts: np.array([labels_by_text[t] for t in texts])
    idx = {"Support": 0, "Neutral": 1, "Unclear": 2}

    def _dec(texts):
        out = []
        for t in texts:
            scores = [-0.5, -0.5, -0.5]
            scores[idx[labels_by_text[t]]] = 1.5
            out.append(scores)
        return np.array(out)

    pipe.decision_function.side_effect = _dec
    return pipe


def _mock_sentiment_clf(sentiment_by_text: dict[str, str]):
    clf = MagicMock()
    clf.predict.side_effect = lambda texts: np.array([sentiment_by_text[t] for t in texts])
    clf.decision_function.side_effect = lambda texts: np.array(
        [[1.0 if sentiment_by_text[t] == "positive" else -1.0] for t in texts]
    )
    return clf


def _make_chunk(rows: list[dict]) -> pd.DataFrame:
    cols = ["corpus_id", "unique_id", "source_dataset", "source_record_id", "question_slug",
            "language", "stakeholder_group", "duplicate_text_group_id", "original_text"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


class TestProcessChunkCorpus:
    def test_pool400_record_excluded_not_labeled(self):
        text = "this record is already in the pool400 training data"
        chunk = _make_chunk([{"original_text": text, "source_dataset": "ds1", "source_record_id": "1"}])
        pool400_keys = {("ds1", "1", rcpl.normalise_text(text))}
        rows, labels, rules, short, pool_excl, dup, unclear = rcpl.process_chunk(
            chunk, 0, -1, _mock_stance_pipe({}), _mock_sentiment_clf({}), set(), pool400_keys
        )
        assert pool_excl == 1
        assert labels == {}
        assert rows[0][10] is True  # excluded_pool400_record column
        assert rows[0][18] == "already_in_pool400_training_data"

    def test_eligible_non_pool400_record_gets_prediction(self):
        text = "a sufficiently long genuinely new consultation comment about open data"
        chunk = _make_chunk([{"original_text": text, "source_dataset": "ds1", "source_record_id": "2"}])
        stance_pipe = _mock_stance_pipe({text: "Support"})
        sentiment_clf = _mock_sentiment_clf({text: "positive"})
        rows, labels, rules, short, pool_excl, dup, unclear = rcpl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set(), set()
        )
        assert pool_excl == 0
        assert labels == {"Support": 1}
        assert rows[0][13] == "Support"  # stance_raw
        assert rows[0][15] == "Support"  # pseudo_label
        assert rows[0][17] is True       # eligible_for_training

    def test_too_short_rejected(self):
        chunk = _make_chunk([{"original_text": "ok", "source_dataset": "ds1", "source_record_id": "3"}])
        rows, labels, rules, short, pool_excl, dup, unclear = rcpl.process_chunk(
            chunk, 0, -1, _mock_stance_pipe({}), _mock_sentiment_clf({}), set(), set()
        )
        assert short == 1
        assert rows[0][18] == "too_short_text"

    def test_duplicate_group_second_occurrence_excluded(self):
        t1, t2 = "first occurrence of this duplicated consultation comment", "second occurrence of this duplicated consultation comment"
        chunk = _make_chunk([
            {"original_text": t1, "source_dataset": "ds1", "source_record_id": "4", "duplicate_text_group_id": "g1"},
            {"original_text": t2, "source_dataset": "ds1", "source_record_id": "5", "duplicate_text_group_id": "g1"},
        ])
        stance_pipe = _mock_stance_pipe({t1: "Neutral", t2: "Neutral"})
        sentiment_clf = _mock_sentiment_clf({t1: "positive", t2: "positive"})
        rows, labels, rules, short, pool_excl, dup, unclear = rcpl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set(), set()
        )
        assert dup == 1
        assert rows[0][17] is True and rows[1][17] is False
        assert labels == {"Neutral": 1}

    def test_unclear_kept_separate(self):
        text = "an ambiguous consultation remark that is hard to classify either way"
        chunk = _make_chunk([{"original_text": text, "source_dataset": "ds1", "source_record_id": "6"}])
        stance_pipe = _mock_stance_pipe({text: "Unclear"})
        sentiment_clf = _mock_sentiment_clf({text: "negative"})
        rows, labels, rules, short, pool_excl, dup, unclear = rcpl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set(), set()
        )
        assert unclear == 1
        assert labels == {}
        assert rows[0][15] == "Unclear"


def _write_fixture_corpus_csv(path: Path, pool400_text: str) -> None:
    cols = ["corpus_id", "unique_id", "source_dataset", "source_record_id", "source_row_number",
            "source_field", "question_slug", "language", "stakeholder_group", "duplicate_text_group_id",
            "text_key", "original_text"]
    rows = [
        ["0", "0", "ds1", "100", "100", "f1", "q1", "English", "Public", "", pool400_text.lower(), pool400_text],
        ["1", "1", "ds1", "101", "101", "f1", "q1", "English", "Public", "", "", "a brand new consultation comment about transparency"],
        ["2", "2", "ds1", "102", "102", "f1", "q1", "English", "Public", "", "", "ok"],
        ["3", "3", "ds1", "103", "103", "f1", "q1", "English", "Public", "dupg1", "", "a duplicated comment about the budget process"],
        ["4", "4", "ds1", "104", "104", "f1", "q1", "English", "Public", "dupg1", "", "a duplicated comment about the budget process"],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def _write_fixture_pool400_csv(path: Path, text: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "source_dataset", "source_record_id", "original_text", "stance"])
        w.writerow(["p1", "ds1", "100", text, "Support"])


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="Real model artifacts or pool400 file not present; skipping integration test.")
class TestRunEndToEndRealModels:
    def test_fresh_run_excludes_pool400_and_writes_manifest(self, tmp_path, monkeypatch):
        pool400_text = "this exact comment is already part of the training pool"
        inp = tmp_path / "corpus.csv"
        pool = tmp_path / "pool400.csv"
        _write_fixture_corpus_csv(inp, pool400_text)
        _write_fixture_pool400_csv(pool, pool400_text)
        monkeypatch.setattr(rcpl, "POOL400_FILE", pool)
        out_dir = tmp_path / "out_v1"
        rc = rcpl.run(inp, out_dir, batch_size=2, max_rows=None, model="linsvm",
                      resume=False, min_free_ram_mb=1.0, log_every=1)
        assert rc == 0
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["excluded_already_in_pool400"] == 1
        assert manifest["rejected_too_short"] == 1
        assert manifest["duplicate_non_first_excluded"] == 1
        with open(out_dir / "pseudo_labels.csv", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == rcpl.OUTPUT_COLUMNS

    def test_rerun_on_completed_dir_refused(self, tmp_path, monkeypatch):
        pool400_text = "this exact comment is already part of the training pool"
        inp = tmp_path / "corpus.csv"
        pool = tmp_path / "pool400.csv"
        _write_fixture_corpus_csv(inp, pool400_text)
        _write_fixture_pool400_csv(pool, pool400_text)
        monkeypatch.setattr(rcpl, "POOL400_FILE", pool)
        out_dir = tmp_path / "out_v1"
        assert rcpl.run(inp, out_dir, 2, None, "linsvm", False, 1.0, 1) == 0
        assert rcpl.run(inp, out_dir, 2, None, "linsvm", False, 1.0, 1) == 2
