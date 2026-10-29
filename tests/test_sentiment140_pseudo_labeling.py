"""
tests/test_sentiment140_pseudo_labeling.py
===========================================
Unit tests for models/pseudo_labeling.py (the labeling rule) and
scripts/run_sentiment140_pseudo_labeling_v1.py (chunked/resumable pipeline).

Design mirrors tests/test_final_stance_inference.py: pure-rule tests need no models; the
chunk-processing tests mock the stance pipeline and sentiment classifier; only the full
end-to-end run() tests touch the real saved artifacts, and those are skipped when the
artifacts are absent.
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

from models.pseudo_labeling import (  # noqa: E402
    MIN_CHARS,
    PSEUDO_LABEL_DISCLAIMER,
    RULE_HEURISTIC_OPPOSE,
    RULE_NEUTRAL_DIRECT_POSITIVE_SENTIMENT,
    RULE_SUPPORT_DIRECT,
    RULE_SUPPORT_DIRECT_POSITIVE_SENTIMENT,
    RULE_UNCLEAR_KEPT_SEPARATE,
    assign_pseudo_label,
    assign_pseudo_labels_batch,
)
import scripts.run_sentiment140_pseudo_labeling_v1 as rspl  # noqa: E402

_STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
_SENTIMENT_MODEL = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1" / "model.joblib"
_ARTIFACTS_PRESENT = _STANCE_RUN.is_dir() and _SENTIMENT_MODEL.is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Rule mapping (pure unit, no models)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignPseudoLabelRule:
    def test_negative_support_kept_support(self):
        r = assign_pseudo_label("negative", "Support")
        assert r.pseudo_label == "Support"
        assert r.rule == RULE_SUPPORT_DIRECT

    def test_negative_neutral_becomes_oppose(self):
        r = assign_pseudo_label("negative", "Neutral")
        assert r.pseudo_label == "Oppose"
        assert r.rule == RULE_HEURISTIC_OPPOSE

    def test_negative_unclear_kept_separate_not_oppose(self):
        """Unclear must never be forced into Oppose, even under negative sentiment."""
        r = assign_pseudo_label("negative", "Unclear")
        assert r.pseudo_label == "Unclear"
        assert r.rule == RULE_UNCLEAR_KEPT_SEPARATE

    def test_positive_unclear_kept_separate(self):
        r = assign_pseudo_label("positive", "Unclear")
        assert r.pseudo_label == "Unclear"
        assert r.rule == RULE_UNCLEAR_KEPT_SEPARATE

    def test_positive_support_kept_support(self):
        r = assign_pseudo_label("positive", "Support")
        assert r.pseudo_label == "Support"
        assert r.rule == RULE_SUPPORT_DIRECT_POSITIVE_SENTIMENT

    def test_positive_neutral_stays_neutral_not_assumed_support(self):
        """Positive sentiment must never be assumed to mean Support."""
        r = assign_pseudo_label("positive", "Neutral")
        assert r.pseudo_label == "Neutral"
        assert r.rule == RULE_NEUTRAL_DIRECT_POSITIVE_SENTIMENT

    def test_invalid_stance_raw_rejected(self):
        with pytest.raises(ValueError):
            assign_pseudo_label("negative", "Oppose")  # stance model never outputs Oppose

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(ValueError):
            assign_pseudo_label("neutral", "Support")  # sentiment model is binary only

    def test_batch_matches_scalar(self):
        sentiments = np.array(["negative", "positive", "negative", "positive"])
        stances = np.array(["Support", "Neutral", "Neutral", "Unclear"])
        labels, rules = assign_pseudo_labels_batch(sentiments, stances)
        expected = [assign_pseudo_label(s, st) for s, st in zip(sentiments, stances)]
        assert list(labels) == [e.pseudo_label for e in expected]
        assert list(rules) == [e.rule for e in expected]

    def test_disclaimer_mentions_pseudo_label_and_oppose_risk(self):
        assert "PSEUDO-LABEL" in PSEUDO_LABEL_DISCLAIMER
        assert "Oppose" in PSEUDO_LABEL_DISCLAIMER


# ─────────────────────────────────────────────────────────────────────────────
# Chunk processing (mocked models, no I/O, no real artifacts)
# ─────────────────────────────────────────────────────────────────────────────

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
    cols = ["record_id", "source_dataset", "source_record_id", "label", "duplicate_group_id", "cleaned_text"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


class TestProcessChunk:
    def test_too_short_text_rejected_no_prediction(self):
        chunk = _make_chunk([{"cleaned_text": "ok", "record_id": "r1", "label": "negative"}])
        stance_pipe = _mock_stance_pipe({})
        sentiment_clf = _mock_sentiment_clf({})
        rows, labels, rules, short, dup, unclear = rspl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set()
        )
        assert short == 1
        assert labels == {} and rules == {}
        assert rows[0][14] == "too_short_text"  # exclusion_reason column

    def test_eligible_text_gets_prediction_and_counts(self):
        text = "this is a sufficiently long piece of text"
        chunk = _make_chunk([{"cleaned_text": text, "record_id": "r1", "label": "negative"}])
        stance_pipe = _mock_stance_pipe({text: "Support"})
        sentiment_clf = _mock_sentiment_clf({text: "negative"})
        rows, labels, rules, short, dup, unclear = rspl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set()
        )
        assert short == 0
        assert labels == {"Support": 1}
        row = rows[0]
        assert row[9] == "Support"       # stance_raw
        assert row[11] == "Support"      # pseudo_label
        assert row[13] is True           # eligible_for_training

    def test_duplicate_group_first_occurrence_eligible_second_excluded(self):
        text1, text2 = "first occurrence text here", "second occurrence text here"
        chunk = _make_chunk([
            {"cleaned_text": text1, "record_id": "r1", "label": "positive", "duplicate_group_id": "g1"},
            {"cleaned_text": text2, "record_id": "r2", "label": "positive", "duplicate_group_id": "g1"},
        ])
        stance_pipe = _mock_stance_pipe({text1: "Neutral", text2: "Neutral"})
        sentiment_clf = _mock_sentiment_clf({text1: "positive", text2: "positive"})
        seen: set[str] = set()
        rows, labels, rules, short, dup, unclear = rspl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, seen
        )
        assert dup == 1
        assert rows[0][6] is False and rows[0][13] is True       # first: not non-first dup, eligible
        assert rows[1][6] is True and rows[1][13] is False        # second: non-first dup, excluded
        assert rows[1][14] == "duplicate_non_first_occurrence"
        assert labels == {"Neutral": 1}   # only the first occurrence counted as training-eligible

    def test_unclear_excluded_from_training_counts(self):
        text = "a genuinely ambiguous statement about something"
        chunk = _make_chunk([{"cleaned_text": text, "record_id": "r1", "label": "negative"}])
        stance_pipe = _mock_stance_pipe({text: "Unclear"})
        sentiment_clf = _mock_sentiment_clf({text: "negative"})
        rows, labels, rules, short, dup, unclear = rspl.process_chunk(
            chunk, 0, -1, stance_pipe, sentiment_clf, set()
        )
        assert unclear == 1
        assert labels == {}
        assert rows[0][11] == "Unclear"
        assert rows[0][13] is False
        assert rows[0][14] == "unresolved_unclear_kept_separate"

    def test_resume_skips_already_processed_rows(self):
        text_a, text_b = "row already processed previously", "row not yet processed at all"
        chunk = _make_chunk([
            {"cleaned_text": text_a, "record_id": "r1", "label": "positive"},
            {"cleaned_text": text_b, "record_id": "r2", "label": "positive"},
        ])
        stance_pipe = _mock_stance_pipe({text_b: "Support"})
        sentiment_clf = _mock_sentiment_clf({text_b: "positive"})
        # resume_from_row = 0 means global row index 0 (text_a) was already written; only row 1 should process
        rows, labels, rules, short, dup, unclear = rspl.process_chunk(
            chunk, 0, 0, stance_pipe, sentiment_clf, set()
        )
        assert len(rows) == 1
        assert rows[0][0] == "r2"

    def test_disclaimer_present_on_every_row(self):
        text = "another sufficiently long example sentence"
        chunk = _make_chunk([{"cleaned_text": text, "record_id": "r1", "label": "negative"}])
        stance_pipe = _mock_stance_pipe({text: "Support"})
        sentiment_clf = _mock_sentiment_clf({text: "negative"})
        rows, *_ = rspl.process_chunk(chunk, 0, -1, stance_pipe, sentiment_clf, set())
        assert rows[0][-1] == PSEUDO_LABEL_DISCLAIMER


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end run() : write-once, resume, manifest/provenance (real artifacts)
# ─────────────────────────────────────────────────────────────────────────────

def _write_fixture_csv(path: Path, n_rows: int = 12) -> None:
    cols = ["record_id", "source_dataset", "source_record_id", "consultation_id", "question_id",
            "question_text", "original_text", "cleaned_text", "language", "task_type", "label",
            "is_synthetic", "duplicate_group_id", "exclusion_reason"]
    texts = [
        "I really support this open data initiative moving forward",
        "This is a terrible policy and it should be cancelled immediately",
        "The meeting happened on a Tuesday afternoon downtown",
        "I am not sure what this means for anyone involved here",
        "ok",  # too short, rejected
        "I really support this open data initiative moving forward",  # exact duplicate of row 0 text
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n_rows):
            t = texts[i % len(texts)]
            dup_group = "fixdup1" if t == texts[0] else ""
            w.writerow([f"fixture:{i}", "fixture_dataset", str(i), "", "", "", t, t, "English",
                        "sentiment_classification", "negative" if i % 2 == 0 else "positive",
                        "False", dup_group, ""])


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="Real model artifacts not present; skipping integration test.")
class TestRunEndToEndRealModels:
    def test_fresh_run_writes_manifest_and_schema(self, tmp_path):
        inp = tmp_path / "fixture.csv"
        _write_fixture_csv(inp)
        out_dir = tmp_path / "out_v1"
        rc = rspl.run(inp, out_dir, batch_size=4, max_rows=None, model="linsvm",
                      resume=False, min_free_ram_mb=1.0, log_every=1)
        assert rc == 0
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert "Oppose" in PSEUDO_LABEL_DISCLAIMER or True  # disclaimer content checked elsewhere
        with open(out_dir / "pseudo_labels.csv", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == rspl.OUTPUT_COLUMNS
        assert manifest["rejected_too_short"] >= 1
        assert manifest["duplicate_non_first_excluded"] >= 1

    def test_rerun_on_completed_dir_refused(self, tmp_path):
        inp = tmp_path / "fixture.csv"
        _write_fixture_csv(inp)
        out_dir = tmp_path / "out_v1"
        assert rspl.run(inp, out_dir, 4, None, "linsvm", False, 1.0, 1) == 0
        rc_again = rspl.run(inp, out_dir, 4, None, "linsvm", False, 1.0, 1)
        assert rc_again == 2

    def test_incomplete_without_resume_refused_then_resume_succeeds(self, tmp_path, monkeypatch):
        inp = tmp_path / "fixture.csv"
        _write_fixture_csv(inp, n_rows=12)
        out_dir = tmp_path / "out_v1"

        real_vm = rspl.psutil.virtual_memory
        calls = {"n": 0}

        def fake_vm():
            calls["n"] += 1
            if calls["n"] <= 1:
                class _Low:
                    available = 1 * 1024 * 1024  # force a pause on the very first chunk
                return _Low()
            return real_vm()

        monkeypatch.setattr(rspl.psutil, "virtual_memory", fake_vm)
        rc_paused = rspl.run(inp, out_dir, batch_size=4, max_rows=None, model="linsvm",
                             resume=False, min_free_ram_mb=500.0, log_every=1)
        assert rc_paused == 3
        checkpoint = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["status"] == "in_progress"

        monkeypatch.setattr(rspl.psutil, "virtual_memory", real_vm)
        rc_no_resume = rspl.run(inp, out_dir, 4, None, "linsvm", False, 500.0, 1)
        assert rc_no_resume == 2

        rc_resumed = rspl.run(inp, out_dir, batch_size=4, max_rows=None, model="linsvm",
                              resume=True, min_free_ram_mb=1.0, log_every=1)
        assert rc_resumed == 0
        with open(out_dir / "pseudo_labels.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        record_ids = [r[0] for r in rows]
        assert len(record_ids) == len(set(record_ids)), "resume must not duplicate rows"
        assert len(record_ids) == 12

    def test_resume_rejects_changed_input_file(self, tmp_path, monkeypatch):
        inp = tmp_path / "fixture.csv"
        _write_fixture_csv(inp)
        out_dir = tmp_path / "out_v1"

        real_vm = rspl.psutil.virtual_memory

        def fake_vm_low():
            class _Low:
                available = 1 * 1024 * 1024
            return _Low()

        monkeypatch.setattr(rspl.psutil, "virtual_memory", fake_vm_low)
        assert rspl.run(inp, out_dir, 4, None, "linsvm", False, 500.0, 1) == 3
        monkeypatch.setattr(rspl.psutil, "virtual_memory", real_vm)

        with open(inp, "a", encoding="utf-8") as f:
            f.write("extra:999,fixture_dataset,999,,,,changed,changed,English,sentiment_classification,negative,False,,\n")

        rc = rspl.run(inp, out_dir, 4, None, "linsvm", True, 1.0, 1)
        assert rc == 1
