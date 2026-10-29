#!/usr/bin/env python
"""
scripts/run_sentiment140_pseudo_labeling_v1.py
===============================================
Memory-safe, chunked, resumable pseudo-labeling of the Sentiment140 canonical dataset
(1,600,000 rows) using the existing verified PolicyLens stance model
(data/stance_training_ai_working_labels/run_pool400_ai_v1_3class) and the existing verified
Sentiment140 sentiment model (data/sentiment_training/sentiment140_hashing_sgd_v1), combined
via the fixed rule in models/pseudo_labeling.py.

READ BEFORE USE
----------------
Pseudo-labels are NOT ground truth. The stance model is trained on 239-399 AI-generated
working labels for "stance toward Open Government" (a narrow Canadian-consultation construct),
not generic policy stance, and the sentiment model is a Twitter tone classifier, not a stance
model. The "Oppose" pseudo-label specifically is produced by an unverified heuristic that the
project's own in-domain check (data/sentiment_evaluation/consultation_domain_check_v1) found
to have a ~98% false-positive rate against the few genuine Oppose examples available. See
models/pseudo_labeling.py for the exact rule and models/stance_inference_v2.py for the original
heuristic this one was derived from. The project owner was shown this evidence (2026-10-29) and
chose to proceed anyway; every output row and the run manifest repeat the caveat so it is never
lost downstream.

Memory safety
-------------
Reads the input CSV in fixed-size chunks (pandas chunksize, default 2000 rows), never holds the
full file or full prediction set in memory, checks available RAM via psutil before each chunk
and aborts (saving a resumable checkpoint) if free RAM drops below --min-free-ram-mb. Output is
written incrementally, one chunk at a time, to a single CSV via the stdlib csv module.

Resume
------
The output directory holds checkpoint.json (status "in_progress" | "completed"). Re-running
without --resume against an existing, incomplete output directory is refused (exit 2) to avoid
accidental duplicate rows; pass --resume to continue from the last fully-written row. A
"completed" run's directory is write-once: start a new --out-dir (new version) to re-run.

Usage
-----
    python scripts/run_sentiment140_pseudo_labeling_v1.py --max-rows 2000 \\
        --out-dir data/pseudo_labeling/sentiment140_stance_pilot_v1
    python scripts/run_sentiment140_pseudo_labeling_v1.py \\
        --out-dir data/pseudo_labeling/sentiment140_stance_v1
    python scripts/run_sentiment140_pseudo_labeling_v1.py --resume \\
        --out-dir data/pseudo_labeling/sentiment140_stance_v1

Exit codes
----------
0  success (run completed, or chunk-limited partial run under --max-rows completed as requested)
1  error (model load failure, missing input, malformed checkpoint)
2  output directory exists and is either already completed, or incomplete without --resume
3  paused: available RAM dropped below --min-free-ram-mb; checkpoint saved, rerun with --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.new_comment_inference import load_verified_pipeline  # noqa: E402
from models.stance_inference_v2 import _load_verified_sentiment_model  # noqa: E402
from models.pseudo_labeling import (  # noqa: E402
    MIN_CHARS,
    PSEUDO_LABEL_DISCLAIMER,
    assign_pseudo_label,
    decision_margin,
)

DEFAULT_INPUT = ROOT / "data" / "processed" / "sentiment140_train_canonical.csv"
STANCE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
DEFAULT_OUT_DIR = ROOT / "data" / "pseudo_labeling" / "sentiment140_stance_v1"
TEXT_COLUMN = "cleaned_text"

OUTPUT_COLUMNS = [
    "record_id", "source_dataset", "source_record_id", "source_sentiment140_label",
    "char_len", "duplicate_group_id", "is_duplicate_non_first",
    "sentiment_prediction", "sentiment_margin", "stance_raw", "stance_margin",
    "pseudo_label", "rule", "eligible_for_training", "exclusion_reason",
    "cleaned_text", "disclaimer",
]

KNOWN_LIMITATIONS = [
    "Sentiment140 is general Twitter text, not government-policy consultation submissions; "
    "'stance toward Open Government' predicted on it is an out-of-domain extrapolation.",
    "The stance model was trained on only 239-399 AI-generated (not human-verified) working "
    "labels and has never seen a real Oppose example.",
    "The 'Oppose' pseudo-label is a heuristic (negative sentiment AND non-Support stance), not a "
    "validated classifier output. The project's own in-domain check found 53/73 negative-sentiment "
    "pool400 records would be heuristically flagged Oppose, against only 1 genuine Oppose record "
    "(data/sentiment_evaluation/consultation_domain_check_v1/results.json). The true false-positive "
    "rate on Sentiment140 (fully out-of-domain) is expected to be at least as high.",
    "The sentiment model is itself a social-media tone classifier (79% macro-F1 on tweets), not a "
    "stance classifier, and was never evaluated on consultation text as ground truth.",
    "Pseudo-labels must never be presented as human-verified stance or as gold-standard evaluation data.",
]


class MemoryPressureError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()  # outside the project root (e.g. a test fixture); keep the absolute path


def checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint.json"


def output_csv_path(out_dir: Path) -> Path:
    return out_dir / "pseudo_labels.csv"


def load_checkpoint(out_dir: Path) -> dict[str, Any] | None:
    p = checkpoint_path(out_dir)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_checkpoint(out_dir: Path, state: dict[str, Any]) -> None:
    """Atomic-ish write: write to a temp file then replace, so a crash mid-write can't corrupt it.

    Retries the rename a few times: on Windows, antivirus/indexing can transiently hold a newly
    written small file (WinError 5 / PermissionError) for a few milliseconds.
    """
    p = checkpoint_path(out_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            tmp.replace(p)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    raise last_exc  # pragma: no cover - only reached if the lock never clears


def fresh_state(config: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_utc": None,
        "config": config,
        "input_sha256": input_sha256,
        "last_row_index": -1,          # 0-based index of the last input row fully written
        "rows_processed": 0,
        "rows_written": 0,
        "seen_duplicate_groups": [],   # duplicate_group_id values whose first occurrence was already seen
        "label_counts": {},            # pseudo_label -> count, training-eligible rows only
        "rule_counts": {},
        "rejected_too_short": 0,
        "duplicate_non_first": 0,
        "unresolved_unclear": 0,
        "elapsed_seconds_prior_runs": 0.0,
    }


def process_chunk(
    chunk: pd.DataFrame,
    start_row_index: int,
    resume_from_row: int,
    stance_pipe,
    sentiment_clf,
    seen_duplicate_groups: set[str],
) -> tuple[list[list[Any]], dict[str, int], dict[str, int], int, int, int]:
    """
    Process one chunk, skipping any rows already written on a prior run (<= resume_from_row).
    Returns (output_rows, label_counts_delta, rule_counts_delta, rejected_too_short_delta,
    duplicate_non_first_delta, unresolved_unclear_delta).
    """
    label_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    rejected_too_short = 0
    duplicate_non_first = 0
    unresolved_unclear = 0
    out_rows: list[list[Any]] = []

    keep_mask = [start_row_index + i > resume_from_row for i in range(len(chunk))]
    if not any(keep_mask):
        return out_rows, label_counts, rule_counts, 0, 0, 0

    chunk = chunk.reset_index(drop=True)
    texts_all = chunk[TEXT_COLUMN].tolist()
    stripped_all = [t.strip() for t in texts_all]
    too_short_all = [len(t) < MIN_CHARS for t in stripped_all]

    eligible_positions = [i for i, (keep, short) in enumerate(zip(keep_mask, too_short_all)) if keep and not short]
    sentiment_preds: dict[int, str] = {}
    sentiment_margins: dict[int, float] = {}
    stance_preds: dict[int, str] = {}
    stance_margins: dict[int, float] = {}
    if eligible_positions:
        eligible_texts = [texts_all[i] for i in eligible_positions]
        s_pred = sentiment_clf.predict(eligible_texts)
        s_marg = sentiment_clf.decision_function(eligible_texts)
        st_pred = stance_pipe.predict(eligible_texts)
        st_marg = decision_margin(stance_pipe, eligible_texts)
        import numpy as np
        s_marg_abs = np.abs(np.asarray(s_marg).reshape(len(eligible_texts), -1)).max(axis=1)
        for k, pos in enumerate(eligible_positions):
            sentiment_preds[pos] = str(s_pred[k])
            sentiment_margins[pos] = round(float(s_marg_abs[k]), 4)
            stance_preds[pos] = str(st_pred[k])
            stance_margins[pos] = round(float(st_marg[k]), 4)

    for i in range(len(chunk)):
        if not keep_mask[i]:
            continue
        row = chunk.iloc[i]
        dup_group = str(row.get("duplicate_group_id", "")).strip()
        is_dup_non_first = False
        if dup_group:
            if dup_group in seen_duplicate_groups:
                is_dup_non_first = True
            else:
                seen_duplicate_groups.add(dup_group)

        if too_short_all[i]:
            rejected_too_short += 1
            out_rows.append([
                row.get("record_id", ""), row.get("source_dataset", ""), row.get("source_record_id", ""),
                row.get("label", ""), len(stripped_all[i]), dup_group, is_dup_non_first,
                "", "", "", "", "", "", False, "too_short_text",
                texts_all[i], PSEUDO_LABEL_DISCLAIMER,
            ])
            continue

        sentiment = sentiment_preds[i]
        stance_raw = stance_preds[i]
        result = assign_pseudo_label(sentiment, stance_raw)
        pseudo_label, rule = result.pseudo_label, result.rule

        exclusion_reason = ""
        eligible_for_training = True
        if is_dup_non_first:
            exclusion_reason = "duplicate_non_first_occurrence"
            eligible_for_training = False
            duplicate_non_first += 1
        elif pseudo_label == "Unclear":
            exclusion_reason = "unresolved_unclear_kept_separate"
            eligible_for_training = False
            unresolved_unclear += 1

        if eligible_for_training:
            label_counts[pseudo_label] = label_counts.get(pseudo_label, 0) + 1
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

        out_rows.append([
            row.get("record_id", ""), row.get("source_dataset", ""), row.get("source_record_id", ""),
            row.get("label", ""), len(stripped_all[i]), dup_group, is_dup_non_first,
            sentiment, sentiment_margins[i], stance_raw, stance_margins[i],
            pseudo_label, rule, eligible_for_training, exclusion_reason,
            texts_all[i], PSEUDO_LABEL_DISCLAIMER,
        ])

    return out_rows, label_counts, rule_counts, rejected_too_short, duplicate_non_first, unresolved_unclear


def merge_counts(base: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    for k, v in delta.items():
        base[k] = base.get(k, 0) + v
    return base


def run(
    input_path: Path,
    out_dir: Path,
    batch_size: int,
    max_rows: int | None,
    model: str,
    resume: bool,
    min_free_ram_mb: float,
    log_every: int,
) -> int:
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    existing_checkpoint = load_checkpoint(out_dir) if out_dir.exists() else None
    if out_dir.exists() and existing_checkpoint is None:
        print(f"[ERROR] {out_dir} exists and has no checkpoint.json; refusing to touch an "
              f"unrelated/foreign directory. Choose a new --out-dir.", file=sys.stderr)
        return 2
    if existing_checkpoint is not None and existing_checkpoint["status"] == "completed":
        print(f"[STOP] {out_dir} already holds a completed run (write-once). "
              f"Use a new --out-dir to re-run.", file=sys.stderr)
        return 2
    if existing_checkpoint is not None and not resume:
        print(f"[ERROR] {out_dir} holds an incomplete run (last_row_index="
              f"{existing_checkpoint['last_row_index']}). Pass --resume to continue, or choose a "
              f"new --out-dir.", file=sys.stderr)
        return 2

    print("=" * 76)
    print("  PolicyLens -- Sentiment140 Stance Pseudo-Labeling v1")
    print("=" * 76)
    print(f"  Input          : {input_path}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Model variant  : {model}")
    print(f"  Min free RAM   : {min_free_ram_mb} MB")
    print(f"  Resume         : {resume}")
    print()
    print("Hashing input file (this reads the whole 1.6M-row file once, streamed) ...")
    input_sha256 = sha256_file(input_path)
    print(f"  input sha256   : {input_sha256}")

    config = {
        "script": "scripts/run_sentiment140_pseudo_labeling_v1.py",
        "input": _rel(input_path), "batch_size": batch_size, "max_rows": max_rows,
        "model_variant": model, "min_free_ram_mb": min_free_ram_mb,
        "stance_run": _rel(STANCE_RUN),
        "sentiment_model": "data/sentiment_training/sentiment140_hashing_sgd_v1",
        "rule_module": "models/pseudo_labeling.py",
    }

    if existing_checkpoint is not None:
        if existing_checkpoint["input_sha256"] != input_sha256:
            print("[ERROR] Input file hash does not match the checkpoint's recorded hash; "
                  "refusing to resume against a changed input file.", file=sys.stderr)
            return 1
        state = existing_checkpoint
        state["config"]["max_rows"] = max_rows  # allow raising/removing a row cap across resumes
        print(f"Resuming after row index {state['last_row_index']} "
              f"({state['rows_written']} rows already written).")
    else:
        out_dir.mkdir(parents=True)
        state = fresh_state(config, input_sha256)
        with open(output_csv_path(out_dir), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(OUTPUT_COLUMNS)
        save_checkpoint(out_dir, state)

    print("Loading verified stance pipeline ...")
    try:
        stance_pipe, stance_meta = load_verified_pipeline(STANCE_RUN, model)
    except Exception as exc:
        print(f"[ERROR] Stance model load failed: {exc}", file=sys.stderr)
        return 1
    print("Loading verified Sentiment140 sentiment model ...")
    try:
        sentiment_clf = _load_verified_sentiment_model()
    except Exception as exc:
        print(f"[ERROR] Sentiment model load failed: {exc}", file=sys.stderr)
        return 1

    seen_duplicate_groups: set[str] = set(state["seen_duplicate_groups"])
    resume_from_row = state["last_row_index"]
    t0 = time.time()
    batch_idx = 0
    global_row_index = 0
    stop_for_memory = False
    max_row_index = (max_rows - 1) if max_rows is not None else None  # max_rows is a row COUNT; this is the last 0-based index to keep

    with open(output_csv_path(out_dir), "a", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        for chunk in pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=batch_size):
            n_in_chunk = len(chunk)
            chunk_start = global_row_index
            chunk_end = global_row_index + n_in_chunk - 1

            if max_row_index is not None and chunk_start > max_row_index:
                break
            if max_row_index is not None and chunk_end > max_row_index:
                chunk = chunk.iloc[: max_row_index - chunk_start + 1]
                n_in_chunk = len(chunk)
                chunk_end = max_row_index

            mem = psutil.virtual_memory()
            avail_mb = mem.available / (1024 * 1024)
            if avail_mb < min_free_ram_mb:
                print(f"\n[PAUSED] Available RAM ({avail_mb:.1f} MB) below safety floor "
                      f"({min_free_ram_mb} MB) at row {chunk_start:,}. Checkpoint saved; "
                      f"rerun with --resume.", file=sys.stderr)
                stop_for_memory = True
                break

            if chunk_end > resume_from_row:
                (rows, label_delta, rule_delta, short_delta, dup_delta, unclear_delta) = process_chunk(
                    chunk, chunk_start, resume_from_row, stance_pipe, sentiment_clf, seen_duplicate_groups
                )
                for r in rows:
                    writer.writerow(r)
                state["label_counts"] = merge_counts(state["label_counts"], label_delta)
                state["rule_counts"] = merge_counts(state["rule_counts"], rule_delta)
                state["rejected_too_short"] += short_delta
                state["duplicate_non_first"] += dup_delta
                state["unresolved_unclear"] += unclear_delta
                state["rows_written"] += len(rows)

            state["rows_processed"] = chunk_end + 1
            state["last_row_index"] = chunk_end
            state["seen_duplicate_groups"] = sorted(seen_duplicate_groups)
            state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            batch_idx += 1

            if batch_idx % max(1, log_every) == 0:
                out_f.flush()
                save_checkpoint(out_dir, state)
                elapsed = time.time() - t0
                rate = state["rows_processed"] / max(elapsed, 0.001)
                print(f"  batch {batch_idx:5d} | rows {state['rows_processed']:9,d} | "
                      f"rate {rate:7,.0f} rows/s | avail RAM {avail_mb/1024:.2f} GB | "
                      f"accepted {sum(state['label_counts'].values()):,} | "
                      f"rejected_short {state['rejected_too_short']:,} | "
                      f"dup_non_first {state['duplicate_non_first']:,} | "
                      f"unclear {state['unresolved_unclear']:,}", flush=True)

            global_row_index += n_in_chunk
            if max_row_index is not None and chunk_end >= max_row_index:
                break

    elapsed_this_run = time.time() - t0
    state["elapsed_seconds_prior_runs"] = round(state.get("elapsed_seconds_prior_runs", 0.0) + elapsed_this_run, 2)
    reached_end_of_input = max_rows is None or state["rows_processed"] >= max_rows
    # also treat "ran out of rows before max_rows" as completed
    if not stop_for_memory:
        state["status"] = "completed"
    save_checkpoint(out_dir, state)

    if stop_for_memory:
        return 3

    output_sha256 = sha256_file(output_csv_path(out_dir))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed",
        "config": config,
        "disclaimer": PSEUDO_LABEL_DISCLAIMER,
        "known_limitations": KNOWN_LIMITATIONS,
        "rows_processed": state["rows_processed"],
        "rows_written": state["rows_written"],
        "label_counts_training_eligible": state["label_counts"],
        "rule_counts_training_eligible": state["rule_counts"],
        "rejected_too_short": state["rejected_too_short"],
        "duplicate_non_first_excluded": state["duplicate_non_first"],
        "unresolved_unclear_excluded": state["unresolved_unclear"],
        "elapsed_seconds_total": state["elapsed_seconds_prior_runs"],
        "inputs_sha256": {_rel(input_path): input_sha256},
        "model_artifacts_sha256": {
            "stance": stance_meta.get("artifacts_sha256", {}),
            "sentiment_model.joblib": json.loads(
                (ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1" / "manifest.json")
                .read_text(encoding="utf-8")
            ).get("outputs_sha256", {}).get("model.joblib"),
        },
        "outputs_sha256": {output_csv_path(out_dir).name: output_sha256, "checkpoint.json": sha256_file(checkpoint_path(out_dir))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    print()
    print("[COMPLETED]" if not reached_end_of_input else "[COMPLETED - reached requested row limit or end of input]")
    print(f"  Rows processed            : {state['rows_processed']:,}")
    print(f"  Accepted (Support/Neutral/Oppose, training-eligible): {sum(state['label_counts'].values()):,}")
    print(f"    by label                : {state['label_counts']}")
    print(f"  Rejected (too short)      : {state['rejected_too_short']:,}")
    print(f"  Duplicate (non-first)     : {state['duplicate_non_first']:,}")
    print(f"  Unresolved (Unclear)      : {state['unresolved_unclear']:,}")
    print(f"  Output CSV                : {output_csv_path(out_dir)}")
    print(f"  Manifest                  : {out_dir / 'manifest.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=None, help="Process only the first N rows (0-based row index cap); for pilots.")
    ap.add_argument("--model", choices=["linsvm", "logreg"], default="linsvm")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-free-ram-mb", type=float, default=150.0)
    ap.add_argument("--log-every", type=int, default=25, help="Log and checkpoint every N batches.")
    a = ap.parse_args()
    return run(a.input, a.out_dir, a.batch_size, a.max_rows, a.model, a.resume, a.min_free_ram_mb, a.log_every)


if __name__ == "__main__":
    raise SystemExit(main())
