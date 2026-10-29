"""
scripts/train_stance_baselines.py
=================================
Supervised STANCE baselines (TF-IDF + Logistic Regression, TF-IDF + Linear SVM) trained ONLY on genuine,
double-annotated human labels (agreement or human adjudication). See models/stance_classification/pipeline.py.

    python scripts/train_stance_baselines.py --check-only     # readiness; never fits, never writes
    python scripts/train_stance_baselines.py                  # fit + evaluate + save, only if labels allow it
    python scripts/train_stance_baselines.py --rounds pilot --tag pilot_only --exclude-labels Unclear

Exit codes: 0 trained (or check passed), 1 invalid annotation/adjudication files, 2 output exists,
            3 blocked: labels missing or insufficient for a valid held-out evaluation (nothing fitted, nothing saved).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.stance_classification import labels as lb  # noqa: E402

OUTPUT_ROOT = ROOT / "data" / "stance_training"
AI_OUTPUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"     # AI working-label runs only
RUN_V1_CORPUS = ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv"
NEAR_DUP = ROOT / "data" / "full_corpus" / "concern_discovery_v1" / "duplicates" / "near_duplicate_groups.csv"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(p).resolve())


def corpus_groups(gold: pd.DataFrame, corpus_path: Path = RUN_V1_CORPUS, dup_path: Path = NEAR_DUP) -> tuple[pd.Series | None, dict[str, Any]]:
    """Map annotated records to concern-discovery near-duplicate groups via dataset + record id + normalised text."""
    if not corpus_path.is_file() or not dup_path.is_file() or gold.empty:
        return None, {"mapped": 0, "note": "corpus near-duplicate groups unavailable; lexical grouping only"}
    from models.concern_discovery.schema import normalise_text
    c = pd.read_csv(corpus_path, dtype=str, keep_default_na=False)
    g = pd.read_csv(dup_path, dtype=str, keep_default_na=False)
    gid = dict(zip(g["unique_id"].astype(int), g["dup_group_id"].astype(int)))
    key = {(a, b, normalise_text(t)): gid[int(u)] for a, b, t, u in zip(c["source_dataset"], c["source_record_id"], c["original_text"], c["unique_id"])}
    s = pd.Series([key.get((a, b, normalise_text(t))) for a, b, t in zip(gold["source_dataset"], gold["source_record_id"], gold["original_text"])],
                  index=gold.index, dtype="float")
    return s, {"mapped": int(s.notna().sum()), "of": int(len(s))}


def build_config(a: argparse.Namespace) -> dict[str, Any]:
    from models.stance_classification.pipeline import DEFAULT_CONFIG
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text(encoding="utf-8")))
    for k in ("test_size", "seed", "min_gold", "min_class_count"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.exclude_labels:
        bad = sorted(set(a.exclude_labels) - set(lb.SCHEMA_LABELS))
        if bad:
            raise SystemExit(f"--exclude-labels must be schema labels {lb.SCHEMA_LABELS}; got {bad}")
        cfg["exclude_labels"] = list(a.exclude_labels)
    return cfg


def write_report(run_dir: Path, meta: dict[str, Any], metrics: dict[str, Any], ai_mode: bool = False) -> None:
    if ai_mode:
        head = ["# Stance baseline run on AI-generated working labels (UNVERIFIED)", "",
                f"Status: **{meta['status']}** ({meta['created_utc']}). **{lb.AI_DISCLAIMER}** Label source: one AI-assisted "
                f"draft ({meta.get('ai_working_label_file')}); no human annotation, merging or voting. Stance toward Open "
                "Government; not sentiment, not topic. Every 'macro-F1', 'precision' and 'recall' below is agreement with "
                "the AI labels.", ""]
    else:
        head = ["# Stance baseline training run", "",
                f"Status: **{meta['status']}** ({meta['created_utc']}). Trained only on genuine double-annotated human labels "
                "(agreement or human adjudication). Stance toward Open Government; not sentiment, not topic.", ""]
    L = head + [
         f"Train {metrics['n_train']} / test {metrics['n_test']} records; {metrics['n_groups']} leakage groups (no group crosses the split).", "",
         "| Model | Test macro-F1 | 95% CI (group bootstrap) | Accuracy | Grouped CV on train (mean ± sd) |", "|---|---:|---|---:|---|"]
    base = metrics["majority_class_baseline"]
    L.append(f"| majority class ({base['predicts']}) | {base['macro_f1']} | {base['macro_f1_ci95_group_bootstrap']} | {base['accuracy']} | — |")
    for m, r in metrics["models"].items():
        cv = r["grouped_cv_on_train"]
        cvs = f"{cv['macro_f1_mean']} ± {cv['macro_f1_std']} ({cv['folds']} folds)" if cv.get("status") == "ok" else cv.get("reason", "")
        L.append(f"| {m} | {r['test']['macro_f1']} | {r['test']['macro_f1_ci95_group_bootstrap']} | {r['test']['accuracy']} | {cvs} |")
    L += ["", "Class distribution: " + json.dumps(metrics["class_distribution"]), ""]
    for m, r in metrics["models"].items():
        L += [f"## {m}: per-class results (test)", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        L += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for c, v in r["test"]["per_class"].items()]
        cm = r["test"]["confusion_matrix"]
        L += ["", "Confusion matrix (rows true, columns predicted; order " + ", ".join(cm["labels"]) + "): " + json.dumps(cm["rows_true_cols_pred"]), ""]
    L += ["Small test sets give wide intervals; results describe this stratified sample, not population accuracy.", ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Train TF-IDF stance baselines on genuine human labels only")
    ap.add_argument("--check-only", action="store_true", help="report readiness; never fit or write")
    ap.add_argument("--rounds", nargs="+", choices=list(lb.ROUND_ORDER), default=list(lb.ROUND_ORDER))
    ap.add_argument("--tag", help="run directory name (default: UTC timestamp)")
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--config", help="JSON overriding DEFAULT_CONFIG keys")
    ap.add_argument("--test-size", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--min-gold", type=int)
    ap.add_argument("--min-class-count", type=int)
    ap.add_argument("--exclude-labels", nargs="*")
    ap.add_argument("--ai-working-labels", type=Path,
                    help="separate mode: ONE AI-assisted draft of the pilot pool as unverified working labels; "
                         "writes to data/stance_training_ai_working_labels and never reads human annotation files")
    a = ap.parse_args(argv)
    cfg = build_config(a)
    ai_mode = a.ai_working_labels is not None
    if ai_mode:
        if a.output_root.resolve() == OUTPUT_ROOT.resolve():
            a.output_root = AI_OUTPUT_ROOT                    # AI runs never share the human run folder
        if a.rounds != list(lb.ROUND_ORDER) and a.rounds != ["pilot"]:
            raise SystemExit("--ai-working-labels covers the pilot pool only")
        a.rounds = ["pilot"]
        st, gold = lb.load_ai_working_labels(a.ai_working_labels)
        statuses, gold = [st], (gold if gold is not None else pd.DataFrame())
        print(f"[AI WORKING LABELS] {lb.AI_DISCLAIMER}")
    else:
        statuses, gold = lb.build_gold(tuple(a.rounds))
    for st in statuses:
        print(f"[{st.state.upper():7}] {st.round}: {st.detail}")
        for e in st.errors[3:]:
            print(f"          {e}")
    if any(st.state == "invalid" for st in statuses):
        print("[INVALID] fix the annotation/adjudication files above; nothing was fitted or written.", file=sys.stderr)
        return 1
    if all(st.state == "missing" for st in statuses):
        if ai_mode:
            print("[BLOCKED] AI working-label file not found. Pipeline not fitted; no model saved.")
        else:
            print("[BLOCKED] No genuine stance labels: both annotators' completed files are required "
                  "(data/annotation/completed/). Pipeline not fitted; no model saved.")
        return 3
    usable = gold[gold["gold_stance"] != ""]
    print(f"gold records: {len(usable)}; classes {usable['gold_stance'].value_counts().to_dict()}; "
          f"excluded {gold.loc[gold['excluded_reason'] != '', 'excluded_reason'].value_counts().to_dict()}")
    from models.stance_classification import pipeline as sp
    cg, mapping = corpus_groups(gold)
    try:
        prep = sp.prepare(gold, cfg, cg)
    except sp.TrainingBlocked as exc:
        print(f"[BLOCKED] {exc}. Pipeline not fitted; no model saved.")
        return 3
    print(f"split: {len(prep['train_idx'])} train / {len(prep['test_idx'])} test; labels {prep['labels']}; dropped {prep['dropped']}")
    if a.check_only:
        print("[READY] labels allow a valid grouped held-out evaluation (check only; nothing fitted).")
        return 0
    run_dir = a.output_root / f"run_{a.tag or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2
    result = sp.train_and_evaluate(prep, cfg)
    inputs = {}
    if ai_mode:
        ref = Path(lb.aaa.ROUNDS["pilot"]["reference"])
        inputs[_rel(ref)] = _sha(ref)
        inputs[_rel(a.ai_working_labels)] = _sha(a.ai_working_labels)
    else:
        for r in a.rounds:
            c = lb.aaa.ROUNDS[r]
            for k in ("reference", "completed_a", "completed_b"):
                p = Path(c[k])
                if p.is_file():
                    inputs[_rel(p)] = _sha(p)
            adj = lb.adjudication_path(r)
            if adj.is_file():
                inputs[_rel(adj)] = _sha(adj)
    prov = {"rounds": a.rounds, "inputs_sha256": inputs, "gold_records_used": int(len(prep["data"])),
            "label_sources": prep["data"]["label_source"].value_counts().to_dict(),
            "excluded_records": gold.loc[gold["excluded_reason"] != "", "excluded_reason"].value_counts().to_dict(),
            "corpus_group_mapping": mapping}
    if ai_mode:
        prov |= {"human_labels_only": False, "label_type": "ai_generated_unverified", "disclaimer": lb.AI_DISCLAIMER,
                 "ai_working_label_file": _rel(a.ai_working_labels), "merged_or_voted": False}
    meta = sp.save_run(run_dir, prep, result, cfg, prov)
    write_report(run_dir, meta, json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), ai_mode)
    for m, r in result["models"].items():
        print(f"{m}: test macro-F1 {r['test']['macro_f1']} (CI {r['test']['macro_f1_ci95_group_bootstrap']})")
    print(f"[TRAINED] saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
