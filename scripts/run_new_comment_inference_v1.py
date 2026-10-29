"""
scripts/run_new_comment_inference_v1.py
=======================================
Run supported new-comment inference (models/new_comment_inference.py).

    python scripts/run_new_comment_inference_v1.py --input my_comments.csv --text-column text --out results.csv
    python scripts/run_new_comment_inference_v1.py --smoke-test     # writes data/inference/new_comment_inference_v1/

The smoke test uses 300 corpus entries that are NOT in the 400-record labelled pool (no shared source record and no
shared normalised text) plus synthetic edge cases (empty, whitespace, one word, very long, French). It checks that
the pipeline runs end to end, records prediction distributions and runtime, and confirms that concern-group
assignment is refused. It has no labels, so it measures nothing about accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.new_comment_inference import PROVENANCE_NOTE, NewCommentPredictor  # noqa: E402

OUT = ROOT / "data" / "inference" / "new_comment_inference_v1"
EDGE_CASES = ["", "   ", "ok", "Transparency.", "x" * 5000,
              "Le gouvernement devrait publier toutes les données sur les contrats publics.",
              "I strongly oppose publishing any more government data; it wastes money."]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def smoke_test() -> int:
    if OUT.exists():
        print(f"[STOP] {OUT} exists", file=sys.stderr)
        return 2
    import psutil
    ram0 = psutil.virtual_memory().available // 2**20
    t0 = time.time()
    pred = NewCommentPredictor()
    load_s = time.time() - t0
    corpus = pd.read_csv(ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    pool = pd.read_csv(ROOT / "data" / "annotation" / "expansion_pool_400_v2.csv", dtype=str, keep_default_na=False)
    used_rec = set(pool.source_dataset + "#" + pool.source_record_id)
    used_txt = set(pool.original_text.map(normalise_text))
    cand = corpus[~(corpus.source_dataset + "#" + corpus.source_record_id).isin(used_rec) & ~corpus.original_text.map(normalise_text).isin(used_txt)]
    sample = pd.concat([cand[cand.source_dataset == ds].sample(n=100, random_state=42) for ds in sorted(cand.source_dataset.unique())])
    texts = sample.original_text.tolist() + EDGE_CASES
    t1 = time.time()
    res = pred.predict(texts)
    infer_s = time.time() - t1
    res.insert(0, "corpus_id", sample.corpus_id.tolist() + [f"edge_{i}" for i in range(len(EDGE_CASES))])
    res.insert(1, "language", sample.language.tolist() + ["synthetic"] * len(EDGE_CASES))
    s = res.iloc[:len(sample)]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": PROVENANCE_NOTE,
               "model": pred.model, "stance_run": pred.stance_run.relative_to(ROOT).as_posix(), "sentiment_run": pred.sentiment_run.relative_to(ROOT).as_posix(),
               "sample": {"entries": int(len(sample)), "per_dataset": sample.source_dataset.value_counts().to_dict(),
                          "excluded": "entries sharing a source record or normalised text with the 400-record labelled pool"},
               "stance_distribution": s.stance_suggestion.value_counts().to_dict(),
               "stance_by_language": {k: g.stance_suggestion.value_counts().to_dict() for k, g in s.groupby("language")},
               "sentiment_distribution": s.sentiment_suggestion.value_counts().to_dict(),
               "stance_margin_median": float(pd.to_numeric(s.stance_margin).median()),
               "edge_cases": res.iloc[len(sample):][["input_status", "stance_suggestion", "sentiment_suggestion"]].assign(text=[t[:40] for t in EDGE_CASES]).to_dict("records"),
               "concern_group_status": res.concern_group_status.iloc[0], "concern_groups_assigned": int(res.concern_group.notna().sum()),
               "runtime_s": {"load": round(load_s, 2), "inference": round(infer_s, 3)}, "available_ram_mb_before": int(ram0),
               "encoder_loaded": False, "encoder_note": "multilingual MiniLM is cached locally but not needed: no validated cluster-assignment method exists",
               "accuracy_measured": False}
    OUT.mkdir(parents=True)
    res.assign(text=res.text.str.slice(0, 500)).to_csv(OUT / "smoke_test_predictions.csv", index=False, lineterminator="\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"outputs_sha256": {f.name: sha(f) for f in sorted(OUT.iterdir())}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", choices=["linsvm", "logreg"], default="linsvm")
    a = ap.parse_args(argv)
    if a.smoke_test:
        return smoke_test()
    if not a.input or not a.out:
        ap.error("--input and --out are required unless --smoke-test")
    if a.out.exists():
        print(f"[STOP] {a.out} exists", file=sys.stderr)
        return 2
    df = pd.read_csv(a.input, dtype=str, keep_default_na=False)
    res = NewCommentPredictor(model=a.model).predict(df[a.text_column].tolist())
    res.to_csv(a.out, index=False, lineterminator="\n")
    print(f"{len(res)} comments -> {a.out}. {PROVENANCE_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
