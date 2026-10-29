"""
scripts/train_sentiment_ai_labels_v1.py
=======================================
Separate experiment: consultation-domain SENTIMENT (tone) baselines trained on AI-generated sentiment working labels
for the 400-record v2 pool. Kept apart from stance: different labels, different output folder, no shared features.

Same protocol as scripts/train_stance_ai_labels_v2.py (grouped 60/20/20 train/validation/test, C chosen on validation,
refit on train+validation, single test evaluation, majority and stratified-random baselines, split-seed sensitivity).
Classes: Positive / Negative / Neutral. Mixed (12 records) is excluded by design: a tone mixture is not a class a
bag-of-words baseline can learn from 12 examples, and it is reported as dropped.

    python scripts/train_sentiment_ai_labels_v1.py
Output: data/sentiment_training/consultation_ai_labels_tfidf_v1/ (write-once)
Metrics are agreement with AI labels, not human-verified accuracy.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

from models.stance_classification import pipeline as sp  # noqa: E402
from scripts.train_stance_ai_labels_v2 import EXTRA_SEEDS, C_GRID, cfg_with_c, run_protocol  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402

OUT = ROOT / "data" / "sentiment_training" / "consultation_ai_labels_tfidf_v1"
POOL = ROOT / "data" / "annotation" / "expansion_pool_400_v2.csv"
AI_SENT = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_sentiment_labels_pool400.csv"
LABELS = ["Negative", "Neutral", "Positive"]
DISCLAIMER = ("Sentiment labels are AI-generated working labels, not human annotations. Metrics measure agreement with "
              "those labels on held-out consultation records. Sentiment is tone, not stance.")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        print(f"[STOP] {OUT} exists", file=sys.stderr)
        return 2
    pool = pd.read_csv(POOL, dtype=str, keep_default_na=False)
    ai = pd.read_csv(AI_SENT, dtype=str, keep_default_na=False)
    assert pool.pilot_id.tolist() == ai.pilot_id.tolist() and pool.original_text.tolist() == ai.original_text.tolist()
    d = pool.drop(columns=["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]).copy()
    d["ai_sentiment"] = ai.ai_sentiment
    cg, mapping = corpus_groups(d)
    keep = d.ai_sentiment.isin(LABELS)
    dropped = d.loc[~keep, "ai_sentiment"].value_counts().to_dict()
    d, cg = d[keep].reset_index(drop=True), cg[keep].reset_index(drop=True)
    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    groups = sp.leakage_groups(d, cg, cfg["near_dup_jaccard"], cfg["near_dup_containment"])
    X, y = d.original_text.to_numpy(), d.ai_sentiment.to_numpy()
    res = run_protocol(X, y, groups, LABELS, cfg, cfg["seed"])
    tr, va, te = res["idx"]
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        if set(y[idx]) != set(LABELS):
            print(f"[BLOCKED] {name} lacks classes", file=sys.stderr)
            return 3
    rows = []
    for s in [cfg["seed"]] + EXTRA_SEEDS:
        r = res if s == cfg["seed"] else run_protocol(X, y, groups, LABELS, cfg, s)
        rows.append({"seed": s, **{m: r["models"][m]["test"]["macro_f1"] for m in sp.MODELS}, "majority": r["majority"]["macro_f1"]})
    split = np.full(len(d), "", dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    d["split"], d["leakage_group"] = split, groups
    metrics = {"label_type": "ai_generated_unverified", "disclaimer": DISCLAIMER, "labels": LABELS, "dropped": {"excluded_by_design": dropped},
               "n": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))}, "n_groups": int(len(np.unique(groups))),
               "corpus_group_mapping": mapping,
               "class_distribution": {s: d.loc[d.split == s, "ai_sentiment"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict() for s in ("train", "validation", "test")},
               "baselines": {"majority_class": res["majority"], "stratified_random": res["stratified_random"]},
               "models": {m: {k: v for k, v in r.items() if k != "_pred"} for m, r in res["models"].items()},
               "c_grid": C_GRID,
               "split_sensitivity": {"per_seed": rows, "summary": {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4), "sd": round(float(np.std([r[k] for r in rows])), 4)}
                                                                  for k in list(sp.MODELS) + ["majority"]}}}
    OUT.mkdir(parents=True)
    import joblib
    arts = {}
    for m, pipe in res["pipelines"].items():
        joblib.dump(pipe, OUT / f"pipeline_{m}.joblib")
        arts[f"pipeline_{m}.joblib"] = sha(OUT / f"pipeline_{m}.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"base": cfg, "final_per_model": {m: cfg_with_c(cfg, m, res["models"][m]["selected_C"]) for m in sp.MODELS}}, indent=2) + "\n", encoding="utf-8")
    (OUT / "label_mapping.json").write_text(json.dumps({"labels": LABELS, "construct": "sentiment (tone) of consultation text; NOT stance"}, indent=2) + "\n", encoding="utf-8")
    tp = d.iloc[te][["pilot_id", "source_language", "ai_sentiment", "original_text"]].copy()
    for m in sp.MODELS:
        tp[f"pred_{m}"] = res["models"][m]["_pred"]
    tp.to_csv(OUT / "test_predictions.csv", index=False, lineterminator="\n")
    d[["pilot_id", "source_language", "ai_sentiment", "leakage_group", "split"]].to_csv(OUT / "split_assignments.csv", index=False, lineterminator="\n")
    for f in ("metrics.json", "config.json", "label_mapping.json", "test_predictions.csv", "split_assignments.csv"):
        arts[f] = sha(OUT / f)
    meta = {"status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/train_sentiment_ai_labels_v1.py", "sklearn_version": sklearn.__version__, "python": platform.python_version(),
            "human_labels_only": False, "label_type": "ai_generated_unverified", "disclaimer": DISCLAIMER,
            "artifacts_sha256": arts, "inputs_sha256": {"data/annotation/expansion_pool_400_v2.csv": sha(POOL),
                                                        "data/ai_labels/stance_sentiment_ai_v1/ai_sentiment_labels_pool400.csv": sha(AI_SENT)}}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    b = metrics["baselines"]
    print(json.dumps({"n": metrics["n"], "dropped": dropped, "dist": metrics["class_distribution"], "majority": b["majority_class"]["macro_f1"],
                      "random": b["stratified_random"]["macro_f1_mean"],
                      **{m: (v["selected_C"], v["test"]["macro_f1"], v["test"]["macro_f1_ci95_group_bootstrap"], {c: x["f1"] for c, x in v["test"]["per_class"].items()})
                         for m, v in metrics["models"].items()}, "sens": metrics["split_sensitivity"]["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
