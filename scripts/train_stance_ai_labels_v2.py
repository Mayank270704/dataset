"""
scripts/train_stance_ai_labels_v2.py
====================================
TF-IDF + Logistic Regression and TF-IDF + Linear SVM STANCE baselines on AI-generated WORKING labels for the
400-record v2 pool (data/ai_labels/stance_sentiment_ai_v1). Labels are AI-generated and unverified; every metric is
agreement with those AI labels, not human-verified accuracy.

Protocol (fixed before looking at test results)
    1. Labels loaded and validated by models.stance_classification.labels.load_ai_working_labels (round expansion_v2).
    2. Classes with < min_class_count records are dropped and reported (Oppose has 1, Mixed 0).
    3. Leakage groups: pipeline.leakage_groups (same source record, pool duplicate group, corpus near-duplicate group,
       template key, char-5-gram near-duplicates). Normalised identical texts always share a group (asserted).
    4. Grouped stratified split: test = first fold of StratifiedGroupKFold(5) (~20 %); validation = first fold of
       StratifiedGroupKFold(4) on the remainder (~20 %); train = rest (~60 %). No group crosses any boundary.
    5. C is selected per model on VALIDATION macro-F1 only (grid below); the selected model is refitted on
       train+validation and evaluated ONCE on test. Majority-class and stratified-random baselines use the same test set.
    6. Secondary: the whole protocol repeated over extra split seeds (split-sensitivity), never used for selection.

    python scripts/train_stance_ai_labels_v2.py --tag pool400_ai_v1_3class
    python scripts/train_stance_ai_labels_v2.py --tag pool400_ai_v1_support_vs_neutral --exclude-labels Unclear
Write-once: exits 2 if the run directory exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402

LABELS_FILE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_working_labels_pool400.csv"
PROVENANCE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "provenance.json"
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
C_PREFERENCE = [1.0, 0.3, 3.0, 0.1, 10.0]          # tie-break order (closest to the v1 default first)
EXTRA_SEEDS = [43, 44, 45, 46]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def three_way_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, te = next(sg.split(np.zeros(len(y)), y, groups))
    sg2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_r, va_r = next(sg2.split(np.zeros(len(rest)), y[rest], groups[rest]))
    tr, va = rest[tr_r], rest[va_r]
    assert not (set(groups[tr]) & set(groups[va]) or set(groups[tr]) & set(groups[te]) or set(groups[va]) & set(groups[te]))
    return tr, va, te


def cfg_with_c(cfg: dict[str, Any], model: str, c: float) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    out[model]["C"] = c
    return out


def select_and_fit(model: str, X, y, tr, va, labels, cfg) -> tuple[Any, float, dict[str, float]]:
    curve = {}
    for c in C_GRID:
        pipe = sp.build_pipeline(model, cfg_with_c(cfg, model, c)).fit(X[tr], y[tr])
        curve[c] = float(f1_score(y[va], pipe.predict(X[va]), labels=labels, average="macro", zero_division=0))
    best = max(curve.values())
    chosen = next(c for c in C_PREFERENCE if abs(curve[c] - best) < 1e-12)
    trva = np.concatenate([tr, va])
    final = sp.build_pipeline(model, cfg_with_c(cfg, model, chosen)).fit(X[trva], y[trva])
    return final, chosen, {str(k): round(v, 4) for k, v in curve.items()}


def run_protocol(X, y, groups, labels, cfg, seed) -> dict[str, Any]:
    tr, va, te = three_way_split(y, groups, seed)
    res: dict[str, Any] = {"idx": (tr, va, te), "models": {}, "pipelines": {}}
    maj = pd.Series(y[np.concatenate([tr, va])]).value_counts().idxmax()
    res["majority"] = sp.evaluate(y[te], np.full(len(te), maj, dtype=object), labels, groups[te], cfg["bootstrap"], seed) | {"predicts": maj}
    prior = pd.Series(y[np.concatenate([tr, va])]).value_counts(normalize=True)
    rng = np.random.default_rng(seed)
    rnd = [f1_score(y[te], rng.choice(prior.index.to_numpy(), size=len(te), p=prior.values), labels=labels,
                    average="macro", zero_division=0) for _ in range(1000)]
    res["stratified_random"] = {"macro_f1_mean": round(float(np.mean(rnd)), 4),
                                "macro_f1_p2.5_p97.5": [round(float(np.percentile(rnd, 2.5)), 4), round(float(np.percentile(rnd, 97.5)), 4)]}
    for m in sp.MODELS:
        pipe, c, curve = select_and_fit(m, X, y, tr, va, labels, cfg)
        pred = pipe.predict(X[te])
        trva = np.concatenate([tr, va])
        res["pipelines"][m] = pipe
        res["models"][m] = {"selected_C": c, "validation_macro_f1_by_C": curve,
                            "test": sp.evaluate(y[te], pred, labels, groups[te], cfg["bootstrap"], seed),
                            "train_plus_validation_macro_f1_refit": round(float(f1_score(y[trva], pipe.predict(X[trva]), labels=labels, average="macro", zero_division=0)), 4),
                            "test_predicted_distribution": pd.Series(pred).value_counts().to_dict(),
                            "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_), "_pred": pred}
    return res


def write_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    L = ["# Stance baselines on AI-generated working labels (400-record v2 pool, UNVERIFIED)", "",
         f"**{lb.AI_DISCLAIMER}**", "",
         "Label source: data/ai_labels/stance_sentiment_ai_v1 (pilot 50 = pre-existing AI draft 1; expansion 350 = new "
         "single-pass AI labels under guidelines v1.1). Stance toward Open Government; not sentiment.", "",
         f"Labels used: {metrics['labels']}. Dropped: {json.dumps(metrics['dropped'])}.",
         f"Split (seed {metrics['seed']}): train {metrics['n']['train']} / validation {metrics['n']['validation']} / test "
         f"{metrics['n']['test']} records; {metrics['n_groups']} leakage groups; no group or normalised text crosses a boundary.", "",
         "Class distribution: " + json.dumps(metrics["class_distribution"]), "",
         "| Model | Selected C (validation) | Test macro-F1 | 95% CI (group bootstrap) | Accuracy |", "|---|---:|---:|---|---:|"]
    b = metrics["baselines"]["majority_class"]
    L.append(f"| majority class ({b['predicts']}) | - | {b['macro_f1']} | {b['macro_f1_ci95_group_bootstrap']} | {b['accuracy']} |")
    r = metrics["baselines"]["stratified_random"]
    L.append(f"| stratified random (1000 draws) | - | {r['macro_f1_mean']} | {r['macro_f1_p2.5_p97.5']} (draw spread) | - |")
    for m, v in metrics["models"].items():
        L.append(f"| {m} | {v['selected_C']} | {v['test']['macro_f1']} | {v['test']['macro_f1_ci95_group_bootstrap']} | {v['test']['accuracy']} |")
    L += ["", "Validation macro-F1 by C: " + json.dumps({m: v["validation_macro_f1_by_C"] for m, v in metrics["models"].items()}), ""]
    for m, v in metrics["models"].items():
        L += [f"## {m}: per-class (test)", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        L += [f"| {c} | {x['precision']} | {x['recall']} | {x['f1']} | {x['support']} |" for c, x in v["test"]["per_class"].items()]
        cm = v["test"]["confusion_matrix"]
        L += ["", f"Confusion matrix (rows true, cols predicted; {', '.join(cm['labels'])}): {json.dumps(cm['rows_true_cols_pred'])}",
              f"Predicted distribution on test: {json.dumps(v['test_predicted_distribution'])}; fit on train+validation macro-F1 {v['train_plus_validation_macro_f1_refit']}", ""]
    s = metrics["split_sensitivity"]
    L += ["## Split sensitivity (secondary; same protocol, other split seeds)", "", "| Seed | " + " | ".join(s["models"]) + " | majority |", "|---|" + "---:|" * (len(s["models"]) + 1)]
    for row in s["per_seed"]:
        L.append(f"| {row['seed']} | " + " | ".join(str(row[m]) for m in s["models"]) + f" | {row['majority']} |")
    L += ["", "Mean ± sd over all seeds: " + json.dumps(s["summary"]), "",
          "Test sets are small; intervals are wide. Results describe this stratified 400-record sample, not the corpus."]
    (run_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--labels", type=Path, default=LABELS_FILE)
    ap.add_argument("--exclude-labels", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2
    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed
    cfg["exclude_labels"] = list(a.exclude_labels)
    st, gold = lb.load_ai_working_labels(a.labels, "expansion_v2")
    print(f"[AI WORKING LABELS] {lb.AI_DISCLAIMER}\n[{st.state}] {st.detail}")
    if st.state != "ok":
        return 1
    cg, mapping = corpus_groups(gold)
    d = gold.copy()
    d["_cg"] = cg
    dropped: dict[str, Any] = {}
    if cfg["exclude_labels"]:
        m = d.gold_stance.isin(cfg["exclude_labels"])
        dropped["excluded_by_config"] = d.loc[m, "gold_stance"].value_counts().to_dict()
        d = d[~m]
    cnt = d.gold_stance.value_counts()
    rare = cnt[cnt < cfg["min_class_count"]]
    if len(rare):
        dropped["below_min_class_count"] = rare.to_dict()
        d = d[~d.gold_stance.isin(rare.index)]
    d = d.reset_index(drop=True)
    labels = [x for x in lb.SCHEMA_LABELS if x in set(d.gold_stance)]
    groups = sp.leakage_groups(d, d["_cg"], cfg["near_dup_jaccard"], cfg["near_dup_containment"])
    norm = d.original_text.map(normalise_text)
    assert norm.groupby(groups).nunique().sum() >= 1
    for t, g in pd.Series(groups).groupby(norm.values):
        assert g.nunique() == 1, f"normalised-identical text split across groups: {t[:60]!r}"
    X, y = d.original_text.to_numpy(), d.gold_stance.to_numpy()
    res = run_protocol(X, y, groups, labels, cfg, a.seed)
    tr, va, te = res["idx"]
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        if set(y[idx]) != set(labels):
            print(f"[BLOCKED] {name} split lacks classes {sorted(set(labels) - set(y[idx]))}", file=sys.stderr)
            return 3
    split = np.full(len(d), "", dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    d["split"], d["leakage_group"] = split, groups
    for s_a in ("train", "validation"):                           # normalised-text leakage check
        assert not (set(norm[d.split == s_a]) & set(norm[d.split == "test"]))
    sens_rows = []
    for s in [a.seed] + EXTRA_SEEDS:
        r = res if s == a.seed else run_protocol(X, y, groups, labels, cfg, s)
        sens_rows.append({"seed": s, **{m: r["models"][m]["test"]["macro_f1"] for m in sp.MODELS}, "majority": r["majority"]["macro_f1"]})
    summ = {k: {"mean": round(float(np.mean([r[k] for r in sens_rows])), 4), "sd": round(float(np.std([r[k] for r in sens_rows])), 4)}
            for k in list(sp.MODELS) + ["majority"]}
    metrics = {
        "label_type": "ai_generated_unverified", "disclaimer": lb.AI_DISCLAIMER, "labels": labels, "dropped": dropped,
        "seed": a.seed, "n": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "n_groups": int(len(np.unique(groups))), "corpus_group_mapping": mapping,
        "class_distribution": {s: d.loc[d.split == s, "gold_stance"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()
                               for s in ("train", "validation", "test")} | {"all": d.gold_stance.value_counts().reindex(labels).fillna(0).astype(int).to_dict()},
        "baselines": {"majority_class": res["majority"], "stratified_random": res["stratified_random"]},
        "models": {m: {k: v for k, v in r.items() if k != "_pred"} for m, r in res["models"].items()},
        "c_grid": C_GRID, "selection": "validation macro-F1; refit on train+validation; single test evaluation",
        "split_sensitivity": {"models": list(sp.MODELS), "per_seed": sens_rows, "summary": summ},
    }
    run_dir.mkdir(parents=True)
    import joblib
    arts = {}
    for m, pipe in res["pipelines"].items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)
    final_cfg = {m: cfg_with_c(cfg, m, res["models"][m]["selected_C"]) for m in sp.MODELS}
    files = {"metrics.json": metrics, "preprocessing_config.json": {"base": cfg, "final_per_model": final_cfg, "c_grid": C_GRID},
             "label_mapping.json": {"labels": labels, "index": {x: i for i, x in enumerate(labels)}, "schema": lb.SCHEMA_LABELS,
                                    "construct": "stance toward the Open Government initiative/policy (guidelines v1.1); not sentiment, not topic"}}
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)
    cols = ["pilot_id", "source_dataset", "source_record_id", "source_row_number", "source_field", "question_slug",
            "source_language", "gold_stance", "label_source", "leakage_group", "split", "original_text"]
    d[cols].rename(columns={"gold_stance": "ai_stance"}).to_csv(run_dir / "split_assignments.csv", index=False, lineterminator="\n")
    tp = d.iloc[te][["pilot_id", "source_language", "question_slug", "gold_stance", "original_text"]].rename(columns={"gold_stance": "ai_stance"})
    for m in sp.MODELS:
        tp[f"pred_{m}"] = res["models"][m]["_pred"]
    tp.to_csv(run_dir / "test_predictions.csv", index=False, lineterminator="\n")
    for f in ("split_assignments.csv", "test_predictions.csv"):
        arts[f] = _sha(run_dir / f)
    meta = {"status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/train_stance_ai_labels_v2.py", "argv": sys.argv[1:] if argv is None else argv,
            "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(sp.MODELS),
            "artifacts_sha256": arts, "human_labels_only": False, "label_type": "ai_generated_unverified",
            "disclaimer": lb.AI_DISCLAIMER, "merged_or_voted": False,
            "inputs_sha256": {_rel(a.labels): _sha(a.labels), _rel(PROVENANCE): _sha(PROVENANCE)}}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(run_dir, metrics)
    print(json.dumps({"n": metrics["n"], "labels": labels, "dropped": dropped,
                      "majority": res["majority"]["macro_f1"], "random": res["stratified_random"]["macro_f1_mean"],
                      **{m: (v["selected_C"], v["test"]["macro_f1"], v["test"]["macro_f1_ci95_group_bootstrap"]) for m, v in metrics["models"].items()},
                      "sensitivity": summ}))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
