#!/usr/bin/env python
"""
scripts/train_stance_oppose_filtering_v1.py
=============================================
Controlled follow-up to run_pool400_plus_all_sources_v1: tries to reduce noisy Oppose
pseudo-labels in training, via two variants, and reports whether it actually helps.

Variant "high_confidence" -- keep only pseudo-labelled rows whose stance decision-margin (and,
for Oppose rows specifically, sentiment decision-margin too) is at or above a percentile cutoff.
The cutoff percentile (25/50/75, i.e. how aggressive the filter is) is a hyperparameter, selected
EXACTLY like C is selected elsewhere in this project: tried on the pool400 VALIDATION split only,
never on the held-out test split, then the winning cutoff is used to refit on train+validation and
evaluated ONCE on test. This is the same "select on validation, single test evaluation" protocol
as scripts/train_stance_ai_labels_v2.py -- it is not reused code-for-code (a different kind of
hyperparameter), but it is the same methodology, applied the same way, for the same reason: to
avoid tuning anything on the data used to report the final number.

Variant "no_oppose" -- drop every Oppose-labelled pseudo-row from training entirely (no threshold,
no tuning needed). The model then has ZERO Oppose training examples (pool400 itself has only 1,
already dropped below min_class_count in every run), so it structurally cannot predict Oppose --
identical behavior to the original run_pool400_ai_v1_3class model for that class, while still
carrying whatever Support/Neutral benefit (if any) the pseudo-labels provide.

Preserved four-class framework
-------------------------------
The output label space is still documented as Support/Neutral/Oppose/Unclear (ALL_FOUR_LABELS) in
every run's label_mapping.json, matching every prior run in this project. Unclear is NEVER
augmented by pseudo-labels in any variant (the pseudo-labeling rule always keeps Unclear
raw-stance predictions out of the training-eligible pseudo pool by design -- see
models/pseudo_labeling.py); its only training signal, in every variant including this one, is the
32 genuine pool400 Unclear records, unchanged. Whether a given variant's fitted model can actually
output "Oppose" depends on whether any Oppose rows survive that variant's filter -- this is checked
and reported explicitly, not assumed.

No claim of validated Oppose accuracy is made: the held-out test set has 1 genuine Oppose example
project-wide (dropped below min_class_count, same as every prior run), so Oppose precision/recall
cannot be computed against real ground truth in any variant.

Usage
    python scripts/train_stance_oppose_filtering_v1.py --variant high_confidence --tag pool400_plus_filtered_s140_corpus_v1
    python scripts/train_stance_oppose_filtering_v1.py --variant no_oppose --tag pool400_plus_no_oppose_v1
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

from models.concern_discovery.schema import normalise_text  # noqa: E402
from models.stance_classification import labels as lb  # noqa: E402
from models.stance_classification import pipeline as sp  # noqa: E402
from models.stance_classification.pipeline import load_trained_pipeline  # noqa: E402
from scripts.train_stance_baselines import corpus_groups  # noqa: E402
from scripts.train_stance_expanded_v1 import (  # noqa: E402
    ALL_FOUR_LABELS, AI_LABEL_SOURCE, ORIGINAL_RUN_DIR, POOL_LABELS_FILE, POOL_PROVENANCE,
    reservoir_sample_pseudo_labels, three_way_split,
)
from scripts.train_stance_all_sources_v1 import (  # noqa: E402
    CORPUS_LABEL_SOURCE, CORPUS_PSEUDO_CSV, S140_LABEL_SOURCE, SENTIMENT140_ONLY_RUN_DIR,
    SENTIMENT140_TEST_CSV, SENTIMENT140_TRAIN_CSV, SENTIMENT140_TRAIN_MANIFEST,
    load_simple_pseudo_csv,
)

ALL_SOURCES_RUN_DIR = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_plus_all_sources_v1"
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"
THRESHOLD_GRID_PERCENTILES = [25, 50, 75]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def load_all_pseudo_sources(seed: int, per_class_cap: int) -> pd.DataFrame:
    """Identical sourcing to scripts/train_stance_all_sources_v1.py: all corpus + all s140-test +
    a capped seeded reservoir sample of s140-train. Deterministic given the same seed/cap."""
    s140_sample, s140_report = reservoir_sample_pseudo_labels(SENTIMENT140_TRAIN_CSV, per_class_cap, seed)
    s140_train_block = pd.DataFrame({
        "original_text": s140_sample["cleaned_text"].values, "gold_stance": s140_sample["pseudo_label"].values,
        "label_source": S140_LABEL_SOURCE, "source_dataset": s140_sample["source_dataset"].values,
        "source_record_id": s140_sample["source_record_id"].values, "rule": s140_sample["rule"].values,
        "sentiment_prediction": s140_sample["sentiment_prediction"].values,
        "stance_margin_pseudo": s140_sample["stance_margin"].astype(float).values,
        "sentiment_margin_pseudo": s140_sample["sentiment_margin"].astype(float).values,
    })
    s140_test_block = load_simple_pseudo_csv(SENTIMENT140_TEST_CSV, "cleaned_text", S140_LABEL_SOURCE)
    corpus_block = load_simple_pseudo_csv(CORPUS_PSEUDO_CSV, "original_text", CORPUS_LABEL_SOURCE)
    pseudo = pd.concat([corpus_block, s140_test_block, s140_train_block], ignore_index=True)
    return pseudo, s140_report


def filter_high_confidence(pseudo: pd.DataFrame, percentile: float) -> tuple[pd.DataFrame, dict[str, float]]:
    stance_cut = float(np.percentile(pseudo["stance_margin_pseudo"], percentile))
    sentiment_cut = float(np.percentile(pseudo["sentiment_margin_pseudo"], percentile))
    keep = (pseudo["stance_margin_pseudo"] >= stance_cut) & (
        (pseudo["gold_stance"] != "Oppose") | (pseudo["sentiment_margin_pseudo"] >= sentiment_cut)
    )
    return pseudo[keep].copy(), {"percentile": percentile, "stance_margin_cutoff": round(stance_cut, 4),
                                  "sentiment_margin_cutoff_oppose_only": round(sentiment_cut, 4)}


def fit_eval(cfg: dict[str, Any], model: str, train_df: pd.DataFrame, eval_X: np.ndarray, eval_y: np.ndarray):
    pipe = sp.build_pipeline(model, cfg).fit(train_df.original_text.to_numpy(), train_df.gold_stance.to_numpy())
    pred = pipe.predict(eval_X)
    return pipe, pred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", required=True, choices=["high_confidence", "no_oppose"])
    ap.add_argument("--per-class-cap-sentiment140", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2

    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed

    # ── 1. pool400: identical split protocol, but keep train/validation SEPARATE this time ──
    st, gold = lb.load_ai_working_labels(POOL_LABELS_FILE, "expansion_v2")
    print(f"[AI WORKING LABELS] {lb.AI_DISCLAIMER}\n[{st.state}] {st.detail}")
    if st.state != "ok":
        return 1
    cg, corpus_mapping = corpus_groups(gold)
    d = gold.copy()
    d["_cg"] = cg
    cnt = d.gold_stance.value_counts()
    rare = cnt[cnt < cfg["min_class_count"]]
    dropped_pool = rare.to_dict() if len(rare) else {}
    if len(rare):
        d = d[~d.gold_stance.isin(rare.index)]
    d = d.reset_index(drop=True)
    pool_labels = [x for x in lb.SCHEMA_LABELS if x in set(d.gold_stance)]
    groups = sp.leakage_groups(d, d["_cg"], cfg["near_dup_jaccard"], cfg["near_dup_containment"])
    norm = d.original_text.map(normalise_text)
    for t, g in pd.Series(groups).groupby(norm.values):
        assert g.nunique() == 1, f"normalised-identical text split across groups: {t[:60]!r}"

    y_pool = d.gold_stance.to_numpy()
    tr, va, te = three_way_split(y_pool, groups, a.seed)
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        if set(y_pool[idx]) != set(pool_labels):
            print(f"[BLOCKED] pool400 {name} split lacks classes {sorted(set(pool_labels) - set(y_pool[idx]))}", file=sys.stderr)
            return 3

    def pool_block(idx):
        sub = d.iloc[idx]
        return pd.DataFrame({
            "original_text": sub["original_text"].values, "gold_stance": sub["gold_stance"].values,
            "label_source": AI_LABEL_SOURCE, "source_dataset": sub["source_dataset"].values,
            "source_record_id": sub["source_record_id"].values,
            "rule": "", "sentiment_prediction": "", "stance_margin_pseudo": np.nan, "sentiment_margin_pseudo": np.nan,
        })

    pool_tr, pool_va, pool_te = pool_block(tr), pool_block(va), pool_block(te)
    X_test, y_test = pool_te.original_text.to_numpy(), pool_te.gold_stance.to_numpy()
    X_val, y_val = pool_va.original_text.to_numpy(), pool_va.gold_stance.to_numpy()
    groups_test = groups[te]
    print(f"pool400 usable: {len(d)} (dropped {dropped_pool}); train {len(pool_tr)} / "
          f"validation {len(pool_va)} / test {len(pool_te)} (test held out, NEVER used for threshold selection)")

    # ── 2. all pseudo sources (same sourcing as run_pool400_plus_all_sources_v1) ──
    pseudo, s140_report = load_all_pseudo_sources(a.seed, a.per_class_cap_sentiment140)
    print(f"Unfiltered combined pseudo pool: {len(pseudo)} rows; by class {pseudo.gold_stance.value_counts().to_dict()}")

    selection: dict[str, Any] = {}
    if a.variant == "high_confidence":
        # ── select percentile on VALIDATION ONLY, never on test ──
        search: dict[str, dict[str, Any]] = {}
        for p in THRESHOLD_GRID_PERCENTILES:
            filtered, cutoffs = filter_high_confidence(pseudo, p)
            row = {"cutoffs": cutoffs, "n_kept": len(filtered),
                   "by_class": filtered.gold_stance.value_counts().to_dict()}
            for m in sp.MODELS:
                train_df = pd.concat([pool_tr, filtered], ignore_index=True)
                _, pred = fit_eval(cfg, m, train_df, X_val, y_val)
                row[f"{m}_validation_macro_f1"] = round(float(
                    f1_score(y_val, pred, labels=pool_labels, average="macro", zero_division=0)
                ), 4)
            search[str(p)] = row
            print(f"  percentile {p}: kept {row['n_kept']} rows {row['by_class']}; "
                  f"val macro-F1 linsvm={row['linsvm_validation_macro_f1']} logreg={row['logreg_validation_macro_f1']}")
        best_by_model = {}
        for m in sp.MODELS:
            scores = {p: search[str(p)][f"{m}_validation_macro_f1"] for p in THRESHOLD_GRID_PERCENTILES}
            best = max(scores.values())
            chosen = min(p for p in THRESHOLD_GRID_PERCENTILES if abs(scores[p] - best) < 1e-12)  # tie -> least aggressive
            best_by_model[m] = chosen
        selection = {"grid": THRESHOLD_GRID_PERCENTILES, "search": search, "selected_percentile_by_model": best_by_model,
                     "selection_rule": "max validation macro-F1 (pool400 validation split, never the test split); ties broken toward the LOWEST (least aggressive) percentile"}
        print(f"Selected percentiles (validation-only): {best_by_model}")
    else:
        selection = {"rule": "no threshold search; all Oppose pseudo-rows dropped unconditionally"}

    # ── 3. final refit per model on train+validation, evaluate ONCE on test ──
    final_pipelines: dict[str, Any] = {}
    new_metrics: dict[str, Any] = {}
    original_metrics: dict[str, Any] = {}
    final_train_composition: dict[str, Any] = {}
    for m in sp.MODELS:
        if a.variant == "high_confidence":
            p = selection["selected_percentile_by_model"][m]
            filtered, cutoffs = filter_high_confidence(pseudo, p)
        else:
            filtered = pseudo[pseudo["gold_stance"] != "Oppose"].copy()
            cutoffs = {}
        train_final = pd.concat([pool_tr, pool_va, filtered], ignore_index=True)
        final_train_composition[m] = {
            "n": len(train_final), "by_class": train_final.gold_stance.value_counts().to_dict(),
            "by_source": train_final.label_source.value_counts().to_dict(),
            "cutoffs_used": cutoffs, "oppose_rows_survived_filter": int((filtered["gold_stance"] == "Oppose").sum()),
        }
        pipe = sp.build_pipeline(m, cfg).fit(train_final.original_text.to_numpy(), train_final.gold_stance.to_numpy())
        final_pipelines[m] = pipe
        pred = pipe.predict(X_test)
        new_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
            "model_classes_": sorted(pipe.classes_.tolist()) if hasattr(pipe, "classes_") else sorted(pipe.named_steps["clf"].classes_.tolist()),
            "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_),
        }
        original_pipe = load_trained_pipeline(ORIGINAL_RUN_DIR, m)
        orig_pred = original_pipe.predict(X_test)
        original_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, orig_pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(orig_pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
        }

    s140_only_metrics = json.loads((SENTIMENT140_ONLY_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    all_sources_metrics = json.loads((ALL_SOURCES_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))

    # ── 4. save (write-once) ──
    run_dir.mkdir(parents=True)
    import joblib
    arts: dict[str, str] = {}
    for m, pipe in final_pipelines.items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)

    disclaimer = (
        lb.AI_DISCLAIMER + f" ADDITIONALLY: this run ('{a.variant}' variant) attempts to reduce noisy Oppose "
        "pseudo-labels in training. Oppose is UNVERIFIED everywhere -- the held-out test set has only 1 "
        "genuine Oppose example project-wide and cannot score Oppose precision/recall. Unclear is never "
        "augmented by pseudo-labels in any variant; its only signal is the pool400 records."
    )
    unclear_handling_note = (
        f"Unclear training examples: {int((pool_tr.gold_stance=='Unclear').sum() + (pool_va.gold_stance=='Unclear').sum())} "
        "(from pool400 train+validation only; 0 from any pseudo-label source in any variant -- the pseudo-labeling "
        "rule in models/pseudo_labeling.py always excludes Unclear from the training-eligible pseudo pool)."
    )
    metrics = {
        "variant": a.variant, "disclaimer": disclaimer, "unclear_handling": unclear_handling_note,
        "pool400": {"usable": len(d), "dropped": dropped_pool, "labels_evaluated": pool_labels,
                    "n_train": len(pool_tr), "n_validation": len(pool_va), "n_test_held_out": len(pool_te),
                    "corpus_group_mapping": corpus_mapping},
        "unfiltered_pseudo_pool": {"n": len(pseudo), "by_class": pseudo.gold_stance.value_counts().to_dict(),
                                   "sentiment140_train_reservoir": s140_report},
        "threshold_selection": selection,
        "final_training_composition_per_model": final_train_composition,
        "held_out_test_set": {"n": len(pool_te), "class_distribution": pd.Series(y_test).value_counts().to_dict(),
                              "note": "Identical pool400 test split used by every prior run (same seed/protocol). "
                                      "Never used for threshold selection -- selection used the validation split only."},
        "model_1_original": original_metrics,
        "model_2_sentiment140_only": {m: s140_only_metrics["new_model"][m]["test_vs_pool_labels"] for m in sp.MODELS},
        "model_3_all_sources_unfiltered": {m: all_sources_metrics["model_3_all_sources"][m]["test_vs_pool_labels"] for m in sp.MODELS},
        "model_4_this_run": new_metrics,
        "comparison_note": ("All four models evaluated on the IDENTICAL 80-record held-out pool400 test split "
                            "(same seed/protocol throughout this project). macro_f1 computed over classes present "
                            "in the test set (Support/Neutral/Unclear). test_predicted_distribution_all_4_labels "
                            "shows Oppose-firing behavior descriptively -- not a scored metric (no genuine Oppose "
                            "ground truth in the test set)."),
    }
    files = {
        "metrics.json": metrics,
        "preprocessing_config.json": {"base": cfg, "note": "same TF-IDF/model config as run_pool400_ai_v1_3class"},
        "label_mapping.json": {"labels": pool_labels, "all_model_output_labels": ALL_FOUR_LABELS, "schema": lb.SCHEMA_LABELS,
                               "construct": "stance toward the Open Government initiative/policy (guidelines v1.1); "
                                            "four-class output framework preserved (Support/Neutral/Oppose/Unclear); "
                                            "whether this run's fitted model can output Oppose depends on whether any "
                                            "Oppose rows survived this variant's filter -- see "
                                            "final_training_composition_per_model.*.oppose_rows_survived_filter in metrics.json"},
    }
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)

    pool_te[["source_dataset", "source_record_id", "gold_stance", "original_text"]].assign(split="test").rename(
        columns={"gold_stance": "ai_stance"}
    ).to_csv(run_dir / "held_out_test_set.csv", index=False, lineterminator="\n")
    arts["held_out_test_set.csv"] = _sha(run_dir / "held_out_test_set.csv")

    meta = {
        "status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_stance_oppose_filtering_v1.py", "argv": sys.argv[1:] if argv is None else argv,
        "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(sp.MODELS),
        "artifacts_sha256": arts, "human_labels_only": False, "label_type": "mixed_ai_and_pseudo_unverified",
        "disclaimer": disclaimer, "merged_or_voted": False,
        "inputs_sha256": {
            _rel(POOL_LABELS_FILE): _sha(POOL_LABELS_FILE), _rel(POOL_PROVENANCE): _sha(POOL_PROVENANCE),
            _rel(SENTIMENT140_TRAIN_MANIFEST): _sha(SENTIMENT140_TRAIN_MANIFEST),
            _rel(SENTIMENT140_TEST_CSV.parent / "manifest.json"): _sha(SENTIMENT140_TEST_CSV.parent / "manifest.json"),
            _rel(CORPUS_PSEUDO_CSV.parent / "manifest.json"): _sha(CORPUS_PSEUDO_CSV.parent / "manifest.json"),
        },
        "does_not_modify": [
            "data/stance_training_ai_working_labels/run_pool400_ai_v1_3class",
            "data/stance_training_ai_working_labels/run_pool400_plus_pseudo_s140_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_all_sources_v1",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "variant": a.variant, "pool_test_n": len(pool_te),
        "linsvm": {"m1_original": original_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
                   "m2_s140only": s140_only_metrics["new_model"]["linsvm"]["test_vs_pool_labels"]["macro_f1"],
                   "m3_allsources": all_sources_metrics["model_3_all_sources"]["linsvm"]["test_vs_pool_labels"]["macro_f1"],
                   "m4_this_run": new_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"]},
        "logreg": {"m1_original": original_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
                   "m2_s140only": s140_only_metrics["new_model"]["logreg"]["test_vs_pool_labels"]["macro_f1"],
                   "m3_allsources": all_sources_metrics["model_3_all_sources"]["logreg"]["test_vs_pool_labels"]["macro_f1"],
                   "m4_this_run": new_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"]},
    }, default=str))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
