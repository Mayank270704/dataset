#!/usr/bin/env python
"""
scripts/train_stance_sentiment140_400k_v1.py
================================================
Trains a new stance classifier on a reproducible, exactly-sized, class-aware-stratified sample
(size set by --sample-size, default 400,000) of the ALREADY-COMPLETED Sentiment140 pseudo-labelling
output, combined with every other eligible labelled PolicyLens source (pool400, in-domain corpus
remainder, Sentiment140-test). Does NOT rerun pseudo-labelling, does NOT download anything, and
does NOT touch the Lok Sabha dataset.

Exact source file (located, not regenerated)
----------------------------------------------
    data/pseudo_labeling/sentiment140_stance_v1/pseudo_labels.csv
      -- 1,600,000 rows processed, 1,570,590 eligible_for_training==True (the pseudo-labelling run
         already completed and audited; see data/audit_reports/run_pseudo_labeling_and_training_audit_v1).

Sampling method (same for any --sample-size)
-----------------------------------------------
Class-aware stratified: targets are as close to equal-per-class as --sample-size/3 allows (see
make_class_targets). All three classes must have at least their target available
(302,433 / 656,034 / 612,123 eligible at the 400k/800k scale tested); a shortfall in any class
STOPS the run and reports the actual count rather than silently sampling fewer rows.

Sampling is two-stage, both stages seeded and reproducible, to stay memory-safe on an 8GB laptop:
  1. A single chunked streaming pass over the 1.6M-row CSV (never loaded fully into memory; see
     reservoir_sample_pseudo_labels, reused unchanged from scripts/train_stance_expanded_v1.py)
     draws an intermediate uniform reservoir of up to `overshoot_cap` rows per class.
  2. Each class's intermediate reservoir (itself already a uniform random sample of the full
     eligible pool, by the reservoir-sampling guarantee) is deterministically sub-sampled down to
     its exact target count with pandas' seeded .sample(), giving an exact, reproducible total
     without ever holding more than `overshoot_cap`*3 rows in memory.

When data/stance_training_ai_working_labels/run_pool400_plus_sentiment140_400k_v1 exists, its
metrics.json is automatically loaded for a 3-way comparison (this run vs. that run vs. the
original baseline) -- no re-training of the 400k run, just reading its saved result.

What this does NOT do
-----------------------
* Does not rerun pseudo-labelling or touch the Lok Sabha dataset.
* Does not overwrite data/stance_training_ai_working_labels/run_pool400_ai_v1_3class, the 400k
  run, or any other existing run directory; writes to a new, write-once directory.
* Does not claim validated Oppose accuracy: the held-out test set has only 1 genuine Oppose
  example project-wide (dropped below min_class_count, same as every prior run).

Usage
    python scripts/train_stance_sentiment140_400k_v1.py --tag pool400_plus_sentiment140_400k_v1 --sample-size 400000
    python scripts/train_stance_sentiment140_400k_v1.py --tag pool400_plus_sentiment140_800k_v1 --sample-size 800000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

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
    CORPUS_LABEL_SOURCE, CORPUS_PSEUDO_CSV, S140_LABEL_SOURCE, SENTIMENT140_TEST_CSV,
    SENTIMENT140_TRAIN_CSV, SENTIMENT140_TRAIN_MANIFEST, load_simple_pseudo_csv,
)

SENTIMENT140_TRAIN_FULL_MANIFEST = SENTIMENT140_TRAIN_MANIFEST
OUT_ROOT = ROOT / "data" / "stance_training_ai_working_labels"
SENTIMENT140_400K_RUN_DIR = OUT_ROOT / "run_pool400_plus_sentiment140_400k_v1"
SAMPLE_CLASSES = ("Support", "Neutral", "Oppose")


def make_class_targets(total: int, classes: tuple[str, ...] = SAMPLE_CLASSES) -> dict[str, int]:
    """As-equal-as-possible per-class split of `total` across `classes`; any remainder from
    integer division goes one-each to the first classes in order, so the sum is always exact."""
    base, remainder = divmod(total, len(classes))
    return {c: base + (1 if i < remainder else 0) for i, c in enumerate(classes)}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)


def stratified_sample_exact(csv_path: Path, class_targets: dict[str, int], seed: int,
                             overshoot_margin: int = 2_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact-count class-aware stratified sample, memory-safe (one chunked streaming pass).

    Returns (sampled_df, report). report['shortfall'] lists any class whose eligible pool is
    smaller than its target -- the caller must stop and report this, never silently proceed with
    fewer rows than requested.
    """
    overshoot_cap = max(class_targets.values()) + overshoot_margin
    reservoir, seen_report = reservoir_sample_pseudo_labels(csv_path, overshoot_cap, seed)
    shortfall = {}
    pieces = []
    for cls, target in class_targets.items():
        available = seen_report.get(cls, {}).get("seen", 0)
        if available < target:
            shortfall[cls] = {"target": target, "available": available}
            continue
        class_block = reservoir[reservoir["pseudo_label"] == cls]
        sub = class_block.sample(n=target, random_state=seed).reset_index(drop=True)
        pieces.append(sub)
    sampled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=reservoir.columns)
    report = {
        "class_targets": class_targets, "seed": seed, "overshoot_cap": overshoot_cap,
        "available_per_class_eligible_pool": {c: seen_report.get(c, {}).get("seen", 0) for c in class_targets},
        "intermediate_reservoir_size_per_class": {c: seen_report.get(c, {}).get("sampled", 0) for c in class_targets},
        "final_sampled_per_class": sampled["pseudo_label"].value_counts().to_dict() if len(sampled) else {},
        "final_total": len(sampled), "shortfall": shortfall,
    }
    return sampled, report


def check_pairwise_text_overlap(blocks: dict[str, pd.Series]) -> dict[str, int]:
    """Normalised-text overlap counts between every pair of named text series. Pure diagnostic;
    does not modify anything. Returns {'nameA|nameB': n_overlapping_normalised_texts}."""
    normed = {name: set(s.map(normalise_text)) for name, s in blocks.items()}
    names = list(normed)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f"{a}|{b}"] = len(normed[a] & normed[b])
    return out


def dedup_by_normalised_text(df: pd.DataFrame, text_col: str, priority_order: list[str],
                              priority_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop exact (post-normalisation) duplicate text across the WHOLE combined frame, keeping
    the first occurrence in `priority_order` (most-trustworthy source wins). Reports how many
    rows were dropped and from which source."""
    order_rank = {name: i for i, name in enumerate(priority_order)}
    df = df.copy()
    df["_norm"] = df[text_col].map(normalise_text)
    df["_rank"] = df[priority_col].map(order_rank).fillna(len(priority_order))
    df = df.sort_values("_rank", kind="stable")
    before = len(df)
    dup_mask = df.duplicated(subset="_norm", keep="first")
    dropped_by_source = df[dup_mask][priority_col].value_counts().to_dict()
    deduped = df[~dup_mask].drop(columns=["_norm", "_rank"]).reset_index(drop=True)
    report = {"rows_before": before, "rows_after": len(deduped), "rows_dropped": int(dup_mask.sum()),
              "dropped_by_source": dropped_by_source}
    return deduped, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-size", type=int, default=400_000,
                     help="Exact total Sentiment140-train tweets to sample, split as evenly as "
                          "possible across Support/Neutral/Oppose (default: 400,000).")
    a = ap.parse_args(argv)
    run_dir = OUT_ROOT / f"run_{a.tag}"
    if run_dir.exists():
        print(f"[STOP] {run_dir} exists", file=sys.stderr)
        return 2
    if not SENTIMENT140_TRAIN_CSV.is_file():
        print(f"[ERROR] {SENTIMENT140_TRAIN_CSV} not found -- pseudo-labelling must already be complete.", file=sys.stderr)
        return 1

    mem_checkpoints: dict[str, float] = {"start": _rss_mb()}
    cfg = json.loads(json.dumps(sp.DEFAULT_CONFIG))
    cfg["seed"] = a.seed
    timings: dict[str, float] = {}

    # ── 1. locate + verify the exact pseudo-labelled Sentiment140 source file ──
    s140_manifest = json.loads(SENTIMENT140_TRAIN_MANIFEST.read_text(encoding="utf-8"))
    print(f"[SOURCE] {_rel(SENTIMENT140_TRAIN_CSV)}  rows_written={s140_manifest['rows_written']}  "
          f"eligible_for_training={sum(s140_manifest['label_counts_training_eligible'].values())}  "
          f"by_class={s140_manifest['label_counts_training_eligible']}")

    # ── 2. exact N-tweet class-aware stratified sample (memory-safe, chunked) ──
    class_targets = make_class_targets(a.sample_size)
    t0 = time.perf_counter()
    s140_sample, s140_sample_report = stratified_sample_exact(SENTIMENT140_TRAIN_CSV, class_targets, a.seed)
    timings["sentiment140_sampling_seconds"] = round(time.perf_counter() - t0, 2)
    mem_checkpoints["after_sentiment140_sampling"] = _rss_mb()
    if s140_sample_report["shortfall"]:
        print(f"[STOP] insufficient eligible Sentiment140 records for target sample: "
              f"{s140_sample_report['shortfall']}", file=sys.stderr)
        return 3
    print(f"[SAMPLE] Sentiment140 {a.sample_size}-tweet sample: {s140_sample_report['final_total']} rows, "
          f"by class {s140_sample_report['final_sampled_per_class']} "
          f"(elapsed {timings['sentiment140_sampling_seconds']}s, RSS {mem_checkpoints['after_sentiment140_sampling']} MB)")

    # ── 3. pool400 (identical loading + three-way split protocol as every prior run) ──
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
            return 4

    def pool_block(idx: np.ndarray) -> pd.DataFrame:
        sub = d.iloc[idx]
        return pd.DataFrame({
            "original_text": sub["original_text"].values, "gold_stance": sub["gold_stance"].values,
            "label_source": AI_LABEL_SOURCE, "source_dataset": sub["source_dataset"].values,
            "source_record_id": sub["source_record_id"].values,
            "rule": "", "sentiment_prediction": "", "stance_margin_pseudo": np.nan, "sentiment_margin_pseudo": np.nan,
        })

    pool_tr, pool_va, pool_te = pool_block(tr), pool_block(va), pool_block(te)
    X_test, y_test = pool_te.original_text.to_numpy(), pool_te.gold_stance.to_numpy()
    groups_test = groups[te]
    print(f"[POOL400] usable {len(d)} (dropped {dropped_pool}); train {len(pool_tr)} / "
          f"validation {len(pool_va)} / test {len(pool_te)} (test held out, identical seed/protocol project-wide)")

    # ── 4. other eligible sources (corpus in full, Sentiment140-test in full) ──
    corpus_block = load_simple_pseudo_csv(CORPUS_PSEUDO_CSV, "original_text", CORPUS_LABEL_SOURCE)
    s140_test_block = load_simple_pseudo_csv(SENTIMENT140_TEST_CSV, "cleaned_text", S140_LABEL_SOURCE)
    s140_train_block = pd.DataFrame({
        "original_text": s140_sample["cleaned_text"].values, "gold_stance": s140_sample["pseudo_label"].values,
        "label_source": S140_LABEL_SOURCE, "source_dataset": s140_sample["source_dataset"].values,
        "source_record_id": s140_sample["source_record_id"].values, "rule": s140_sample["rule"].values,
        "sentiment_prediction": s140_sample["sentiment_prediction"].values,
        "stance_margin_pseudo": s140_sample["stance_margin"].astype(float).values,
        "sentiment_margin_pseudo": s140_sample["sentiment_margin"].astype(float).values,
    })
    print(f"[SOURCES] corpus_block={len(corpus_block)}  s140_test_block={len(s140_test_block)}  "
          f"s140_train_block={len(s140_train_block)}")

    # ── 5. text-overlap check between TRAINING-CANDIDATE sources, VALIDATION, and TEST -- BEFORE combining/training ──
    overlap = check_pairwise_text_overlap({
        "pool_train": pool_tr.original_text, "pool_validation": pool_va.original_text,
        "pool_test": pool_te.original_text, "corpus_pseudo": corpus_block.original_text,
        "sentiment140_test_pseudo": s140_test_block.original_text,
        "sentiment140_train_400k_sample": s140_train_block.original_text,
    })
    print("[OVERLAP CHECK] pairwise normalised-text overlaps (expect 0 for every pair touching "
          f"pool_validation or pool_test): {overlap}")
    test_or_val_leak_pairs = {k: v for k, v in overlap.items()
                               if ("pool_test" in k or "pool_validation" in k) and v > 0
                               and not (k == "pool_train|pool_validation" or k == "pool_train|pool_test"
                                        or k == "pool_validation|pool_test")}
    if test_or_val_leak_pairs:
        print(f"[WARNING] unexpected overlap between a training-candidate source and validation/test: "
              f"{test_or_val_leak_pairs} -- excluding overlapping rows from training.", file=sys.stderr)

    # ── 6. combine (train + validation, per project convention when no new hyperparameter search is run) ──
    train_final = pd.concat([pool_tr, pool_va, corpus_block, s140_test_block, s140_train_block], ignore_index=True)
    train_final["_source_group"] = (["pool400"] * (len(pool_tr) + len(pool_va)) + ["corpus"] * len(corpus_block)
                                     + ["sentiment140_test"] * len(s140_test_block)
                                     + ["sentiment140_train_400k"] * len(s140_train_block))

    # ── 7. deduplicate by normalised text across the FULL combined set (pool400 wins ties, then corpus, then s140) ──
    train_final, dedup_report = dedup_by_normalised_text(
        train_final, "original_text", ["pool400", "corpus", "sentiment140_test", "sentiment140_train_400k"], "_source_group"
    )
    print(f"[DEDUP] combined set before={dedup_report['rows_before']} after={dedup_report['rows_after']} "
          f"dropped={dedup_report['rows_dropped']} by_source={dedup_report['dropped_by_source']}")

    train_class_dist = train_final.gold_stance.value_counts().to_dict()
    train_source_dist = train_final.label_source.value_counts().to_dict()
    print(f"[COMBINED TRAIN SET] n={len(train_final)} by_class={train_class_dist} by_source={train_source_dist}")
    mem_checkpoints["after_combine_and_dedup"] = _rss_mb()

    X_train, y_train = train_final.original_text.to_numpy(), train_final.gold_stance.to_numpy()

    # ── 8. train LinearSVM + Logistic Regression (reused sp.build_pipeline: sparse TF-IDF, CPU) ──
    pipelines: dict[str, Any] = {}
    fit_timings: dict[str, float] = {}
    for m in sp.MODELS:
        t0 = time.perf_counter()
        pipe = sp.build_pipeline(m, cfg).fit(X_train, y_train)
        fit_timings[m] = round(time.perf_counter() - t0, 2)
        pipelines[m] = pipe
        mem_checkpoints[f"after_fit_{m}"] = _rss_mb()
        print(f"[TRAIN] {m}: fit in {fit_timings[m]}s, vocab_size={len(pipe.named_steps['tfidf'].vocabulary_)}, "
              f"RSS={mem_checkpoints[f'after_fit_{m}']} MB")

    # ── 9. evaluate vs original baseline on the SAME held-out test set ──
    new_metrics: dict[str, Any] = {}
    original_metrics: dict[str, Any] = {}
    for m in sp.MODELS:
        pred = pipelines[m].predict(X_test)
        new_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
            "model_classes_": sorted(pipelines[m].named_steps["clf"].classes_.tolist()),
        }
        original_pipe = load_trained_pipeline(ORIGINAL_RUN_DIR, m)
        orig_pred = original_pipe.predict(X_test)
        original_metrics[m] = {
            "test_vs_pool_labels": sp.evaluate(y_test, orig_pred, pool_labels, groups_test, cfg["bootstrap"], a.seed),
            "test_predicted_distribution_all_4_labels": pd.Series(orig_pred).value_counts().reindex(ALL_FOUR_LABELS, fill_value=0).to_dict(),
        }
        print(f"[RESULT] {m}: macro_f1={new_metrics[m]['test_vs_pool_labels']['macro_f1']} "
              f"(original={original_metrics[m]['test_vs_pool_labels']['macro_f1']}) "
              f"Oppose-predictions-on-test={new_metrics[m]['test_predicted_distribution_all_4_labels'].get('Oppose', 0)}")

    # ── 10. save (write-once), new versioned directory ──
    run_dir.mkdir(parents=True)
    import joblib
    arts: dict[str, str] = {}
    for m, pipe in pipelines.items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)

    # verify the SAVED (reloaded from disk) models can actually predict -- real smoke test, not just reusing the in-memory object
    smoke_test: dict[str, Any] = {}
    sample_texts = X_test[:3].tolist()
    for m in sp.MODELS:
        reloaded = joblib.load(run_dir / f"pipeline_{m}.joblib")
        preds = reloaded.predict(sample_texts).tolist()
        smoke_test[m] = {"sample_texts_n": len(sample_texts), "predictions": preds, "status": "ok"}
    print(f"[SMOKE TEST] reloaded-from-disk predictions: {smoke_test}")

    disclaimer = (
        lb.AI_DISCLAIMER + " ADDITIONALLY: training data mixes pool400 AI-working-labels with "
        "pseudo-labels from THREE sources -- the in-domain consultation corpus remainder "
        f"(self-training risk), ALL Sentiment140-test pseudo-labels, and an exact {a.sample_size:,}-tweet "
        "class-aware stratified sample of the Sentiment140-train pseudo-labels (seed "
        f"{a.seed}, out-of-domain, heuristic Oppose rule with a known high in-domain false-positive "
        "rate). Oppose is UNVERIFIED everywhere in this run; the held-out test set has only 1 "
        "genuine Oppose example project-wide and cannot evaluate it."
    )
    unclear_handling_note = (
        f"Unclear training examples: {int((pool_tr.gold_stance=='Unclear').sum() + (pool_va.gold_stance=='Unclear').sum())} "
        "(from pool400 train+validation only; 0 from any pseudo-label source -- the pseudo-labeling "
        "rule in models/pseudo_labeling.py always excludes Unclear from the training-eligible pseudo pool)."
    )
    mem_checkpoints["end"] = _rss_mb()
    metrics = {
        "disclaimer": disclaimer, "unclear_handling": unclear_handling_note,
        "source_file_located": {
            "path": _rel(SENTIMENT140_TRAIN_CSV), "manifest": _rel(SENTIMENT140_TRAIN_MANIFEST),
            "rows_written": s140_manifest["rows_written"],
            "eligible_for_training": sum(s140_manifest["label_counts_training_eligible"].values()),
            "eligible_by_class": s140_manifest["label_counts_training_eligible"],
        },
        "sentiment140_400k_sample": s140_sample_report,
        "pool400": {"usable": len(d), "dropped": dropped_pool, "labels_evaluated": pool_labels,
                    "n_train": len(pool_tr), "n_validation": len(pool_va), "n_test_held_out": len(pool_te),
                    "corpus_group_mapping": corpus_mapping},
        "sources": {
            "corpus_indomain_all": {"n": len(corpus_block), "by_class": corpus_block.gold_stance.value_counts().to_dict()},
            "sentiment140_test_all": {"n": len(s140_test_block), "by_class": s140_test_block.gold_stance.value_counts().to_dict()},
            "sentiment140_train_400k_sample": {"n": len(s140_train_block), "by_class": s140_train_block.gold_stance.value_counts().to_dict()},
        },
        "pairwise_text_overlap_check_before_training": overlap,
        "deduplication": dedup_report,
        "combined_training_set": {"n": len(train_final), "by_class": train_class_dist, "by_source": train_source_dist},
        "held_out_test_set": {"n": len(pool_te), "class_distribution": pd.Series(y_test).value_counts().to_dict(),
                              "note": "Identical pool400 test split used project-wide (same seed/protocol); never used for threshold/hyperparameter tuning; checked for 0 text overlap with every training source before training (see pairwise_text_overlap_check_before_training)."},
        "model_1_original_baseline": original_metrics,
        "model_2_this_run": new_metrics,
        "model_3_sentiment140_400k_run": (
            {m: json.loads((SENTIMENT140_400K_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
                ["model_2_this_run"][m]["test_vs_pool_labels"] for m in sp.MODELS}
            if SENTIMENT140_400K_RUN_DIR.is_dir() else "not found -- comparison skipped"
        ),
        "comparison_note": ("Both models evaluated on the IDENTICAL 80-record held-out pool400 test "
                            "split used by every run in this project. macro_f1 computed over classes "
                            "present in the test set (Support/Neutral/Unclear). "
                            "test_predicted_distribution_all_4_labels shows Oppose-firing behavior "
                            "descriptively -- not a scored metric (no genuine Oppose ground truth in "
                            "the test set)."),
        "timings_seconds": {**timings, "fit": fit_timings},
        "memory_rss_mb_checkpoints": mem_checkpoints,
        "memory_rss_mb_peak_observed": max(mem_checkpoints.values()),
        "smoke_test_reloaded_models_predict": smoke_test,
    }
    files = {
        "metrics.json": metrics,
        "preprocessing_config.json": {"base": cfg, "note": "same TF-IDF/model config as run_pool400_ai_v1_3class"},
        "label_mapping.json": {"labels": pool_labels, "all_model_output_labels": ALL_FOUR_LABELS, "schema": lb.SCHEMA_LABELS,
                               "construct": "stance toward the Open Government initiative/policy (guidelines v1.1); "
                                            "four-class output framework preserved (Support/Neutral/Oppose/Unclear)",
                               "unclear_handling": unclear_handling_note},
    }
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)

    # the selected N-tweet sample, saved on its own (deliverable #8)
    sample_filename = f"sentiment140_{a.sample_size}_sample.csv"
    s140_train_block.assign(split="train").to_csv(run_dir / sample_filename, index=False, lineterminator="\n")
    arts[sample_filename] = _sha(run_dir / sample_filename)

    train_final.drop(columns=["_source_group"], errors="ignore").assign(split="train_or_validation").to_csv(
        run_dir / "combined_training_data_provenance.csv", index=False, lineterminator="\n"
    )
    arts["combined_training_data_provenance.csv"] = _sha(run_dir / "combined_training_data_provenance.csv")

    pool_te[["source_dataset", "source_record_id", "gold_stance", "original_text"]].assign(split="test").rename(
        columns={"gold_stance": "ai_stance"}
    ).to_csv(run_dir / "held_out_test_set.csv", index=False, lineterminator="\n")
    arts["held_out_test_set.csv"] = _sha(run_dir / "held_out_test_set.csv")

    meta = {
        "status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_stance_sentiment140_400k_v1.py", "argv": sys.argv[1:] if argv is None else argv,
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
            "data/stance_training_ai_working_labels/run_pool400_plus_filtered_s140_corpus_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_no_oppose_v1",
            "data/stance_training_ai_working_labels/run_combined_dataset_retrain_v1",
            "data/stance_training_ai_working_labels/run_pool400_plus_sentiment140_400k_v1",
            "indian/ (Lok Sabha not used)",
        ],
        "no_new_pseudo_labeling_performed": True, "no_new_datasets_downloaded": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    summary = {
        "test_n": len(pool_te), "combined_train_n": len(train_final),
        "linsvm_macro_f1": new_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "logreg_macro_f1": new_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "original_linsvm_macro_f1": original_metrics["linsvm"]["test_vs_pool_labels"]["macro_f1"],
        "original_logreg_macro_f1": original_metrics["logreg"]["test_vs_pool_labels"]["macro_f1"],
        "peak_rss_mb": metrics["memory_rss_mb_peak_observed"],
    }
    if isinstance(metrics["model_3_sentiment140_400k_run"], dict):
        summary["sentiment140_400k_run_linsvm_macro_f1"] = metrics["model_3_sentiment140_400k_run"]["linsvm"]["macro_f1"]
        summary["sentiment140_400k_run_logreg_macro_f1"] = metrics["model_3_sentiment140_400k_run"]["logreg"]["macro_f1"]
    print(json.dumps(summary, default=str))
    print(f"[TRAINED] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
