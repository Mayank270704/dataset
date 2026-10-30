"""
indian/scripts/run_tweets_analysis_v1.py
========================================
Kaggle "Dataset of Indian Politics Tweets & Reactions" (social-media dataset, NOT consultation data).

    python -I indian/scripts/run_tweets_analysis_v1.py stats        # validation, quality profile, label check
    python -I indian/scripts/run_tweets_analysis_v1.py annotation   # blank annotation templates + sampling plan
    python -I indian/scripts/run_tweets_analysis_v1.py weak         # weak-label experiment (Sentiment140 model)

Likes/retweets are engagement counts, never sentiment labels. Weak-model predictions are stored separately and are
never ground truth. Author IDs stay pseudonymous. Outputs: indian/processed/tweets/{analysis_v1, annotation_v1,
weak_labels_sentiment140_v1}/ (write-once).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "indian"
SRC = IND / "processed" / "kaggle_indian_politics_tweets_clean.csv"
ORIG = IND / "kaggle" / "Dataset of Indian Politics Tweets & Reactions" / "tweets.csv"
OUT = IND / "processed" / "tweets"
S140 = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1"
LICENCE = "unverified: no Kaggle licence/dataset card locally; X/Twitter content terms apply; local research only"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    d["likes_n"] = pd.to_numeric(d.likes, errors="coerce")
    d["retweets_n"] = pd.to_numeric(d.retweets, errors="coerce")
    d["ts"] = pd.to_datetime(d.created_at_utc, utc=True, errors="coerce")
    d["month"] = d.ts.dt.strftime("%Y-%m")
    return d


def newdir(p: Path) -> bool:
    if p.exists():
        print(f"[SKIP] {p.relative_to(ROOT)} exists")
        return False
    p.mkdir(parents=True)
    return True


def stats() -> int:
    d = load()
    orig_header = pd.read_csv(ORIG, nrows=0).columns.tolist()
    qs = [0, .25, .5, .75, .9, .99, 1]
    per_author = d.author_pseudo_id.value_counts()
    res = {
        "rows": len(d), "unique_tweet_row_ids": int(d.tweet_row_id.nunique()), "columns": [c for c in d.columns if c not in ("likes_n", "retweets_n", "ts", "month")],
        "original_columns": orig_header,
        "label_check": {"sentiment_or_stance_columns_in_original": [c for c in orig_header if any(k in c.lower() for k in ("sentiment", "label", "stance", "polarity", "emotion"))],
                        "conclusion": "no genuine sentiment/stance labels; 'Reactions' = Likes and Retweets (engagement counts)"},
        "missing_by_column": {k: int(v) for k, v in (d.drop(columns=["likes_n", "retweets_n", "ts", "month"]) == "").sum().items() if v},
        "unparseable_dates": int(d.ts.isna().sum()), "unparseable_likes": int(d.likes_n.isna().sum()), "unparseable_retweets": int(d.retweets_n.isna().sum()),
        "date_range": [str(d.ts.min()), str(d.ts.max())], "tweets_per_month": d.month.value_counts().sort_index().to_dict(),
        "char_count": {str(k): float(v) for k, v in d.char_count.astype(int).quantile(qs).items()},
        "duplicate_text_groups": int(d.exact_text_duplicate_group.replace("", np.nan).nunique()),
        "tweets_in_duplicate_groups": int((d.exact_text_duplicate_group != "").sum()),
        "largest_duplicate_group": int(d[d.exact_text_duplicate_group != ""].exact_text_duplicate_group.value_counts().max()),
        "engagement": {"likes_quantiles": {str(k): float(v) for k, v in d.likes_n.quantile(qs).items()},
                       "retweets_quantiles": {str(k): float(v) for k, v in d.retweets_n.quantile(qs).items()},
                       "share_zero_likes": round(float((d.likes_n == 0).mean()), 4), "share_zero_retweets": round(float((d.retweets_n == 0).mean()), 4),
                       "note": "engagement counts are not sentiment labels"},
        "authors": {"distinct_pseudonymous": int(per_author.size), "tweets_per_author_quantiles": {str(k): float(v) for k, v in per_author.quantile(qs).items()},
                    "top_author_share": round(float(per_author.iloc[0] / len(d)), 4), "authors_with_ge_50_tweets": int((per_author >= 50).sum())},
        "script_detected": d.script_detected.value_counts().to_dict(),
        "share_with_url": round(float((d.has_url == "True").mean()), 4), "mention_count_median": float(d.mention_count.astype(int).median()),
        "share_containing_phrase_indian_politics": round(float(d.tweet_text.str.contains("indian politics", case=False).mean()), 4),
        "repaired_line_break_records": int((d.repaired_line_break == "True").sum()),
        "licence_status": LICENCE,
    }
    a = OUT / "analysis_v1"
    if newdir(a):
        (a / "quality_profile.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("rows", "unique_tweet_row_ids", "missing_by_column", "date_range", "duplicate_text_groups", "tweets_in_duplicate_groups", "label_check")}, default=str))
    print(json.dumps(res["engagement"]), json.dumps(res["authors"]))
    return 0


def annotation() -> int:
    t = OUT / "annotation_v1"
    if not newdir(t):
        return 0
    d = load()
    rng = np.random.default_rng(42)
    # one candidate per duplicate-text group; at most 2 tweets per author; stratified by month
    cand = d[(d.exact_text_duplicate_group == "") | ~d.duplicated("exact_text_duplicate_group")]
    cand = cand.iloc[rng.permutation(len(cand))]
    cand = cand[cand.groupby("author_pseudo_id").cumcount() < 2]
    months = sorted(cand.month.dropna().unique())
    per_month = int(np.ceil(1500 / len(months)))
    study = pd.concat([g.head(per_month) for _, g in cand.groupby("month")]).head(1500)
    study = study.iloc[rng.permutation(len(study))].reset_index(drop=True)
    study["study_split"] = np.where(np.arange(len(study)) < 300, "locked_test_300", "train_dev")
    pilot_ids = set(study[study.study_split == "train_dev"].tweet_row_id.head(200))
    study["annotation_batch"] = np.where(study.tweet_row_id.isin(pilot_ids), "pilot_200", np.where(study.study_split == "locked_test_300", "test", "main"))
    plan = study[["tweet_row_id", "created_at_utc", "month", "author_pseudo_id", "script_detected", "exact_text_duplicate_group", "study_split", "annotation_batch"]]
    plan.to_csv(t / "sampling_plan_1500.csv", index=False, lineterminator="\n")
    tmpl = study[study.annotation_batch == "pilot_200"][["tweet_row_id", "created_at_utc", "author_pseudo_id", "tweet_text_masked", "script_detected"]].assign(
        sentiment="", target_entity="", sarcasm_or_irony="", language_note="", review_notes="", annotator_id="")
    for k in ("annotator_A", "annotator_B"):
        tmpl.to_csv(t / f"tweets_pilot_annotation_template_v1_{k}.csv", index=False, lineterminator="\n")
    man = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": 42, "study_size": int(len(study)),
           "locked_test": int((study.study_split == "locked_test_300").sum()), "pilot": int(len(tmpl)),
           "rules": ["one tweet per exact-text duplicate group", "at most 2 tweets per pseudonymous author", f"<= {per_month} per month",
                     "test split fixed now and never used for development", "pilot drawn from train_dev only", "texts shown with @mentions and URLs masked"],
           "per_month": study.month.value_counts().sort_index().to_dict(), "source_sha256": sha(SRC), "licence_status": LICENCE,
           "outputs_sha256": {f.name: sha(f) for f in sorted(t.iterdir())}}
    (t / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("study_size", "locked_test", "pilot", "per_month")}))
    return 0


def weak() -> int:
    w = OUT / "weak_labels_sentiment140_v1"
    if w.exists():
        print("[SKIP] exists")
        return 0
    sys.path.insert(0, str(ROOT))
    from models.sentiment_analysis.streaming_baseline import Sentiment140StreamingClassifier
    man = json.loads((S140 / "manifest.json").read_text(encoding="utf-8"))
    want = man.get("outputs_sha256", {}).get("model.joblib")
    if want != sha(S140 / "model.joblib"):
        print("[STOP] Sentiment140 model hash does not match its manifest", file=sys.stderr)
        return 1
    t0 = time.time()
    clf = Sentiment140StreamingClassifier.load(S140 / "model.joblib")
    d = load()
    p = clf.predict_proba(d.tweet_text.tolist())[:, 1]
    conf = np.maximum(p, 1 - p)
    out = pd.DataFrame({"tweet_row_id": d.tweet_row_id, "weak_label": np.where(p >= 0.5, "positive", "negative"),
                        "prob_positive": p.round(4), "confidence": conf.round(4),
                        "label_source": "weak_model_prediction_NOT_ground_truth:sentiment140_hashing_sgd_v1"})
    w.mkdir(parents=True)
    out.to_csv(w / "weak_predictions.csv", index=False, lineterminator="\n")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": "data/sentiment_training/sentiment140_hashing_sgd_v1/model.joblib (HashingVectorizer + SGD log-loss; Sentiment140 English tweets, 2009; binary)",
           "model_sha256_verified_against_manifest": True, "model_in_domain_test_macro_f1": 0.7911,
           "status": "WEAK LABELS ONLY: unvalidated on Indian political tweets; no human labels exist to measure accuracy",
           "limitations": ["binary: forces every tweet to negative/positive (no neutral class)", "trained on 2009 US-centric tweets; political/Indian vocabulary and Hinglish unseen",
                           "earlier PolicyLens check: this model pushed neutral consultation text toward positive", "cannot detect sarcasm",
                           "agreement must be measured on the locked human test set before any use"],
           "rows": len(out), "distribution": out.weak_label.value_counts().to_dict(),
           "confidence_quantiles": {str(k): float(v) for k, v in out.confidence.quantile([.1, .25, .5, .75, .9]).items()},
           "share_confidence_below_0.6": round(float((conf < 0.6).mean()), 4),
           "positive_share_by_script": {k: round(float((out.weak_label[d.script_detected == k] == "positive").mean()), 4) for k in d.script_detected.unique()},
           "positive_share_by_month": {k: round(float((out.weak_label[d.month == k] == "positive").mean()), 4) for k in sorted(d.month.dropna().unique())},
           "runtime_s": round(time.time() - t0, 1), "outputs_sha256": {"weak_predictions.csv": sha(w / "weak_predictions.csv")}, "licence_status": LICENCE}
    (w / "summary.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("rows", "distribution", "confidence_quantiles", "share_confidence_below_0.6", "positive_share_by_script", "runtime_s")}))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit({"stats": stats, "annotation": annotation, "weak": weak}[sys.argv[1]]())
