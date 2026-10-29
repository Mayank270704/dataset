"""
scripts/run_sentiment_domain_check_v1.py
========================================
Inference-only check of the two existing SOCIAL-MEDIA sentiment baselines on Canadian Open Government consultation
text. Nothing is (re)trained here. Sentiment is tone; it is never used as stance.

Models (read-only, hash-checked against their manifests)
    data/sentiment_training/sentiment140_hashing_sgd_v1    English tweets, binary negative/positive
    data/sentiment_training/sentimix_hinglish_baseline_v1  Hindi-English code-mixed tweets, negative/neutral/positive

Samples
    A  full consultation corpus (data/full_corpus/run_v1/corpus/corpus_entries.csv, 10,108 entries): prediction
       distribution, confidence, language effects, failure indicators.
    B  400-record v2 pool with AI-generated sentiment working labels (data/ai_labels/stance_sentiment_ai_v1): agreement
       with AI labels (NOT human-verified accuracy), plus stance-vs-sentiment cross-tab showing they differ.

    python scripts/run_sentiment_domain_check_v1.py
Output: data/sentiment_evaluation/consultation_domain_check_v1/ (write-once)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

from models.sentiment_analysis.sentimix_baseline import SentiMixClassifier  # noqa: E402
from models.sentiment_analysis.streaming_baseline import Sentiment140StreamingClassifier  # noqa: E402

OUT = ROOT / "data" / "sentiment_evaluation" / "consultation_domain_check_v1"
S140 = ROOT / "data" / "sentiment_training" / "sentiment140_hashing_sgd_v1"
SMIX = ROOT / "data" / "sentiment_training" / "sentimix_hinglish_baseline_v1"
CORPUS = ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv"
AI_SENT = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_sentiment_labels_pool400.csv"
AI_STANCE = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1" / "ai_working_labels_pool400.csv"
LOW_CONF = 0.6


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def verify_model(d: Path) -> str:
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    rec = man.get("outputs_sha256", {})
    want = next((v for k, v in rec.items() if k.endswith("model.joblib")), None)
    got = sha(d / "model.joblib")
    if want is not None and want != got:
        raise SystemExit(f"{d.name}/model.joblib does not match its manifest; refusing to load")
    return "verified" if want else "no recorded hash"


def predict_all(texts: pd.Series, s140, smix) -> pd.DataFrame:
    p1 = s140.predict_proba(texts)
    p3 = smix.predict_proba(texts)
    out = pd.DataFrame({"s140_label": np.where(p1[:, 1] >= 0.5, "positive", "negative"),
                        "s140_conf": p1.max(axis=1).round(4),
                        "smix_label": np.array(smix.target_names_)[p3.argmax(axis=1)] if hasattr(smix, "target_names_") else smix.predict(texts),
                        "smix_conf": p3.max(axis=1).round(4)})
    return out


def dist(s: pd.Series) -> dict:
    return {k: int(v) for k, v in s.value_counts().items()}


def prf(y, p, labels) -> dict:
    pr, rc, f, n = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"macro_f1": round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4),
            "accuracy": round(float((np.asarray(y) == np.asarray(p)).mean()), 4), "n": int(len(y)),
            "per_class": {l: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f[i]), 4), "support": int(n[i])}
                          for i, l in enumerate(labels)},
            "confusion_matrix": {"labels": labels, "rows_true_cols_pred": confusion_matrix(y, p, labels=labels).tolist()}}


def main() -> int:
    if OUT.exists():
        print(f"[STOP] {OUT} exists", file=sys.stderr)
        return 2
    model_check = {"sentiment140": verify_model(S140), "sentimix": verify_model(SMIX)}
    s140 = Sentiment140StreamingClassifier.load(S140 / "model.joblib")
    smix = SentiMixClassifier.load(SMIX / "model.joblib")

    # ── A: full corpus ──
    c = pd.read_csv(CORPUS, dtype=str, keep_default_na=False)
    pa = pd.concat([c[["corpus_id", "unique_id", "source_dataset", "question_slug", "language"]], predict_all(c.original_text, s140, smix)], axis=1)
    ntok = c.original_text.str.split().str.len()
    pa["n_tokens"] = ntok
    lang = pa.language.where(pa.language.isin(["English", "French"]), "other")
    A = {"entries": int(len(pa)),
         "sentiment140": {"distribution": dist(pa.s140_label),
                          "by_language": {l: dist(g.s140_label) for l, g in pa.groupby(lang)},
                          "confidence_quantiles": pa.s140_conf.quantile([.1, .25, .5, .75, .9]).round(4).to_dict(),
                          "share_conf_below_0.6": round(float((pa.s140_conf < LOW_CONF).mean()), 4),
                          "share_conf_below_0.6_by_language": {l: round(float((g.s140_conf < LOW_CONF).mean()), 4) for l, g in pa.groupby(lang)},
                          "note": "binary model: every text is forced to negative or positive; it has no neutral class"},
         "sentimix": {"distribution": dist(pa.smix_label),
                      "by_language": {l: dist(g.smix_label) for l, g in pa.groupby(lang)},
                      "confidence_quantiles": pa.smix_conf.quantile([.1, .25, .5, .75, .9]).round(4).to_dict(),
                      "share_conf_below_0.6": round(float((pa.smix_conf < LOW_CONF).mean()), 4)},
         "cross_model_agreement_on_polar": None,
         "failure_indicators": {"empty_or_whitespace": int((c.original_text.str.strip() == "").sum()),
                                "texts_le_2_tokens": int((ntok <= 2).sum()),
                                "texts_gt_200_tokens": int((ntok > 200).sum()),
                                "non_english_entries": int((pa.language != "English").sum())},
         "by_source_dataset_s140_positive_share": {k: round(float((g.s140_label == "positive").mean()), 4) for k, g in pa.groupby("source_dataset")}}
    polar = pa[pa.smix_label != "neutral"]
    A["cross_model_agreement_on_polar"] = {"n": int(len(polar)), "agree": round(float((polar.smix_label == polar.s140_label).mean()), 4)}

    # ── B: AI-labelled pool ──
    ai = pd.read_csv(AI_SENT, dtype=str, keep_default_na=False)
    st = pd.read_csv(AI_STANCE, dtype=str, keep_default_na=False)
    assert ai.pilot_id.tolist() == st.pilot_id.tolist()
    pb = pd.concat([ai[["pilot_id", "ai_sentiment", "source_language"]], st[["stance"]].rename(columns={"stance": "ai_stance"}),
                    predict_all(ai.original_text, s140, smix)], axis=1)
    pol = pb[pb.ai_sentiment.isin(["Positive", "Negative"])]
    tri = pb[pb.ai_sentiment.isin(["Positive", "Negative", "Neutral"])]
    B = {"label_status": "AI-generated sentiment working labels; agreement figures are NOT human-verified accuracy",
         "ai_label_distribution": dist(pb.ai_sentiment),
         "sentiment140_vs_ai_on_polar_records": prf(pol.ai_sentiment.str.lower(), pol.s140_label, ["negative", "positive"]),
         "sentiment140_predictions_on_ai_neutral": dist(pb.loc[pb.ai_sentiment == "Neutral", "s140_label"]),
         "sentimix_vs_ai_3class": prf(tri.ai_sentiment.str.lower(), tri.smix_label, ["negative", "neutral", "positive"]),
         "sentimix_vs_ai_by_language": {l: prf(g.ai_sentiment.str.lower(), g.smix_label, ["negative", "neutral", "positive"])["macro_f1"]
                                        for l, g in tri.groupby("source_language")},
         "ai_stance_by_ai_sentiment": pd.crosstab(pb.ai_stance, pb.ai_sentiment).to_dict(),
         "ai_negative_sentiment_with_support_stance": int(((pb.ai_sentiment == "Negative") & (pb.ai_stance == "Support")).sum())}

    OUT.mkdir(parents=True)
    pa.to_csv(OUT / "corpus_predictions.csv", index=False, lineterminator="\n")
    pb.to_csv(OUT / "pool400_predictions_vs_ai_labels.csv", index=False, lineterminator="\n")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "construct": "sentiment (tone); not stance",
           "models": {"sentiment140_hashing_sgd_v1": {"trained_on": "Sentiment140 English tweets (binary)", "manifest_hash_check": model_check["sentiment140"],
                                                      "in_domain_test_macro_f1": 0.7911, "in_domain_note": "tweet test set; NOT consultation accuracy",
                                                      "licence": "no licence file in workspace; Sentiment140 terms unverified (academic dataset, Go et al. 2009)"},
                      "sentimix_hinglish_baseline_v1": {"trained_on": "SemEval-2020 SentiMix Hindi-English code-mixed tweets (3-class)", "manifest_hash_check": model_check["sentimix"],
                                                        "in_domain_test_macro_f1": 0.6614, "in_domain_note": "Hinglish tweet test set; NOT consultation accuracy",
                                                        "licence": "no licence file in the local SentiMix copy; terms unverified",
                                                        "language_compatibility": "trained on romanised Hindi-English; consultation corpus is English/French. Language mismatch."}},
           "dependencies": {"sklearn": __import__("sklearn").__version__, "pandas": pd.__version__, "numpy": np.__version__},
           "A_full_corpus": A, "B_pool400_vs_ai_labels": B,
           "inputs_sha256": {rel(p): sha(p) for p in (CORPUS, AI_SENT, AI_STANCE, S140 / "model.joblib", SMIX / "model.joblib")}}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    man = {"outputs_sha256": {f.name: sha(f) for f in sorted(OUT.iterdir())}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"A_s140": A["sentiment140"]["distribution"], "A_smix": A["sentimix"]["distribution"],
                      "A_lowconf": (A["sentiment140"]["share_conf_below_0.6"], A["sentimix"]["share_conf_below_0.6"]),
                      "B_s140_polar": B["sentiment140_vs_ai_on_polar_records"]["macro_f1"], "B_s140_on_neutral": B["sentiment140_predictions_on_ai_neutral"],
                      "B_smix": B["sentimix_vs_ai_3class"]["macro_f1"], "B_smix_lang": B["sentimix_vs_ai_by_language"],
                      "neg_support": B["ai_negative_sentiment_with_support_stance"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
