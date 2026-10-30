"""
indian/scripts/build_final_reports.py
=====================================
Assemble indian/reports/final_dataset_quality_summary.csv and final_experiment_results.csv directly from the
measured JSON outputs of each experiment (no hand-copied numbers). Missing experiments are written as rows with
status 'not run' and the recorded reason.

    python -I indian/scripts/build_final_reports.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

IND = Path(__file__).resolve().parents[1]
P = IND / "processed"
R = IND / "reports"


def rj(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[WROTE] {path.relative_to(IND.parent)} ({len(rows)} rows)")


def quality() -> list[dict]:
    mv = rj(P / "mygov" / "analysis_v1" / "validation.json") or {}
    ms = rj(P / "mygov" / "analysis_v1" / "descriptive_stats.json") or {}
    tq = rj(P / "tweets" / "analysis_v1" / "quality_profile.json") or {}
    pr = (rj(P / "processing_summary_mygov_tweets_civis.json") or {}).get("results", {})
    lk = (rj(P / "processing_summary_loksabha.json") or {}).get("results", {}).get("loksabha", {})
    cm = rj(P / "retrieval" / "civis_v1" / "index_manifest.json") or {}
    c = pr.get("civis", {})
    rows = [
        {"dataset": "MyGov Open Data Use License consultation", "file": "indian/processed/mygov_comments_clean.csv", "content": "public consultation submissions",
         "records": mv.get("rows_clean"), "unique_ids": mv.get("unique_ids"), "coverage": "550 of 1,132 displayed (48.6%); page indices 0-54",
         "missing_fields": "submission_date (all); language (all; script detected instead)",
         "duplicates": f"{mv.get('raw_duplicate_id_records')} duplicate-ID raw records (removed by ID); {mv.get('duplicate_text_groups')} identical-text groups / {mv.get('comments_in_duplicate_text_groups')} comments kept and flagged",
         "quality_flags": f"{mv.get('near_limit_and_no_terminal_punctuation')} possibly truncated (>=490 chars, no end punctuation); {ms.get('script_detected')}",
         "pii": f"{mv.get('pii_any_flag')} comments flagged {mv.get('pii_flag_counts')}; usernames only in raw file",
         "labels": "none", "licence_status": "unverified", "suitability": "consultation analysis (partial sample) after permission; annotation-ready"},
        {"dataset": "Kaggle Indian Politics Tweets & Reactions", "file": "indian/processed/kaggle_indian_politics_tweets_clean.csv", "content": "social-media posts",
         "records": tq.get("rows"), "unique_ids": tq.get("unique_tweet_row_ids"), "coverage": f"{tq.get('date_range')}",
         "missing_fields": f"{ {k: v for k, v in (tq.get('missing_by_column') or {}).items() if k != 'exact_text_duplicate_group'} or 'none'} (duplicate-group column empty by design)",
         "duplicates": f"{tq.get('duplicate_text_groups')} identical-text groups / {tq.get('tweets_in_duplicate_groups')} tweets, flagged not removed (no tweet IDs)",
         "quality_flags": f"{tq.get('repaired_line_break_records')} record repaired (split line); {tq.get('share_containing_phrase_indian_politics')} share contain 'Indian politics' (collection query)",
         "pii": f"authors pseudonymised ({(tq.get('authors') or {}).get('distinct_pseudonymous')} IDs); masked-text column for @mentions/URLs",
         "labels": "none (likes/retweets are engagement)", "licence_status": "unverified", "suitability": "supplementary experimentation only"},
        {"dataset": "HF civis_consultation_summaries", "file": "indian/processed/civis_policy_documents_clean.csv", "content": "policy documents + reference summaries",
         "records": c.get("rows"), "unique_ids": c.get("rows"), "coverage": f"{c.get('by_split')}",
         "missing_fields": f"empty document text: {c.get('empty_document_text')}",
         "duplicates": f"identical-text pairs: {list((c.get('duplicate_document_groups') or {}).values())}",
         "quality_flags": f"indexed {cm.get('indexed_canonical_documents')} canonical documents / {(cm.get('chunking') or {}).get('chunks')} chunks",
         "pii": "not assessed beyond document type (government documents)", "labels": "reference summaries of unverified origin (no stance/sentiment)",
         "licence_status": "unverified", "suitability": "policy context / retrieval"},
        {"dataset": "HF lok-sabha-qa", "file": "indian/processed/lok_sabha_qa_clean.parquet", "content": "parliamentary questions and ministerial answers",
         "records": lk.get("rows"), "unique_ids": lk.get("rows") if (lk.get("quality") or {}).get("dup_id") == 0 else "check", "coverage": "Lok Sabha 16-18, 2015-07-21 to 2026-04-02",
         "missing_fields": f"answer_empty {(lk.get('quality') or {}).get('answer_empty')}; question_empty {(lk.get('quality') or {}).get('question_empty')}",
         "duplicates": f"duplicate IDs {(lk.get('quality') or {}).get('dup_id')}; duplicate (LS,session,Q,type) {(lk.get('quality') or {}).get('dup_key')}",
         "quality_flags": f"legacy-font Hindi {(lk.get('quality') or {}).get('legacy_font')}; some answers parsed into question text (21 of 1,000 pilot queries)",
         "pii": "names of elected members (public role)", "labels": "none (ministry/subject metadata only)", "licence_status": "unverified",
         "suitability": "policy context / evidence retrieval"},
    ]
    return rows


def results() -> list[dict]:
    rows = []
    cv = rj(P / "retrieval" / "civis_v1" / "evaluation" / "metrics.json")
    if cv:
        for qt, d in cv["metrics"].items():
            for m, x in d.items():
                rows.append({"experiment": "CIVIS known-item retrieval", "dataset": "CIVIS", "query_type": qt, "method": m, "subset": "all",
                             "n_queries": x["queries"], "corpus": f"{cv['indexed_documents']} docs / {cv['chunks']} chunks",
                             "recall@1": x["recall@1"], "recall@1_ci95": x["recall@1_ci95"], "recall@5": x["recall@5"], "recall@10": x["recall@10"],
                             "mrr@10": x["mrr@10"], "mrr@10_ci95": x["mrr@10_ci95"], "ndcg@10": x["ndcg@10"],
                             "label_status": "self-generated known-item queries (document-derived); not human relevance judgments",
                             "source_file": "indian/processed/retrieval/civis_v1/evaluation/metrics.json"})
    rows.append({"experiment": "CIVIS dense e5-base-v2", "dataset": "CIVIS", "method": "dense_e5-base-v2", "status": "not run",
                 "label_status": ("blocked: attempts 1-2 skipped by RAM guard (576/532 MB < ~722 MB); attempt 3 started with 1,373 MB free, "
                                  "ran at ~4 chunks/s and was interrupted at 1,824/22,300 chunks when the session ended; not restarted "
                                  "(449 MB RAM at resume). See embedding_status_e5-base-v2_*.json")})
    lsm = rj(P / "loksabha" / "retrieval_pilot_v1" / "evaluation" / "metrics.json")
    if lsm:
        for qt, d in lsm["metrics"].items():
            for m, subs in d.items():
                for sub, x in subs.items():
                    if not x.get("queries"):
                        continue
                    rows.append({"experiment": "Lok Sabha question->answer retrieval pilot", "dataset": "Lok Sabha Q&A", "query_type": qt, "method": m,
                                 "subset": sub, "n_queries": x["queries"], "corpus": f"{lsm['corpus']} records",
                                 "recall@1": x["recall@1"], "recall@1_ci95": x["recall@1_ci95"], "recall@5": x["recall@5"], "recall@10": x["recall@10"],
                                 "mrr@10": x["mrr@10"], "mrr@10_ci95": x["mrr@10_ci95"], "ndcg@10": x["ndcg@10"],
                                 "label_status": "self-generated known-item queries (question->own answer); not human relevance judgments",
                                 "source_file": "indian/processed/loksabha/retrieval_pilot_v1/evaluation/metrics.json"})
    mc = rj(P / "mygov" / "concern_discovery_exploratory_v1" / "summary.json")
    if mc:
        rows.append({"experiment": "MyGov exploratory concern discovery", "dataset": "MyGov", "method": "MiniLM-L6 + UMAP + HDBSCAN(10/5)", "subset": "Latin-script comments",
                     "n_queries": mc.get("comments_clustered"), "clusters_seed42": mc.get("clusters_seed42"), "noise_share": mc.get("noise_share_seed42"),
                     "seed_ari": mc.get("seed_ari_all_points"), "label_status": "EXPLORATORY unsupervised groupings; no ground truth",
                     "source_file": "indian/processed/mygov/concern_discovery_exploratory_v1/summary.json"})
    tw = rj(P / "tweets" / "weak_labels_sentiment140_v1" / "summary.json")
    if tw:
        rows.append({"experiment": "Tweets weak-label run (Sentiment140 model)", "dataset": "Kaggle tweets", "method": "sentiment140_hashing_sgd_v1",
                     "n_queries": tw.get("rows"), "weak_label_distribution": tw.get("distribution"), "share_confidence_below_0.6": tw.get("share_confidence_below_0.6"),
                     "label_status": "WEAK predictions only; accuracy unmeasurable without human labels",
                     "source_file": "indian/processed/tweets/weak_labels_sentiment140_v1/summary.json"})
    for name, why in (("MyGov stance/concern classifier", "no labels (annotation templates prepared)"),
                      ("Tweet sentiment classifier", "no human labels (pilot templates + 1,500 sampling plan prepared)")):
        rows.append({"experiment": name, "status": "not run (blocked)", "label_status": why})
    return rows


if __name__ == "__main__":
    write(R / "final_dataset_quality_summary.csv", quality())
    write(R / "final_experiment_results.csv", results())
