"""
indian/scripts/run_mygov_analysis_v1.py
=======================================
MyGov "Government Open Data Use License - India" consultation: validation, descriptive statistics, annotation
template (blank labels), and an EXPLORATORY unsupervised concern-discovery run.

The 550 comments are 48.6 % of the 1,132 displayed submissions (page indices 0-54, the most recent half by display
order). Nothing here represents all submissions. No stance, sentiment or concern labels are generated.

    python -I indian/scripts/run_mygov_analysis_v1.py stats        # validation + descriptive stats + annotation template
    python -I indian/scripts/run_mygov_analysis_v1.py clusters     # exploratory concern discovery (memory-guarded)
Outputs: indian/processed/mygov/{analysis_v1, annotation_v1, concern_discovery_exploratory_v1}/ (write-once)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "indian"
PROC = IND / "processed"
OUT = PROC / "mygov"
CLEAN, RAW, PII = PROC / "mygov_comments_clean.csv", PROC / "mygov_comments_raw.csv", PROC / "mygov_pii_review_flags.csv"
LICENCE = "unverified: MyGov/MeitY reuse terms not present locally; local research only"
CLAUSE_TAG = re.compile(r"#\s*clause\s*(\d+)", re.I)
CLAUSE_MENTION = re.compile(r"\bclause\s*(?:no\.?\s*)?(\d+)", re.I)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new_dir(p: Path) -> bool:
    if p.exists():
        print(f"[SKIP] {p.relative_to(ROOT)} exists (write-once)")
        return False
    p.mkdir(parents=True)
    return True


def q(s: pd.Series) -> dict:
    return {k: round(float(v), 1) for k, v in s.quantile([0, .1, .25, .5, .75, .9, 1]).items()}


def stage_stats() -> int:
    a, t = OUT / "analysis_v1", OUT / "annotation_v1"
    c = pd.read_csv(CLEAN, dtype=str, keep_default_na=False)
    r = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    p = pd.read_csv(PII, dtype=str, keep_default_na=False)
    b = lambda s: s == "True"
    val = {
        "rows_clean": len(c), "unique_ids": int(c.comment_id.nunique()), "ids_numeric": bool(c.comment_id.str.fullmatch(r"\d+").all()),
        "raw_rows": len(r), "raw_duplicate_id_records": int(b(r.is_duplicate_id).sum()),
        "raw_unique_ids_equal_clean": set(r.comment_id) == set(c.comment_id),
        "pii_file_aligned": p.comment_id.tolist() == c.comment_id.tolist(),
        "empty_texts": int((c.comment_text.str.strip() == "").sum()),
        "missing_by_column": {k: int(v) for k, v in (c == "").sum().items() if v},
        "duplicate_text_groups": int(c.exact_text_duplicate_group.replace("", np.nan).nunique()),
        "comments_in_duplicate_text_groups": int((c.exact_text_duplicate_group != "").sum()),
        "near_500_char_limit": int(b(c.near_500_char_limit).sum()),
        "near_limit_and_no_terminal_punctuation": int((b(c.near_500_char_limit) & b(c.ends_without_terminal_punctuation)).sum()),
        "pii_any_flag": int(b(p.any_flag).sum()), "pii_flag_counts": {k: int(b(p[k]).sum()) for k in p.columns if k not in ("comment_id", "any_flag", "review_note")},
    }
    val["all_checks_pass"] = (val["rows_clean"] == val["unique_ids"] == 550 and val["ids_numeric"] and val["raw_unique_ids_equal_clean"]
                              and val["pii_file_aligned"] and val["empty_texts"] == 0)
    tags = c.comment_text.map(lambda x: sorted({int(m) for m in CLAUSE_TAG.findall(x)}))
    ment = c.comment_text.map(lambda x: sorted({int(m) for m in CLAUSE_MENTION.findall(x)}))
    clause_any = [sorted(set(x) | set(y)) for x, y in zip(tags, ment)]
    wc, cc = c.word_count.astype(int), c.char_count.astype(int)
    stats = {
        "coverage": "550 unique of 1,132 displayed (48.6 %); page indices 0-54; most recent half by display order; not representative of all submissions",
        "char_count": q(cc), "word_count": q(wc),
        "share_under_10_words": round(float((wc < 10).mean()), 4), "share_ge_490_chars": round(float((cc >= 490).mean()), 4),
        "script_detected": c.script_detected.value_counts().to_dict(),
        "clause_references": {"comments_with_hash_clause_tag": int((tags.str.len() > 0).sum()),
                              "comments_with_any_clause_number": int(sum(1 for x in clause_any if x)),
                              "comments_mentioning_word_clause": int(c.comment_text.str.contains("clause", case=False).sum()),
                              "by_clause_number": dict(sorted(Counter(n for x in clause_any for n in x).items()))},
        "attachments": {"comments_with_attachment": int(b(c.has_attachment).sum())},
        "reply_count_displayed": c.reply_count_displayed.astype(int).value_counts().sort_index().to_dict(),
        "comments_with_url": int(b(c.contains_url).sum()),
        "per_page_comments": c.source_page_index.value_counts().describe().to_dict(),
        "top_terms_note": "no term statistics are reported as findings; see exploratory clustering",
    }
    if new_dir(a):
        (a / "validation.json").write_text(json.dumps(val, indent=2) + "\n", encoding="utf-8")
        (a / "descriptive_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")
        pd.DataFrame({"comment_id": c.comment_id, "clause_numbers_detected": [";".join(map(str, x)) for x in clause_any],
                      "hash_clause_tag": tags.str.len() > 0}).to_csv(a / "clause_references_detected.csv", index=False, lineterminator="\n")
    if new_dir(t):
        rng = np.random.default_rng(42)
        # pilot batch: 50 comments, stratified by length tercile, one per duplicate-text group, fixed seed
        uniq = c[(c.exact_text_duplicate_group == "") | ~c.duplicated("exact_text_duplicate_group")].copy()
        uniq["len_bin"] = pd.qcut(uniq.char_count.astype(int), 3, labels=False)
        pilot = set()
        for _, g in uniq.groupby("len_bin"):
            pilot |= set(g.comment_id.iloc[rng.permutation(len(g))[: 17 if len(pilot) < 34 else 16]])
        tmpl = pd.DataFrame({
            "comment_id": c.comment_id, "annotation_batch": np.where(c.comment_id.isin(pilot), "pilot_50", "main"),
            "source_page_index": c.source_page_index, "comment_text": c.comment_text,
            "possible_truncation": b(c.near_500_char_limit) & b(c.ends_without_terminal_punctuation),
            "script_detected": c.script_detected, "has_attachment": c.has_attachment, "pii_flag": p.any_flag.values,
            "clause_numbers_detected": [";".join(map(str, x)) for x in clause_any],
            "exact_text_duplicate_group": c.exact_text_duplicate_group,
            "stance": "", "clause_referenced": "", "concern_type": "", "requested_action": "", "evidence_span": "",
            "sentiment": "", "review_notes": "", "annotator_id": ""})
        for k in ("annotator_A", "annotator_B"):
            tmpl.to_csv(t / f"mygov_annotation_template_v1_{k}.csv", index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        (t / "README.md").write_text(
            "# MyGov annotation files (v1)\n\nBlank templates. Codebook: `indian/reports/mygov_annotation_codebook_v1.md`.\n"
            "Annotate `annotation_batch == pilot_50` first, independently; compute agreement before the main batch.\n"
            "Detected clause numbers and flags are aids, not labels. No label in these files is pre-filled.\n"
            f"Licence: {LICENCE}. Comments contain personal data (see pii_flag); do not share outside the annotation team.\n",
            encoding="utf-8")
        (t / "manifest.json").write_text(json.dumps({"pilot_size": int(len(pilot)), "seed": 42, "source_sha256": sha(CLEAN),
                                                      "outputs_sha256": {f.name: sha(f) for f in sorted(t.iterdir()) if f.suffix == ".csv"}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_pass": val["all_checks_pass"], "clause": stats["clause_references"], "words": stats["word_count"],
                      "pilot": len(pilot) if t.exists() else None}, default=str))
    return 0


def stage_clusters() -> int:
    import psutil
    d = OUT / "concern_discovery_exploratory_v1"
    if d.exists():
        print("[SKIP] exists")
        return 0
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(IND))
    from retrieval import civis as cv
    res = {"disk_free_mb": shutil.disk_usage(ROOT).free // 2**20, "ram_available_mb": psutil.virtual_memory().available // 2**20}
    c = pd.read_csv(CLEAN, dtype=str, keep_default_na=False)
    # all-MiniLM-L6-v2 is English-only: non-Latin comments are excluded from clustering and listed, not translated
    keep = c.script_detected == "Latin"
    name = "all-MiniLM-L6-v2"
    w = cv.cached_weights(cv.ENCODERS[name]["model_id"])
    need = int(w.stat().st_size / 2**20 * 1.25) + 200 if w else None
    status = {"encoder": name, "resources_before": res, "ram_needed_mb_estimate": need}
    if w is None or res["ram_available_mb"] < need:
        status["status"] = "skipped: " + ("encoder weights incomplete" if w is None else f"{res['ram_available_mb']} MB RAM < ~{need} MB")
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(status["status"])
        return 4
    import torch
    import hdbscan
    import umap
    from run_full_corpus_clustering import load_encoder_low_memory
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import adjusted_rand_score
    t0 = time.time()
    tok, model, _ = load_encoder_low_memory(cv.ENCODERS[name]["model_id"])
    torch.set_num_threads(6)
    sub = c[keep].reset_index(drop=True)
    E, trunc = cv.encode(sub.comment_text.tolist(), tok, model, max_length=256, batch_size=32)
    del model
    labels = {}
    for s in (42, 43, 44):
        red = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=s).fit_transform(E)
        labels[s] = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(red)
    ari = {f"{a}-{b}": round(float(adjusted_rand_score(labels[a], labels[b])), 4) for a, b in ((42, 43), (42, 44), (43, 44))}
    lab = labels[42]
    # c-TF-IDF keywords per cluster (seed 42)
    vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=2)
    X = vec.fit_transform(sub.comment_text)
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    for k in sorted(set(lab) - {-1}):
        m = lab == k
        tf = np.asarray(X[m].sum(0)).ravel()
        idf = np.log(1 + X.shape[0] / np.maximum(1, np.asarray((X > 0).sum(0)).ravel()))
        top = terms[np.argsort(-(tf / max(tf.sum(), 1) * idf))[:8]]
        cen = E[m].mean(0)
        cen /= np.linalg.norm(cen)
        rep = sub.comment_id[m].to_numpy()[np.argsort(-(E[m] @ cen))[:3]]
        # stability: best Jaccard of this cluster's members against clusters in the other seeds
        jac = []
        for s in (43, 44):
            o = labels[s]
            best = max((len(set(np.where(m)[0]) & set(np.where(o == j)[0])) / len(set(np.where(m)[0]) | set(np.where(o == j)[0]))
                        for j in set(o) - {-1}), default=0.0)
            jac.append(best)
        clusters.append({"cluster": int(k), "size": int(m.sum()), "top_terms": " ".join(top), "representative_comment_ids": ";".join(rep),
                         "stability_best_jaccard_seed43": round(jac[0], 3), "stability_best_jaccard_seed44": round(jac[1], 3),
                         "duplicate_text_members": int((sub.exact_text_duplicate_group[m] != "").sum()),
                         "possibly_truncated_members": int(((sub.near_500_char_limit[m] == "True") & (sub.ends_without_terminal_punctuation[m] == "True")).sum())})
    d.mkdir(parents=True)
    pd.DataFrame(clusters).to_csv(d / "clusters.csv", index=False, lineterminator="\n")
    pd.DataFrame({"comment_id": sub.comment_id, **{f"cluster_seed{s}": labels[s] for s in labels}}).to_csv(d / "assignments.csv", index=False, lineterminator="\n")
    c.loc[~keep, ["comment_id", "script_detected"]].assign(reason="non-Latin script; English-only encoder; not clustered").to_csv(d / "excluded_non_latin.csv", index=False, lineterminator="\n")
    out = status | {"status": "ok (EXPLORATORY: unsupervised groupings, not concern labels; no human validation)",
                    "comments_clustered": int(len(sub)), "excluded_non_latin": int((~keep).sum()), "truncated_at_256_tokens": trunc,
                    "umap": {"n_components": 5, "n_neighbors": 15, "min_dist": 0.0, "metric": "cosine"},
                    "hdbscan": {"min_cluster_size": 10, "min_samples": 5, "note": "smaller than the Canadian 25/5 because n=534; chosen a priori, not tuned"},
                    "clusters_seed42": int(len(clusters)), "noise_share_seed42": round(float((lab == -1).mean()), 4),
                    "clusters_per_seed": {s: int(len(set(v) - {-1})) for s, v in labels.items()},
                    "noise_share_per_seed": {s: round(float((v == -1).mean()), 4) for s, v in labels.items()},
                    "seed_ari_all_points": ari, "runtime_s": round(time.time() - t0, 1),
                    "resources_after": {"ram_available_mb": psutil.virtual_memory().available // 2**20},
                    "coverage_caveat": "partial, recency-biased sample of the consultation; groupings describe these 534 comments only",
                    "licence_status": LICENCE}
    (d / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("clusters_seed42", "noise_share_seed42", "clusters_per_seed", "seed_ari_all_points", "runtime_s")}, default=str))
    print(pd.DataFrame(clusters)[["cluster", "size", "top_terms", "stability_best_jaccard_seed43", "stability_best_jaccard_seed44"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit({"stats": stage_stats, "clusters": stage_clusters}[sys.argv[1]]())
