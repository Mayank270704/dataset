"""
indian/scripts/process_indian_datasets.py
=========================================
Source-specific cleaning and validation for every dataset under indian/. Local files only; no network access;
original downloads are never modified. Every output is write-once (the script skips an output that already exists
and says so). Sources are NEVER merged with each other or with the Canadian corpus.

Sections (run all, or name some):
    mygov    validate the existing MyGov extraction against the saved HTML; PII review flags; annotation pool
    tweets   Kaggle "Dataset of Indian Politics Tweets & Reactions" -> separate social-media dataset
    civis    Hugging Face civis_consultation_summaries -> policy documents + reference summaries (not comments)
    loksabha Hugging Face lok-sabha-qa -> parliamentary questions and answers (not comments); streamed by row group

    python -I indian/scripts/process_indian_datasets.py            # all sections
    python -I indian/scripts/process_indian_datasets.py tweets civis
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                     # indian/
PROC = ROOT / "processed"
MYGOV = ROOT / "mygov_consultations"
TWEETS = ROOT / "kaggle" / "Dataset of Indian Politics Tweets & Reactions" / "tweets.csv"
CIVIS = ROOT / "huggingface_datasets" / "civis_consultation_summaries"
LOKSABHA = ROOT / "huggingface_datasets" / "lok-sabha-qa"
LICENCE_UNVERIFIED = "unverified: no licence or terms file present locally; confirm before reuse"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
SIGNOFF_RE = re.compile(r"(?m)^\s*(?i:regards|thanks|thank you|sincerely|yours (?:truly|faithfully|sincerely)|warm regards|best regards)\b[,.!]?\s*$|"
                        r"\b(?i:regards|thanks|sincerely)\s*,\s*[A-Z][a-z]+|\b(?i:on behalf of)\b")
KRUTIDEV_RE = re.compile(r"[¼½]")                               # legacy Hindi font artefacts (heuristic)

RESULTS: dict[str, dict] = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.parent).as_posix()


def script_of(t: str) -> str:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return "none"
    dev = sum("ऀ" <= c <= "ॿ" for c in letters)
    lat = sum(c.isascii() for c in letters)
    shares = {"Latin": lat, "Devanagari": dev, "other": len(letters) - dev - lat}
    top = max(shares, key=shares.get)
    return top if shares[top] / len(letters) >= 0.9 else "mixed"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> bool:
    if path.exists():
        print(f"[SKIP] {rel(path)} exists (write-once)")
        return False
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[WROTE] {rel(path)} ({len(rows):,} rows)")
    return True


def missing(rows: list[dict], fields: list[str]) -> dict[str, int]:
    return {f: sum(1 for r in rows if r.get(f) in ("", None)) for f in fields}


# ── MyGov ─────────────────────────────────────────────────────────────────────

def mygov() -> None:
    from bs4 import BeautifulSoup
    raw_p, clean_p = PROC / "mygov_comments_raw.csv", PROC / "mygov_comments_clean.csv"
    summ = json.loads((PROC / "mygov_extraction_summary.json").read_text(encoding="utf-8"))
    checks = {}
    for p in (raw_p, clean_p):
        checks[f"{p.name}_sha256_matches_extraction_summary"] = sha(p) == summ["outputs_sha256"][p.name]
    raw = list(csv.DictReader(raw_p.open(encoding="utf-8")))
    clean = list(csv.DictReader(clean_p.open(encoding="utf-8")))
    # independent re-count from the saved HTML (not from the extraction script)
    html_ids, per_file = [], {}
    for h in sorted(MYGOV.glob("*.html")):
        s = h.read_text(encoding="utf-8")
        ids = re.findall(r'id="comment_main_content_(\d+)"', s)
        per_file[h.name] = len(ids)
        html_ids += ids
        if sha(h) != summ["inputs_sha256"].get(h.name):
            checks.setdefault("html_hash_mismatches", []).append(h.name)
    checks["html_files"] = len(per_file)
    checks["html_comment_rows_total"] = len(html_ids)
    checks["html_unique_ids"] = len(set(html_ids))
    checks["raw_rows_equal_html_rows"] = len(raw) == len(html_ids)
    checks["clean_ids_equal_html_unique_ids"] = {r["comment_id"] for r in clean} == set(html_ids)
    checks["clean_ids_unique"] = len({r["comment_id"] for r in clean}) == len(clean)
    checks["clean_has_no_username_column"] = "username" not in clean[0]
    # text fidelity: every clean text equals the raw text up to whitespace collapsing
    rawtxt = {r["comment_id"]: r["comment_text"] for r in raw if r["is_duplicate_id"] == "False"}
    ws = lambda t: re.sub(r"\s+", " ", t).strip()
    checks["clean_text_matches_raw_modulo_whitespace"] = sum(ws(rawtxt[r["comment_id"]]) == ws(r["comment_text"]) for r in clean)
    # spot check 25 records directly against HTML
    import random
    random.seed(7)
    ok = 0
    for r in random.sample(raw, 25):
        soup = BeautifulSoup((MYGOV / r["source_file"]).read_text(encoding="utf-8"), "html.parser")
        node = soup.find(id=f"comment_main_content_{r['comment_id']}")
        ok += bool(node) and node.select_one(".comment_body").decode_contents() == r["comment_html"]
    checks["html_spot_check_25_exact"] = ok

    # PII / identifying-information review (flags only; matched strings are not copied)
    pii, ann = [], []
    for r in clean:
        t = r["comment_text"]
        f = {"comment_id": r["comment_id"],
             "email": bool(EMAIL_RE.search(t)), "phone_like": bool(PHONE_RE.search(t)),
             "aadhaar_like_12_digits": bool(AADHAAR_RE.search(t)), "pan_like": bool(PAN_RE.search(t)),
             "url": bool(URL_RE.search(t)), "signoff_or_on_behalf_of": bool(SIGNOFF_RE.search(t)),
             "at_mention": bool(MENTION_RE.search(re.sub(EMAIL_RE, "", t)))}
        f["any_flag"] = any(v for k, v in f.items() if k != "comment_id")
        f["review_note"] = "manual review before any sharing; flags are heuristics"
        pii.append(f)
        ann.append({"comment_id": r["comment_id"], "consultation_title": r["consultation_title"],
                    "consultation_question": "Public consultation on the draft Government Open Data Use License - India (MyGov, 2016)",
                    "comment_text": t, "near_500_char_limit": r["near_500_char_limit"],
                    "ends_without_terminal_punctuation": r["ends_without_terminal_punctuation"], "script_detected": r["script_detected"],
                    "pii_flag": f["any_flag"], "exact_text_duplicate_group": r["exact_text_duplicate_group"],
                    "stance": "", "sentiment": "", "concern": "", "requested_action": "", "review_notes": "", "annotator_id": ""})
    write_csv(PROC / "mygov_pii_review_flags.csv", pii)
    write_csv(PROC / "mygov_annotation_pool_blank.csv", ann)
    n = lambda k: sum(1 for r in clean if r[k] == "True")
    RESULTS["mygov"] = {
        "dataset": "MyGov: Public consultation on Government Open Data Use License - India", "source_type": "Indian government public consultation submissions",
        "files": {"combined_raw": rel(raw_p), "clean_analytical": rel(clean_p), "pii_review": "indian/processed/mygov_pii_review_flags.csv",
                  "annotation_pool_blank": "indian/processed/mygov_annotation_pool_blank.csv"},
        "rows_raw": len(raw), "rows_clean": len(clean), "columns_clean": list(clean[0]),
        "validation": checks,
        "missing_values_clean": {"submission_date": len(clean), "language": len(clean), **{k: v for k, v in missing(clean, list(clean[0])).items() if v and k not in ("submission_date", "language", "exact_text_duplicate_group")}},
        "duplicates": {"duplicate_id_records_in_raw": sum(r["is_duplicate_id"] == "True" for r in raw),
                       "exact_text_groups_distinct_ids_kept": len({r["exact_text_duplicate_group"] for r in clean if r["exact_text_duplicate_group"]})},
        "truncation": {"near_500_char_limit": n("near_500_char_limit"),
                       "near_limit_and_no_terminal_punctuation": sum(r["near_500_char_limit"] == "True" and r["ends_without_terminal_punctuation"] == "True" for r in clean)},
        "pii_flags": {k: sum(p[k] for p in pii) for k in pii[0] if k not in ("comment_id", "review_note")},
        "coverage": "550 unique of 1,132 displayed submissions (48.6%); page indices 0-54 saved, 55-113 not saved; most recent half by display order",
        "labels": "none (no stance, sentiment or concern labels exist)",
        "licence_status": LICENCE_UNVERIFIED,
        "suitability": "consultation analysis (partial, recency-biased sample) after reuse permission is confirmed; annotation-ready",
    }


# ── Kaggle tweets ─────────────────────────────────────────────────────────────

def tweets() -> None:
    rows = list(csv.reader(TWEETS.open(encoding="utf-8", newline="")))
    header, body = rows[0], rows[1:]
    recs, repaired, unparsed = [], [], []
    i = 0
    while i < len(body):
        r = body[i]
        if len(r) == 6 and r[0].isdigit():
            recs.append((r, False))
            i += 1
        elif len(r) == 4 and r[0].isdigit() and i + 1 < len(body) and len(body[i + 1]) == 3:
            nxt = body[i + 1]                                     # unquoted line break inside the tweet text
            recs.append(([r[0], r[1], r[2], r[3] + "\n" + nxt[0], nxt[1], nxt[2]], True))
            repaired.append(r[0])
            i += 2
        else:
            unparsed.append(i + 2)
            i += 1
    users: dict[str, str] = {}
    norm = lambda t: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip().lower()
    tcount = Counter(norm(r[3]) for r, _ in recs)
    grp = {t: f"T{k + 1:05d}" for k, t in enumerate(t for t, c in tcount.items() if c > 1)}
    out = []
    for r, rep in recs:
        idx, date, user, text, likes, rts = r
        uid = users.setdefault(user, f"U{len(users) + 1:06d}")
        masked = MENTION_RE.sub("@[user]", URL_RE.sub("[url]", text))
        out.append({"tweet_row_id": int(idx), "created_at_utc": date, "author_pseudo_id": uid, "tweet_text": text,
                    "tweet_text_masked": masked, "likes": likes, "retweets": rts,
                    "char_count": len(text), "mention_count": len(MENTION_RE.findall(text)), "has_url": bool(URL_RE.search(text)),
                    "script_detected": script_of(text), "exact_text_duplicate_group": grp.get(norm(text), ""),
                    "is_empty_text": not text.strip(), "repaired_line_break": rep,
                    "source_file": rel(TWEETS), "licence_status": LICENCE_UNVERIFIED})
    write_csv(PROC / "kaggle_indian_politics_tweets_clean.csv", out)
    dates = sorted(o["created_at_utc"] for o in out if o["created_at_utc"])
    RESULTS["tweets"] = {
        "dataset": "Kaggle (folder name): Dataset of Indian Politics Tweets & Reactions", "source_type": "social-media posts (Twitter/X), not consultation submissions",
        "files": {"original": rel(TWEETS), "clean": "indian/processed/kaggle_indian_politics_tweets_clean.csv"},
        "rows_original_csv_records": len(body), "rows_clean": len(out), "repaired_split_records": repaired, "unparsed_lines": unparsed,
        "columns_clean": list(out[0]), "date_range": [dates[0], dates[-1]] if dates else None,
        "missing_values_clean": {k: v for k, v in missing(out, list(out[0])).items() if v and k != "exact_text_duplicate_group"},
        "duplicates": {"exact_text_duplicate_groups": len(grp), "records_in_groups": sum(c for c in tcount.values() if c > 1),
                       "note": "flagged, not removed: no tweet ID exists to prove the same post"},
        "distinct_authors": len(users), "script": dict(Counter(o["script_detected"] for o in out)),
        "pii": "author handles replaced by pseudonymous IDs in the clean file; tweet_text keeps @mentions (original); tweet_text_masked masks mentions and URLs",
        "labels": "none (likes/retweets are engagement counts, not sentiment or stance labels)",
        "licence_status": LICENCE_UNVERIFIED,
        "suitability": "supplementary experimentation only (separate social-media sentiment/topic study); not consultation data",
    }


# ── CIVIS ─────────────────────────────────────────────────────────────────────

def civis() -> None:
    out = []
    for split in ("train", "test"):
        j = json.loads((CIVIS / f"{split}.json").read_text(encoding="utf-8"))
        for k in sorted(j["Title"], key=int):
            out.append({"civis_record_id": f"{split}_{int(k):03d}", "split": split, "source_index": int(k),
                        "title": j["Title"][k], "document_text": j["Text"][k], "reference_summary": j["Summary"][k]})
    tcount = Counter(o["document_text"] for o in out if o["document_text"].strip())
    grp = {t: f"D{k + 1:03d}" for k, t in enumerate(t for t, c in tcount.items() if c > 1)}
    for o in out:
        t = o["document_text"]
        o.update({"document_char_count": len(t), "summary_char_count": len(o["reference_summary"]),
                  "document_text_empty": not t.strip(), "duplicate_document_group": grp.get(t, "") if t.strip() else "",
                  "duplicate_across_splits": bool(t.strip()) and sum(1 for x in out if x["document_text"] == t and x["split"] != o["split"]) > 0,
                  "content_type": "policy/consultation document text + reference summary (not citizen comments)",
                  "source_file": f"indian/huggingface_datasets/civis_consultation_summaries/{o['split']}.json", "licence_status": LICENCE_UNVERIFIED})
    write_csv(PROC / "civis_policy_documents_clean.csv", out)
    meta = [{k: v for k, v in o.items() if k not in ("document_text", "reference_summary")} for o in out]
    write_csv(PROC / "civis_policy_documents_metadata.csv", meta)
    RESULTS["civis"] = {
        "dataset": "Hugging Face (folder name): civis_consultation_summaries", "source_type": "Indian draft policy / consultation documents with reference summaries",
        "files": {"clean": "indian/processed/civis_policy_documents_clean.csv", "metadata": "indian/processed/civis_policy_documents_metadata.csv"},
        "rows": len(out), "by_split": dict(Counter(o["split"] for o in out)), "columns_clean": list(out[0]),
        "empty_document_text": [o["civis_record_id"] for o in out if o["document_text_empty"]],
        "duplicate_document_groups": {g: [o["civis_record_id"] + ": " + o["title"][:60] for o in out if o["duplicate_document_group"] == g] for g in sorted(set(grp.values()))},
        "document_chars_median_max": [sorted(o["document_char_count"] for o in out)[len(out) // 2], max(o["document_char_count"] for o in out)],
        "labels": "reference summaries only (usable as summarisation targets once provenance/licence verified); no stance/sentiment labels",
        "licence_status": LICENCE_UNVERIFIED,
        "suitability": "policy context (document grounding, summarisation); not consultation-comment analysis",
        "dedup_decision": "no rows removed: duplicate texts have different titles, so the pairing cannot be resolved locally; flagged instead",
    }


# ── Lok Sabha ─────────────────────────────────────────────────────────────────

def loksabha() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    meta_p, clean_p = PROC / "lok_sabha_qa_metadata.csv", PROC / "lok_sabha_qa_clean.parquet"
    if meta_p.exists() or clean_p.exists():
        print("[SKIP] lok sabha outputs exist (write-once)")
        return
    shards = sorted(LOKSABHA.glob("train-*.parquet"))
    schema = pa.schema([("qa_id", pa.string()), ("lok_sabha", pa.int64()), ("session", pa.int64()), ("question_no", pa.int64()),
                        ("question_type", pa.string()), ("date", pa.string()), ("subject", pa.string()), ("ministry", pa.string()),
                        ("member_count", pa.int64()), ("members", pa.list_(pa.string())), ("question_text", pa.string()), ("answer_text", pa.string()),
                        ("question_word_count", pa.float64()), ("answer_word_count", pa.float64()), ("num_pages", pa.float64()),
                        ("pdf_url", pa.string()), ("pdf_url_hindi", pa.string()), ("question_empty", pa.bool_()), ("answer_empty", pa.bool_()),
                        ("legacy_hindi_font_artifacts", pa.bool_()), ("source_shard", pa.string())])
    mfields = [f.name for f in schema if f.name not in ("members", "question_text", "answer_text")]
    writer = pq.ParquetWriter(clean_p, schema, compression="zstd")
    mf = meta_p.open("w", newline="", encoding="utf-8")
    mw = csv.DictWriter(mf, fieldnames=mfields, lineterminator="\n")
    mw.writeheader()
    n, ids, keys = 0, set(), set()
    stats = Counter()
    try:
        for s in shards:
            pf = pq.ParquetFile(s)
            for rg in range(pf.num_row_groups):
                cols = [c for c in pf.schema_arrow.names if c != "full_text"]      # full_text not needed; halves memory
                d = pf.read_row_group(rg, columns=cols).to_pylist()
                recs = []
                for r in d:
                    q, a = r.get("question_text") or "", r.get("answer_text") or ""
                    rec = {"qa_id": r["id"], "lok_sabha": r["lok_no"], "session": r["session_no"], "question_no": r["ques_no"],
                           "question_type": r["type"], "date": r["date"], "subject": r["subject"], "ministry": r["ministry"],
                           "member_count": len(r["members"] or []), "members": r["members"] or [], "question_text": q or None,
                           "answer_text": a or None, "question_word_count": r["question_word_count"], "answer_word_count": r["answer_word_count"],
                           "num_pages": r["num_pages"], "pdf_url": r["pdf_url"], "pdf_url_hindi": r["pdf_url_hindi"],
                           "question_empty": not q.strip(), "answer_empty": not a.strip(),
                           "legacy_hindi_font_artifacts": bool(KRUTIDEV_RE.search(a) or KRUTIDEV_RE.search(q)), "source_shard": s.name}
                    stats["question_empty"] += rec["question_empty"]
                    stats["answer_empty"] += rec["answer_empty"]
                    stats["legacy_font"] += rec["legacy_hindi_font_artifacts"]
                    stats["dup_id"] += rec["qa_id"] in ids
                    ids.add(rec["qa_id"])
                    k = (rec["lok_sabha"], rec["session"], rec["question_no"], rec["question_type"])
                    stats["dup_key"] += k in keys
                    keys.add(k)
                    for f in ("subject", "ministry", "date", "pdf_url"):
                        stats[f"missing_{f}"] += not (rec[f] or "")
                    recs.append(rec)
                writer.write_table(pa.Table.from_pylist(recs, schema=schema))
                mw.writerows({k: v for k, v in r.items() if k in mfields} for r in recs)
                n += len(recs)
                del d, recs
    finally:
        writer.close()
        mf.close()
    print(f"[WROTE] {rel(clean_p)} and {rel(meta_p)} ({n:,} rows)")
    RESULTS["loksabha"] = {
        "dataset": "Hugging Face (folder name): lok-sabha-qa", "source_type": "Lok Sabha parliamentary questions and ministerial answers (not consultation comments)",
        "files": {"clean_parquet": rel(clean_p), "metadata_csv": rel(meta_p)},
        "format_note": "text kept in Parquet: a CSV of 150k Q&A texts would be several hundred MB; the CSV holds metadata only. full_text (whole PDF text incl. headers) dropped from the analytical copy; it remains in the original shards.",
        "rows": n, "columns_clean": [f.name for f in schema], "quality": dict(stats),
        "labels": "none for stance/sentiment; ministry/subject are metadata, usable only as topic labels with caution",
        "pii": "member names are elected officials acting publicly; kept as a list, not used as features by default",
        "licence_status": LICENCE_UNVERIFIED,
        "suitability": "policy context (government responses, ministry-topic grounding); not consultation-comment analysis",
    }


def main(argv: list[str]) -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    secs = {"mygov": mygov, "tweets": tweets, "civis": civis, "loksabha": loksabha}
    todo = argv or list(secs)
    PROC.mkdir(exist_ok=True)
    for k in todo:
        print(f"== {k}")
        secs[k]()
    out = PROC / f"processing_summary_{'_'.join(todo)}.json"
    if out.exists():
        print(f"[SKIP] {rel(out)} exists")
    else:
        out.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "network_requests": 0,
                                   "merged_sources": False, "results": RESULTS}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(f"[WROTE] {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
