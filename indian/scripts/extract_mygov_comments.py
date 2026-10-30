"""
indian/scripts/extract_mygov_comments.py
========================================
Extract the submission records present in the locally saved MyGov consultation pages
("Public consultation on Government Open Data Use License - India").

Local files only: no network requests, no downloaded JavaScript is executed, and the original files are never
modified. HTML is parsed as text with BeautifulSoup (html.parser).

Outputs (write-once; the script refuses to overwrite):
    indian/processed/mygov_comments_raw.csv     every record found, one row per (file, comment); duplicates kept and flagged
    indian/processed/mygov_comments_clean.csv   one row per unique comment ID; analytical fields only; no usernames
    indian/processed/mygov_page_inventory.csv   one row per saved HTML file
    indian/processed/mygov_extraction_summary.json   counts, checks and SHA-256 of inputs and outputs

    python indian/scripts/extract_mygov_comments.py
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
from urllib.parse import unquote

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]                      # indian/
SRC = ROOT / "mygov_consultations"
OUT = ROOT / "processed"
STEM = "Public consultation on Government Open Data Use License - India _ MyGov.in"
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
TERMINAL_RE = re.compile(r"[.!?।)\]\"'”]\s*$")              # ends with . ! ? danda ) ] or a closing quote


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def file_number(name: str) -> int | None:
    m = re.fullmatch(re.escape(STEM) + r" ?(\d*)\.html", name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 0                  # 0 = the unnumbered first save


def page_index(url: str | None) -> int | None:
    """Comment-view page index from '?page=0,N' in the saved-from URL; the bare index.html is page 0."""
    if not url:
        return None
    m = re.search(r"[?&]page=([^&#]+)", unquote(url))
    if m:
        parts = m.group(1).split(",")
        return int(parts[-1]) if parts[-1].isdigit() else None
    return 0 if url.rstrip("/").endswith("index.html") else None


def text_of(node) -> str:
    """Comment text with <br>/<p> boundaries kept as line breaks; nothing else altered."""
    for br in node.find_all("br"):
        br.replace_with("\n")
    return node.get_text("", strip=False).strip()


def script_of(t: str) -> str:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return "none"
    dev = sum("ऀ" <= c <= "ॿ" for c in letters)
    lat = sum(c.isascii() for c in letters)
    other = len(letters) - dev - lat
    shares = {"Latin": lat, "Devanagari": dev, "other": other}
    top = max(shares, key=shares.get)
    return top if shares[top] / len(letters) >= 0.9 else "mixed"


def parse_file(path: Path) -> tuple[dict, list[dict], list[str]]:
    s = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(s, "html.parser")
    saved = re.search(r"saved from url=\(\d+\)(\S+)", s)
    mirror = re.search(r"Mirrored from (\S+) by HTTrack Website Copier/[\d.x]+ \[[^\]]*\], ([^-]+?GMT)", s)
    nxt = soup.select_one("li.pager-next a")
    shown = re.search(r"Showing (\d+) Submission", s)
    title = soup.title.get_text(strip=True).split("|")[0].strip() if soup.title else ""
    saved_url = saved.group(1) if saved else ""
    failures, recs = [], []
    rows = soup.find_all("article", class_="comment_row")
    for pos, art in enumerate(rows, 1):
        col = art.find(class_="comment_col1")
        cid = (col.get("id", "") if col else "").replace("comment_main_content_", "")
        body = art.select_one(".comment_body")
        if not cid.isdigit() or body is None:
            failures.append(f"{path.name} position {pos}: missing comment id or body")
            continue
        html_body = body.decode_contents()
        text = text_of(body)
        user = art.select_one(".username")
        avatar = art.select_one(".comment_user img")
        date = art.select_one(".date_time")
        rc = art.find("a", id="reply_count")
        rcount = re.search(r"\((\d+)\)", rc.get_text()) if rc else None
        files = art.select(".field-name-field-gi-comment-file a")
        share = art.select_one("a.share-n-track-facebook")
        recs.append({
            "comment_id": cid, "position_on_page": pos,
            "username": user.get_text(strip=True) if user else "",
            "avatar_local_file": (avatar.get("src", "").split("/")[-1] if avatar else ""),
            "relative_date_as_displayed": date.get_text(strip=True) if date else "",
            "comment_text": text, "comment_html": html_body,
            "reply_count_displayed": int(rcount.group(1)) if rcount else 0,
            "replies_present_in_file": bool(art.select_one(".replies") and art.select_one(".replies").get_text(strip=True)),
            "attachment_names": ";".join(a.get_text(strip=True) for a in files),
            "attachment_urls": ";".join(a.get("href", "") for a in files),
            "comment_permalink": (re.search(r"u=(\S+?)/?$", unquote(share.get("href", ""))).group(1) if share and "u=" in share.get("href", "") else ""),
        })
    meta = {"source_file": path.name, "file_number": file_number(path.name), "file_size_bytes": path.stat().st_size,
            "file_sha256": sha(path), "resources_dir_present": (path.parent / (path.stem + "_files")).is_dir(),
            "consultation_title": title, "saved_from_url": saved_url, "page_index": page_index(saved_url),
            "httrack_mirrored_from": mirror.group(1) if mirror else "", "httrack_mirror_time": mirror.group(2).strip() if mirror else "",
            "next_page_url": nxt.get("href", "") if nxt else "", "next_page_index": page_index(nxt.get("href")) if nxt else None,
            "displayed_total_submissions": int(shown.group(1)) if shown else None,
            "comment_rows_found": len(rows), "comments_extracted": len(recs), "extraction_failures": len(failures)}
    return meta, recs, failures


def main() -> int:
    outs = {k: OUT / f"mygov_{k}" for k in ("comments_raw.csv", "comments_clean.csv", "page_inventory.csv", "extraction_summary.json")}
    if any(p.exists() for p in outs.values()):
        print("[STOP] an output already exists; nothing written", file=sys.stderr)
        return 2
    html_files = sorted((p for p in SRC.glob("*.html")), key=lambda p: (file_number(p.name) is None, file_number(p.name) or 0, p.name))
    pages, raw, failures = [], [], []
    for p in html_files:
        meta, recs, fail = parse_file(p)
        pages.append(meta)
        failures += fail
        for r in recs:
            raw.append({"record_seq": len(raw) + 1, "source_file": meta["source_file"], "file_number": meta["file_number"],
                        "page_index": meta["page_index"], "saved_from_url": meta["saved_from_url"],
                        "consultation_title": meta["consultation_title"], **r})

    # duplicates by comment ID; the canonical copy is the first occurrence in numbered-file order (file 0 = unnumbered save is sorted first,
    # so prefer numbered files by re-ordering: numbered files 1..N first, then the unnumbered file)
    order = sorted(raw, key=lambda r: (r["file_number"] == 0, r["file_number"], r["position_on_page"]))
    first: dict[str, int] = {}
    for r in order:
        first.setdefault(r["comment_id"], r["record_seq"])
    for r in raw:
        r["canonical_record_seq"] = first[r["comment_id"]]
        r["is_duplicate_id"] = r["record_seq"] != first[r["comment_id"]]

    # page-level duplicate detection (identical comment-ID sequence)
    seq_by_file = {}
    for r in raw:
        seq_by_file.setdefault(r["source_file"], []).append(r["comment_id"])
    seen_seq: dict[tuple, str] = {}
    for m in sorted(pages, key=lambda m: (m["file_number"] == 0, m["file_number"] or 0)):
        key = tuple(seq_by_file.get(m["source_file"], []))
        m["duplicate_of_file"] = seen_seq.get(key, "")
        seen_seq.setdefault(key, m["source_file"])

    uniq = [r for r in raw if not r["is_duplicate_id"]]
    norm = lambda t: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip().lower()
    tgroups = Counter(norm(r["comment_text"]) for r in uniq)
    gid = {t: i + 1 for i, t in enumerate(t for t, c in tgroups.items() if c > 1)}
    clean = []
    for r in sorted(uniq, key=lambda r: (r["page_index"], r["position_on_page"])):
        t = re.sub(r"[ \t]+", " ", r["comment_text"])          # whitespace only; characters unchanged
        t = re.sub(r"\n\s*\n+", "\n", t).strip()
        clean.append({
            "comment_id": r["comment_id"], "consultation_title": r["consultation_title"],
            "consultation_archive_url": "https://static.mygov.in/archive/mygov/en/public/group-issue/public-consultation-government-open-data-use-license-india/index.html",
            "source_file": r["source_file"], "source_page_index": r["page_index"], "position_on_page": r["position_on_page"],
            "comment_text": t, "char_count": len(t), "word_count": len(t.split()),
            "submission_date": "", "submission_date_note": "not available in saved HTML (relative date only)",
            "language": "", "script_detected": script_of(t),
            "has_attachment": bool(r["attachment_urls"]), "attachment_count": len([a for a in r["attachment_urls"].split(";") if a]),
            "reply_count_displayed": r["reply_count_displayed"],
            "exact_text_duplicate_group": gid.get(norm(r["comment_text"]), ""),
            "contains_email": bool(EMAIL_RE.search(t)), "contains_phone_like_number": bool(PHONE_RE.search(t)),
            "contains_url": bool(URL_RE.search(t)), "is_empty_text": not t,
            "near_500_char_limit": len(r["comment_text"]) >= 490,
            "ends_without_terminal_punctuation": bool(t) and not TERMINAL_RE.search(t),
            "raw_record_seq": r["record_seq"],
        })

    OUT.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerows(rows)

    write(outs["comments_raw.csv"], raw)
    write(outs["comments_clean.csv"], clean)
    write(outs["page_inventory.csv"], pages)

    idx = sorted({m["page_index"] for m in pages if m["page_index"] is not None})
    expected = set(range(0, max(idx) + 1)) if idx else set()
    nexts = {m["next_page_index"] for m in pages if m["next_page_index"] is not None}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network_requests": 0, "originals_modified": False,
        "html_files_inspected": len(pages), "distinct_page_indices": len(idx), "page_index_range": [min(idx), max(idx)] if idx else None,
        "missing_page_indices_within_range": sorted(expected - set(idx)),
        "unsaved_pages_linked_by_next": sorted(nexts - set(idx)),
        "duplicate_page_files": {m["source_file"]: m["duplicate_of_file"] for m in pages if m["duplicate_of_file"]},
        "records_extracted_raw": len(raw), "unique_comment_ids": len(uniq), "duplicate_id_records": sum(r["is_duplicate_id"] for r in raw),
        "exact_text_duplicate_groups_among_unique_ids": len(gid),
        "records_in_text_duplicate_groups": sum(c for c in tgroups.values() if c > 1),
        "extraction_failures": failures,
        "displayed_total_submissions": sorted({m["displayed_total_submissions"] for m in pages}),
        "share_of_displayed_total_recovered": round(len(uniq) / 1132, 4),
        "distinct_usernames_raw_only": len({r["username"] for r in uniq}),
        "clean_flags": {k: sum(bool(c[k]) for c in clean) for k in ("has_attachment", "contains_email", "contains_phone_like_number", "contains_url", "is_empty_text", "near_500_char_limit")},
        "near_500_limit_and_no_terminal_punctuation": sum(c["near_500_char_limit"] and c["ends_without_terminal_punctuation"] for c in clean),
        "max_raw_text_chars": max(len(r["comment_text"]) for r in uniq) if uniq else 0,
        "adjacent_id_order_exceptions": [[a["comment_id"], b["comment_id"]] for a, b in zip(clean, clean[1:]) if int(a["comment_id"]) <= int(b["comment_id"])],
        "script_detected": dict(Counter(c["script_detected"] for c in clean)),
        "comment_id_range": [min(int(r["comment_id"]) for r in uniq), max(int(r["comment_id"]) for r in uniq)] if uniq else None,
        "ids_strictly_descending_in_page_order": all(int(a["comment_id"]) > int(b["comment_id"]) for a, b in zip(clean, clean[1:])),
        "reuse_permission": "UNVERIFIED: no MyGov licence or terms-of-use text is present in the saved files; confirm before any reuse or publication",
        "inputs_sha256": {m["source_file"]: m["file_sha256"] for m in pages},
    }
    summary["outputs_sha256"] = {p.name: sha(p) for k, p in outs.items() if k != "extraction_summary.json"}
    outs["extraction_summary.json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("inputs_sha256", "outputs_sha256")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
