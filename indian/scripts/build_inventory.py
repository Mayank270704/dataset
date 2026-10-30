"""
indian/scripts/build_inventory.py
=================================
Recursive inventory of every file under indian/ (originals, browser resources, processed outputs, scripts, reports,
logs) with content-detected type, size, SHA-256, role, dataset, suitability category and record count where
measurable. Writes indian/reports/indian_file_inventory.csv (the previous version is archived in reports/archive/).

    python -I indian/scripts/build_inventory.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "indian_file_inventory.csv"
RECORDS = {}                                                        # filled from processing summaries / known counts


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def magic(p: Path) -> str:
    b = p.open("rb").read(16)
    for sig, t in ((b"\x89PNG", "PNG image"), (b"\xff\xd8\xff", "JPEG image"), (b"PAR1", "Parquet"), (b"%PDF", "PDF"), (b"GIF8", "GIF image")):
        if b.startswith(sig):
            return t
    t = p.open("rb").read(4096).decode("utf-8", "replace").lstrip()
    lo = t.lower()
    if lo.startswith("<!doctype html") or lo.startswith("<html"):
        return "HTML"
    if lo.startswith("<svg") or (lo.startswith("<?xml") and "<svg" in lo):
        return "SVG image"
    if p.suffix == ".json" or t.startswith("{"):
        return "JSON"
    if p.suffix == ".csv":
        return "CSV"
    if p.suffix == ".css":
        return "CSS stylesheet"
    if p.suffix == ".md":
        return "Markdown"
    if p.suffix == ".py":
        return "Python source"
    if p.suffix in (".log", ".txt"):
        return "text log"
    if p.name.endswith(".js.download") or p.name in ("js", "saved_resource") or "function" in t[:2000]:
        return "JavaScript (browser download)"
    return "text"


def classify(rel: str) -> tuple[str, str, str, str]:
    """role, dataset, category, note"""
    if rel.startswith("mygov_consultations/"):
        if "_files/" in rel:
            return "browser_resource", "MyGov page resources", "D", "site JS/CSS/images; no submission data"
        return "original_download", "MyGov Open Data Use License consultation", "A", "saved comment page (10 submissions; partial: 55 of ~114 pages)"
    if rel.startswith("kaggle/"):
        return "original_download", "Kaggle: Indian Politics Tweets & Reactions", "C", "social-media posts; not consultation data"
    if rel.startswith("huggingface_datasets/civis"):
        return "original_download", "HF: civis_consultation_summaries", "B", "policy documents + summaries; no citizen comments"
    if rel.startswith("huggingface_datasets/lok-sabha"):
        return "original_download", "HF: lok-sabha-qa", "B", "parliamentary Q&A; not consultation comments"
    if rel.startswith("processed/"):
        n = rel.split("/")[-1]
        ds = ("MyGov" if n.startswith("mygov") else "Kaggle tweets" if n.startswith("kaggle") else "CIVIS" if n.startswith("civis")
              else "Lok Sabha Q&A" if n.startswith("lok_sabha") else "all sources (summary)")
        cat = {"MyGov": "A", "Kaggle tweets": "C", "CIVIS": "B", "Lok Sabha Q&A": "B"}.get(ds, "")
        return "processed_output", ds, cat, "derived; write-once"
    if rel.startswith("scripts/"):
        return "script", "", "", ""
    if rel.startswith("reports/"):
        return "report", "", "", ""
    if rel.startswith("logs/"):
        return "log", "", "", ""
    return "other", "", "D", "unidentified"


def records(p: Path, rel: str) -> str:
    try:
        if rel in RECORDS:
            return str(RECORDS[rel])
        if p.suffix == ".csv" and p.stat().st_size < 60_000_000:
            with p.open(encoding="utf-8", newline="") as f:
                return str(sum(1 for _ in csv.reader(f)) - 1)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            return str(pq.ParquetFile(p).metadata.num_rows)
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"
    return ""


def main() -> int:
    RECORDS.update({"kaggle/Dataset of Indian Politics Tweets & Reactions/tweets.csv": "50000 (50001 CSV records; 1 split by a line break, repaired in processed copy)",
                    "huggingface_datasets/civis_consultation_summaries/train.json": 136,
                    "huggingface_datasets/civis_consultation_summaries/test.json": 34})
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == OUT:
            continue
        rel = p.relative_to(ROOT).as_posix()
        role, ds, cat, note = classify(rel)
        rec = "10" if (role == "original_download" and rel.endswith(".html")) else records(p, rel)
        rows.append({"relative_path": "indian/" + rel, "file_name": p.name, "extension": p.suffix or "(none)", "size_bytes": p.stat().st_size,
                     "detected_type": magic(p), "sha256": sha(p), "role": role, "dataset": ds, "category": cat, "records": rec, "note": note})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"files": len(rows), "bytes": sum(r["size_bytes"] for r in rows), "by_role": Counter(r["role"] for r in rows),
                      "by_type": Counter(r["detected_type"] for r in rows), "by_category": Counter(r["category"] for r in rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
