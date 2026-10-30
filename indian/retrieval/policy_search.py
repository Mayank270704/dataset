"""
indian/retrieval/policy_search.py
=================================
Local command-line search over two SEPARATE indexes:
  * CIVIS policy documents (164 canonical documents, 22,300 chunks; indian/processed/retrieval/civis_v1)
  * Lok Sabha Q&A pilot corpus (10,000 records; indian/processed/loksabha/retrieval_pilot_v1)

Each source is searched and ranked independently; scores are never compared or merged across sources.
MyGov comments and tweets are NOT indexed. Results are text-similarity matches, not verified answers.

    python -I indian/retrieval/policy_search.py "open data licence attribution" --top 5
    python -I indian/retrieval/policy_search.py "road safety rules" --source civis --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IND))
from retrieval import civis as cv  # noqa: E402
from retrieval import loksabha as ls  # noqa: E402

CIVIS_DIR = IND / "processed" / "retrieval" / "civis_v1"
CIVIS_SRC = IND / "processed" / "civis_policy_documents_clean.csv"
LS_DIR = IND / "processed" / "loksabha" / "retrieval_pilot_v1"
LS_PQ = IND / "processed" / "lok_sabha_qa_clean.parquet"
DISCLAIMER = ("Results are ranked by keyword (BM25) similarity within each source. They are candidate evidence, not verified "
              "answers; read the source before relying on it. Licence and reuse permission for both sources are UNVERIFIED.")


@dataclass
class Result:
    source: str
    rank: int
    score: float
    score_type: str
    record_id: str
    title_or_question: str
    snippet: str
    date: str
    issuer_or_ministry: str
    url: str
    provenance: str
    quality_flags: str
    licence_status: str


def _snippet(text: str, query: str, width: int = 240) -> str:
    t = re.sub(r"\s+", " ", text)
    words = [w for w in re.findall(r"\w{4,}", query.lower())]
    pos = min([p for p in (t.lower().find(w) for w in words) if p >= 0], default=0)
    s = max(0, pos - width // 3)
    return ("…" if s else "") + t[s:s + width] + ("…" if s + width < len(t) else "")


class CivisIndex:
    name = "CIVIS policy documents"

    def __init__(self):
        self.docs = pd.read_csv(CIVIS_DIR / "documents.csv", dtype=str, keep_default_na=False)
        self.ch = pd.read_csv(CIVIS_DIR / "chunks.csv", keep_default_na=False)
        src = pd.read_csv(CIVIS_SRC, dtype=str, keep_default_na=False)
        text = dict(zip(src.civis_record_id, src.document_text))
        self.ctext = [text[k][a:b] for k, a, b in zip(self.ch.doc_key, self.ch.start, self.ch.end)]
        self.keys = self.docs.doc_key.tolist()
        self.cdoc = self.ch.doc_key.map({k: i for i, k in enumerate(self.keys)}).to_numpy()
        self.bm = cv.BM25().fit(self.ctext)

    def search(self, query: str, top: int) -> list[Result]:
        cs = self.bm.score(query)
        ds = cv.doc_scores_from_chunks(cs, self.cdoc, len(self.keys))
        out = []
        for r, i in enumerate(cv.rank_order(ds)[:top], 1):
            if not np.isfinite(ds[i]) or ds[i] <= 0:
                break
            best = int(np.argmax(np.where(self.cdoc == i, cs, -np.inf)))
            d = self.docs.iloc[i]
            out.append(Result("CIVIS", r, round(float(ds[i]), 3), "bm25 (best chunk)", d.member_record_ids, d.titles,
                              _snippet(self.ctext[best], query), "", "", "",
                              f"civis_policy_documents_clean.csv; chunk {self.ch.chunk_id[best]} chars {self.ch.start[best]}-{self.ch.end[best]}; "
                              f"split {d.splits}; policy document (not citizen comments); summaries of unverified origin",
                              "identical-text records merged" if ";" in d.member_record_ids else "", d.licence_status))
        return out


class LokSabhaIndex:
    name = "Lok Sabha Q&A (10,000-record pilot corpus)"

    def __init__(self):
        self.c = pd.read_csv(LS_DIR / "corpus.csv", dtype={"qa_id": str}, keep_default_na=False)
        tx = ls.read_texts(LS_PQ, set(self.c.qa_id), cols=("qa_id", "question_text", "answer_text"))
        self.t = self.c[["qa_id"]].merge(tx, on="qa_id", how="left", validate="one_to_one")
        self.bm = cv.BM25().fit(self.t.answer_text.tolist())

    def search(self, query: str, top: int) -> list[Result]:
        s = self.bm.score(query)
        out = []
        for r, i in enumerate(cv.rank_order(s)[:top], 1):
            if s[i] <= 0:
                break
            m = self.c.iloc[i]
            flags = ";".join(f for f, v in (("legacy_hindi_font_artifacts", m.legacy_hindi_font_artifacts),) if str(v) == "True")
            out.append(Result("Lok Sabha Q&A", r, round(float(s[i]), 3), "bm25 (answer text)", m.qa_id, m.subject,
                              _snippet(self.t.answer_text[i] or "", query), m.date, m.ministry, m.pdf_url,
                              f"lok_sabha_qa_clean.parquet; Lok Sabha {m.lok_sabha}, session {m.session}, {m.question_type} Q{m.question_no}; "
                              "parliamentary question and ministerial answer (not a consultation comment)", flags, m.licence_status))
        return out


def search(query: str, sources=("civis", "loksabha"), top: int = 5) -> dict[str, list[Result]]:
    idx = {"civis": CivisIndex, "loksabha": LokSabhaIndex}
    return {k: idx[k]().search(query, top) for k in sources}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Search CIVIS policy documents and Lok Sabha Q&A separately (local, keyword).")
    ap.add_argument("query")
    ap.add_argument("--source", choices=["civis", "loksabha", "both"], default="both")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = ("civis", "loksabha") if a.source == "both" else (a.source,)
    res = search(a.query, src, a.top)
    if a.json:
        print(json.dumps({"query": a.query, "disclaimer": DISCLAIMER, "results_by_source": {k: [asdict(r) for r in v] for k, v in res.items()}},
                         indent=2, ensure_ascii=False))
        return 0
    print(f"Query: {a.query}\n{DISCLAIMER}\n")
    for k, rs in res.items():
        print(f"=== {('CIVIS policy documents' if k == 'civis' else 'Lok Sabha Q&A (pilot corpus)')} — ranked separately ===")
        if not rs:
            print("  (no keyword match)\n")
        for r in rs:
            print(f"  [{r.rank}] {r.title_or_question[:110]}\n      id: {r.record_id} | {r.score_type} {r.score}"
                  + (f" | {r.date} | {r.issuer_or_ministry}" if r.date else "")
                  + (f"\n      url: {r.url}" if r.url else "")
                  + f"\n      {r.snippet}\n      provenance: {r.provenance}"
                  + (f"\n      flags: {r.quality_flags}" if r.quality_flags else "")
                  + f"\n      licence: {r.licence_status}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
