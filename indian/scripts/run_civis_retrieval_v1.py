"""
indian/scripts/run_civis_retrieval_v1.py
========================================
CIVIS policy-document retrieval experiment v1 (local, offline). See indian/reports/multi_dataset_research_plan.md §B.

Stages (each artifact is write-once; existing files are reused, never overwritten):
    index     canonical documents, chunks (offsets + hashes, no copied text), preflight checks
    embed     dense chunk embeddings per encoder (float16 .npy); skipped with a recorded reason if the encoder's
              weights are incomplete or RAM is insufficient
    evaluate  BM25, dense and hybrid known-item retrieval with title / summary queries -> metrics + per-query ranks

    python -I indian/scripts/run_civis_retrieval_v1.py index
    python -I indian/scripts/run_civis_retrieval_v1.py embed --encoder all-MiniLM-L6-v2
    python -I indian/scripts/run_civis_retrieval_v1.py embed --encoder e5-base-v2
    python -I indian/scripts/run_civis_retrieval_v1.py evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                    # project root
IND = ROOT / "indian"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(IND))
sys.path.insert(0, str(ROOT / "scripts"))                       # existing low-memory encoder loader

from retrieval import civis as cv  # noqa: E402

SRC = IND / "processed" / "civis_policy_documents_clean.csv"
OUT = IND / "processed" / "retrieval" / "civis_v1"
CHUNK_SIZE, CHUNK_OVERLAP, MAX_LEN = 1000, 100, 256
MIN_FREE_DISK_MB = 300


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json_new(p: Path, obj) -> None:
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resources() -> dict:
    import psutil
    return {"disk_free_mb": shutil.disk_usage(ROOT).free // 2**20, "ram_available_mb": psutil.virtual_memory().available // 2**20}


def load_docs():
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    docs, mapping, excluded = cv.canonical_documents(df)
    return df, docs, mapping, excluded


# ── index ─────────────────────────────────────────────────────────────────────

def stage_index() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "chunks.csv").exists():
        print("[SKIP] index exists")
        return 0
    t0 = time.time()
    res = resources()
    if res["disk_free_mb"] < MIN_FREE_DISK_MB:
        print(f"[STOP] only {res['disk_free_mb']} MB disk free", file=sys.stderr)
        return 3
    df, docs, mapping, excluded = load_docs()
    rows, chunks = [], []
    for d in docs:
        spans = cv.chunk_text(d.text, CHUNK_SIZE, CHUNK_OVERLAP)
        rows.append({"doc_key": d.doc_key, "member_record_ids": ";".join(d.member_ids), "titles": " || ".join(d.titles),
                     "splits": ";".join(d.splits), "char_count": len(d.text), "n_chunks": len(spans),
                     "text_sha256": cv.sha_text(d.text), "content_type": "policy/consultation document (not citizen comments)",
                     "summary_note": cv.SUMMARY_NOTE, "source_file": "indian/processed/civis_policy_documents_clean.csv",
                     "licence_status": cv.LICENCE_STATUS})
        for i, (a, b) in enumerate(spans):
            chunks.append({"chunk_id": f"{d.doc_key}#{i:05d}", "doc_key": d.doc_key, "chunk_no": i, "start": a, "end": b,
                           "chunk_sha256": cv.sha_text(d.text[a:b])})
    meta_only = df[df.civis_record_id.isin(excluded)][["civis_record_id", "split", "title"]].assign(
        status="metadata only: empty document text, not indexed", summary_note=cv.SUMMARY_NOTE, licence_status=cv.LICENCE_STATUS)
    pd.DataFrame(rows).to_csv(OUT / "documents.csv", index=False, lineterminator="\n")
    meta_only.to_csv(OUT / "documents_metadata_only.csv", index=False, lineterminator="\n")
    pd.DataFrame(chunks).to_csv(OUT / "chunks.csv", index=False, lineterminator="\n")
    write_json_new(OUT / "index_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "indian/processed/civis_policy_documents_clean.csv",
        "source_sha256": sha(SRC), "records": len(df), "excluded_empty_records": excluded,
        "indexed_canonical_documents": len(docs), "records_covered_by_index": len(mapping),
        "identical_text_groups": {d.doc_key: d.member_ids for d in docs if len(d.member_ids) > 1},
        "chunking": {"size_chars": CHUNK_SIZE, "overlap_chars": CHUNK_OVERLAP, "boundary": "whitespace", "chunks": len(chunks),
                     "note": "chunk text is not copied; reconstruct from the source CSV with start/end offsets"},
        "resources_at_start": res, "runtime_s": round(time.time() - t0, 2), "licence_status": cv.LICENCE_STATUS})
    print(f"[INDEX] {len(docs)} canonical documents ({len(mapping)} records), {len(chunks)} chunks; excluded {excluded}")
    return 0


# ── embed ─────────────────────────────────────────────────────────────────────

def stage_embed(name: str) -> int:
    spec = cv.ENCODERS[name]
    emb_p, meta_p = OUT / f"chunk_embeddings_{name}.npy", OUT / f"chunk_embeddings_{name}.json"
    if emb_p.exists():
        print(f"[SKIP] {emb_p.name} exists")
        return 0
    w = cv.cached_weights(spec["model_id"])
    res = resources()
    status = {"encoder": name, "model_id": spec["model_id"], "resources_before": res}
    if w is None:
        status["status"] = "skipped: no complete local weights (no download attempted)"
    else:
        need = int(w.stat().st_size / 2**20 * 1.25) + 200
        status |= {"weights_file": str(w), "weights_mb": w.stat().st_size // 2**20, "ram_needed_mb_estimate": need}
        if res["ram_available_mb"] < need:
            status["status"] = f"skipped: {res['ram_available_mb']} MB RAM available < ~{need} MB needed"
    if "status" in status:
        print(f"[EMBED] {name}: {status['status']}")
        p = OUT / f"embedding_status_{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        write_json_new(p, status)
        return 4
    import torch
    from run_full_corpus_clustering import load_encoder_low_memory     # existing low-memory loader (scripts/)
    t0 = time.time()
    tok, model, path = load_encoder_low_memory(spec["model_id"])
    torch.set_num_threads(6)                                            # after weights are read (see loader note)
    load_s = time.time() - t0
    _, docs, _, _ = load_docs()
    text = {d.doc_key: d.text for d in docs}
    ch = pd.read_csv(OUT / "chunks.csv", keep_default_na=False)
    passages = [spec["passage_prefix"] + text[k][a:b] for k, a, b in zip(ch.doc_key, ch.start, ch.end)]
    t1 = time.time()
    last = [t1]

    def prog(n):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"  {n}/{len(passages)} chunks, {n / (time.time() - t1):.1f}/s", flush=True)
    emb, trunc = cv.encode(passages, tok, model, max_length=MAX_LEN, batch_size=16, progress=prog)
    enc_s = time.time() - t1
    if emb.shape != (len(ch), spec["dim"]):
        raise ValueError(f"unexpected embedding shape {emb.shape}")
    np.save(emb_p, emb.astype(np.float16))
    write_json_new(meta_p, status | {"status": "ok", "dim": spec["dim"], "dtype_saved": "float16", "rows": len(ch),
                                     "chunks_sha256": sha(OUT / "chunks.csv"), "max_length_tokens": MAX_LEN,
                                     "chunks_truncated_at_max_length": trunc, "passage_prefix": spec["passage_prefix"],
                                     "load_s": round(load_s, 1), "encode_s": round(enc_s, 1), "chunks_per_s": round(len(ch) / enc_s, 2),
                                     "embeddings_sha256": sha(emb_p), "embeddings_mb": round(emb_p.stat().st_size / 2**20, 2),
                                     "model_licence": spec["licence"], "resources_after": resources()})
    print(f"[EMBED] {name}: {emb.shape} in {enc_s:.0f}s; truncated {trunc}")
    return 0


# ── evaluate ──────────────────────────────────────────────────────────────────

def build_queries(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    q = []
    for _, r in df.iterrows():
        if r.civis_record_id not in mapping:
            continue                                                    # empty-text record: no indexed target
        tgt = mapping[r.civis_record_id]
        q.append({"query_id": f"title::{r.civis_record_id}", "query_type": "title", "record_id": r.civis_record_id, "target_doc_key": tgt, "query": r.title})
        q.append({"query_id": f"summary_first_sentence::{r.civis_record_id}", "query_type": "summary_first_sentence", "record_id": r.civis_record_id,
                  "target_doc_key": tgt, "query": cv.first_sentence(r.reference_summary)})
        q.append({"query_id": f"summary_full::{r.civis_record_id}", "query_type": "summary_full", "record_id": r.civis_record_id,
                  "target_doc_key": tgt, "query": r.reference_summary})
    return pd.DataFrame(q)


def stage_evaluate() -> int:
    ev = OUT / "evaluation"
    if (ev / "metrics.json").exists():
        print("[SKIP] evaluation exists")
        return 0
    ev.mkdir(exist_ok=True)
    t0 = time.time()
    df, docs, mapping, excluded = load_docs()
    ch = pd.read_csv(OUT / "chunks.csv", keep_default_na=False)
    keys = [d.doc_key for d in docs]
    kidx = {k: i for i, k in enumerate(keys)}
    cdoc = ch.doc_key.map(kidx).to_numpy()
    text = {d.doc_key: d.text for d in docs}
    ctext = [text[k][a:b] for k, a, b in zip(ch.doc_key, ch.start, ch.end)]
    nchunks = np.bincount(cdoc, minlength=len(docs))
    queries = build_queries(df, mapping)
    tgt = queries.target_doc_key.map(kidx).to_numpy()

    tb = time.time()
    bm = cv.BM25().fit(ctext)
    bm_fit_s = time.time() - tb
    scores: dict[str, list[np.ndarray]] = {"bm25": []}
    for qt in queries["query"]:
        scores["bm25"].append(cv.doc_scores_from_chunks(bm.score(qt), cdoc, len(docs)))
    timings = {"bm25_fit_s": round(bm_fit_s, 2)}

    encoders_used, encoder_notes = [], {}
    for name, spec in cv.ENCODERS.items():
        ep, mp = OUT / f"chunk_embeddings_{name}.npy", OUT / f"chunk_embeddings_{name}.json"
        if not ep.exists():
            encoder_notes[name] = "not evaluated: no embeddings (see embedding_status_*.json)"
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta["embeddings_sha256"] != sha(ep) or meta["chunks_sha256"] != sha(OUT / "chunks.csv"):
            encoder_notes[name] = "not evaluated: embeddings/chunks hash mismatch"
            continue
        import torch
        from run_full_corpus_clustering import load_encoder_low_memory
        tok, model, _ = load_encoder_low_memory(spec["model_id"])
        torch.set_num_threads(6)
        E = np.load(ep).astype(np.float32)
        te = time.time()
        Q, qtrunc = cv.encode([spec["query_prefix"] + x for x in queries["query"]], tok, model, max_length=MAX_LEN, batch_size=16)
        timings[f"{name}_query_encode_s"] = round(time.time() - te, 1)
        del model, tok
        S = Q @ E.T
        scores[f"dense_{name}"] = [cv.doc_scores_from_chunks(S[i], cdoc, len(docs)) for i in range(len(queries))]
        orders_b = [cv.rank_order(s) for s in scores["bm25"]]
        orders_d = [cv.rank_order(s) for s in scores[f"dense_{name}"]]
        scores[f"hybrid_rrf_{name}"] = [cv.rrf([ob, od], len(docs)) for ob, od in zip(orders_b, orders_d)]
        scores[f"hybrid_minmax_{name}"] = [cv.minmax_fusion(b, d, 0.5) for b, d in zip(scores["bm25"], scores[f"dense_{name}"])]
        encoders_used.append(name)
        encoder_notes[name] = {"dim": spec["dim"], "query_prefix": spec["query_prefix"], "queries_truncated": qtrunc}
        del E, S, Q

    ranks = pd.DataFrame({"query_id": queries.query_id, "query_type": queries.query_type, "record_id": queries.record_id,
                          "target_doc_key": queries.target_doc_key})
    top1 = {}
    for m, sl in scores.items():
        orders = [cv.rank_order(s) for s in sl]
        ranks[f"rank_{m}"] = [cv.first_relevant_rank(o, {t}) for o, t in zip(orders, tgt)]
        top1[m] = np.array([o[0] for o in orders])
    metrics = {}
    for qt in ("title", "summary_first_sentence", "summary_full"):
        sel = (ranks.query_type == qt).to_numpy()
        metrics[qt] = {}
        for m in scores:
            r = ranks.loc[sel, f"rank_{m}"].to_numpy()
            wrong = sel & (ranks[f"rank_{m}"].to_numpy() > 1)
            metrics[qt][m] = cv.metrics_from_ranks(r) | {
                "wrong_top1_median_chunks": float(np.median(nchunks[top1[m][wrong]])) if wrong.any() else None}

    # leakage diagnostics for self-generated queries
    title_rows = queries[queries.query_type == "title"]
    in_text = [str(t).lower().strip() in text[k].lower() for t, k in zip(title_rows["query"], title_rows.target_doc_key)]
    def overlap(q, d):
        qt = set(bm.vec.build_analyzer()(q))
        return len(qt & set(bm.vec.build_analyzer()(d))) / max(1, len(qt))
    fs = queries[queries.query_type == "summary_first_sentence"]
    full = queries[queries.query_type == "summary_full"]
    diag = {"title_verbatim_in_target_text_share": round(float(np.mean(in_text)), 4),
            "summary_first_sentence_term_overlap_with_target_median": round(float(np.median([overlap(q, text[k]) for q, k in zip(fs["query"], fs.target_doc_key)])), 4),
            "summary_full_term_overlap_with_target_median": round(float(np.median([overlap(q, text[k]) for q, k in zip(full["query"], full.target_doc_key)])), 4),
            "doc_chunks_median": float(np.median(nchunks)), "doc_chunks_max": int(nchunks.max())}
    ranks.to_csv(ev / "per_query_ranks.csv", index=False, lineterminator="\n")
    queries.to_csv(ev / "queries.csv", index=False, lineterminator="\n")
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "task": "known-item retrieval: each query's relevant item is its own source document (identical-text members share one canonical target)",
           "indexed_documents": len(docs), "chunks": len(ch), "queries_per_type": {k: int(v) for k, v in queries.query_type.value_counts().items()},
           "excluded_empty_records": excluded, "methods": list(scores), "encoders_used": encoders_used, "encoder_notes": encoder_notes,
           "fusion": {"rrf_k": 60, "minmax_alpha": 0.5, "note": "fixed a priori; not tuned on these queries"},
           "metrics": metrics, "leakage_diagnostics": diag, "timings": timings | {"evaluate_total_s": round(time.time() - t0, 1)},
           "resources_after": resources(), "licence_status": cv.LICENCE_STATUS,
           "inputs_sha256": {"source": sha(SRC), "chunks.csv": sha(OUT / "chunks.csv")}}
    write_json_new(ev / "metrics.json", out)
    print(json.dumps({qt: {m: {k: v[k] for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")} for m, v in d.items()} for qt, d in metrics.items()}, indent=1))
    print(json.dumps(diag))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["index", "embed", "evaluate"])
    ap.add_argument("--encoder", choices=list(cv.ENCODERS))
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.stage == "index":
        return stage_index()
    if a.stage == "embed":
        if not a.encoder:
            ap.error("--encoder required")
        return stage_embed(a.encoder)
    return stage_evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
