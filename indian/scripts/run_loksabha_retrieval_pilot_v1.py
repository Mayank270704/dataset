"""
indian/scripts/run_loksabha_retrieval_pilot_v1.py
=================================================
Lok Sabha question-to-answer retrieval PILOT (local, offline, memory-bounded).

    python -I indian/scripts/run_loksabha_retrieval_pilot_v1.py sample     # deterministic pilot corpus + queries
    python -I indian/scripts/run_loksabha_retrieval_pilot_v1.py embed      # MiniLM answer embeddings (RAM-guarded)
    python -I indian/scripts/run_loksabha_retrieval_pilot_v1.py evaluate   # BM25, dense, hybrid -> metrics

Pilot design (fixed before results):
  * eligible = question and answer both non-empty (148,783 of 150,808)
  * corpus = 10,000 eligible records, stratified by Lok Sabha, seed 42; queries = 1,000 corpus records, seed 42
  * a query's relevant set = every corpus record whose normalised answer text equals its own answer
  * query records whose raw question contains an answer marker or the answer's opening text are EXCLUDED (leakage)
  * query text = cleaned question (no member names, number or 'Will the Minister of X be pleased to state');
    second query type = subject line
  * legacy-font Hindi records are indexed unchanged and flagged; metrics reported with and without them
Outputs: indian/processed/loksabha/retrieval_pilot_v1/ (write-once per file)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "indian"
sys.path.insert(0, str(IND))
sys.path.insert(0, str(ROOT / "scripts"))
from retrieval import civis as cv  # noqa: E402  (BM25, encoder, fusion, metrics)
from retrieval import loksabha as ls  # noqa: E402

PQ = IND / "processed" / "lok_sabha_qa_clean.parquet"
OUT = IND / "processed" / "loksabha" / "retrieval_pilot_v1"
N_CORPUS, N_QUERIES, SEED, MAX_LEN = 10_000, 1_000, 42, 256
ENC = "all-MiniLM-L6-v2"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def res() -> dict:
    import psutil
    return {"disk_free_mb": shutil.disk_usage(ROOT).free // 2**20, "ram_available_mb": psutil.virtual_memory().available // 2**20}


def stage_sample() -> int:
    if (OUT / "corpus.csv").exists():
        print("[SKIP] sample exists")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = ls.read_metadata(PQ)
    elig = m[~m.question_empty & ~m.answer_empty].copy()
    rng = np.random.default_rng(SEED)
    frac = N_CORPUS / len(elig)
    parts = []
    for lok, g in elig.groupby("lok_sabha"):
        k = int(round(len(g) * frac))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    corpus = pd.concat(parts).head(N_CORPUS).reset_index(drop=True)
    tx = ls.read_texts(PQ, set(corpus.qa_id))
    corpus = corpus.merge(tx, on="qa_id", how="left", validate="one_to_one")
    corpus["answer_key"] = corpus.answer_text.map(ls.answer_key)
    corpus["question_devanagari_share"] = corpus.question_text.map(ls.devanagari_share).round(3)
    corpus["answer_devanagari_share"] = corpus.answer_text.map(ls.devanagari_share).round(3)
    lk = [ls.leakage_flags(q, a) for q, a in zip(corpus.question_text, corpus.answer_text)]
    corpus = pd.concat([corpus, pd.DataFrame(lk)], axis=1)
    corpus["query_clean"] = [ls.clean_question(q, list(mb) if mb is not None else []) for q, mb in zip(corpus.question_text, corpus.members)]
    # queries: sample, then exclude leakage / empty cleaned query
    qi = np.sort(rng.choice(len(corpus), size=N_QUERIES, replace=False))
    qs = corpus.iloc[qi].copy()
    excl = qs.q_has_answer_marker | qs.q_contains_answer_head | (qs.query_clean.str.len() < 15)
    qs["excluded_reason"] = np.where(qs.q_has_answer_marker | qs.q_contains_answer_head, "answer leakage in question text",
                                     np.where(qs.query_clean.str.len() < 15, "cleaned question too short", ""))
    keep_cols = ["qa_id", "lok_sabha", "session", "question_no", "question_type", "date", "subject", "ministry", "pdf_url",
                 "legacy_hindi_font_artifacts", "answer_key", "question_devanagari_share", "answer_devanagari_share",
                 "q_has_answer_marker", "q_contains_answer_head", "q_shares_8gram_with_answer"]
    corpus[keep_cols].assign(licence_status=ls.LICENCE_STATUS).to_csv(OUT / "corpus.csv", index=False, lineterminator="\n")
    qs[["qa_id", "subject", "query_clean", "excluded_reason", "q_shares_8gram_with_answer", "question_devanagari_share",
        "legacy_hindi_font_artifacts"]].to_csv(OUT / "queries.csv", index=False, lineterminator="\n")
    dupkeys = corpus.answer_key.value_counts()
    info = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "indian/processed/lok_sabha_qa_clean.parquet",
            "source_sha256": sha(PQ), "records_total": int(len(m)), "eligible_q_and_a": int(len(elig)),
            "excluded_from_eligibility": {"question_empty": int(m.question_empty.sum()), "answer_empty": int(m.answer_empty.sum())},
            "corpus": int(len(corpus)), "corpus_by_lok_sabha": corpus.lok_sabha.value_counts().sort_index().to_dict(),
            "corpus_legacy_font_flagged": int(corpus.legacy_hindi_font_artifacts.sum()),
            "corpus_answer_text_groups_with_duplicates": int((dupkeys > 1).sum()), "corpus_records_in_duplicate_answer_groups": int(dupkeys[dupkeys > 1].sum()),
            "queries_sampled": N_QUERIES, "queries_excluded": qs.excluded_reason[qs.excluded_reason != ""].value_counts().to_dict(),
            "queries_used": int((~excl).sum()), "queries_sharing_8gram_with_answer": int(qs.q_shares_8gram_with_answer[~excl].sum()),
            "queries_devanagari_majority": int((qs.question_devanagari_share[~excl] > 0.5).sum()),
            "seed": SEED, "runtime_s": round(time.time() - t0, 1), "resources": res(), "licence_status": ls.LICENCE_STATUS}
    (OUT / "sample_manifest.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(info, default=str))
    return 0


def corpus_texts() -> pd.DataFrame:
    c = pd.read_csv(OUT / "corpus.csv", dtype={"qa_id": str}, keep_default_na=False)
    tx = ls.read_texts(PQ, set(c.qa_id), cols=("qa_id", "answer_text"))
    return c[["qa_id"]].merge(tx, on="qa_id", how="left", validate="one_to_one")


def stage_embed() -> int:
    ep = OUT / f"answer_embeddings_{ENC}.npy"
    if ep.exists():
        print("[SKIP] embeddings exist")
        return 0
    spec = cv.ENCODERS[ENC]
    w = cv.cached_weights(spec["model_id"])
    r = res()
    need = int(w.stat().st_size / 2**20 * 1.25) + 300 if w else None
    if w is None or r["ram_available_mb"] < need or r["disk_free_mb"] < 300:
        st = {"status": "skipped", "reason": "weights incomplete" if w is None else f"resources {r} below need ~{need} MB RAM / 300 MB disk"}
        (OUT / f"embedding_status_{ENC}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(st)
        return 4
    import torch
    from run_full_corpus_clustering import load_encoder_low_memory
    tok, model, _ = load_encoder_low_memory(spec["model_id"])
    torch.set_num_threads(6)
    a = corpus_texts()
    t0 = time.time()
    last = [t0]

    def prog(n):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"  {n}/{len(a)} answers, {n / (time.time() - t0):.1f}/s", flush=True)
    E, trunc = cv.encode(a.answer_text.tolist(), tok, model, max_length=MAX_LEN, batch_size=16, progress=prog)
    np.save(ep, E.astype(np.float16))
    (OUT / f"answer_embeddings_{ENC}.json").write_text(json.dumps({
        "encoder": spec["model_id"], "dim": spec["dim"], "rows": len(a), "unit": f"answer text, first {MAX_LEN} tokens (rest truncated)",
        "answers_truncated": trunc, "encode_s": round(time.time() - t0, 1), "corpus_sha256": sha(OUT / "corpus.csv"),
        "embeddings_sha256": sha(ep), "embeddings_mb": round(ep.stat().st_size / 2**20, 2), "resources_before": r, "resources_after": res()}, indent=2) + "\n", encoding="utf-8")
    print(f"[EMBED] {E.shape} in {time.time() - t0:.0f}s; truncated {trunc}")
    return 0


def stage_evaluate() -> int:
    ev = OUT / "evaluation"
    if (ev / "metrics.json").exists():
        print("[SKIP] evaluation exists")
        return 0
    ev.mkdir(exist_ok=True)
    t0 = time.time()
    c = pd.read_csv(OUT / "corpus.csv", dtype={"qa_id": str}, keep_default_na=False)
    q = pd.read_csv(OUT / "queries.csv", dtype={"qa_id": str}, keep_default_na=False)
    q = q[q.excluded_reason == ""].reset_index(drop=True)
    a = corpus_texts()
    assert a.qa_id.tolist() == c.qa_id.tolist()
    idx = {k: i for i, k in enumerate(c.qa_id)}
    by_key = c.groupby("answer_key").indices
    rel = [set(by_key[c.answer_key[idx[x]]]) for x in q.qa_id]
    tb = time.time()
    bm = cv.BM25().fit(a.answer_text.tolist())
    timings = {"bm25_fit_s": round(time.time() - tb, 1)}
    del a
    qtexts = {"question_clean": q.query_clean.tolist(), "subject": q.subject.tolist()}
    scores = {qt: {"bm25": [bm.score(t) for t in txts]} for qt, txts in qtexts.items()}
    ep = OUT / f"answer_embeddings_{ENC}.npy"
    dense_note = "not run: no embeddings (see embedding_status_*.json)"
    if ep.exists():
        meta = json.loads((OUT / f"answer_embeddings_{ENC}.json").read_text(encoding="utf-8"))
        if meta["embeddings_sha256"] == sha(ep) and meta["corpus_sha256"] == sha(OUT / "corpus.csv"):
            import torch
            from run_full_corpus_clustering import load_encoder_low_memory
            tok, model, _ = load_encoder_low_memory(cv.ENCODERS[ENC]["model_id"])
            torch.set_num_threads(6)
            E = np.load(ep).astype(np.float32)
            for qt, txts in qtexts.items():
                Q, _ = cv.encode(txts, tok, model, max_length=MAX_LEN, batch_size=32)
                S = Q @ E.T
                scores[qt][f"dense_{ENC}"] = list(S)
                orders = [(cv.rank_order(b), cv.rank_order(d)) for b, d in zip(scores[qt]["bm25"], S)]
                scores[qt][f"hybrid_rrf_{ENC}"] = [cv.rrf(list(o), len(c)) for o in orders]
                scores[qt][f"hybrid_minmax_{ENC}"] = [cv.minmax_fusion(b, d, 0.5) for b, d in zip(scores[qt]["bm25"], S)]
            dense_note = {"encoder": meta["encoder"], "dim": meta["dim"], "unit": meta["unit"], "answers_truncated": meta["answers_truncated"]}
            del E, model
        else:
            dense_note = "not run: embedding/corpus hash mismatch"
    subsets = {"all": np.ones(len(q), bool),
               "excluding_legacy_font": ~q.legacy_hindi_font_artifacts.astype(str).eq("True").to_numpy(),
               "excluding_8gram_overlap": ~q.q_shares_8gram_with_answer.astype(str).eq("True").to_numpy(),
               "english_question_only": (q.question_devanagari_share.astype(float) <= 0.5).to_numpy(),
               "devanagari_question_only": (q.question_devanagari_share.astype(float) > 0.5).to_numpy()}
    ranks = pd.DataFrame({"qa_id": q.qa_id})
    metrics = {}
    for qt, ms in scores.items():
        metrics[qt] = {}
        for m, sl in ms.items():
            rk = np.array([cv.first_relevant_rank(cv.rank_order(s), r) for s, r in zip(sl, rel)])
            ranks[f"rank_{qt}_{m}"] = rk
            metrics[qt][m] = {name: (cv.metrics_from_ranks(rk[mask]) if mask.sum() else {"queries": 0}) for name, mask in subsets.items()}
    ranks.to_csv(ev / "per_query_ranks.csv", index=False, lineterminator="\n")
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "task": "known-item question->answer retrieval over a 10,000-record pilot corpus; relevant = records with identical normalised answer text",
           "corpus": len(c), "queries_used": len(q), "subset_sizes": {k: int(v.sum()) for k, v in subsets.items()},
           "dense": dense_note, "fusion": {"rrf_k": 60, "minmax_alpha": 0.5, "note": "fixed a priori, not tuned"},
           "metrics": metrics, "timings": timings | {"evaluate_total_s": round(time.time() - t0, 1)}, "resources_after": res(),
           "licence_status": ls.LICENCE_STATUS, "inputs_sha256": {"corpus.csv": sha(OUT / "corpus.csv"), "queries.csv": sha(OUT / "queries.csv")}}
    (ev / "metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    for qt in metrics:
        for m in metrics[qt]:
            x = metrics[qt][m]["all"]
            print(f"{qt:15} {m:32} R@1 {x['recall@1']:.3f} R@5 {x['recall@5']:.3f} R@10 {x['recall@10']:.3f} MRR@10 {x['mrr@10']:.3f}")
    print(json.dumps(out["subset_sizes"]), json.dumps(out["timings"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit({"sample": stage_sample, "embed": stage_embed, "evaluate": stage_evaluate}[sys.argv[1]]())
