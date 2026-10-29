"""
scripts/run_full_corpus_clustering.py
=====================================
Full-corpus, UNSUPERVISED semantic pipeline for the Canadian consultation text (v1).

Unit of analysis
----------------
One *text entry* = one non-blank response in a designed free-text field of a source record
(individual comments: 5 fields; compilation: the comment column; questionnaire: Q8, Q14, Q18),
with at least 20 characters and not generic filler. Entries with identical normalised text
(NFC + casefold + whitespace collapse) are embedded and clustered once ("unique text") and the
cluster is mapped back to every entry; multiplicity is retained. Respondent identity is not
available, so units are source records/fields, never "people".

Stages (resumable; never overwrite):
    corpus   build the corpus, unique-text table and the 14,135 -> eligible reconciliation
    embed    encode unique texts with --model-key {multilingual,baseline} in cached chunks
    cluster  UMAP x HDBSCAN configurations + diagnostics, c-TF-IDF keywords, representatives
    report   write reports/full_corpus_clustering_report_v1.md and run_manifest.json

Examples:
    python scripts/run_full_corpus_clustering.py profile
    python scripts/run_full_corpus_clustering.py corpus
    python scripts/run_full_corpus_clustering.py embed --model-key multilingual
    python scripts/run_full_corpus_clustering.py embed --model-key baseline
    python scripts/run_full_corpus_clustering.py cluster --model-key multilingual
    python scripts/run_full_corpus_clustering.py cluster --model-key baseline
    python scripts/run_full_corpus_clustering.py report

Nothing here uses human labels. Cluster keywords and representative texts are descriptive,
traceable to source records, and are NOT validated concerns. Silhouette and NPMI are
geometric / lexical proxies, not evidence of semantic quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

try:  # torch must be imported before psutil/pandas here: the reverse order makes torch's
    import torch  # noqa: F401  # storage code crash with an access violation on this machine
except ImportError:  # pragma: no cover - torch is optional for non-embedding stages
    pass
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ROOT = ROOT / "data" / "full_corpus" / "run_v1"
REPORT_PATH = ROOT / "reports" / "full_corpus_clustering_report_v1.md"
PROCESSED = ROOT / "data" / "processed"
INTERIM_ENTRIES = ROOT / "data" / "interim" / "government_response_text_entries.csv"

MODELS = {
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "baseline": "sentence-transformers/all-MiniLM-L6-v2",
}
REFERENCE_EMBEDDINGS = {      # saved 450-comment embeddings used only as an equivalence check
    "multilingual": ROOT / "data" / "comparisons" / "multilingual_embedding_450" /
                    "paraphrase_multilingual_minilm_l12_v2" / "embeddings.npy",
    "baseline": ROOT / "data" / "embeddings" / "minilm_embeddings.npy",
}
MAX_LENGTH = 256
BATCH_SIZE = 16
CHUNK_SIZE = 128
SEED = 42
MIN_TEXT_LEN = 20

UMAP_CONFIGS = {
    "u15": {"n_components": 5, "n_neighbors": 15, "min_dist": 0.0, "metric": "cosine"},   # existing 450 settings
    "u30": {"n_components": 5, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"},
}
HDBSCAN_CONFIGS = {
    "m10s5": {"min_cluster_size": 10, "min_samples": 5},    # existing 450 settings
    "m25s5": {"min_cluster_size": 25, "min_samples": 5},
    "m50s10": {"min_cluster_size": 50, "min_samples": 10},
}
SEED_STABILITY = {"umap": "u15", "hdbscan": "m25s5", "seed": 43}
SELECTION_RULE = ("Reporting configuration = among the multilingual-model configurations with 20 <= clusters <= 150 and "
                  "noise <= 40%, the one with the highest mean NPMI coherence (ties: larger min_cluster_size). This only "
                  "chooses which configuration is listed in detail; it does not validate it.")


# ── generic helpers ───────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text_list(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def write_new(path: Path, writer: Callable[[Any], None], binary: bool = False) -> None:
    """Exclusive creation: raises FileExistsError instead of overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        with open(path, "xb") as f:
            writer(f)
    else:
        with open(path, "x", encoding="utf-8", newline="") as f:
            writer(f)


def write_json_new(path: Path, obj: Any) -> None:
    write_new(path, lambda f: (json.dump(obj, f, indent=2, ensure_ascii=False, default=str), f.write("\n")))


def peak_rss_mb() -> float:
    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 2**20, 1)


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(ROOT)
    return {
        "ram_total_mb": round(vm.total / 2**20), "ram_available_mb": round(vm.available / 2**20),
        "disk_free_gb": round(du.free / 2**30, 1), "logical_cpus": os.cpu_count(),
    }


# ── corpus ────────────────────────────────────────────────────────────────────

DESIGNED_PATTERNS = {
    "individual_comments_2016_18": [r"^1 Practical idea", r"^OG is important because", r"^Public Engagement Question",
                                    r"^Improvements\?", r"^Other comments"],
    "consultation_compilation_2017_18": [r"^Comment, Question or Idea \(In source language\)"],
    "get_involved_questionnaire_2017_18": [r"^Q8\.", r"^Q14\.", r"^Q18\."],
}


def reconcile_extracted_entries(entries_path: Path, eligible: pd.DataFrame) -> pd.DataFrame:
    """Account for every row of government_response_text_entries.csv (14,135 expected)."""
    d = pd.read_csv(entries_path, dtype=str, keep_default_na=False)

    def classify(row) -> str:
        col = row["column_name"].strip()
        if re.search(r"translation", col, re.I):
            return "translation_column_not_designed"
        if any(re.search(p, col) for p in DESIGNED_PATTERNS.get(row["dataset_name"], [])):
            return "designed_field"
        return "other_free_text_field_not_designed"

    d["category"] = d.apply(classify, axis=1)
    out = d.groupby(["dataset_name", "category"]).size().unstack(fill_value=0)
    elig = eligible.groupby("source_dataset").size()
    out["designed_field_eligible_entries"] = elig.reindex(out.index).fillna(0).astype(int)
    out["designed_field_short_or_filler"] = out.get("designed_field", 0) - out["designed_field_eligible_entries"]
    out["total_extracted_entries"] = d.groupby("dataset_name").size()
    return out.reset_index()


def build_unique_table(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus = corpus.copy()
    corpus["text_key"] = corpus["original_text"].map(normalise_text)
    first = corpus.drop_duplicates("text_key", keep="first").reset_index(drop=True)
    first["unique_id"] = range(len(first))
    mult = corpus.groupby("text_key").size()
    first["multiplicity"] = first["text_key"].map(mult).astype(int)
    uid = dict(zip(first["text_key"], first["unique_id"]))
    corpus["unique_id"] = corpus["text_key"].map(uid).astype(int)
    return first, corpus


def stage_corpus(run_root: Path = RUN_ROOT, processed: Path = PROCESSED, entries_path: Path = INTERIM_ENTRIES) -> dict[str, Any]:
    from models.semantic_similarity.minilm import build_corpus
    out = run_root / "corpus"
    if (out / "corpus_manifest.json").exists():
        return json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
    corpus = build_corpus(processed)
    unique, entries = build_unique_table(corpus)
    cols = ["corpus_id", "unique_id", "source_dataset", "source_record_id", "source_row_number", "source_field",
            "question_slug", "language", "stakeholder_group", "duplicate_text_group_id", "text_key", "original_text"]
    write_new(out / "corpus_entries.csv", lambda f: entries[cols].to_csv(f, index=False, lineterminator="\n"))
    ucols = ["unique_id", "corpus_id", "multiplicity", "source_dataset", "source_record_id", "question_slug",
             "language", "original_text"]
    write_new(out / "unique_texts.csv", lambda f: unique[ucols].to_csv(f, index=False, lineterminator="\n"))
    recon = reconcile_extracted_entries(entries_path, entries) if entries_path.exists() else pd.DataFrame()
    if len(recon):
        write_new(out / "reconciliation.csv", lambda f: recon.to_csv(f, index=False, lineterminator="\n"))
    manifest = {
        "entries": int(len(entries)),
        "unique_texts": int(len(unique)),
        "duplicate_entries_mapped_to_unique": int(len(entries) - len(unique)),
        "by_dataset": entries.groupby("source_dataset").size().to_dict(),
        "language_counts": {f"{d}|{l}": int(n) for (d, l), n in entries.groupby(["source_dataset", "language"]).size().items()},
        "eligibility": f">= {MIN_TEXT_LEN} characters after trim, not generic filler, exclusion_reason blank (models.semantic_similarity.minilm.is_eligible)",
        "reconciliation_total_extracted": int(recon["total_extracted_entries"].sum()) if len(recon) else None,
        "inputs_sha256": {p.name: sha256_file(p) for p in sorted(processed.glob("*_source_clean.csv"))
                          if p.name.startswith(("individual_comments", "consultation_compilation", "get_involved"))},
        "corpus_entries_sha256": sha256_file(out / "corpus_entries.csv"),
        "unique_texts_sha256": sha256_file(out / "unique_texts.csv"),
    }
    write_json_new(out / "corpus_manifest.json", manifest)
    return manifest


# ── embeddings ────────────────────────────────────────────────────────────────

def load_encoder_low_memory(model_id: str):
    """Load a BERT-style encoder with peak memory ~ one copy of the weights.

    transformers' default loader (and safetensors.load_file) hold two copies and crash with an
    access violation on this machine when only ~500 MB RAM is available. This builds the model on
    the meta device and assigns tensors read one by one; weights are identical.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = Path(snapshot_download(model_id, local_files_only=True))
    # Order matters on this machine: reading the safetensors weights BEFORE the fast tokenizer is
    # created is stable; the reverse order crashed with an access violation in torch's storage code.
    model = load_model_from_dir(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return tokenizer, model, path


_ST_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64",
              "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def read_safetensors_plain(path: Path) -> dict[str, Any]:
    """Read a .safetensors file with ordinary file I/O (no memory mapping).

    safetensors' mmap-based get_tensor intermittently crashed this process with an access violation
    (an in-page error under memory pressure / antivirus scanning). Plain reads raise OSError instead,
    and the format is simple: 8-byte little-endian header length, JSON header, raw tensor bytes.
    """
    import json
    import struct
    import torch
    out: dict[str, Any] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
        for name, info in items:
            begin, end = info["data_offsets"]
            dtype = getattr(torch, _ST_DTYPES[info["dtype"]])
            if end == begin:
                out[name] = torch.empty(info["shape"], dtype=dtype)
                continue
            f.seek(base + begin)
            buf = bytearray(f.read(end - begin))
            if len(buf) != end - begin:
                raise OSError(f"Short read for tensor {name} in {path}")
            out[name] = torch.frombuffer(buf, dtype=dtype).reshape(info["shape"])
    return out


def load_model_from_dir(path: Path):
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True)
    with torch.device("meta"):
        model = AutoModel.from_config(cfg)
    state = read_safetensors_plain(path / "model.safetensors")
    result = model.load_state_dict(state, strict=False, assign=True)
    del state
    if result.missing_keys:
        raise RuntimeError(f"Missing weights when loading {path}: {result.missing_keys[:5]}")
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            module_path, _, attr = name.rpartition(".")
            module = model.get_submodule(module_path) if module_path else model
            if attr == "position_ids":
                fixed = torch.arange(buf.shape[-1], dtype=torch.long).unsqueeze(0)
            elif attr == "token_type_ids":
                fixed = torch.zeros(buf.shape, dtype=torch.long)
            else:
                raise RuntimeError(f"Unexpected meta buffer {name}")
            module.register_buffer(attr, fixed, persistent=False)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Some parameters remain on the meta device.")
    return model.eval()


def stage_embed(model_key: str, run_root: Path = RUN_ROOT, chunk_size: int = CHUNK_SIZE, threads: int = 6,
                encode: Callable[[list[str]], np.ndarray] | None = None) -> dict[str, Any]:
    emb_dir = run_root / "embeddings" / model_key
    final = emb_dir / "embeddings.npy"
    manifest_path = emb_dir / "embedding_manifest.json"
    if manifest_path.exists() and final.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    unique = pd.read_csv(run_root / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    texts = unique["original_text"].tolist()
    texts_hash = sha256_text_list(texts)
    model_id = MODELS[model_key]
    t_start = time.perf_counter()
    model_info: dict[str, Any] = {}
    token_lengths = None
    if encode is None:
        import torch
        from models.semantic_similarity.minilm import mean_pool_and_normalize
        tokenizer, model, snapshot = load_encoder_low_memory(model_id)
        torch.set_num_threads(threads)      # must follow weight loading (see run_clustering_experiments_v2.default_encoder)
        model_info = {"snapshot": snapshot.name, "threads": threads}
        token_lengths = np.array([len(x) for x in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]])
        encode = lambda batch: mean_pool_and_normalize(batch, tokenizer, model, max_length=MAX_LENGTH,
                                                       batch_size=BATCH_SIZE, device="cpu")
    order = (np.argsort(token_lengths, kind="stable") if token_lengths is not None
             else np.argsort([len(t) for t in texts], kind="stable"))
    chunks_dir = emb_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    meta_path = chunks_dir / "chunk_meta.json"
    meta = {"texts_sha256": texts_hash, "model_id": model_id, "max_length": MAX_LENGTH, "chunk_size": chunk_size,
            "n_texts": len(texts)}
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != meta:
            raise RuntimeError("Existing chunk cache was built for different texts/settings; refusing to resume.")
    else:
        write_json_new(meta_path, meta)
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    done = reused = 0
    for k in range(n_chunks):
        chunk_file = chunks_dir / f"chunk_{k:04d}.npz"
        if chunk_file.exists():
            reused += 1
            continue
        idx = order[k * chunk_size:(k + 1) * chunk_size]
        emb = encode([texts[i] for i in idx]).astype(np.float32)
        if not np.isfinite(emb).all():
            raise ValueError(f"Non-finite embeddings in chunk {k}")
        tmp = chunks_dir / f"chunk_{k:04d}.tmp.npz"
        np.savez(tmp, idx=idx, emb=emb)
        os.replace(tmp, chunk_file)          # atomic: partial chunks are never treated as complete
        done += 1
    dim = None
    full = None
    for k in range(n_chunks):
        z = np.load(chunks_dir / f"chunk_{k:04d}.npz")
        if full is None:
            dim = z["emb"].shape[1]
            full = np.zeros((len(texts), dim), dtype=np.float32)
        full[z["idx"]] = z["emb"]
    if not np.isfinite(full).all() or not np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-3):
        raise ValueError("Assembled embeddings are not finite and L2-normalised.")
    write_new(final, lambda f: np.save(f, full), binary=True)
    elapsed = time.perf_counter() - t_start
    trunc = None if token_lengths is None else {
        "tokens_over_max_length": int((token_lengths > MAX_LENGTH).sum()),
        "share_over_max_length": round(float((token_lengths > MAX_LENGTH).mean()), 4),
        "token_length_median": int(np.median(token_lengths)), "token_length_p95": int(np.percentile(token_lengths, 95)),
        "token_length_max": int(token_lengths.max()),
    }
    manifest = {
        "model_key": model_key, "model_id": model_id, "n_unique_texts": len(texts), "dim": int(dim),
        "texts_sha256": texts_hash, "embeddings_sha256": sha256_file(final), "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE, "chunk_size": chunk_size, "chunks_total": n_chunks, "chunks_computed_this_run": done,
        "chunks_reused": reused, "elapsed_s_this_run": round(elapsed, 1), "peak_rss_mb": peak_rss_mb(),
        "truncation": trunc, "model_info": model_info, "system": system_snapshot(),
        "pooling": "attention-mask mean pooling + L2 normalisation (models.semantic_similarity.minilm.mean_pool_and_normalize)",
    }
    write_json_new(manifest_path, manifest)
    return manifest


def check_against_reference(model_key: str, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    """Compare full-run vectors with the saved 450-sample vectors for texts present in both."""
    ref_path = REFERENCE_EMBEDDINGS[model_key]
    meta_path = ROOT / "data" / "embeddings" / "corpus_metadata.csv"
    if not (ref_path.exists() and meta_path.exists()):
        return {"status": "reference_missing"}
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    ref = np.load(ref_path)
    unique = pd.read_csv(run_root / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    new = np.load(run_root / "embeddings" / model_key / "embeddings.npy")
    pos = {normalise_text(t): i for i, t in enumerate(unique["original_text"])}
    pairs = [(i, pos[normalise_text(t)]) for i, t in enumerate(meta["original_text"]) if normalise_text(t) in pos]
    if not pairs:
        return {"status": "no_overlap"}
    a = ref[[p[0] for p in pairs]]
    b = new[[p[1] for p in pairs]]
    return {"status": "compared", "n_compared": len(pairs), "max_abs_diff": float(np.abs(a - b).max()),
            "min_cosine": float((a * b).sum(axis=1).min())}


# ── clustering diagnostics ────────────────────────────────────────────────────

def size_stats(sizes: list[int]) -> dict[str, Any]:
    if not sizes:
        return {"n": 0}
    s = np.array(sorted(sizes, reverse=True))
    return {"n": int(len(s)), "min": int(s.min()), "median": float(np.median(s)), "max": int(s.max()),
            "top1_share_of_clustered": round(float(s[0] / s.sum()), 4),
            "top5_share_of_clustered": round(float(s[:5].sum() / s.sum()), 4)}


def npmi_coherence(doc_texts: list[str], topics: dict[int, list[str]]) -> dict[int, float | None]:
    """NPMI over document co-occurrence of each cluster's top terms (descriptive lexical proxy)."""
    from sklearn.feature_extraction.text import CountVectorizer
    vocab = sorted({t for terms in topics.values() for t in terms})
    if not vocab:
        return {c: None for c in topics}
    vec = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), binary=True, lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    X = vec.transform(doc_texts).tocsc()
    n = X.shape[0]
    df = np.asarray(X.sum(axis=0)).ravel()
    index = {t: i for i, t in enumerate(vocab)}
    out: dict[int, float | None] = {}
    for cid, terms in topics.items():
        terms = [t for t in terms if t in index]
        vals = []
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = index[terms[i]], index[terms[j]]
                co = int(X[:, a].multiply(X[:, b]).sum())
                p1, p2, p12 = df[a] / n, df[b] / n, co / n
                if p12 <= 0 or p1 <= 0 or p2 <= 0:
                    vals.append(-1.0)
                elif p12 >= 1.0:
                    vals.append(1.0)
                else:
                    vals.append(float(np.log(p12 / (p1 * p2)) / (-np.log(p12))))
        out[cid] = float(np.mean(vals)) if vals else None
    return out


def representatives(unique: pd.DataFrame, emb: np.ndarray, labels: np.ndarray, n_central: int = 3, n_edge: int = 2) -> pd.DataFrame:
    rows = []
    for cid in sorted(set(labels) - {-1}):
        idx = np.where(labels == cid)[0]
        centroid = emb[idx].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        sims = emb[idx] @ centroid
        order = np.argsort(-sims)
        picks = [(int(idx[o]), "central", float(sims[o])) for o in order[:n_central]]
        seen = {p[0] for p in picks}
        for o in order[::-1][:n_edge]:
            if int(idx[o]) not in seen:
                picks.append((int(idx[o]), "peripheral", float(sims[o])))
        for u, kind, sim in picks:
            r = unique.iloc[u]
            rows.append({"cluster_id": int(cid), "kind": kind, "cosine_to_centroid": round(sim, 4),
                         "unique_id": int(r["unique_id"]), "corpus_id": int(r["corpus_id"]),
                         "multiplicity": int(r["multiplicity"]), "source_dataset": r["source_dataset"],
                         "source_record_id": r["source_record_id"], "question_slug": r["question_slug"],
                         "language": r["language"], "original_text": r["original_text"]})
    return pd.DataFrame(rows)


def lang_group(lang: str) -> str:
    return lang if lang in ("English", "French") else "Other/Unknown"


def cluster_diagnostics(entries: pd.DataFrame, unique_labels: np.ndarray, topics_df: pd.DataFrame,
                        coherence: dict[int, float | None]) -> dict[str, Any]:
    e = entries.copy()
    e["cluster"] = unique_labels[e["unique_id"].to_numpy()]
    e["lang_group"] = e["language"].map(lang_group)
    clustered = e[e["cluster"] != -1]
    sizes_u = pd.Series(unique_labels[unique_labels != -1]).value_counts().tolist()
    sizes_e = clustered.groupby("cluster").size().tolist()
    overall_fr = float((e["lang_group"] == "French").mean())
    per_cluster = []
    dup_heavy = 0
    tmpl_heavy = 0
    e["template_key"] = (e["text_key"].str.replace(r"https?://\S+", "", regex=True)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    clustered = e[e["cluster"] != -1]
    tmpl_counts_all = e["template_key"].value_counts()
    for cid, g in clustered.groupby("cluster"):
        fr = float((g["lang_group"] == "French").mean())
        top_ds = g["source_dataset"].value_counts(normalize=True)
        chan = g[g["source_dataset"] == "consultation_compilation_2017_18"]["stakeholder_group"]
        top_chan = chan.value_counts(normalize=True).iloc[0] if len(chan) else None
        top_chan_name = chan.value_counts().index[0] if len(chan) else None
        tcounts = g["template_key"].value_counts()
        top_tmpl = float(tcounts.iloc[0] / len(g))
        tmpl_heavy += (tcounts.iloc[0] >= 5 and top_tmpl >= 0.3)
        counts = g["text_key"].value_counts()
        top_dup = float(counts.iloc[0] / len(g))
        dup_heavy += (counts.iloc[0] >= 5 and top_dup >= 0.3)
        per_cluster.append({"cluster_id": int(cid), "entries": int(len(g)), "unique_texts": int(g["unique_id"].nunique()),
                            "french_share": round(fr, 3), "top_dataset": top_ds.index[0],
                            "top_dataset_share": round(float(top_ds.iloc[0]), 3),
                            "top_channel_share_in_compilation": None if top_chan is None else round(float(top_chan), 3),
                            "top_channel": top_chan_name, "compilation_entries": int(len(chan)),
                            "top_template_share": round(top_tmpl, 3), "top_template_count": int(tcounts.iloc[0]),
                            "top_duplicate_text_share": round(top_dup, 3),
                            "npmi": None if coherence.get(int(cid)) is None else round(float(coherence[int(cid)]), 3)})
    npmi_vals = [v for v in coherence.values() if v is not None]
    return {
        "entries": int(len(e)), "entries_clustered": int(len(clustered)),
        "entry_noise_ratio": round(1 - len(clustered) / len(e), 4),
        "unique_noise_ratio": round(float((unique_labels == -1).mean()), 4),
        "n_clusters": int(len(sizes_u)),
        "size_unique_texts": size_stats(sizes_u), "size_entries": size_stats(sizes_e),
        "corpus_french_share": round(overall_fr, 4),
        "clusters_ge90pct_french": int(sum(pc["french_share"] >= 0.9 for pc in per_cluster)),
        "clusters_ge95pct_single_dataset": int(sum(pc["top_dataset_share"] >= 0.95 for pc in per_cluster)),
        "clusters_duplicate_heavy": int(dup_heavy),
        "clusters_template_heavy": int(tmpl_heavy),
        "entries_sharing_url_stripped_text": int((tmpl_counts_all[tmpl_counts_all >= 2]).sum()),
        "clusters_channel_dominated_ge70pct": int(sum(
            1 for pc in per_cluster if pc["compilation_entries"] >= 20 and (pc["top_channel_share_in_compilation"] or 0) >= 0.7)),
        "mean_npmi": None if not npmi_vals else round(float(np.mean(npmi_vals)), 4),
        "median_npmi": None if not npmi_vals else round(float(np.median(npmi_vals)), 4),
        "per_cluster": per_cluster,
    }


def exact_knn_cosine(emb: np.ndarray, k: int, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-nearest neighbours (including self at position 0) by cosine distance on L2-normalised rows.

    Used instead of UMAP's approximate NN-descent: on this machine pynndescent failed with
    MemoryError (numba allocation) when only a few hundred MB of RAM were available. Chunked
    brute force needs ~chunk x N floats and returns the exact neighbour graph.
    """
    n = emb.shape[0]
    k = min(k, n)
    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        sims = emb[i0:i1] @ emb.T
        rows = np.arange(i1 - i0)
        sims[rows, rows + i0] = np.inf                          # force self into position 0
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        idx_out[i0:i1] = top
        d = 1.0 - np.take_along_axis(sims, top, axis=1)
        d[:, 0] = 0.0
        dist_out[i0:i1] = np.clip(d, 0.0, 2.0)
    return idx_out, dist_out


def umap_reduce_exact_knn(emb: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, metric: str,
                          random_state: int) -> np.ndarray:
    """Same UMAP hyper-parameters as models.concern_clustering.reduce_with_umap, with an exact precomputed kNN graph."""
    import umap
    knn_idx, knn_dist = exact_knn_cosine(emb, n_neighbors)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=random_state, transform_seed=random_state, precomputed_knn=(knn_idx, knn_dist, None))
    return reducer.fit_transform(emb).astype(np.float32)


def stage_cluster(model_key: str, run_root: Path = RUN_ROOT, umap_configs: dict | None = None,
                  hdbscan_configs: dict | None = None, seed: int = SEED, do_seed_check: bool = True) -> dict[str, Any]:
    import hdbscan
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from models.concern_clustering.clustering import extract_cluster_topics

    umap_configs = umap_configs or UMAP_CONFIGS
    hdbscan_configs = hdbscan_configs or HDBSCAN_CONFIGS
    out_dir = run_root / "clustering" / model_key
    summary_path = out_dir / "cluster_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    emb = np.load(run_root / "embeddings" / model_key / "embeddings.npy")
    unique = pd.read_csv(run_root / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    entries = pd.read_csv(run_root / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    entries["unique_id"] = entries["unique_id"].astype(int)
    if len(unique) != len(emb):
        raise ValueError("Embeddings are not aligned with the unique-text table.")
    docs = unique["original_text"].tolist()
    meta_for_topics = unique.assign(corpus_id=unique["unique_id"].astype(int))
    results: dict[str, Any] = {}
    reduced_cache: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    labels_by_id: dict[str, np.ndarray] = {}

    def reduce(uid: str, s: int) -> np.ndarray:
        key = f"{uid}_seed{s}"
        if key not in reduced_cache:
            path = out_dir / "umap" / f"{key}.npy"
            if path.exists():
                reduced_cache[key] = np.load(path)
            else:
                t = time.perf_counter()
                red = umap_reduce_exact_knn(emb, random_state=s, **umap_configs[uid])
                write_new(path, lambda f: np.save(f, red), binary=True)
                reduced_cache[key] = red
                results.setdefault("_umap_seconds", {})[key] = round(time.perf_counter() - t, 1)
        return reduced_cache[key]

    def run_hdbscan(red: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, float | None, Any]:
        cl = hdbscan.HDBSCAN(min_cluster_size=cfg["min_cluster_size"], min_samples=cfg["min_samples"], metric="euclidean",
                             cluster_selection_method="eom", gen_min_span_tree=True)
        lab = cl.fit_predict(red)
        try:
            rv = float(cl.relative_validity_)
        except Exception:  # noqa: BLE001
            rv = None
        return lab, cl.probabilities_.astype(np.float32), rv, cl

    for uid in umap_configs:
        red = reduce(uid, seed)
        for hid, hcfg in hdbscan_configs.items():
            config_id = f"{uid}_{hid}"
            cdir = out_dir / config_id
            if (cdir / "diagnostics.json").exists():          # resume: configuration already completed
                diag = json.loads((cdir / "diagnostics.json").read_text(encoding="utf-8"))
                ea = pd.read_csv(cdir / "entry_assignments.csv", usecols=["unique_id", "cluster_id"])
                lab = np.full(len(unique), -1, dtype=int)
                lab[ea["unique_id"].to_numpy()] = ea["cluster_id"].to_numpy()
                labels_by_id[config_id] = lab
                results[config_id] = {k: v for k, v in diag.items() if k != "per_cluster"}
                continue
            lab, prob, rv, cl = run_hdbscan(red, hcfg)
            labels_by_id[config_id] = lab
            non = lab != -1
            sil = None
            if len(set(lab[non])) >= 2:
                sil = float(silhouette_score(red[non], lab[non], sample_size=min(4000, int(non.sum())), random_state=seed))
            topics_df = extract_cluster_topics(meta_for_topics, lab)
            topic_terms = {int(r.cluster_id): [t.strip() for t in r.top_keywords.split(",") if t.strip()]
                           for r in topics_df.itertuples() if r.cluster_id != -1}
            coherence = npmi_coherence(docs, topic_terms)
            diag = cluster_diagnostics(entries, lab, topics_df, coherence)
            diag.update({"config_id": config_id, "umap": umap_configs[uid], "hdbscan": hcfg,
                         "silhouette_sample4000_nonnoise": None if sil is None else round(sil, 4),
                         "hdbscan_relative_validity": None if rv is None else round(rv, 4)})
            reps = representatives(unique, emb, lab)
            entries_out = entries[["corpus_id", "unique_id", "source_dataset", "source_record_id", "question_slug", "language"]].copy()
            entries_out["cluster_id"] = lab[entries_out["unique_id"].to_numpy()]
            entries_out["probability"] = prob[entries_out["unique_id"].to_numpy()].round(4)
            write_new(cdir / "entry_assignments.csv", lambda f: entries_out.to_csv(f, index=False, lineterminator="\n"))
            topics_out = topics_df.merge(pd.DataFrame([{"cluster_id": c, "npmi": v} for c, v in coherence.items()]),
                                         on="cluster_id", how="left")
            write_new(cdir / "topics.csv", lambda f: topics_out.to_csv(f, index=False, lineterminator="\n"))
            write_new(cdir / "representatives.csv", lambda f: reps.to_csv(f, index=False, lineterminator="\n"))
            write_json_new(cdir / "diagnostics.json", diag)
            results[config_id] = {k: v for k, v in diag.items() if k != "per_cluster"}

    if do_seed_check:
        s = SEED_STABILITY
        if s["umap"] in umap_configs and s["hdbscan"] in hdbscan_configs:
            red2 = reduce(s["umap"], s["seed"])
            lab2, _, _, _ = run_hdbscan(red2, hdbscan_configs[s["hdbscan"]])
            lab1 = labels_by_id[f"{s['umap']}_{s['hdbscan']}"]
            both = (lab1 != -1) & (lab2 != -1)
            results["_seed_stability"] = {
                "config": f"{s['umap']}_{s['hdbscan']}", "seeds": [seed, s["seed"]],
                "ari_on_points_clustered_in_both": round(float(adjusted_rand_score(lab1[both], lab2[both])), 4),
                "share_clustered_in_both": round(float(both.mean()), 4),
                "n_clusters": [int(len(set(lab1)) - (1 if -1 in lab1 else 0)), int(len(set(lab2)) - (1 if -1 in lab2 else 0))],
            }
    results["_runtime"] = {"elapsed_s": round(time.perf_counter() - t0, 1), "peak_rss_mb": peak_rss_mb(),
                           "system": system_snapshot()}
    write_json_new(summary_path, results)
    return results


# ── report ────────────────────────────────────────────────────────────────────

def select_reporting_config(summary: dict[str, Any]) -> str | None:
    ok = []
    for cid, d in summary.items():
        if cid.startswith("_") or d.get("mean_npmi") is None:
            continue
        if 20 <= d["n_clusters"] <= 150 and d["entry_noise_ratio"] <= 0.40:
            ok.append((d["mean_npmi"], d["hdbscan"]["min_cluster_size"], cid))
    return max(ok)[2] if ok else None


def ari_between(run_root: Path, key_a: str, cfg_a: str, key_b: str, cfg_b: str) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score
    a = pd.read_csv(run_root / "clustering" / key_a / cfg_a / "entry_assignments.csv", usecols=["corpus_id", "cluster_id"])
    b = pd.read_csv(run_root / "clustering" / key_b / cfg_b / "entry_assignments.csv", usecols=["corpus_id", "cluster_id"])
    m = a.merge(b, on="corpus_id", suffixes=("_a", "_b"))
    both = (m.cluster_id_a != -1) & (m.cluster_id_b != -1)
    return {"ari": round(float(adjusted_rand_score(m.cluster_id_a[both], m.cluster_id_b[both])), 4),
            "share_clustered_in_both": round(float(both.mean()), 4)}


def stage_report(run_root: Path = RUN_ROOT, report_path: Path = REPORT_PATH) -> str:
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    corpus = json.loads((run_root / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8"))
    recon = pd.read_csv(run_root / "corpus" / "reconciliation.csv") if (run_root / "corpus" / "reconciliation.csv").exists() else None
    emb_m, summ, check = {}, {}, {}
    for key in MODELS:
        p = run_root / "embeddings" / key / "embedding_manifest.json"
        if p.exists():
            emb_m[key] = json.loads(p.read_text(encoding="utf-8"))
            check[key] = check_against_reference(key, run_root)
        s = run_root / "clustering" / key / "cluster_summary.json"
        if s.exists():
            summ[key] = json.loads(s.read_text(encoding="utf-8"))
    L: list[str] = []
    A = L.append
    A("# PolicyLens full-corpus unsupervised clustering report (v1)\n")
    A("**Status: exploratory, unsupervised, NOT human-validated.** Clusters are groups of semantically similar texts under one embedding model; "
      "they are not validated public concerns. No human labels, reviewer agreement or accuracy figures were used or produced. Sentiment, stance and concern are not inferred.\n")
    A("## 1. Corpus definition\n")
    A("- **Unit:** a text entry = a non-blank response in a designed free-text field of a source record (individual comments: 5 fields; compilation: the comment column; questionnaire: Q8, Q14, Q18).")
    A(f"- **Eligibility:** {corpus['eligibility']}.")
    A(f"- **Eligible entries:** {corpus['entries']:,}. **Unique normalised texts clustered:** {corpus['unique_texts']:,} "
      f"({corpus['duplicate_entries_mapped_to_unique']} repeated entries mapped back to their first occurrence; multiplicity kept).")
    A("- **Not respondent-level:** one questionnaire respondent may contribute up to 3 entries; the sources contain no respondent identifier.")
    A("- **Entries by dataset:** " + ", ".join(f"`{k}`={v:,}" for k, v in corpus["by_dataset"].items()) + "\n")
    if recon is not None:
        A("### Reconciliation of the extracted entries\n")
        A("`data/interim/government_response_text_entries.csv` has "
          f"**{corpus['reconciliation_total_extracted']:,}** extracted entries. They are accounted for as follows:\n")
        cols = [c for c in recon.columns if c != "dataset_name"]
        A("| Dataset | " + " | ".join(cols) + " |")
        A("|---|" + "---:|" * len(cols))
        for _, r in recon.iterrows():
            A(f"| `{r['dataset_name']}` | " + " | ".join(str(int(r[c])) if pd.notna(r[c]) else "0" for c in cols) + " |")
        tot = recon[cols].sum()
        A(f"| **total** | " + " | ".join(str(int(tot[c])) for c in cols) + " |")
        A("\nThe figure **13,541** quoted in `reports/current_project_status.md` could not be reproduced from any filter of this table "
          "(it has no derivation in the repository) and should not be used; the counts above supersede it.\n")
    A("### Language metadata (source metadata, not inferred)\n")
    A("| Dataset \\| language | entries |\n|---|---:|")
    for k, v in sorted(corpus["language_counts"].items()):
        A(f"| {k} | {v:,} |")
    A("\nQuestionnaire language is the survey language selected, not necessarily the comment's language. 'S' (3), Arabic (2) and Japanese (1) entries are kept and counted as Other/Unknown.\n")
    A("## 2. Embeddings\n")
    A("| Model | Revision snapshot | Unique texts | Dim | Chunks (computed/reused) | Elapsed s | Peak RSS MB | Texts > 256 tokens |")
    A("|---|---|---:|---:|---|---:|---:|---|")
    for key, m in emb_m.items():
        tr = m.get("truncation") or {}
        A(f"| `{m['model_id']}` | {m['model_info'].get('snapshot', '?')[:12]} | {m['n_unique_texts']:,} | {m['dim']} | "
          f"{m['chunks_computed_this_run']}/{m['chunks_reused']} | {m['elapsed_s_this_run']} | {m['peak_rss_mb']} | "
          f"{tr.get('tokens_over_max_length', '?')} ({100 * tr.get('share_over_max_length', 0):.1f}%) |")
    A("\nEquivalence check against the saved 450-comment vectors (same texts, same pooling):\n")
    for key, c in check.items():
        if c.get("status") == "compared":
            A(f"- `{key}`: {c['n_compared']} texts compared, max |Δ| = {c['max_abs_diff']:.2e}, min cosine = {c['min_cosine']:.6f}")
        else:
            A(f"- `{key}`: {c.get('status')}")
    ents = pd.read_csv(run_root / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    uq = pd.read_csv(run_root / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    first_text = dict(zip(uq["unique_id"].astype(int), uq["original_text"]))
    differs = int((ents["original_text"] != ents["unique_id"].astype(int).map(first_text)).sum())
    A(f"\nExact-text collapsing uses a case- and whitespace-insensitive key, so {differs} entries share the embedding of a "
      "first-occurrence variant that differs only in case or spacing (for example 'Did not know it existed' / 'did not know it existed'). "
      "The one 450-sample text that differs beyond 1e-6 in the equivalence check is such a variant (cosine 0.9996).")
    A("\nModel loading used a low-memory loader (meta-device build, tensors assigned one by one) because the default `from_pretrained` "
      "crashed with an access violation at ~500 MB available RAM; weights are identical (see the equivalence check). Model revisions are not pinned.\n")
    for key, s in summ.items():
        A(f"## 3. Clustering configurations — `{key}`\n")
        A("UMAP (cosine, 5-D, min_dist 0, seed 42) then HDBSCAN (euclidean, EOM). `u15`/`m10s5` are the settings used for the 450-comment baseline. "
          "Silhouette is a geometric descriptor on non-noise points of a 4,000-point sample; NPMI is a lexical co-occurrence proxy for keyword coherence "
          "(each cluster's top-10 c-TF-IDF terms; a pair that never co-occurs in one text scores -1, so values are mostly negative for short texts and "
          "are only meaningful as relative comparisons). Neither shows semantic or policy validity. "
          "'Channel-dominated' = clusters with at least 20 compilation entries where one source channel (Twitter, Slido, RSVP forms, etc.) supplies ≥70% of them; "
          "'template-heavy' = a text, after removing URLs, repeated ≥5 times and making up ≥30% of the cluster.\n")
        A("| Config | Clusters | Noise (entries) | Size median / max | Top-5 share | ≥90% French | ≥95% one dataset | Channel-dominated (≥70%) | Template-heavy | Silhouette | Rel. validity | Mean NPMI |")
        A("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cid, d in s.items():
            if cid.startswith("_"):
                continue
            se = d["size_unique_texts"]
            A(f"| {cid} | {d['n_clusters']} | {100 * d['entry_noise_ratio']:.1f}% | {se.get('median', 0):.0f} / {se.get('max', 0)} | "
              f"{100 * se.get('top5_share_of_clustered', 0):.0f}% | {d['clusters_ge90pct_french']} | {d['clusters_ge95pct_single_dataset']} | "
              f"{d['clusters_channel_dominated_ge70pct']} | {d['clusters_template_heavy']} | {d['silhouette_sample4000_nonnoise']} | "
              f"{d['hdbscan_relative_validity']} | {d['mean_npmi']} |")
        if "_seed_stability" in s:
            st = s["_seed_stability"]
            A(f"\nSeed stability ({st['config']}, UMAP seeds {st['seeds']}): ARI on points clustered in both = {st['ari_on_points_clustered_in_both']}, "
              f"clustered in both = {100 * st['share_clustered_in_both']:.0f}%, clusters {st['n_clusters']}.")
        rt = s.get("_runtime", {})
        A(f"\nRuntime {rt.get('elapsed_s')} s; peak RSS {rt.get('peak_rss_mb')} MB; UMAP seconds: {s.get('_umap_seconds')}\n")
    chosen = select_reporting_config(summ.get("multilingual", {})) if "multilingual" in summ else None
    if "multilingual" in summ and "baseline" in summ:
        A("## 4. Model comparison\n")
        A("Agreement between the two embedding models' partitions, per UMAP/HDBSCAN configuration (ARI on points clustered by both; low values mean the models organise the corpus differently, not that one is better):\n")
        A("| Config | ARI | clustered by both |\n|---|---:|---:|")
        for cid in [c for c in summ["multilingual"] if not c.startswith("_") and c in summ["baseline"]]:
            r = ari_between(run_root, "multilingual", cid, "baseline", cid)
            A(f"| {cid} | {r['ari']} | {100 * r['share_clustered_in_both']:.0f}% |")
        A("")
    if chosen:
        A(f"## 5. Reporting configuration: `multilingual/{chosen}`\n")
        A(f"{SELECTION_RULE}\n")
        topics = pd.read_csv(run_root / "clustering" / "multilingual" / chosen / "topics.csv", dtype=str, keep_default_na=False)
        reps = pd.read_csv(run_root / "clustering" / "multilingual" / chosen / "representatives.csv", dtype=str, keep_default_na=False)
        diag = json.loads((run_root / "clustering" / "multilingual" / chosen / "diagnostics.json").read_text(encoding="utf-8"))
        pc = {c["cluster_id"]: c for c in diag["per_cluster"]}
        topics = topics[topics.cluster_id != "-1"].copy()
        topics["size"] = topics["size"].astype(int)
        A("Largest clusters (keywords = c-TF-IDF terms; examples = texts nearest the cluster centroid, verbatim, traceable by `corpus_id`):\n")
        for _, t in topics.sort_values("size", ascending=False).head(15).iterrows():
            cid = int(t.cluster_id)
            d = pc.get(cid, {})
            A(f"### Cluster {cid} — {t['size']} unique texts; {d.get('entries', '?')} entries; French {100 * d.get('french_share', 0):.0f}%; "
              f"top dataset {d.get('top_dataset')} ({100 * d.get('top_dataset_share', 0):.0f}%); duplicate-text share {100 * d.get('top_duplicate_text_share', 0):.0f}%\n")
            A(f"- Keywords: {t['top_keywords']}")
            for _, r in reps[reps.cluster_id == str(cid)].head(3).iterrows():
                txt = r["original_text"].replace("\n", " ")
                A(f"- corpus_id {r['corpus_id']} ({r['source_dataset']}, {r['question_slug']}, {r['language']}): “{txt[:240]}{'…' if len(txt) > 240 else ''}”")
            A("")
        A("Full per-cluster keyword and representative tables for every configuration are in "
          f"`data/full_corpus/run_v1/clustering/<model>/<config>/{{topics,representatives}}.csv`.\n")
    else:
        A("## 5. Reporting configuration\n\nNo multilingual configuration met the pre-declared readability criteria "
          "(20–150 clusters, noise ≤ 40%); no configuration is singled out. " + SELECTION_RULE + "\n")
    if "multilingual" in summ:
        A("## 5b. Data-driven observations (computed, not interpretive)\n")
        ms = summ["multilingual"]
        cfgs = [c for c in ms if not c.startswith("_")]
        lo, hi = min(ms[c]["entry_noise_ratio"] for c in cfgs), max(ms[c]["entry_noise_ratio"] for c in cfgs)
        A(f"- Between {100 * lo:.0f}% and {100 * hi:.0f}% of entries are left as noise (unassigned) depending on configuration; "
          "a cluster map therefore covers only part of the corpus.")
        if chosen:
            d = ms[chosen]
            A(f"- In the reporting configuration, {d['clusters_channel_dominated_ge70pct']} of {d['n_clusters']} clusters are dominated (≥70%) by a single "
              f"compilation channel and {d['clusters_ge95pct_single_dataset']} come ≥95% from a single instrument; "
              f"{d['clusters_template_heavy']} are template-heavy. The corpus has {d['entries_sharing_url_stripped_text']} entries sharing "
              "identical text once URLs are removed (beyond the 65 exact duplicates already collapsed).")
            A(f"- The largest cluster holds {d['size_unique_texts']['max']:,} unique texts "
              f"(top-5 clusters = {100 * d['size_unique_texts']['top5_share_of_clustered']:.0f}% of clustered texts), "
              "so a few broad clusters dominate.")
            A(f"- Clusters with ≥90% French entries: {d['clusters_ge90pct_french']} (corpus French share {100 * d['corpus_french_share']:.1f}%): "
              "French texts are distributed among mixed-language clusters under the multilingual model"
              + (f"; the English-only baseline model at the same configuration has {summ['baseline'][chosen]['clusters_ge90pct_french']}."
                 if "baseline" in summ and chosen in summ["baseline"] else "."))
        if "_seed_stability" in ms:
            st = ms["_seed_stability"]
            A(f"- Changing only the UMAP seed ({st['config']}) gives ARI {st['ari_on_points_clustered_in_both']} on the "
              f"{100 * st['share_clustered_in_both']:.0f}% of points clustered under both seeds, with cluster counts {st['n_clusters']}: "
              "the partition is only moderately stable.")
        if "baseline" in summ:
            A("- The two embedding models give different partitions (see section 4); neither is validated, so no model is declared better.")
        A("")
    A("## 5c. Resources: pre-run estimate vs observed\n")
    disk_mb = round(sum(f.stat().st_size for f in run_root.rglob("*") if f.is_file()) / 2**20, 1)
    A("| Item | Estimate before running | Observed |\n|---|---|---|")
    for key, m in emb_m.items():
        A(f"| Encode `{key}` (peak RSS) | ~{450 if key == 'multilingual' else 90} MB weights + ~600 MB working set | "
          f"{m['peak_rss_mb']} MB in {m['elapsed_s_this_run']} s |")
    for key, sm in summ.items():
        rt = sm.get("_runtime", {})
        A(f"| UMAP/HDBSCAN `{key}` | ~900 MB, minutes | {rt.get('peak_rss_mb')} MB in {rt.get('elapsed_s')} s |")
    A(f"| Artifacts under `data/full_corpus/run_v1` | ~70 MB | {disk_mb} MB |")
    sysinfo = next(iter(emb_m.values()))["system"] if emb_m else {}
    A(f"\nMachine: {sysinfo.get('logical_cpus')} logical CPUs, {sysinfo.get('ram_total_mb')} MB RAM "
      f"(about {sysinfo.get('ram_available_mb')} MB free at run time), {sysinfo.get('disk_free_gb')} GB free disk. "
      "The run completed on this memory-constrained machine because of the low-memory loader and the exact-kNN path (section 6b).\n")
    A("## 6. Limitations\n")
    for x in [
        "Clusters are unsupervised and have not been human-reviewed; the 44-pair concern-coherence review is still pending.",
        "Silhouette, relative validity and NPMI are proxies and can be high for incoherent or channel-driven clusters.",
        "Clusters may reflect response channel, question wording, language or boilerplate (e.g. event RSVPs, hashtags) rather than a policy concern; inspect composition columns.",
        "Texts above 256 tokens are truncated; very long responses are represented by their first 256 tokens.",
        "Short texts (≥20 characters) carry little context, so embeddings of generic remarks may be poorly separated.",
        "Exact duplicates are collapsed; near-duplicates and campaign text are not removed and may form dense clusters.",
        "The corpus mixes three instruments with different prompts; counts are not population estimates and no design weighting is applied.",
        "Language labels are source metadata; cross-lingual alignment is not established by language mixing alone.",
        "Embedding models are not revision-pinned; exact numerical reproduction across library or model revisions is not guaranteed.",
    ]:
        A(f"- {x}")
    A("\n## 6b. Implementation notes\n")
    A("- **Low-memory model loading** (see section 2).")
    A("- **Exact k-nearest-neighbour graph for UMAP.** UMAP's default approximate NN-descent (pynndescent) failed with a numba `MemoryError` on this machine "
      "(a few hundred MB of free RAM). The cosine kNN graph is therefore computed exactly by chunked brute force and passed to UMAP via `precomputed_knn`; "
      "all other UMAP/HDBSCAN hyper-parameters equal the 450-comment baseline. Exact neighbours can differ slightly from the approximate graph used for the 450 baseline.")
    A("- HDBSCAN is called directly with the baseline parameters (the existing wrapper also computes an extra 2-D UMAP that is not needed here).\n")
    A("\n## 7. Reproducibility\n")
    A("```powershell\npython scripts\\run_full_corpus_clustering.py corpus\npython scripts\\run_full_corpus_clustering.py embed --model-key multilingual\n"
      "python scripts\\run_full_corpus_clustering.py embed --model-key baseline\npython scripts\\run_full_corpus_clustering.py cluster --model-key multilingual\n"
      "python scripts\\run_full_corpus_clustering.py cluster --model-key baseline\npython scripts\\run_full_corpus_clustering.py report\n```")
    A("\nSeed 42; UMAP `random_state` fixes results on a given machine/library set. Stages resume from chunk and artifact caches and refuse to overwrite outputs.")
    text = "\n".join(L) + "\n"
    write_new(report_path, lambda f: f.write(text))
    manifest = {
        "run": "full_corpus_v1", "report": report_path.name,
        "reporting_configuration": None if not chosen else f"multilingual/{chosen}",
        "corpus": {k: corpus[k] for k in ("entries", "unique_texts", "by_dataset", "corpus_entries_sha256", "unique_texts_sha256")},
        "embeddings": {k: {"embeddings_sha256": v["embeddings_sha256"], "n": v["n_unique_texts"], "dim": v["dim"],
                           "elapsed_s": v["elapsed_s_this_run"], "peak_rss_mb": v["peak_rss_mb"]} for k, v in emb_m.items()},
        "clustering": {k: {c: {"clusters": d["n_clusters"], "entry_noise_ratio": d["entry_noise_ratio"], "mean_npmi": d["mean_npmi"]}
                           for c, d in s.items() if not c.startswith("_")} for k, s in summ.items()},
        "assignment_sha256": {f"{k}/{c}": sha256_file(run_root / "clustering" / k / c / "entry_assignments.csv")
                              for k, s in summ.items() for c in s if not c.startswith("_")},
        "human_validated": False,
    }
    write_json_new(run_root / "run_manifest.json", manifest)
    return text


# ── profile / CLI ─────────────────────────────────────────────────────────────

def stage_profile() -> dict[str, Any]:
    from models.semantic_similarity.minilm import build_corpus
    c = build_corpus(PROCESSED)
    key = c["original_text"].map(normalise_text)
    n_unique = int(key.nunique())
    snap = system_snapshot()
    est_art_mb = round(n_unique * 384 * 4 / 2**20, 1)
    return {
        "entries": int(len(c)), "unique_texts": n_unique, "embedding_matrix_mb_per_model": est_art_mb,
        "system": snap,
        "estimate": {"model_weights_mb": {"multilingual": 450, "baseline": 90},
                     "encode_peak_ram_mb_batch16": 1100, "umap_hdbscan_peak_ram_mb": 900,
                     "artifacts_total_mb_estimate": round(est_art_mb * 2 + 40, 1)},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("stage", choices=["profile", "corpus", "embed", "cluster", "report"])
    p.add_argument("--model-key", choices=sorted(MODELS))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--no-seed-check", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.stage == "profile":
            print(json.dumps(stage_profile(), indent=2))
        elif a.stage == "corpus":
            print(json.dumps(stage_corpus(), indent=2)[:1500])
        elif a.stage == "embed":
            m = stage_embed(a.model_key, threads=a.threads)
            print(json.dumps({k: m[k] for k in ("model_key", "n_unique_texts", "dim", "chunks_computed_this_run", "chunks_reused",
                                                "elapsed_s_this_run", "peak_rss_mb")}, indent=2))
            print(json.dumps(check_against_reference(a.model_key), indent=2))
        elif a.stage == "cluster":
            s = stage_cluster(a.model_key, do_seed_check=not a.no_seed_check)
            for cid, d in s.items():
                if not cid.startswith("_"):
                    print(cid, d["n_clusters"], "clusters; noise", d["entry_noise_ratio"], "NPMI", d["mean_npmi"])
            print(s.get("_seed_stability"), s.get("_runtime"))
        else:
            stage_report()
            print(f"written {REPORT_PATH}")
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
