"""
indian/scripts/run_civis_e5_chunked.py
========================================
Resumable, checkpointed intfloat/e5-base-v2 embedding for the CIVIS corpus.

Companion to:  run_civis_retrieval_v1.py embed --encoder e5-base-v2
Output:        same artifact paths, identical embedding values

Motivation
----------
The original embed stage accumulates all 22,300 embeddings in a Python list
before calling np.vstack and np.save.  On this 8 GB laptop that leaves no margin
for interruption; any crash loses all progress.  This script saves a validated
checkpoint shard every N chunks so a restart loses at most one shard of work.

Stages
------
(default)   encode missing/stale shards → assemble → final .npy + .json
--validate  inspect checkpoints and final output; do not encode
--dry-run   check config and RAM; print plan; do not encode or write

Final outputs  (identical paths used by run_civis_retrieval_v1.py evaluate)
---------------------------------------------------------------------------
  civis_v1/chunk_embeddings_e5-base-v2.npy    float16, shape (N, 768)
  civis_v1/chunk_embeddings_e5-base-v2.json   metadata + SHA256s

Checkpoints  (kept on interrupt; delete only with --cleanup-ckpts)
------------------------------------------------------------------
  civis_v1/e5_ckpt/shard_{start:06d}.npy    float32, shape (shard_size, 768)
  civis_v1/e5_ckpt/shard_{start:06d}.json   per-shard metadata and checksums

Shard atomicity protocol
------------------------
  1. encode shard → validate shape + finite values in memory
  2. write to shard_NNNNNN.tmp.npy
  3. os.replace(.tmp.npy → .npy)   atomic; .npy is now present
  4. write shard_NNNNNN.json       completion marker with checksums

If killed between steps 3 and 4 the .npy exists without .json; _validate_shard
reports "orphan .npy" and the shard is re-encoded on resume.

Usage
-----
    python -I indian/scripts/run_civis_e5_chunked.py
    python -I indian/scripts/run_civis_e5_chunked.py --batch-size 4 --shard-size 500
    python -I indian/scripts/run_civis_e5_chunked.py --dry-run
    python -I indian/scripts/run_civis_e5_chunked.py --validate
    python -I indian/scripts/run_civis_e5_chunked.py --cleanup-ckpts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "indian"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(IND))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval import civis as cv  # noqa: E402

# ── configuration  (values must match run_civis_retrieval_v1.py exactly) ──────

SRC: Path = IND / "processed" / "civis_policy_documents_clean.csv"
OUT: Path = IND / "processed" / "retrieval" / "civis_v1"
MAX_LEN: int = 256              # line 41 of run_civis_retrieval_v1.py
ENCODER_NAME: str = "e5-base-v2"
SPEC: dict = cv.ENCODERS[ENCODER_NAME]   # model_id, dim, passage_prefix, …
EMB_P: Path = OUT / f"chunk_embeddings_{ENCODER_NAME}.npy"
META_P: Path = OUT / f"chunk_embeddings_{ENCODER_NAME}.json"
CKPT_DIR: Path = OUT / "e5_ckpt"

# Paths this script must never write to.
_PROTECTED: list[Path] = [
    OUT / "chunk_embeddings_all-MiniLM-L6-v2.npy",
    OUT / "chunk_embeddings_all-MiniLM-L6-v2.json",
    OUT / "evaluation" / "metrics.json",
    OUT / "evaluation" / "paired_tests.json",
    OUT / "evaluation" / "per_query_ranks.csv",
]


# ── utilities ─────────────────────────────────────────────────────────────────

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_json_new(p: Path, obj: object) -> None:
    """Write JSON; raise FileExistsError rather than overwrite."""
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite: {p}")
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _config_hash(chunks_sha256: str) -> str:
    """Short hash of every parameter that affects the numerical embedding values.

    Changing model_id, passage_prefix, max_length, or the corpus text invalidates
    all existing checkpoints.  Batch size and shard size do NOT affect values and
    are intentionally excluded.
    """
    key = json.dumps(
        {
            "model_id": SPEC["model_id"],
            "passage_prefix": SPEC["passage_prefix"],
            "max_length": MAX_LEN,
            "chunks_sha256": chunks_sha256,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(key).hexdigest()[:16]


# ── corpus loading ─────────────────────────────────────────────────────────────

def _load_passages() -> tuple[list[str], str]:
    """Build the ordered passage list exactly as run_civis_retrieval_v1.py lines 136-139.

    Returns (passages, sha256_of_chunks_csv).
    """
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    docs, _mapping, _excluded = cv.canonical_documents(df)
    text = {d.doc_key: d.text for d in docs}
    ch = pd.read_csv(OUT / "chunks.csv", keep_default_na=False)
    passages = [
        SPEC["passage_prefix"] + text[k][a:b]
        for k, a, b in zip(ch.doc_key, ch.start, ch.end)
    ]
    return passages, _sha256(OUT / "chunks.csv")


# ── checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt_paths(start: int) -> tuple[Path, Path]:
    return CKPT_DIR / f"shard_{start:06d}.npy", CKPT_DIR / f"shard_{start:06d}.json"


def _validate_shard(start: int, end: int, cfg_hash: str) -> tuple[bool, str]:
    """Return (ok, reason_string).

    A shard is considered valid only when:
      - both .npy and .json exist
      - config_hash, start, end, and dim match the current run's values
      - finite_ok and norms_ok flags are True in the stored metadata
      - SHA256 of the .npy matches the stored checksum
      - the loaded array's shape equals (end - start, SPEC["dim"])

    Any single check failure returns False with a human-readable reason.
    """
    npy_p, json_p = _ckpt_paths(start)

    if not json_p.exists() and not npy_p.exists():
        return False, "not yet encoded"
    if not json_p.exists():
        # Process was killed between os.replace and json write.
        return False, "orphan .npy (no completion marker); will re-encode"
    if not npy_p.exists():
        return False, "missing .npy despite completion marker; will re-encode"

    try:
        meta = json.loads(json_p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot parse metadata: {exc}"

    if meta.get("config_hash") != cfg_hash:
        return False, f"config mismatch (stored {meta.get('config_hash')!r})"
    if meta.get("start") != start or meta.get("end") != end:
        return False, (
            f"index mismatch: metadata ({meta.get('start')}, {meta.get('end')}) "
            f"!= expected ({start}, {end})"
        )
    if meta.get("dim") != SPEC["dim"]:
        return False, f"dim mismatch: stored {meta.get('dim')} != expected {SPEC['dim']}"
    if not meta.get("finite_ok"):
        return False, "finite_ok=False in stored metadata"
    if not meta.get("norms_ok"):
        return False, "norms_ok=False in stored metadata"

    actual_sha = _sha256(npy_p)
    if actual_sha != meta.get("emb_sha256", ""):
        return False, "SHA256 mismatch — file modified after checkpoint was written"

    # Shape check via memory-mapped read (avoids loading the full array).
    emb = np.load(npy_p, mmap_mode="r")
    if emb.shape != (end - start, SPEC["dim"]):
        return False, f"shape {emb.shape} != expected ({end - start}, {SPEC['dim']})"

    return True, "ok"


def _delete_shard(start: int) -> None:
    """Remove both checkpoint files for a stale or corrupt shard."""
    for p in _ckpt_paths(start):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _save_shard(
    start: int,
    end: int,
    emb: np.ndarray,
    truncated: int,
    cfg_hash: str,
) -> None:
    """Atomically save a float32 shard with validation metadata.

    Raises ValueError if emb contains non-finite values (checked before writing).
    """
    finite_ok = bool(np.isfinite(emb).all())
    if not finite_ok:
        raise ValueError(f"Non-finite values in shard {start}–{end}; not saving")

    norms = np.linalg.norm(emb, axis=1)
    norms_ok = bool(np.allclose(norms, 1.0, atol=1e-3))

    npy_p, json_p = _ckpt_paths(start)
    tmp_npy = npy_p.with_suffix(".tmp.npy")

    np.save(tmp_npy, emb.astype(np.float32))
    os.replace(tmp_npy, npy_p)          # atomic: .npy is now present

    # Write completion marker only after .npy is in place.
    json_p.write_text(
        json.dumps(
            {
                "start": start,
                "end": end,
                "dim": int(emb.shape[1]),
                "shape": list(emb.shape),
                "dtype": "float32",
                "config_hash": cfg_hash,
                "finite_ok": finite_ok,
                "norms_ok": norms_ok,
                "norm_min": round(float(norms.min()), 6),
                "norm_max": round(float(norms.max()), 6),
                "norm_mean": round(float(norms.mean()), 6),
                "truncated": truncated,
                "emb_sha256": _sha256(npy_p),
                "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ── metadata recovery ─────────────────────────────────────────────────────────

def _recover_metadata(*, shard_size: int) -> int:
    """
    EMB_P is present but META_P is missing — the process was killed between
    os.replace (which atomically placed EMB_P) and _write_json_new (META_P).

    Verify the embedding against the current corpus; if all checks pass, write
    a recovered META_P.  Any verification failure → return 5 without touching files.
    """
    import shutil

    import psutil

    print(
        f"[RECOVER] {EMB_P.name} present, {META_P.name} absent — "
        "likely killed after assembly. Verifying embedding before writing metadata …"
    )

    # 1. Load corpus to derive provenance.
    try:
        passages, chunks_sha256 = _load_passages()
    except Exception as exc:  # noqa: BLE001
        print(f"[STOP] Cannot load corpus for provenance check: {exc}", file=sys.stderr)
        return 5
    n_total = len(passages)
    cfg_hash = _config_hash(chunks_sha256)
    del passages

    # 2. Verify embedding (shape, finite, L2-normalised).
    emb = np.load(EMB_P).astype(np.float32)
    if emb.shape != (n_total, SPEC["dim"]):
        print(
            f"[STOP] shape {emb.shape} ≠ expected ({n_total}, {SPEC['dim']}); "
            "provenance check failed.",
            file=sys.stderr,
        )
        return 5
    if not np.isfinite(emb).all():
        print("[STOP] non-finite values in EMB_P; provenance check failed.", file=sys.stderr)
        return 5
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(
            f"[STOP] L2 norms not ≈ 1.0 (mean={norms.mean():.6f}); "
            "provenance check failed.",
            file=sys.stderr,
        )
        return 5
    del emb, norms

    # 3. Sum truncation counts from surviving checkpoints (best-effort).
    total_truncated: int | str = "unknown"
    n_shards = math.ceil(n_total / shard_size)
    t_sum, all_ok = 0, True
    for i in range(n_shards):
        s = i * shard_size
        e = min(s + shard_size, n_total)
        ok_ck, _ = _validate_shard(s, e, cfg_hash)
        if not ok_ck:
            all_ok = False
            break
        _, json_p_ck = _ckpt_paths(s)
        t_sum += json.loads(json_p_ck.read_text(encoding="utf-8")).get("truncated", 0)
    if all_ok:
        total_truncated = t_sum

    # 4. Write recovered metadata — only reachable if all verifications passed.
    _write_json_new(
        META_P,
        {
            "encoder": ENCODER_NAME,
            "model_id": SPEC["model_id"],
            "status": "ok (recovered)",
            "dim": SPEC["dim"],
            "dtype_saved": "float16",
            "rows": n_total,
            "chunks_sha256": chunks_sha256,
            "max_length_tokens": MAX_LEN,
            "chunks_truncated_at_max_length": total_truncated,
            "passage_prefix": SPEC["passage_prefix"],
            "batch_size_used": "unknown (recovered after interruption)",
            "shard_size_arg": shard_size,
            "embeddings_sha256": _sha256(EMB_P),
            "embeddings_mb": round(EMB_P.stat().st_size / 2**20, 2),
            "model_licence": SPEC["licence"],
            "config_hash": cfg_hash,
            "resources_after": {
                "disk_free_mb": shutil.disk_usage(ROOT).free // 2**20,
                "ram_available_mb": psutil.virtual_memory().available // 2**20,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "indian/scripts/run_civis_e5_chunked.py",
            "note": (
                "metadata recovered after interrupted assembly; "
                "embedding verified: shape, finite, L2-norms ≈ 1.0, corpus hash"
            ),
        },
    )
    print(f"[RECOVER] {META_P.name} written; embedding verified OK.")
    return 0


# ── embedding run ──────────────────────────────────────────────────────────────

def run_embed(
    *,
    batch_size: int,
    shard_size: int,
    cleanup: bool,
    dry_run: bool,
) -> int:
    # Developer-facing assertion: output paths must not appear in the protected list.
    for pp in _PROTECTED:
        if pp in (EMB_P, META_P):
            raise RuntimeError(
                f"BUG: {pp.name} is both a protected path and a target output"
            )

    # Write-once guard for the final artifacts.
    if EMB_P.exists() and META_P.exists():
        print(f"[SKIP] {EMB_P.name} already exists; nothing to do.")
        return 0
    if EMB_P.exists() and not META_P.exists():
        return _recover_metadata(shard_size=shard_size)
    if META_P.exists():
        print(
            f"[STOP] {META_P.name} exists but {EMB_P.name} does not — "
            "inconsistent state; resolve manually.",
            file=sys.stderr,
        )
        return 5

    # Weights check (no download).
    w = cv.cached_weights(SPEC["model_id"])
    if w is None:
        print(
            f"[STOP] No local weights for {SPEC['model_id']}. "
            "Do not download here; resolve the cache path manually.",
            file=sys.stderr,
        )
        return 3

    # RAM preflight (estimate, not a guarantee).
    import psutil

    weights_mb = w.stat().st_size // 2**20
    guard_mb = int(weights_mb * 1.25) + 200
    batch_hidden_mb = batch_size * MAX_LEN * SPEC["dim"] * 4 // 2**20
    shard_accum_mb = shard_size * SPEC["dim"] * 4 // 2**20
    # conservative: shard float32 + one vstack copy + hidden states + misc
    peak_est_mb = guard_mb + batch_hidden_mb + shard_accum_mb * 2 + 60
    avail_mb = psutil.virtual_memory().available // 2**20

    print(
        f"[RAM]  available={avail_mb} MB | model-load guard={guard_mb} MB | "
        f"encode-peak estimate={peak_est_mb} MB"
    )
    print(
        "       Passing the guard does not guarantee sufficient memory; "
        "peak estimate includes shard accumulation and transformer hidden states."
    )
    if avail_mb < guard_mb:
        print(
            f"[STOP] {avail_mb} MB available < {guard_mb} MB load guard.",
            file=sys.stderr,
        )
        return 4
    if avail_mb < peak_est_mb:
        print(
            f"[WARN] Only {avail_mb - guard_mb} MB above load guard; "
            "encoding may page to disk and run slowly."
        )

    # Load corpus.
    print("[LOAD] Reading corpus and building passage list …")
    t_load = time.time()
    passages, chunks_sha256 = _load_passages()
    n_total = len(passages)
    cfg_hash = _config_hash(chunks_sha256)
    n_shards = math.ceil(n_total / shard_size)
    print(
        f"[LOAD] {n_total} passages | shard_size={shard_size} → {n_shards} shards | "
        f"batch_size={batch_size} | cfg_hash={cfg_hash} ({time.time() - t_load:.1f}s)"
    )

    if dry_run:
        print("[DRY-RUN] Config verified; no encoding or file writes will occur.")
        return 0

    # Inventory checkpoints.
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shards_to_do: list[tuple[int, int]] = []
    shards_reused = 0
    for i in range(n_shards):
        s = i * shard_size
        e = min(s + shard_size, n_total)
        ok, reason = _validate_shard(s, e, cfg_hash)
        if ok:
            shards_reused += 1
        else:
            if reason not in ("not yet encoded",):
                print(
                    f"  [CKPT] shard {s:06d}–{e:06d}: {reason} → "
                    "deleting stale files and re-encoding"
                )
                _delete_shard(s)
            shards_to_do.append((s, e))

    print(
        f"[CKPT] {shards_reused}/{n_shards} shards already valid; "
        f"{len(shards_to_do)} to encode"
    )

    if shards_to_do:
        import torch
        from run_full_corpus_clustering import load_encoder_low_memory

        print("[MODEL] Loading e5-base-v2 with low-memory loader …")
        t_model = time.time()
        tok, model, _ = load_encoder_low_memory(SPEC["model_id"])
        torch.set_num_threads(6)
        print(f"[MODEL] Loaded in {time.time() - t_model:.1f}s")

        t_enc = time.time()
        for shard_idx, (s, e) in enumerate(shards_to_do):
            shard_passages = passages[s:e]
            last_log: list[float] = [time.time()]

            # Capture by value with default arguments to avoid late-binding.
            def _prog(
                n: int,
                _s: int = s,
                _total: int = n_total,
                _t_enc: float = t_enc,
            ) -> None:
                if time.time() - last_log[0] > 60:
                    last_log[0] = time.time()
                    abs_done = _s + n
                    elapsed = time.time() - _t_enc
                    rate = abs_done / elapsed if elapsed > 0 else 0.0
                    print(f"  {abs_done}/{_total} chunks, {rate:.1f}/s", flush=True)

            t_shard = time.time()
            emb, trunc = cv.encode(
                shard_passages,
                tok,
                model,
                max_length=MAX_LEN,
                batch_size=batch_size,
                progress=_prog,
            )
            _save_shard(s, e, emb, trunc, cfg_hash)
            del emb

            shard_elapsed = time.time() - t_shard
            rate = (e - s) / shard_elapsed if shard_elapsed > 0 else 0.0
            print(
                f"  [SHARD] {s:06d}–{e:06d} saved "
                f"({shard_idx + 1}/{len(shards_to_do)}) {rate:.1f}/s trunc={trunc}",
                flush=True,
            )

        del model, tok

    # Assemble final matrix.
    print("[ASSEMBLE] Loading and stacking shards …")
    t_asm = time.time()
    parts: list[np.ndarray] = []
    total_truncated = 0
    for i in range(n_shards):
        s = i * shard_size
        e = min(s + shard_size, n_total)
        ok, reason = _validate_shard(s, e, cfg_hash)
        if not ok:
            print(
                f"[STOP] Shard {s:06d}–{e:06d} invalid at assembly: {reason}",
                file=sys.stderr,
            )
            return 6
        _npy_p, json_p = _ckpt_paths(s)
        total_truncated += json.loads(json_p.read_text(encoding="utf-8")).get("truncated", 0)
        parts.append(np.load(_npy_p))

    full = np.vstack(parts)
    del parts

    # Final validation.
    if full.shape != (n_total, SPEC["dim"]):
        print(
            f"[STOP] assembled shape {full.shape} != ({n_total}, {SPEC['dim']})",
            file=sys.stderr,
        )
        return 7
    if not np.isfinite(full).all():
        print("[STOP] non-finite values in assembled matrix", file=sys.stderr)
        return 7
    norms = np.linalg.norm(full, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(
            f"[STOP] L2 norms not ≈ 1.0: mean={norms.mean():.6f} "
            f"min={norms.min():.6f} max={norms.max():.6f}",
            file=sys.stderr,
        )
        return 7

    print(
        f"[ASSEMBLE] shape={full.shape} finite=True "
        f"norms_mean={norms.mean():.6f} ({time.time() - t_asm:.1f}s)"
    )

    # Atomically publish the final .npy, then write the metadata JSON.
    full_f16 = full.astype(np.float16)
    del full, norms

    tmp_emb = EMB_P.with_suffix(".tmp.npy")
    np.save(tmp_emb, full_f16)
    os.replace(tmp_emb, EMB_P)

    avail_after = psutil.virtual_memory().available // 2**20
    _write_json_new(
        META_P,
        {
            "encoder": ENCODER_NAME,
            "model_id": SPEC["model_id"],
            "status": "ok",
            "dim": SPEC["dim"],
            "dtype_saved": "float16",
            "rows": n_total,
            "chunks_sha256": chunks_sha256,
            "max_length_tokens": MAX_LEN,
            "chunks_truncated_at_max_length": total_truncated,
            "passage_prefix": SPEC["passage_prefix"],
            "batch_size_used": batch_size,
            "shard_size": shard_size,
            "n_shards": n_shards,
            "shards_reused_on_this_run": shards_reused,
            "embeddings_sha256": _sha256(EMB_P),
            "embeddings_mb": round(EMB_P.stat().st_size / 2**20, 2),
            "model_licence": SPEC["licence"],
            "config_hash": cfg_hash,
            "total_runtime_s": round(time.time() - t_load, 1),
            "resources_after": {
                "disk_free_mb": shutil.disk_usage(ROOT).free // 2**20,
                "ram_available_mb": avail_after,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "indian/scripts/run_civis_e5_chunked.py",
            "note": (
                "checkpointed run; artifacts are compatible with "
                "run_civis_retrieval_v1.py evaluate"
            ),
        },
    )

    size_mb = EMB_P.stat().st_size // 2**20
    print(f"[DONE] {EMB_P.name} ({size_mb} MB) | metadata: {META_P.name}")

    if cleanup:
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        print(f"[CLEANUP] Removed {CKPT_DIR}")

    return 0


# ── validation-only mode ───────────────────────────────────────────────────────

def run_validate() -> int:
    """Inspect existing checkpoints and final output; do not encode."""
    print("=== e5-base-v2 CIVIS embedding inspection ===")

    if EMB_P.exists() and META_P.exists():
        meta = json.loads(META_P.read_text(encoding="utf-8"))
        size_mb = EMB_P.stat().st_size // 2**20
        print(f"Final .npy : {EMB_P}  ({size_mb} MB)")
        print(
            f"  status={meta.get('status')}  rows={meta.get('rows')}  "
            f"dim={meta.get('dim')}"
        )
        actual_sha = _sha256(EMB_P)
        sha_ok = actual_sha == meta.get("embeddings_sha256", "")
        print(f"  SHA256: {'MATCH' if sha_ok else 'MISMATCH'}")
        if sha_ok:
            emb = np.load(EMB_P).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1)
            print(
                f"  shape={emb.shape}  finite={np.isfinite(emb).all()}  "
                f"norms_mean={norms.mean():.6f}"
            )
        return 0

    if EMB_P.exists() or META_P.exists():
        print(
            "[WARN] Only one of the final output files is present — "
            "inconsistent state."
        )

    if not CKPT_DIR.exists():
        print("[INFO] No checkpoint directory; no partial progress.")
        return 1

    # Load corpus to derive the expected config hash.
    cfg_hash: str | None = None
    try:
        passages, chunks_sha256 = _load_passages()
        cfg_hash = _config_hash(chunks_sha256)
        del passages
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load corpus (needed for config hash): {exc}")

    shard_jsons = sorted(CKPT_DIR.glob("shard_*.json"))
    print(f"\nCheckpoint dir: {CKPT_DIR}")
    print(f"  {len(shard_jsons)} shard metadata file(s) found")

    ok_count = bad_count = 0
    for json_p in shard_jsons:
        meta = json.loads(json_p.read_text(encoding="utf-8"))
        s, e = meta.get("start"), meta.get("end")
        if cfg_hash is not None and s is not None and e is not None:
            ok, reason = _validate_shard(s, e, cfg_hash)
            label = "OK" if ok else f"BAD: {reason}"
            if ok:
                ok_count += 1
            else:
                bad_count += 1
        else:
            label = "UNKNOWN (corpus unavailable)"
        print(f"  shard {s!s:>6}–{e!s:<6}: {label}")

    print(f"\nSummary: {ok_count} valid, {bad_count} invalid/unknown")
    return 0 if bad_count == 0 else 1


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Model forward-pass batch size (default 8; smaller = less peak RAM)",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=500,
        metavar="N",
        help="Chunks per checkpoint shard (default 500)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Check config and RAM; print plan; do not encode or write",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Inspect existing checkpoints and output; do not encode",
    )
    ap.add_argument(
        "--cleanup-ckpts",
        action="store_true",
        help="Delete checkpoint directory after successful assembly (default: keep)",
    )
    a = ap.parse_args(argv)

    if not (1 <= a.batch_size <= 64):
        ap.error("--batch-size must be between 1 and 64")
    if not (10 <= a.shard_size <= 5000):
        ap.error("--shard-size must be between 10 and 5000")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.validate:
        return run_validate()
    return run_embed(
        batch_size=a.batch_size,
        shard_size=a.shard_size,
        cleanup=a.cleanup_ckpts,
        dry_run=a.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
