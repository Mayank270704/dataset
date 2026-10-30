"""
Tests for indian/scripts/run_civis_e5_chunked.py.

All tests are offline-only: no model loading, no corpus reading,
no embedding, no network access, no modifications to existing artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

# ── import the module under test ──────────────────────────────────────────────

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "indian"))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.run_civis_e5_chunked as ec  # noqa: E402


# ── 1. Static configuration sanity ────────────────────────────────────────────

def test_encoder_name_is_e5_not_minilm():
    assert "e5" in ec.ENCODER_NAME.lower()
    assert "minilm" not in ec.ENCODER_NAME.lower()


def test_emb_p_filename_contains_encoder_name():
    assert ec.ENCODER_NAME in ec.EMB_P.name


def test_meta_p_filename_contains_encoder_name():
    assert ec.ENCODER_NAME in ec.META_P.name


def test_emb_p_and_meta_p_are_not_in_protected():
    assert ec.EMB_P not in ec._PROTECTED
    assert ec.META_P not in ec._PROTECTED


def test_protected_list_includes_minilm_npy():
    names = [p.name for p in ec._PROTECTED]
    assert any("MiniLM" in n and n.endswith(".npy") for n in names)


def test_protected_list_includes_evaluation_metrics():
    names = [p.name for p in ec._PROTECTED]
    assert "metrics.json" in names


def test_protected_list_includes_paired_tests():
    names = [p.name for p in ec._PROTECTED]
    assert "paired_tests.json" in names


def test_max_len_matches_original_script():
    """MAX_LEN must equal 256 — the same constant used in run_civis_retrieval_v1.py."""
    assert ec.MAX_LEN == 256


def test_ckpt_dir_is_under_out():
    assert ec.CKPT_DIR.parent == ec.OUT


def test_spec_dim_is_768():
    """e5-base-v2 produces 768-dimensional embeddings."""
    assert ec.SPEC["dim"] == 768


def test_spec_passage_prefix():
    assert ec.SPEC["passage_prefix"] == "passage: "


def test_emb_p_and_meta_p_share_stem():
    assert ec.EMB_P.stem == ec.META_P.stem


# ── 2. config_hash ─────────────────────────────────────────────────────────────

def test_config_hash_is_deterministic():
    h1 = ec._config_hash("abc123")
    h2 = ec._config_hash("abc123")
    assert h1 == h2


def test_config_hash_length_is_16():
    h = ec._config_hash("abc")
    assert len(h) == 16


def test_config_hash_changes_with_chunks_sha():
    h1 = ec._config_hash("sha_A")
    h2 = ec._config_hash("sha_B")
    assert h1 != h2


def test_config_hash_is_hex():
    h = ec._config_hash("any")
    assert all(c in "0123456789abcdef" for c in h)


# ── 3. _validate_shard: missing / corrupt cases ────────────────────────────────

def test_validate_shard_both_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    ok, reason = ec._validate_shard(0, 500, "cfg0")
    assert not ok
    assert "not yet encoded" in reason


def test_validate_shard_orphan_npy(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    npy_p, _json_p = ec._ckpt_paths(0)
    np.save(npy_p, np.ones((500, 768), dtype=np.float32))
    # no .json written
    ok, reason = ec._validate_shard(0, 500, "cfg0")
    assert not ok
    assert "orphan" in reason


def test_validate_shard_config_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "hash_A")
    ok, reason = ec._validate_shard(0, 10, "hash_B")   # different hash
    assert not ok
    assert "config mismatch" in reason


def test_validate_shard_index_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    # ask for a different end index than what was saved
    ok, reason = ec._validate_shard(0, 20, "cfg1")
    assert not ok
    assert "index mismatch" in reason


def test_validate_shard_finite_flag_false(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    npy_p, json_p = ec._ckpt_paths(0)
    # write a .npy with finite data but mark finite_ok=False in metadata
    emb = np.ones((10, 768), dtype=np.float32)
    np.save(npy_p, emb)
    sha = hashlib.sha256(npy_p.read_bytes()).hexdigest()
    json_p.write_text(json.dumps({
        "start": 0, "end": 10, "dim": 768, "shape": [10, 768],
        "config_hash": "cfg1", "finite_ok": False, "norms_ok": True,
        "norm_min": 1.0, "norm_max": 1.0, "norm_mean": 1.0,
        "truncated": 0, "emb_sha256": sha, "saved_utc": "2026-01-01T00:00:00+00:00",
    }, indent=2) + "\n", encoding="utf-8")
    ok, reason = ec._validate_shard(0, 10, "cfg1")
    assert not ok
    assert "finite_ok" in reason


def test_validate_shard_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    # corrupt the .npy after saving
    npy_p, _ = ec._ckpt_paths(0)
    with open(npy_p, "ab") as f:
        f.write(b"\x00")
    ok, reason = ec._validate_shard(0, 10, "cfg1")
    assert not ok
    assert "SHA256" in reason


def test_validate_shard_shape_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    npy_p, json_p = ec._ckpt_paths(0)
    # save array with wrong number of rows
    emb_wrong = np.ones((5, 768), dtype=np.float32)  # only 5 rows, expect 10
    np.save(npy_p, emb_wrong)
    sha = hashlib.sha256(npy_p.read_bytes()).hexdigest()
    json_p.write_text(json.dumps({
        "start": 0, "end": 10, "dim": 768, "shape": [5, 768],
        "config_hash": "cfg1", "finite_ok": True, "norms_ok": True,
        "norm_min": 1.0, "norm_max": 1.0, "norm_mean": 1.0,
        "truncated": 0, "emb_sha256": sha, "saved_utc": "2026-01-01T00:00:00+00:00",
    }, indent=2) + "\n", encoding="utf-8")
    ok, reason = ec._validate_shard(0, 10, "cfg1")
    assert not ok
    assert "shape" in reason


def test_validate_shard_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    npy_p, json_p = ec._ckpt_paths(0)
    np.save(npy_p, np.ones((10, 768), dtype=np.float32))
    json_p.write_text("not valid json", encoding="utf-8")
    ok, reason = ec._validate_shard(0, 10, "cfg1")
    assert not ok
    assert "cannot parse" in reason


# ── 4. _save_shard ─────────────────────────────────────────────────────────────

def test_save_shard_roundtrip(tmp_path, monkeypatch):
    """Save a synthetic shard; _validate_shard must pass on the same config_hash."""
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 768)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 50, emb, 3, "cfg_roundtrip")
    ok, reason = ec._validate_shard(0, 50, "cfg_roundtrip")
    assert ok, f"Expected valid shard, got: {reason}"


def test_save_shard_produces_two_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.ones((10, 768), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    npy_p, json_p = ec._ckpt_paths(0)
    assert npy_p.exists()
    assert json_p.exists()


def test_save_shard_no_tmp_file_left(tmp_path, monkeypatch):
    """The .tmp.npy file must be removed after os.replace."""
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.ones((10, 768), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    npy_p, _ = ec._ckpt_paths(0)
    tmp_npy = npy_p.with_suffix(".tmp.npy")
    assert not tmp_npy.exists()


def test_save_shard_raises_on_non_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.full((10, 768), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite"):
        ec._save_shard(0, 10, emb, 0, "cfg1")


def test_save_shard_norms_ok_flag(tmp_path, monkeypatch):
    """Embeddings already L2-normalised → norms_ok=True in metadata."""
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    _, json_p = ec._ckpt_paths(0)
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    assert meta["norms_ok"] is True


def test_save_shard_records_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 7, "cfg1")
    _, json_p = ec._ckpt_paths(0)
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    assert meta["truncated"] == 7


def test_save_shard_metadata_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.random.randn(10, 768).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(100, 110, emb, 2, "cfg99")
    _, json_p = ec._ckpt_paths(100)
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    assert meta["start"] == 100
    assert meta["end"] == 110
    assert meta["dim"] == 768
    assert meta["dtype"] == "float32"
    assert meta["config_hash"] == "cfg99"
    assert "emb_sha256" in meta
    assert "saved_utc" in meta


# ── 5. _delete_shard ───────────────────────────────────────────────────────────

def test_delete_shard_removes_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    emb = np.ones((10, 768), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ec._save_shard(0, 10, emb, 0, "cfg1")
    ec._delete_shard(0)
    npy_p, json_p = ec._ckpt_paths(0)
    assert not npy_p.exists()
    assert not json_p.exists()


def test_delete_shard_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CKPT_DIR", tmp_path)
    ec._delete_shard(999)   # nothing exists; must not raise


# ── 6. _write_json_new ─────────────────────────────────────────────────────────

def test_write_json_new_raises_if_exists(tmp_path):
    p = tmp_path / "out.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ec._write_json_new(p, {"key": "value"})


def test_write_json_new_creates_file(tmp_path):
    p = tmp_path / "new.json"
    ec._write_json_new(p, {"k": 1})
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1}


# ── 7. CLI argument validation ────────────────────────────────────────────────

def test_cli_batch_size_zero_is_rejected():
    with pytest.raises(SystemExit):
        ec.main(["--batch-size", "0"])


def test_cli_batch_size_65_is_rejected():
    with pytest.raises(SystemExit):
        ec.main(["--batch-size", "65"])


def test_cli_shard_size_9_is_rejected():
    with pytest.raises(SystemExit):
        ec.main(["--shard-size", "9"])


def test_cli_shard_size_5001_is_rejected():
    with pytest.raises(SystemExit):
        ec.main(["--shard-size", "5001"])
