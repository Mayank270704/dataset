"""
indian/scripts/integrity_check.py
=================================
Snapshot and verify SHA-256 hashes of protected files: everything under indian/ that exists at snapshot time
(raw downloads, processed datasets, earlier experiment artifacts, code, reports), plus the original project code and
research artifacts (models/, scripts/, app/, data/, reports/, frontend/frontend/src). New files created later are
allowed; any protected file that is modified or deleted is reported.

    python -I indian/scripts/integrity_check.py snapshot indian/logs/integrity/baseline_pipeline_v1.json
    python -I indian/scripts/integrity_check.py verify   indian/logs/integrity/baseline_pipeline_v1.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ["indian", "models", "scripts", "app", "data", "reports", "frontend/frontend/src"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "node_modules", ".next"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def files():
    for s in SCOPES:
        base = ROOT / s
        if not base.exists():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not (SKIP_PARTS & set(p.parts)) and not rel.startswith("indian/logs/"):
                yield rel, p


def main(argv: list[str]) -> int:
    mode, path = argv[0], ROOT / argv[1]
    if mode == "snapshot":
        if path.exists():
            print(f"[STOP] {path} exists")
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        snap = {rel: sha(p) for rel, p in files()}
        path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scopes": SCOPES,
                                    "files": len(snap), "sha256": snap}, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"snapshot": str(path.relative_to(ROOT)), "files": len(snap)}))
        return 0
    base = json.loads(path.read_text(encoding="utf-8"))["sha256"]
    changed, missing = [], []
    for rel, h in base.items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != h:
            changed.append(rel)
    now = {rel for rel, _ in files()}
    res = {"verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "baseline_files": len(base),
           "unchanged": len(base) - len(changed) - len(missing), "changed": changed, "missing": missing,
           "new_files_since_baseline": len(now - set(base))}
    print(json.dumps(res, indent=1))
    return 0 if not changed and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
