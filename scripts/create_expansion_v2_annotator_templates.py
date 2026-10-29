"""
scripts/create_expansion_v2_annotator_templates.py
==================================================
Creates two identical, blank, 400-row annotator templates from the v2 pool:

    data/annotation/expansion_v2_annotator_1.csv   (working copy source for annotator_A)
    data/annotation/expansion_v2_annotator_2.csv   (working copy source for annotator_B)
    data/annotation/expansion_v2_templates_manifest.json

The templates are byte-for-byte copies of ``expansion_pool_400_v2.csv`` (same 20 columns, same row
order). Rows 1-50 (``pilot_*``) are the pilot and rows 51-400 (``expansion_v2_*``) are the 350
additional records; the pilot rows must never be counted among the 350.

The script verifies the pool against its manifest first, refuses to overwrite anything, and never
creates labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_annotation_expansion_pool_v2 as builder  # noqa: E402

ANNOTATION = builder.ANNOTATION
TEMPLATE_1 = ANNOTATION / "expansion_v2_annotator_1.csv"
TEMPLATE_2 = ANNOTATION / "expansion_v2_annotator_2.csv"
TEMPLATES_MANIFEST = ANNOTATION / "expansion_v2_templates_manifest.json"


def create_templates(
    pool_path: Path = builder.POOL_PATH,
    manifest_path: Path = builder.MANIFEST_PATH,
    template_1: Path = TEMPLATE_1,
    template_2: Path = TEMPLATE_2,
    templates_manifest: Path = TEMPLATES_MANIFEST,
) -> dict:
    outputs = [template_1, template_2, templates_manifest]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(existing))

    pool_bytes = pool_path.read_bytes()
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["outputs_sha256"].get(pool_path.name) != pool_sha:
        raise ValueError("Pool SHA-256 does not match its manifest; refusing to create templates.")
    pool = pd.read_csv(pool_path, dtype=str, keep_default_na=False)
    if len(pool) != manifest["stats"]["pool_rows"]:
        raise ValueError("Pool row count does not match the manifest.")
    if (pool[builder.ANNOTATION_COLUMNS] != "").any().any():
        raise ValueError("The pool already contains annotation values; refusing to create blank templates.")
    n_pilot = manifest["stats"]["pilot_rows"]
    if not pool.pilot_id.iloc[:n_pilot].str.startswith("pilot_").all():
        raise ValueError("First rows are not the pilot segment.")

    for path in (template_1, template_2):
        with open(path, "xb") as f:      # exclusive creation: never overwrites
            f.write(pool_bytes)
    info = {
        "purpose": "Blank annotator templates for the v2 pool. Copy before annotating; never edit in place.",
        "source_pool": pool_path.name,
        "source_pool_sha256": pool_sha,
        "templates": {template_1.name: pool_sha, template_2.name: pool_sha},
        "rows": len(pool),
        "pilot_segment_rows": n_pilot,
        "expansion_segment_rows": len(pool) - n_pilot,
        "segment_rule": "pilot_id starting with 'pilot_' = pilot (rows 1-50); 'expansion_v2_' = the 350 additional records. "
                        "Pilot rows are not part of the 350.",
        "completed_file_names": {
            "annotator_A": "data/annotation/completed/expansion_v2_annotator_A_completed.csv",
            "annotator_B": "data/annotation/completed/expansion_v2_annotator_B_completed.csv",
        },
        "labels_created": 0,
    }
    with open(templates_manifest, "x", encoding="utf-8", newline="") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info


def main() -> int:
    try:
        info = create_templates()
    except (FileExistsError, ValueError, FileNotFoundError) as exc:
        print(f"[STOP] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Created {TEMPLATE_1.name} and {TEMPLATE_2.name} ({info['rows']} blank rows each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
