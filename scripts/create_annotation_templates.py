"""
scripts/create_annotation_templates.py
======================================
Creates independent annotation template files for two human annotators
from the validated pilot dataset (data/annotation/pilot_annotation_50.csv):
  - data/annotation/annotator_1.csv
  - data/annotation/annotator_2.csv

Rules:
  1. Identical 50 pilot records in identical order.
  2. Identical provenance columns preserved.
  3. All human annotation fields (stance, concern, requested_action,
     evidence_span, review_notes, annotator_id) strictly blank.
  4. Separate files to prevent accidental leakage during independent labeling.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"
PILOT_CSV = ANNOTATION_DIR / "pilot_annotation_50.csv"
ANNOTATOR_1_CSV = ANNOTATION_DIR / "annotator_1.csv"
ANNOTATOR_2_CSV = ANNOTATION_DIR / "annotator_2.csv"

HUMAN_ANNOTATION_COLUMNS = [
    "stance",
    "concern",
    "requested_action",
    "evidence_span",
    "review_notes",
    "annotator_id",
]


def create_templates() -> None:
    if not PILOT_CSV.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {PILOT_CSV}")

    df = pd.read_csv(PILOT_CSV, dtype=str, keep_default_na=False)

    # Ensure all annotation columns are blank
    for col in HUMAN_ANNOTATION_COLUMNS:
        df[col] = ""

    # Write annotator 1 template
    df.to_csv(ANNOTATOR_1_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Created {ANNOTATOR_1_CSV.relative_to(PROJECT_ROOT)} with {len(df)} records.")

    # Write annotator 2 template
    df.to_csv(ANNOTATOR_2_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Created {ANNOTATOR_2_CSV.relative_to(PROJECT_ROOT)} with {len(df)} records.")


if __name__ == "__main__":
    create_templates()
