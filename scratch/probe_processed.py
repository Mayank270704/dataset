"""Quick probe of processed files."""
import pandas as pd
from pathlib import Path

ROOT = Path('.')
proc = ROOT / 'data' / 'processed'

for p in sorted(proc.iterdir()):
    if p.suffix != '.csv':
        continue
    if p.stat().st_size > 50_000_000:
        print(f'SKIP (large): {p.name}')
        continue
    try:
        full = pd.read_csv(p, encoding='utf-8', low_memory=False)
        print(f'{p.name}: {len(full)} rows, {len(full.columns)} cols')
        print(f'  columns: {list(full.columns[:12])}')
        if 'exclusion_reason' in full.columns:
            print(f'  exclusion_reason: {full["exclusion_reason"].fillna("OK").value_counts().to_dict()}')
        if 'label' in full.columns:
            print(f'  label dist: {full["label"].value_counts().head(5).to_dict()}')
        if 'source_language' in full.columns:
            print(f'  source_language: {full["source_language"].value_counts().head(5).to_dict()}')
        if 'language_label' in full.columns:
            print(f'  language_label: {full["language_label"].value_counts().head(5).to_dict()}')
        if 'language' in full.columns:
            print(f'  language: {full["language"].value_counts().head(5).to_dict()}')
        miss = full.isna().sum()
        print(f'  missing any: {miss[miss > 0].to_dict()}')
    except Exception as e:
        print(f'ERROR {p.name}: {e}')
    print()
