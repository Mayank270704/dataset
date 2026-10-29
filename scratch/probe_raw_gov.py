"""Probe raw + government source-clean files."""
import pandas as pd
import json
from pathlib import Path

ROOT = Path('.')

# Government source-clean files
gov_files = [
    ROOT / 'data' / 'processed' / 'consultation_compilation_2017_18_source_clean.csv',
    ROOT / 'data' / 'processed' / 'get_involved_questionnaire_2017_18_source_clean.csv',
]
for p in gov_files:
    try:
        df = pd.read_csv(p, encoding='utf-8', low_memory=False)
        print(f'{p.name}: {len(df)} rows, {len(df.columns)} cols')
        print(f'  cols: {list(df.columns[:10])}')
        if 'exclusion_reason' in df.columns:
            print(f'  exclusion_reason: {df["exclusion_reason"].fillna("OK").value_counts().to_dict()}')
        if 'source_language' in df.columns:
            print(f'  source_language: {df["source_language"].value_counts().head(6).to_dict()}')
        miss = df.isna().sum()
        miss = miss[miss > 0]
        print(f'  missing cols: {len(miss)} cols with NaN')
        # Check for exact duplicate rows
        dups = df.duplicated().sum()
        print(f'  exact dup rows: {dups}')
        print()
    except Exception as e:
        print(f'ERROR {p.name}: {e}')

# Raw files (small ones)
print('=== RAW FILES ===')
raw_files = [
    (ROOT / 'ap2016-18_individual_comments.csv', 'latin-1'),
    (ROOT / 'PolicyLens_Hindi_English_Hinglish_Sentiment.csv', 'utf-8'),
]
for p, enc in raw_files:
    try:
        df = pd.read_csv(p, encoding=enc, low_memory=False)
        print(f'{p.name}: {len(df)} rows, {len(df.columns)} cols')
        print(f'  cols: {list(df.columns[:8])}')
        dups = df.duplicated().sum()
        miss = df.isna().sum()
        print(f'  exact dup rows: {dups}')
        print(f'  missing cols: {miss[miss>0].to_dict()}')
    except Exception as e:
        try:
            df = pd.read_csv(p, encoding='latin-1', low_memory=False)
            print(f'{p.name} (latin-1): {len(df)} rows')
        except Exception as e2:
            print(f'ERROR: {e2}')
    print()

# Sentiment140 test (small)
p = ROOT / 'trainingandtestdata' / 'testdata.manual.2009.06.14.csv'
try:
    df = pd.read_csv(p, encoding='latin-1', header=None,
                     names=['target','ids','date','flag','user','text'])
    print(f'{p.name}: {len(df)} rows, target dist: {df["target"].value_counts().to_dict()}')
    print(f'  exact dups: {df.duplicated().sum()}')
    print(f'  blank text: {(df["text"].str.strip() == "").sum()}')
except Exception as e:
    print(f'ERROR: {e}')
print()

# hinemo9 raw jsonl (just probe sizes)
hinemo_base = ROOT / 'hinemo9-v1.0.1' / 'im-rakeshsahani-hinemo9-764fb36' / 'data'
for jf in ['train.jsonl', 'dev.jsonl', 'test.jsonl', 'gold.jsonl', 'full.jsonl']:
    p = hinemo_base / jf
    if p.exists():
        records = []
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        print(f'{p.name}: {len(df)} rows, cols={list(df.columns[:8])}')
        if 'label' in df.columns:
            print(f'  label dist: {df["label"].value_counts().head(5).to_dict()}')
        if 'language_label' in df.columns:
            print(f'  lang: {df["language_label"].value_counts().head(4).to_dict()}')
        if 'label_source' in df.columns:
            print(f'  label_source: {df["label_source"].value_counts().to_dict()}')
        dups = df.duplicated().sum()
        print(f'  exact dup rows: {dups}')
