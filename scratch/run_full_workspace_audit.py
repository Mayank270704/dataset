"""
Full workspace dataset inspection script.
Gathers verifiable data for all dataset files in the workspace.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path('.')

def audit_file(path: Path):
    res = {
        'path': str(path),
        'filename': path.name,
        'size_bytes': path.stat().st_size,
    }
    
    # Check extension
    suffix = path.suffix.lower()
    
    # Try reading as CSV or TSV or JSONL or TXT
    if suffix in ['.csv', '.tsv']:
        sep = '\t' if suffix == '.tsv' else ','
        # Try utf-8 then cp1252/latin-1
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']:
            try:
                # If large (> 20MB), read sample or handle carefully
                if path.stat().st_size > 20_000_000:
                    # just read header and line count
                    header_df = pd.read_csv(path, sep=sep, encoding=enc, nrows=10, low_memory=False)
                    with open(path, 'rb') as f:
                        line_count = sum(1 for _ in f)
                    res['row_count'] = line_count - 1 # approximate header
                    res['columns'] = list(header_df.columns)
                    res['encoding'] = enc
                    res['is_large'] = True
                    return res
                else:
                    df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                    res['encoding'] = enc
                    break
            except Exception:
                continue
        if df is not None:
            res['row_count'] = len(df)
            res['columns'] = [str(c) for c in df.columns]
            res['missing_values'] = df.isna().sum().to_dict()
            res['total_missing'] = int(df.isna().sum().sum())
            res['exact_duplicate_rows'] = int(df.duplicated().sum())
            
            # Check blank text across string columns
            blank_count = 0
            text_dups = {}
            for col in df.select_dtypes(include=['object']):
                s = df[col].dropna().astype(str)
                blanks = (s.str.strip() == '').sum()
                blank_count += int(blanks)
                # check duplicates in text column if plausible text
                if 'text' in col.lower() or 'sentence' in col.lower() or 'comment' in col.lower():
                    clean_s = s[s.str.strip() != ''].str.strip().str.lower()
                    n_dup_groups = (clean_s.value_counts() > 1).sum()
                    text_dups[col] = int(n_dup_groups)
            res['blank_text_cells'] = blank_count
            res['text_duplicate_groups'] = text_dups
            
            # Language & label distribution
            for lang_col in ['language', 'language_label', 'source_language', 'lang']:
                if lang_col in df.columns:
                    res['language_distribution'] = df[lang_col].value_counts().head(5).to_dict()
            for lab_col in ['label', 'sentiment', 'target', 'stance', 'primary_emotion']:
                if lab_col in df.columns:
                    res['label_distribution'] = df[lab_col].value_counts().head(5).to_dict()
            return res
            
    elif suffix == '.jsonl':
        lines = 0
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    lines += 1
                    if len(records) < 5000:
                        records.append(json.loads(line_str))
        df = pd.DataFrame(records)
        res['row_count'] = lines
        res['columns'] = list(df.columns)
        res['encoding'] = 'utf-8'
        res['exact_duplicate_rows'] = int(df.duplicated().sum())
        if 'language_label' in df.columns:
            res['language_distribution'] = df['language_label'].value_counts().head(5).to_dict()
        if 'label' in df.columns:
            res['label_distribution'] = df['label'].value_counts().head(5).to_dict()
        return res

    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            res['row_count'] = len(data)
            if data and isinstance(data[0], dict):
                res['columns'] = list(data[0].keys())
        elif isinstance(data, dict):
            res['row_count'] = 1
            res['columns'] = list(data.keys())
        res['encoding'] = 'utf-8'
        return res
        
    elif suffix == '.txt':
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            line_count = sum(1 for _ in f)
        res['row_count'] = line_count
        res['columns'] = ['raw_lines']
        res['encoding'] = 'utf-8'
        return res

    return res

if __name__ == '__main__':
    # test on annotation & interim files
    sample_files = [
        ROOT / 'data' / 'annotation' / 'pilot_annotation_50.csv',
        ROOT / 'data' / 'annotation' / 'annotator_1.csv',
        ROOT / 'data' / 'annotation' / 'annotator_2.csv',
        ROOT / 'data' / 'clustering' / 'cluster_labels.csv',
        ROOT / 'data' / 'clustering' / 'cluster_topics.csv',
        ROOT / 'data' / 'embeddings' / 'corpus_metadata.csv',
    ]
    for sf in sample_files:
        print(audit_file(sf))
