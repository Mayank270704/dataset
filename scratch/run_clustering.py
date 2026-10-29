import sys; sys.path.insert(0, '.')
from pathlib import Path
from scripts.concern_clustering import run_clustering
import json

labels_df, topics_df, metrics = run_clustering(
    embeddings_dir=Path('data/embeddings'),
    output_dir=Path('data/clustering'),
    force_recompute=True,
)

print('=== CLUSTER TOPICS ===')
for _, row in topics_df.iterrows():
    cid = row['cluster_id']
    label = 'noise' if cid == -1 else f'cluster {cid}'
    size = row['size']
    kws = row['top_keywords']
    print(f'  {label:>10s}  size={str(size):>4s}  {kws}')

print()
print('=== QUALITY METRICS ===')
for k, v in metrics.items():
    print(f'  {k}: {v}')

print()
print('=== ARTEFACTS ===')
for f in sorted(Path('data/clustering').iterdir()):
    print(f'  {f.name}  ({f.stat().st_size:,} bytes)')
