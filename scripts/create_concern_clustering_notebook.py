"""
scripts/create_concern_clustering_notebook.py
=============================================
Generate notebooks/05_concern_clustering.ipynb.

Run from the project root:
    python scripts/create_concern_clustering_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


def make_md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def make_code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


CELLS = [
    # ── Title ────────────────────────────────────────────────────────────────
    make_md(
        "# PolicyLens — Concern Clustering and Topic Discovery",
        "",
        "**Notebook 05** in the PolicyLens pipeline.",
        "",
        "This notebook implements unsupervised clustering of open-government consultation",
        "comments using the pre-computed MiniLM sentence embeddings produced in",
        "`04_semantic_similarity.ipynb`.  No human stance labels are required.",
        "",
        "## Pipeline",
        "1. Load embeddings and corpus metadata from `data/embeddings/`.",
        "2. Reduce to 5-D with **UMAP** (cosine metric, tight `min_dist=0.0` for HDBSCAN input).",
        "3. Cluster with **HDBSCAN** — density-based, handles noise, no assumed cluster count.",
        "4. Extract per-cluster topic keywords via **c-TF-IDF** (sklearn CountVectorizer only).",
        "5. Report cluster quality metrics: silhouette score and HDBSCAN persistence.",
        "6. Visualise clusters on a 2-D UMAP scatter plot.",
        "7. Language-stratified reporting.",
        "",
        "Artefacts saved to `data/clustering/`:",
        "- `cluster_labels.csv`",
        "- `cluster_topics.csv`",
        "- `umap_2d.npy`",
        "- `clustering_config.json`",
    ),

    # ── Imports & project root ───────────────────────────────────────────────
    make_md("## 1  Setup"),
    make_code(
        "import sys",
        "from pathlib import Path",
        "",
        "# Ensure project root is on sys.path so scripts/ is importable",
        "PROJECT_ROOT = Path().resolve().parent if Path().resolve().name == 'notebooks' else Path().resolve()",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "import warnings",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import matplotlib.cm as cm",
        "",
        "warnings.filterwarnings('ignore', category=UserWarning)",
        "",
        "print(f'Project root: {PROJECT_ROOT}')",
    ),

    # ── Import pipeline functions ────────────────────────────────────────────
    make_code(
        "from scripts.concern_clustering import (",
        "    load_embeddings_and_metadata,",
        "    reduce_with_umap,",
        "    cluster_with_hdbscan,",
        "    extract_cluster_topics,",
        "    compute_quality_metrics,",
        "    save_clustering_artefacts,",
        "    run_clustering,",
        "    NOISE_LABEL,",
        "    UMAP_N_COMPONENTS_CLUSTER,",
        "    UMAP_N_NEIGHBORS,",
        "    UMAP_MIN_DIST_CLUSTER,",
        "    HDBSCAN_MIN_CLUSTER_SIZE,",
        "    HDBSCAN_MIN_SAMPLES,",
        ")",
        "",
        "print('All pipeline functions imported successfully')",
    ),

    # ── Paths ────────────────────────────────────────────────────────────────
    make_code(
        "EMBEDDINGS_DIR = PROJECT_ROOT / 'data' / 'embeddings'",
        "OUTPUT_DIR     = PROJECT_ROOT / 'data' / 'clustering'",
        "",
        "# Confirm embeddings are present",
        "for f in ['minilm_embeddings.npy', 'corpus_metadata.csv', 'corpus_hash.txt']:",
        "    path = EMBEDDINGS_DIR / f",
        "    status = '✓' if path.exists() else '✗  MISSING'",
        "    size   = f'{path.stat().st_size:,} bytes' if path.exists() else ''",
        "    print(f'  {status}  {f}  {size}')",
    ),

    # ── Load embeddings ──────────────────────────────────────────────────────
    make_md("## 2  Load Embeddings and Corpus Metadata"),
    make_code(
        "embeddings, meta_df = load_embeddings_and_metadata(EMBEDDINGS_DIR)",
        "",
        "print(f'Embeddings : {embeddings.shape}  dtype={embeddings.dtype}')",
        "print(f'Metadata   : {len(meta_df)} rows')",
        "print(f'Columns    : {list(meta_df.columns)}')",
        "print()",
        "print('Source dataset distribution:')",
        "print(meta_df['source_dataset'].value_counts().to_string())",
        "print()",
        "print('Language distribution:')",
        "print(meta_df['language'].value_counts().to_string())",
    ),

    # ── Alignment check ──────────────────────────────────────────────────────
    make_code(
        "# Strict alignment: corpus_id must equal the 0-based row index",
        "corpus_ids = meta_df['corpus_id'].astype(int).tolist()",
        "expected   = list(range(len(meta_df)))",
        "assert corpus_ids == expected, f'corpus_id alignment broken: {corpus_ids[:5]} vs {expected[:5]}'",
        "assert embeddings.shape[0] == len(meta_df), 'Row count mismatch between embeddings and metadata'",
        "print(f'Alignment check PASSED — {len(meta_df)} rows, {embeddings.shape[1]}-D embeddings')",
    ),

    # ── UMAP ─────────────────────────────────────────────────────────────────
    make_md(
        "## 3  UMAP Dimensionality Reduction",
        "",
        "We run UMAP twice:",
        "- **5-D** (tight `min_dist=0.0`, cosine metric) → HDBSCAN input",
        "- **2-D** (relaxed `min_dist=0.1`) → visualisation scatter plot",
        "",
        "> *Why two projections?* HDBSCAN benefits from tighter cluster geometry",
        "> (`min_dist=0.0`), while a readable scatter plot needs spread (`min_dist>0`).",
    ),
    make_code(
        "print('Running 5-D UMAP for clustering...')",
        "umap_5d, reducer_5d = reduce_with_umap(",
        "    embeddings,",
        "    n_components=UMAP_N_COMPONENTS_CLUSTER,",
        "    n_neighbors=UMAP_N_NEIGHBORS,",
        "    min_dist=UMAP_MIN_DIST_CLUSTER,",
        "    metric='cosine',",
        "    random_state=42,",
        ")",
        "print(f'5-D UMAP shape : {umap_5d.shape}')",
        "print(f'Finite values  : {np.isfinite(umap_5d).all()}')",
    ),

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    make_md(
        "## 4  HDBSCAN Clustering",
        "",
        "HDBSCAN discovers clusters of variable density without a fixed cluster count.",
        "Points that cannot be assigned are labelled `-1` (noise).",
        "",
        "**Key hyperparameters**:",
        "- `min_cluster_size` — minimum points to form a cluster (increase to merge small groups)",
        "- `min_samples` — controls how conservative the clustering is (higher = more noise)",
        "- `cluster_selection_method='eom'` — excess-of-mass; tends to prefer larger clusters",
    ),
    make_code(
        "print('Running HDBSCAN...')",
        "result = cluster_with_hdbscan(",
        "    umap_5d,",
        "    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,",
        "    min_samples=HDBSCAN_MIN_SAMPLES,",
        ")",
        "",
        "labels = result.labels",
        "n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)",
        "n_noise    = int((labels == NOISE_LABEL).sum())",
        "",
        "print(f'Clusters found : {n_clusters}')",
        "print(f'Noise points   : {n_noise} / {len(labels)} ({n_noise/len(labels)*100:.1f}%)')",
        "print()",
        "print('Cluster sizes:')",
        "for cid in sorted(set(labels)):",
        "    label_name = 'noise' if cid == NOISE_LABEL else f'cluster {cid}'",
        "    count = int((labels == cid).sum())",
        "    bar = '█' * (count // 5)",
        "    print(f'  {label_name:>10s}  {count:4d}  {bar}')",
    ),

    # ── Quality metrics ───────────────────────────────────────────────────────
    make_md(
        "## 5  Cluster Quality Metrics",
        "",
        "**Silhouette score** (range −1 to +1): measures how well each point fits its",
        "own cluster vs. the nearest other cluster.  Computed on 5-D UMAP vectors,",
        "excluding noise points.  Scores > 0.3 indicate reasonable structure.",
        "",
        "**Cluster persistence** (HDBSCAN internal): the range of density thresholds",
        "over which each cluster exists.  Higher persistence → more stable cluster.",
    ),
    make_code(
        "metrics = compute_quality_metrics(umap_5d, labels, result.clusterer)",
        "",
        "print(f'n_clusters        : {metrics[\"n_clusters\"]}')",
        "print(f'n_noise           : {metrics[\"n_noise\"]}')",
        "print(f'noise_fraction    : {metrics[\"noise_fraction\"]:.3f}')",
        "print(f'silhouette_score  : {metrics.get(\"silhouette_score\")}')",
        "",
        "if metrics.get('cluster_persistence'):",
        "    cp = metrics['cluster_persistence']",
        "    print(f'persistence mean  : {cp[\"mean\"]:.4f}')",
        "    print(f'persistence std   : {cp[\"std\"]:.4f}')",
        "    print(f'persistence range : [{cp[\"min\"]:.4f}, {cp[\"max\"]:.4f}]')",
    ),

    # ── c-TF-IDF topics ───────────────────────────────────────────────────────
    make_md(
        "## 6  Cluster Topic Keywords (c-TF-IDF)",
        "",
        "**c-TF-IDF** (Grootendorst 2022) treats each cluster as a 'document class' and",
        "extracts terms that are frequent within a cluster but rare across others.",
        "English *and* French stop-words are filtered out.",
        "",
        "> **Limitation**: the corpus is bilingual (English / French).  Keywords for the",
        "> French cluster will naturally be French terms, and vice versa.  This is",
        "> *expected and informative* — it reflects genuine language stratification rather",
        "> than a modelling error.  For fully language-agnostic topics, consider running",
        "> clustering separately per language.",
    ),
    make_code(
        "topics_df = extract_cluster_topics(meta_df, labels)",
        "",
        "print(topics_df[['cluster_id', 'size', 'top_keywords']].to_string(index=False))",
    ),

    # ── Language breakdown per cluster ────────────────────────────────────────
    make_md("### 6.1  Language Distribution per Cluster"),
    make_code(
        "for _, row in topics_df.iterrows():",
        "    cid = row['cluster_id']",
        "    lang_dist = json.loads(row['language_distribution'])",
        "    label = 'noise' if cid == NOISE_LABEL else f'cluster {cid}'",
        "    dist_str = '  '.join(f'{k}: {v:.0%}' for k, v in lang_dist.items())",
        "    print(f'  {label:>12s}  size={row[\"size\"]:4s}  {dist_str}')",
    ),

    # ── Source dataset breakdown per cluster ────────────────────────────────
    make_code(
        "print('\\nSource dataset distribution per cluster:')",
        "for _, row in topics_df.iterrows():",
        "    cid = row['cluster_id']",
        "    src_dist = json.loads(row['source_distribution'])",
        "    label = 'noise' if cid == NOISE_LABEL else f'cluster {cid}'",
        "    dist_str = '  '.join(f'{k}: {v}' for k, v in src_dist.items())",
        "    print(f'  {label:>12s}  {dist_str}')",
    ),

    # ── 2-D scatter visualisation ─────────────────────────────────────────────
    make_md(
        "## 7  2-D UMAP Scatter Plot",
        "",
        "The 2-D projection is built internally by `cluster_with_hdbscan` from the 5-D",
        "UMAP space (Euclidean metric, `min_dist=0.1` for readability).  Noise points are",
        "plotted in grey.",
    ),
    make_code(
        "umap_2d = result.umap_2d",
        "",
        "fig, ax = plt.subplots(figsize=(10, 7))",
        "",
        "# Colour map — one colour per real cluster",
        "real_cluster_ids = sorted(c for c in set(labels) if c != NOISE_LABEL)",
        "cmap = cm.get_cmap('tab10', max(len(real_cluster_ids), 1))",
        "",
        "# Noise points",
        "noise_mask = labels == NOISE_LABEL",
        "if noise_mask.any():",
        "    ax.scatter(",
        "        umap_2d[noise_mask, 0], umap_2d[noise_mask, 1],",
        "        c='lightgrey', s=10, alpha=0.5, label='noise', zorder=1,",
        "    )",
        "",
        "# Real clusters",
        "for i, cid in enumerate(real_cluster_ids):",
        "    mask = labels == cid",
        "    kw_row = topics_df[topics_df['cluster_id'] == cid].iloc[0]",
        "    kws = str(kw_row['top_keywords']).split(', ')[:3]",
        "    short_label = f'C{cid}: {\" | \".join(kws)}'",
        "    ax.scatter(",
        "        umap_2d[mask, 0], umap_2d[mask, 1],",
        "        c=[cmap(i)], s=18, alpha=0.75, label=short_label, zorder=2,",
        "    )",
        "",
        "ax.set_title('UMAP 2-D Projection — HDBSCAN Clusters', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('UMAP-1', fontsize=11)",
        "ax.set_ylabel('UMAP-2', fontsize=11)",
        "ax.legend(loc='best', fontsize=9, markerscale=1.5)",
        "ax.grid(True, alpha=0.3, linestyle='--')",
        "plt.tight_layout()",
        "plt.savefig(PROJECT_ROOT / 'reports' / 'umap_cluster_scatter.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
        "print('Scatter plot saved → reports/umap_cluster_scatter.png')",
    ),

    # ── Per-language topic analysis ────────────────────────────────────────────
    make_md(
        "## 8  Per-Language Topic Analysis",
        "",
        "Running topic extraction restricted to English-only and French-only",
        "documents within each cluster removes cross-language noise from the keywords.",
    ),
    make_code(
        "for lang in ['English', 'French']:",
        "    lang_mask = meta_df['language'] == lang",
        "    if lang_mask.sum() == 0:",
        "        print(f'No {lang} documents in corpus. Skipping.')",
        "        continue",
        "    lang_labels = labels[lang_mask.values]",
        "    lang_meta   = meta_df[lang_mask].reset_index(drop=True)",
        "    real_in_lang = len(set(lang_labels)) - (1 if NOISE_LABEL in lang_labels else 0)",
        "    if real_in_lang == 0:",
        "        print(f'{lang}: all documents are noise. Skipping topic extraction.')",
        "        continue",
        "    lang_topics = extract_cluster_topics(",
        "        lang_meta, lang_labels,",
        "        extra_stop_words=[],  # language-specific corpus — no cross-language stops",
        "    )",
        "    print(f'\\n{lang} topics:')",
        "    print(lang_topics[['cluster_id','size','top_keywords']].to_string(index=False))",
    ),

    # ── Save artefacts ────────────────────────────────────────────────────────
    make_md("## 9  Save Clustering Artefacts"),
    make_code(
        "config = {",
        "    'umap_n_components_cluster': UMAP_N_COMPONENTS_CLUSTER,",
        "    'umap_n_neighbors': UMAP_N_NEIGHBORS,",
        "    'umap_min_dist_cluster': UMAP_MIN_DIST_CLUSTER,",
        "    'hdbscan_min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,",
        "    'hdbscan_min_samples': HDBSCAN_MIN_SAMPLES,",
        "}",
        "",
        "save_clustering_artefacts(",
        "    meta_df=meta_df,",
        "    result=result,",
        "    topics_df=topics_df,",
        "    metrics=metrics,",
        "    output_dir=OUTPUT_DIR,",
        "    config=config,",
        ")",
        "",
        "for f in ['cluster_labels.csv', 'cluster_topics.csv', 'umap_2d.npy', 'clustering_config.json']:",
        "    path = OUTPUT_DIR / f",
        "    size = path.stat().st_size if path.exists() else 0",
        "    print(f'  ✓  {f}  ({size:,} bytes)')",
    ),

    # ── Full pipeline convenience ──────────────────────────────────────────────
    make_md(
        "## 10  Full Pipeline (convenience helper)",
        "",
        "The `run_clustering()` function wraps the entire pipeline with caching.",
        "Run this cell in subsequent sessions to reload or recompute results.",
    ),
    make_code(
        "# Cached run — reuses saved artefacts if hyperparameters unchanged",
        "labels_df_full, topics_df_full, metrics_full = run_clustering(",
        "    embeddings_dir=EMBEDDINGS_DIR,",
        "    output_dir=OUTPUT_DIR,",
        "    force_recompute=False,",
        ")",
        "",
        "print(f'Labels dataframe : {labels_df_full.shape}')",
        "print(f'Topics dataframe : {topics_df_full.shape}')",
        "print(f'Quality metrics  : {metrics_full}')",
        "labels_df_full.head(5)",
    ),

    # ── Limitations ───────────────────────────────────────────────────────────
    make_md(
        "## 11  Limitations and Caveats",
        "",
        "> [!NOTE]",
        "> The embedding corpus currently contains **450 sampled comments** (sample mode).",
        "> Full-corpus clustering (10,108 comments) will require increasing",
        "> `min_cluster_size` proportionally and may surface more fine-grained topics.",
        "",
        "> [!WARNING]",
        "> **Language confounding** — MiniLM separates English and French comments",
        "> into distinct regions of embedding space.  The two primary clusters found",
        "> correspond largely to language rather than pure thematic differences.  This is",
        "> semantically valid (topics *are* expressed differently across languages) but",
        "> should be interpreted with care.  Consider analysing each language separately",
        "> for theme-level insights.",
        "",
        "> [!NOTE]",
        "> **c-TF-IDF keywords are surface-level** — they reflect term frequency patterns,",
        "> not deep semantic reasoning.  Inspect individual cluster members alongside the",
        "> keywords to validate topic coherence.",
        "",
        "> [!CAUTION]",
        "> UMAP and HDBSCAN are stochastic.  Fix `random_state=42` for reproducibility.",
        "> Changing `n_neighbors`, `min_cluster_size`, or `min_samples` can materially",
        "> alter the number and composition of clusters.",
    ),
]


def build_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def main() -> None:
    out_path = Path(__file__).parent.parent / "notebooks" / "05_concern_clustering.ipynb"
    if out_path.exists():
        print(f"Notebook already exists: {out_path}")
        return
    nb = build_notebook(CELLS)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created: {out_path}")


if __name__ == "__main__":
    main()
