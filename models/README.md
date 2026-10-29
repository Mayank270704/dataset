# PolicyLens Models Architecture

This directory houses all implemented, modular, and planned machine learning models, representation pipelines, and topic discovery systems for **PolicyLens**.

---

## 1. Directory Structure

```text
models/
├── README.md                      # This guide
├── MODEL_REGISTRY.md              # Living registry of model statuses, artifacts, and next steps
├── __init__.py                    # Top-level models package export (lazy-loaded via PEP 562)
├── new_comment_inference.py       # NewCommentPredictor — 3-label consultation stance + sentiment
├── stance_inference_v2.py         # FinalStancePredictor — 4-label stance pipeline (heuristic Oppose)
├── common/                        # Shared path configuration and model constants
│   ├── __init__.py
│   └── config.py                  # Standard paths (DATA_DIR, EMBEDDINGS_DIR, etc.)
├── semantic_similarity/           # MiniLM sentence embeddings & evidence retrieval
│   ├── __init__.py
│   └── minilm.py                  # Core implementation of mean-pooling & cosine search
├── concern_clustering/            # UMAP + HDBSCAN clustering & c-TF-IDF keyword extraction
│   ├── __init__.py
│   └── clustering.py              # Core clustering, c-TF-IDF, and metrics implementation
├── concern_discovery/             # Full concern discovery, deduplication, diagnostics, review
│   └── ...
├── sentiment_analysis/            # Streaming SGD hashing baseline for Sentiment140 & SentiMix
│   └── ...
├── stance_classification/         # Supervised stance classification (linsvm / logreg)
│   └── __init__.py                # Input/output schemas and baseline specifications
└── multilingual/                  # Multilingual representation & cross-lingual models (planned)
    └── __init__.py                # MuRIL and multilingual MiniLM configurations
```

---

## 2. Model Implementation Locations

| Model / Pipeline | Implementation Path | Main Functions |
|---|---|---|
| **Sentence Embeddings & Retrieval** | [`models/semantic_similarity/minilm.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/semantic_similarity/minilm.py) | `mean_pool_and_normalize()`, `build_corpus()`, `embed_and_save_corpus()`, `retrieve_similar()` |
| **Concern Clustering & Topics** | [`models/concern_clustering/clustering.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/concern_clustering/clustering.py) | `reduce_with_umap()`, `cluster_with_hdbscan()`, `extract_cluster_topics()`, `run_clustering()` |
| **Path & Config Resolution** | [`models/common/config.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/common/config.py) | `PROJECT_ROOT`, `DATA_DIR`, `EMBEDDINGS_DIR`, `CLUSTERING_DIR` |
| **Final Stance Inference (v2 Heuristic)** | [`models/stance_inference_v2.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/stance_inference_v2.py) | `FinalStancePredictor`, `predict()`, heuristic Oppose assignment |
| **New Comment Inference (v1 Baseline)** | [`models/new_comment_inference.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/new_comment_inference.py) | `NewCommentPredictor`, `predict()` |
| **Stance Classification (Trained Baseline)** | [`models/stance_classification/train_pool400_ai_labels.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/stance_classification/train_pool400_ai_labels.py) | Linear SVM & Logistic Regression on 400-record AI-labeled pool |
| **Sentiment Analysis (Streaming)** | [`models/sentiment_analysis/streaming_baseline.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/sentiment_analysis/streaming_baseline.py) | `Sentiment140StreamingClassifier`, HashingVectorizer + SGD |
| **Multilingual Models (Planned)** | [`models/multilingual/__init__.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/multilingual/__init__.py) | `MURIL_MODEL_ID`, `MULTILINGUAL_MINILM_MODEL_ID` |

---

## 3. How to Run the Existing Pipelines

### A. Python API
You can import and run the pipelines directly from Python:

```python
from models.semantic_similarity import build_corpus, embed_and_save_corpus, retrieve_similar
from models.concern_clustering import run_clustering
from models.common import PROCESSED_DATA_DIR, EMBEDDINGS_DIR, CLUSTERING_DIR
from transformers import AutoTokenizer, AutoModel

# 1. Generate / retrieve MiniLM embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
corpus_df = build_corpus(PROCESSED_DATA_DIR, sample_per_dataset=150)
embeddings, meta_df = embed_and_save_corpus(corpus_df, tokenizer, model, EMBEDDINGS_DIR)

# 2. Run HDBSCAN clustering and c-TF-IDF keyword discovery
labels_df, topics_df, metrics = run_clustering(
    embeddings_dir=EMBEDDINGS_DIR,
    output_dir=CLUSTERING_DIR,
)
print("Clusters found:", metrics["n_clusters"])
print("Silhouette score:", metrics["silhouette_score"])
```

### B. Command-Line Entrypoints
Backward-compatible CLI runners are available in `scripts/`:

```powershell
# Run semantic similarity pipeline
python scripts/semantic_similarity.py

# Run concern clustering pipeline
python scripts/concern_clustering.py
```

### C. Interactive Notebooks
- [`notebooks/04_semantic_similarity.ipynb`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/notebooks/04_semantic_similarity.ipynb): Interactive embedding exploration, cosine retrieval benchmarks, and self-exclusion verification.
- [`notebooks/05_concern_clustering.ipynb`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/notebooks/05_concern_clustering.ipynb): UMAP 2-D visualization, cluster distribution plots, c-TF-IDF keyword inspection, and per-language stratified analysis.

---

## 4. Where Outputs and Metrics Are Saved

All generated model artifacts are cleanly persisted in `data/`:

| Output File | Format | Description |
|---|---|---|
| `data/embeddings/minilm_embeddings.npy` | NumPy Binary | 450 × 384 L2-normalized float32 sentence embedding matrix |
| `data/embeddings/corpus_metadata.csv` | CSV | 450 rows aligned 1-to-1 with embedding matrix rows |
| `data/embeddings/corpus_hash.txt` | Text | SHA-256 fingerprint for deterministic cache invalidation |
| `data/clustering/cluster_labels.csv` | CSV | 450 records with `cluster_id`, `cluster_probability`, `umap_x`, `umap_y` |
| `data/clustering/cluster_topics.csv` | CSV | Top keywords and size per discovered cluster |
| `data/clustering/umap_2d.npy` | NumPy Binary | 450 × 2 2-D coordinates for interactive scatter plots |
| `data/clustering/clustering_config.json` | JSON | All pipeline hyperparameters and quality metrics (`silhouette_score`, `noise_ratio`, `persistence`) |

---

## 5. Status of Pending Models

The following models are documented in [`models/MODEL_REGISTRY.md`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/MODEL_REGISTRY.md) and currently pending:

1. **TF-IDF + Logistic Regression & Linear SVM (Stance Classification)**:
   - *Dependency*: Scaling up the human-annotated stance dataset beyond the initial 50-row pilot (`data/annotation/pilot_annotation_50.csv`) to >= 300 instances.
2. **Fine-tuned Transformer Stance Model**:
   - *Architecture*: DeBERTa-v3 or RoBERTa with cross-encoder stance classification head.
   - *Dependency*: Expanded double-annotated consensus dataset with inter-annotator agreement metrics.
3. **MuRIL Cross-Lingual Representation**:
   - *Architecture*: `google/muril-base-cased` for cross-lingual zero-shot transfer across Hindi, Hinglish, and English auxiliary corpora.
4. **Multilingual MiniLM**:
   - *Architecture*: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to unify English and French consultation topics into a shared embedding space.

---

## 6. Backward Compatibility & Design Integrity

- **Scripts Compatibility**: [`scripts/semantic_similarity.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/scripts/semantic_similarity.py) and [`scripts/concern_clustering.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/scripts/concern_clustering.py) have been preserved as thin, 100% backward-compatible wrappers that re-export all symbols from `models/`.
- **No Weight Bloat**: Large pretrained model weights are stored in the local Hugging Face cache (`~/.cache/huggingface/hub/`) and never committed into the repository.
- **Existing Directories Preserved**: `data/`, `notebooks/`, `tests/`, and `reports/` maintain their exact locations and roles without data loss or unintended side-effects.
