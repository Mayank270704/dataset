# PolicyLens Project Status Audit

**Audit Timestamp:** 2026-10-12  
**Workspace:** `C:\Users\MAYANK SWAROOP\OneDrive\Desktop\dataset`  
**Purpose:** Factual baseline assessment of data holdings, label counts, model implementations, fitted artifacts, test suite health, blockers, and next steps prior to subsequent development.

---

## 1. Primary Consultation Datasets & Holdings

The project holds three primary Canadian Federal Open Government consultation datasets (2016–2018) in both raw and verified source-clean processed formats:

| Dataset Name | Role / File Path | Rows | Columns | Description & Structure |
|---|---|:---:|:---:|---|
| **Individual Comments (Raw)** | `ap2016-18_individual_comments.csv` | 160 | 19 | Workshop index card comments; contains 14 trailing empty records (13 exact duplicate blank rows). |
| **Individual Comments (Clean)** | `data/processed/individual_comments_2016_18_source_clean.csv` | 160 | 24 | Whitespace normalized, empty markers standardized to null, trailing blanks flagged via `exclusion_reason`. |
| **Consultation Compilation (Raw)** | `ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv` | 5,169 | 63 | Multi-channel engagement (Twitter `#OpenGovCan`, Reddit, workshops, email submissions). |
| **Consultation Compilation (Clean)** | `data/processed/consultation_compilation_2017_18_source_clean.csv` | 5,169 | 68 | Cleaned source comments, standardized text fields, preserved bilingual metadata. |
| **Get Involved Questionnaire (Raw)** | `Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv` | 8,213 | 98 | 2 header rows; nationwide online consultation questionnaire. |
| **Get Involved Questionnaire (Clean)** | `data/processed/get_involved_questionnaire_2017_18_source_clean.csv` | 8,212 | 103 | Dual headers preserved, column names uniquely slugified, standardized nulls. |

- **Total Extracted Substantive Text Responses**: **13,541 responses** (compiled in `data/interim/government_response_text_entries.csv`).
- **Active Unsupervised Representation Sample**: **450 length-filtered, deduplicated comments** (compiled in `data/embeddings/corpus_metadata.csv`).

---

## 2. Status of Human-Verified Labels

An inspection was conducted across all annotation files, processed datasets, and model outputs:

| File / Dataset | Row Count | Genuine Human Stance Labels | Genuine Human Concern/Demand Labels | Status |
|---|:---:|:---:|:---:|---|
| `data/annotation/pilot_annotation_50.csv` | 50 | **0** | **0** | Blank reference template |
| `data/annotation/annotator_1.csv` | 50 | **0** | **0** | Blank blind template for Annotator 1 |
| `data/annotation/annotator_2.csv` | 50 | **0** | **0** | Blank blind template for Annotator 2 |
| `data/annotation/expansion_pool_400.csv` | 400 | **0** | **0** | Blank candidate expansion pool |
| `data/annotation/expansion_annotator_1.csv` | 400 | **0** | **0** | Blank template for Annotator 1 |
| `data/annotation/expansion_annotator_2.csv` | 400 | **0** | **0** | Blank template for Annotator 2 |
| `data/processed/individual_comments_...` | 160 | **0** | **0** | Raw consultation text; no stance annotations |
| `data/processed/consultation_compilation_...` | 5,169 | **0** | **0*** | *Has administrative organizer tag `theme_th_me`, not citizen stance or demand annotations |
| `data/processed/get_involved_questionnaire_...` | 8,212 | **0** | **0** | Survey answers; no stance or demand annotations |
| `data/clustering/cluster_labels.csv` | 450 | **0** | **0** | Unsupervised HDBSCAN cluster IDs (-1, 0, 1), not human labels |

> **Factual Finding:** There are **zero (0) genuine human-verified stance, concern, or demand labels** in the workspace. All annotation CSVs are currently empty templates awaiting human adjudication.

---

## 3. Implementation, Artifact, and Training Audit of Models

| Model / Pipeline Component | Working Implementation Code? | Generated Artifacts in Workspace? | Actually Trained / Fitted? | Notes & Exact Artifact Paths |
|---|:---:|:---:|:---:|---|
| **MiniLM Sentence Embeddings** | **Yes**<br>([`models/semantic_similarity/minilm.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/semantic_similarity/minilm.py)) | **Yes** | **Fitted (Zero-shot Representation Caching)** | `data/embeddings/minilm_embeddings.npy` (450 × 384 matrix), `corpus_metadata.csv`, `corpus_hash.txt`. Model weights reside in Hugging Face local cache (`~/.cache/huggingface/hub/`). Not fine-tuned. |
| **UMAP Dimensionality Reduction** | **Yes**<br>([`models/concern_clustering/clustering.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/concern_clustering/clustering.py)) | **Yes** | **Yes (Fitted on 450 Embeddings)** | Fitted 5-D reduction for HDBSCAN input and 2-D reduction saved at `data/clustering/umap_2d.npy`. |
| **HDBSCAN Density Clustering** | **Yes**<br>([`models/concern_clustering/clustering.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/concern_clustering/clustering.py)) | **Yes** | **Yes (Fitted on 450 5-D UMAP Vectors)** | Partitioned into 2 clusters + 1 noise group. Outputs: `data/clustering/cluster_labels.csv`, `data/clustering/clustering_config.json` (silhouette score = 0.4468, noise ratio = 0.1911). |
| **c-TF-IDF Topic Keywords** | **Yes**<br>([`models/concern_clustering/clustering.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/concern_clustering/clustering.py)) | **Yes** | **Yes (Fitted on Cluster Documents)** | Fitted on grouped text pseudo-documents. Artifact: `data/clustering/cluster_topics.csv`. |
| **TF-IDF + Logistic Regression** | **Yes**<br>([`models/stance_classification/baselines.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/stance_classification/baselines.py)) | **No** | **No** | Pipeline code is complete with anti-leakage splitting, but has **not been fitted on consultation data** due to 0 human labels. Unit tested on synthetic data only. |
| **TF-IDF + Linear SVM** | **Yes**<br>([`models/stance_classification/baselines.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/stance_classification/baselines.py)) | **No** | **No** | Pipeline code is complete, but has **not been fitted on consultation data**. Unit tested on synthetic data only. |
| **Transformer Stance Model** | **No** (Interface only) | **No** | **No** | Planned architecture. |
| **MuRIL Multilingual Model** | **No** (Config only) | **No** | **No** | Planned architecture. |

---

## 4. Feasible Experiments Ready to Run Immediately (Without Inventing Labels)

The following pipelines rely strictly on unsupervised methods or external benchmark data and can execute immediately without synthetic or hallucinated labels:

1. **Full-Corpus Unsupervised Semantic Discovery**:
   - Scale sentence embedding computation and HDBSCAN topic clustering from the current 450-record sample to the **entire 13,541 substantive consultation responses** in `data/interim/government_response_text_entries.csv`.
2. **Interactive Evidence-Linked Policy Retrieval**:
   - Query consultation embeddings with user-defined policy terms (e.g., *"open science data sharing"*, *"indigenous duty to consult"*, *"FOI portal delays"*) using `models.semantic_similarity.retrieve_similar()` to rank nearest citizen submissions by cosine similarity.
3. **Cross-Lingual Representation Alignment Evaluation**:
   - Evaluate whether multilingual sentence encoders (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) bridge the English-French separation observed in `all-MiniLM-L6-v2`, evaluating cross-lingual cosine similarity between translated question prompts.
4. **Thematic Distribution Analysis**:
   - Cross-tabulate discovered HDBSCAN clusters against existing administrative metadata (`theme_th_me`, workshop events, and `q19` sector groups) to quantify stakeholder participation patterns across topics.
5. **Auxiliary Sentiment Baseline Experiments**:
   - Train and benchmark polarity classifiers exclusively on the external datasets (`PolicyLens_Hindi_English_Hinglish_Sentiment.csv`, `SentiMix`, `Sentiment140`), strictly documenting them as social sentiment baselines, not policy consultation stance.

---

## 5. Identified Blockers

### A. Blockers for Supervised Stance Classification
1. **Primary Blocker — Zero Human Labels**: Supervised stance classifiers cannot be trained because `pilot_annotation_50.csv` and annotator templates contain 0 non-blank values.
2. **Inter-Annotator Agreement Undetermined**: Before a consensus gold set can exist, Annotator 1 and Annotator 2 must complete blind templates independently to measure Cohen's Kappa ($\kappa$) and resolve disagreements.
3. **Guardrail Enforced**: `models.stance_classification.validate_labeled_data()` strictly halts execution if verified labels $< 10$ to prevent accidental training on unannotated data.

### B. Blockers for Multilingual Evaluation
1. **Lack of Multilingual Consultation Stance Benchmark**: French consultation comments exist (approx. 17% of responses), but none have human stance annotations. Hindi and Hinglish text exists only in auxiliary YouTube (`HinEmo-9`) and Twitter (`SentiMix`) corpora, not in the primary consultation domain.
2. **Monolingual Vector Bias in MiniLM**: `all-MiniLM-L6-v2` clusters English and French into distinct spatial clusters, preventing language-agnostic topic grouping until a multilingual model is evaluated.

---

## 6. Verified Workspace Test Results

A full test suite run was executed and verified directly on the workspace:

- **Total Tests Passed:** **116 / 116 tests** (0 failures, 0 errors, 39 warnings in 52.07s)
- **Breakdown by Module:**
  - `tests/test_stance_classification.py`: **7 passed** (anti-leakage splitting, vectorizer fit-on-train-only, label validation rejection guardrails, expansion pool integrity)
  - `tests/test_models_structure.py`: **5 passed** (package exports, common paths, backward-compatible wrappers)
  - `tests/test_concern_clustering.py`: **44 passed** (UMAP reduction, HDBSCAN clustering, c-TF-IDF keyword extraction, metrics)
  - `tests/test_semantic_similarity.py`: **11 passed** (MiniLM mean pooling, L2 normalization, hash caching, cosine retrieval)
  - `tests/test_annotation_pilot.py`: **18 passed** (pilot sampling rules, schema compliance, blank label verification)
  - `tests/test_annotation_agreement.py`: **10 passed** (double-blind template alignment, Kappa calculation logic)
  - `tests/test_comprehensive_data_cleaning_audit.py`: **5 passed** (data cleaning audit logic and provenance)
  - `tests/test_policylens_dataset_audit.py`: **5 passed** (canonicalization, text cleaning, duplicate keys)
  - `tests/test_government_consultation_audit.py`: **5 passed** (cell cleaning, column slugification, PII redaction)
  - `tests/test_duplicate_overlap_review.py`: **6 passed** (exact and near duplicate detection)

---

## 7. Recommended Next Action

**Single Recommended Next Action:**  
> **Scale the unsupervised semantic embedding and concern clustering pipeline from the current 450-comment pilot to the full 13,541 substantive consultation responses (`data/interim/government_response_text_entries.csv`).**

### Rationale:
- Supervised stance training is blocked until human annotators manually label the prepared templates (`data/annotation/expansion_annotator_1.csv` and `data/annotation/expansion_annotator_2.csv`). That task requires human subject-matter experts and cannot be automated without violating data integrity rules.
- In contrast, scaling the unsupervised pipeline to all 13,541 comments requires zero invented labels, utilizes verified and passing code, unlocks macro-level topic discovery across all three federal datasets, and directly produces full-coverage citizen concern insights.
