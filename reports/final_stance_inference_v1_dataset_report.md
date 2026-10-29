# PolicyLens – Final Stance Inference v1: Dataset Usage and Label Mapping Report

**Document version:** 1.0  
**Created:** 2026-10-29  
**Pipeline:** `FinalStancePredictor` / `models/stance_inference_v2.py`  
**Script:** `scripts/run_final_stance_inference_v1.py`

---

## Executive Summary

The final stance inference pipeline (v1) produces four-label stance predictions —
**Support / Oppose / Neutral / Unclear** — for new consultation comments by combining
two pre-trained models and a deterministic heuristic rule.

> [!IMPORTANT]
> **Oppose is a heuristic, not a validated classifier output.**  
> The stance model was trained exclusively on **Support / Neutral / Unclear** labels
> (AI-generated working labels from the Canadian consultation pool).
> The **Oppose** label is assigned by a post-hoc rule: if the Sentiment140 model
> predicts *negative* sentiment **and** the stance model predicts anything other than
> *Support*, the text is labelled *Oppose*.  
> This rule has not been validated against human-annotated Oppose examples.
> All outputs should be treated as provisional machine suggestions.

### Pipeline components

| Component | Model | Artifact path |
|---|---|---|
| Stance classifier | TF-IDF + Linear SVM / Logistic Regression | `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/` |
| Sentiment signal | HashingVectorizer + SGD (Sentiment140) | `data/sentiment_training/sentiment140_hashing_sgd_v1/` |

---

## Dataset Inventory and Role Table

| Dataset | Language | Approx. size | Licence | Role in pipeline | Feeds |
|---|---|---|---|---|---|
| **Sentiment140** (Go et al., 2009) | English (informal Twitter) | 1,600,000 training + 498 test rows | CC-BY 4.0 | Training the sentiment tone classifier that drives the Oppose heuristic | Sentiment model |
| **Canadian consultation corpus** (`data/full_corpus/run_v1/corpus/corpus_entries.csv`) | English / French | 10,108 unique corpus entries; 400-record pool used for AI label generation | Open Government Licence – Canada | Source of texts used to generate AI working labels for stance training | Stance model |
| **SentiMix Hindi–English** (SemEval-2020 Task 9) | Hindi–English code-mixed | 15,131 training + 3,000 test | SemEval 2020 data licence (see dataset README) | Separate Hinglish sentiment baseline (`sentimix_hinglish_baseline_v1`). **NOT used in the final stance pipeline**. | (separate SentiMix model only) |
| **Indian political tweets** (`indian/processed/kaggle_indian_politics_tweets_clean.csv`) | Hindi / English | Variable (Kaggle dataset) | Kaggle dataset terms | Reference / retrieval only. **NOT used for training** any component of this pipeline. | — |
| **MyGov comments** (`indian/processed/mygov_comments_clean.csv`) | Hindi / English | Variable (public web scrape) | Public web scrape | Reference / retrieval only. **NOT used for training**. | — |
| **PolicyLens Hindi–English–Hinglish Sentiment** (`PolicyLens_Hindi_English_Hinglish_Sentiment.csv`) | Hindi / English / Hinglish | Variable | Internal (PolicyLens project) | Auxiliary dataset; **NOT merged into stance training** or Sentiment140 training. Kept separate for potential future multilingual work. | — |

> [!NOTE]
> The Indian political tweets dataset, the MyGov comments dataset, and the
> PolicyLens multilingual sentiment CSV are **reference-only** assets.  
> They are not blended into the Canadian consultation stance training, and they
> are not used to train or fine-tune the Sentiment140 sentiment model.  
> Their presence in the workspace does not affect any prediction made by this pipeline.

---

## Label Mapping Table

The following table traces how each raw training label flows through the pipeline to
the final displayed label.

| Original training label | Stance model output | Sentiment condition | Final displayed label | `prediction_method` | `stance_label_source` |
|---|---|---|---|---|---|
| `Support` (AI working label) | `Support` | positive (any) | **Support** | `stance_model_direct` | `ai_working_label_unverified` |
| `Support` (AI working label) | `Support` | negative | **Support** | `stance_model_direct` | `ai_working_label_unverified` |
| `Neutral` (AI working label) | `Neutral` | positive | **Neutral** | `stance_model_direct` | `ai_working_label_unverified` |
| `Neutral` (AI working label) | `Neutral` | **negative** | **Oppose** ⚠️ | `heuristic_oppose` | `heuristic_oppose_unverified` |
| `Unclear` (AI working label) | `Unclear` | positive | **Unclear** | `stance_model_direct` | `ai_working_label_unverified` |
| `Unclear` (AI working label) | `Unclear` | **negative** | **Oppose** ⚠️ | `heuristic_oppose` | `heuristic_oppose_unverified` |
| *(too short / empty)* | — | — | `None` | `None` | `None` |

**Key:**  
- ⚠️ = heuristic label; not produced by a validated classifier.  
- Rows marked `heuristic_oppose` are flagged in every output CSV via the `prediction_method` column.

### Heuristic rule (verbatim)

```
if sentiment_prediction == "negative" AND stance_raw != "Support":
    final_stance = "Oppose"
    prediction_method = "heuristic_oppose"
    stance_label_source = "heuristic_oppose_unverified"
else:
    final_stance = stance_raw      # "Support", "Neutral", or "Unclear"
    prediction_method = "stance_model_direct"
    stance_label_source = "ai_working_label_unverified"
```

---

## Datasets NOT Merged Into Stance Training

The following datasets are explicitly **excluded** from the Canadian consultation
stance model training. They serve different, well-delimited roles:

| Dataset | Why it is NOT in stance training |
|---|---|
| **Indian political tweets** | Different domain (Indian politics, Twitter), different annotation framework, different language register. Kept for retrieval and reference only. |
| **MyGov comments** | Public web scrape without stance annotation. Not merged to avoid domain contamination of the consultation stance labels. |
| **SentiMix Hindi–English** | Used only for the separate `sentimix_hinglish_baseline_v1` sentiment model. Code-mixed tweets ≠ consultation stance; merging would contaminate both tasks. |
| **PolicyLens multilingual sentiment CSV** | Auxiliary multilingual resource; no stance labels; kept separate to preserve dataset identity and avoid mixing social-media sentiment signals into consultation stance training. |
| **Sentiment140** | Used for the sentiment signal only. Twitter sentiment ≠ consultation stance. Must not be used as a stance training source. |

---

## Caveats and Limitations

> [!CAUTION]
> **Do not use pipeline outputs as authoritative stance classifications.**

1. **Oppose is heuristic, not validated.** No human-annotated Oppose examples were
   used to train or evaluate the Oppose prediction path. The heuristic accuracy is
   unknown.

2. **Sentiment domain mismatch.** The Sentiment140 model was trained on informal
   English tweets. Canadian consultation comments are formal, policy-oriented text.
   The classifier may systematically misclassify policy hedging, diplomatic language,
   or French-language comments.

3. **Stance model trained on AI-generated working labels.** The labels assigned
   during stance training were generated by an AI assistant and are explicitly marked
   `ai_generated_unverified`. Macro-F1 against those AI labels (held-out) was modest
   (reported in `run_pool400_ai_v1_3class/metrics.json`); performance on genuine
   human-annotated stances is unknown.

4. **No Oppose training examples.** The stance model outputs only Support / Neutral /
   Unclear. The model architecture has no concept of Oppose; assigning Oppose by
   external rule means that texts genuinely expressing opposition but using positive
   or neutral tone will be mislabelled (false negatives for Oppose).

5. **French and non-English texts.** The Sentiment140 model is English-only.
   French consultation comments and Hindi/Hinglish texts will receive sentiment
   predictions that are linguistically unreliable; downstream Oppose assignments
   for non-English texts carry additional uncertainty.

6. **Short-text rejection.** Texts shorter than 3 characters after stripping
   whitespace are rejected (`input_status = "too_short_or_empty"`). This includes
   empty strings, whitespace-only strings, and two-character inputs.

7. **No concern group or priority assignment.** This pipeline does not assign
   new comments to existing concern clusters (nearest-centroid assignment failed
   pre-set criteria; no UMAP/HDBSCAN model was saved). Priority rankings describe
   existing groups only and are not applicable to single new comments.

---

## Artifact Traceability

All outputs written by `run_final_stance_inference_v1.py` include:

- `results.csv` (or `smoke_test_results.csv`) — one row per input text
- `summary.json` — label distribution and metadata
- `manifest.json` — SHA-256 checksums of all output files, script provenance, model versions

The stance model artifacts are SHA-256 verified by `load_verified_pipeline()` before
loading (see `models/new_comment_inference.py`). The Sentiment140 model artifact
(`model.joblib`) is SHA-256 verified against the value recorded in
`data/sentiment_training/sentiment140_hashing_sgd_v1/manifest.json` before loading
(see `models/stance_inference_v2._load_verified_sentiment_model()`).

---

## Reproduction Commands

```bash
# Smoke test (synthetic cases, writes to data/inference/final_stance_inference_v1/)
cd "c:\Users\MAYANK SWAROOP\OneDrive\Desktop\dataset"
.venv\Scripts\python.exe scripts/run_final_stance_inference_v1.py --smoke-test

# Batch inference on a real CSV
.venv\Scripts\python.exe scripts/run_final_stance_inference_v1.py \
    --input my_comments.csv \
    --text-column text \
    --out results/final_stance_output.csv \
    --model linsvm

# Unit tests
.venv\Scripts\python.exe -m pytest tests/test_final_stance_inference.py -v

# Full relevant test suite
.venv\Scripts\python.exe -m pytest \
    tests/test_final_stance_inference.py \
    tests/test_stance_classification.py \
    tests/test_sentiment140_baseline.py \
    tests/test_models_structure.py -v
```
