# PolicyLens: Sentiment140 Streaming Baseline Report (v1)

**Execution Date:** 2026-10-13 12:26:17 UTC  
**Model Type:** Social-Media Sentiment Baseline (`HashingVectorizer` + `SGDClassifier`)  
**Status:** Completed on full 1,600,000 training records.

> [!WARNING]
> **Domain Boundary & Scientific Integrity Notice:**  
> DOMAIN BOUNDARY DISCLAIMER: This model is a social-media sentiment baseline trained on informal English Twitter posts (Sentiment140; 0=negative, 4=positive). It is NOT a policy consultation stance classifier. It must not be deployed as a stance evaluation model on Canadian or international citizen consultation submissions without genuine consultation-domain human labels.

---

## 1. Executive Summary

| Metric | Training Run | Evaluation on Held-Out Test Set (359 binary rows) |
| :--- | :--- | :--- |
| **Completed Rows** | **1,600,000** (100% full dataset) | **359** (177 Negative / 182 Positive) |
| **Elapsed Training Time** | **34.40 seconds** (46,505 rows/s) | N/A |
| **Macro-F1** | N/A | **0.7911** |
| **Accuracy** | N/A | **79.39%** |
| **Negative F1** | N/A | **0.7673** (P: 0.8652, R: 0.6893) |
| **Positive F1** | N/A | **0.8150** (P: 0.7477, R: 0.8956) |

---

## 2. Binary Evaluation (Negative vs. Positive)

Evaluated strictly on the 359 negative and positive manual test examples (`testdata.manual.2009.06.14.csv`):

* **Binary Test Set Size:** 359 records (177 Negative, 182 Positive)
* **Macro-F1:** `0.7911`
* **Weighted-F1:** `0.7915`
* **Accuracy:** `79.39%`

### Confusion Matrix
| Actual \ Predicted | Predicted Negative (0) | Predicted Positive (4) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (0)** | **122** (True Neg) | **55** (False Pos) | 177 |
| **Actual Positive (4)** | **19** (False Neg) | **163** (True Pos) | 182 |
| **Total Predicted** | 141 | 218 | 359 |

### Per-Class Detailed Performance
* **Negative (Class 0):** Precision = `0.8652`, Recall = `0.6893`, F1 = `0.7673` (Support: 177)
* **Positive (Class 4):** Precision = `0.7477`, Recall = `0.8956`, F1 = `0.8150` (Support: 182)

---

## 3. Neutral Class Handling & Distribution

The manual test set includes 139 neutral tweets (`polarity == 2`). In accordance with research integrity requirements:
* Neutral was **NOT** mapped to either positive or negative during training.
* Neutral was **NOT** included in the binary scoring above.
* **Neutral Prediction Behavior:**
  * Total Neutral Tweets: 139
  * Predicted as Negative: 17 (12.23%)
  * Predicted as Positive: 122 (87.77%)
  * Mean Positive Probability on Neutral: 0.6927

---

## 4. Hardware Resource Usage & Memory Safety

* **Strategy:** Streaming incremental partial fit with `HashingVectorizer(n_features=262,144)` and batch size 25,000.
* **Memory Profile:** Constant heap usage throughout all batches.
  * Starting Available RAM: 0.39 GB
  * Minimum Observed Available RAM: 0.39 GB
  * Peak memory footprint remained safely within host bounds with zero memory errors.
* **Throughput:** 46,505 rows per second.

---

## 5. Artifact Lineage & Verification

All output artifacts saved in versioned directory: `C:\Users\MAYANK SWAROOP\OneDrive\Desktop\dataset\data\sentiment_training\sentiment140_hashing_sgd_v1`:
* `model.joblib`: Serialized inference bundle (SHA-256: `c6737ebf77f04355...`)
* `config.json`: Feature and optimizer hyperparameters
* `label_mapping.json`: Strict class name mapping
* `evaluation_metrics.json`: Full machine-readable evaluation report
* `manifest.json`: Verification manifest with input/output hashes
