# PolicyLens: SentiMix Hindi-English Sentiment Baseline Report (v1)

**Execution Date:** 2026-10-13T12:51:10.918058+00:00  
**Model Architecture:** `HashingVectorizer` ($2^{18}$ features) + `SGDClassifier` (multiclass `log_loss`)  
**Training Corpus:** SemEval-2020 Task 9 SentiMix Hindi-English (15,131 official train records)  
**Evaluation Corpus:** Official held-out test split (3,000 records)  

> [!WARNING]
> **Domain Boundary & Scientific Integrity Notice:**  
> DOMAIN BOUNDARY DISCLAIMER: This model is an auxiliary social-media sentiment baseline trained on code-mixed Hindi-English Twitter posts (SemEval SentiMix 2020; negative, neutral, positive). It is NOT a policy consultation stance classifier. It must not be deployed as a stance evaluation model on Canadian or international citizen consultation submissions without genuine consultation-domain human labels.

---

## 1. Executive Summary

| Metric | Training Profile | Evaluation on Official Test Split (3,000 rows) |
| :--- | :--- | :--- |
| **Dataset Size** | **15,131** records (10 epochs, 151,310 row-updates) | **3,000** held-out records |
| **Elapsed Training Time** | **4.6s** (32,908.0 rows/s) | N/A |
| **Overall Accuracy** | N/A | **65.67%** |
| **Macro-F1** | N/A | **0.6614** |
| **Weighted-F1** | N/A | **0.6589** |

---

## 2. 3-Class Evaluation Breakdown

Evaluated on the 3,000 official held-out test examples (`data/processed/sentimix_test_canonical.csv`):

### Confusion Matrix
| Ground Truth \ Predicted | Predicted Negative | Predicted Neutral | Predicted Positive | Total Support |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Negative** | **561** | **302** | **37** | 900 |
| **Actual Neutral** | **188** | **697** | **215** | 1100 |
| **Actual Positive** | **33** | **255** | **712** | 1000 |
| **Total Predicted** | 782 | 1254 | 964 | 3000 |

### Per-Class Detailed Performance
* **Negative:** Precision = `0.7174`, Recall = `0.6233`, F1 = `0.6671` (Support: 900)
* **Neutral:** Precision = `0.5558`, Recall = `0.6336`, F1 = `0.5922` (Support: 1100)
* **Positive:** Precision = `0.7386`, Recall = `0.7120`, F1 = `0.7251` (Support: 1000)

---

## 3. Methodological Comparison with Upstream Competition Baseline

* Upstream Kaggle / SentiMix reference `SGDClassifier + TfidfVectorizer` achieved **65.53% accuracy**.
* Our memory-efficient `HashingVectorizer` ($2^{18}$ hashed n-grams) + `SGDClassifier` streaming baseline achieves **65.67% accuracy** and **0.6614 Macro-F1**, operating entirely in constant bounded memory (< 50 MB heap overhead).

---

## 4. Hardware Resource Usage & Operational Safety

* **Memory Safety:** Memory-bounded training executed across all 10 epochs without heap spikes. Starting available RAM was **449.9 MB**.
* **Disk Space:** Output artifacts occupy less than 3 MB total.
* **Throughput:** **32,908.0 rows/sec** across batched epochs.

---

## 5. Artifact Lineage & Cryptographic Checksums

All output artifacts are saved in versioned directory: `data/sentiment_training/sentimix_hinglish_baseline_v1/`:
* `model.joblib`: Serialized inference bundle (SHA-256: `ae70a574c303a72277b3de16239e4bd34137baab02cf7e1bbf7a41371ddf49a3`)
* `config.json`: Feature and optimizer hyperparameters (SHA-256: `69a5fecb22e0cd5961ae62236454b59d7c42cb51a45f0ff27f60da8495e58796`)
* `label_mapping.json`: 3-class label mapping schema (SHA-256: `8e80d34842a342805900ae9fba34321ee00ffb89ad1e7802a885a9cd40ee80dc`)
* `evaluation_metrics.json`: Full machine-readable evaluation report (SHA-256: `07333b0314aea236d7c34e94b6f180917ba5f110af2850c9c965f35ea8cbf410`)
* `manifest.json`: Verification manifest with complete input/output provenance
