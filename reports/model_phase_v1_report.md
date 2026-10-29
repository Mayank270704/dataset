# PolicyLens model training and evaluation phase v1

**Date:** 2026-10-14
**Corpus:** Canadian federal Open Government consultations, 2016–18 (English and French).
**Label status:** the project owner authorised AI-generated labels as the working supervision source while human annotations are unavailable. Every stance and sentiment label used for training here is AI-generated and **not human-verified**. Every precision, recall and F1 figure measures agreement with those AI labels on held-out records. None of them is human-verified accuracy.

The frontend was not changed. Official annotation files, reviewer packets and protected artifacts were not modified. The protected-artifact check reports that all 49 recorded protected files match their baseline.

---

## 1. Experiments completed and blocked

| Component | Status | Output |
|---|---|---|
| AI stance labels, 350 expansion records | Completed | `data/ai_labels/stance_sentiment_ai_v1/` |
| AI sentiment labels, 400 records | Completed | same directory, separate file |
| Stance: TF-IDF + LR and TF-IDF + Linear SVM, 3 classes | Completed | `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/` |
| Stance: Support vs Neutral (Unclear excluded) | Completed | `data/stance_training_ai_working_labels/run_pool400_ai_v1_support_vs_neutral/` |
| Stance: Oppose and Mixed classes | **Blocked** | Oppose has 1 AI label and Mixed has 0. The minimum is 5 per class, and that threshold was not lowered. |
| Sentiment140 and SentiMix inference on consultation text | Completed | `data/sentiment_evaluation/consultation_domain_check_v1/` |
| Consultation-domain sentiment baseline on AI labels | Completed | `data/sentiment_training/consultation_ai_labels_tfidf_v1/` |
| Concern-group audit and configuration sensitivity | Completed | `data/full_corpus/concern_audit_v1/` |
| Alternative clustering: question-field centering | Completed, **not adopted** | `data/full_corpus/experiments_field_centered_v1/` |
| Prioritization stability, contributions, evidence | Completed | `data/full_corpus/prioritization_audit_v1/` |
| New-comment stance and sentiment inference | Completed | `models/new_comment_inference.py`, `data/inference/new_comment_inference_v1/` |
| New-comment concern-group assignment | **Blocked** | Nearest-centroid assignment failed its pre-set criteria, and no UMAP/HDBSCAN model was saved. |
| Human-label stance training, agreement, cluster review | **Blocked**, unchanged | No completed human annotation or review files exist. |

## 2. Dataset and label counts

**AI working labels** cover all 400 records of the v2 pool.

- **Pilot (50 records):** reused the pre-existing AI draft 1 unchanged, with no merging or voting.
- **Expansion (350 records):** new labels from a single pass by Claude (`claude-opus-5-5`) under guidelines v1.1. French records were labelled with the standard rules. There was no second pass and no human review. Method, prompt version and limitations are in `provenance.json`.

| Stance | All 400 | Expansion 350 |
|---|---:|---:|
| Neutral | 219 | 201 |
| Support | 140 | 116 |
| Unclear | 40 | 32 |
| Oppose | 1 | 1 |
| Mixed | 0 | 0 |

| Sentiment (tone, not stance) | All 400 |
|---|---:|
| Neutral | 270 |
| Negative | 73 |
| Positive | 45 |
| Mixed | 12 |

Stance and sentiment differ. 19 records carry Negative sentiment and Support stance, because criticism of delivery is not opposition under guidelines v1.1.

## 3. Split methodology

- **Leakage groups:** union-find joins records that share any of the following:
  - a source record;
  - a pool duplicate group;
  - a corpus near-duplicate group;
  - a template key;
  - a character 5-gram near-duplicate (Jaccard ≥ 0.5 or containment ≥ 0.9).
- **Normalised identical texts:** asserted to share a group. Tests confirm that no group and no normalised text crosses a split.
- **Split:** a grouped, stratified split into test (about 20%), validation (about 20%) and train (about 60%). It uses the first fold of StratifiedGroupKFold(5) for test, then of StratifiedGroupKFold(4) for validation, with seed 42.
- **Model selection:** C is chosen from {0.1, 0.3, 1, 3, 10} on validation macro-F1 only. The chosen model is refitted on train plus validation and evaluated once on test.
- **Baselines:** majority class, and stratified random averaged over 1,000 draws.
- **Intervals:** 95% intervals come from a group bootstrap of the test set.
- **Split sensitivity:** the full protocol was repeated for split seeds 43–46. This was never used for selection.

## 4. Metrics versus baselines

### Stance, 3 classes (Support, Neutral, Unclear)

Test set: 80 records (Support 28, Neutral 44, Unclear 8).

| Model | Test macro-F1 | 95% CI | Accuracy | Mean over 5 split seeds |
|---|---:|---|---:|---:|
| Majority (Neutral) | 0.237 | 0.20–0.26 | 0.550 | 0.237 |
| Stratified random | 0.327 | — | — | — |
| Logistic Regression (C=3) | 0.364 | 0.29–0.44 | 0.525 | 0.382 ± 0.016 |
| Linear SVM (C=0.3) | 0.396 | 0.32–0.47 | 0.575 | 0.377 ± 0.023 |

- **Per-class results (SVM):** Support had precision 0.52 and recall 0.57. Neutral had precision 0.61 and recall 0.68.
- **Unclear is never predicted** by either model, so its F1 is 0.
- **Confusion matrix (SVM):** rows are the true labels, columns are the predictions, both in the order Support, Neutral, Unclear: [[16,12,0],[14,30,0],[1,7,0]].
- **Overfitting:** both models fit train plus validation perfectly.
- **Accuracy:** the SVM's accuracy is barely above the majority baseline.

### Stance, Support vs Neutral

Test set: 72 records (Support 28, Neutral 44).

| Model | Test macro-F1 | 95% CI | Accuracy | Mean over 5 split seeds |
|---|---:|---|---:|---:|
| Majority | 0.379 | — | — | 0.379 |
| Stratified random | 0.498 | — | — | — |
| Logistic Regression (C=0.3) | 0.611 | 0.49–0.72 | 0.611 | 0.630 ± 0.035 |
| Linear SVM (C=0.1) | 0.625 | 0.51–0.73 | 0.625 | 0.645 ± 0.030 |

On the Support/Neutral test set, Support recall is 0.79 and Neutral recall is 0.52. The models over-predict Support.

### Existing sentiment models on consultation text

These models are inference only. Tweet-test scores are not consultation accuracy.

| Check | Sentiment140 (binary) | SentiMix (Hinglish, 3 classes) |
|---|---|---|
| Full corpus, 10,108 entries | positive 6,932 / negative 3,176 | neutral 7,395 / positive 2,689 / negative 24 |
| French entries, 1,087 | 1,063 positive | 1,083 neutral |
| Share of predictions with confidence < 0.6 | 32% | 96% |
| Agreement with AI labels (macro-F1) | 0.65 on 118 polar records | 0.35, 3-class |

- **Sentiment140:** has no neutral class. It labelled 213 of the 270 AI-Neutral records positive.
- **SentiMix:** never matched an AI-Negative record. It is trained on romanised Hindi-English, which does not match this corpus's languages.
- **Licences:** neither model's training data has a licence file in the workspace, so the terms are unverified.
- **Conclusion:** neither model is fit for consultation sentiment.

### Consultation-domain sentiment baseline (AI labels)

Classes are Positive, Negative and Neutral. Mixed (12 records) is excluded. The test set has 78 records (Negative 15, Neutral 54, Positive 9).

| Model | Test macro-F1 | 95% CI | Accuracy | Mean over 5 split seeds |
|---|---:|---|---:|---:|
| Majority (Neutral) | 0.273 | — | 0.692 | 0.273 |
| Stratified random | 0.335 | — | — | — |
| Logistic Regression (C=1) | 0.570 | 0.41–0.71 | 0.705 | 0.574 ± 0.061 |
| Linear SVM (C=1) | 0.593 | 0.42–0.73 | 0.756 | 0.561 ± 0.067 |

For the SVM, Negative recall is 0.40 and Positive recall is 0.33.

### Concern discovery: audit of the 74 groups

The selected configuration is `strip__umap_m25s5`, seed 42.

- **Tiers:**
  - 30 candidate;
  - 24 needs_review;
  - 10 non_concern_pattern;
  - 9 unreliable;
  - 1 campaign_or_template.
- **Group sizes:** 25 to 354 entries, median 62. The largest group holds 5.7% of clustered entries.
- **Noise:** 37% of entries are noise, with similar rates in English (37%) and French (36%).
- **Question-driven groups:** 22 groups draw at least 80% of their entries from one question. Of these, 11 come from Q14 ("why", satisfaction with open.canada.ca) and 10 from the compilation comment field.
- **Generic and weak groups:**
  - 18 groups have generic-only labels, such as "open / government".
  - 11 carry the "mixed" flag.
  - 19 have weak cohesion.
- **Language:** no group is majority French, and 6 groups contain no French at all.
- **Configuration sensitivity:** each row compares the selected run with one other run on the core entries.

| Compared with | ARI on core | Groups with best-match Jaccard ≥ 0.5 |
|---|---:|---:|
| Seed 43 | 0.905 | 62 of 74 |
| Seed 44 | 0.920 | 64 |
| Leaf selection | 0.931 | 66 |
| Template-deduplicated | 0.923 | 63 |
| Short texts removed | 0.805 | 36 |
| m50/s10 | 0.587 | 26 |
| Raw (unstripped) text | 0.617 | 40 |
| k-means k=80 | 0.476 | 18 |

**Conclusion:** groups are stable across seeds but depend on the minimum-cluster-size setting and on preprocessing choices.

**Field-centering alternative:** compared with the selected configuration, it produced 80 clusters and less noise (32%). Question-driven groups fell from 22 to 18, and Cramér's V with the question field fell from 0.371 to 0.315. Its minimum seed ARI is 0.556, below the 0.85 selection rule, so it was **not adopted**.

### Prioritization: 55 ranked and 19 not ranked

- **Reproducibility:** the recomputed ranking matches `ranking.csv` exactly.
- **Not-ranked groups:** 10 are non-concern patterns and 9 are unreliable.
- **Resampling stability:** 200 subsamples of 80% of source records gave a mean Kendall tau of 0.906 (minimum 0.859).
  - All ten top-10 groups stay in the top 10 in at least 90% of replicates.
  - Groups 30, 25, 57 and 26 stay in the top 5.
  - The median 5–95% rank interval spans 8 places.
- **Weight sensitivity:** this is the larger risk, with a minimum Kendall tau of 0.49. The median gap between adjacent scores is 0.0029.
- **Contributions:** mean shares of the score are:
  - independent volume 38%;
  - reliability 16%;
  - evidence 13%;
  - independence 13%;
  - dataset diversity 8%;
  - field diversity 6%;
  - tier 5%.
- **Evidence:** 645 quotes were re-checked, 275 ranking quotes and 370 representative examples. Every check passed with no failures:
  - each quote is a substring of its full text;
  - the full text equals the corpus text;
  - source dataset, record, row and field agree with the corpus;
  - each entry belongs to its stated cluster;
  - each text hash matches.

  The raw-file lineage check also passed. All 74 groups have representative examples.

Counts are entries, texts or source records, never people. No impact or urgency labels were created.

### New-comment inference smoke test

The sample is 300 corpus entries that share no source record and no normalised text with the labelled pool, plus 7 edge cases. There are no labels, so no accuracy was measured.

| Item | Result |
|---|---|
| Inference time, 307 texts | 0.24 s |
| Empty, whitespace and 1-word inputs | Returned as "too_short_or_empty" |
| Concern groups assigned | 0, with the refusal reason recorded |

There are two visible failure modes:

- **French:** of 25 French entries, 23 were predicted Neutral.
- **Opposing comments:** an explicitly opposing sentence was predicted Support, because Oppose is untrainable from the available labels.

## 5. Saved paths

| Purpose | Path |
|---|---|
| AI labels, provenance, manifest | `data/ai_labels/stance_sentiment_ai_v1/` |
| Stance pipelines, metrics, splits, test predictions | `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/`, `..._support_vs_neutral/` |
| Sentiment domain check | `data/sentiment_evaluation/consultation_domain_check_v1/` |
| Consultation sentiment pipelines | `data/sentiment_training/consultation_ai_labels_tfidf_v1/` |
| Concern audit | `data/full_corpus/concern_audit_v1/` |
| Field-centering alternative | `data/full_corpus/experiments_field_centered_v1/` |
| Prioritization audit | `data/full_corpus/prioritization_audit_v1/` |
| Inference module and smoke test | `models/new_comment_inference.py`, `data/inference/new_comment_inference_v1/` |
| Logs | `logs/model_phase_v1/` |
| Tests | `tests/test_model_phase_v1.py` |

Every output directory is write-once and has a `manifest.json` with SHA-256 hashes. Pipelines load only if their hashes match.

## 6. Reproduce

Each script refuses to overwrite an existing output. To rerun one, delete or rename its output folder first.

```powershell
python scripts\build_ai_working_labels_v1.py
python scripts\train_stance_ai_labels_v2.py --tag pool400_ai_v1_3class
python scripts\train_stance_ai_labels_v2.py --tag pool400_ai_v1_support_vs_neutral --exclude-labels Unclear
python scripts\run_sentiment_domain_check_v1.py
python scripts\train_sentiment_ai_labels_v1.py
python scripts\run_concern_audit_v1.py
python scripts\run_prioritization_audit_v1.py
python scripts\run_new_comment_inference_v1.py --smoke-test
python scripts\run_new_comment_inference_v1.py --input comments.csv --text-column text --out predictions.csv
python -m pytest tests -q -p no:cacheprovider
```

The full test suite passed with 502 tests. The research evaluation run with `--no-write` reports 50 passed. Its 10 blocked and 2 not-run checks are all pre-existing human-annotation blockers.

## 7. Limitations and frontend-ready outputs

### Limitations

- **Label source:** every label comes from a single AI labeller. No agreement statistic exists, and the labels may carry systematic AI bias. The pilot and expansion labels come from different AI passes.
- **Small test sets:** the stance and sentiment test sets have 72 to 80 records, so confidence intervals span about 0.15 to 0.30.
- **Stance model:** the 3-class model is weak, never predicts Unclear, and cannot detect Oppose or Mixed.
- **French:** French input is poorly handled by every supervised model.
- **Cluster dependence:** clusters depend on configuration. 22 groups are question-driven, and nothing has been human-reviewed.
- **Ranking weights:** rankings are sensitive to the weights chosen. The ranking orders groups for inspection; it does not measure importance.

### Frontend-ready outputs

The frontend was not changed. The following outputs could be integrated later through the read-only ResultsStore, each with the stated caveats:

- **Concern groups:** the verified groups, the ranking with per-signal contributions, and representative quotes that are now fully traceable.
- **Ranking stability:** `prioritization_audit_v1/rank_stability.csv`, with rank intervals and top-10 frequency.
- **Group quality flags:** `concern_audit_v1/group_audit.csv`, with question-driven, generic, mixed and language flags.
- **Model suggestions:** stance and sentiment suggestions, labelled "AI-label-trained, provisional". They are not suitable as headline metrics.
- **Not for display:** Sentiment140 and SentiMix outputs. They are not suitable for consultation text.

## 8. Status per component

| Component | Status |
|---|---|
| Stance, Support/Neutral/Unclear (AI labels) | Trained and evaluated. Weak: macro-F1 about 0.38, majority baseline 0.24. |
| Stance, Support vs Neutral (AI labels) | Trained and evaluated. Macro-F1 about 0.63, majority baseline 0.38. |
| Stance, Oppose and Mixed | Blocked: insufficient labels. |
| Stance, human-verified | Blocked: no human annotations. |
| Sentiment, Sentiment140 and SentiMix on consultation text | Evaluated. Not fit for this domain. |
| Sentiment, consultation baseline (AI labels) | Trained and evaluated. Macro-F1 about 0.57, majority baseline 0.27. |
| Concern discovery | Audited. Stable across seeds but configuration-dependent. Not human-validated. |
| Concern discovery, field-centering alternative | Run. Not adopted because it is seed-unstable. |
| Prioritization | Reproduced and stable under resampling. Weight-sensitive. Evidence fully verified. |
| New-comment inference, stance and sentiment | Supported, with hash checks. |
| New-comment inference, concern group and priority | Unsupported. Explicitly refused. |
