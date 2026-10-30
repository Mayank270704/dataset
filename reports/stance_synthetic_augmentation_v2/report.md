# Controlled synthetic stance augmentation

Only Support, Neutral, and Unclear are modeled. Original reference records are unchanged.
The full reference pool has prior development exposure, so test values below are development estimates. Synthetic examples are training data and are not evidence of real-world accuracy.

Split seed 20261029: 239 original train / 80 validation / 80 test records; 398 grouped source/duplicate units.
Generated 240 rows (80 per class) at `synthetic_training_rows.jsonl` with seed 20261029.
Held-out exact matches: 0; near matches at char cosine >=0.72 or sequence ratio >=0.82: 0.
Synthetic rows appear as `train` only in `split_manifest.jsonl`; validation and test rows are original records.

| Condition | Synthetic/original train | Synthetic rows | Selected C / weight | Accuracy | Macro-F1 | Weighted-F1 | Test N |
|---|---:|---:|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0.1 / balanced | 0.6125 | 0.4051 | 0.5663 | 80 |
| ratio_0.25 | 0.25 | 60 | 0.1 / balanced | 0.625 | 0.4815 | 0.5923 | 80 |
| ratio_0.5 | 0.5 | 120 | 0.3 / balanced | 0.65 | 0.5063 | 0.6305 | 80 |
| ratio_1 | 1 | 239 | 0.1 / balanced | 0.65 | 0.4921 | 0.6152 | 80 |

## baseline

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.6316 | 0.4286 | 0.5106 | 28 |
| Neutral | 0.6066 | 0.8409 | 0.7048 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |
Confusion matrix, reference rows and predicted columns (Support, Neutral, Unclear): `[[12, 16, 0], [7, 37, 0], [0, 8, 0]]`
Training counts: {"original_train_total": 239, "original_train_by_class": {"Support": 84, "Neutral": 131, "Unclear": 24}, "synthetic_train_total": 0, "synthetic_train_by_class": {"Support": 0, "Neutral": 0, "Unclear": 0}, "final_fit_total_including_original_validation": 319}
Language test subsets: {"English": 72, "French": 8}

## ratio_0.25

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.6316 | 0.4286 | 0.5106 | 28 |
| Neutral | 0.6167 | 0.8409 | 0.7115 | 44 |
| Unclear | 1.0 | 0.125 | 0.2222 | 8 |
Confusion matrix, reference rows and predicted columns (Support, Neutral, Unclear): `[[12, 16, 0], [7, 37, 0], [0, 7, 1]]`
Training counts: {"original_train_total": 239, "original_train_by_class": {"Support": 84, "Neutral": 131, "Unclear": 24}, "synthetic_train_total": 60, "synthetic_train_by_class": {"Support": 20, "Neutral": 20, "Unclear": 20}, "final_fit_total_including_original_validation": 379}
Language test subsets: {"English": 72, "French": 8}

## ratio_0.5

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.6667 | 0.5714 | 0.6154 | 28 |
| Neutral | 0.6604 | 0.7955 | 0.7216 | 44 |
| Unclear | 0.3333 | 0.125 | 0.1818 | 8 |
Confusion matrix, reference rows and predicted columns (Support, Neutral, Unclear): `[[16, 11, 1], [8, 35, 1], [0, 7, 1]]`
Training counts: {"original_train_total": 239, "original_train_by_class": {"Support": 84, "Neutral": 131, "Unclear": 24}, "synthetic_train_total": 120, "synthetic_train_by_class": {"Support": 40, "Neutral": 40, "Unclear": 40}, "final_fit_total_including_original_validation": 439}
Language test subsets: {"English": 72, "French": 8}

## ratio_1

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.7059 | 0.4286 | 0.5333 | 28 |
| Neutral | 0.6393 | 0.8864 | 0.7429 | 44 |
| Unclear | 0.5 | 0.125 | 0.2 | 8 |
Confusion matrix, reference rows and predicted columns (Support, Neutral, Unclear): `[[12, 15, 1], [5, 39, 0], [0, 7, 1]]`
Training counts: {"original_train_total": 239, "original_train_by_class": {"Support": 84, "Neutral": 131, "Unclear": 24}, "synthetic_train_total": 239, "synthetic_train_by_class": {"Support": 80, "Neutral": 80, "Unclear": 79}, "final_fit_total_including_original_validation": 558}
Language test subsets: {"English": 72, "French": 8}

## Limitations

The seed-20261029 grouped holdout was created after prior work exposed the full reference pool. It is not an untouched evaluation of unseen consultations. The Unclear class combines guideline-specific cases, and language-comprehension difficulty depends on annotator ability rather than only comment text; synthetic examples for that case are protocol scenarios and should not be interpreted as a learnable text signal.

Reproduce: `python scripts/experiment_stance_synthetic_augmentation_v1.py --out-dir reports/stance_synthetic_augmentation_rerun1 --seed 20261029 --synthetic-per-class 80 --ratios 0,0.25,0.5,1`
