# Stance model comparison using repository reference labels

The existing stance labels are treated as the project reference targets. This terminology does not assert independent human verification; source provenance remains in data/ai_labels/stance_sentiment_ai_v1/provenance.json.

The previously inspected seed-42 80-row test split is development data. This run creates a new source-record and near-duplicate grouped test fold for this fit. However, the full reference pool has been exposed in prior development runs, so these results are development estimates, not an untouched independent evaluation.

New grouped split (seed 20261029): 239 train / 80 validation / 80 test; 398 leakage groups.
The model schema is Support / Neutral / Unclear. One Oppose reference record is marked `excluded_outside_three_class_model`; all 400 source records, label values and record identifiers are retained in `split_assignments.csv`.
Grouped boundaries join same source records, exact duplicate groups, corpus near-duplicate groups, templates and lexical near-duplicates.
Validation macro-F1 selected configurations before any new-test metrics were computed.

| Test model | Frozen configuration | N | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| selected | word_char_linsvm_weight-balanced_C-0.1 | 80 | 0.6125 | 0.4051 | 0.5663 |
| word_char_validation_best | word_char_linsvm_weight-balanced_C-0.1 | 80 | 0.6125 | 0.4051 | 0.5663 |
| word_only_word_linsvm_validation_best | word_linsvm_weight-balanced_C-3 | 80 | 0.5625 | 0.394 | 0.5368 |
| word_only_word_logreg_validation_best | word_logreg_weight-balanced_C-1 | 80 | 0.55 | 0.4426 | 0.5389 |
| majority_class | always predict Neutral | 80 | 0.55 | 0.2366 | 0.3903 |

## selected: per-class test metrics

| Reference class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.6316 | 0.4286 | 0.5106 | 28 |
| Neutral | 0.6066 | 0.8409 | 0.7048 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |
Confusion matrix, rows reference / columns predicted in Support, Neutral, Unclear order: `[[12, 16, 0], [7, 37, 0], [0, 8, 0]]`

## word_char_validation_best: per-class test metrics

| Reference class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.6316 | 0.4286 | 0.5106 | 28 |
| Neutral | 0.6066 | 0.8409 | 0.7048 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |
Confusion matrix, rows reference / columns predicted in Support, Neutral, Unclear order: `[[12, 16, 0], [7, 37, 0], [0, 8, 0]]`

## word_only_word_linsvm_validation_best: per-class test metrics

| Reference class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.5312 | 0.6071 | 0.5667 | 28 |
| Neutral | 0.5957 | 0.6364 | 0.6154 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |
Confusion matrix, rows reference / columns predicted in Support, Neutral, Unclear order: `[[17, 11, 0], [15, 28, 1], [0, 8, 0]]`

## word_only_word_logreg_validation_best: per-class test metrics

| Reference class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.5 | 0.6071 | 0.5484 | 28 |
| Neutral | 0.6047 | 0.5909 | 0.5977 | 44 |
| Unclear | 0.3333 | 0.125 | 0.1818 | 8 |
Confusion matrix, rows reference / columns predicted in Support, Neutral, Unclear order: `[[17, 11, 0], [16, 26, 2], [1, 6, 1]]`

## majority_class: per-class test metrics

| Reference class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Support | 0.0 | 0.0 | 0.0 | 28 |
| Neutral | 0.55 | 1.0 | 0.7097 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |
Confusion matrix, rows reference / columns predicted in Support, Neutral, Unclear order: `[[0, 28, 0], [0, 44, 0], [0, 8, 0]]`

## Language subsets

English and French test subsets are recorded in `metrics.json` for the selected model, the validation-best word+character model, and word-only baselines. Interpret small subsets cautiously.

## Remaining limitation

This is a new held-out fold for this fit, but every record in the reference pool has been seen in previous development runs. Treat metrics as development estimates, not untouched test performance. It is also not an unseen consultation-source test. A clean external evaluation requires newly acquired reference labels from source records kept out of all previous development.

Reproduce: `python scripts/experiment_stance_reference_models_v2.py --out-dir reports/stance_reference_model_comparison_rerun1 --seed 20261029`
