# Context feature stance benchmark

Existing comment, question, context, and language fields only; no synthetic examples, model downloads, reviewer notes, source labels, or target-derived features. All reported metrics are agreement with previously exposed, unverified repository reference labels, not independent human-verified accuracy.

The 3-repeat x 5-fold outer assignments are copied from the v4 benchmark's shared source/duplicate-grouped manifest. Each input mode uses the same 399 rows and folds. C is selected independently within each outer training fold by three-fold grouped validation. Text vectorizers are fitted inside each inner and outer training fold.

Input coverage is complete for comment, question wording, question context, context, and source language on all 399 rows. Context and question_context are identical in 399/399 rows, so the combined mode includes that field once. Verified comment-language metadata is available on 269 rows; the other 130 rows describe questionnaire language and are marked unknown for the language-aware mode. No rows have missing question/context values, so missing-context subgroup performance is not estimable from this pool.

## Source coverage

| Source | Rows | Comment | Question | Context | Language metadata |
|---|---:|---:|---:|---:|---:|
| consultation_compilation_2017_18 | 179 | 179/179 | 179/179 | 179/179 | 179/179 |
| get_involved_questionnaire_2017_18 | 130 | 130/130 | 130/130 | 130/130 | 130/130 |
| individual_comments_2016_18 | 90 | 90/90 | 90/90 | 90/90 | 90/90 |

| Input mode | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Weighted-F1 mean +/- SD | Unclear P/R/F1 mean |
|---|---:|---:|---:|---:|
| comment_only | 0.6007 +/- 0.0427 | 0.4308 +/- 0.0443 | 0.5751 +/- 0.0406 | 0.1667 / 0.0333 / 0.0548 |
| question_plus_comment | 0.6558 +/- 0.0311 | 0.5007 +/- 0.0483 | 0.6314 +/- 0.0324 | 0.3511 / 0.1250 / 0.1773 |
| context_question_comment | 0.6525 +/- 0.0507 | 0.5032 +/- 0.0618 | 0.6345 +/- 0.0516 | 0.2668 / 0.1417 / 0.1791 |
| verified_language_comment | 0.6224 +/- 0.0402 | 0.4510 +/- 0.0521 | 0.5957 +/- 0.0399 | 0.2000 / 0.0500 / 0.0778 |

## Pooled repeated out-of-fold prediction counts

Each source row appears once per outer repeat.

| Input mode | Support | Neutral | Unclear |
|---|---:|---:|---:|
| comment_only | 423 (35.3%) | 750 (62.7%) | 24 (2.0%) |
| context_question_comment | 400 (33.4%) | 738 (61.7%) | 59 (4.9%) |
| question_plus_comment | 372 (31.1%) | 785 (65.6%) | 40 (3.3%) |
| verified_language_comment | 425 (35.5%) | 749 (62.6%) | 23 (1.9%) |

Per-fold, per-class, confusion, language-subset, inner C-selection, and missing-context outputs are in metrics.json. The prediction file contains record IDs and outcomes but no raw comment/context text.

The context missing subgroup has n=0. The context-present subgroup is reported for every mode; do not infer a missing-context effect from this dataset. Unclear remains a heterogeneous label, including cases where its reason is not recoverable from text.

The best contextual arm is question plus comment (0.6558 accuracy, 0.5007 macro-F1); full context plus question is similar (0.6525, 0.5032). The existing v4 shared-fold benchmark reports production word-only at 0.5898 accuracy / 0.4036 macro-F1 and the staged v1 synthetic recipe at 0.6191 / 0.4674. These cross-report comparisons are contextual rather than fully paired because this benchmark selects C within each outer training fold. No context candidate is staged: Unclear recall remains 0.125-0.142 (15-17 correct among 120 repeated out-of-fold Unclear predictions), the 40 underlying Unclear rows are previously exposed development references, and the current inference API has no question/context input contract. That prevents treating these results as a reliable deployment improvement. Production and existing staging artifacts remain untouched.

Reproduce: python scripts/benchmark_stance_context_v1.py --out-dir reports/stance_context_benchmark_v2
