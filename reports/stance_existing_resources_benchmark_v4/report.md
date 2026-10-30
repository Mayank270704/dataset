# Existing-resource stance grouped benchmark and abstention analysis

No new examples or model downloads were used. TF-IDF recipes, synthetic pools, grouped folds, and cached multilingual MiniLM embeddings were read from existing repository assets. The reference pool has prior development exposure; these are development robustness estimates, not unseen-source results.

Three-class reference rows: 399 ({"Support": 140, "Neutral": 219, "Unclear": 40}); languages: {"English": 364, "French": 35}. Source/duplicate-grouped folds: 3 repeats x 5 folds. Hyperparameters were frozen from the existing validation sweep; no old development test scores were used to select them.

The source pool has 400 records: 140 Support, 219 Neutral, 40 Unclear, and one Oppose. The singleton Oppose record remains in the original source manifest and is excluded from three-class fitting. All 400 rows carry the same AI working-label provenance; this is not independent human ground truth. The 399 modeled rows comprise 364 English and 35 French examples. All 399 map to corpus source records; 398 modeled leakage groups remain after source/duplicate grouping. There are no normalized exact duplicate texts or exact-text cross-label conflicts among the modeled rows.

Available data include comment text, source and record identifiers, question wording/context, language and duplicate metadata, plus label/review provenance. Identifiers and label/review fields are excluded. The current model input is only `original_text`; question wording, question context, `context`, source field, question slug, and language are not features. This matters because guideline Unclear reasons include missing context and language-comprehension inability, while the schema also has a `Mixed` concept not modeled here.

## Prior validation sweep and negative results

The existing prior-exposed validation sweep tested 60 settings over word TF-IDF and word+character TF-IDF with LinearSVC/Logistic Regression, plus cached multilingual MiniLM with Logistic Regression; C, class weight, n-gram/features, and minimum document frequency are recorded in the source metrics. Best per family (validation only):

| Feature/model family | Selected setting | Validation accuracy | Validation macro-F1 |
|---|---|---:|---:|
| word / linsvm | word_linsvm_cw-none_C-0.3 | 0.6375 | 0.4152 |
| word / logreg | word_logreg_cw-none_C-3.0 | 0.6250 | 0.4177 |
| word_char / linsvm | word_char_linsvm_cw-balanced_C-0.3 | 0.5750 | 0.4579 |
| word_char / logreg | word_char_logreg_cw-balanced_C-3.0 | 0.5625 | 0.4469 |
| multilingual_minilm_embeddings / logreg | multilingual_minilm_logreg_cw-balanced_C-3.0 | 0.5500 | 0.5029 |

The validation-best MiniLM recipe did not transfer into this same-fold grouped comparison as the accuracy leader. Cached MiniLM LinearSVC and Logistic Regression are below the word+character synthetic recipes on grouped mean accuracy; no pretrained model download was attempted. The full per-fold, class, confusion, language, configuration, and software-version results are in `metrics.json`.

| Recipe | Accuracy mean ± SD | Macro-F1 mean ± SD | Weighted-F1 mean ± SD | Unclear P/R/F1 mean |
|---|---:|---:|---:|---:|
| production_word_only | 0.5898 ± 0.0434 | 0.4036 ± 0.0303 | 0.5585 ± 0.0404 | 0.0000 / 0.0000 / 0.0000 |
| word_linsvm_validation_best | 0.5915 ± 0.0382 | 0.3796 ± 0.0338 | 0.5384 ± 0.0404 | 0.0000 / 0.0000 / 0.0000 |
| word_logreg_validation_best | 0.5974 ± 0.0425 | 0.3982 ± 0.0336 | 0.5564 ± 0.0416 | 0.0000 / 0.0000 / 0.0000 |
| word_char_linsvm_validation_best | 0.6090 ± 0.0432 | 0.4321 ± 0.0427 | 0.5819 ± 0.0408 | 0.1333 / 0.0250 / 0.0415 |
| word_char_logreg_validation_best | 0.6015 ± 0.0384 | 0.4349 ± 0.0398 | 0.5790 ± 0.0371 | 0.1556 / 0.0333 / 0.0536 |
| candidate_v1_recipe_v4_ratio_0_5 | 0.6191 ± 0.0411 | 0.4674 ± 0.0394 | 0.5985 ± 0.0389 | 0.3222 / 0.0833 / 0.1270 |
| candidate_v2_recipe_clean_ratio_0_5 | 0.6115 ± 0.0484 | 0.4596 ± 0.0466 | 0.5909 ± 0.0459 | 0.3278 / 0.0750 / 0.1148 |
| cached_minilm_logreg_validation_best | 0.5289 ± 0.0439 | 0.4444 ± 0.0413 | 0.5450 ± 0.0385 | 0.1230 / 0.2167 / 0.1541 |
| cached_minilm_linsvm | 0.5590 ± 0.0472 | 0.4420 ± 0.0399 | 0.5616 ± 0.0392 | 0.1034 / 0.1417 / 0.1180 |
| explore_word_unigram_linsvm | 0.5857 ± 0.0416 | 0.4093 ± 0.0316 | 0.5583 ± 0.0402 | 0.0556 / 0.0167 / 0.0255 |
| explore_word_trigram_linsvm | 0.5991 ± 0.0483 | 0.4083 ± 0.0331 | 0.5653 ± 0.0447 | 0.0000 / 0.0000 / 0.0000 |
| explore_word_cap10k_min_df2_linsvm | 0.5622 ± 0.0365 | 0.4127 ± 0.0306 | 0.5453 ± 0.0314 | 0.1333 / 0.0417 / 0.0630 |
| explore_char_2_5_cap15k_min_df1_linsvm | 0.6057 ± 0.0397 | 0.4236 ± 0.0287 | 0.5783 ± 0.0371 | 0.0333 / 0.0083 / 0.0133 |
| majority_class_baseline | 0.5489 ± 0.0024 | 0.2363 ± 0.0007 | 0.3890 ± 0.0027 | 0.0000 / 0.0000 / 0.0000 |

None of the measured configurations reached 90% mean grouped-CV accuracy. The best mean is the v1 word+character synthetic recipe at 0.6191. This is agreement with the repository's unverified AI reference labels, not human-verified stance accuracy.

## Abstention operating points

For each saved validation fold, the top X% of rows by the model's top-two decision-score gap are accepted. Coverage thresholds are fixed without reference labels; metrics are then calculated on accepted rows. Margins are not calibrated probabilities, and these operating curves do not imply that abstention resolves the ambiguous Unclear construct. Full reference-label versus predicted-label confusion matrices are recorded per recipe in `metrics.json`; the raw prediction file enables per-record error review without changing labels.

| Recipe | Target coverage | Actual coverage | Accepted accuracy | Accepted macro-F1 | Accepted Unclear n | Unclear recall among accepted |
|---|---:|---:|---:|---:|---:|---:|
| cached_minilm_linsvm | 100% | 1.000 | 0.5589 | 0.4468 | 120 | 0.1417 |
| cached_minilm_linsvm | 90% | 0.902 | 0.5750 | 0.4559 | 111 | 0.1261 |
| cached_minilm_linsvm | 80% | 0.802 | 0.5938 | 0.4703 | 107 | 0.1215 |
| cached_minilm_linsvm | 70% | 0.702 | 0.6107 | 0.4714 | 89 | 0.0899 |
| cached_minilm_linsvm | 50% | 0.501 | 0.6567 | 0.4946 | 64 | 0.0625 |
| cached_minilm_logreg_validation_best | 100% | 1.000 | 0.5288 | 0.4453 | 120 | 0.2167 |
| cached_minilm_logreg_validation_best | 90% | 0.902 | 0.5463 | 0.4627 | 105 | 0.2476 |
| cached_minilm_logreg_validation_best | 80% | 0.802 | 0.5635 | 0.4795 | 95 | 0.2632 |
| cached_minilm_logreg_validation_best | 70% | 0.702 | 0.5702 | 0.4836 | 86 | 0.2442 |
| cached_minilm_logreg_validation_best | 50% | 0.501 | 0.6150 | 0.5242 | 63 | 0.2698 |
| candidate_v1_recipe_v4_ratio_0_5 | 100% | 1.000 | 0.6190 | 0.4689 | 120 | 0.0833 |
| candidate_v1_recipe_v4_ratio_0_5 | 90% | 0.902 | 0.6370 | 0.4822 | 110 | 0.0818 |
| candidate_v1_recipe_v4_ratio_0_5 | 80% | 0.802 | 0.6469 | 0.4853 | 94 | 0.0745 |
| candidate_v1_recipe_v4_ratio_0_5 | 70% | 0.702 | 0.6560 | 0.4926 | 75 | 0.0800 |
| candidate_v1_recipe_v4_ratio_0_5 | 50% | 0.501 | 0.6750 | 0.4918 | 55 | 0.0545 |
| candidate_v2_recipe_clean_ratio_0_5 | 100% | 1.000 | 0.6115 | 0.4606 | 120 | 0.0750 |
| candidate_v2_recipe_clean_ratio_0_5 | 90% | 0.902 | 0.6269 | 0.4666 | 107 | 0.0654 |
| candidate_v2_recipe_clean_ratio_0_5 | 80% | 0.802 | 0.6469 | 0.4855 | 91 | 0.0769 |
| candidate_v2_recipe_clean_ratio_0_5 | 70% | 0.702 | 0.6631 | 0.4912 | 74 | 0.0676 |
| candidate_v2_recipe_clean_ratio_0_5 | 50% | 0.501 | 0.6983 | 0.5041 | 51 | 0.0588 |
| explore_char_2_5_cap15k_min_df1_linsvm | 100% | 1.000 | 0.6057 | 0.4241 | 120 | 0.0083 |
| explore_char_2_5_cap15k_min_df1_linsvm | 90% | 0.902 | 0.6120 | 0.4279 | 114 | 0.0088 |
| explore_char_2_5_cap15k_min_df1_linsvm | 80% | 0.802 | 0.6271 | 0.4287 | 93 | 0.0000 |
| explore_char_2_5_cap15k_min_df1_linsvm | 70% | 0.702 | 0.6500 | 0.4430 | 77 | 0.0000 |
| explore_char_2_5_cap15k_min_df1_linsvm | 50% | 0.501 | 0.6833 | 0.4593 | 52 | 0.0000 |
| explore_word_cap10k_min_df2_linsvm | 100% | 1.000 | 0.5622 | 0.4117 | 120 | 0.0417 |
| explore_word_cap10k_min_df2_linsvm | 90% | 0.902 | 0.5806 | 0.4263 | 108 | 0.0463 |
| explore_word_cap10k_min_df2_linsvm | 80% | 0.802 | 0.6083 | 0.4414 | 91 | 0.0440 |
| explore_word_cap10k_min_df2_linsvm | 70% | 0.702 | 0.6143 | 0.4278 | 79 | 0.0127 |
| explore_word_cap10k_min_df2_linsvm | 50% | 0.501 | 0.6467 | 0.4423 | 59 | 0.0000 |
| explore_word_trigram_linsvm | 100% | 1.000 | 0.5990 | 0.4085 | 120 | 0.0000 |
| explore_word_trigram_linsvm | 90% | 0.902 | 0.6028 | 0.4113 | 113 | 0.0000 |
| explore_word_trigram_linsvm | 80% | 0.802 | 0.6135 | 0.4162 | 99 | 0.0000 |
| explore_word_trigram_linsvm | 70% | 0.702 | 0.6238 | 0.4179 | 89 | 0.0000 |
| explore_word_trigram_linsvm | 50% | 0.501 | 0.6517 | 0.4342 | 71 | 0.0000 |
| explore_word_unigram_linsvm | 100% | 1.000 | 0.5856 | 0.4109 | 120 | 0.0167 |
| explore_word_unigram_linsvm | 90% | 0.902 | 0.6009 | 0.4164 | 109 | 0.0092 |
| explore_word_unigram_linsvm | 80% | 0.802 | 0.6104 | 0.4214 | 96 | 0.0104 |
| explore_word_unigram_linsvm | 70% | 0.702 | 0.6155 | 0.4248 | 88 | 0.0114 |
| explore_word_unigram_linsvm | 50% | 0.501 | 0.6400 | 0.4230 | 65 | 0.0000 |
| production_word_only | 100% | 1.000 | 0.5898 | 0.4039 | 120 | 0.0000 |
| production_word_only | 90% | 0.902 | 0.6000 | 0.4107 | 112 | 0.0000 |
| production_word_only | 80% | 0.802 | 0.6156 | 0.4192 | 99 | 0.0000 |
| production_word_only | 70% | 0.702 | 0.6286 | 0.4229 | 85 | 0.0000 |
| production_word_only | 50% | 0.501 | 0.6433 | 0.4298 | 71 | 0.0000 |
| word_char_linsvm_validation_best | 100% | 1.000 | 0.6090 | 0.4323 | 120 | 0.0250 |
| word_char_linsvm_validation_best | 90% | 0.902 | 0.6194 | 0.4310 | 112 | 0.0089 |
| word_char_linsvm_validation_best | 80% | 0.802 | 0.6333 | 0.4329 | 92 | 0.0000 |
| word_char_linsvm_validation_best | 70% | 0.702 | 0.6524 | 0.4437 | 82 | 0.0000 |
| word_char_linsvm_validation_best | 50% | 0.501 | 0.7033 | 0.4698 | 51 | 0.0000 |
| word_char_logreg_validation_best | 100% | 1.000 | 0.6015 | 0.4352 | 120 | 0.0333 |
| word_char_logreg_validation_best | 90% | 0.902 | 0.6102 | 0.4435 | 113 | 0.0354 |
| word_char_logreg_validation_best | 80% | 0.802 | 0.6292 | 0.4536 | 97 | 0.0309 |
| word_char_logreg_validation_best | 70% | 0.702 | 0.6488 | 0.4459 | 74 | 0.0000 |
| word_char_logreg_validation_best | 50% | 0.501 | 0.6967 | 0.4760 | 53 | 0.0000 |
| word_linsvm_validation_best | 100% | 1.000 | 0.5915 | 0.3809 | 120 | 0.0000 |
| word_linsvm_validation_best | 90% | 0.902 | 0.6037 | 0.3841 | 111 | 0.0000 |
| word_linsvm_validation_best | 80% | 0.802 | 0.6188 | 0.3813 | 101 | 0.0000 |
| word_linsvm_validation_best | 70% | 0.702 | 0.6286 | 0.3805 | 93 | 0.0000 |
| word_linsvm_validation_best | 50% | 0.501 | 0.6767 | 0.3910 | 73 | 0.0000 |
| word_logreg_validation_best | 100% | 1.000 | 0.5973 | 0.3988 | 120 | 0.0000 |
| word_logreg_validation_best | 90% | 0.902 | 0.6111 | 0.4031 | 110 | 0.0000 |
| word_logreg_validation_best | 80% | 0.802 | 0.6146 | 0.3989 | 98 | 0.0000 |
| word_logreg_validation_best | 70% | 0.702 | 0.6298 | 0.4026 | 89 | 0.0000 |
| word_logreg_validation_best | 50% | 0.501 | 0.6717 | 0.4189 | 72 | 0.0000 |

The additional same-fold exploratory variants cover unigram and 1–3 word n-grams, a 10k feature cap with min_df=2, and character 2–5 grams with a 15k cap/min_df=1. They were not selected using the folds; per-fold results remain descriptive.

## Language subsets

Per-language grouped out-of-fold counts and metrics are in `metrics.json`. French has only 35 distinct reference records, repeated across folds; interpret those estimates cautiously.

## Reproduction

`python scripts/benchmark_stance_existing_resources_v1.py --out-dir reports/stance_existing_resources_benchmark_v3`

Exact record/fold assignments are in `reports/stance_candidate_shared_cv_v2/shared_fold_manifest.jsonl`. Fold predictions and margins are in `grouped_cv_predictions.jsonl`; their SHA-256 is recorded in `metrics.json`.
