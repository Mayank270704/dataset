# Synthetic stance deployment-readiness and grouped CV audit

This is a source/duplicate-grouped robustness audit on the existing reference pool. The pool was exposed during prior development; these are not independent or unseen-source performance estimates.

Eligible original records: 399; reference class counts: `{"Support": 140, "Neutral": 219, "Unclear": 40}`. Groups: 398. Repeated CV: 3 repeats x 5 folds (15 validation folds). Every original row is validated once per repeat, and no leakage group crosses a fold.

V4 selected model reproducibility: ratio ratio_0.5 with C=1.0, class_weight=balanced; validation macro-F1 0.4992. Synthetic v4 rows reproduced exactly: 240. No test metric was used to reconstruct this selection.

Synthetic additions are restricted to each fold's training side. The model configuration is frozen to the v4 selected word+character LinearSVC (C=1.0, balanced) for a controlled augmentation comparison. The current production recipe (word-only LinearSVC C=0.3, balanced) is also included.

| Condition | Accuracy mean ± SD | Macro-F1 mean ± SD | Support P/R/F1 (mean ± SD) | Neutral P/R/F1 (mean ± SD) | Unclear P/R/F1 (mean ± SD) | Mean validation-fold support S/N/U |
|---|---:|---:|---|---|---|---|
| production_recipe_word_only | 0.5898 ± 0.0434 | 0.4036 ± 0.0303 | 0.5313 ± 0.0641 / 0.5452 ± 0.0640 / 0.5359 ± 0.0540 | 0.6326 ± 0.0353 / 0.7263 ± 0.0748 / 0.6750 ± 0.0479 | 0.0000 ± 0.0000 / 0.0000 ± 0.0000 / 0.0000 ± 0.0000 | 28.0 ± 0.0 / 43.8 ± 0.4 / 8.0 ± 0.0 |
| word_char_original_only | 0.5982 ± 0.0401 | 0.4292 ± 0.0425 | 0.5451 ± 0.0605 / 0.5595 ± 0.0583 / 0.5504 ± 0.0487 | 0.6448 ± 0.0339 / 0.7261 ± 0.0602 / 0.6823 ± 0.0409 | 0.1667 ± 0.3086 / 0.0333 ± 0.0572 / 0.0548 ± 0.0942 | 28.0 ± 0.0 / 43.8 ± 0.4 / 8.0 ± 0.0 |
| ratio_0.25 | 0.6141 ± 0.0394 | 0.4688 ± 0.0328 | 0.5649 ± 0.0747 / 0.5929 ± 0.0724 / 0.5757 ± 0.0591 | 0.6629 ± 0.0397 / 0.7231 ± 0.0628 / 0.6904 ± 0.0410 | 0.3555 ± 0.3188 / 0.0917 ± 0.0572 / 0.1403 ± 0.0891 | 28.0 ± 0.0 / 43.8 ± 0.4 / 8.0 ± 0.0 |
| ratio_0.5 | 0.6199 ± 0.0383 | 0.4679 ± 0.0389 | 0.5710 ± 0.0726 / 0.5881 ± 0.0725 / 0.5770 ± 0.0610 | 0.6667 ± 0.0351 / 0.7383 ± 0.0541 / 0.6995 ± 0.0335 | 0.3278 ± 0.3325 / 0.0833 ± 0.0610 / 0.1272 ± 0.0947 | 28.0 ± 0.0 / 43.8 ± 0.4 / 8.0 ± 0.0 |
| ratio_1 | 0.6174 ± 0.0402 | 0.4660 ± 0.0400 | 0.5689 ± 0.0755 / 0.5833 ± 0.0684 / 0.5735 ± 0.0601 | 0.6641 ± 0.0333 / 0.7367 ± 0.0601 / 0.6973 ± 0.0361 | 0.3278 ± 0.3325 / 0.0833 ± 0.0610 / 0.1272 ± 0.0947 | 28.0 ± 0.0 / 43.8 ± 0.4 / 8.0 ± 0.0 |

## Per-fold class support

Support values above are mean ± standard deviation of each validation-fold class count. Total unique original reference counts are Support 140, Neutral 219, Unclear 40; each appears once per repeat.

## Template diversity and label consistency

Unique generated texts in the v4 corpus are counted by class below. Template-level notes are in `template_audit.md`.

| Class | Rows | Unique texts | Template IDs | Rows flagged for review |
|---|---:|---:|---:|---:|
| Neutral | 80 | 49 | 12 | 0 |
| Support | 80 | 57 | 12 | 0 |
| Unclear | 80 | 35 | 20 | 28 |

## Representative CV errors

Examples below come from original validation-fold predictions and preserve the source reference labels and IDs. The separate `representative_errors.jsonl` contains the same examples in machine-readable form.

### production_recipe_word_only

- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_004` (source `consultation_compilation_2017_18` / `46`): “Digital divides differentiates contributions to amplify the already empowered.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- Unclear reference; predicted Neutral, Support, Support; errors in 3/3 repeats; record `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”

### word_char_original_only

- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_030` (source `consultation_compilation_2017_18` / `866`): “La conférence « Open dialogue on Open government » est en cours. @IAP2StLaurent #GouvertCan Un gouvernement ouvert ... comment cela résonne pour vous? Peut-on faire mieux qu’à l’heure actuelle? https://t.co/vISzeWOK5x”
- Unclear reference; predicted Neutral, Support, Support; errors in 3/3 repeats; record `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”

### ratio_0.5

- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- Support reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- Neutral reference; predicted Support, Support, Support; errors in 3/3 repeats; record `expansion_v2_015` (source `consultation_compilation_2017_18` / `263`): “Internal efforts to improve services for public still come up against restrictions and red tape that slow change - discourages cultural change”
- Unclear reference; predicted Neutral, Support, Support; errors in 3/3 repeats; record `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- Unclear reference; predicted Neutral, Neutral, Neutral; errors in 3/3 repeats; record `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”

## Deployment decision

Paired comparison for ratio 0.5 vs production: mean macro-F1 difference 0.0643 with 14/15 fold wins; vs word+character original-only: 0.0387 with 12/15 fold wins. Candidate gate for further staging: True. Fold differences are descriptive and are not independent confidence intervals.

Production artifact SHA-256 unchanged: `5078100cca85e7cd6410a61c7d5738d6c28c51f78d5b2fed43ce08f79a735843`. The audit did not modify it. CV supports retaining a versioned staging candidate for follow-up review, while the flagged synthetic label risks and lack of new-source evaluation block replacing the production artifact. Rollback is the current unchanged route to `data\stance_training_ai_working_labels\run_pool400_ai_v1_3class/pipeline_linsvm.joblib`; no routing/config change is part of this audit.

This dataset is small, uses existing reference labels previously exposed in model development, and has only 40 Unclear records (about eight per fold). The language-comprehension subcase is not text-identifiable. New source-held-out reference data and review of the flagged synthetic ambiguity templates remain necessary before a deployment change.

Reproduce: `python scripts/audit_stance_synthetic_deployment_v1.py --out-dir reports/stance_synthetic_deployment_audit_rerun --seed 20261029 --folds 5 --repeats 3 --synthetic-per-class 120`
