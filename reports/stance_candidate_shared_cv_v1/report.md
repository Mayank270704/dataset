# Shared-fold production and stance-candidate comparison

All results use the same repeated source/duplicate-grouped folds. Every model is re-fit on each fold's training originals; synthetic additions never enter its validation fold. The pre-trained v1 candidate is not directly scored because it was fit on the complete reference pool. The reported v1 and v2 candidate recipes are fold-local re-trains.

Eligible records 399; class counts `{"Support": 140, "Neutral": 219, "Unclear": 40}`; groups 398; 3 repeats x 5 folds. The existing reference pool has prior development exposure, so these are not independent source-held-out estimates.

| Recipe | Accuracy mean ± SD | Macro-F1 mean ± SD | Support P/R/F1 mean ± SD | Neutral P/R/F1 mean ± SD | Unclear P/R/F1 mean ± SD |
|---|---:|---:|---|---|---|
| production_word_only | 0.5898 ± 0.0434 | 0.4036 ± 0.0303 | 0.5313 ± 0.0641 / 0.5452 ± 0.0640 / 0.5359 ± 0.0540 (n=28.0000 ± 0.0000) | 0.6326 ± 0.0353 / 0.7263 ± 0.0748 / 0.6750 ± 0.0479 (n=43.8000 ± 0.4140) | 0.0000 ± 0.0000 / 0.0000 ± 0.0000 / 0.0000 ± 0.0000 (n=8.0000 ± 0.0000) |
| word_char_original_only | 0.5982 ± 0.0401 | 0.4292 ± 0.0425 | 0.5451 ± 0.0605 / 0.5595 ± 0.0583 / 0.5504 ± 0.0487 (n=28.0000 ± 0.0000) | 0.6448 ± 0.0339 / 0.7261 ± 0.0602 / 0.6823 ± 0.0409 (n=43.8000 ± 0.4140) | 0.1667 ± 0.3086 / 0.0333 ± 0.0572 / 0.0548 ± 0.0942 (n=8.0000 ± 0.0000) |
| candidate_v1_recipe | 0.6191 ± 0.0389 | 0.4677 ± 0.0410 | 0.5701 ± 0.0728 / 0.5881 ± 0.0700 / 0.5759 ± 0.0566 (n=28.0000 ± 0.0000) | 0.6673 ± 0.0356 / 0.7367 ± 0.0602 / 0.6989 ± 0.0355 (n=43.8000 ± 0.4140) | 0.3333 ± 0.3318 / 0.0833 ± 0.0610 / 0.1282 ± 0.0952 (n=8.0000 ± 0.0000) |
| candidate_v2_recipe | 0.6115 ± 0.0484 | 0.4596 ± 0.0466 | 0.5746 ± 0.0962 / 0.5857 ± 0.0554 / 0.5774 ± 0.0636 (n=28.0000 ± 0.0000) | 0.6532 ± 0.0336 / 0.7260 ± 0.0721 / 0.6866 ± 0.0455 (n=43.8000 ± 0.4140) | 0.3278 ± 0.3838 / 0.0750 ± 0.0634 / 0.1148 ± 0.0992 (n=8.0000 ± 0.0000) |

## Paired fold differences

Positive values favor the first named recipe. Fold wins count macro-F1 wins across the identical 15 validation folds; differences are descriptive, not confidence intervals.

| Recipe | Comparator | Accuracy delta mean | Macro-F1 delta mean ± SD | Wins / ties / losses |
|---|---|---:|---:|---:|
| word_char_original_only | production_word_only | +0.0083 | +0.0256 ± 0.0443 | 11 / 0 / 4 |
| word_char_original_only | candidate_v1_recipe | -0.0209 | -0.0385 ± 0.0448 | 4 / 0 / 11 |
| candidate_v1_recipe | production_word_only | +0.0292 | +0.0641 ± 0.0466 | 13 / 0 / 2 |
| candidate_v2_recipe | production_word_only | +0.0216 | +0.0560 ± 0.0505 | 12 / 0 / 3 |
| candidate_v2_recipe | candidate_v1_recipe | -0.0076 | -0.0081 ± 0.0236 | 4 / 1 / 10 |

## Confusion matrices

Counts are pooled repeated out-of-fold predictions; each original record contributes once per repeat (so pooled N is three times the eligible record count). Rows are reference labels; columns are predicted labels in Support, Neutral, Unclear order.

### production_word_only

`[[229, 189, 2], [174, 477, 6], [32, 88, 0]]` (pooled N=1197)

### word_char_original_only

`[[235, 179, 6], [167, 477, 13], [32, 84, 4]]` (pooled N=1197)

### candidate_v1_recipe

`[[247, 164, 9], [160, 484, 13], [31, 79, 10]]` (pooled N=1197)

### candidate_v2_recipe

`[[246, 170, 4], [161, 477, 19], [28, 83, 9]]` (pooled N=1197)

## Language subsets

Metrics are means across folds containing each language. Sample sizes count all repeated out-of-fold appearances; they are not unique records.

- **production_word_only:** English n=1092, accuracy 0.5899 ± 0.0478, macro-F1 0.4050 ± 0.0326; French n=105, accuracy 0.5958 ± 0.1953, macro-F1 0.2664 ± 0.0804
- **word_char_original_only:** English n=1092, accuracy 0.6007 ± 0.0384, macro-F1 0.4332 ± 0.0423; French n=105, accuracy 0.5608 ± 0.1896, macro-F1 0.2748 ± 0.0966
- **candidate_v1_recipe:** English n=1092, accuracy 0.6190 ± 0.0422, macro-F1 0.4734 ± 0.0446; French n=105, accuracy 0.6208 ± 0.1885, macro-F1 0.3435 ± 0.1243
- **candidate_v2_recipe:** English n=1092, accuracy 0.6096 ± 0.0500, macro-F1 0.4647 ± 0.0514; French n=105, accuracy 0.6275 ± 0.1895, macro-F1 0.3326 ± 0.1137

## False positives and false negatives

Each row includes an unchanged reference record ID, original text, and the error's repeat count. The machine-readable file contains the same examples.

### production_word_only

- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_004` (source `consultation_compilation_2017_18` / `46`): “Digital divides differentiates contributions to amplify the already empowered.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_004` (source `consultation_compilation_2017_18` / `46`): “Digital divides differentiates contributions to amplify the already empowered.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, Support, error in 3/3 repeats; `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 3/3 repeats; `expansion_v2_265` (source `get_involved_questionnaire_2017_18` / `7392`): “Cadre du gouvernement fédéral à la retraite. Présentement avec un emploi occasionnel à INFC.”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 2/3 repeats; `expansion_v2_196` (source `get_involved_questionnaire_2017_18` / `2787`): “Open by Default - DWS”
- false_positive for Unclear: ref=Support, predicted=Unclear, error in 1/1 repeats; `expansion_v2_256` (source `get_involved_questionnaire_2017_18` / `6652`): “Making government actions open by default rather than closed by default.”

### candidate_v1_recipe

- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_031` (source `consultation_compilation_2017_18` / `875`): “letting third party agencies run departments for profit. not responsible to follow government rules. need monitoring.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_negative for Neutral: ref=Neutral, predicted=Support, Unclear, error in 3/3 repeats; `expansion_v2_030` (source `consultation_compilation_2017_18` / `866`): “La conférence « Open dialogue on Open government » est en cours. @IAP2StLaurent #GouvertCan Un gouvernement ouvert ... comment cela résonne pour vous? Peut-on faire mieux qu’à l’heure actuelle? https://t.co/vISzeWOK5x”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, Support, error in 3/3 repeats; `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”
- false_positive for Unclear: ref=Support, predicted=Unclear, error in 3/3 repeats; `expansion_v2_105` (source `consultation_compilation_2017_18` / `3677`): “@mpacielloif u guys wanted to get in with canadian gov, this is a good way - The call for proposals on the #OpenbyDefault accessibility challenge is live! Check it out now: https://t.co/dfAhtFZX6i #OpenGovCan https://t.co/Hda0aMFAme”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 3/3 repeats; `expansion_v2_301` (source `individual_comments_2016_18` / `47`): “Conditioned by actual needs - we shouldn't be doing this just because we can, or because it's cheaper (it's not).”
- false_positive for Unclear: ref=Support, predicted=Unclear, error in 3/3 repeats; `expansion_v2_341` (source `individual_comments_2016_18` / `128`): “Please support CODE 2016.”

### candidate_v2_recipe

- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_negative for Support: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_positive for Support: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_015` (source `consultation_compilation_2017_18` / `263`): “Internal efforts to improve services for public still come up against restrictions and red tape that slow change - discourages cultural change”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_005` (source `consultation_compilation_2017_18` / `54`): “Taxi net was just retired at CRA where they burned data on CD and moved it around by taxi. We need to ensure we are keeping our IT modern enough to meet business requirements.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_014` (source `consultation_compilation_2017_18` / `252`): “FYI. SCC is not available for 2 weeks to troubleshoot a client issue impacting the distribution of weather information. I guess ensuring Canadians have accurate weather information is not a priority for Shared Services. ECCC has people working 24x7 to generate accurate information, but shared services cannot make sure it is distributed correctly? Seems a disconnect here.”
- false_negative for Neutral: ref=Neutral, predicted=Support, error in 3/3 repeats; `expansion_v2_015` (source `consultation_compilation_2017_18` / `263`): “Internal efforts to improve services for public still come up against restrictions and red tape that slow change - discourages cultural change”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_001` (source `consultation_compilation_2017_18` / `30`): “Yes!!! to advisory council”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_002` (source `consultation_compilation_2017_18` / `41`): “Types of available formats are changing. Technology is changing and adapting. It is difficult to work through the mindsets within government to find support. Digital changes are held back because of a culture resiting change or unwilllingness to share.”
- false_positive for Neutral: ref=Support, predicted=Neutral, error in 3/3 repeats; `expansion_v2_019` (source `consultation_compilation_2017_18` / `484`): “Good example: Using github for tracking”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, Support, error in 3/3 repeats; `expansion_v2_010` (source `consultation_compilation_2017_18` / `153`): “Endorsement of the statement IRADA, a Pakistani civil society & rights group, endorses the statement.”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_025` (source `consultation_compilation_2017_18` / `740`): “capacity to document answers”
- false_negative for Unclear: ref=Unclear, predicted=Neutral, error in 3/3 repeats; `expansion_v2_028` (source `consultation_compilation_2017_18` / `791`): “Majority of people don't have knoweldge or perspective to know what to do.”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 3/3 repeats; `expansion_v2_225` (source `get_involved_questionnaire_2017_18` / `4816`): “We hear an announcement...........then later we hear the truth.”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 3/3 repeats; `expansion_v2_301` (source `individual_comments_2016_18` / `47`): “Conditioned by actual needs - we shouldn't be doing this just because we can, or because it's cheaper (it's not).”
- false_positive for Unclear: ref=Neutral, predicted=Unclear, error in 2/2 repeats; `expansion_v2_016` (source `consultation_compilation_2017_18` / `293`): “phone lines always busy”

## Synthetic template review

V4 defects are listed in the v2 corpus report; the conflicting support/rejection templates and language-comprehension examples are absent from v2 model-training rows. The v2 corpus contains only insufficient-context, fragment and off-topic Unclear cases. Language protocol rows have no `reference_stance`, have `model_eligible=false`, and are excluded from folds.
Exact and near duplicate counts against all original records: v1 0 / 0; v2 0 / 0. v2 is exact-text deduplicated during generation. Model learning transfer is assessed on original fold texts, which do not share synthetic template IDs; the small Unclear class and prior pool exposure still limit interpretation.

## Deployment decision

The production artifact hash remained `5078100cca85e7cd6410a61c7d5738d6c28c51f78d5b2fed43ce08f79a735843`. The comparison does not modify production or either existing candidate. If v2 does not consistently improve on v1 and production, retain the existing v1 candidate for traceability, do not create/promote v2, and acquire new source-held-out labels before reconsidering.

Reproduce: `python scripts/compare_stance_candidates_grouped_cv_v2.py --out-dir reports/stance_candidate_shared_cv_rerun --seed 20261029 --folds 5 --repeats 3 --per-class 120`
