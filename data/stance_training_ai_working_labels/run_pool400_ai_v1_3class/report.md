# Stance baselines on AI-generated working labels (400-record v2 pool, UNVERIFIED)

**Labels are AI-generated working labels, not human annotations. Metrics measure agreement with these AI labels on held-out records; they are not human-verified stance accuracy, gold-standard performance or inter-annotator agreement, and do not support generalization claims.**

Label source: data/ai_labels/stance_sentiment_ai_v1 (pilot 50 = pre-existing AI draft 1; expansion 350 = new single-pass AI labels under guidelines v1.1). Stance toward Open Government; not sentiment.

Labels used: ['Support', 'Neutral', 'Unclear']. Dropped: {"below_min_class_count": {"Oppose": 1}}.
Split (seed 42): train 239 / validation 80 / test 80 records; 398 leakage groups; no group or normalised text crosses a boundary.

Class distribution: {"train": {"Support": 84, "Neutral": 131, "Unclear": 24}, "validation": {"Support": 28, "Neutral": 44, "Unclear": 8}, "test": {"Support": 28, "Neutral": 44, "Unclear": 8}, "all": {"Support": 140, "Neutral": 219, "Unclear": 40}}

| Model | Selected C (validation) | Test macro-F1 | 95% CI (group bootstrap) | Accuracy |
|---|---:|---:|---|---:|
| majority class (Neutral) | - | 0.2366 | [0.2034, 0.2646] | 0.55 |
| stratified random (1000 draws) | - | 0.3268 | [0.2279, 0.4308] (draw spread) | - |
| logreg | 3.0 | 0.3641 | [0.287, 0.4391] | 0.525 |
| linsvm | 0.3 | 0.3958 | [0.3181, 0.4714] | 0.575 |

Validation macro-F1 by C: {"logreg": {"0.1": 0.3696, "0.3": 0.3847, "1.0": 0.3725, "3.0": 0.3859, "10.0": 0.3753}, "linsvm": {"0.1": 0.3626, "0.3": 0.4017, "1.0": 0.3645, "3.0": 0.3753, "10.0": 0.3753}}

## logreg: per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Support | 0.4571 | 0.5714 | 0.5079 | 28 |
| Neutral | 0.5778 | 0.5909 | 0.5843 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |

Confusion matrix (rows true, cols predicted; Support, Neutral, Unclear): [[16, 12, 0], [18, 26, 0], [1, 7, 0]]
Predicted distribution on test: {"Neutral": 45, "Support": 35}; fit on train+validation macro-F1 1.0

## linsvm: per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Support | 0.5161 | 0.5714 | 0.5424 | 28 |
| Neutral | 0.6122 | 0.6818 | 0.6452 | 44 |
| Unclear | 0.0 | 0.0 | 0.0 | 8 |

Confusion matrix (rows true, cols predicted; Support, Neutral, Unclear): [[16, 12, 0], [14, 30, 0], [1, 7, 0]]
Predicted distribution on test: {"Neutral": 49, "Support": 31}; fit on train+validation macro-F1 1.0

## Split sensitivity (secondary; same protocol, other split seeds)

| Seed | logreg | linsvm | majority |
|---|---:|---:|---:|
| 42 | 0.3641 | 0.3958 | 0.2366 |
| 43 | 0.3646 | 0.3568 | 0.2366 |
| 44 | 0.4024 | 0.3738 | 0.2366 |
| 45 | 0.3954 | 0.4102 | 0.2366 |
| 46 | 0.3844 | 0.3506 | 0.2366 |

Mean ± sd over all seeds: {"logreg": {"mean": 0.3822, "sd": 0.0156}, "linsvm": {"mean": 0.3774, "sd": 0.0227}, "majority": {"mean": 0.2366, "sd": 0.0}}

Test sets are small; intervals are wide. Results describe this stratified 400-record sample, not the corpus.
