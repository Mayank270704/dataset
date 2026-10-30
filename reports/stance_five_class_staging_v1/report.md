# Five-class stance synthetic staging evaluation

This is a staging-only synthetic-data experiment. Original records and the production artifact remain unchanged. Synthetic data is marked training-only and never appears in an evaluation fold. Scores measure agreement with previously exposed, unverified repository reference labels.

Original reference counts are Support 140, Neutral 219, Oppose 1, Mixed 0, Unclear 40. The saved shared grouped folds contain only the 399 Support/Neutral/Unclear records. Thus original evaluation support is zero for Oppose and Mixed; their real-world precision/recall/generalization cannot be established by this dataset.

Both conditions use the same saved 3x5 source/duplicate-grouped folds and context-question-comment feature constructor. Each fold fits all vectorizers on training text only. The baseline sees original fold training rows only. The augmented model appends synthetic Oppose/Mixed/Unclear samples to each training split only; count and C are selected using grouped validation on original rows. The test labels are never used for selection.

## Overall metrics

Macro-F1 below is the requested five-class macro with zero-support Oppose/Mixed included as 0; observed-class macro-F1 is also shown to make three-class comparison interpretable.

| Model | Repeated OOF rows | Accuracy | Five-class macro-F1 | Observed-class macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 1197 | 0.6525 | 0.3048 | 0.5080 | 0.6368 |
| synthetic_augmented | 1197 | 0.6483 | 0.2889 | 0.4814 | 0.6276 |

## baseline: per-class precision, recall, F1 and support

Support is repeated OOF reference count, not synthetic training count.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Support | 0.5906 | 0.5667 | 0.5784 | 420 |
| Neutral | 0.7156 | 0.8006 | 0.7557 | 657 |
| Oppose | 0.0000 | 0.0000 | 0.0000 | 0 |
| Mixed | 0.0000 | 0.0000 | 0.0000 | 0 |
| Unclear | 0.2881 | 0.1417 | 0.1899 | 120 |

Confusion matrix (rows reference, columns predicted; order Support, Neutral, Oppose, Mixed, Unclear): `[[238, 156, 0, 0, 26], [115, 526, 0, 0, 16], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [50, 53, 0, 0, 17]]`.

## synthetic_augmented: per-class precision, recall, F1 and support

Support is repeated OOF reference count, not synthetic training count.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Support | 0.5893 | 0.5500 | 0.5690 | 420 |
| Neutral | 0.7095 | 0.8143 | 0.7583 | 657 |
| Oppose | 0.0000 | 0.0000 | 0.0000 | 0 |
| Mixed | 0.0000 | 0.0000 | 0.0000 | 0 |
| Unclear | 0.1961 | 0.0833 | 0.1170 | 120 |

Confusion matrix (rows reference, columns predicted; order Support, Neutral, Oppose, Mixed, Unclear): `[[231, 164, 0, 0, 25], [106, 535, 0, 0, 16], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [55, 55, 0, 0, 10]]`.

Final training configuration selected in grouped validation: C=3.0, synthetic count/class=40 for Oppose/Mixed/Unclear.

Candidate artifact SHA-256 `e772daeab6b4fa69f53ef56dd6ba3e57320b9afe58c1ce933c09551a6d0ce322`. Training manifest SHA-256 `d8acd7d259c5a8876333ed38bf0e17c97eb73562fedc801014f27d2b34a8d484`. Production artifact SHA-256 `5078100cca85e7cd6410a61c7d5738d6c28c51f78d5b2fed43ce08f79a735843`.

No 90% accuracy claim is supported. Oppose and Mixed need independent original examples before their generalization can be measured. Synthetic training agreement is not evidence of real-world accuracy.

Reproduce without overwriting the saved run: `python scripts/generate_stance_five_class_synthetic_v1.py --out-dir reports/stance_five_class_synthetic_reproduction` then `python scripts/train_stance_five_class_staging_v1.py --synthetic-dir reports/stance_five_class_synthetic_reproduction --out-dir reports/stance_five_class_staging_reproduction --candidate-dir models/candidates/stance_context_wordchar_5class_synth_v1_reproduction`.
