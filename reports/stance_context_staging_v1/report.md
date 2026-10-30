# Context-aware stance staging prototype

This is a separate staging-only model and route. The production artifact and response contract remain unchanged. Results are agreement with previously exposed, unverified repository reference labels, not external or human-validated accuracy.

The context model uses question wording, question context, and comment text. It excludes labels, provenance, source IDs, reviewer notes/decisions, sentiment, and language metadata. All 399 rows and the exact saved 3x5 source/duplicate-grouped folds were used. Each outer training split independently selected C through three-fold grouped validation; all vectorization was fitted within training partitions. The production comparator refits the deployed word-only recipe (word 1-2 gram TF-IDF, C=0.3, balanced) inside each outer training fold. The actual deployed artifact was trained on this reference pool and is not evaluated directly on its own training rows.

| Model | OOF rows (repeated) | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Weighted-F1 mean +/- SD |
|---|---:|---:|---:|---:|
| context_candidate | 1197 | 0.6525 +/- 0.0507 | 0.5032 +/- 0.0618 | 0.6345 +/- 0.0516 |
| production | 1197 | 0.5898 +/- 0.0434 | 0.4036 +/- 0.0303 | 0.5585 +/- 0.0404 |

## context_candidate: per-class mean precision / recall / F1

| Class | Precision | Recall | F1 | Mean support per fold |
|---|---:|---:|---:|---:|
| Support | 0.5932 | 0.5619 | 0.5744 | 28.0 |
| Neutral | 0.7170 | 0.8038 | 0.7561 | 43.8 |
| Unclear | 0.2668 | 0.1417 | 0.1791 | 8.0 |

Pooled repeated out-of-fold confusion matrix (rows reference, columns predicted; order Support, Neutral, Unclear): `[[236, 157, 27], [114, 528, 15], [50, 53, 17]]`.

## production: per-class mean precision / recall / F1

| Class | Precision | Recall | F1 | Mean support per fold |
|---|---:|---:|---:|---:|
| Support | 0.5313 | 0.5452 | 0.5359 | 28.0 |
| Neutral | 0.6326 | 0.7263 | 0.6750 | 43.8 |
| Unclear | 0.0000 | 0.0000 | 0.0000 | 8.0 |

Pooled repeated out-of-fold confusion matrix (rows reference, columns predicted; order Support, Neutral, Unclear): `[[229, 189, 2], [174, 477, 6], [32, 88, 0]]`.

## Unclear behavior

Context candidate: 17 correct Unclear predictions from 120 reference Unclear instances across repeats. Production: 0 correct from 120. The full row shows errors into Support and Neutral; no threshold or class-balance forcing was used.

Final artifact C=1.0 was selected using five-fold grouped validation on the full available training pool (macro-F1, then accuracy, then lower C). The evaluation metrics above remain from the untouched shared outer folds.

Artifact: `models/candidates/stance_context_wordchar_v1/pipeline_context_wordchar.artifact` (SHA-256 `b85b25bee4dd11154e97953b77d3ab25cd96a65ae60fffe523faf8d900d0601b`). Training rows manifest SHA-256 `29e244e18352057688723dd2600b35b3c8281520684cbe86000cc2b828ed9eac`. Production artifact SHA-256 `5078100cca85e7cd6410a61c7d5738d6c28c51f78d5b2fed43ce08f79a735843`.

Demo route: `POST /api/inference/stance/context-staging`. Rollback: disable the staging route/UI panel; production endpoints and artifact were not switched.

Reproduce: `python scripts/build_stance_context_staging_v1.py`
