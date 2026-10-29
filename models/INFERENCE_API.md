# Consultation inference API

The API exposes the existing TF-IDF baselines through `POST /api/inference/consultation`. It does not train or
write artifacts. Models load lazily on the first inference request for each model name and remain cached in the
API process. Only the four small scikit-learn pipelines below are loaded; no transformer or embedding model is
loaded.

## Request and response

```json
{"model":"linsvm","texts":["How do we increase participation across government?"]}
```

`model` is `linsvm` (default) or `logreg`; `texts` contains 1–16 strings, each at most 5,000 characters. The
response includes `stance_suggestion`, `sentiment_suggestion`, input status, and the model's decision margin for
each task. A margin is not a probability or calibrated confidence. The response marks `human_validated: false`
and includes the AI-label provenance and caveats. Empty or shorter-than-three-character comments receive no
prediction. Submitted text is processed in memory and is not persisted by this route; request logs omit queries
and bodies.

## Exact active model artifacts

All paths are relative to the repository root. The verified inference implementation is
`models/new_comment_inference.py`, called by the read-only API in `app/api.py`.
Its Python interface is `NewCommentPredictor(model="linsvm"|"logreg").predict(texts)`; the existing CLI is
`scripts/run_new_comment_inference_v1.py` (`--input`, `--text-column`, `--out`, `--model`). The API wraps that
predictor; it does not train models.

| Task / model | Saved model | Artifact SHA-256 | Saved evaluation |
|---|---|---|---|
| Stance / Logistic Regression | `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_logreg.joblib` | `fd618afd8684d942dc88ade814231e8b9f5605bf989881446307ec67f9fbd24c` | `.../metrics.json` |
| Stance / Linear SVM | `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib` | `5078100cca85e7cd6410a61c7d5738d6c28c51f78d5b2fed43ce08f79a735843` | `.../metrics.json` |
| Consultation sentiment / Logistic Regression | `data/sentiment_training/consultation_ai_labels_tfidf_v1/pipeline_logreg.joblib` | `c2f62b4ce2ca24e7605e2f785229c820ff6df41ac53cfe7bc9bf71cfa0bb353b` | `.../metrics.json` |
| Consultation sentiment / Linear SVM | `data/sentiment_training/consultation_ai_labels_tfidf_v1/pipeline_linsvm.joblib` | `212c9161b02caeef4d28a1bd3459374e67df1c07793a815da0f74f877de52f6c` | `.../metrics.json` |

Before `joblib.load`, inference verifies every artifact checksum and training-input checksum in the run's
`metadata.json`, completed-run status, declared model, AI-label status, label-map/metrics consistency, caveat,
and saved TF-IDF/model configuration. Training inputs are:

* Stance: `data/ai_labels/stance_sentiment_ai_v1/ai_working_labels_pool400.csv` and
  `data/ai_labels/stance_sentiment_ai_v1/provenance.json`.
* Sentiment: `data/annotation/expansion_pool_400_v2.csv` and
  `data/ai_labels/stance_sentiment_ai_v1/ai_sentiment_labels_pool400.csv`.

The 400-record working labels are AI-generated and not human-verified. The pilot 50 stance labels reuse an
AI-assisted draft; expansion stance and sentiment labels were produced in a single AI pass without human review.
The provenance record is `data/ai_labels/stance_sentiment_ai_v1/provenance.json`.

## Training configuration and saved results

Both runs use lowercase word TF-IDF, unigram/bigram features, `min_df=1`, at most 20,000 features, sublinear TF,
and token pattern `(?u)\\b\\w\\w+\\b`. They use grouped leakage-aware train/validation/test splits, validation
macro-F1 for C selection, refit on train+validation, one held-out test evaluation, and split-seed sensitivity.
See each run's `preprocessing_config.json` (stance) or `config.json` (sentiment), `metrics.json`,
`label_mapping.json`, and `metadata.json` for the complete settings and checksums.

| Task | Labels used | Split sizes (train / validation / test) | Model | Selected C | Held-out macro-F1 | Held-out accuracy |
|---|---|---:|---|---:|---:|---:|
| Stance | Support, Neutral, Unclear | 239 / 80 / 80 | Logistic Regression | 3.0 | 0.3641 | 0.5250 |
| Stance | Support, Neutral, Unclear | 239 / 80 / 80 | Linear SVM | 0.3 | 0.3958 | 0.5750 |
| Sentiment (tone) | Negative, Neutral, Positive | 232 / 78 / 78 | Logistic Regression | 1.0 | 0.5699 | 0.7051 |
| Sentiment (tone) | Negative, Neutral, Positive | 232 / 78 / 78 | Linear SVM | 1.0 | 0.5926 | 0.7564 |

These are agreement scores against unverified AI working labels, not human-validated model accuracy. Stance
`Oppose` had one labeled record and was dropped; sentiment `Mixed` (12 records) was excluded by design. The active
API uses the 3-class stance run; older pilot and support-vs-neutral model files are not selected.
Other saved stance files are `data/stance_training_ai_working_labels/run_pilot_primary_draft_v1/pipeline_logreg.joblib`,
`data/stance_training_ai_working_labels/run_pilot_primary_draft_v1/pipeline_linsvm.joblib`,
`data/stance_training_ai_working_labels/run_pool400_ai_v1_support_vs_neutral/pipeline_logreg.joblib`, and
`data/stance_training_ai_working_labels/run_pool400_ai_v1_support_vs_neutral/pipeline_linsvm.joblib`. The pilot
run uses 50 AI-draft records; support-vs-neutral omits Unclear. Neither is selected by the API.

## Known failure modes and unavailable features

The saved smoke test in `data/inference/new_comment_inference_v1/` runs the existing predictor on 300 held-out
corpus entries plus edge cases. Its manifest checksums cover the predictions and summary; `summary.json` explicitly
states that accuracy was not measured. One saved adversarial-looking edge case, “I strongly oppose publishing any
more government data; it wastes money,” receives a `Support` stance suggestion. The model cannot output `Oppose`
at all. Very short text gets no prediction; long, noisy, French, code-mixed, or out-of-domain inputs can receive
plausible-looking but unsupported labels. Sentiment predicts tone, not policy stance.

The API does not assign concern groups or priority to new comments. The saved audit found nearest-centroid
assignment failed its pre-set acceptance threshold, and no fitted UMAP/HDBSCAN assignment model is saved.
Collective-demand generation is also not implemented: the candidate ledger is not a generated or reviewed demand.
