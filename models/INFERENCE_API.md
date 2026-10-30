# Consultation inference API

## Stance endpoint correction (2026-10-29)

`POST /api/inference/stance` and v2 routing return the direct output of the saved three-class stance model: Support, Neutral or Unclear. Sentiment does not determine stance. Oppose and Mixed are not classes in the saved artifact. Compatibility response fields for sentiment remain present as null; sentiment is not run for this endpoint. The UI shows the provisional stance and brief reference-set caveat, not raw margins or internal model diagnostics. See `models/stance_inference_v2.py` and `reports/stance_reference_model_comparison_v2/report.md`.

## Separate context staging prototype

`POST /api/inference/stance/context-staging` is a separate staging-only endpoint. It accepts `{ "items": [{ "question": "...", "context": "...", "comment": "...", "language": "..." }] }` with 1-16 items, subject to the existing 128 KB request-body limit. Question is optional (maximum 2,000 characters), context is optional (maximum 5,000), comment is required (maximum 5,000), and comments shorter than 3 non-space characters return `too_short_or_empty` without a prediction. Missing question/context values use `[NO_QUESTION]` and `[NO_CONTEXT]` during feature construction. Language metadata is accepted up to 100 characters, including unsupported values, and ignored by the model.

The route only accepts `items` at the request root, and each item only accepts question, context, comment, and language. Labels, provenance, reviewer fields, IDs, and sentiment are rejected. Outputs are Support, Neutral, or Unclear. This route does not change `/api/inference/stance` or `/api/inference/consultation`; its versioned artifact is `models/candidates/stance_context_wordchar_v1/`. See its `candidate_manifest.json` and `README.md` for hashes and rollback instructions.

The active route still loads the word-only C=0.3 LinearSVC run at `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/`. The word+character synthetic candidate in `models/candidates/stance_wordchar_synthetic_v1/` is not connected to inference. `reports/stance_candidate_shared_cv_v2/report.md` documents the latest same-fold candidate comparison and development-only limitations.

## Five-class synthetic staging endpoint

`POST /api/inference/stance/five-class-staging` is isolated from both production routes and the existing three-class context stage. Request shape: `{ "items": [{ "question": "...", "context": "...", "comment": "..." }] }`. It accepts 1-16 items; question/context are optional, comments are required, question/context/comment are limited to 2,000/5,000/5,000 characters. Whitespace comments shorter than three characters return a no-prediction status. The only allowed item fields are question, context, and comment; target labels, provenance, sentiment, and reviewer metadata are rejected.

The model uses the shared training feature order `[CONTEXT] ... [QUESTION] ... [COMMENT] ...`, including `[NO_CONTEXT]` and `[NO_QUESTION]` markers. Its supported output schema is Support, Neutral, Oppose, Mixed, and Unclear. The versioned artifact is `models/candidates/stance_context_wordchar_5class_synth_v1/`; its manifest verifies artifact checksum, feature contract, and label classes. The training provenance and reproducible grouped evaluation are in `models/candidates/stance_context_wordchar_5class_synth_v1/training_manifest.jsonl` and `reports/stance_five_class_staging_v1/`.

The original 3x5 grouped evaluation has 399 records (Support 140, Neutral 219, Unclear 40), and excludes the sole Oppose record because the shared folds were defined for those three classes. There are no original Mixed examples. Thus evaluation support is zero for Oppose and Mixed and the synthetic candidate is not ready for production. Synthetic examples are training-only and do not count as validation evidence. To rebuild, run `python scripts/generate_stance_five_class_synthetic_v1.py` followed by `python scripts/train_stance_five_class_staging_v1.py`. Rollback is simply to stop calling this staging route; the production artifacts and routes are unchanged.

Current experiment metrics are development estimates because this reference-label pool has appeared in prior model-development splits. A clean external evaluation requires new source groups kept out of all prior development.

The API exposes the existing TF-IDF baselines through `POST /api/inference/consultation`. It does not train or
write artifacts. Models load lazily on the first inference request for each model name and remain cached in the
API process. Only the four small scikit-learn pipelines below are loaded; no transformer or embedding model is
loaded.

## Request and response

```json
{"model":"linsvm","texts":["How do we increase participation across government?"]}
```

`model` is `linsvm` (default) or `logreg`; `texts` contains 1–16 strings, each at most 5,000 characters. The
response includes `stance_suggestion`, `sentiment_suggestion`, input status, and legacy decision-margin fields. A
margin is not a calibrated confidence score. The response marks `human_validated: false` and identifies the
repository reference-label role. Empty or shorter-than-three-character comments receive no
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
`metadata.json`, completed-run status, declared model, reference-label role, label-map/metrics consistency, caveat,
and saved TF-IDF/model configuration. Training inputs are:

* Stance: `data/ai_labels/stance_sentiment_ai_v1/ai_working_labels_pool400.csv` and
  `data/ai_labels/stance_sentiment_ai_v1/provenance.json`.
* Sentiment: `data/annotation/expansion_pool_400_v2.csv` and
  `data/ai_labels/stance_sentiment_ai_v1/ai_sentiment_labels_pool400.csv`.

The stance labels are the repository's project reference targets. Their source history, including how the pilot and
expansion labels were assembled, remains in `data/ai_labels/stance_sentiment_ai_v1/provenance.json`; preserving this
terminology does not assert independent human verification. The separate sentiment model has its own label source.

## Training configuration and saved results

Both runs use lowercase word TF-IDF, unigram/bigram features, `min_df=1`, at most 20,000 features, sublinear TF,
and token pattern `(?u)\\b\\w\\w+\\b`. They use grouped leakage-aware train/validation/test splits, validation
macro-F1 for C selection, refit on train+validation, one held-out test evaluation, and split-seed sensitivity.
See each run's `preprocessing_config.json` (stance) or `config.json` (sentiment), `metrics.json`,
`label_mapping.json`, and `metadata.json` for the complete settings and checksums.

| Task | Labels used | Split sizes (train / validation / development / test) | Model | Selected C | Development macro-F1 | Development accuracy |
|---|---|---:|---|---:|---:|---:|
| Stance | Support, Neutral, Unclear | 239 / 80 / 80 | Logistic Regression | 3.0 | 0.3641 | 0.5250 |
| Stance | Support, Neutral, Unclear | 239 / 80 / 80 | Linear SVM | 0.3 | 0.3958 | 0.5750 |
| Sentiment (tone) | Negative, Neutral, Positive | 232 / 78 / 78 | Logistic Regression | 1.0 | 0.5699 | 0.7051 |
| Sentiment (tone) | Negative, Neutral, Positive | 232 / 78 / 78 | Linear SVM | 1.0 | 0.5926 | 0.7564 |

These are agreement scores against the repository reference labels on the historic seed-42 development split; they
are not external-source performance estimates. Stance `Oppose` had one reference record and `Mixed` had zero; neither
is part of the active three-class model. The active
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
