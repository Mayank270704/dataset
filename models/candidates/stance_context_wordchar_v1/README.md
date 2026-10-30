# Context aware stance candidate v1

This candidate is for a controlled staging demo only. It is not the production artifact and does not alter either existing production route. The API route is `POST /api/inference/stance/context-staging`.

Feature construction is centralized in `models/context_stance_features.py` and shared by training and inference. Inputs are question wording, optional question context, and comment text. Missing question/context fields become explicit sentinel tokens. Comments with fewer than three non-space characters receive no prediction. Language metadata is ignored. Only Support, Neutral, and Unclear are emitted.

The exact artifact checksum, training data checksum, row manifest checksum, saved fold checksum, configuration, class counts, selected C validation results, and the production artifact checksum are in `candidate_manifest.json`. The artifact is `pipeline_context_wordchar.artifact`; the training row manifest is `training_rows.jsonl`.

Reproduce the candidate and grouped-fold comparison with:

```powershell
python scripts/build_stance_context_staging_v1.py
```

Rollback: no production switch occurred. Stop using the staging endpoint or remove its dedicated UI panel. Keep the production route loading `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib`.
