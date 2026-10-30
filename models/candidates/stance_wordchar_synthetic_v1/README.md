# Stance word+character synthetic candidate v1

This versioned artifact is a staging candidate only. It is not wired into API or UI inference. The production artifact and its SHA-256 are recorded in `candidate_metadata.json`.

Training used all eligible original reference records plus a 0.5 synthetic/original ratio, with the v4 validation-selected word+character LinearSVC configuration (C=1.0, balanced). The grouped CV staging gate supports keeping this candidate for further review, but the pool has prior development exposure and several Unclear synthetic templates carry label risk. Do not promote without reviewing those templates and obtaining source-held-out reference evaluation.

Rollback: no production change occurred; leave the current route pointing to `data/stance_training_ai_working_labels/run_pool400_ai_v1_3class/pipeline_linsvm.joblib`.
