# Stance-training pipeline: PowerShell usage guide

Supervised **stance** baselines (TF-IDF + Logistic Regression, TF-IDF + Linear SVM) for the Canadian Open Government
consultation comments. Stance means position toward the Open Government initiative or policy, as defined in
`reports/annotation_guidelines_v1.1.md`. It is not sentiment and not topic; neither is used as a label or feature.

**Human-annotation training path: BLOCKED.** No completed annotator files exist in `data/annotation/completed/`, so the
human-only path has not been fitted. A separate saved baseline does use the repository stance reference labels; see
section 9 and the model registry. `--check-only` currently prints:

```text
[MISSING] pilot: completed annotator file(s) not found: ['pilot_annotator_A_completed.csv', 'pilot_annotator_B_completed.csv']
[MISSING] expansion_v2: completed annotator file(s) not found: ['expansion_v2_annotator_A_completed.csv', 'expansion_v2_annotator_B_completed.csv']
[BLOCKED] No genuine stance labels: both annotators' completed files are required (data/annotation/completed/). Pipeline not fitted; no model saved.
```

Draft files in `data/reviewer_handoff/ai_assisted_drafts_review_only/` are not human annotations and must never be
placed in `data/annotation/completed/`. They may only be loaded by the legacy reference-label mode in section 9.

Run every command from the project root with the system Python 3.13 (the project `.venv` is empty).

## 1. Files the pipeline reads

| Input | Path | Notes |
|---|---|---|
| Reference pools | `data/annotation/pilot_annotation_50.csv`, `data/annotation/expansion_pool_400_v2.csv` | text and source ids; never modified |
| Completed annotation, pilot | `data/annotation/completed/pilot_annotator_A_completed.csv`, `..._B_completed.csv` | both required |
| Completed annotation, v2 | `data/annotation/completed/expansion_v2_annotator_A_completed.csv`, `..._B_completed.csv` | both required |
| Human adjudication (optional) | `data/annotation/adjudicated/pilot_adjudicated.csv`, `expansion_v2_adjudicated.csv` | columns `pilot_id`, `adjudicated_stance`, `adjudication_rationale` |
| Corpus near-duplicate groups (optional) | `data/full_corpus/run_v1/corpus/corpus_entries.csv`, `data/full_corpus/concern_discovery_v1/duplicates/near_duplicate_groups.csv` | used for leakage grouping when present |

Completed files are validated with the same rules as `scripts/analyze_annotation_agreement.py`: unchanged columns, row ids
and order; annotator ids `annotator_A` / `annotator_B`; verbatim evidence spans. In addition, any non-blank stance outside
`Support`, `Oppose`, `Mixed`, `Neutral`, `Unclear` is an error.

## 2. Which records become training labels

| Situation | Result |
|---|---|
| Human `adjudicated_stance` with a rationale | gold label, source `adjudicated` |
| Both annotators chose the same stance | gold label, source `agreement` |
| Same stance, but a note says `non-English` (guidelines §7) | excluded: `language_inability` |
| Annotators disagree, no adjudication | excluded: `unresolved_disagreement` |
| Either annotation blank | excluded: `missing_annotation` |
| `pilot_*` rows inside the v2 file | excluded: labelled in the pilot round |

Training needs at least 40 usable gold records in at least two classes, each class with at least 5 records. Rarer classes
are dropped and reported. A model is fitted only if the grouped held-out set has at least 10 records, at least two classes,
and no test class missing from training.

## 3. Commands

Check readiness. This never fits a model and never writes a file:

```powershell
python scripts\train_stance_baselines.py --check-only
```

Train, evaluate and save. Nothing is written unless both models were fitted and evaluated:

```powershell
python scripts\train_stance_baselines.py                          # both rounds, default settings
python scripts\train_stance_baselines.py --rounds pilot --tag pilot_only
python scripts\train_stance_baselines.py --exclude-labels Unclear --tag no_unclear
python scripts\train_stance_baselines.py --config my_stance_config.json --tag custom
```

| Option | Default | Meaning |
|---|---|---|
| `--rounds` | `pilot expansion_v2` | rounds to read |
| `--tag` | UTC timestamp | run folder `data\stance_training\run_<tag>` (must not exist) |
| `--output-root` | `data\stance_training` | parent folder for runs |
| `--test-size` | 0.25 | held-out share (first fold of a stratified group split) |
| `--seed` | 42 | split, bootstrap and model seed |
| `--min-gold` | 40 | minimum usable gold records |
| `--min-class-count` | 5 | classes with fewer records are dropped and reported |
| `--exclude-labels` | none | schema labels to leave out, e.g. `Unclear` |
| `--config` | none | JSON overriding keys of `DEFAULT_CONFIG` in `models/stance_classification/pipeline.py` |

Fixed model settings, never tuned on the test set: TF-IDF word 1-2 grams, at most 20,000 features, sublinear tf, accents
kept. Logistic Regression uses C = 1, lbfgs and balanced class weights. LinearSVC uses C = 1 and balanced class weights.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | trained and saved, or `--check-only` found the labels sufficient |
| 1 | invalid annotation or adjudication file; nothing fitted or written |
| 2 | the run folder already exists |
| 3 | blocked: labels missing or insufficient for a valid held-out evaluation; nothing fitted or written |

## 4. Producing the adjudication file

After both annotators' files exist, run the agreement analysis. It writes a disagreement table with blank adjudication
columns. A human copies it, fills `adjudicated_stance` and `adjudication_rationale` for the rows they resolve, and saves
it under the name the pipeline reads:

```powershell
python scripts\analyze_annotation_agreement.py --round pilot
New-Item -ItemType Directory -Force data\annotation\adjudicated | Out-Null
Copy-Item data\annotation\agreement_analysis\pilot_v1\disagreements.csv data\annotation\adjudicated\pilot_adjudicated.csv
# a human edits data\annotation\adjudicated\pilot_adjudicated.csv; then
python scripts\train_stance_baselines.py --check-only
```

Rows left blank stay excluded. Adjudication is never automated.

## 5. Leakage control and evaluation

* **Groups.** Records sharing a source record, an exact-duplicate group, a corpus near-duplicate group, a template, or
  near-identical wording (character 5-gram Jaccard ≥ 0.5 or containment ≥ 0.9) form one group. A group is never split
  across train and test.
* **Fitting.** TF-IDF is fitted inside each scikit-learn `Pipeline` on the training split only.
* **Reported metrics.** Macro-F1 on the held-out set with a 95% group-bootstrap interval, plus per-class precision, recall,
  F1 and support. Also a confusion matrix, train/test/all class distributions, a majority-class baseline, and grouped
  cross-validation within the training split.
* **Interpretation.** The pools are stratified with fixed quotas, so results describe these samples, not population accuracy.

## 6. Outputs of a successful run

`data\stance_training\run_<tag>\` is write-once and contains:

| File | Content |
|---|---|
| `pipeline_logreg.joblib`, `pipeline_linsvm.joblib` | fitted Pipelines |
| `label_mapping.json` | labels used, index, full schema, construct definition |
| `preprocessing_config.json` | the exact configuration |
| `metrics.json` | all metrics above, plus dropped classes |
| `split_assignments.csv` | every record with `pilot_id`, round, source ids, gold label, label source, leakage group, split |
| `metadata.json` | status `trained_and_evaluated`, SHA-256 of every artifact, input-file hashes, versions |
| `report.md` | readable summary |

Load a saved model. The hash is re-checked before unpickling, so only load runs you created:

```powershell
python -c "from pathlib import Path; from models.stance_classification.pipeline import load_trained_pipeline as L; m = L(Path('data/stance_training/run_<tag>'), 'logreg'); print(m.predict(['example comment']))"
```

## 7. Integration with the research evaluation

```powershell
python scripts\run_research_evaluation.py --no-write
```

Section 5 reports `supervised stance baselines (TF-IDF + LR/SVM)` as:

| Status | When |
|---|---|
| BLOCKED | no completed files, or too few labels; this is the current state |
| FAILED | annotation files are invalid, or a saved run's artifacts no longer match their hashes |
| PASSED | a saved run is intact; its held-out macro-F1 and the majority baseline are shown |

## 8. Tests

```powershell
python -m pytest tests\test_stance_training.py tests\test_stance_classification.py tests\test_stance_ai_working_labels.py -q -p no:cacheprovider
```

The tests cover missing labels, invalid labels and adjudication, gold rules, duplicate leakage, reproducible splits,
fit-on-train-only TF-IDF, evaluation output and serialization with tamper refusal. `test_stance_ai_working_labels.py`
covers the AI-working-label mode of section 9 and checks that the human path, thresholds and research-evaluation item
are unchanged. The tests use synthetic fixture labels in temporary folders only; no test writes into `data\annotation`.

## 9. Existing project reference-label mode

This legacy input mode can train the same baselines from the repository's existing stance reference-label files. The
reference values are authoritative targets for project model development. Source history is preserved in
`data/ai_labels/stance_sentiment_ai_v1/provenance.json`; use of the project reference set does not imply independent
human verification or external-source performance.

```powershell
# readiness only: never fits, never writes
python scripts\train_stance_baselines.py --ai-working-labels data\reviewer_handoff\ai_assisted_drafts_review_only\person_1\policylens_annotator_1_AI_assisted_draft.csv --check-only

# fit, evaluate and save against the reference set (the run folder must not exist yet)
python scripts\train_stance_baselines.py --ai-working-labels data\reviewer_handoff\ai_assisted_drafts_review_only\person_1\policylens_annotator_1_AI_assisted_draft.csv --tag <new_tag>
```

**Input compatibility**

* The original single-draft pilot input remains accepted for compatibility. The current 400-record reference set is the
  default for the reproducible three-class comparison.
* Reference stance values and `pilot_id` / source-record mappings are loaded unchanged.
* Files inside `data\annotation\completed\` or `data\annotation\adjudicated\` are refused (exit 1). The mode never reads
  the human annotation files.
* The human-annotation path remains separate; this mode never writes or populates completed annotation files.

**Training protocol.** The pipeline retains grouped splitting, validation-based C selection, per-class metrics and the
three-class output schema Support / Neutral / Unclear. Oppose and Mixed are not introduced by this path.

**Where results go.** Existing run artifacts remain in `data\stance_training_ai_working_labels\run_<tag>\`; this
directory name is retained for compatibility. The files are those of section 6, with the following source-role fields:

| Field | Value in this mode |
|---|---|
| `metadata.json` → `label_role` | `repository_reference_labels` |
| `metadata.json` → `human_labels_only` | `false` (legacy source-role metadata) |
| `metadata.json` → `label_type` | retained original source-provenance value for artifact compatibility |
| `split_assignments.csv` | `reference_stance`, source identifiers and original label-source mapping |
| `report.md` | repository reference-label terminology and a source-history note |

`load_trained_pipeline` (section 6) also loads these runs. Preserve source-role metadata when loading or comparing them.

**Research evaluation.** The section-7 item counts only runs in `data\stance_training\`; project reference-label runs
remain a separate development track.

**Existing run.** `data\stance_training_ai_working_labels\run_pilot_primary_draft_v1\` used the person-1 draft (50 records;
37 train, 13 held out) and is described in its `report.md`. With 13 held-out records, of which 2 are `Unclear`, its
confidence intervals are wide. Grouped cross-validation inside the training split was much lower than the held-out
score, so read the held-out figures as unstable.
