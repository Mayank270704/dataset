# PolicyLens research evaluation

Generated: 2026-10-12T18:34:23+00:00

Summary: BLOCKED=4, NOT RUN=3, PASSED=27

## Interpretation rules

- Cluster assignments, silhouette and noise ratios are unsupervised descriptors and are NOT human-validated concerns.
- Sentiment (tone), stance (position toward Open Government) and concern (issue raised) are distinct; none is inferred from another.
- Sentiment corpora (Sentiment140, SentiMix, HinEmo-9, PolicyLens sentiment) are auxiliary and are not ground truth for consultation stance.
- The annotation pools are stratified with fixed quotas; they do not estimate population frequencies without design weights.
- Identical-text or duplicate-group rules prove identical text only, never that two comments came from the same respondent.


## 1. Protected artifact integrity

- **PASSED** — protected files unchanged: 49 recorded files match the baseline

## 2. Expansion pool v2 verification

- **PASSED** — artifacts_exist: pool, ledger, manifest and pilot present
- **PASSED** — manifest_output_hashes: pool and ledger SHA-256 match the manifest
- **PASSED** — manifest_input_hashes: pilot and source_clean inputs unchanged since the build
- **PASSED** — row_counts: pool=400, ledger=11031, manifest pool=400, ledger=11031
- **PASSED** — pilot_rows_verbatim_and_first: first 50 rows equal the pilot file; 2 appended columns
- **PASSED** — segments_labelled_by_id: rows 1-50 are pilot_*, remaining are expansion_v2_*
- **PASSED** — annotation_columns_blank: non-blank annotation cells: 0
- **PASSED** — source_record_uniqueness: repeated (source_dataset, source_row_number): 0
- **PASSED** — no_pilot_record_overlap: new rows sharing a pilot source record: 0
- **PASSED** — exact_text_duplicate_policy: repeated normalised texts in pool: 0
- **PASSED** — duplicate_group_unique: non-blank duplicate_text_group_id appears at most once (an identical-text key, not a respondent ID)
- **PASSED** — source_text_provenance_verbatim: rows not matching source cell (trimmed): 0
- **PASSED** — eligibility_rules: rows violating length/filler/exclusion rules: 0
- **PASSED** — ledger_completeness: ledger=11031 vs non-blank designed cells=11031; selected=350
- **PASSED** — language_metadata: values=['English', 'French']; questionnaire basis states survey language: True
- **PASSED** — language_scope_exclusions_ledgered: out-of-scope entries ledgered: 3
- **PASSED** — dataset_quotas: achieved={'consultation_compilation_2017_18': 160, 'get_involved_questionnaire_2017_18': 120, 'individual_comments_2016_18': 70}; quotas={'individual_comments_2016_18': 70, 'consultation_compilation_2017_18': 160, 'get_involved_questionnaire_2017_18': 120}
- **PASSED** — stratum_allocation_matches_manifest: individual_comments_2016_18: ok; get_involved_questionnaire_2017_18: ok; compilation themes: ok
- **PASSED** — allocation_shortfall_events: no shortfall/redistribution events
- **PASSED** — annotator_templates_match_pool: 2/2 templates present; each equals the pool (header, rows, blank labels)

## 3. Existing 450-comment baseline (UNSUPERVISED, not human-validated)

- **PASSED** — artifacts match the 44-pair review fingerprints: 6 embedding/cluster/report fingerprints match
- **PASSED** — stored comparison: all_minilm_l6_v2: clusters=2, noise_fraction=0.0, silhouette=0.5956 (geometric descriptor; not concern validity)
- **PASSED** — stored comparison: paraphrase_multilingual_minilm_l12_v2: clusters=6, noise_fraction=0.0867, silhouette=0.2942 (geometric descriptor; not concern validity)
- **PASSED** — interpretation: Cluster count, silhouette, and language mixing are descriptive and do not establish policy-concern coherence. Review representative source comments.

## 4. Annotation readiness and human-dependent analyses

- **PASSED** — pilot templates present and blank: non-blank annotation cells: 0
- **PASSED** — v2 templates present and blank: non-blank annotation cells: 0
- **BLOCKED** — pilot agreement analysis: awaiting human annotation: completed annotator_A/B files not present
- **BLOCKED** — expansion_v2 agreement analysis: awaiting human annotation: completed annotator_A/B files not present
- **BLOCKED** — 44-pair concern-coherence review: awaiting two independent bilingual reviewers (human_review/completed/ missing)

## 5. Supervised models and sentiment/stance separation

- **BLOCKED** — supervised stance baselines (TF-IDF + LR/SVM): no adjudicated human stance labels exist; models.stance_classification.validate_labeled_data refuses unlabeled data
- **NOT RUN** — concern-priority scoring: intentionally not implemented before concern validity is human-assessed
- **NOT RUN** — full-corpus (13,541) embedding and clustering: not executed here; requires a separate, resource-checked decision after the 44-pair review
- **NOT RUN** — external sentiment baselines: auxiliary only; polarity is not stance toward Open Government
