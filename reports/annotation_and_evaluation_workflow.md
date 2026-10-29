# PolicyLens annotation and evaluation workflow

Exact commands, run from the project root. Use the system Python 3.13 that has pandas/numpy/pytest (the project `.venv` is empty). Every command that writes refuses to overwrite.

## 0. One-time setup (already done once; refuses to repeat)

```powershell
python scripts\create_expansion_v2_annotator_templates.py   # writes expansion_v2_annotator_1.csv / _2.csv (blank, 400 rows)
python scripts\run_research_evaluation.py --init-baseline   # records SHA-256 of protected artifacts
```

## 1. Annotation copies (for each annotator)

PowerShell:

```powershell
New-Item -ItemType Directory -Force data\annotation\working | Out-Null
New-Item -ItemType Directory -Force data\annotation\completed | Out-Null

# Pilot (50 rows)
Copy-Item data\annotation\annotator_1.csv data\annotation\working\pilot_annotator_A_working.csv
Copy-Item data\annotation\annotator_2.csv data\annotation\working\pilot_annotator_B_working.csv

# v2 expansion (400 rows: 50 pilot + 350 additional)
Copy-Item data\annotation\expansion_v2_annotator_1.csv data\annotation\working\expansion_v2_annotator_A_working.csv
Copy-Item data\annotation\expansion_v2_annotator_2.csv data\annotation\working\expansion_v2_annotator_B_working.csv
```

Git Bash equivalent: `mkdir -p data/annotation/working data/annotation/completed` and `cp <template> <working copy>`.

When finished, save each annotator's file (CSV UTF-8, columns and rows unchanged) as:

| Round | Annotator | Completed file |
|---|---|---|
| pilot | A | `data/annotation/completed/pilot_annotator_A_completed.csv` |
| pilot | B | `data/annotation/completed/pilot_annotator_B_completed.csv` |
| v2 | A | `data/annotation/completed/expansion_v2_annotator_A_completed.csv` |
| v2 | B | `data/annotation/completed/expansion_v2_annotator_B_completed.csv` |

Mapping: `annotator_1.*` templates are for `annotator_A`, `annotator_2.*` for `annotator_B`.

## 2. Agreement analysis (after both completed files exist)

```powershell
# Dry check: validate and print a summary, write nothing
python scripts\analyze_annotation_agreement.py --round pilot --check-only
python scripts\analyze_annotation_agreement.py --round expansion_v2 --check-only

# Write the analysis to a NEW directory (refuses an existing one)
python scripts\analyze_annotation_agreement.py --round pilot
python scripts\analyze_annotation_agreement.py --round expansion_v2
# custom location:  --output-dir data\annotation\agreement_analysis\expansion_v2_rerun2
```

Outputs per round (`data/annotation/agreement_analysis/<round>_v1/`): `agreement_report.md`, `agreement_metrics.json`, `disagreements.csv` (blank adjudication columns), `validation_issues.csv`.

- The pilot notebook `notebooks/03_annotation_agreement_analysis.ipynb` continues to work on the same completed pilot files.
- Exit codes: 0 ok or pending, 1 invalid files, 2 completed files missing.
- The tool does not compute agreement on blank files and does not resolve disagreements.

## 3. Reading the results correctly

- **Pilot vs expansion:** the pilot (50) and the v2 expansion (350) are separate analyses. In the v2 file the first 50 `pilot_*` rows are an optional, separately reported "pilot_segment"; they are never part of the 350 and never pooled.
- **Stratified samples:** quotas are fixed per stratum. Agreement describes annotator reliability on these records; it does not estimate population prevalence. See `reports/expansion_pool_v2_methodology_notes.md` for design weights.
- **Stance is not sentiment.** Concern and requested action are free text; only descriptive overlap rates are reported.
- **Duplicate groups prove identical text, not identical respondents.**

## 4. Consolidated, non-destructive evaluation

```powershell
python scripts\verify_expansion_pool_v2.py                    # 20 data-quality checks on the v2 pool (exit 1 on failure)
python scripts\run_research_evaluation.py                     # all checks; writes reports\research_evaluation\run_<UTC>\
python scripts\run_research_evaluation.py --no-write          # print only
python scripts\run_research_evaluation.py --tag manual_check  # named report folder (must not exist)
```

Each item is `PASSED`, `FAILED`, `WARN`, `BLOCKED` (needs human annotation) or `NOT RUN` (deliberately). It does not re-embed, re-cluster, train, or create labels.

## 5. Tests

```powershell
python -m pytest tests -q -p no:cacheprovider
```

## 6. Human-dependent steps that remain

1. Two independent bilingual reviewers complete the 44-pair concern-coherence templates; then run `python scripts\analyze_concern_review_agreement.py`.
2. Two annotators complete and return the pilot files; calibrate guidelines (kappa target 0.60).
3. Adjudicate pilot disagreements (never automated).
4. Two annotators complete the 350 additional v2 records.
5. Only then: supervised stance baselines, and any decision on full-corpus embedding or concern prioritisation.

## 7. Coordinated reviewer handoff (all three human-evaluation tracks)

Stance annotation, the 44-pair review and the comment-group review (`data/full_corpus/concern_discovery_v1/human_review/`) are handed off together through
`data/reviewer_handoff/REVIEWER_HANDOFF_GUIDE.md` and stage folders built by `python scripts\human_review_handoff.py build-packets --stage N`.
Returned files are checked and stored with `python scripts\human_review_handoff.py accept ...`; `status` is read-only. The order of work, exact file
names and analysis commands are in `reports/concern_discovery_report_v1.md`, section 11. Schemas and the analysis commands above are unchanged.
