# Expansion pool v2: methodology notes and limitations

Companion to `reports/expansion_pool_v2_report.md` (generated, not edited). Evidence: `data/annotation/expansion_pool_400_v2_manifest.json` and `data/annotation/expansion_pool_400_v2_ledger.csv`.

## 1. The `duplicate_text_group_id` rule

**What the field is.** It was assigned during the data audit (`scripts/create_dataset_audit_notebook.py`, lines ~608-628): for each source row, the text of the designed response columns is joined, normalised, and identical joined texts share a group ID. It is therefore a **row-level key for identical response text**.

**What it is not.**
- It is not a respondent identifier. The sources contain no respondent ID.
- Two different people can write the same short statement ("access to information").
- One person can submit different texts in several records.
- A matching ID does not show that two comments came from the same person, campaign or channel.

**Artefacts.** In the questionnaire, `dup_000001` covers 4,097 rows whose six target text fields are all empty; it reflects blank rows and is irrelevant to the selected fields.

**How v2 uses it, and why.**
- At most one entry per non-blank group ID is eligible, and at most one entry per normalised exact-text key (policy A). This avoids annotating the same words repeatedly and stops identical text appearing on both sides of a later train/test split.
- Retention is by a deterministic rule (lowest dataset name, numeric row, field order). It is arbitrary with respect to meaning, not a judgement about which respondent is "real".
- Every non-retained entry is in the ledger with reason `duplicate_text_key_not_retained` (12 compilation, 52 questionnaire) or `duplicate_group_not_retained` (6 questionnaire), plus the retained entry's identity, so multiplicity can be recovered.

**Consequence.** Frequently repeated statements are represented once, so unweighted counts in the pool understate how common they are in the source. Use the ledger to reconstruct multiplicities if frequency matters.

## 2. Compilation theme quota rule

**Rule (implemented in `allocate_quotas`).** 160 records across 15 strata:
1. Equal share = 160 // 15 = 10 per stratum.
2. "Not Applicable" and "Blank/unspecified" are **capped** at the equal share.
3. The remainder (10) goes one each to the 10 largest named themes by eligible count.

Result: 11 records in each of Open Dialogue, Open Data, Education, Governance & Resourcing, User-Centric Thinking, Financial Transparency, Open Information, Culture of Openness, Open Science and Other; 10 each in Service, Reconciliation & Data Sovereignty, Social Innovation, Not Applicable and Blank/unspecified.

**Rationale.**
- Coverage: with proportional allocation, "Not Applicable" (28% of eligible records) and the two largest themes would crowd out small themes (Social Innovation has only 36 eligible records).
- "Not Applicable" and blank carry little thematic information, so they are capped rather than dropped.
- Every stratum could be filled; no shortfall or redistribution occurred.

**Design table (compilation only).** Each compilation record has exactly one text entry, so these fractions are exact. They exclude the 20 compilation records already in the pilot, which were drawn by a different procedure.

| Theme stratum | Eligible records | Share | Quota | Inclusion fraction | Records represented per selected |
|---|---:|---:|---:|---:|---:|
| Not Applicable | 1,344 | 28.0% | 10 | 0.007 | 134.4 |
| Open Dialogue | 969 | 20.2% | 11 | 0.011 | 88.1 |
| Open Data | 775 | 16.2% | 11 | 0.014 | 70.5 |
| Education | 227 | 4.7% | 11 | 0.048 | 20.6 |
| Governance & Resourcing | 226 | 4.7% | 11 | 0.049 | 20.5 |
| User-Centric Thinking | 224 | 4.7% | 11 | 0.049 | 20.4 |
| Financial Transparency | 223 | 4.6% | 11 | 0.049 | 20.3 |
| Open Information | 212 | 4.4% | 11 | 0.052 | 19.3 |
| Culture of Openness | 163 | 3.4% | 11 | 0.067 | 14.8 |
| Open Science | 106 | 2.2% | 11 | 0.104 | 9.6 |
| Other | 101 | 2.1% | 11 | 0.109 | 9.2 |
| Service | 98 | 2.0% | 10 | 0.102 | 9.8 |
| Reconciliation & Data Sovereignty | 80 | 1.7% | 10 | 0.125 | 8.0 |
| Social Innovation | 36 | 0.8% | 10 | 0.278 | 3.6 |
| Blank/unspecified | 12 | 0.3% | 10 | 0.833 | 1.2 |

**Limitations.**
- **Not representative without weights.** Inclusion fractions range from 0.007 to 0.83, so a pooled, unweighted percentage from this sample over-represents small themes by more than 100x relative to "Not Applicable" (Blank/unspecified vs Not Applicable). Any prevalence estimate needs design weights (records represented per selected) and still applies only to the deduplicated eligible compilation records.
- **Themes are administrative tags.** `theme_th_me` was assigned by the consultation organisers. It is not a validated concern category, and annotators see it in `context`, which may prime their judgement. It must not be used as a model feature for stance or concern.
- **Tie-handling.** The remainder rule is deterministic but arbitrary: moving a unit from one named theme to another would be equally defensible.
- **Individual and questionnaire** strata (field) are equal splits (14 and 40). Their records can have several eligible fields, so entry-level availability overstates record availability and inclusion fractions there are approximate.
- **Pooled counts across datasets** (70/160/120) are fixed by design; they do not reflect each dataset's size or response volume.

## 3. What the pool supports

- Annotator-reliability estimates (kappa) and guideline calibration on varied, deduplicated records.
- A labelled set for later supervised experiments, with group-aware splitting on the text key.

## 4. What it does not support

- Population prevalence of any stance, concern or theme without weights.
- Claims about distinct respondents or about public opinion.
- Any inference from sentiment datasets to consultation stance.
