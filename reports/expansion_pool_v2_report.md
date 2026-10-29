# PolicyLens expansion pool v2 report

Status: **annotation pool prepared; no human labels exist**. All annotation columns are blank.

## Files

- Pool: `data/annotation/expansion_pool_400_v2.csv` (400 rows = 50 pilot + 350 new)
- Ledger: `data/annotation/expansion_pool_400_v2_ledger.csv` (11031 entries)
- Manifest: `data/annotation/expansion_pool_400_v2_manifest.json` (seed, quotas, SHA-256 of inputs/outputs)
- The original `data/annotation/expansion_pool_400.csv` and its annotator copies are unchanged and superseded by this pool.

## Allocation

| Dataset | Quota | Achieved | Eligible records | Eligible entries |
|---|---:|---:|---:|---:|
| `individual_comments_2016_18` | 70 | 70 | 126 | 375 |
| `consultation_compilation_2017_18` | 160 | 160 | 4796 | 4796 |
| `get_involved_questionnaire_2017_18` | 120 | 120 | 3386 | 4757 |

Within-dataset strata (quota = equal split; compilation caps "Not Applicable" and "Blank/unspecified" at the base share, and gives the remainder to the largest named themes):

**`consultation_compilation_2017_18`**

Equal share per stratum: 10; remainder distributed to non-capped strata: 10.

| Stratum | Eligible entries | Initial quota | Final quota |
|---|---:|---:|---:|
| Blank/unspecified | 12 | 10 | 10 |
| Culture of Openness | 163 | 11 | 11 |
| Education | 227 | 11 | 11 |
| Financial Transparency | 223 | 11 | 11 |
| Governance & Resourcing | 226 | 11 | 11 |
| Not Applicable | 1344 | 10 | 10 |
| Open Data | 775 | 11 | 11 |
| Open Dialogue | 969 | 11 | 11 |
| Open Information | 212 | 11 | 11 |
| Open Science | 106 | 11 | 11 |
| Other | 101 | 11 | 11 |
| Reconciliation & Data Sovereignty | 80 | 10 | 10 |
| Service | 98 | 10 | 10 |
| Social Innovation | 36 | 10 | 10 |
| User-Centric Thinking | 224 | 11 | 11 |

No shortfall; no redistribution needed.

**`get_involved_questionnaire_2017_18`**

Equal share per stratum: 40; remainder distributed to non-capped strata: 0.

| Stratum | Eligible entries | Initial quota | Final quota |
|---|---:|---:|---:|
| q14 | 2109 | 40 | 40 |
| q18 | 614 | 40 | 40 |
| q8 | 2034 | 40 | 40 |

No shortfall; no redistribution needed.

**`individual_comments_2016_18`**

Equal share per stratum: 14; remainder distributed to non-capped strata: 0.

| Stratum | Eligible entries | Initial quota | Final quota |
|---|---:|---:|---:|
| improvements | 63 | 14 | 14 |
| og_is_important_because | 101 | 14 | 14 |
| other_comments | 18 | 14 | 14 |
| practical_idea | 107 | 14 | 14 |
| public_engagement_question | 86 | 14 | 14 |

No shortfall; no redistribution needed.

## Exclusion ledger summary (entries by reason)

| Dataset | Reason | Entries |
|---|---|---:|
| `consultation_compilation_2017_18` | duplicate_text_key_not_retained | 12 |
| `consultation_compilation_2017_18` | eligible_not_drawn | 4636 |
| `consultation_compilation_2017_18` | generic_filler | 1 |
| `consultation_compilation_2017_18` | language_out_of_scope | 3 |
| `consultation_compilation_2017_18` | pilot_item | 20 |
| `consultation_compilation_2017_18` | pilot_text_overlap | 1 |
| `consultation_compilation_2017_18` | selected | 160 |
| `consultation_compilation_2017_18` | too_short | 335 |
| `get_involved_questionnaire_2017_18` | duplicate_group_not_retained | 6 |
| `get_involved_questionnaire_2017_18` | duplicate_text_key_not_retained | 52 |
| `get_involved_questionnaire_2017_18` | eligible_not_drawn | 4571 |
| `get_involved_questionnaire_2017_18` | generic_filler | 28 |
| `get_involved_questionnaire_2017_18` | pilot_item | 10 |
| `get_involved_questionnaire_2017_18` | pilot_record_overlap | 7 |
| `get_involved_questionnaire_2017_18` | same_record_other_field_selected | 66 |
| `get_involved_questionnaire_2017_18` | selected | 120 |
| `get_involved_questionnaire_2017_18` | too_short | 527 |
| `individual_comments_2016_18` | eligible_not_drawn | 167 |
| `individual_comments_2016_18` | pilot_item | 20 |
| `individual_comments_2016_18` | pilot_record_overlap | 49 |
| `individual_comments_2016_18` | same_record_other_field_selected | 138 |
| `individual_comments_2016_18` | selected | 70 |
| `individual_comments_2016_18` | too_short | 32 |

Blank cells in the designed fields are not ledgered row by row; their counts are in the manifest (`stats.blank_cells_not_in_ledger`).

## Language

`source_language` comes from source metadata only; text language is never inferred, and unrecognised or blank values are `Unknown`. `language_basis` records the source column.

| Dataset | New-row language counts |
|---|---|
| `consultation_compilation_2017_18` | {"English": 156, "French": 4} |
| `get_involved_questionnaire_2017_18` | {"English": 97, "French": 23} |
| `individual_comments_2016_18` | {"English": 68, "French": 2} |

Pilot rows (50): {"English": 44, "French": 6}

## Provenance and context

- Pilot rows are copied verbatim and first; new rows use the pilot's context format and question wording.
- `original_text` is the source cell trimmed of leading/trailing whitespace, as in the pilot.
- `source_record_id` equals `source_row_number`; native IDs (`row_rang_e`, `id`, `unique_identifier`) are in the ledger.

## Limitations

- **Not respondent-unique.** The sources contain no respondent identifiers. Uniqueness is only at the source-record level `(source_dataset, source_row_number)`. The same person may have submitted several records, and one card may appear in both the individual-comments and compilation files under different text.
- **Text duplicates:** matching normalised text does not prove a shared respondent. One occurrence per key is retained by a deterministic rule; others are in the ledger.
- **Questionnaire language** is the survey language the respondent used to complete the form, not necessarily the language of the comment.
- **Stratified, not representative.** Strata quotas are equal splits chosen for coverage, so the pool does not estimate population frequencies without weights; "Not Applicable" and blank themes are deliberately capped.
- **French coverage is incidental.** No language quota was set. French counts follow each dataset's availability (individual comments have very few French records).
- Arabic and Japanese records are excluded from this English/French pool, not from the corpus.
- Near-duplicate (non-exact) texts are not removed.

