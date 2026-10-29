# PolicyLens exploratory concern prioritization (v1)

**Status:** computed and automatically verified (PASSED output validation; evidence verification PASSED (275 ranked-concern examples, 10 collective-demand entries verbatim)). **Not human-validated, and not a measure of public importance.** Generated 2026-10-13T09:26:46+00:00 from `data/full_corpus/concern_prioritization_v1/`.

## 1. Interpretation rules

- Exploratory prioritization: the score orders concern groups for human inspection. It is not a validated measure of public importance, urgency, policy impact or representativeness, and it was not fitted to any human judgment (none exists).
- Counts are of entries (raw occurrences), unique texts, independent texts (near-duplicate groups) and source records. None of them is a count of respondents; repeated or templated comments are not independent people.
- Concern groups are unsupervised clusters, not ground-truth labels; their descriptions are extractive and not human-validated.
- Corpus: Canadian federal Open Government consultations 2016-18. Nothing here transfers to Indian MCA or other consultations.

## 2. Inputs and integrity

Inputs are the concern-discovery outputs and the original corpus, read-only and hash-checked. No embedding, clustering or model was run.

| Check | Status | Detail |
|---|---|---|
| concern-discovery inputs match run_manifest.json | PASSED | 4 files; selected configuration strip__umap_m25s5 |
| original corpus matches run_v1 manifest | PASSED | 10,108 entries |
| entry_assignments schema | PASSED | 13 required columns present, 10,108 rows |
| concern_groups schema | PASSED | 7 required columns present, 74 rows |
| representative_examples schema | PASSED | 6 required columns present, 370 rows |
| near_duplicate_groups schema | PASSED | 4 required columns present, 10,043 rows |
| assignment consistency | PASSED | 74 clusters; sizes agree; states {'clustered': 6226, 'noise': 3760, 'excluded_content_free_after_stripping': 122} |
| source ids and metadata agree with the original corpus | PASSED | dataset, record id, field, language |

## 3. Counting units

| Unit | What it counts | Corpus total |
|---|---|---:|
| Entries | raw occurrences: non-blank designed free-text responses | 10,108 |
| Unique texts | normalised distinct texts | 10,043 |
| Independent texts | near-duplicate groups; a repeated or templated text counts once | 9,757 |
| Source records | distinct dataset + record id; a record is a submission, row or card, not a person | 8,414 |

Entry states: {'clustered': 6226, 'noise': 3760, 'excluded_content_free_after_stripping': 122}. Noise and excluded entries are never ranked; a group's `nearby_noise_entries` (noise whose nearest group it is) is reported as context only, because their membership is uncertain.

## 4. Method

A group is ranked if its tier is candidate, needs_review or campaign_or_template and at least one supporting comment verifies verbatim against the original corpus. Each signal is scaled to [0, 1] across the ranked groups; the score is Σ weight × scaled signal, and each term is stored as a separate contribution column in `ranking.csv`.

| Signal | Definition | Weight |
|---|---|---:|
| `independent_volume` | log(1 + independent texts) / log(1 + largest ranked value) | 0.40 |
| `independence` | 1 − duplicate influence; duplicate influence = 1 − independent texts / entries | 0.10 |
| `dataset_diversity` | normalised Shannon entropy of independent texts over the 3 source datasets | 0.10 |
| `field_diversity` | normalised Shannon entropy over consultation fields (dataset + question) | 0.10 |
| `reliability` | mean best-match Jaccard of the group across UMAP seeds 42/43/44 | 0.15 |
| `tier_quality` | candidate 1.0, needs_review 0.5, campaign_or_template 0.5 | 0.05 |
| `evidence` | verified supporting comments, capped at 3, divided by 3 | 0.10 |

Weights source: defaults (models/concern_discovery/prioritization.py). They are documented judgement calls set before inspecting the ranking and are not fitted to any outcome; volume carries the largest weight but is measured in independent texts so that repetition cannot dominate. Baseline: rank by raw entries (ties: unique texts).

## 5. Multi-signal ranking (top 15)

| Rank | Freq. rank | Cluster | Tier | Label | Entries | Unique | Independent | Records | Score | `independent_volume` | `independence` | `dataset_diversity` | `field_diversity` | `reliability` | `tier_quality` | `evidence` |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 30 | candidate | open / government | 354 | 353 | 348 | 349 | 0.910 | 0.400 | 0.098 | 0.081 | 0.064 | 0.117 | 0.050 | 0.100 |
| 2 | 3 | 25 | candidate | transparency / trust | 205 | 202 | 200 | 205 | 0.885 | 0.362 | 0.098 | 0.078 | 0.057 | 0.140 | 0.050 | 0.100 |
| 3 | 2 | 57 | candidate | data / government | 251 | 251 | 249 | 251 | 0.862 | 0.377 | 0.099 | 0.074 | 0.052 | 0.110 | 0.050 | 0.100 |
| 4 | 8 | 26 | candidate | privacy / information | 128 | 128 | 127 | 126 | 0.853 | 0.331 | 0.099 | 0.078 | 0.059 | 0.135 | 0.050 | 0.100 |
| 5 | 11 | 18 | candidate | data / canada | 119 | 119 | 119 | 117 | 0.840 | 0.327 | 0.100 | 0.081 | 0.058 | 0.124 | 0.050 | 0.100 |
| 6 | 12 | 56 | candidate | citizens / engagement | 109 | 109 | 107 | 109 | 0.838 | 0.320 | 0.098 | 0.090 | 0.066 | 0.114 | 0.050 | 0.100 |
| 7 | 9 | 5 | candidate | immigration / immigrants | 127 | 127 | 127 | 120 | 0.818 | 0.331 | 0.100 | 0.038 | 0.051 | 0.147 | 0.050 | 0.100 |
| 8 | 13 | 9 | candidate | indigenous / peoples | 107 | 106 | 106 | 105 | 0.815 | 0.319 | 0.099 | 0.066 | 0.044 | 0.137 | 0.050 | 0.100 |
| 9 | 28 | 34 | candidate | information / access | 63 | 61 | 60 | 62 | 0.811 | 0.281 | 0.095 | 0.090 | 0.068 | 0.127 | 0.050 | 0.100 |
| 10 | 7 | 71 | candidate | open / data | 143 | 142 | 141 | 141 | 0.809 | 0.339 | 0.099 | 0.065 | 0.045 | 0.112 | 0.050 | 0.100 |
| 11 | 20 | 70 | candidate | electoral / reform | 78 | 76 | 76 | 77 | 0.796 | 0.297 | 0.097 | 0.061 | 0.049 | 0.142 | 0.050 | 0.100 |
| 12 | 5 | 68 | needs_review | government / accountability | 165 | 164 | 164 | 165 | 0.793 | 0.349 | 0.099 | 0.070 | 0.054 | 0.096 | 0.025 | 0.100 |
| 13 | 15 | 37 | needs_review | environment / climate | 101 | 101 | 101 | 99 | 0.785 | 0.316 | 0.100 | 0.050 | 0.053 | 0.141 | 0.025 | 0.100 |
| 14 | 24 | 2 | candidate | nations | 66 | 65 | 64 | 65 | 0.782 | 0.285 | 0.097 | 0.059 | 0.042 | 0.149 | 0.050 | 0.100 |
| 15 | 35 | 24 | candidate | information / disclosure | 51 | 50 | 50 | 51 | 0.777 | 0.269 | 0.098 | 0.070 | 0.057 | 0.133 | 0.050 | 0.100 |

Signal columns show each contribution (weight × scaled value); they sum to the score. All ranked groups: `ranking.csv`.

## 6. Frequency-only baseline (top 15)

| Freq. rank | Multi rank | Cluster | Label | Entries | Independent | Duplicate influence | Stability |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | 30 | open / government | 354 | 348 | 0.02 | 0.78 |
| 2 | 3 | 57 | data / government | 251 | 249 | 0.01 | 0.74 |
| 3 | 2 | 25 | transparency / trust | 205 | 200 | 0.02 | 0.93 |
| 4 | 25 | 72 | data | 198 | 197 | 0.01 | 0.56 |
| 5 | 12 | 68 | government / accountability | 165 | 164 | 0.01 | 0.64 |
| 6 | 18 | 39 | information / difficult | 144 | 142 | 0.01 | 0.83 |
| 7 | 10 | 71 | open / data | 143 | 141 | 0.01 | 0.75 |
| 8 | 4 | 26 | privacy / information | 128 | 127 | 0.01 | 0.90 |
| 9 | 7 | 5 | immigration / immigrants | 127 | 127 | 0.00 | 0.98 |
| 10 | 26 | 59 | people / income | 126 | 126 | 0.00 | 0.64 |
| 11 | 5 | 18 | data / canada | 119 | 119 | 0.00 | 0.83 |
| 12 | 6 | 56 | citizens / engagement | 109 | 107 | 0.02 | 0.76 |
| 13 | 8 | 9 | indigenous / peoples | 107 | 106 | 0.01 | 0.91 |
| 14 | 23 | 7 | women / gender | 104 | 101 | 0.03 | 0.95 |
| 15 | 13 | 37 | environment / climate | 101 | 101 | 0.00 | 0.94 |

## 7. How the rankings differ and why

- 55 groups ranked. Rank agreement with the frequency baseline: Spearman 0.7071, Kendall τ 0.53. Mean absolute rank change 9.891; largest 26.
- Top 5: 3 groups shared; entered [18, 26]; left [68, 72].
- Top 10: 6 groups shared; entered [9, 18, 34, 56]; left [39, 59, 68, 72].
- Top 20: 13 groups shared; entered [2, 4, 8, 17, 24, 34, 65]; left [7, 15, 31, 52, 59, 60, 72].
- Raw entries versus unique texts: Spearman 1.0. Among ranked groups the median duplicate influence is 0.000 (largest 0.38, cluster 6), so removing repeats changes few positions here; most differences come from reliability, diversity, tier and evidence.
- Volume is log-scaled: the largest ranked group gets a volume value of 1.00 and the smallest 0.50, although they differ 19-fold in independent texts. This deliberately limits how far size alone decides the order, which is what allows small but stable and diverse groups to move up.

Largest moves (explanations are computed from each group's contributions relative to the ranked-group mean):

- **Cluster 4 (democracy / real, candidate)**: frequency rank 46 → multi rank 20. up 26 vs frequency rank; above-average contributions: dataset_diversity +0.039, reliability +0.027; below-average: independent_volume -0.063, independence -0.020. Raw values: 32 entries, 25 independent texts, stability 1.00, 6 fields, 3 datasets, duplicate influence 0.22.
- **Cluster 24 (information / disclosure, candidate)**: frequency rank 35 → multi rank 15. up 20 vs frequency rank; above-average contributions: tier_quality +0.011, field_diversity +0.011; below-average: independent_volume -0.017. Raw values: 51 entries, 50 independent texts, stability 0.89, 5 fields, 3 datasets, duplicate influence 0.02.
- **Cluster 34 (information / access, candidate)**: frequency rank 28 → multi rank 9. up 19 vs frequency rank; above-average contributions: dataset_diversity +0.029, field_diversity +0.022; below-average: none. Raw values: 63 entries, 60 independent texts, stability 0.85, 9 fields, 3 datasets, duplicate influence 0.05.
- **Cluster 17 (online / media, candidate)**: frequency rank 32 → multi rank 16. up 16 vs frequency rank; above-average contributions: reliability +0.021, tier_quality +0.011; below-average: independent_volume -0.015. Raw values: 52 entries, 52 independent texts, stability 0.96, 7 fields, 3 datasets, duplicate influence 0.00.
- **Cluster 73 (data / available, needs_review)**: frequency rank 27 → multi rank 51. down 24 vs frequency rank; above-average contributions: none; below-average: reliability -0.047, tier_quality -0.014. Raw values: 63 entries, 63 independent texts, stability 0.51, 3 fields, 2 datasets, duplicate influence 0.00.
- **Cluster 52 (direction / start, needs_review)**: frequency rank 19 → multi rank 41. down 22 vs frequency rank; above-average contributions: independent_volume +0.013; below-average: dataset_diversity -0.015, tier_quality -0.014. Raw values: 79 entries, 78 independent texts, stability 0.82, 5 fields, 3 datasets, duplicate influence 0.01.
- **Cluster 72 (data, needs_review)**: frequency rank 4 → multi rank 25. down 21 vs frequency rank; above-average contributions: independent_volume +0.075; below-average: reliability -0.039, field_diversity -0.015. Raw values: 198 entries, 197 independent texts, stability 0.56, 8 fields, 3 datasets, duplicate influence 0.01.
- **Cluster 15 (canadians / canada, candidate)**: frequency rank 16 → multi rank 37. down 21 vs frequency rank; above-average contributions: independent_volume +0.027, tier_quality +0.011; below-average: dataset_diversity -0.030, reliability -0.015. Raw values: 96 entries, 96 independent texts, stability 0.72, 4 fields, 2 datasets, duplicate influence 0.00.

## 8. Ranking examples with source evidence

Quotes are verbatim spans of the cited original comment (verified against `run_v1/corpus/corpus_entries.csv`, with SHA-256). The group label is a list of frequent terms, not a validated concern statement.

**1. Cluster 30 — open / government** (candidate; 354 entries, 348 independent texts, 349 source records; score 0.910)

- [corpus_id 7646 · get_involved_questionnaire_2017_18 · record 1291 · q14 · English] “Just beginning to learn about open government”
- [corpus_id 4088 · consultation_compilation_2017_18 · record 3823 · comment_question_or_idea · French] “D’autres sujets reliés à l’idée du gouvernement ouvert qui vous semblent importants”
- [corpus_id 9411 · get_involved_questionnaire_2017_18 · record 7912 · q14 · French] “Faute de connaissances appronfondies sur ce qu'est le gouvernement ouvert!”

**2. Cluster 25 — transparency / trust** (candidate; 205 entries, 200 independent texts, 205 source records; score 0.885)

- [corpus_id 4795 · consultation_compilation_2017_18 · record 4611 · comment_question_or_idea · French] “la transparence au niveau des comptes publiques et depense du government... une plus grande transparence et une participation”
- [corpus_id 6614 · get_involved_questionnaire_2017_18 · record 5460 · q8 · English] “OPINION: Transparency is not only opening the books and showing what we what do, and look what we've done, it's also about having integrity and honesty from”…
- [corpus_id 4686 · consultation_compilation_2017_18 · record 4484 · comment_question_or_idea · English] “Proactive disclosure, transparency, accountability”

**3. Cluster 57 — data / government** (candidate; 251 entries, 249 independent texts, 251 source records; score 0.862)

- [corpus_id 4385 · consultation_compilation_2017_18 · record 4136 · comment_question_or_idea · English] “How can government gather and share examples or uses of open data in action?”
- [corpus_id 3473 · consultation_compilation_2017_18 · record 3200 · comment_question_or_idea · French] “Sous quels formats seraient rendus accessibles les données gouvernementales aux citoyens?”
- [corpus_id 4862 · consultation_compilation_2017_18 · record 4697 · comment_question_or_idea · English] “How to better drive use of government data.”

**4. Cluster 26 — privacy / information** (candidate; 128 entries, 127 independent texts, 126 source records; score 0.853)

- [corpus_id 3481 · consultation_compilation_2017_18 · record 3208 · comment_question_or_idea · French] “définir les limites de la vie privée vs les données publiques”
- [corpus_id 4845 · consultation_compilation_2017_18 · record 4678 · comment_question_or_idea · English] “Safeguard for privacy - where appropriate”
- [corpus_id 10098 · get_involved_questionnaire_2017_18 · record 8084 · q18 · French] “Loi sur l'Accès à l'Information et Loi sur la protection des renseignements personnels”

**5. Cluster 18 — data / canada** (candidate; 119 entries, 119 independent texts, 117 source records; score 0.840)

- [corpus_id 2179 · consultation_compilation_2017_18 · record 1882 · comment_question_or_idea · English] “Open data - more engagement with Canadians to educate them on the subject”
- [corpus_id 8404 · get_involved_questionnaire_2017_18 · record 4122 · q14 · English] “It does provide Canadians with important data.”
- [corpus_id 6033 · get_involved_questionnaire_2017_18 · record 3348 · q8 · English] “-Statistics Canada raw data being available directly to citizens, including public servants (not just academics) -Open Data (accessible, timely/current, broad”…

**6. Cluster 56 — citizens / engagement** (candidate; 109 entries, 107 independent texts, 109 source records; score 0.838)

- [corpus_id 4684 · consultation_compilation_2017_18 · record 4481 · comment_question_or_idea · English] “Public engagement, citizen centred services, policy codesign”
- [corpus_id 6577 · get_involved_questionnaire_2017_18 · record 5337 · q8 · English] “ability to share our suggestions with government and response on citizens suggestions”
- [corpus_id 3335 · consultation_compilation_2017_18 · record 3060 · comment_question_or_idea · English] “citizen engagement how can they contribute, what are their priorities”

Signals that take the same value for every ranked group in this run, and therefore add a constant without changing the order: `evidence`. Here they act only through the eligibility rule.

## 9. Robustness to the weights

Each weight was dropped, halved and doubled (renormalised): 21 variants. Minimum Kendall τ against the default ranking: 0.4882. Groups in the default top 10 that stay in the top 10 under every variant: [18, 25, 26, 30, 56].

Most influential perturbations: `drop_independent_volume` τ = 0.4882; `drop_reliability` τ = 0.7306; `drop_dataset_diversity` τ = 0.7549; `half_independent_volume` τ = 0.7603. Positions that change under plausible weights should not be read as meaningful differences.

## 10. Collective-demand candidates (repeated text)

Near-duplicate groups with at least 5 entries were screened: 17 groups; 1 state something substantive. Their entry counts measure how often the same wording was submitted. They may reflect a coordinated campaign and are **not** counts of people.

| Group | Entries | Unique texts | Source records | Datasets | Assigned clusters | Lead text (verbatim) |
|---:|---:|---:|---:|---|---|---|
| 3993 | 10 | 10 | 10 | {"consultation_compilation_2017_18": 10} | {"6": 10} | [corpus_id 3997] “I want to see Canada make a comittment for a publicly accessible registry of beneficial owners of companies and properties to combat”… |

Every member entry, with source ids and verbatim text, is in `collective_demand_evidence.csv` (10 rows).

Screened out: group 7325 (14 entries) — no-opinion / awareness or promotional wording; group 7438 (14 entries) — no-opinion / awareness or promotional wording; group 9132 (12 entries) — no-opinion / awareness or promotional wording; group 2835 (8 entries) — no-opinion / awareness or promotional wording; group 7331 (8 entries) — no-opinion / awareness or promotional wording; group 7510 (8 entries) — no-opinion / awareness or promotional wording; group 7562 (8 entries) — no-opinion / awareness or promotional wording; group 7323 (7 entries) — no-opinion / awareness or promotional wording; group 3660 (6 entries) — no-opinion / awareness or promotional wording; group 7504 (6 entries) — no-opinion / awareness or promotional wording; group 7425 (5 entries) — no-opinion / awareness or promotional wording; group 7464 (5 entries) — no-opinion / awareness or promotional wording; group 7676 (5 entries) — no-opinion / awareness or promotional wording; group 7742 (5 entries) — no-opinion / awareness or promotional wording; group 2762 (24 entries) — no substantive text once links, mentions and hashtags are removed; group 3617 (22 entries) — no substantive text once links, mentions and hashtags are removed.

## 11. Groups not ranked

19 of 74 groups are not ranked: non_concern_pattern (10 groups, 1,019 entries); unreliable (9 groups, 751 entries). Reasons per group are in `not_ranked.csv`.

## 12. Limitations

- No human relevance or importance judgments exist; the ranking is not validated and was not tuned to any outcome. Weights are judgement calls; section 9 shows how much positions depend on them.
- Counts are entries, texts and records, never people. A questionnaire submission can add up to three entries, and duplicates across instruments cannot be traced to a person.
- Volume reflects which consultation channels and questions collected text, not how widely a concern is held in the population; the instruments differ in size and design.
- Group boundaries come from one embedding model and one clustering configuration; agreement between encoders is low (concern-discovery report, section 5), so a different configuration would give different groups and ranks.
- Reliability uses only UMAP-seed stability; tiers and flags are heuristic diagnostics, not quality judgments.
- Diversity is measured over datasets and question fields only; no stakeholder identity, demographic or urgency information is used or inferred.
- Broad, generic groups can rank high on volume and diversity; their labels are frequent terms and should be read with the evidence comments.
- Collective-demand screening uses keyword rules for no-opinion and promotional wording, and the 5-entry threshold is a convention.

## 13. Reproduction

```powershell
python scripts\run_concern_prioritization.py run                       # default weights
python scripts\run_concern_prioritization.py run --weights my_weights.json --out data\full_corpus\concern_prioritization_alt
python scripts\run_concern_prioritization.py report
python -m pytest tests\test_concern_prioritization.py -q -p no:cacheprovider
```

Outputs are write-once; `run_manifest.json` records SHA-256 of every input and output. Runtime 1.75 s.

