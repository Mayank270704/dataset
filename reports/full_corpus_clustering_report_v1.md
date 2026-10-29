# PolicyLens full-corpus unsupervised clustering report (v1)

**Status: exploratory, unsupervised, NOT human-validated.** Clusters are groups of semantically similar texts under one embedding model; they are not validated public concerns. No human labels, reviewer agreement or accuracy figures were used or produced. Sentiment, stance and concern are not inferred.

## 1. Corpus definition

- **Unit:** a text entry = a non-blank response in a designed free-text field of a source record (individual comments: 5 fields; compilation: the comment column; questionnaire: Q8, Q14, Q18).
- **Eligibility:** >= 20 characters after trim, not generic filler, exclusion_reason blank (models.semantic_similarity.minilm.is_eligible).
- **Eligible entries:** 10,108. **Unique normalised texts clustered:** 10,043 (65 repeated entries mapped back to their first occurrence; multiplicity kept).
- **Not respondent-level:** one questionnaire respondent may contribute up to 3 entries; the sources contain no respondent identifier.
- **Entries by dataset:** `consultation_compilation_2017_18`=4,832, `get_involved_questionnaire_2017_18`=4,832, `individual_comments_2016_18`=444

### Reconciliation of the extracted entries

`data/interim/government_response_text_entries.csv` has **14,135** extracted entries. They are accounted for as follows:

| Dataset | designed_field | other_free_text_field_not_designed | translation_column_not_designed | designed_field_eligible_entries | designed_field_short_or_filler | total_extracted_entries |
|---|---:|---:|---:|---:|---:|---:|
| `consultation_compilation_2017_18` | 5168 | 0 | 527 | 4832 | 336 | 5695 |
| `get_involved_questionnaire_2017_18` | 5387 | 2577 | 0 | 4832 | 555 | 7964 |
| `individual_comments_2016_18` | 476 | 0 | 0 | 444 | 32 | 476 |
| **total** | 11031 | 2577 | 527 | 10108 | 923 | 14135 |

The figure **13,541** quoted in `reports/current_project_status.md` could not be reproduced from any filter of this table (it has no derivation in the repository) and should not be used; the counts above supersede it.

### Language metadata (source metadata, not inferred)

| Dataset \| language | entries |
|---|---:|
| consultation_compilation_2017_18|Arabic | 2 |
| consultation_compilation_2017_18|English | 4,530 |
| consultation_compilation_2017_18|French | 299 |
| consultation_compilation_2017_18|Japanese | 1 |
| get_involved_questionnaire_2017_18|English | 4,064 |
| get_involved_questionnaire_2017_18|French | 768 |
| individual_comments_2016_18|English | 421 |
| individual_comments_2016_18|French | 20 |
| individual_comments_2016_18|S | 3 |

Questionnaire language is the survey language selected, not necessarily the comment's language. 'S' (3), Arabic (2) and Japanese (1) entries are kept and counted as Other/Unknown.

## 2. Embeddings

| Model | Revision snapshot | Unique texts | Dim | Chunks (computed/reused) | Elapsed s | Peak RSS MB | Texts > 256 tokens |
|---|---|---:|---:|---|---:|---:|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | e8f8c211226b | 10,043 | 384 | 79/0 | 126.4 | 763.3 | 78 (0.8%) |
| `sentence-transformers/all-MiniLM-L6-v2` | 1110a243fdf4 | 10,043 | 384 | 79/0 | 77.6 | 494.6 | 74 (0.7%) |

Equivalence check against the saved 450-comment vectors (same texts, same pooling):

- `multilingual`: 450 texts compared, max |Δ| = 5.70e-03, min cosine = 0.999627
- `baseline`: 450 texts compared, max |Δ| = 1.60e-07, min cosine = 1.000000

Exact-text collapsing uses a case- and whitespace-insensitive key, so 20 entries share the embedding of a first-occurrence variant that differs only in case or spacing (for example 'Did not know it existed' / 'did not know it existed'). The one 450-sample text that differs beyond 1e-6 in the equivalence check is such a variant (cosine 0.9996).

Model loading used a low-memory loader (meta-device build, tensors assigned one by one) because the default `from_pretrained` crashed with an access violation at ~500 MB available RAM; weights are identical (see the equivalence check). Model revisions are not pinned.

## 3. Clustering configurations — `multilingual`

UMAP (cosine, 5-D, min_dist 0, seed 42) then HDBSCAN (euclidean, EOM). `u15`/`m10s5` are the settings used for the 450-comment baseline. Silhouette is a geometric descriptor on non-noise points of a 4,000-point sample; NPMI is a lexical co-occurrence proxy for keyword coherence (each cluster's top-10 c-TF-IDF terms; a pair that never co-occurs in one text scores -1, so values are mostly negative for short texts and are only meaningful as relative comparisons). Neither shows semantic or policy validity. 'Channel-dominated' = clusters with at least 20 compilation entries where one source channel (Twitter, Slido, RSVP forms, etc.) supplies ≥70% of them; 'template-heavy' = a text, after removing URLs, repeated ≥5 times and making up ≥30% of the cluster.

| Config | Clusters | Noise (entries) | Size median / max | Top-5 share | ≥90% French | ≥95% one dataset | Channel-dominated (≥70%) | Template-heavy | Silhouette | Rel. validity | Mean NPMI |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| u15_m10s5 | 179 | 42.2% | 21 / 295 | 17% | 0 | 25 | 10 | 1 | 0.566 | 0.2243 | -0.3389 |
| u15_m25s5 | 66 | 31.2% | 57 / 1154 | 38% | 0 | 9 | 7 | 0 | 0.4516 | 0.2863 | -0.177 |
| u15_m50s10 | 32 | 31.4% | 134 / 1287 | 53% | 0 | 5 | 1 | 0 | 0.4473 | 0.2149 | -0.1709 |
| u30_m10s5 | 151 | 43.5% | 22 / 298 | 18% | 0 | 30 | 8 | 1 | 0.5254 | 0.1374 | -0.3212 |
| u30_m25s5 | 68 | 39.0% | 60 / 608 | 28% | 0 | 8 | 2 | 0 | 0.5409 | 0.2011 | -0.1904 |
| u30_m50s10 | 33 | 34.0% | 121 / 1321 | 54% | 0 | 5 | 1 | 0 | 0.4435 | 0.1297 | -0.1939 |

Seed stability (u15_m25s5, UMAP seeds [42, 43]): ARI on points clustered in both = 0.6976, clustered in both = 57%, clusters [66, 77].

Runtime 38.7 s; peak RSS 671.6 MB; UMAP seconds: {'u15_seed42': 11.7, 'u30_seed42': 8.6, 'u15_seed43': 6.8}

## 3. Clustering configurations — `baseline`

UMAP (cosine, 5-D, min_dist 0, seed 42) then HDBSCAN (euclidean, EOM). `u15`/`m10s5` are the settings used for the 450-comment baseline. Silhouette is a geometric descriptor on non-noise points of a 4,000-point sample; NPMI is a lexical co-occurrence proxy for keyword coherence (each cluster's top-10 c-TF-IDF terms; a pair that never co-occurs in one text scores -1, so values are mostly negative for short texts and are only meaningful as relative comparisons). Neither shows semantic or policy validity. 'Channel-dominated' = clusters with at least 20 compilation entries where one source channel (Twitter, Slido, RSVP forms, etc.) supplies ≥70% of them; 'template-heavy' = a text, after removing URLs, repeated ≥5 times and making up ≥30% of the cluster.

| Config | Clusters | Noise (entries) | Size median / max | Top-5 share | ≥90% French | ≥95% one dataset | Channel-dominated (≥70%) | Template-heavy | Silhouette | Rel. validity | Mean NPMI |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| u15_m10s5 | 167 | 36.4% | 20 / 1063 | 28% | 1 | 40 | 8 | 1 | 0.622 | 0.258 | -0.1976 |
| u15_m25s5 | 68 | 37.1% | 59 / 1063 | 32% | 1 | 13 | 6 | 1 | 0.6275 | 0.288 | -0.078 |
| u15_m50s10 | 32 | 33.1% | 111 / 1072 | 58% | 1 | 5 | 2 | 0 | 0.545 | 0.2529 | -0.0067 |
| u30_m10s5 | 135 | 38.2% | 25 / 1064 | 30% | 1 | 30 | 8 | 1 | 0.6116 | 0.278 | -0.1898 |
| u30_m25s5 | 61 | 37.1% | 57 / 1064 | 42% | 1 | 9 | 7 | 1 | 0.5719 | 0.2542 | -0.0888 |
| u30_m50s10 | 29 | 32.1% | 110 / 1271 | 60% | 1 | 3 | 1 | 0 | 0.5638 | 0.4138 | 0.0252 |

Runtime 30.4 s; peak RSS 597.5 MB; UMAP seconds: {'u15_seed42': 11.7, 'u30_seed42': 8.0}

## 4. Model comparison

Agreement between the two embedding models' partitions, per UMAP/HDBSCAN configuration (ARI on points clustered by both; low values mean the models organise the corpus differently, not that one is better):

| Config | ARI | clustered by both |
|---|---:|---:|
| u15_m10s5 | 0.2671 | 42% |
| u15_m25s5 | 0.2858 | 48% |
| u15_m50s10 | 0.4951 | 53% |
| u30_m10s5 | 0.3168 | 40% |
| u30_m25s5 | 0.3209 | 44% |
| u30_m50s10 | 0.5129 | 52% |

## 5. Reporting configuration: `multilingual/u15_m50s10`

Reporting configuration = among the multilingual-model configurations with 20 <= clusters <= 150 and noise <= 40%, the one with the highest mean NPMI coherence (ties: larger min_cluster_size). This only chooses which configuration is listed in detail; it does not validate it.

Largest clusters (keywords = c-TF-IDF terms; examples = texts nearest the cluster centroid, verbatim, traceable by `corpus_id`):

### Cluster 9 — 1287 unique texts; 1289 entries; French 10%; top dataset get_involved_questionnaire_2017_18 (54%); duplicate-text share 0%

- Keywords: canada, canadians, canadian, government, https, opengovcan, open, people, country, need
- corpus_id 3585 (consultation_compilation_2017_18, comment_question_or_idea, English): “Help @TBS_Canada Shape Canada’s Action plan on Open Government. https://t.co/tDXGheZnAn #opengovcan”
- corpus_id 4592 (consultation_compilation_2017_18, comment_question_or_idea, French): “Des sujets pour les citoyens Canadians”
- corpus_id 3755 (consultation_compilation_2017_18, comment_question_or_idea, English): “Help @TBS_Canada Shape Canada’s Action plan on Open Government. https://t.co/psa9iW60Kg”

### Cluster 14 — 1159 unique texts; 1164 entries; French 8%; top dataset consultation_compilation_2017_18 (73%); duplicate-text share 0%

- Keywords: data, open data, open, données, use, information, access, datasets, science, data open
- corpus_id 4385 (consultation_compilation_2017_18, comment_question_or_idea, English): “How can government gather and share examples or uses of open data in action?”
- corpus_id 3291 (consultation_compilation_2017_18, comment_question_or_idea, English): “How should the government use open data & engagement to foster data literacy and effective use of data?”
- corpus_id 4917 (consultation_compilation_2017_18, comment_question_or_idea, English): “Priorities for open data”

### Cluster 4 — 623 unique texts; 624 entries; French 3%; top dataset consultation_compilation_2017_18 (98%); duplicate-text share 0%

- Keywords: opengovcan, https, opengovcan https, opengov, https opengovcan, opendata, gcdigital, opengovpart, thanks, great
- corpus_id 3682 (consultation_compilation_2017_18, comment_question_or_idea, English): “#opengovcan #opensource https://t.co/COqkpEgVhs”
- corpus_id 2757 (consultation_compilation_2017_18, comment_question_or_idea, English): “@OpenGovCan @opengovpart @CivicTechTO @CivicActionGTHA @SamaraCDA https://t.co/vfH4YPnCpP”
- corpus_id 3651 (consultation_compilation_2017_18, comment_question_or_idea, English): “#OpenGovCan https://t.co/uNzBea49w5”

### Cluster 15 — 306 unique texts; 306 entries; French 9%; top dataset get_involved_questionnaire_2017_18 (55%); duplicate-text share 0%

- Keywords: open government, open, government, gov, open gov, gouvernement ouvert, ouvert, gouvernement, government open, public
- corpus_id 7646 (get_involved_questionnaire_2017_18, q14, English): “Just beginning to learn about open government”
- corpus_id 4088 (consultation_compilation_2017_18, comment_question_or_idea, French): “D’autres sujets reliés à l’idée du gouvernement ouvert qui vous semblent importants”
- corpus_id 3420 (consultation_compilation_2017_18, comment_question_or_idea, English): “how to do "open government" properly”

### Cluster 31 — 285 unique texts; 287 entries; French 15%; top dataset get_involved_questionnaire_2017_18 (59%); duplicate-text share 1%

- Keywords: government, accountability, public, employees, govt, departments, accountable, gouvernement, service, publique
- corpus_id 7271 (get_involved_questionnaire_2017_18, q8, French): “Un gouvernement responsable et décisif”
- corpus_id 10017 (get_involved_questionnaire_2017_18, q18, French): “Maitrise en administration publique”
- corpus_id 9841 (get_involved_questionnaire_2017_18, q18, English): “Accountability and Citizens knowing the duties of government officials”

### Cluster 17 — 249 unique texts; 258 entries; French 26%; top dataset get_involved_questionnaire_2017_18 (97%); duplicate-text share 2%

- Keywords: site, website, visited, ai, haven, site ai, visited site, know website, existed, encore
- corpus_id 8949 (get_involved_questionnaire_2017_18, q14, English): “I haven't visit the site”
- corpus_id 7583 (get_involved_questionnaire_2017_18, q14, English): “I haven't gone to the website”
- corpus_id 9182 (get_involved_questionnaire_2017_18, q14, French): “Je n'ai pas encore visité le site”

### Cluster 22 — 210 unique texts; 219 entries; French 12%; top dataset get_involved_questionnaire_2017_18 (91%); duplicate-text share 1%

- Keywords: don, don know, know, comment, familiar, opinion, know don, assez, knowledge, experience
- corpus_id 9223 (get_involved_questionnaire_2017_18, q14, French): “Je ne suis pas bien informé à ce sujet”
- corpus_id 9417 (get_involved_questionnaire_2017_18, q14, French): “Pas assez informé pour pouvoir répondre.”
- corpus_id 7833 (get_involved_questionnaire_2017_18, q14, English): “I currently do not know enough about it.”

### Cluster 26 — 179 unique texts; 179 entries; French 4%; top dataset consultation_compilation_2017_18 (82%); duplicate-text share 1%

- Keywords: engagement, community, groups, participation, outreach, organizations, support, civil, communities, civil society
- corpus_id 6418 (get_involved_questionnaire_2017_18, q8, English): “Public outreach and stakeholder engagement”
- corpus_id 2439 (consultation_compilation_2017_18, comment_question_or_idea, English): “include stakeholders in shaping the engagement process”
- corpus_id 3338 (consultation_compilation_2017_18, comment_question_or_idea, English): “how can engagement shift to include nonprofit/social sector in the co-creation of enabling policy process”

### Cluster 25 — 176 unique texts; 176 entries; French 19%; top dataset get_involved_questionnaire_2017_18 (81%); duplicate-text share 1%

- Keywords: income, seniors, pension, help, jobs, people, life, personnes, programs, tax
- corpus_id 7289 (get_involved_questionnaire_2017_18, q8, French): “le gouvernement aide d'avantage les familles et les personnes riches mais qu'en est -il de la classe moyenne et des personnes de 65 ans et plus avec nos maigres revenus car nous n'avons pas eu le choix de vivre avec un maigre salaire sans a…”
- corpus_id 2513 (consultation_compilation_2017_18, comment_question_or_idea, French): “un revenu minimum garantira une meilleure santé pour tous, abolira le cafouillis de l'éligibilité aux programmes sociaux, favorisera du temps pour l'éducation ou la formation, aidera la persévérance des petits commerçants et entrepreneurs, …”
- corpus_id 5974 (get_involved_questionnaire_2017_18, q8, English): “Low income/Disability issues regarding Health, Housing, and Government involvement”

### Cluster 11 — 166 unique texts; 169 entries; French 11%; top dataset consultation_compilation_2017_18 (52%); duplicate-text share 2%

- Keywords: transparency, trust, accountability, transparence, transparent, transparency accountability, government, public, accountability transparency, government transparency
- corpus_id 4795 (consultation_compilation_2017_18, comment_question_or_idea, French): “la transparence au niveau des comptes publiques et depense du government... une plus grande transparence et une participation”
- corpus_id 5568 (get_involved_questionnaire_2017_18, q8, English): “transparency and accountability”
- corpus_id 234 (individual_comments_2016_18, og_is_important_because, English): “Transparency leads to accountability.”

### Cluster 10 — 160 unique texts; 161 entries; French 7%; top dataset consultation_compilation_2017_18 (52%); duplicate-text share 1%

- Keywords: privacy, information, disclosure, act, security, access information, personal, access, information act, proactive disclosure
- corpus_id 3481 (consultation_compilation_2017_18, comment_question_or_idea, French): “définir les limites de la vie privée vs les données publiques”
- corpus_id 10098 (get_involved_questionnaire_2017_18, q18, French): “Loi sur l'Accès à l'Information et Loi sur la protection des renseignements personnels”
- corpus_id 6379 (get_involved_questionnaire_2017_18, q8, English): “Privacy of personal information in open government environment - how tight the government will control its citizens with all that information received and shared”

### Cluster 23 — 153 unique texts; 159 entries; French 12%; top dataset get_involved_questionnaire_2017_18 (96%); duplicate-text share 1%

- Keywords: haven, looked, used, ai, encore, haven looked, checked, looked haven, seen, services
- corpus_id 8935 (get_involved_questionnaire_2017_18, q14, English): “I have not looked into it yet.”
- corpus_id 9313 (get_involved_questionnaire_2017_18, q14, French): “Je n'y ai pas encore eu recours.”
- corpus_id 8685 (get_involved_questionnaire_2017_18, q14, English): “Haven't checked it out yet.”

### Cluster 29 — 152 unique texts; 153 entries; French 23%; top dataset get_involved_questionnaire_2017_18 (69%); duplicate-text share 1%

- Keywords: ideas, direction, improvement, right direction, step, room, trop, start, room improvement, step right
- corpus_id 7507 (get_involved_questionnaire_2017_18, q14, English): “Improving, but more to be done.”
- corpus_id 5164 (consultation_compilation_2017_18, comment_question_or_idea, French): “qui fonctionne bien - tous les avenues d'engagement”
- corpus_id 9312 (get_involved_questionnaire_2017_18, q14, French): “L'intention est bonne mais il reste du travail à faire pour rendre l'information utile”

### Cluster 8 — 149 unique texts; 155 entries; French 8%; top dataset get_involved_questionnaire_2017_18 (94%); duplicate-text share 2%

- Keywords: open canada, canada ca, ca, canada, open, heard open, aware open, heard, visited open, aware
- corpus_id 7762 (get_involved_questionnaire_2017_18, q14, English): “not aware of Open.Canada.ca”
- corpus_id 7341 (get_involved_questionnaire_2017_18, q14, English): “I am not familiar with Open.Canada.ca”
- corpus_id 8280 (get_involved_questionnaire_2017_18, q14, English): “I have not seen much information on Open.Canada.ca”

### Cluster 24 — 147 unique texts; 147 entries; French 12%; top dataset get_involved_questionnaire_2017_18 (69%); duplicate-text share 1%

- Keywords: tax, spending, taxes, money, debt, budget, government, government spending, spent, taxation
- corpus_id 6039 (get_involved_questionnaire_2017_18, q8, English): “*Accountability for excessive government spending”
- corpus_id 7115 (get_involved_questionnaire_2017_18, q8, French): “Gouvernement responsable avec les finances publiques. Des dépenses controlees et des coupures aux bons endroits.”
- corpus_id 7081 (get_involved_questionnaire_2017_18, q8, French): “Investir dans la fonction publique afin de réduire les délais de traitment des dossiers des contribuables (Impôt entre autres)”

Full per-cluster keyword and representative tables for every configuration are in `data/full_corpus/run_v1/clustering/<model>/<config>/{topics,representatives}.csv`.

## 5b. Data-driven observations (computed, not interpretive)

- Between 31% and 44% of entries are left as noise (unassigned) depending on configuration; a cluster map therefore covers only part of the corpus.
- In the reporting configuration, 1 of 32 clusters are dominated (≥70%) by a single compilation channel and 5 come ≥95% from a single instrument; 0 are template-heavy. The corpus has 189 entries sharing identical text once URLs are removed (beyond the 65 exact duplicates already collapsed).
- The largest cluster holds 1,287 unique texts (top-5 clusters = 53% of clustered texts), so a few broad clusters dominate.
- Clusters with ≥90% French entries: 0 (corpus French share 10.8%): French texts are distributed among mixed-language clusters under the multilingual model; the English-only baseline model at the same configuration has 1.
- Changing only the UMAP seed (u15_m25s5) gives ARI 0.6976 on the 57% of points clustered under both seeds, with cluster counts [66, 77]: the partition is only moderately stable.
- The two embedding models give different partitions (see section 4); neither is validated, so no model is declared better.

## 5c. Resources: pre-run estimate vs observed

| Item | Estimate before running | Observed |
|---|---|---|
| Encode `multilingual` (peak RSS) | ~450 MB weights + ~600 MB working set | 763.3 MB in 126.4 s |
| Encode `baseline` (peak RSS) | ~90 MB weights + ~600 MB working set | 494.6 MB in 77.6 s |
| UMAP/HDBSCAN `multilingual` | ~900 MB, minutes | 671.6 MB in 38.7 s |
| UMAP/HDBSCAN `baseline` | ~900 MB, minutes | 597.5 MB in 30.4 s |
| Artifacts under `data/full_corpus/run_v1` | ~70 MB | 78.0 MB |

Machine: 12 logical CPUs, 7860 MB RAM (about 128 MB free at run time), 10.4 GB free disk. The run completed on this memory-constrained machine because of the low-memory loader and the exact-kNN path (section 6b).

## 6. Limitations

- Clusters are unsupervised and have not been human-reviewed; the 44-pair concern-coherence review is still pending.
- Silhouette, relative validity and NPMI are proxies and can be high for incoherent or channel-driven clusters.
- Clusters may reflect response channel, question wording, language or boilerplate (e.g. event RSVPs, hashtags) rather than a policy concern; inspect composition columns.
- Texts above 256 tokens are truncated; very long responses are represented by their first 256 tokens.
- Short texts (≥20 characters) carry little context, so embeddings of generic remarks may be poorly separated.
- Exact duplicates are collapsed; near-duplicates and campaign text are not removed and may form dense clusters.
- The corpus mixes three instruments with different prompts; counts are not population estimates and no design weighting is applied.
- Language labels are source metadata; cross-lingual alignment is not established by language mixing alone.
- Embedding models are not revision-pinned; exact numerical reproduction across library or model revisions is not guaranteed.

## 6b. Implementation notes

- **Low-memory model loading** (see section 2).
- **Exact k-nearest-neighbour graph for UMAP.** UMAP's default approximate NN-descent (pynndescent) failed with a numba `MemoryError` on this machine (a few hundred MB of free RAM). The cosine kNN graph is therefore computed exactly by chunked brute force and passed to UMAP via `precomputed_knn`; all other UMAP/HDBSCAN hyper-parameters equal the 450-comment baseline. Exact neighbours can differ slightly from the approximate graph used for the 450 baseline.
- HDBSCAN is called directly with the baseline parameters (the existing wrapper also computes an extra 2-D UMAP that is not needed here).


## 7. Reproducibility

```powershell
python scripts\run_full_corpus_clustering.py corpus
python scripts\run_full_corpus_clustering.py embed --model-key multilingual
python scripts\run_full_corpus_clustering.py embed --model-key baseline
python scripts\run_full_corpus_clustering.py cluster --model-key multilingual
python scripts\run_full_corpus_clustering.py cluster --model-key baseline
python scripts\run_full_corpus_clustering.py report
```

Seed 42; UMAP `random_state` fixes results on a given machine/library set. Stages resume from chunk and artifact caches and refuse to overwrite outputs.
