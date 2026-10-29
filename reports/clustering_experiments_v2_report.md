# PolicyLens clustering experiments v2

**Exploratory, unsupervised, NOT human-validated.** All figures are proxies computed without labels; no configuration is declared the winner and no cluster is a validated concern. Silhouette is not used for selection.

## 1. Design

- **Population:** the 10,108 eligible entries / 10,043 unique texts of `data/full_corpus/run_v1` (multilingual embeddings), for every experiment.
- **Variants:** `raw` (reference), `strip` (URLs/@mentions/#hashtags removed; 121 texts content-free and not clustered), `strip_dedup` (33 unique texts collapsed into template representatives; entries inherit the cluster), `no_short` (texts under 40 characters not clustered: 1899).
- **Comparable core C:** 8,023 unique texts (≥ 40 characters after stripping). Coverage and flag shares below are computed on C so that all variants are compared on the same texts.
- Texts changed by stripping: 1,354; stripped embeddings were computed only for those, all others reuse run_v1 vectors.
- **Regression:** `raw__umap_m50s10` reproduces run_v1/u15_m50s10; `raw__umap_m25s5` reproduces run_v1/u15_m25s5.

## 2. Results (seed 42; proxies only)

| Experiment | Clusters | Coverage (all) | Coverage (core) | Noise (core) | Driven-cluster share | Low/mixed share | Effective substantive coverage | Cohesion | Seed ARI mean (min) | EN coverage | FR coverage | NPMI | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `raw__umap_m25s5` | 66 | 69% | 66% | 34% | 13% | 54% | 57% | 0.6692 | 0.78 (0.70) | 67% | 65% | -0.177 | 42.4 |
| `raw__umap_m50s10` | 32 | 69% | 67% | 33% | 8% | 64% | 62% | 0.6485 | 0.97 (0.96) | 67% | 67% | -0.1709 | 21.4 |
| `strip__umap_m25s5` | 74 | 62% | 60% | 40% | 5% | 27% | 57% | 0.6843 | 0.93 (0.90) | 60% | 62% | -0.2292 | 43.6 |
| `strip__umap_m50s10` | 33 | 63% | 62% | 38% | 3% | 58% | 60% | 0.6452 | 0.96 (0.94) | 62% | 64% | -0.1559 | 43.3 |
| `strip_dedup__umap_m25s5` | 78 | 59% | 57% | 43% | 6% | 31% | 53% | 0.6881 | — | 56% | 59% | -0.2486 | 15.7 |
| `strip_dedup__umap_m50s10` | 36 | 66% | 66% | 34% | 3% | 58% | 64% | 0.6431 | — | 65% | 68% | -0.1941 | 15.9 |
| `no_short__umap_m25s5` | 60 | 47% | 59% | 41% | 5% | 36% | 56% | 0.6766 | — | 59% | 62% | -0.1647 | 12.9 |
| `no_short__umap_m50s10` | 25 | 48% | 61% | 39% | 4% | 61% | 59% | 0.6381 | — | 61% | 61% | -0.0856 | 11.9 |
| `strip__umap_m25s5_leaf` | 86 | 55% | 53% | 47% | 6% | 32% | 50% | 0.694 | — | 53% | 55% | -0.259 | 15.9 |
| `raw__pca20_hdb_m25s5` | 11 | 15% | 11% | 89% | 20% | 55% | 9% | 0.711 | — | 11% | 12% | -0.0554 | 2.6 |
| `strip__pca20_hdb_m25s5` | 3 | 28% | 26% | 74% | 0% | 89% | 26% | 0.5558 | — | 25% | 29% | 0.005 | 2.4 |
| `raw__kmeans_k40` | 40 | 100% | 100% | 0% | 8% | 43% | 92% | 0.6388 | — | 100% | 100% | -0.0735 | 6.5 |
| `strip__kmeans_k40` | 40 | 99% | 100% | 0% | 2% | 36% | 98% | 0.6305 | — | 100% | 100% | -0.0543 | 5.1 |
| `strip__kmeans_k80` | 80 | 99% | 100% | 0% | 4% | 33% | 96% | 0.6611 | — | 100% | 100% | -0.2081 | 8.4 |
| `strip__umap_m25s5_englishonly` | 72 | 53% | 58% | 42% | 5% | 30% | 55% | 0.6773 | — | 58% | 0% | -0.1612 | 14.3 |
| `strip__umap_m10s5_frenchonly` | 26 | 8% | 78% | 22% | 2% | 51% | 77% | 0.6825 | — | 0% | 78% | -0.3217 | 2.0 |

Columns: *Driven-cluster share* = share of clustered core entries sitting in clusters flagged channel/template-driven (social markup ≥50%, a repeated template ≥5 and ≥30%, or one compilation channel ≥70%); *Effective substantive coverage* = coverage × (1 − driven share), a heuristic that rewards covering the core without relying on flagged clusters; *Cohesion* = entry-weighted mean cosine to cluster centroid; *Seed ARI* = pairwise ARI over UMAP seeds 42/43/44 on core points clustered under both. Direct baselines (PCA/k-means) have no seed ARI (single seed) and k-means assigns every point, so its noise is 0 by construction.

## 3. Cross-experiment agreement (ARI on core points clustered in both)

| Comparison | ARI | n | share of core |
|---|---:|---:|---:|
| raw__umap_m25s5 vs strip__umap_m25s5 | 0.6172 | 4225 | 0.5266 |
| raw__umap_m50s10 vs strip__umap_m50s10 | 0.8817 | 4468 | 0.5569 |
| strip__umap_m25s5 vs strip_dedup__umap_m25s5 | 0.9232 | 4214 | 0.5252 |
| strip__umap_m25s5 vs no_short__umap_m25s5 | 0.8046 | 3942 | 0.4913 |
| strip__umap_m25s5 vs strip__kmeans_k40 | 0.4462 | 4817 | 0.6004 |
| strip__umap_m25s5 vs strip__pca20_hdb_m25s5 | 0.0296 | 1581 | 0.1971 |
| joint strip__umap_m25s5 vs strip__umap_m25s5_englishonly (English texts only) | 0.8447 | 3421 | 0.4802 |
| joint strip__umap_m25s5 vs strip__umap_m10s5_frenchonly (French texts only) | 0.3338 | 450 | 0.5034 |

## 3b. Cluster-level diagnostics of the run_v1 reference configuration (`raw__umap_m50s10`)

Noise (unassigned): 3,176 of 10,108 entries (31.4%); language {'English': 2835, 'French': 340, 'S': 1}; instrument {'consultation_compilation_2017_18': 1607, 'get_involved_questionnaire_2017_18': 1383, 'individual_comments_2016_18': 186}.

| Cluster | Entries | Share | EN / FR / other | Top dataset | Exact-dup | URL / # / @ | Template | Short | Top channel | Probability | Cohesion | Flags | Proxy coherence | Descriptive label |
|---:|---:|---:|---|---|---:|---|---:|---:|---|---:|---:|---|---|---|
| 9 | 1289 | 12.8% | 1158 / 130 / 1 | get_involved_questionnaire 54% | 0% | 18% / 16% / 18% | 0% | 3% | Twitter 51% | 0.969 | 0.6447 | broad | low_or_mixed | canada, canadians, canadian |
| 14 | 1164 | 11.5% | 1076 / 87 / 1 | consultation_compilation 73% | 1% | 1% / 1% / 2% | 0% | 15% | Commitment Brainstorm 24% | 0.972 | 0.6108 | broad | low_or_mixed | data, open data, open |
| 4 | 624 | 6.2% | 600 / 21 / 3 | consultation_compilation 98% | 0% | 63% / 64% / 68% | 11% | 10% | Twitter 96% | 0.882 | 0.6335 | social, channel, one-instrument | low_or_mixed | opengovcan, https, opengovcan https |
| 15 | 306 | 3.0% | 279 / 27 / 0 | get_involved_questionnaire 55% | 0% | 1% / 1% / 1% | 0% | 6% | Individual Comment 23% | 0.797 | 0.724 | — | moderate | open government, open, government |
| 31 | 287 | 2.8% | 245 / 42 / 0 | get_involved_questionnaire 59% | 1% | 0% / 0% / 0% | 1% | 17% | Commitment Brainstorm 37% | 0.973 | 0.6384 | — | moderate | government, accountability, public |
| 17 | 258 | 2.6% | 192 / 66 / 0 | get_involved_questionnaire 97% | 6% | 1% / 0% / 0% | 2% | 53% | Commitment Brainstorm 25% | 0.768 | 0.7584 | short, one-instrument | moderate | site, website, visited |
| 22 | 219 | 2.2% | 193 / 26 / 0 | get_involved_questionnaire 91% | 7% | 0% / 0% / 0% | 1% | 57% | Commitment Brainstorm 58% | 0.817 | 0.6076 | short | low_or_mixed | don, don know, know |
| 26 | 179 | 1.8% | 171 / 8 / 0 | consultation_compilation 82% | 0% | 1% / 1% / 0% | 1% | 24% | Slido Text 31% | 0.872 | 0.6219 | — | low_or_mixed | engagement, community, groups |
| 25 | 176 | 1.7% | 143 / 33 / 0 | get_involved_questionnaire 81% | 0% | 0% / 0% / 2% | 1% | 18% | Slido Text 25% | 0.921 | 0.5549 | — | low_or_mixed | income, seniors, pension |
| 11 | 169 | 1.7% | 151 / 18 / 0 | consultation_compilation 52% | 3% | 4% / 4% / 4% | 2% | 21% | Slido Text 23% | 0.757 | 0.6877 | — | moderate | transparency, trust, accountability |
| 10 | 161 | 1.6% | 150 / 11 / 0 | consultation_compilation 52% | 1% | 1% / 1% / 0% | 1% | 23% | Slido Text 23% | 0.724 | 0.6395 | — | moderate | privacy, information, disclosure |
| 23 | 159 | 1.6% | 140 / 19 / 0 | get_involved_questionnaire 96% | 8% | 0% / 0% / 0% | 1% | 68% | Commitment Brainstorm 71% | 0.958 | 0.6442 | short, one-instrument | moderate | haven, looked, used |
| 8 | 155 | 1.5% | 143 / 12 / 0 | get_involved_questionnaire 94% | 6% | 0% / 0% / 0% | 2% | 32% | Commitment Brainstorm 50% | 0.724 | 0.8006 | — | moderate | open canada, canada ca, ca |
| 29 | 153 | 1.5% | 118 / 35 / 0 | get_involved_questionnaire 69% | 1% | 0% / 1% / 0% | 1% | 43% | Commitment Brainstorm 42% | 0.668 | 0.5396 | — | low_or_mixed | ideas, direction, improvement |
| 24 | 147 | 1.5% | 130 / 17 / 0 | get_involved_questionnaire 69% | 0% | 1% / 0% / 0% | 1% | 19% | Slido Text 22% | 0.879 | 0.6131 | — | low_or_mixed | tax, spending, taxes |
| 20 | 140 | 1.4% | 117 / 23 / 0 | get_involved_questionnaire 74% | 0% | 1% / 0% / 0% | 1% | 24% | Commitment Brainstorm 67% | 0.819 | 0.6456 | — | moderate | information, difficult, info |
| 30 | 130 | 1.3% | 125 / 5 / 0 | get_involved_questionnaire 63% | 2% | 1% / 0% / 1% | 2% | 15% | Reddit 50% | 0.731 | 0.627 | — | low_or_mixed | vote, electoral, reform |
| 7 | 129 | 1.3% | 121 / 8 / 0 | get_involved_questionnaire 56% | 2% | 1% / 1% / 1% | 2% | 8% | Facilitator Notes 36% | 0.713 | 0.6904 | — | moderate | indigenous, peoples, indigenous peoples |
| 0 | 126 | 1.2% | 94 / 32 / 0 | get_involved_questionnaire 85% | 0% | 1% / 1% / 2% | 1% | 24% | Commitment Brainstorm 32% | 0.746 | 0.6635 | — | moderate | immigration, immigrants, refugee |
| 1 | 112 | 1.1% | 96 / 16 / 0 | consultation_compilation 90% | 0% | 9% / 10% / 9% | 1% | 12% | Slido Text 52% | 0.814 | 0.6834 | — | moderate | women, gender, feminist |
| 21 | 89 | 0.9% | 83 / 6 / 0 | get_involved_questionnaire 100% | 7% | 0% / 0% / 0% | 2% | 51% | —  | 0.85 | 0.7097 | short, one-instrument | moderate | heard, ve heard, heard time |
| 5 | 89 | 0.9% | 72 / 17 / 0 | consultation_compilation 51% | 0% | 1% / 1% / 2% | 1% | 27% | Commitment Brainstorm 38% | 0.967 | 0.6415 | — | moderate | education, youth, students |
| 19 | 89 | 0.9% | 70 / 19 / 0 | get_involved_questionnaire 85% | 0% | 0% / 2% / 0% | 1% | 18% | Event RSVP 2017 25% | 0.891 | 0.6304 | — | moderate | environment, climate, environmental |
| 27 | 81 | 0.8% | 77 / 4 / 0 | consultation_compilation 62% | 0% | 0% / 0% / 0% | 1% | 24% | Commitment Brainstorm 26% | 0.965 | 0.7218 | — | moderate | citizens, citizen, engagement |
| 6 | 76 | 0.8% | 66 / 10 / 0 | get_involved_questionnaire 93% | 0% | 0% / 0% / 0% | 1% | 12% | Slido Questions 50% | 0.855 | 0.6217 | — | low_or_mixed | previous, survey, important |
| 12 | 67 | 0.7% | 62 / 5 / 0 | get_involved_questionnaire 97% | 21% | 0% / 0% / 0% | 4% | 82% | Commitment Brainstorm 50% | 0.998 | 0.7871 | short, one-instrument | moderate | existed, did know, know |
| 16 | 65 | 0.6% | 61 / 4 / 0 | get_involved_questionnaire 55% | 5% | 2% / 0% / 0% | 5% | 28% | Commitment Brainstorm 48% | 0.962 | 0.7099 | — | moderate | information, access, access information |
| 13 | 63 | 0.6% | 52 / 11 / 0 | get_involved_questionnaire 78% | 0% | 0% / 0% / 0% | 2% | 6% | Commitment Brainstorm 79% | 0.993 | 0.713 | — | moderate | websites, information, government websites |
| 18 | 60 | 0.6% | 49 / 11 / 0 | get_involved_questionnaire 90% | 0% | 0% / 0% / 0% | 2% | 65% | Commitment Brainstorm 33% | 0.999 | 0.6234 | short | moderate | learning, learn, new |
| 2 | 60 | 0.6% | 46 / 14 / 0 | get_involved_questionnaire 57% | 0% | 0% / 0% / 0% | 2% | 25% | Commitment Brainstorm 44% | 0.959 | 0.6816 | — | moderate | diversity, minorities, inclusion |
| 3 | 59 | 0.6% | 52 / 7 / 0 | get_involved_questionnaire 71% | 3% | 0% / 0% / 0% | 3% | 19% | Table notes - Flipchart 29% | 0.981 | 0.6766 | — | moderate | nations, premières nations, premières |
| 28 | 51 | 0.5% | 48 / 3 / 0 | consultation_compilation 57% | 0% | 0% / 0% / 2% | 2% | 35% | Commitment Brainstorm 38% | 1.0 | 0.622 | — | low_or_mixed | discussion, sessions, discussions |

Verbatim representatives with `corpus_id` for every cluster of every experiment are in each experiment's `cluster_table.csv` (`representatives` column). Descriptive labels are keyword lists only and are not policy concerns.


## 4. Reading the results

- **Stripping URLs/mentions/hashtags** (`umap_m25s5`): driven-cluster share 13% → 5%, core noise 34% → 40%, clusters 66 → 74, seed ARI 0.7772 → 0.925.
- **Template collapse** (`strip_dedup` vs `strip`, m25s5): clusters 74 → 78, driven share 5% → 6%, core coverage 60% → 57%.
- **Excluding short texts** (`no_short` vs `strip`, m25s5): clusters 74 → 60, driven share 5% → 5%, core noise 40% → 41% (all-entry coverage 62% → 47% because short texts are not clustered).
- **Direct-embedding baselines:** PCA(20-D)+HDBSCAN leaves 74% of the core unassigned with 3 clusters (density clustering in the full embedding space is not usable here); k-means (k=40) assigns every text, so its coverage is 100% by construction and its clusters include texts a density method treats as noise; its agreement with the UMAP partition is ARI 0.4462.
- **Language-separate runs:** English-only 72 clusters (core coverage of English 58%); French-only 26 clusters (core coverage of French 78%). Joint run coverage: English 60%, French 62%.
- **Shortlist for human review** (non-direct, non-subset configurations with mean seed ARI ≥ 0.6, ranked by effective substantive coverage): `raw__umap_m50s10`, `strip__umap_m50s10`. This is a review shortlist, not a validated selection.

## 5. Evidence-linked summaries

Extractive summaries (frequent terms, verbatim central examples with `corpus_id`, language counts, uncertainty flags) for the shortlisted configurations and the run_v1 reference are in `data/full_corpus/experiments_v2/summaries/*.json`; full per-cluster diagnostics are in `<experiment>/cluster_table.csv`. Each summary is assembled from computed fields only; examples are verbatim source texts.

### Sample from `raw__umap_m50s10` (largest 6 clusters)

- **Keyword group (descriptive only, not a validated concern): canada, canadians, canadian** — Cluster 9: 1289 entries (12.8% of eligible). Frequent terms: canada, canadians, canadian, government, https, opengovcan, open, people, country, need. Most central verbatim examples: [corpus_id 3585] “Help @TBS_Canada Shape Canada’s Action plan on Open Government. https://t.co/tDXGheZnAn #opengovcan”; [corpus_id 4592] “Des sujets pour les citoyens Canadians”; [corpus_id 3755] “Help @TBS_Canada Shape Canada’s Action plan on Open Government. https://t.co/psa9iW60Kg”. Language: 1158 English / 130 French / 1 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: very large cluster (>= 8% of eligible entries); large clusters tend to be generic.
- **Keyword group (descriptive only, not a validated concern): data, open data, open** — Cluster 14: 1164 entries (11.5% of eligible). Frequent terms: data, open data, open, données, use, information, access, datasets, science, data open. Most central verbatim examples: [corpus_id 4385] “How can government gather and share examples or uses of open data in action?”; [corpus_id 3291] “How should the government use open data & engagement to foster data literacy and effective use of data?”; [corpus_id 4917] “Priorities for open data”. Language: 1076 English / 87 French / 1 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: very large cluster (>= 8% of eligible entries); large clusters tend to be generic.
- **Keyword group (descriptive only, not a validated concern): opengovcan, https, opengovcan https** — Cluster 4: 624 entries (6.2% of eligible). Frequent terms: opengovcan, https, opengovcan https, opengov, https opengovcan, opendata, gcdigital, opengovpart, thanks, great. Most central verbatim examples: [corpus_id 3682] “#opengovcan #opensource https://t.co/COqkpEgVhs”; [corpus_id 2757] “@OpenGovCan @opengovpart @CivicTechTO @CivicActionGTHA @SamaraCDA https://t.co/vfH4YPnCpP”; [corpus_id 3651] “#OpenGovCan https://t.co/uNzBea49w5”. Language: 600 English / 21 French / 3 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: likely channel/template-driven: social-media markup, one source channel dominates; almost entirely from a single instrument.
- **Keyword group (descriptive only, not a validated concern): open government, open, government** — Cluster 15: 306 entries (3.0% of eligible). Frequent terms: open government, open, government, gov, open gov, gouvernement ouvert, ouvert, gouvernement, government open, public. Most central verbatim examples: [corpus_id 7646] “Just beginning to learn about open government”; [corpus_id 4088] “D’autres sujets reliés à l’idée du gouvernement ouvert qui vous semblent importants”; [corpus_id 3420] “how to do "open government" properly”. Language: 279 English / 27 French / 0 other. Proxy coherence: moderate (descriptive; not validated).
- **Keyword group (descriptive only, not a validated concern): government, accountability, public** — Cluster 31: 287 entries (2.8% of eligible). Frequent terms: government, accountability, public, employees, govt, departments, accountable, gouvernement, service, publique. Most central verbatim examples: [corpus_id 7271] “Un gouvernement responsable et décisif”; [corpus_id 10017] “Maitrise en administration publique”; [corpus_id 9841] “Accountability and Citizens knowing the duties of government officials”. Language: 245 English / 42 French / 0 other. Proxy coherence: moderate (descriptive; not validated).
- **Keyword group (descriptive only, not a validated concern): site, website, visited** — Cluster 17: 258 entries (2.6% of eligible). Frequent terms: site, website, visited, ai, haven, site ai, visited site, know website, existed, encore. Most central verbatim examples: [corpus_id 8949] “I haven't visit the site”; [corpus_id 7583] “I haven't gone to the website”; [corpus_id 9182] “Je n'ai pas encore visité le site”. Language: 192 English / 66 French / 0 other. Proxy coherence: moderate (descriptive; not validated). Uncertainty: median response is under 40 characters; almost entirely from a single instrument.

### Sample from `strip__umap_m50s10` (largest 6 clusters)

- **Keyword group (descriptive only, not a validated concern): data, open data, open** — Cluster 18: 1110 entries (11.0% of eligible). Frequent terms: data, open data, open, information, use, données, access, government, datasets, sets. Most central verbatim examples: [corpus_id 4385] “How can government gather and share examples or uses of open data in action?”; [corpus_id 3291] “How should the government use open data & engagement to foster data literacy and effective use of data?”; [corpus_id 30] “How can government departments / agencies prioritize data / information that is made open?”. Language: 1042 English / 67 French / 1 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: very large cluster (>= 8% of eligible entries); large clusters tend to be generic.
- **Keyword group (descriptive only, not a validated concern): canada, canadians, canadian** — Cluster 6: 1024 entries (10.1% of eligible). Frequent terms: canada, canadians, canadian, government, need, country, people, money, open, make. Most central verbatim examples: [corpus_id 4592] “Des sujets pour les citoyens Canadians”; [corpus_id 5994] “well managed & efficient government operations; transparent accountable and responsive federal government; safe & secure Canada Healthy Canadians”; [corpus_id 7047] “une anlayse rigoureuse des enjeux et des besoins des Canadiens afin d'assurer une mise en oeuvre qui réflète la façon dont nos programmes et services sont utili…”. Language: 911 English / 112 French / 1 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: very large cluster (>= 8% of eligible entries); large clusters tend to be generic.
- **Keyword group (descriptive only, not a validated concern): haven, don, don know** — Cluster 24: 380 entries (3.8% of eligible). Frequent terms: haven, don, don know, know, looked, familiar, comment, opinion, encore, ai. Most central verbatim examples: [corpus_id 9229] “Je n'ai pas encore beaucoup exploré”; [corpus_id 9223] “Je ne suis pas bien informé à ce sujet”; [corpus_id 8622] “Not familiar enough with it yet.”. Language: 337 English / 43 French / 0 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: top keywords appear in few member texts (mean support 0.12); median response is under 40 characters.
- **Keyword group (descriptive only, not a validated concern): open government, open, government** — Cluster 11: 340 entries (3.4% of eligible). Frequent terms: open government, open, government, gouvernement ouvert, open gov, ouvert, gov, gouvernement, government open, public. Most central verbatim examples: [corpus_id 7646] “Just beginning to learn about open government”; [corpus_id 4088] “D’autres sujets reliés à l’idée du gouvernement ouvert qui vous semblent importants”; [corpus_id 7679] “Addresses the main concerns of Open Government”. Language: 308 English / 32 French / 0 other. Proxy coherence: moderate (descriptive; not validated).
- **Keyword group (descriptive only, not a validated concern): site, website, ai** — Cluster 16: 265 entries (2.6% of eligible). Frequent terms: site, website, ai, visited, haven, site ai, visited site, know website, existed, web. Most central verbatim examples: [corpus_id 8949] “I haven't visit the site”; [corpus_id 7583] “I haven't gone to the website”; [corpus_id 9182] “Je n'ai pas encore visité le site”. Language: 197 English / 68 French / 0 other. Proxy coherence: moderate (descriptive; not validated). Uncertainty: median response is under 40 characters; almost entirely from a single instrument.
- **Keyword group (descriptive only, not a validated concern): thanks, great, session** — Cluster 25: 205 entries (2.0% of eligible). Frequent terms: thanks, great, session, awesome, discussion, team, event, presentation, ll, thank. Most central verbatim examples: [corpus_id 3829] “Looking forward to this session #opendata, #opengov, #opengovcan https://t.co/YWpWYC24Bv”; [corpus_id 3858] “Hey #Calgary! Looking forward to talking #OpenGovCan with you - don't forget to sign up for one of our two events this Tuesday! #yyc https://t.co/88VL4SXlvc”; [corpus_id 2895] “@rosh_vr @OpenGovCan Thanks for taking part!”. Language: 198 English / 7 French / 0 other. Proxy coherence: low_or_mixed (descriptive; not validated). Uncertainty: likely channel/template-driven: social-media markup, one source channel dominates; top keywords appear in few member texts (mean support 0.10).

## 6. Resources

Total runtime 265.8 s; peak RSS 488.4 MB; machine: 12 CPUs, 7860 MB RAM (~436 MB free), 10.4 GB disk. UMAP uses exact chunked kNN (see run_v1 report); no new model weights beyond the cached multilingual encoder were loaded, and only changed texts were re-embedded.

## 7. Limitations

- No human labels exist; every figure is a proxy and none shows that a cluster is a coherent policy concern.
- Flags are heuristic thresholds (social markup ≥50%, template ≥5 repeats and ≥30%, channel ≥70%, short median <40 chars, broad ≥8% of entries); they identify clusters needing review, not invalid ones.
- 'Channel-concentrated' can also describe a legitimately distinct engagement activity; it is flagged, not discarded.
- Stripping removes information (hashtags can carry topic); stripped texts shorter than 20 characters are excluded from clustering in the `strip*` and `no_short` variants.
- Template collapse uses a URL/mention/hashtag-stripped exact key; near-duplicates are not collapsed.
- Language subsets use source metadata of each unique text's first occurrence; Arabic/Japanese/'S' entries are not in either language run.
- Only the multilingual encoder was varied in preprocessing; model sensitivity was evaluated in run_v1.
- Seed ARI covers the UMAP seed only; HDBSCAN and k-means are deterministic given the reduced space/seed.

## 8. Reproducibility

```powershell
python scripts\run_clustering_experiments_v2.py run
python scripts\run_clustering_experiments_v2.py report
```

Outputs: `data/full_corpus/experiments_v2/` (one folder per experiment: `entry_assignments.csv`, `cluster_table.csv`, `metrics.json`, `labels_seed*.npy`), `experiments_summary.json`, `summaries/`, `preprocess/`.
