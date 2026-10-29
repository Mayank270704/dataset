# PolicyLens concern discovery report (v1)

**Status:** pipeline implemented and run; outputs are **NOT human-validated**. Generated 2026-10-13T04:03:10+00:00 from `data/full_corpus/concern_discovery_v1/`.

## 1. Scope and interpretation rules

- Corpus: Canadian federal Open Government consultations 2016-18 (English/French). Results do not transfer to, or validate performance on, Indian MCA or any other consultation.
- Clusters are unsupervised groupings of semantically similar texts, not ground-truth concern labels.
- Every metric in this run is an internal proxy (geometry, seed agreement, metadata association); none is evidence of semantic or policy validity. Only completed independent human reviews are such evidence.
- Descriptions are deterministic and extractive: computed term shares and verbatim quotes with source ids. They are not validated because they read plausibly.
- Sentiment (tone) and stance (position) are separate constructs and are not inferred here.

## 2. Inputs and integrity

The pipeline consumes the existing eligible corpus and cached artifacts; it loads no encoder and re-clusters nothing.

| Check | Status | Detail |
|---|---|---|
| input files present | PASSED | run_v1 corpus/embeddings and experiments_v2 preprocessing cache found |
| corpus file hashes | PASSED | corpus_entries.csv and unique_texts.csv match the run_v1 manifest |
| corpus_entries schema | PASSED | 11 required columns present |
| unique_texts schema | PASSED | 8 required columns present |
| corpus_id unique integers | PASSED | 10,108 entries |
| entry -> unique mapping | PASSED | 10,043 unique texts; multiplicities consistent |
| entry text identity | PASSED | every entry text normalises to its unique text |
| source datasets | PASSED | consultation_compilation_2017_18=4832, get_involved_questionnaire_2017_18=4832, individual_comments_2016_18=444 |
| source_record_id metadata | PASSED | no blank values |
| question_slug metadata | PASSED | no blank values |
| language metadata | PASSED | no blank values |
| embeddings | PASSED | 10,043 x 384, finite, L2-normalised |
| corpus counts vs run_v1 manifest | PASSED | 10,108 entries / 10,043 unique texts unchanged |
| stripped embeddings | PASSED | 1,233 texts re-embedded in experiments v2 overlaid on run_v1 vectors |
| labels raw__umap_m25s5 seed 42 | PASSED | 66 clusters; states [-1] |
| labels raw__umap_m25s5 seed 43 | PASSED | 77 clusters; states [-1] |
| labels raw__umap_m25s5 seed 44 | PASSED | 67 clusters; states [-1] |
| labels raw__umap_m50s10 seed 42 | PASSED | 32 clusters; states [-1] |
| labels raw__umap_m50s10 seed 43 | PASSED | 35 clusters; states [-1] |
| labels raw__umap_m50s10 seed 44 | PASSED | 33 clusters; states [-1] |
| labels strip__umap_m25s5 seed 42 | PASSED | 74 clusters; states [-2, -1] |
| labels strip__umap_m25s5 seed 43 | PASSED | 77 clusters; states [-2, -1] |
| labels strip__umap_m25s5 seed 44 | PASSED | 80 clusters; states [-2, -1] |
| labels strip__umap_m50s10 seed 42 | PASSED | 33 clusters; states [-2, -1] |
| labels strip__umap_m50s10 seed 43 | PASSED | 36 clusters; states [-2, -1] |
| labels strip__umap_m50s10 seed 44 | PASSED | 34 clusters; states [-2, -1] |

Corpus definition unchanged: 10,108 eligible entries from the run_v1 manifest (the eligibility rule is documented in `reports/full_corpus_clustering_report_v1.md`). Source ids, original text, language, source dataset and consultation field are carried through to every output row.

## 3. Methods

1. **Duplicates and templates.** Unique texts are linked when their template key matches (URLs, mentions, hashtags, punctuation and case removed) or when they are among each other's 10 nearest neighbours with cosine ≥ 0.92 *and* character 5-gram Jaccard ≥ 0.5 (or containment ≥ 0.9). The lexical condition keeps paraphrases and translations out of duplicate groups. Groups with ≥ 5 entries are *repeated template groups*. Identical text never implies the same author.
2. **Candidate configurations.** The four UMAP+HDBSCAN configurations of experiments v2 with three-seed runs (`raw`/`strip` × `m25s5`/`m50s10`) are re-assessed from their saved labels. Noise is kept as noise: no comment is forced into a cluster.
3. **Per-cluster diagnostics.** Cohesion (cosine to centroid), separation (margin to the nearest other centroid, share of members closer to another centroid, centroid-to-centroid cosine), seed stability (best-match Jaccard of member sets across UMAP seeds 42/43/44), distinct near-duplicate groups, template share, language/instrument/field/channel composition, keyword support, and transparent EN/FR lexicons for no-opinion and promotional responses.
4. **Tiers.** `candidate`: no diagnostic flag; a candidate concern group pending human review; `needs_review`: plausible group with a quality warning (mixed, weak, broad, moderately unstable, language- or channel-concentrated); `campaign_or_template`: dominated by a repeated template/campaign; the concern may be real but rests on few independent texts; `non_concern_pattern`: coherent response pattern that is not a policy concern (no-opinion/awareness answers, social-media promotion or event logistics); `unreliable`: too few distinct texts, unstable across random seeds, or (raw-text clusterings) grouped by social-media markup rather than content; not summarised as a concern. Precedence: unreliable > campaign_or_template > non_concern_pattern > needs_review > candidate. Thresholds are in `config.json`.
5. **Selection.** Primary configuration = among CANDIDATE_CONFIGS (the four UMAP+HDBSCAN configurations of experiments v2 that have three-seed runs) those with minimum pairwise seed ARI >= 0.85; choose the highest share of comparable-core entries in candidate-tier clusters (stable groups with no diagnostic warning); ties broken by mean seed ARI. It selects what is summarised and reviewed; it does not validate it. Revised once, after a development run and before the reported run: the first rule maximised 'usable coverage' (candidate + needs_review + campaign_or_template). It rewarded configurations whose largest clusters are broad catch-all groups (one cluster of 12.8% of entries keyed on 'canada, canadians'), which are not inspectable concern groups. Outcomes under both rules are reported side by side.
6. **Confounds.** Bias-corrected Cramér's V and NMI between cluster labels and language, instrument, question field, compilation channel and repeated-template membership, each against a 200-permutation null.
7. **Descriptions.** Deterministic and extractive. No local abstractive summariser with weights is available (the cache holds only `facebook/bart-large-cnn/config.json`), so the fallback is used by design. Each description states the share of member texts containing up to four c-TF-IDF terms (only terms in ≥ 10% of members) and quotes the first maximal-marginal-relevance representative verbatim with its `corpus_id`. Representatives: MMR (λ = 0.7), one per near-duplicate group, ≥ 40 characters preferred, a language with ≥ 15% of members guaranteed one example. `verify_grounding` re-checks every quote, citation and percentage; the run aborts on any violation. Result: **PASSED (every quote verbatim, every citation a member, every term share recomputed)**.
8. **Human review.** A blinded two-reviewer package (cluster intrusion + coherence; then summary support) with a withheld key. See section 8.

## 4. Duplicates, near-duplicates and templates

| Measure | Value |
|---|---:|
| unique texts | 10,043 |
| groups | 9,757 |
| unique texts in multi text groups | 426 |
| multi text groups | 140 |
| unique texts with exact repeats | 51 |
| repeated template groups | 17 |
| entries in repeated template groups | 167 |
| largest group entries | 24 |

Largest repeated template groups and where the selected configuration put them (cluster id: unique texts; -1 noise, -2 excluded):

- 24 entries / 24 texts, consultation_compilation_2017_18: “@OpenGovCan https://t.co/9H0icZRlmS” → {'-2': 24}
- 22 entries / 22 texts, consultation_compilation_2017_18: “#opengovcan https://t.co/pAmNnFmtC1” → {'-2': 22}
- 14 entries / 12 texts, get_involved_questionnaire_2017_18: “This is the first I've heard of it.” → {'45': 12}
- 14 entries / 7 texts, get_involved_questionnaire_2017_18: “I did not know it existed” → {'35': 7}
- 12 entries / 6 texts, get_involved_questionnaire_2017_18: “Je ne connais pas ce site.” → {'41': 6}
- 10 entries / 10 texts, consultation_compilation_2017_18: “I want to see Canada make a comittment for a publicly accessible registry of beneficial owners of companies” → {'6': 10}

## 5. Configuration comparison (internal proxies only)

| Configuration | Clusters | Noise (pop.) | Seed ARI mean / min | Mean cluster stability | Cohesion | Simplified silhouette | Candidate | Needs review | Campaign | Non-concern | Unreliable | Candidate coverage (core) | Usable coverage (core) | Language V (null p95) | Instrument V | Field V |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `raw__umap_m25s5` | 66 | 31.2% | 0.7772 / 0.7046 | 0.7152 | 0.6692 | 0.2336 | 26 | 18 | 1 | 6 | 15 | 20.6% | 46.9% | 0.146 (0.0616) | 0.4866 | 0.3576 |
| `raw__umap_m50s10` | 32 | 31.4% | 0.9693 / 0.9551 | 0.7881 | 0.6485 | 0.2406 | 12 | 10 | 0 | 6 | 4 | 14.7% | 54.8% | 0.1203 (0.0489) | 0.4456 | 0.3425 |
| `strip__umap_m25s5` **(selected)** | 74 | 37.6% | 0.925 / 0.9045 | 0.763 | 0.6843 | 0.2491 | 30 | 24 | 1 | 10 | 9 | 28.3% | 45.9% | 0.1636 (0.0762) | 0.4442 | 0.3547 |
| `strip__umap_m50s10` | 33 | 36.1% | 0.9575 / 0.94 | 0.8218 | 0.6452 | 0.2407 | 15 | 9 | 0 | 5 | 4 | 18.6% | 52.4% | 0.1177 (0.0632) | 0.4044 | 0.3285 |

Eligible under the stability requirement (min seed ARI ≥ 0.85): `raw__umap_m50s10`, `strip__umap_m25s5`, `strip__umap_m50s10`. Selected: **`strip__umap_m25s5`**. Usable coverage counts entries in candidate, needs_review and campaign_or_template clusters. These numbers rank configurations by internal criteria only.

Outcome under each selection criterion (the revision is described in section 3):

| Criterion | Configuration chosen |
|---|---|
| candidate_coverage_core (current rule) | `strip__umap_m25s5` |
| usable_concern_coverage_core (first rule) | `raw__umap_m50s10` |
| usable_non_broad_coverage_core (sensitivity) | `strip__umap_m25s5` |

### Confound analysis of the selected configuration

| Attribute | n | Cramér's V | Null mean / p95 | Permutation p | NMI |
|---|---:|---:|---|---:|---:|
| language | 6,226 | 0.1636 | 0.0216 / 0.0762 | 0.005 | 0.0141 |
| source dataset | 6,226 | 0.4442 | 0.0147 / 0.0473 | 0.005 | 0.0957 |
| question field | 6,226 | 0.3547 | 0.012 / 0.0352 | 0.005 | 0.1771 |
| repeated template membership | 6,226 | 0.3814 | 0.0174 / 0.0612 | 0.005 | 0.0202 |
| compilation channel | 2,648 | 0.2727 | 0.0191 / 0.0667 | 0.005 | 0.1715 |

Cross-lingual mixing: French is 11.2% of clustered entries. Of the 52 clusters where ≥ 5 French entries would be expected at that base rate, 50 contain French entries and 4 hold less than a quarter of the expected French count; 0 clusters are French-majority. Associations with the permutation p at its floor are statistically distinguishable from chance; a non-zero V with instrument or field is expected because questions differ in topic, so V measures how strongly the partition follows the instrument, not whether that is wrong.

### Embedding-model comparison (run_v1 partitions)

| Partition | Model | Clusters | Noise | French-majority clusters | ≥ 90% French | Language V | Instrument V |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline/u15_m25s5` | sentence-transformers/all-MiniLM-L6-v2 (English) | 68 | 37.1% | 1 | 1 | 0.6911 | 0.4683 |
| `baseline/u15_m50s10` | sentence-transformers/all-MiniLM-L6-v2 (English) | 32 | 33.1% | 1 | 1 | 0.6898 | 0.4752 |
| `multilingual/u15_m25s5` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 66 | 31.2% | 0 | 0 | 0.146 | 0.4866 |
| `multilingual/u15_m50s10` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 32 | 31.4% | 0 | 0 | 0.1203 | 0.4456 |

The English-only encoder separates French comments by language more than the multilingual encoder does; this is the reason the multilingual encoder is used. Agreement between the two models' partitions (ARI 0.27–0.51, run_v1 report) is low, so the grouping is model-dependent.

## 6. Concern groups of `strip__umap_m25s5`

| Tier | Clusters | Entries |
|---|---:|---:|
| unreliable | 9 | 751 |
| campaign_or_template | 1 | 29 |
| non_concern_pattern | 10 | 1,019 |
| needs_review | 24 | 1,757 |
| candidate | 30 | 2,670 |
| noise (unassigned) | — | 3,760 |
| excluded (content-free after stripping) | — | 122 |

All groups, largest first (stability = mean best-match Jaccard across seeds; margin = mean cosine margin to nearest other centroid):

| Cluster | Tier | Entries | Distinct texts | EN/FR | Stability | Cohesion | Margin | Label (top supported terms) | Flags |
|---:|---|---:|---:|---|---:|---:|---:|---|---|
| 30 | candidate | 354 | 348 | 321/33 | 0.7824 | 0.7178 | 0.0939 | open / government | — |
| 16 | unreliable | 290 | 290 | 256/34 | 0.2865 | 0.6932 | 0.0273 | — | unstable |
| 41 | non_concern_pattern | 267 | 216 | 199/68 | 0.9791 | 0.7467 | 0.1547 | site / ai | no_opinion_pattern, single_field, single_instrument |
| 57 | candidate | 251 | 249 | 231/19 | 0.7355 | 0.7042 | 0.0452 | data / government | — |
| 25 | candidate | 205 | 200 | 185/20 | 0.933 | 0.6715 | 0.0673 | transparency / trust | — |
| 72 | needs_review | 198 | 197 | 191/7 | 0.5601 | 0.6588 | 0.0795 | data | moderately_unstable |
| 66 | unreliable | 176 | 175 | 158/18 | 0.3108 | 0.6603 | 0.0469 | — | unstable |
| 12 | non_concern_pattern | 169 | 137 | 155/14 | 0.9674 | 0.7869 | 0.1907 | canada / open | no_opinion_pattern |
| 68 | needs_review | 165 | 164 | 141/24 | 0.6389 | 0.655 | 0.0585 | government / accountability | moderately_unstable, weak_cohesion |
| 62 | non_concern_pattern | 153 | 150 | 147/6 | 0.8396 | 0.5474 | 0.0928 | great / thanks | promotional_pattern, channel_concentrated, mixed, weak_cohesion, single_field |
| 39 | needs_review | 144 | 142 | 119/25 | 0.8302 | 0.6337 | 0.0575 | information / difficult | mixed, weak_cohesion |
| 71 | candidate | 143 | 141 | 137/6 | 0.7499 | 0.714 | 0.0776 | open / data | — |
| 26 | candidate | 128 | 127 | 121/7 | 0.9005 | 0.6583 | 0.0905 | privacy / information | — |
| 5 | candidate | 127 | 127 | 95/32 | 0.9767 | 0.6611 | 0.171 | immigration / immigrants | — |
| 59 | needs_review | 126 | 126 | 93/33 | 0.6371 | 0.5861 | 0.0757 | people / income | moderately_unstable, mixed, weak_cohesion |
| 18 | candidate | 119 | 119 | 113/5 | 0.8273 | 0.7371 | 0.0769 | data / canada | — |
| 56 | candidate | 109 | 107 | 103/6 | 0.7589 | 0.707 | 0.0699 | citizens / engagement | — |
| 45 | non_concern_pattern | 108 | 86 | 102/6 | 0.7252 | 0.6771 | 0.1295 | heard / time | no_opinion_pattern, single_field, single_instrument |
| 9 | candidate | 107 | 106 | 99/8 | 0.9104 | 0.7074 | 0.1047 | indigenous / peoples | — |
| 7 | candidate | 104 | 101 | 88/16 | 0.9535 | 0.6913 | 0.1635 | women / gender | — |
| 37 | needs_review | 101 | 101 | 80/21 | 0.942 | 0.6038 | 0.1585 | environment / climate | mixed, weak_cohesion |
| 15 | candidate | 96 | 96 | 86/10 | 0.7178 | 0.727 | 0.0634 | canadians / canada | — |
| 60 | needs_review | 95 | 95 | 86/9 | 0.784 | 0.6518 | 0.0843 | community / groups | mixed, weak_cohesion |
| 31 | candidate | 88 | 84 | 69/19 | 0.8542 | 0.6714 | 0.1142 | science / open | — |
| 52 | needs_review | 79 | 78 | 67/12 | 0.8232 | 0.6127 | 0.1151 | direction / start | mixed, weak_cohesion |
| 70 | candidate | 78 | 76 | 74/4 | 0.9483 | 0.6758 | 0.1187 | electoral / reform | — |
| 61 | candidate | 70 | 70 | 70/0 | 0.7066 | 0.6674 | 0.0903 | engagement / participation | — |
| 53 | unreliable | 70 | 67 | 64/6 | 0.4874 | 0.6583 | 0.0951 | — | unstable |
| 64 | needs_review | 68 | 68 | 62/6 | 0.8849 | 0.6394 | 0.1711 | spending / debt | weak_cohesion |
| 35 | non_concern_pattern | 67 | 46 | 62/5 | 0.5254 | 0.7871 | 0.2024 | know / existed | moderately_unstable, no_opinion_pattern, mixed, single_field, single_instrument |
| 2 | candidate | 66 | 64 | 59/7 | 0.9923 | 0.6578 | 0.1306 | nations | — |
| 19 | needs_review | 66 | 66 | 57/9 | 0.9632 | 0.6481 | 0.1626 | important / questions | weak_cohesion |
| 63 | needs_review | 65 | 64 | 63/2 | 0.6255 | 0.6259 | 0.0928 | discussion / session | moderately_unstable, mixed, weak_cohesion |
| 32 | needs_review | 64 | 64 | 59/5 | 0.6743 | 0.6125 | 0.0793 | feedback / decision | moderately_unstable, weak_cohesion |
| 34 | candidate | 63 | 60 | 58/5 | 0.8468 | 0.7087 | 0.1285 | information / access | — |
| 73 | needs_review | 63 | 63 | 60/3 | 0.5084 | 0.675 | 0.1396 | data / available | moderately_unstable |
| 54 | non_concern_pattern | 62 | 43 | 58/4 | 0.5016 | 0.745 | 0.114 | don / know | moderately_unstable, no_opinion_pattern |
| 48 | needs_review | 61 | 61 | 41/20 | 0.6897 | 0.5628 | 0.1126 | — | moderately_unstable, mixed, weak_cohesion |
| 29 | needs_review | 60 | 60 | 55/5 | 0.6828 | 0.666 | 0.0635 | digital / government | moderately_unstable |
| 21 | non_concern_pattern | 59 | 52 | 51/8 | 0.9071 | 0.7542 | 0.1221 | canada / plan | promotional_pattern, channel_concentrated, single_field, single_instrument |
| 58 | unreliable | 55 | 55 | 50/5 | 0.2702 | 0.6533 | 0.1584 | — | unstable, weak_cohesion |
| 8 | candidate | 54 | 52 | 41/13 | 0.8966 | 0.6934 | 0.1652 | diversity / inclusion | — |
| 50 | non_concern_pattern | 54 | 52 | 42/12 | 0.6773 | 0.6977 | 0.1039 | used / haven | moderately_unstable, no_opinion_pattern, single_field, single_instrument |
| 17 | candidate | 52 | 52 | 50/2 | 0.9623 | 0.6667 | 0.0948 | online / media | — |
| 24 | candidate | 51 | 50 | 46/5 | 0.8889 | 0.6571 | 0.0722 | information / disclosure | — |
| 65 | candidate | 51 | 51 | 47/4 | 0.8962 | 0.6867 | 0.1562 | tax / dollars | — |
| 49 | non_concern_pattern | 51 | 35 | 51/0 | 0.9579 | 0.8099 | 0.1361 | haven / looked | no_opinion_pattern, single_field, single_instrument |
| 40 | candidate | 51 | 50 | 48/3 | 0.8373 | 0.7099 | 0.1384 | consultation | — |
| 22 | needs_review | 50 | 48 | 48/2 | 0.541 | 0.7909 | 0.0511 | government / open | moderately_unstable |
| 0 | needs_review | 47 | 47 | 31/16 | 0.97 | 0.6507 | 0.1883 | language / official | weak_cohesion |
| 10 | needs_review | 47 | 47 | 46/1 | 0.9056 | 0.6605 | 0.1247 | questions / answer | channel_concentrated |
| 1 | candidate | 47 | 47 | 46/1 | 0.9792 | 0.7 | 0.2343 | phoenix / pay | — |
| 51 | needs_review | 44 | 44 | 31/13 | 0.834 | 0.5809 | 0.0943 | just | mixed, weak_cohesion |
| 3 | needs_review | 42 | 42 | 42/0 | 0.9524 | 0.6555 | 0.1769 | cra / time | weak_cohesion |
| 43 | needs_review | 42 | 42 | 36/6 | 0.6942 | 0.7129 | 0.124 | youth / education | moderately_unstable |
| 13 | candidate | 42 | 42 | 37/5 | 0.9297 | 0.7124 | 0.0791 | african / community | — |
| 47 | candidate | 41 | 41 | 40/1 | 0.9878 | 0.6668 | 0.1278 | new / learn | — |
| 20 | unreliable | 39 | 38 | 38/1 | 0.4129 | 0.7494 | 0.0422 | — | unstable, mixed |
| 14 | needs_review | 35 | 35 | 31/4 | 0.5089 | 0.7249 | 0.1303 | canadian / money | moderately_unstable |
| 55 | unreliable | 34 | 25 | 28/6 | 0.2264 | 0.7249 | 0.0908 | — | unstable, no_opinion_pattern |
| 67 | candidate | 32 | 32 | 32/0 | 0.7877 | 0.7397 | 0.0965 | data / training | — |
| 4 | candidate | 32 | 25 | 29/3 | 1.0 | 0.7868 | 0.2033 | democracy / real | — |
| 23 | needs_review | 32 | 32 | 26/6 | 0.881 | 0.6385 | 0.151 | justice / court | weak_cohesion |
| 44 | needs_review | 32 | 31 | 25/7 | 0.6662 | 0.696 | 0.1404 | education / éducation | moderately_unstable |
| 28 | candidate | 31 | 31 | 26/5 | 0.9375 | 0.7641 | 0.0986 | information / websites | — |
| 38 | needs_review | 31 | 31 | 20/11 | 0.6339 | 0.6908 | 0.181 | trouve / cherche | moderately_unstable |
| 46 | unreliable | 31 | 30 | 29/2 | 0.4688 | 0.6943 | 0.1673 | — | unstable, no_opinion_pattern |
| 6 | campaign_or_template | 29 | 18 | 29/0 | 0.9349 | 0.8263 | 0.1681 | registry / beneficial | template_driven, single_field, single_instrument |
| 33 | non_concern_pattern | 29 | 22 | 29/0 | 1.0 | 0.6306 | 0.1653 | latest / thanks | promotional_pattern, channel_concentrated, weak_cohesion |
| 27 | unreliable | 28 | 28 | 22/6 | 0.431 | 0.745 | 0.0981 | — | unstable |
| 42 | unreliable | 28 | 28 | 27/1 | 0.3636 | 0.6422 | 0.1185 | — | unstable, weak_cohesion |
| 36 | candidate | 27 | 27 | 22/5 | 0.9821 | 0.685 | 0.2339 | marijuana / cannabis | — |
| 11 | candidate | 26 | 26 | 23/2 | 0.9074 | 0.6926 | 0.1822 | open / culture | — |
| 69 | candidate | 25 | 25 | 23/2 | 0.7701 | 0.7114 | 0.1226 | party / minister | — |

### Evidence-backed descriptions (largest candidate and needs-review groups)

- **Cluster 30 — open / government** (candidate, 354 entries). Candidate concern (unvalidated). Member comments most often mention “open” (81%; corpus 15%), “government” (68%; corpus 18%) (share of the 353 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 7646]: “Just beginning to learn about open government”.
  - [corpus_id 4088, consultation_compilation_2017_18, record 3823, comment_question_or_idea, French] “D’autres sujets reliés à l’idée du gouvernement ouvert qui vous semblent importants”
  - [corpus_id 9411, get_involved_questionnaire_2017_18, record 7912, q14, French] “Faute de connaissances appronfondies sur ce qu'est le gouvernement ouvert!”
- **Cluster 57 — data / government** (candidate, 251 entries). Candidate concern (unvalidated). Member comments most often mention “data” (78%; corpus 14%), “government” (41%; corpus 18%), “open” (37%; corpus 15%), “information” (15%; corpus 9%) (share of the 251 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 4385]: “How can government gather and share examples or uses of open data in action?”.
  - [corpus_id 3473, consultation_compilation_2017_18, record 3200, comment_question_or_idea, French] “Sous quels formats seraient rendus accessibles les données gouvernementales aux citoyens?”
  - [corpus_id 4862, consultation_compilation_2017_18, record 4697, comment_question_or_idea, English] “How to better drive use of government data.”
- **Cluster 25 — transparency / trust** (candidate, 205 entries). Candidate concern (unvalidated). Member comments most often mention “transparency” (55%; corpus 2%), “trust” (14%; corpus 1%), “transparent” (11%; corpus 1%), “accountability” (11%; corpus 2%) (share of the 202 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 4795]: “la transparence au niveau des comptes publiques et depense du government... une plus grande transparence et une participation”.
  - [corpus_id 6614, get_involved_questionnaire_2017_18, record 5460, q8, English] “OPINION: Transparency is not only opening the books and showing what we what do, and look what we've done, it's also about having integrity and honesty from everyone involved.”
  - [corpus_id 4686, consultation_compilation_2017_18, record 4484, comment_question_or_idea, English] “Proactive disclosure, transparency, accountability”
- **Cluster 72 — data** (needs_review, 198 entries). Candidate concern with quality warnings (unvalidated). Member comments most often mention “data” (76%; corpus 14%) (share of the 197 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 1830]: “(data) acessibility: make data/info easy to acess”.
  - [corpus_id 1166, consultation_compilation_2017_18, record 795, comment_question_or_idea, English] “Would be helpful to have better descriptions and details on datasets.”
  - [corpus_id 3307, consultation_compilation_2017_18, record 3032, comment_question_or_idea, English] “a process that makes future data better documented and available”
- **Cluster 68 — government / accountability** (needs_review, 165 entries). Candidate concern with quality warnings (unvalidated). Member comments most often mention “government” (35%; corpus 18%), “accountability” (27%; corpus 2%) (share of the 164 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 5930]: “Accountability across different governments (i.e. making former government officials accountable for poor/negligent decisions during their tenure)”.
  - [corpus_id 7195, get_involved_questionnaire_2017_18, record 7787, q8, French] “Des leaders qui mentent moins. Des hauts fonctionnaires imputables. Arrêter des programmes hors de contrôle qui gaspille les fonds publiques( contrôle des armes à feu, systeme de paie Phénix, etc)”
  - [corpus_id 7112, get_involved_questionnaire_2017_18, record 7382, q8, French] “la culture organisationnelle au sein de la fonction publique fédérale et des décisions intelligentes: nous sommes en 2017 = Phénix n'a pas sa place dans une société intelligente.”
- **Cluster 39 — information / difficult** (needs_review, 144 entries). Candidate concern with quality warnings (unvalidated). Member comments most often mention “information” (61%; corpus 9%), “difficult” (12%; corpus 1%) (share of the 144 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 9207]: “Difficile de trouver l'information rechercher”.
  - [corpus_id 7934, get_involved_questionnaire_2017_18, record 2382, q14, English] “Information is still not well-organized/accessible to public”
  - [corpus_id 9236, get_involved_questionnaire_2017_18, record 7089, q14, French] “L'information est difficile à trouver et n'est pas à jour”
- **Cluster 71 — open / data** (candidate, 143 entries). Candidate concern (unvalidated). Member comments most often mention “open” (87%; corpus 15%), “data” (85%; corpus 14%) (share of the 142 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 4517]: “open data and the need for standardization”.
  - [corpus_id 1311, consultation_compilation_2017_18, record 940, comment_question_or_idea, English] “offering tools to facilitate basic use of open data”
  - [corpus_id 4629, consultation_compilation_2017_18, record 4414, comment_question_or_idea, English] “Existing and policies on open data and how they many change given the advance in technology”
- **Cluster 26 — privacy / information** (candidate, 128 entries). Candidate concern (unvalidated). Member comments most often mention “privacy” (52%; corpus 1%), “information” (27%; corpus 9%), “security” (16%; corpus 1%), “personal” (12%; corpus 1%) (share of the 128 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 3481]: “définir les limites de la vie privée vs les données publiques”.
  - [corpus_id 4845, consultation_compilation_2017_18, record 4678, comment_question_or_idea, English] “Safeguard for privacy - where appropriate”
  - [corpus_id 10098, get_involved_questionnaire_2017_18, record 8084, q18, French] “Loi sur l'Accès à l'Information et Loi sur la protection des renseignements personnels”
- **Cluster 5 — immigration / immigrants** (candidate, 127 entries). Candidate concern (unvalidated). Member comments most often mention “immigration” (52%; corpus 1%), “immigrants” (16%; corpus 1%) (share of the 127 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 7221]: “faciliter laloi de limmigration et lobtention du visas”.
  - [corpus_id 9759, get_involved_questionnaire_2017_18, record 4274, q18, English] “Current immigration and settlement system.”
  - [corpus_id 7036, get_involved_questionnaire_2017_18, record 7007, q8, French] “peut être plus d'informations sur immigration”
- **Cluster 59 — people / income** (needs_review, 126 entries). Candidate concern with quality warnings (unvalidated). Member comments most often mention “people” (17%; corpus 6%), “income” (16%; corpus 1%), “help” (11%; corpus 2%) (share of the 126 distinct member texts containing each term; corpus = share of all texts in the clustering population). Most representative comment [corpus_id 2513]: “un revenu minimum garantira une meilleure santé pour tous, abolira le cafouillis de l'éligibilité aux programmes sociaux, favorisera du temps pour l'éducation ou la formation, aidera la persévérance des petits”.
  - [corpus_id 5974, get_involved_questionnaire_2017_18, record 3114, q8, English] “Low income/Disability issues regarding Health, Housing, and Government involvement”
  - [corpus_id 5736, get_involved_questionnaire_2017_18, record 2014, q8, English] “Fair and secure workplace Poverty reduction”

Full descriptions and up to five source-linked examples per cluster: `selected/concern_groups.csv`, `configurations/strip__umap_m25s5/concern_summaries.json`, `configurations/strip__umap_m25s5/representative_examples.csv` (with `text_sha256`).

## 7. Qualitative error analysis

Examples are selected by rule (most negative margins, nearest/farthest noise, least stable clusters); they are illustrations for human inspection, not error rates.

- **campaign_or_template** cluster 6 (29 entries; flags template_driven, single_field, single_instrument): [4006] “We want to see Canada establish a public registry of beneficial owners of companies and properties”; [550] “I think it's crucial Canada create an open registry of beneficial owners!”
- **non_concern_pattern** cluster 41 (267 entries; flags no_opinion_pattern, single_field, single_instrument): [7559] “Have never visited the site; didn't know about it”; [8904] “I did not know that website existed, until now. I will go and check it out.”
- **unreliable** cluster 16 (290 entries; flags unstable): [7286] “favoriser et protéger les canadiens au lieu des immigrants”; [5822] “democracy and efficiency may not work simultaneously. Hope government of Canada can find a balance.”

**Members closer to another cluster's centroid** (possible mixing or boundary cases):

- [1693] in cluster 30 (open / government), nearer to 61 (engagement / participation), margin -0.3423: “Cultural commitment to change - senior level commitment”
- [1794] in cluster 25 (transparency / trust), nearer to 6 (registry / beneficial), margin -0.3058: “beneficial ownership registry”
- [46] in cluster 30 (open / government), nearer to 8 (diversity / inclusion), margin -0.2421: “How to influence cultural change?”
- [6108] in cluster 26 (privacy / information), nearer to 2 (nations), margin -0.2112: “Standing more for human rights around the Globe”
- [8061] in cluster 45 (heard / time), nearer to 41 (site / ai), margin -0.203: “I clicked on a button that asked if I would do a survey AFTER visiting a web page--however the survey launched”

**Noise texts closest to a cluster** (candidates for missed membership):

- [6635] nearest cluster 22 (government / open), cosine 0.8684: “I find this survey stale in its approach and presentation.”
- [4592] nearest cluster 16 ((no label)), cosine 0.8553: “Des sujets pour les citoyens Canadians”
- [4676] nearest cluster 57 (data / government), cosine 0.8532: “Innovation au sein de la fonction publique et l'utilisation des données ouvert dans l'appareil gouvernemental”
- [1383] nearest cluster 22 (government / open), cosine 0.849: “Take a look at the model of Open Government policy-making developed by the Provincial Government in Alberta”
- [6638] nearest cluster 15 (canadians / canada), cosine 0.846: “the government needs to start working for the majority of canadian, we need to look after our seniors raise there”

**Noise texts farthest from every cluster** (genuine outliers or idiosyncratic comments):

- [6987] cosine 0.1989: “Individual grooming of a person”
- [3712] cosine 0.1786: “Why test for Listeria monocytogenes in ready-to-eat (RTE) products?”
- [3159] cosine 0.1537: “@'patlaj @OpenGovCan It's now supported by default in Edge too https://t.co/4CEz8paXwH which should kickstart Chrome”
- [1984] cosine 0.1501: “jon craft watts the mc template asap”
- [1478] cosine 0.1349: “Delete - Zz Xbox zxzzzxz”

Noise texts have median length 103 characters versus 79 for clustered texts.

**Least stable clusters across seeds:** cluster 55 ((no label), unreliable) Jaccard 0.2264; cluster 58 ((no label), unreliable) Jaccard 0.2702; cluster 16 ((no label), unreliable) Jaccard 0.2865.

**Cross-lingual groupings** (most central English and French member of the same group; language is source metadata, so a questionnaire answer written in French in the English-language survey is listed as English):

- cluster 38 (trouve / cherche, French share 0.355): EN [7759] “I am getting most of what I search for.” / FR [9322] “Site détaillé, facile de trouver ce que l'on cherche,”
- cluster 0 (language / official, French share 0.34): EN [890] “How can we make it easier? Need flexiblity on official languages” / FR [413] “offrir ces consultations en francais, ou de moins en format bilingue”
- cluster 48 (None, French share 0.328): EN [8699] “cumbersome hard to understand” / FR [9405] “parfois difficile de trouver des réponses”
- cluster 51 (just, French share 0.295): EN [8548] “je viens de le découvrir” / FR [10066] “oui, sur l'enverennement”

## 8. Human evaluation status

**Internal metrics versus semantic evidence.** Sections 4–7 contain internal evidence only: geometry, seed agreement, metadata association, lexical support and rule-selected examples. They show that the groups are reproducible and where they are confounded. They do not show that any group is a coherent policy concern or that any description is faithful. That requires the human judgments below.

- **Concern-cluster review package (new):** `data/full_corpus/concern_discovery_v1/human_review/`. Task A: 32 intrusion/coherence items (strata {'suitable': 24, 'control': 8}; tiers {'candidate': 12, 'needs_review': 11, 'unreliable': 4, 'non_concern_pattern': 4, 'campaign_or_template': 1}). Task B: 28 summary-support items. Leakage check: PASSED. Templates contain comment text only; rows are shuffled independently per reviewer; the key is in `_key_do_not_share/` and is read only after both reviewers' files validate against the templates.
- **Result: BLOCKED.** Missing: reviewer_1_task_a.csv, reviewer_1_task_b.csv, reviewer_2_task_a.csv, reviewer_2_task_b.csv. No reviewer judgments exist, so **no intrusion accuracy, agreement statistic, coherence rating or summary-faithfulness result is reported**. This is a research limitation, not a result.
- **pilot templates present and blank** (Annotation readiness and human-dependent analyses): PASSED — non-blank annotation cells: 0
- **v2 templates present and blank** (Annotation readiness and human-dependent analyses): PASSED — non-blank annotation cells: 0
- **pilot agreement analysis** (Annotation readiness and human-dependent analyses): BLOCKED — awaiting human annotation: completed annotator_A/B files not present
- **expansion_v2 agreement analysis** (Annotation readiness and human-dependent analyses): BLOCKED — awaiting human annotation: completed annotator_A/B files not present
- **44-pair concern-coherence review** (Annotation readiness and human-dependent analyses): BLOCKED — awaiting two independent bilingual reviewers (human_review/completed/ missing)
- Stance annotation and concern-cluster review measure different constructs; neither is used as a label for the other, and no stance or sentiment is inferred.

## 9. Limitations

- No completed human review exists; tiers and descriptions are unvalidated proxies.
- Noise is substantial (about a third of the clustering population) and noise texts are longer than clustered texts, so long, multi-issue comments are under-represented in the concern groups.
- All groupings depend on one multilingual MiniLM encoder (384-D; texts beyond its maximum length, 0.8% per the run_v1 report, are truncated) and on UMAP/HDBSCAN settings; agreement between encoders is low.
- Seed stability covers UMAP seeds only (three runs). Per-cluster Jaccard with three seeds is a coarse estimate.
- Thresholds for tiers, near-duplicates and lexicons are heuristic. They were set and revised during development on this same corpus (the promotional-pattern rule, a markup-driven flag for raw-text clusterings and the selection criterion were changed after inspecting development runs; see section 3). They were not tuned against human judgments, because none exist. The review package includes control clusters so the flags can be tested against people.
- The no-opinion and promotional lexicons are English/French keyword rules; they miss paraphrases and can match substantive comments that quote such phrases.
- c-TF-IDF terms come from experiments v2 and are bag-of-words; a term's share in members says it is frequent, not that it states the concern.
- Questionnaire language is the survey language selected, not necessarily the comment's language; 6 Other/Unknown entries are kept.
- Entries are not respondents: one respondent may contribute up to three questionnaire entries, and the sources carry no respondent id.
- The corpus is Canadian federal Open Government consultation data (2016-18). Nothing here validates use on Indian MCA consultations or other domains.

## 10. Reproduction

```powershell
python scripts\run_concern_discovery.py run              # ~1-2 min, no model loading
python scripts\run_concern_discovery.py review-package
python scripts\run_concern_discovery.py report
# after reviewers save completed/reviewer_{1,2}_task_{a,b}.csv:
python scripts\run_concern_discovery.py analyze-reviews
python -m pytest tests\test_concern_discovery.py -q
```

Outputs are write-once; `run_manifest.json` records SHA-256 of every input and output. Runtime of this run: 66.6 s.


## 11. Human-evaluation handoff and remaining requirements

Status checked 2026-10-13T04:27:44+00:00. **No completed reviewer file exists; no part of PolicyLens is human-validated.** Every human-evaluation result below is pending.

Reviewer-facing guide: `data/reviewer_handoff/REVIEWER_HANDOFF_GUIDE.md`. Stage folders: `data/reviewer_handoff/packets/stage_N/person_N/` (built by `scripts/human_review_handoff.py`; allow-list only, leak-audited, no answer keys).

### Evidence status: automated verification versus human validation

| Kind of evidence | Status | What it shows |
|---|---|---|
| Automated verification (tests, integrity hashes, grounding checks, leak audits, intake validation) | Done; full test suite and research evaluation pass with no FAILED item | The software, files and packets behave as specified; reviewers can receive and return files safely |
| Internal proxy metrics (sections 4-7: cohesion, stability, confounds, tiers) | Computed | How reproducible and how confounded the groupings are; **not** whether they are coherent policy concerns |
| Human-validated findings (agreement, intrusion accuracy, coherence, summary support, stance labels) | **None. 0 of 10 completed reviewer files exist** | Nothing yet. Each analysis reports BLOCKED until both people's files are accepted |

### Remaining requirements

| Track | What is needed | Completed files (exact paths) | Current status |
|---|---|---|---|
| stance_pilot | 2 independent annotators label 50 pilot comments (stance, concern, requested action) | `data/annotation/completed/pilot_annotator_A_completed.csv`<br>`data/annotation/completed/pilot_annotator_B_completed.csv` | BLOCKED: 2 of 2 completed file(s) missing |
| pair_review_44 | 2 independent bilingual reviewers rate 44 comment pairs (0-3 or U) | `data/comparisons/multilingual_embedding_450/human_review/completed/reviewer_1_completed.csv`<br>`data/comparisons/multilingual_embedding_450/human_review/completed/reviewer_2_completed.csv` | BLOCKED: 2 of 2 completed file(s) missing |
| cluster_task_a | 2 independent bilingual reviewers: intruder + coherence on 32 items | `data/full_corpus/concern_discovery_v1/human_review/completed/reviewer_1_task_a.csv`<br>`data/full_corpus/concern_discovery_v1/human_review/completed/reviewer_2_task_a.csv` | BLOCKED: 2 of 2 completed file(s) missing |
| cluster_task_b | same reviewers, after task A is accepted: summary support on 28 items | `data/full_corpus/concern_discovery_v1/human_review/completed/reviewer_1_task_b.csv`<br>`data/full_corpus/concern_discovery_v1/human_review/completed/reviewer_2_task_b.csv` | BLOCKED: 2 of 2 completed file(s) missing |
| stance_v2 | after pilot calibration (kappa target 0.60) and all concern files: 350 additional comments | `data/annotation/completed/expansion_v2_annotator_A_completed.csv`<br>`data/annotation/completed/expansion_v2_annotator_B_completed.csv` | BLOCKED: 2 of 2 completed file(s) missing |

### Order of work (prevents cross-task contamination)

1. Stage 1, in parallel and independently: stance pilot, 44-pair review, comment-group task A.
2. Stage 2: comment-group task B is released per reviewer only after that reviewer's task A has been accepted (SHA-256 receipt). Five task-A intruder comments reappear in task B next to their own group, so task A must be locked first.
3. Stage 3: only after every stage-1/2 file of both people is accepted, the two annotators may discuss pilot disagreements and calibrate the stance guidelines; then the 350-record stance round is released. Machine-written descriptions (task B) are therefore seen only after each person's own concern judgments are locked; the stance round still follows task B, so its free-text concern field may be anchored by those descriptions (documented residual risk).

### Intake and analysis commands

```powershell
python scripts\human_review_handoff.py status          # read-only: MISSING / VALID / INVALID per file, readiness per track
python scripts\human_review_handoff.py accept --track <track> --person <1|2> --file <returned.csv>
python scripts\human_review_handoff.py accept --track <track> --person <1|2> --file <its completed/ path>   # file already saved in completed/: receipt in place
python scripts\human_review_handoff.py build-packets --stage 2 --person <1|2>   # after that person's task A is accepted
python scripts\human_review_handoff.py build-packets --stage 3                 # after all stage-1/2 files + pilot analysis
python scripts\analyze_annotation_agreement.py --round pilot --check-only      # then without --check-only
python scripts\analyze_concern_review_agreement.py
python scripts\run_concern_discovery.py analyze-reviews
python scripts\analyze_annotation_agreement.py --round expansion_v2 --check-only   # then without --check-only
python scripts\run_research_evaluation.py --no-write
```

### Completed-file requirements and what each analysis does when they are not met

| Analysis | Requires | Missing reviewer | Invalid file |
|---|---|---|---|
| Stance (`analyze_annotation_agreement.py`) | both annotators' files; columns, row ids and order unchanged; `annotator_A`/`annotator_B`; stance in Support/Oppose/Mixed/Neutral/Unclear; verbatim evidence spans | status `missing_files`, exit 2, nothing written | exit 1, errors listed, nothing written |
| 44-pair (`analyze_concern_review_agreement.py`) | both reviewers' files; 44 pair ids; values 0-3/U; pair text and source references unchanged (checked against the catalog since this round) | `no_completed_files`, exit 1, nothing written | `validation_failed`, exit 1, nothing written. Blank ratings are tolerated and reported as `either_blank` (established behaviour); intake rejects them first |
| Comment-group (`run_concern_discovery.py analyze-reviews`) | both reviewers' task A and task B files; columns/items/text unchanged (line-ending differences tolerated); valid values, no blanks | `BLOCKED`, exit 0, answer key not opened, nothing written | `INVALID`, exit 1, answer key not opened |
| Intake (`human_review_handoff.py`) | one file at a time against its own blank template and row order; never opens a key, catalog or model output | `MISSING` (or `INVALID` if an accepted file disappeared) | `INVALID` with the cells to fix; `accept` refuses to store it, to replace an accepted file, or to count one submission twice |

### Reviewer instructions verified

- IDs: every template has unique, non-blank ids; comment-group ids are opaque random tokens; the two reviewers' templates hold the same items in independent orders (comment-group) or the same order (stance, 44-pair).
- No leakage: the leak audit finds no cluster ids, tiers, strata, model names, scores or expected answers in any template or packet; `pair_catalog.csv`, `selection_manifest.json`, `review_package_report.md`, `package_manifest.json` and `_key_do_not_share/` are excluded by allow-list.
- Bilingual handling: reviewers judge each comment in its own language without machine translation; language metadata is the survey language for questionnaire rows; uninterpretable comments get `U` (review tracks) or `Unclear` with `non-English: <language>` (stance), and notes use the tags `[language]`, `[context]`, `[ambiguous]`, `[not a concern]`.
- Context: 44-pair and stance templates include the question prompt; comment-group items deliberately show text only (a prompt would reveal question-driven grouping), so reviewers mark answers that need their prompt as uninterpretable.
- File safety: 8 comment-group and 6 other template comments begin with `@`, and 8 comment-group and 12 other comments contain line breaks; the guide gives spreadsheet import settings that keep them as text, and intake detects formula corruption.

### Operational audit of the handoff (2026-10-13)

Run from the project root in Windows PowerShell. `accept` was exercised in a disposable copy of the project with placeholder
submissions (`U` / `Unclear` only); no reviewer file was written into the project and no rating was created.

| Check | Result |
|---|---|
| `status`, `audit`, `build-packets` on the project | status: 10 of 10 files MISSING, exit 0; audit PASSED; stage 2 and 3 BLOCKED (exit 2); rebuilding stage 1 refused (exit 1) |
| Invalid / incomplete / corrupted submissions (blank template, one blank cell, `#NAME?` cell, edited comment, duplicated row, missing rows, semicolon-delimited, cp1252, truncated, empty) | all INVALID with the cell or reason named; nothing stored, no receipt |
| Wrong person (reviewer 2's task-A file offered as person 1) | **defect found and fixed:** previously accepted; now rejected ("this is person 2's file") because each comment-group reviewer has a distinct row order |
| Same submission for both people | **defect found and fixed:** previously accepted twice; byte-identical submissions are now refused, an id naming the other person is rejected, identical judgments raise a warning |
| Accepted file deleted, then replaced | **defect found and fixed:** previously shown as MISSING and replaceable; now INVALID ("accepted file is missing") and a second acceptance for the same path is refused |
| File saved directly into `completed/` (as the track READMEs instruct) | **defect found and fixed:** could never be receipted, so task B stayed blocked; `accept --file <that completed path>` now validates and receipts it in place |
| Accepted file edited afterwards | INVALID: SHA-256 differs from the receipt (or structure check fails) |
| Task B release | BLOCKED until that reviewer's task A is accepted; then a 3-file packet (handoff guide, task-B template, comment-group review guide) passes the leak audit |
| Receipt log | `data/reviewer_handoff/receipts.jsonl` is append-only by convention, not cryptographically protected; keep a backup copy outside the project |
| CLI root handling | **defect found and fixed:** `status`/`audit` ignored a non-default project root; now explicit |
| Documented analysis commands and flags | match the scripts (`--round pilot|expansion_v2`, `--check-only`, `analyze-reviews`, `--no-write`; the 44-pair script takes no arguments) |

### Known limitations of the human-evaluation design

- In task A the intruder is the longest of the six comments in 10 of 32 items (about 5 expected by chance). Intrusion accuracy should be reported alongside accuracy on items where the intruder is not the longest; this is disclosed here and not to reviewers.
- 4 to 12 comments overlap between pairs of tracks; seeing a comment twice is unavoidable with one corpus and is mitigated only by the stage order.
- Two reviewers per track give agreement estimates with wide confidence intervals; results describe these samples, not the population.

