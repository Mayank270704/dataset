# PolicyLens -- Methodology Appendix (v1)

**Status:** Exploratory pipeline completed; outputs NOT human-validated.
**Generated:** 2026-10-13
**Corpus:** Canadian Federal Open Government Consultations 2016-2018 (English / French).
**Scope note:** Every result in this document is derived from the Canadian federal consultation corpus only. No finding transfers to, or validates performance on, any other consultation domain (e.g., Indian MCA).

---

## A. Dataset Description

### A.1 Primary Consultation Datasets

Three datasets were released by the Government of Canada under open-government licences and collected from the Treasury Board of Canada Secretariat's 2016-2018 Open Government consultations. All three are in the project root (raw) and data/processed/ (source-clean copies).

| Dataset | Role | Raw rows | Processed rows | Languages |
|---|:---:|:---:|:---:|---|
| Individual Index Card Comments (p2016-18_individual_comments.csv) | Workshop index cards | 160 | 160 | English / French |
| Consultation Compilation 2017-18 (ConsultationData2017Engagement_...csv) | Multi-channel engagement (Twitter #OpenGovCan, Reddit, workshop facilitation, email) | 5,169 | 5,169 | English / French |
| Get Involved Questionnaire 2017-18 (Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv) | National online questionnaire | 8,213 | 8,212 | English / French |

**Data cleaning.** Each raw file is loaded once and processed by scripts/policylens_government_consultation_audit.py and scripts/comprehensive_data_cleaning_audit.py. Transformations are limited to: whitespace normalization (collapse runs of spaces/tabs, strip leading/trailing whitespace and blank lines), standardizing null representations (N/A, nil, None, empty -> NaN), and column-name slugification (lower-case, ASCII, underscores). No text content is altered, deleted, or translated. Exact transformation provenance is stored in data/processed/*_source_clean.csv and audited in 
eports/data_quality_report.md.

### A.2 Eligible Full-Corpus

Substantive text fields were extracted across all three instruments using scripts/policylens_dataset_audit.py. Each row in the extraction (data/interim/government_response_text_entries.csv) carries the source dataset name, source row number, source field/column header, and original text verbatim. An entry is **eligible** if its text (after stripping URLs, @-mentions, and #-hashtags) is at least 20 characters long and is not a fill-value null. Exact eligibility and deduplication rules are in scripts/run_full_corpus_clustering.py (function is_eligible).

| Corpus metric | Value |
|---|---:|
| Eligible entries (corpus_entries.csv) | 10,108 |
| Unique texts (unique_texts.csv) | 10,043 |
| Source records | 8,414 |

Entries versus unique texts differ because the same text may appear in multiple records (e.g., a Twitter reply quoted in a facilitated workshop). A unique text is the canonical normalized form; its multiplicity is its count of source occurrences. **Multiplicity is not a count of people.**

### A.3 Annotation and Human-Evaluation Pools

Three stratified sampling pools were constructed for human evaluation:

| Pool | Rows | Instrument quotas | Purpose |
|---|:---:|---|---|
| Pilot annotation (data/annotation/pilot_annotation_50.csv) | 50 | Proportional to eligible entries | Stance/concern annotation calibration |
| Expansion pool v2 (data/annotation/expansion_pool_400_v2.csv) | 400 (incl. pilot) | 70 individual comments; 160 compilation; 120 questionnaire | Full stance/concern annotation |
| 44-pair review (data/comparisons/multilingual_embedding_450/human_review/) | 44 pairs | Stratified by cluster tier | Cross-lingual pair coherence |

Stratification and sampling are documented in 
eports/expansion_pool_v2_report.md and 
eports/expansion_pool_v2_methodology_notes.md. Quotas are fixed and do not estimate population proportions; design weights would be required for any prevalence estimate.

---

## B. Full Pipeline Architecture

The pipeline is implemented in five stages, each producing write-once versioned artifacts. No stage modifies an earlier stage's outputs.

`
Stage 1: Sentence Embeddings
  models/semantic_similarity/minilm.py
  data/embeddings/  (450-comment pilot)
  data/full_corpus/run_v1/embeddings/  (full corpus, multilingual encoder)

Stage 2: Unsupervised Clustering Experiments
  scripts/run_full_corpus_clustering.py   -> data/full_corpus/run_v1/
  scripts/run_clustering_experiments_v2.py -> data/full_corpus/experiments_v2/

Stage 3: Concern Discovery
  scripts/run_concern_discovery.py        -> data/full_corpus/concern_discovery_v1/

Stage 4: Concern Prioritization
  scripts/run_concern_prioritization.py   -> data/full_corpus/concern_prioritization_v1/

Stage 5: Source-Evidence Traceability Audit
  scripts/verify_evidence_traceability.py -> evidence_traceability_ledger.csv
                                          -> evidence_traceability_manifest.json
                                          -> reports/source_evidence_traceability_report_v1.md
`

Each stage reads only from prior-stage artifacts (never from intermediate or in-progress outputs), validates input hashes against the prior stage's manifest before proceeding, writes all outputs via a write-once helper (raises FileExistsError on conflict), and records SHA-256 of every output in a 
un_manifest.json.

---

## C. Stage 1 -- Sentence Embeddings

### C.1 Encoder

**Model:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Hugging Face hub; weights cached locally).
**Rationale:** The multilingual encoder is used throughout because the English-only ll-MiniLM-L6-v2 places French and English comments in separate spatial clusters (language Cramer's V approx 0.69 vs. 0.12 for the multilingual model under identical UMAP + HDBSCAN settings), which would conflate language with topic in cluster labels.

### C.2 Encoding

Mean-pooling over token embeddings with attention-mask weighting; L2-normalized to the unit sphere. Batch size 64, maximum sequence length 256 tokens.

### C.3 Full-Corpus Run

The 10,043 unique texts were embedded once; embeddings are cached in data/full_corpus/run_v1/embeddings/multilingual/embeddings.npy (shape 10,043 x 384, float32, L2-normalized). The cache is invalidated if the unique-texts file hash changes. For the experiments-v2 preprocessing variants, only the 1,233 texts that change under URL/mention/hashtag stripping were re-embedded; all others reuse the cached run-v1 vectors.

---

## D. Stage 2 -- Clustering Experiments

### D.1 Dimensionality Reduction (UMAP)

**Library:** umap-learn
**Parameters:**  n_components = 5 (for HDBSCAN input), n_components = 2 (for visualization); n_neighbors in {25, 50}; min_dist in {5, 10} (scaled x 10^-1); metric = cosine; random_state in {42, 43, 44} (three seeds for stability assessment).
**kNN computation:** chunked exact kNN (no approximate search) on the L2-normalized embedding matrix.

### D.2 Density Clustering (HDBSCAN)

**Library:** hdbscan
**Parameters:** min_cluster_size in {25, 50}; min_samples in {5, 10}; cluster_selection_method = eom; metric = euclidean (on the 5-D UMAP projection). Noise points (label -1) are retained and never forced into a cluster.

### D.3 Preprocessing Variants

| Variant | Description | Effect |
|---|---|---|
| 
aw | No preprocessing beyond eligibility | Reference; may group by social-media markup |
| strip | URLs, @-mentions, #-hashtags removed before embedding | Reduces driven-cluster share from 13% to 5% |
| strip_dedup | Texts identical after stripping collapsed to one canonical | Template deflation |
| 
o_short | Texts < 40 characters excluded from clustering | Removes short affirmations |

Full experiment grid and metrics are in 
eports/clustering_experiments_v2_report.md.

### D.4 Evaluation Metrics (Proxy Only)

All metrics are internal proxies. None establishes that a cluster is a coherent policy concern.

| Metric | Definition |
|---|---|
| Cohesion | Entry-weighted mean cosine to cluster centroid (higher = more compact) |
| Coverage | Share of eligible entries assigned to any cluster |
| Seed ARI | Pairwise Adjusted Rand Index between UMAP seeds 42/43/44 on entries clustered in both |
| Driven-cluster share | Share of clustered core entries in clusters flagged as channel/template-driven |
| Effective substantive coverage | coverage x (1 - driven-cluster share) |
| Simplified silhouette | Mean (a - b) / max(a, b) over cluster members; a = within-cluster, b = nearest-cluster distance |

---

## E. Stage 3 -- Concern Discovery

### E.1 Configuration Selection

The four UMAP + HDBSCAN configurations with three-seed runs are assessed. A configuration is eligible if its minimum pairwise seed ARI >= 0.85. Among eligible configurations, the one with the highest share of comparable-core entries in **candidate-tier** clusters is selected. Ties are broken by mean seed ARI.

**Selected configuration:** strip__umap_m25s5 (min seed ARI = 0.90; candidate-core coverage = 28.3%).

### E.2 Diagnostic Tiers

Every cluster is assigned exactly one tier using a precedence rule (higher = overrides lower):

| Tier | Precedence | Definition |
|---|:---:|---|
| unreliable | 5 | < 15 distinct near-duplicate groups, mean seed Jaccard < 0.5, or social-markup share >= 50% |
| campaign_or_template | 4 | A repeated-template group accounts for >= 30% of members and >= 5 independent texts |
| 
on_concern_pattern | 3 | No-opinion/awareness pattern (>= 40% members) or social promotion/event-logistics pattern |
| 
eeds_review | 2 | One or more quality flags: moderately_unstable, mixed, weak_cohesion, broad, channel_concentrated, etc. |
| candidate | 1 | No diagnostic flag; a candidate concern group pending human review |

Thresholds are in data/full_corpus/concern_discovery_v1/config.json.

### E.3 Duplicate and Near-Duplicate Detection

Two texts are near-duplicates if their 5-gram Jaccard similarity >= 0.5 or one contains the other (containment >= 0.9), AND their cosine similarity (multilingual embedding space) >= 0.92. The cosine condition keeps translations and paraphrases out of duplicate groups. Identical text groups with >= 5 entries are flagged as repeated template groups.

### E.4 Extractive Cluster Descriptions

Descriptions are deterministic and extractive (no local abstractive summarizer available). The extractive fallback:
1. **Terms:** c-TF-IDF term frequencies; only terms in >= 10% of member texts are reported.
2. **Representatives:** MMR (lambda = 0.7) selects up to 5; one per near-duplicate group; texts >= 40 chars preferred; language coverage >= 15% guaranteed one example.
3. **Grounding verification:** erify_grounding re-checks every quoted span and every term share; the pipeline aborts on any violation.

**Grounding result:** PASSED (every quote verbatim, every citation a member, every term share recomputed).

### E.5 Summary Statistics

| Configuration | Clusters | Candidate | Needs review | Campaign | Non-concern | Unreliable | Noise |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| strip__umap_m25s5 (selected) | 74 | 30 | 24 | 1 | 10 | 9 | 3,760 entries |

---

## F. Stage 4 -- Concern Prioritization

### F.1 Eligibility for Ranking

A group is ranked if (a) its tier is candidate, 
eeds_review, or campaign_or_template, AND (b) at least one supporting comment verifies verbatim against the original corpus.

### F.2 Multi-Signal Score

Each signal is min-max scaled to [0, 1] across the ranked set; the score is a weighted sum score = sum(w_k * s_k):

| Signal | Definition | Default weight |
|---|---|:---:|
| independent_volume | log(1 + independent texts) / log(1 + max) | 0.40 |
| independence | 1 - duplicate influence | 0.10 |
| dataset_diversity | Normalised Shannon entropy over source datasets | 0.10 |
| ield_diversity | Normalised Shannon entropy over consultation fields | 0.10 |
| 
eliability | Mean best-match Jaccard across UMAP seeds 42/43/44 | 0.15 |
| 	ier_quality | Ordinal map: candidate = 1.0; needs_review = campaign_or_template = 0.5 | 0.05 |
| evidence | Verified supporting comments / 3 (capped at 3) | 0.10 |

Weights are documented judgement calls; they were set before inspecting the ranking and are not fitted to any outcome. Robustness analysis over 21 weight perturbations: minimum Kendall tau = 0.49. Groups in top 10 under every variant: clusters 18, 25, 26, 30, 56.

### F.3 Frequency Baseline

Multi-signal vs. frequency-only baseline: Spearman rho = 0.71, Kendall tau = 0.53.

### F.4 Counting Units

| Unit | What it counts | Total in corpus |
|---|---|:---:|
| Entries | Raw occurrences (non-blank designed free-text responses) | 10,108 |
| Unique texts | Normalised distinct texts | 10,043 |
| Independent texts | Near-duplicate groups; a repeated text counts once | 9,757 |
| Source records | Distinct (dataset, record ID) pairs | 8,414 |

None of these units is a count of respondents.

### F.5 Collective-Demand Candidates

17 repeated-text groups with >= 5 entries screened; 1 states something substantive (calling for a public beneficial-ownership registry; 10 entries from 10 source records). Remaining 16 excluded as no-opinion/promotional wording or content-free after stripping.

---

## G. Stage 5 -- Source-Evidence Traceability

4-tier provenance verified for every cited evidence quote:

| Tier | Stage | Artefact |
|---|---|---|
| 0 | Raw submission | Original government CSV (physical row, column header, uncleaned text) |
| 1 | Source-clean | data/processed/*_source_clean.csv |
| 2 | Full-corpus entry | data/full_corpus/run_v1/corpus/corpus_entries.csv |
| 3 | Prioritized evidence | 
anking_evidence.csv / collective_demand_evidence.csv / 
epresentative_examples.csv |

**655 quotes audited** (275 ranked-concern, 370 representative-example, 10 collective-demand). 100% grounded (0 hallucinated). 590 of 655 (90.1%) are exact substrings of the raw uncleaned submission cell; the remaining 65 (9.9%) are exact substrings of the Tier-1 source-clean cell (raw cell had outer carriage-returns or trailing spaces stripped).

---

## H. Human-Evaluation Protocol

### H.1 Stance Annotation

**Task:** Two independent bilingual annotators label each comment with one of {Support, Oppose, Mixed, Neutral, Unclear} toward Open Government, plus free-text concern and requested-action fields.
**Pool:** 50-comment pilot; 350-comment expansion v2 after calibration (kappa target 0.60).
**Guardrail:** models.stance_classification.validate_labeled_data() halts training if fewer than 10 adjudicated labels or only one class exists.

### H.2 44-Pair Bilingual Review

**Task:** Two independent bilingual reviewers each rate 44 matched comment pairs on a 0-3 scale for semantic coherence, without seeing cluster labels, tier information, or model scores.
**Agreement analysis:** scripts/analyze_concern_review_agreement.py. Outputs: quadratic Cohen's kappa.

### H.3 Cluster Intrusion + Coherence Review (Task A) and Summary-Support Review (Task B)

**Task A:** 32 items; each shows 5 true cluster members + 1 intruder. Reviewer identifies intruder and rates coherence of 5 true members (0-3 scale).
**Task B:** Released per reviewer only after that reviewer's Task A is accepted (SHA-256 receipt). Rates summary support and flags misleading claims.
**Leakage prevention:** Templates contain no cluster IDs, tier names, priority scores, model outputs, or answer keys.

### H.4 Stage Order

1. **Stage 1 (parallel):** Stance pilot, 44-pair review, cluster Task A.
2. **Stage 2 (per person):** Cluster Task B, released only after that person's Task A is accepted.
3. **Stage 3 (after all Stage 1/2 files and pilot analysis):** Pilot calibration; then 350-record stance expansion.

### H.5 File Intake and Receipting

scripts/human_review_handoff.py accept validates: column headers, row count/order, id columns (against blank template SHA-256), formula-injection cells, byte-identity with other person's submission, and SHA-256 receipt stored in data/reviewer_handoff/receipts.jsonl (append-only). Refuses to overwrite an accepted file or accept a second submission for same track/person.

---

## I. Reproducibility

`powershell
# Stage 1: Full-corpus embeddings + run_v1 clustering
python scripts\run_full_corpus_clustering.py run

# Stage 2: Clustering experiments v2
python scripts\run_clustering_experiments_v2.py run
python scripts\run_clustering_experiments_v2.py report

# Stage 3: Concern discovery
python scripts\run_concern_discovery.py run
python scripts\run_concern_discovery.py review-package
python scripts\run_concern_discovery.py report

# Stage 4: Concern prioritization
python scripts\run_concern_prioritization.py run
python scripts\run_concern_prioritization.py report

# Stage 5: Source-evidence traceability
python scripts\verify_evidence_traceability.py

# Consolidated evaluation (read-only, no model runs)
python scripts\run_research_evaluation.py --no-write
python scripts\run_research_evaluation.py

# Full test suite (398+ tests)
python -m pytest tests -q -p no:cacheprovider
`

### I.1 Artifact Manifest Chain

Each stage writes 
un_manifest.json recording SHA-256 hashes of every input and output. The next stage validates its inputs against the prior manifest before proceeding. Any hash mismatch raises CorpusIntegrityError.

### I.2 Seeded Operations

All non-deterministic operations use explicit seeds: UMAP (random_state in {42, 43, 44}), k-means (random_state = 42). Three UMAP seeds are used for every candidate configuration; seed ARI measures partition stability under seed variation.

### I.3 Runtime and Resources

| Stage | Runtime (approx.) | Peak RAM |
|---|:---:|:---:|
| Full-corpus embedding (10,043 texts, GPU not required) | ~12 min | ~2 GB |
| Run v1 clustering | ~25 min | ~2 GB |
| Clustering experiments v2 (16 variants) | ~4.4 min | ~490 MB |
| Concern discovery | ~2 min | ~600 MB |
| Concern prioritization | ~2 s | ~200 MB |
| Source-evidence traceability | ~15 s | ~400 MB |

Machine: 12 CPUs, 7,860 MB RAM, Windows 10.

---

## J. Limitations and Validity Threats

### J.1 Construct Validity

- **Clusters are not validated concerns.** Semantic similarity does not imply thematic coherence, policy relevance, or representativeness. Tiers and diagnostic flags are heuristics, not validity judgments.
- **Prioritization is not importance.** The multi-signal score orders groups for inspection. It does not measure public importance, urgency, policy impact, or representativeness. It was not fitted to any human judgment.
- **Volume is not prevalence.** Entry counts, unique text counts, and source-record counts are not counts of people. Instruments differ in size and design; no design weights are applied.

### J.2 Internal Validity

- **Model dependence.** Agreement between multilingual and English-only encoder partitions is low (ARI 0.27-0.51 in the run-v1 comparison). Encoder choice substantially changes cluster structure.
- **Threshold sensitivity.** Tier boundaries were set before inspecting results. Ranks 6-15 are weight-dependent in the sensitivity analysis.
- **Near-duplicate criteria.** Cosine >= 0.92 threshold chosen to exclude paraphrases; lowering it merges semantically related but independently phrased comments.

### J.3 External Validity

- **Domain.** All findings are specific to the Canadian federal Open Government consultation corpus (2016-2018). No finding transfers to the Indian MCA consultation corpus, other jurisdictions, or other policy domains.
- **Generalization.** Annotation pools are stratified quota samples, not probability samples. Agreement statistics describe the annotators on the sampled records and cannot be generalized to the full corpus without design weights.

### J.4 Known Methodological Limitations

- **Task A intruder length bias.** The intruder is the longest of the six comments in 10 of 32 items (~3.3 expected by chance). Intrusion accuracy should be reported separately for items where the intruder is and is not the longest; this is disclosed in 
eports/concern_discovery_report_v1.md Section 11.
- **Cross-track comment overlap.** 4-12 comments appear in multiple tracks. Seeing a comment twice is unavoidable with a single corpus; the stage order is the only mitigation.
- **Two reviewers.** Two independent reviewers per track give agreement estimates with wide confidence intervals. Results describe these samples only.

---

## K. Cross-Reference Table: Artifact to Script to Report to Test

| Artifact | Generating script | Report | Primary test file |
|---|---|---|---|
| data/full_corpus/run_v1/ | 
un_full_corpus_clustering.py | ull_corpus_clustering_report_v1.md | 	est_full_corpus_pipeline.py |
| data/full_corpus/experiments_v2/ | 
un_clustering_experiments_v2.py | clustering_experiments_v2_report.md | 	est_clustering_experiments_v2.py |
| data/full_corpus/concern_discovery_v1/ | 
un_concern_discovery.py | concern_discovery_report_v1.md | 	est_concern_discovery.py |
| data/full_corpus/concern_prioritization_v1/ | 
un_concern_prioritization.py | concern_prioritization_report_v1.md | 	est_concern_prioritization.py |
| evidence_traceability_ledger.csv | erify_evidence_traceability.py | source_evidence_traceability_report_v1.md | 	est_evidence_traceability.py |
| data/annotation/expansion_pool_400_v2.csv | create_annotation_expansion_pool_v2.py | expansion_pool_v2_report.md | 	est_expansion_pool_v2.py |
| data/reviewer_handoff/ | human_review_handoff.py build-packets | concern_discovery_report_v1.md Section 11 | 	est_human_review_handoff.py |
| Research evaluation | 
un_research_evaluation.py | 
eports/research_evaluation/ | 	est_research_evaluation_entrypoint.py |

**Current evaluation summary:** PASSED=42, BLOCKED=10, NOT RUN=2 (as of 2026-10-13). No FAILED items. All BLOCKED items require human reviewer files not yet returned.
