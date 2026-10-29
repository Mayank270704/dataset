# PolicyLens Source-Evidence Traceability & Lineage Audit Report (v1)

**Audit Status:** PASSED (100% Grounded)  
**Audit Timestamp:** 2026-10-13T10:23:14+00:00  
**Scope:** End-to-end 4-tier provenance verification across all cited evidence quotes in PolicyLens research outputs.

## 1. Executive Summary

PolicyLens prioritizes policy concerns and collective citizen demands directly from free-text consultation comments. To guarantee that automated clustering and extractive summarization do not hallucinate, misattribute, or alter citizen claims, this audit verifies the complete 4-tier chain of custody from raw government submissions to prioritized evidence quotes.

- **Total Evidence Quotes Audited:** **655 quotes** across 3 evidence tracks:
  - Ranked concern citations: **275**
  - Representative cluster examples: **370**
  - Collective citizen demands: **10**
- **Verbatim Grounding in Raw Government Files:** **590 of 655 (90.08%)**
- **Total Verified Grounded Citations:** **655 of 655 (100.0%)**
- **Ungrounded / Hallucinated Quotes Found:** **0**

## 2. Four-Tier Provenance Architecture

| Tier | Stage | Source File / Artifact | Verification Checked |
|---|---|---|---|
| **Tier 0** | **Raw Submission** | 3 original government CSVs (`ap2016-18`, `ConsultationData2017`, `Get Involved`) | Physical file row, raw column header, uncleaned text, encoding integrity |
| **Tier 1** | **Source Clean** | `data/processed/*_source_clean.csv` | Standardized nulls, whitespace trimming, column slugification, redactions |
| **Tier 2** | **Full Corpus** | `data/full_corpus/run_v1/corpus/corpus_entries.csv` | Length & substantive eligibility, duplicate group assignment, language tag |
| **Tier 3** | **Prioritized Evidence** | `ranking_evidence.csv`, `collective_demand_evidence.csv` | Substring match, cluster membership, multi-signal priority rank |

## 3. Evidence Distribution Across Instruments and Languages

### By Consultation Instrument

| Instrument Dataset | Evidence Quotes | Share of Evidence |
|---|---:|---:|
| `get_involved_questionnaire_2017_18` | 362 | 55.3% |
| `consultation_compilation_2017_18` | 258 | 39.4% |
| `individual_comments_2016_18` | 35 | 5.3% |

### By Consultation Language

| Language | Evidence Quotes | Share |
|---|---:|---:|
| English | 495 | 75.6% |
| French | 158 | 24.1% |
| S | 2 | 0.3% |

### By Match Status

| Match Status | Count | Interpretation |
|---|---:|---|
| `VERBATIM_RAW` | 590 (90.1%) | Exact character-for-character substring of raw uncleaned citizen submission cell. |
| `VERBATIM_CLEAN` | 65 (9.9%) | Exact substring of cleaned source cell (raw cell had outer carriage-returns or trailing spaces). |

## 4. Top Ranked Concern Lineage Trace (Sample)

| Rank | Cluster | Corpus ID | Instrument | Physical Row | Column Header | Verbatim Quote Excerpt | Status |
|---:|---:|---:|---|---:|---|---|:---:|
| 1 | 30 | 7646 | `get_involved_questionnaire_2017_18` | 1291 | Q14. Please tell us why: | “Just beginning to learn about open government” | **VERBATIM_RAW** |
| 1 | 30 | 4088 | `consultation_compilation_2017_18` | 3823 | Comment, Question or Idea… | “D’autres sujets reliés à l’idée du gouverneme…” | **VERBATIM_RAW** |
| 1 | 30 | 9411 | `get_involved_questionnaire_2017_18` | 7912 | Q14. Please tell us why: | “Faute de connaissances appronfondies sur ce q…” | **VERBATIM_RAW** |
| 1 | 30 | 466 | `consultation_compilation_2017_18` | 26 | Comment, Question or Idea… | “How does #mansplanning apply to open governme…” | **VERBATIM_RAW** |
| 1 | 30 | 7679 | `get_involved_questionnaire_2017_18` | 1427 | Q14. Please tell us why: | “Addresses the main concerns of Open Governmen…” | **VERBATIM_RAW** |
| 2 | 25 | 4795 | `consultation_compilation_2017_18` | 4611 | Comment, Question or Idea… | “la transparence au niveau des comptes publiqu…” | **VERBATIM_RAW** |
| 2 | 25 | 6614 | `get_involved_questionnaire_2017_18` | 5460 | Q8. Did we miss anything?… | “OPINION: Transparency is not only opening the…” | **VERBATIM_RAW** |
| 2 | 25 | 4686 | `consultation_compilation_2017_18` | 4484 | Comment, Question or Idea… | “Proactive disclosure, transparency, accountab…” | **VERBATIM_RAW** |
| 2 | 25 | 4376 | `consultation_compilation_2017_18` | 4127 | Comment, Question or Idea… | “What is stopping transparency for becoming th…” | **VERBATIM_RAW** |
| 2 | 25 | 3517 | `consultation_compilation_2017_18` | 3248 | Comment, Question or Idea… | “Agencies that depend on each other should hav…” | **VERBATIM_RAW** |

## 5. Methodological Interpretation & Limitations

- **Zero Hallucination Guarantee:** 100% of cited evidence spans trace to real, physical survey rows submitted during public consultations.
- **Extractive Substrings:** Quotes are strictly extractive spans, not model-generated syntheses or LLM rewrites.
- **Preservation of Context:** The lineage ledger (`evidence_traceability_ledger.csv`) includes both the extracted quote and the full citizen response, enabling human reviewers to verify that quotes are not taken out of context.
- **Scope Limitation:** The Canadian federal Open Government consultation data (2016–2018) does not transfer to Indian MCA or other jurisdictions.
