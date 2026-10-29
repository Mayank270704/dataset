# PolicyLens Annotation Guidelines v1.0

**Project:** PolicyLens — Government Open-Government Consultation Analysis  
**Document version:** 1.0  
**Prepared:** 2026-10-12  
**Status:** Pilot guideline (pre-adjudication)

---

## 1. Purpose and Scope

This document defines the annotation schema for manually labelling free-text responses
collected during Canada's Open Government consultation process (2016–2018).  
The goal of annotation is to enable supervised and semi-supervised modelling of **citizen
policy stances**, **policy concerns**, and **requested government actions** — without
altering or fabricating any source data.

Annotation applies **only** to pre-selected records in the pilot CSV
(`data/annotation/pilot_annotation_50.csv`).  Source data files are strictly read-only.

---

## 2. Source Fields Selected for Annotation

Three Canadian open-government consultation datasets were audited. The following free-text fields were selected because the
first-stage audit (notebook `01_dataset_audit_and_cleaning.ipynb`) assessed them as
`review_for_policy_concerns_suggestions_or_requested_actions` and they contain substantive
citizen text of sufficient length and non-generic content.

### 2.1 `individual_comments_2016_18`

Collected via index cards at 7 in-person workshops (2016–2018).  Each row represents one
participant at one event.

| Slug | Original question wording | Notes |
|---|---|---|
| `public_engagement_question` | "Public Engagement Question (Also in Compilation)" | Core open-ended question per workshop |
| `practical_idea` | "1 Practical idea (Also in Compilation)" | One actionable idea from participant |
| `og_is_important_because` | "OG is important because / Il est important d'avoir un gouvernement ouvert parce que" | Rationale/value statement |
| `improvements` | "Improvements? / Améliorations" | Suggested improvements to the consultation process |
| `other_comments` | "Other comments / Autres commentaires" | Any additional remarks |

### 2.2 `consultation_compilation_2017_18`

Multi-channel compilation from facilitator notes, social media (Twitter hashtag
`#OpenGovCan`), Reddit, email submissions, and online ideas boards.  Each row is one
discrete comment, idea, or question captured during a consultation event or channel.

| Slug | Original column heading | Notes |
|---|---|---|
| `comment_question_or_idea` | "Comment, Question or Idea (In source language)" | The main substantive text of each record |
| `context_prompt` | "Context/Prompt / Contexte/Message-guide" | The question or activity prompt that elicited the comment |
| `theme` | "Theme / Thème" | Government-assigned thematic category (not annotated by this project) |

### 2.3 `get_involved_questionnaire_2017_18`

Online structured questionnaire (n = 8,212).  Most fields are Likert/checkbox.
The following open-text fields were selected:

| Slug | Original question wording | Notes |
|---|---|---|
| `q2r_other` | "Q2R. Other (please specify)" — transparency priorities | Free-text addition to checkbox list |
| `q4r_other` | "Q4R. Other (please specify)6" — accountability needs | Free-text addition to checkbox list |
| `q6r_other` | "Q6R. Other (please specify)" — participation methods | Free-text addition to checkbox list |
| `q8` | "Q8. Did we miss anything? …tell us about the important aspects of open government that drive your interest" | Main open-ended catch-all |
| `q14` | "Q14. Please tell us why:" — satisfaction with open.canada.ca | Rationale for satisfaction rating |
| `q18` | "Q18. If you have knowledge or experience with an open government topic…" | Expertise and experience sharing |

---

## 3. Annotation Schema and Rules

Every selected pilot record receives **six** annotation fields.
All human annotation columns in the pilot CSV are initially blank. Annotators must enter their judgements manually.
Do not fill in predicted labels, guesses, or automated values.

### 3.1 `stance` — Overall attitude toward Open Government policy or initiative

Allowed values:
- `Support`
- `Oppose`
- `Mixed`
- `Neutral`
- `Unclear`

| Value | Definition |
|---|---|
| `Support` | The respondent expresses general or specific agreement with, enthusiasm for, or endorsement of Open Government initiatives or the government's direction. |
| `Oppose` | The respondent expresses disagreement, skepticism, criticism, or rejection of a specific Open Government policy, initiative, or the consultation itself. |
| `Mixed` | The respondent's text genuinely expresses both supportive and opposing positions (e.g., praising openness in principle while criticizing the current implementation). |
| `Neutral` | The text is relevant but expresses no discernible positive or negative position (states facts, procedural observations, or purely descriptive commentary). |
| `Unclear` | Insufficient context, ambiguity, language limitations, or no defensible interpretation exists. |

**Key Stance Rules:**
- Judge the author's position toward the issue or proposal using the available comment and question context.
- **Do not infer opposition merely from criticism** of execution, nor **support merely from positive sentiment** or polite phrasing.
- Use `Mixed` only when the text genuinely expresses both support and opposition.
- Use `Neutral` when the text is relevant but expresses no discernible position.
- Use `Unclear` when there is insufficient context, ambiguity, language limitations, or no defensible interpretation.
- **Do not force a stance** for irrelevant, non-substantive, or ambiguous responses.
- French-language responses: annotators who cannot reliably interpret French must use `Unclear` rather than guessing, and record this in `review_notes`.

### 3.2 `concern` — Specific policy or procedural issue raised

A concise noun phrase describing the issue raised, preferably no more than 15 words.

- Use `none` when no substantive concern is expressed.
- Use `unclear` when a concern cannot be reliably determined.
- Express in English regardless of source language.
- Summarize the underlying issue concisely; do not copy multi-sentence verbatim text.
- If multiple concerns exist, list the primary one; document secondary concerns in `review_notes`.

### 3.3 `requested_action` — Requested change or government action

A concise verb phrase describing the requested change or action, preferably no more than 20 words.

- Use `none` when no action is requested.
- Use `unclear` when the intended action cannot be determined.
- **Do not invent an action simply because a comment describes a problem.** Only annotate an action if the respondent explicitly or implicitly calls for a change.
- Distinguish process changes (how government consults) from policy changes (what government implements).

### 3.4 `evidence_span` — Exact supporting text

An exact substring copied directly from `original_text` that supports the annotation.

- Maximum 200 characters.
- Must be a verbatim, literal substring of `original_text` (do not paraphrase or modify punctuation).
- If no defensible span exists (e.g., generic response or ambiguous text), leave blank and document why in `review_notes`.

### 3.5 `review_notes` and `annotator_id`

- `review_notes`: Optional human notes explaining ambiguities, secondary concerns, multilingual aspects (e.g., French text), or off-topic content.
- `annotator_id`: Must be entered by the human annotator conducting the review (e.g., `annotator_A`, `annotator_B`). Never pre-populate or automate this field.

---

## 4. Handling Special Cases

### 4.1 Blank or Generic Responses
- If a response consists purely of generic filler (e.g., `none`, `n/a`, `.`, `thanks`), it should have been excluded during sampling. If encountered: `stance = Neutral`, `concern = none`, `requested_action = none`, `evidence_span = (blank)`.

### 4.2 Multilingual Content (French / English)
- Canadian consultations include bilingual English and French submissions.
- Annotators fluent in French should annotate French comments according to the standard definitions, writing `concern` and `requested_action` in English.
- Annotators who cannot reliably interpret French must select `Unclear` for stance and `unclear` for concern/action, noting `"non-English: French"` in `review_notes`. Never use automated machine translation tools or guess.

### 4.3 Off-Topic or Non-Policy Submissions
- If a submission is completely unrelated to government policy, transparency, or public services:
  `stance = Unclear`, `concern = unclear`, `requested_action = unclear`, `evidence_span = (blank)`, with `"off-topic"` noted in `review_notes`.

### 4.4 Repeated Text and Duplicate Groups
- Records sharing a `duplicate_text_group_id` have identical text but may come from different participants.
- Pilot sampling guarantees that at most one record per duplicate text group is included, and no underlying respondent record is sampled more than once.
- Each pilot record must be evaluated independently on its own merits without carrying forward assumptions from other rows.

---

## 5. Sampling Method for the Pilot Dataset

### 5.1 Eligibility Rules
1. Non-blank text with length >= 20 characters after stripping whitespace.
2. Exclude trailing confirmed all-blank rows from `individual_comments_2016_18` (`row_rang_e` is empty).
3. Exclude records with documented `exclusion_reason` in the clean dataset.
4. Exclude normalized generic filler words (`none`, `n/a`, `.`, `thanks`, etc.).
5. Duplicate text group deduplication: at most the first occurrence per group is eligible.
6. Record-level deduplication: avoid selecting the same underlying record (`source_row_number`) more than once.

### 5.2 Allocation Target
- `individual_comments_2016_18`: 20 records (stratified across the 5 policy-relevant free-text fields: 4 each).
- `consultation_compilation_2017_18`: 20 records (stratified across government consultation themes).
- `get_involved_questionnaire_2017_18`: 10 records (stratified across open-ended questions Q8, Q14, Q18: 4, 3, 3).
- Total target: 50 records.

### 5.3 Determinism and Random Seed
- Seed: `42` (`random.Random(42)`).
- Selection is fully deterministic and logged with exact row identifiers in `data/annotation/pilot_sampling_log.md`.

---

## 6. Pilot CSV Schema and Column Mapping

File: `data/annotation/pilot_annotation_50.csv`

The CSV contains both standard provenance fields and established repository column names:

| Pilot Column | Provenance Role | Description |
|---|---|---|
| `pilot_id` | Identifier | Unique pilot identifier (`pilot_001` to `pilot_050`) |
| `source_dataset` | Provenance | Source dataset identifier (`individual_comments_2016_18`, etc.) |
| `source_record_id` | Provenance | Identifier of source record (alias of `source_row_number`) |
| `source_file` | Provenance | Original raw file name |
| `source_row_number` | Provenance | 1-indexed row number in the processed clean CSV |
| `source_field` | Provenance | Field/question slug (alias of `question_slug`) |
| `question_slug` | Provenance | Clean slug of the question/field |
| `question_wording` | Context | Full original question wording as presented to participants |
| `question_context` | Context | Context string (alias of `context`) |
| `context` | Context | Context metadata (event, sector, prompt, theme) |
| `duplicate_text_group_id` | Audit | Duplicate text group identifier from interim review (if any) |
| `original_text` | Source Text | Verbatim comment text exactly as present in the processed file |
| `stance` | Annotation | **Blank** — Human annotation (`Support`, `Oppose`, `Mixed`, `Neutral`, `Unclear`) |
| `concern` | Annotation | **Blank** — Human annotation (concise noun phrase <= 15 words) |
| `requested_action` | Annotation | **Blank** — Human annotation (concise verb phrase <= 20 words) |
| `evidence_span` | Annotation | **Blank** — Human annotation (verbatim substring <= 200 chars) |
| `review_notes` | Annotation | **Blank** — Human annotator optional observations |
| `annotator_id` | Annotation | **Blank** — Human annotator ID entered upon annotation |

---

## 7. Quality Control and Two-Annotator Workflow

### 7.1 Two-Annotator Independent Protocol
To prevent accidental leakage and ensure research integrity:
1. **Separate Template Files:**
   - Annotator 1 works exclusively in [`data/annotation/annotator_1.csv`](file:///c:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/data/annotation/annotator_1.csv).
   - Annotator 2 works exclusively in [`data/annotation/annotator_2.csv`](file:///c:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/data/annotation/annotator_2.csv).
2. **Strict Blind Labeling:**
   - Both annotators must annotate independently without viewing each other's annotations, discussing specific comments, or exchanging files prior to round completion.
   - Annotator identity should be recorded by each annotator in `annotator_id` upon starting.
3. **No Automated or Pre-Filled Labels:**
   - Templates are initialized with blank human annotation columns (`stance`, `concern`, `requested_action`, `evidence_span`, `review_notes`, `annotator_id`).
   - Do not use model predictions, heuristics, or external tools to pre-populate labels.
4. **Inter-Annotator Agreement Analysis:**
   - Once both annotators submit completed files, execute [`notebooks/03_annotation_agreement_analysis.ipynb`](file:///c:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/notebooks/03_annotation_agreement_analysis.ipynb) to validate IDs, check permitted labels, compute Cohen's Kappa ($\kappa$) for `stance`, generate the confusion matrix, and compile the disagreement table.
5. **Target Calibration Threshold:**
   - Inter-annotator agreement target: Cohen's Kappa $\kappa \ge 0.60$ for stance.
   - If $\kappa < 0.60$, annotator guidelines must be calibrated before proceeding to larger-scale annotation.
6. **Adjudication Round:**
   - Flagged disagreements are reviewed in a structured adjudication session.
   - Original annotations in `annotator_1.csv` and `annotator_2.csv` remain unaltered as immutable audit records; resolved gold-standard labels are exported to `data/annotation/pilot_adjudicated_50.csv`.
7. **Verbatim Text Integrity:**
   - Annotators must never modify `original_text`, `pilot_id`, or any provenance column.

---

## 8. Illustrative Examples (Hypothetical Reference)

*(Note: The following examples are hypothetical illustrative examples constructed for annotator guidance and training; they demonstrate the application of schema rules to typical consultation-style feedback.)*

### Example 1 — Support with actionable request
- **Context:** Workshop on open data standards
- **Hypothetical Text:** "Open data needs to be cost-effective. We should mandate standard APIs across all departments."
- **stance:** `Support`
- **concern:** `cost of implementing open data`
- **requested_action:** `mandate standard APIs across all departments`
- **evidence_span:** `mandate standard APIs across all departments`
- **review_notes:** `Supportive of open data overall, with cost concern and specific API action`

### Example 2 — Oppose / Criticism of barriers
- **Context:** Online ideas board regarding government transparency
- **Hypothetical Text:** "The information is stuck in departmental silos between federal and provincial levels; get rid of the silos."
- **stance:** `Oppose`
- **concern:** `data silos between federal and provincial jurisdictions`
- **requested_action:** `eliminate jurisdictional data silos`
- **evidence_span:** `get rid of the silos`
- **review_notes:** `Expresses clear frustration with current fragmented access`

### Example 3 — Indigenous consultation obligation
- **Context:** Catch-all questionnaire question (Q8)
- **Hypothetical Text:** "Honour of the Crown towards Indigenous peoples requires publicizing and respecting the legal duty to consult."
- **stance:** `Support`
- **concern:** `inadequate respect for duty to consult Indigenous peoples`
- **requested_action:** `publicize and respect duty to consult obligations`
- **evidence_span:** `respecting the legal duty to consult`
- **review_notes:** `Policy governance recommendation on treaty obligations`

### Example 4 — Descriptive / Neutral comment
- **Context:** Questionnaire experience sharing (Q18)
- **Hypothetical Text:** "I worked as an open data analyst for the municipal government for three years until 2015."
- **stance:** `Neutral`
- **concern:** `none`
- **requested_action:** `none`
- **evidence_span:** *(blank)*
- **review_notes:** `Purely biographical background; no policy position or request`

### Example 5 — French language entry (Annotator unable to interpret French)
- **Context:** In-person workshop (OG is important because)
- **Hypothetical Text:** "Je m'intéresse aux contextes de développement des données ouvertes et du web sémantique."
- **stance:** `Unclear`
- **concern:** `unclear`
- **requested_action:** `unclear`
- **evidence_span:** *(blank)*
- **review_notes:** `non-English: French (annotator not fluent in French)`

---

## 9. Artifact Directory and Traceability

| Artifact | Location |
|---|---|
| Annotation Guidelines | `reports/annotation_guidelines_v1.md` |
| Pilot Dataset CSV | `data/annotation/pilot_annotation_50.csv` |
| Sampling Log | `data/annotation/pilot_sampling_log.md` |
| Pilot Generation Script | `scripts/create_annotation_pilot.py` |
| Test Suite | `tests/test_annotation_pilot.py` |
