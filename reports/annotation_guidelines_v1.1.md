# PolicyLens Annotation Guidelines v1.1

**Status:** Draft for the 50-comment pilot. Not yet tested with annotators; agreement results may require a v1.2.
**Relationship to v1:** `reports/annotation_guidelines_v1.md` is unchanged and remains the record of the original schema. v1.1 replaces **only** the stance rules (v1 §3.1) and the worked examples (v1 §8). Source fields, sampling, provenance columns, special cases (v1 §4) and the two-annotator workflow (v1 §7) still apply.

**Schema is unchanged.** Same six annotation columns, same five stance labels: `Support`, `Oppose`, `Mixed`, `Neutral`, `Unclear`. No label has been added or renamed.

---

## 1. What stance is measured toward (the stance target)

> **Stance target:** the author's position on the **Open Government (OG) initiative or policy direction**: the goals and principles of transparency, open data and information, accountability, and public participation in government, **and any specific OG policy or proposal the comment addresses.**

Several other things appear in comments. They are **not** the stance target. Record them in `concern` and `requested_action`; they do not decide `stance` on their own.

| Object the comment is about | Examples | Decides `stance`? | Where it is recorded |
|---|---|---|---|
| OG initiative / policy / proposal | "publish all contracts", "OG is a waste of money", "open data should be the default" | **Yes** | `stance`, plus `concern`/`requested_action` |
| A consultation question or the consultation process | session format, notice, who was invited, "this survey is too long" | No | `concern`, `requested_action` (process change) |
| An implementation issue | portal usability, slow request handling, poor data quality | No (see rule 3 below for the one exception) | `concern`, `requested_action` |
| A government service unrelated to OG | passport wait times, an unrelated program | No | `concern`, `requested_action` |

**Rule of thumb:** ask "what does this person think about government being open, transparent, accountable or participatory, or about the specific OG measure named?". Do not ask "is this person happy?".

## 2. Sentiment is not stance

- **Sentiment** is the emotional tone of the wording: positive (praise, gratitude, enthusiasm), negative (anger, frustration, sarcasm), or neutral. It is a reading aid only. **It is not recorded in any column.**
- **Stance** is the position taken toward the stance target in §1.

They can differ in either direction:

| Tone | Position | Example type |
|---|---|---|
| Negative | `Support` | Angry that government is **not open enough** |
| Positive / polite | `Oppose` | Friendly wording, but argues **against** an OG measure |
| Positive | `Neutral` | Praise for an event or website, no position on OG itself |

**Never choose a stance from tone, politeness, praise or complaint alone.**

## 3. Stance labels

| Label | Use when |
|---|---|
| `Support` | The text endorses an OG goal or specific OG policy, **or asks for more or better openness, transparency, accountability or participation**, even if angry, critical of current practice, or sceptical about delivery. |
| `Oppose` | The text **rejects or argues against** an OG goal or a specific OG policy/proposal, or says government should not pursue it. |
| `Mixed` | The text takes a supporting position **and** an opposing position toward the stance target (for example supports one OG measure but rejects another). |
| `Neutral` | The text is understandable and relevant, but establishes **no position toward the stance target**: facts, questions, experience, process or service feedback, procedural remarks. "Neutral" means "no stance established", **not** "no opinion". |
| `Unclear` | Insufficient context, genuine ambiguity between readings, fragments, off-topic text, or the annotator cannot read the language (§7). |

### Resolution of the `Oppose` contradiction

v1 §3.1 defined `Oppose` as including "skepticism, criticism", while its own rules said not to infer opposition from criticism. **v1.1 resolves this: criticism, frustration, distrust, sarcasm or negative wording alone never imply `Oppose`.**

- Criticism of **how** OG is carried out, or doubt about whether government will deliver, goes in `concern`.
- `Oppose` requires the text to reject or argue against the OG goal or policy itself.
- If a text criticizes execution **and** asks for more openness, that is `Support`.
- If it criticizes execution and expresses **no** position on OG itself, that is `Neutral`.

## 4. Quick decision order

1. **Can you understand the comment, using its question context?** If not: `Unclear`. (Language inability: see §7.)
2. **Is it unrelated to government policy, transparency or public services?** `Unclear`, note "off-topic" (v1 §4.3).
3. **Does it take a position on the stance target (§1)?**
   - Supports and also opposes: `Mixed`
   - Only supports (including demands for more openness): `Support`
   - Only rejects or argues against: `Oppose`
   - No position (facts, questions, process or service feedback): `Neutral`
4. Fill `concern`, `requested_action` and `evidence_span` (§6).

## 5. Handling specific kinds of text

| Situation | Guidance |
|---|---|
| **Question** | A question alone is `Neutral` unless its wording itself expresses a position. `concern`: the topic asked about. `requested_action`: `none` unless the text explicitly asks for something to be done. |
| **Procedural / consultation feedback** (venue, notice, format, time) | Target is the process, not OG: `Neutral`. `concern`: the process issue. `requested_action`: the process change if one is asked for. |
| **Mixed views** | `Mixed` only for supporting **and** opposing positions on the target. Praising OG while criticizing delivery is `Support`, not `Mixed`. Record the secondary concern in `review_notes`. |
| **Unclear intent** | If two readings are both defensible and one would change the label, use `Unclear` and explain in `review_notes`. Do not pick the more likely-sounding reading. |
| **Not enough context** (fragment, reference to an unseen earlier item) | `Unclear`. The question prompt shows what was asked, not what the author thinks, so do not infer a position from the prompt alone. Satisfaction ratings in the questionnaire are **not** shown to you. |
| **Biographical / descriptive** | `Neutral`, `concern`: `none`, `requested_action`: `none`. |
| **Bare complaint about a tool or service** | `Neutral`. It becomes `Support` only if the text itself asks for more or better openness, transparency, accountability or participation. This boundary is where annotators are most likely to disagree; explain close calls in `review_notes`. |

## 6. Other fields (unchanged from v1)

- `concern`: concise English noun phrase, up to 15 words, or `none` / `unclear`. Primary concern only; secondary concerns go in `review_notes`.
- `requested_action`: concise English verb phrase, up to 20 words, or `none` / `unclear`. Do not invent an action because a problem is described. Distinguish process changes from policy changes.
- `evidence_span`: exact substring of `original_text`, up to 200 characters, copied without changes (keep original accents and punctuation). Choose the words that best show the **position toward the stance target**. Leave blank if none exists and say why in `review_notes`.
- `review_notes`, `annotator_id`: see v1 §3.5. Use `annotator_A` or `annotator_B`.

## 7. French and other non-English text

Use the existing v1 §4.2 policy:

- A **French-fluent** annotator labels French comments with the standard rules, writing `concern` and `requested_action` in English.
- An annotator who **cannot reliably read French** must use `Unclear` / `unclear` / `unclear` and write `non-English: French` in `review_notes`. No machine translation and no guessing.

**Do not confuse the two reasons for `Unclear`.**
- Language inability: always include the note `non-English: French`.
- Genuine ambiguity in a comment you can read: use `Unclear` with a note such as `ambiguous: <reason>`, **without** the `non-English` note.

**Limitation:** a label of `Unclear` from language inability says nothing about the comment's stance. If an annotator is not French-fluent, those rows will inflate `Unclear` and disagreement counts. The notes allow them to be separated before interpreting agreement.

## 8. Worked examples

All examples below are **hypothetical**. They were written for training and are **not** taken from, and carry no implied labels for, any consultation record in the pilot or expansion files. No human-verified labels exist yet.

### 8.1 Correction of v1 Example 2

- **Hypothetical text:** "The information is stuck in departmental silos between federal and provincial levels; get rid of the silos."
- **v1 label:** `Oppose`. **v1.1 label: `Support`.**
- **Reasoning:**
  - The text asks for information to be shared across jurisdictions, which is a demand for **more openness**. Under §3 that is `Support` for the OG goal.
  - The negative wording ("stuck", "get rid of") is **sentiment** and describes the current situation. It is not a rejection of OG.
  - `Oppose` in v1 rested on tone, which §2 and §3 now exclude.
- `concern`: `data silos between federal and provincial jurisdictions`. `requested_action`: `eliminate jurisdictional data silos`. `evidence_span`: `get rid of the silos`.
- **Remaining limitation:** if the full source record showed that the author wanted OG abandoned, the label would change. This text alone does not show that.

### 8.2 Paired examples: sentiment and stance differ (hypothetical)

| # | Hypothetical text | Sentiment | Stance | Why |
|---|---|---|---|---|
| A | "It is infuriating that departmental records are not published by default. They must be." | Negative | `Support` | Angry wording, but demands more openness. |
| B | "Thank you for the warm welcome and the great session. Still, government should not publish departmental working data at all; it wastes public money." | Positive | `Oppose` | Polite and grateful, but argues against an OG measure. |
| C | "The workshop was well organized and the facilitators were friendly. It ran 30 minutes over." | Positive | `Neutral` | Process feedback only; no position on OG itself. `concern`: `session ran over planned time`; `requested_action`: `none`. |

### 8.3 Mixed, question and insufficient-context examples (hypothetical)

| Hypothetical text | Stance | Reasoning |
|---|---|---|
| "Publishing departmental datasets is long overdue and I fully support it. But publishing individual-level case records would harm privacy and should not go ahead." | `Mixed` | Supports one OG measure and rejects another. |
| "Will the datasets also be available in French?" | `Neutral` | A question; no position. `concern`: `availability of datasets in French`; `requested_action`: `none`. |
| "Needs more work." | `Unclear` | Cannot tell what is meant or whether it concerns OG. |

### 8.4 Review of the other v1 examples

| v1 example | v1.1 status |
|---|---|
| 1 (APIs, cost concern) | Unchanged: `Support`. |
| 3 (duty to consult) | Retained as `Support`. The text calls for publicizing and respecting a consultation duty, a participation demand. A reasonable annotator could treat the legal duty as outside the OG target; if so, record the reason in `review_notes`. |
| 4 (biographical) | Unchanged: `Neutral`. |
| 5 (French, annotator cannot read it) | Unchanged: `Unclear` with the `non-English: French` note. |

## 9. Workflow reminders

- Two annotators label independently (v1 §7). Do not compare, share or pre-fill labels. Do not use model outputs, cluster assignments or automated tools.
- The agreement target remains κ ≥ 0.60 for `stance`. If it is missed, guidelines are calibrated before the 350-row expansion is annotated.
- `concern` and `requested_action` are free text and are not scored with kappa.
- See `data/annotation/README_annotators.md` for file handling.
