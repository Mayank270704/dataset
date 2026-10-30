# Stance review queue: concise annotation guide

This queue is for independent review of existing original consultation records. The current reference label is shown for traceability only; do not treat it or either model prediction as an answer key. Preserve the record text and identifiers. Reviewer fields are intentionally blank. Reviewers should label independently before adjudication.

## Stance target and labels

Follow `reports/annotation_guidelines_v1.1.md`. Judge the author's position toward the Open Government initiative, transparency/accountability/participation goals, or a specific OG policy. Sentiment alone is not stance. Under the active three-class model schema, the available model labels are **Support**, **Neutral**, and **Unclear**. The broader annotation guideline also describes Oppose and Mixed; this workflow does not add either label to the model schema. If a record appears to reject OG or express both support and rejection, record that in reviewer notes and request schema-owner adjudication rather than forcing it into one of the three model labels.

- **Support:** endorses an OG goal or policy, or asks for more/better openness, transparency, accountability, or participation. Criticism of delivery can still be Support when it calls for improvement.
- **Neutral:** understandable and relevant, but takes no position toward OG itself: facts, questions, process feedback, or service feedback without an OG position.
- **Unclear:** use only for the guideline-defined cases below, not as a general uncertainty bucket.

## Record an Unclear reason separately

Use one or more proposed reason codes in notes, and explain the evidence. These reason codes are for analysis and do not replace the stance label.

- **Insufficient context (`insufficient_context`):** depends on an unseen statement, prompt, or antecedent; context does not establish the author's position.
- **Genuine ambiguity (`genuine_ambiguity`):** two plausible readings of the text would change the stance label. State both readings.
- **Fragment (`fragment`):** incomplete phrase/list that does not establish a position or intended target. Do not label short-but-clear endorsements as fragments.
- **Off-topic (`off_topic`):** wholly unrelated to government policy, transparency, or public services.
- **Language-comprehension protocol (`language_comprehension_protocol`):** only when the assigned reviewer cannot reliably read the language. The guideline requires a French-fluent reviewer for French text; a reviewer unable to read it records Unclear and `non-English: French`. This is a limitation of the reviewer, not a text-inferable stance, and must be distinguished from genuine ambiguity. Do not use machine translation or guess.
- **Unintelligible/no content (`unintelligible_or_no_content`):** text cannot be interpreted or contains no usable content, consistent with the applicable guideline.

The CSV's `proposed_reason_code` only surfaces hints already present in the existing record notes. It may say `review_reason_unrecorded`; reviewers must decide the reason themselves. French-language records are prioritized for review, but are not automatically assigned the language-comprehension reason.

## Review signals and confidence

Production and candidate predictions are majority votes over the three repeated source/duplicate-grouped out-of-fold predictions in the shared-fold report. A disagreement or prior fold error is a review-priority signal, not evidence that the reference label is wrong. Candidate predictions are the fold-local v1 recipe, not direct predictions from the full-data staging artifact. No calibrated confidence score exists in the saved fold predictions, so this queue does not invent one or use a raw SVM margin as a probability.

## Fields and workflow

Keep `record_id`, `source_id`, `language`, `text`, and `current_reference_label` unchanged. Enter independent decisions only in `reviewer_1_label` and `reviewer_2_label`; fill `adjudicated_label` after disagreement resolution. Use `reviewer_notes` for evidence, Unclear reason, ambiguity alternatives, a possible Oppose/Mixed case, or a source/context limitation. Never copy a model prediction into a reviewer field.
