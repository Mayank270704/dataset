# Stance review queue: concise annotation guide

This queue is for independent review of existing original consultation records. The current reference label is shown for traceability only; do not treat it or either model prediction as an answer key. Preserve record text and identifiers. Reviewer fields are intentionally blank. Reviewers label independently before adjudication.

Follow `reports/annotation_guidelines_v1.1.md`. Judge the author's position toward the Open Government initiative, transparency/accountability/participation goals, or a specific OG policy. Sentiment alone is not stance. The model schema remains **Support**, **Neutral**, and **Unclear**. The broader guideline also describes Oppose and Mixed; this workflow does not add those labels to the model. If a record appears to reject OG or express both support and rejection, explain that in reviewer notes and request schema-owner adjudication rather than forcing a model label.

- **Support:** endorses an OG goal or policy, or asks for more/better openness, transparency, accountability, or participation. Criticism of delivery can still be Support when it calls for improvement.
- **Neutral:** understandable and relevant, but takes no position toward OG itself: facts, questions, process feedback, or service feedback without an OG position.
- **Unclear:** use only for guideline-defined cases below, not as a general uncertainty bucket.

## Unclear reason codes

Record one or more reasons in reviewer notes and cite the evidence. Proposed codes only surface hints already present in record notes; `review_reason_unrecorded` means the reviewer must decide.

- `insufficient_context`: an unseen statement, prompt, or antecedent is needed; context does not establish the author's position.
- `genuine_ambiguity`: two plausible readings would change the stance label; state both readings.
- `fragment`: incomplete phrase/list that does not establish a position or target. Short but clear endorsements are not fragments.
- `off_topic`: wholly unrelated to government policy, transparency, or public services.
- `language_comprehension_protocol`: only if the assigned reviewer cannot reliably read the language. The guideline requires a French-fluent reviewer for French text. An annotator unable to read it records Unclear and `non-English: French`; this is reviewer inability, not text-inferable stance. Do not use machine translation or guess.
- `unintelligible_or_no_content`: no interpretable usable content, consistent with the guideline.

French records are prioritized, but are not automatically assigned a language-comprehension reason. Distinguish language inability from genuine ambiguity in text the reviewer can read.

## Prediction signals

Production and candidate predictions are majority votes over the three repeated source/duplicate-grouped out-of-fold predictions. Disagreement and prior fold error are review-priority signals, not evidence that the reference label is wrong. The candidate vote comes from the fold-local v1 recipe, not the full-data staging artifact. The saved predictions contain no calibrated confidence scores; no probability or raw SVM margin is supplied.

Keep record/source IDs, language, text, and current reference label unchanged. Enter independent decisions only in the reviewer fields; fill adjudication after disagreements are resolved. Never copy model predictions into reviewer fields.
