# V4 synthetic template audit

Review against reports/annotation_guidelines_v1.1.md. This is a rule-based text review, not new human annotation.

| Reference class | Template ID | Rows | Unique texts | Review status | Note |
|---|---|---:|---:|---|---|
| Neutral | neutral_experience | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_fr | 6 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_fr_question | 6 | 3 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_fr_report | 6 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_fr_schedule | 6 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_location | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_process | 7 | 3 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_question | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_question_report | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_report | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_schedule | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Neutral | neutral_service | 7 | 7 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_access | 7 | 6 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_accessible | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_accountability | 7 | 5 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_default | 7 | 5 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_fr | 6 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_fr_accountability | 6 | 3 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_fr_default | 6 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_fr_participation | 6 | 5 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_participation | 7 | 4 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_publish | 7 | 5 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_request | 7 | 6 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Support | support_review | 7 | 6 | guideline_check_pass | Explicit stance or no stance established under the annotation guidelines. |
| Unclear | unclear_ambiguity_1 | 4 | 3 | label_risk | Explicitly supports openness and rejects publication of records; guidelines may classify this as Mixed. |
| Unclear | unclear_ambiguity_2 | 4 | 1 | review | Could be Unclear, but could be Support if the objection concerns implementation only. |
| Unclear | unclear_ambiguity_3 | 4 | 3 | plausible_unclear | Contradictory positions toward the same change make intent indeterminate. |
| Unclear | unclear_ambiguity_4 | 4 | 1 | review | Wording is semantically awkward and could imply a Support stance. |
| Unclear | unclear_context_1 | 4 | 2 | plausible_unclear | Refers to an unseen option or prior discussion; stance target cannot be recovered. |
| Unclear | unclear_context_2 | 4 | 4 | plausible_unclear | Refers to an unseen option or prior discussion; stance target cannot be recovered. |
| Unclear | unclear_context_3 | 4 | 3 | plausible_unclear | Refers to an unseen option or prior discussion; stance target cannot be recovered. |
| Unclear | unclear_context_4 | 4 | 3 | plausible_unclear | Refers to an unseen option or prior discussion; stance target cannot be recovered. |
| Unclear | unclear_fragment_1 | 4 | 3 | plausible_unclear | Incomplete text does not establish a stance. |
| Unclear | unclear_fragment_2 | 4 | 2 | plausible_unclear | Incomplete text does not establish a stance. |
| Unclear | unclear_fragment_3 | 4 | 1 | plausible_unclear | Incomplete text does not establish a stance. |
| Unclear | unclear_fragment_4 | 4 | 1 | plausible_unclear | Incomplete text does not establish a stance. |
| Unclear | unclear_language_fr_1 | 4 | 1 | protocol_only | This label depends on annotator reading ability, which text alone cannot establish. |
| Unclear | unclear_language_fr_2 | 4 | 1 | protocol_only | This label depends on annotator reading ability, which text alone cannot establish. |
| Unclear | unclear_language_fr_3 | 4 | 1 | protocol_only | This label depends on annotator reading ability, which text alone cannot establish. |
| Unclear | unclear_language_fr_4 | 4 | 1 | protocol_only | This label depends on annotator reading ability, which text alone cannot establish. |
| Unclear | unclear_off_topic_1 | 4 | 1 | plausible_unclear | Clearly unrelated to the consultation stance target. |
| Unclear | unclear_off_topic_2 | 4 | 1 | plausible_unclear | Clearly unrelated to the consultation stance target. |
| Unclear | unclear_off_topic_3 | 4 | 1 | plausible_unclear | Clearly unrelated to the consultation stance target. |
| Unclear | unclear_off_topic_4 | 4 | 1 | plausible_unclear | Clearly unrelated to the consultation stance target. |

The `unclear_ambiguity_1` template explicitly combines support for openness with rejection of publication. Under the current guidelines, this may be Mixed rather than Unclear; its four generated copies are suspect synthetic targets. Other genuine-ambiguity templates also merit bilingual annotation review. French-reading difficulty is annotator-specific and cannot be inferred from an ordinary French consultation comment.

Generated text repeats: v4 has 57/80 unique Support texts, 49/80 Neutral, and 35/80 Unclear. Repeated template families are more concentrated in Unclear. The grouped CV evaluates transfer onto original comments that do not share these synthetic template IDs, but cannot establish robustness to new consultation sources.
