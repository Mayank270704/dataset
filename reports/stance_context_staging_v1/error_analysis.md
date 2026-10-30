# Context stance staging: error analysis and abstention

Analysis uses the existing repeated grouped out-of-fold predictions (1197 prediction events across 399 records and three repeats; predictions SHA-256 `935f395c5eb14c4790b95aed558166596287e5c67e2c1fcf4b8b22bd044dfc07`). These are development estimates against unverified repository reference labels. No reference label was changed.

## Class errors

False positive means predicted class differs from the reference class; false negative means reference class differs from the prediction.

| Class | False positives | False negatives |
|---|---:|---:|
| Support | 164 | 184 |
| Neutral | 210 | 129 |
| Unclear | 42 | 103 |

The row/column confusion matrix in Support, Neutral, Unclear order is `[[236, 157, 27], [114, 528, 15], [50, 53, 17]]`.

| Reference | Predicted | Error events | Unique records |
|---|---|---:|---:|
| Support | Neutral | 157 | 63 |
| Support | Unclear | 27 | 13 |
| Neutral | Support | 114 | 56 |
| Neutral | Unclear | 15 | 9 |
| Unclear | Support | 50 | 20 |
| Unclear | Neutral | 53 | 21 |

## Linguistic/context review (hypotheses)

Context was present for all modeled rows. It generally identifies prompt, theme, event, or response type; for short text it often does not reveal whether the author endorses a position. The patterns below are qualitative hypotheses, not adjudicated causes.

### Support reference predicted Neutral

Support reference comments often phrase a recommendation, question, or criticism without a direct endorsement. The model appears to prefer descriptive or interrogative wording; question/context sometimes frames the comment as an open idea rather than an explicit stance.

### Support reference predicted Unclear

Support reference comments can be short suggestions or proposals with little first-person or endorsement wording. The topic context helps identify subject matter but may not show whether a fragment is intended as support.

### Neutral reference predicted Support

Neutral references include constructive proposals, words such as 'need', 'better', or 'support', and questions. Lexical positivity or recommendation language may be mistaken for taking a side; the prompt often invites ideas rather than a position.

### Neutral reference predicted Unclear

Some references are terse, tangential, or contain uncertainty. In these cases Unclear may be linguistically plausible, though it disagrees with the reference label; broader context does not always clarify the author's position.

### Unclear reference predicted Support

The 20 unique affected records include endorsements of third-party statements, fragments, topic labels, retweets/links, and recommendations. Positive or action-oriented tokens can look supportive even when the author position is not recoverable. Context often identifies a brainstorming topic but does not establish authorship or endorsement.

### Unclear reference predicted Neutral

The 21 unique affected records include topic fragments, unrelated/off-topic text, explicit uncertainty, and mixed or critical statements. The model often defaults to Neutral for non-directional wording; context may explain the topic but cannot resolve missing or ambiguous author intent.

## Unclear predicted Support or Neutral

The pooled matrix has 50 Unclear-to-Support and 53 Unclear-to-Neutral error events. These correspond to 20 and 21 unique records respectively; repeated folds account for multiple events per record, and some records switch between these predictions. Their record IDs are listed for audit; none were relabeled.

- Unclear to Support IDs: expansion_v2_010, expansion_v2_037, expansion_v2_038, expansion_v2_045, expansion_v2_046, expansion_v2_060, expansion_v2_064, expansion_v2_068, expansion_v2_078, expansion_v2_080, expansion_v2_145, expansion_v2_148, expansion_v2_155, expansion_v2_204, expansion_v2_290, expansion_v2_337, pilot_002, pilot_017, pilot_021, pilot_040
- Unclear to Neutral IDs: expansion_v2_025, expansion_v2_028, expansion_v2_032, expansion_v2_084, expansion_v2_132, expansion_v2_135, expansion_v2_145, expansion_v2_155, expansion_v2_161, expansion_v2_168, expansion_v2_218, expansion_v2_236, expansion_v2_243, expansion_v2_278, expansion_v2_279, expansion_v2_313, pilot_015, pilot_021, pilot_027, pilot_030, pilot_032

Among the 40 Unclear records, the model produced the same prediction in all three repeats for 30 records; ten had predictions that varied across repeats. This instability is consistent with weak margins or fold-sensitive wording, but is not proof of a particular cause.

Input question/context fields were missing on 0 of the 399 modeled rows. That does not rule out insufficient evidence inside a non-empty prompt. Potential causes inspected include fragments or unrelated content (insufficient evidence), mixed or vague wording (ambiguity), prompt framing that invites ideas/questions without specifying a stance (framing), and fold-dependent predictions (instability). These remain hypotheses; no single cause can be assigned from this evaluation.

## Grouped-validation abstention experiment

For each outer fold, select a top-two LinearSVC margin threshold on three-fold grouped validation within the outer training records; maximize accepted validation accuracy subject to at least 50% validation coverage. Apply only that frozen threshold to the outer validation fold.

Outer grouped validation: accepted 652 / 1197 (54.5%) with accepted-prediction accuracy 0.7255. This is selective accuracy among accepted cases, not overall accuracy.

| Reference class | Accepted | Available | Class coverage |
|---|---:|---:|---:|
| Support | 208 | 420 | 49.5% |
| Neutral | 380 | 657 | 57.8% |
| Unclear | 64 | 120 | 53.3% |

Accepted-only confusion matrix: `[[116, 89, 3], [24, 355, 1], [22, 40, 2]]`.

The accepted-only Unclear row is [22, 40, 2]; only 2 of 64 accepted Unclear references were predicted Unclear. The policy improves accepted-case accuracy mainly on Support and Neutral and does not make Unclear reliable.

Selective accuracy is conditional on accepted predictions and is not overall accuracy. Top-two LinearSVC margins are uncalibrated, fold-specific ranking signals. This is an analysis-only policy and is not enabled in the staging route. The threshold uses grouped validation only; outer grouped folds provide held-out evaluation and were not used to select the threshold.

Reproduce: `python scripts/analyze_stance_context_staging_v1.py`
