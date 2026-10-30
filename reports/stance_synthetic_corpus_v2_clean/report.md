# Synthetic stance corpus v2: quality-gated generation

The v4 corpus remains unchanged. This version disables unclear templates with Mixed-label risk and excludes annotator-language-comprehension examples from model training.

Generated 360 model-eligible rows (120 each for Support, Neutral and Unclear) with seed 20261029. Exact normalized duplicate count: 0.
Generated 8 separate French protocol scenarios in `language_protocol_examples.jsonl`. They have no `reference_stance`, `model_eligible=false`, and are never included in model training.

Unclear training rows cover only insufficient context, fragments, and off-topic text. Genuine ambiguity templates are disabled until examples can be supplied with explicit, guideline-grounded readings; language-comprehension cases are conditional on annotator ability.

| Class | Rows | Unique texts | Subtypes |
|---|---:|---:|---|
| Neutral | 120 | 120 | no_stance_established |
| Support | 120 | 120 | explicit_stance |
| Unclear | 120 | 120 | fragment, insufficient_context, off_topic |

## Disabled v4 templates

- `unclear_ambiguity_1`: Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.
- `unclear_ambiguity_2`: Could be Support if the caveat is about implementation rather than the stance target.
- `unclear_ambiguity_3`: Directly contradictory support and opposition; no stable intended label is evident.
- `unclear_ambiguity_4`: Semantically awkward wording can imply Support rather than Unclear.
- `unclear_language_fr_1`: Annotator reading ability is not visible in the comment text.
- `unclear_language_fr_2`: Annotator reading ability is not visible in the comment text.
- `unclear_language_fr_3`: Annotator reading ability is not visible in the comment text.
- `unclear_language_fr_4`: Annotator reading ability is not visible in the comment text.

## Row-by-row review of all disabled v4 examples

- `synthetic-20261029-00162` / `unclear_ambiguity_1` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “I support more openness, but publishing service data would go too far.” — Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.
- `synthetic-20261029-00165` / `unclear_language_fr_1` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas lire ce commentaire en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00167` / `unclear_ambiguity_2` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “The proposal sounds right, although I cannot accept it as written.” — Could be Support if the caveat is about implementation rather than the stance target.
- `synthetic-20261029-00170` / `unclear_language_fr_2` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne comprends pas le texte en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00172` / `unclear_ambiguity_3` / Unclear (genuine_ambiguity; precautionarily_disabled): “I can accept the change to public participation, but I do not support it.” — Directly contradictory support and opposition; no stable intended label is evident.
- `synthetic-20261029-00175` / `unclear_language_fr_3` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas évaluer cette réponse écrite en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00177` / `unclear_ambiguity_4` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “More information is welcome, except for the information that should stay public.” — Semantically awkward wording can imply Support rather than Unclear.
- `synthetic-20261029-00180` / `unclear_language_fr_4` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne sais pas lire les commentaires en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00182` / `unclear_ambiguity_1` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “I support more openness, but publishing service data would go too far.” — Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.
- `synthetic-20261029-00185` / `unclear_language_fr_1` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas lire ce commentaire en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00187` / `unclear_ambiguity_2` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “The proposal sounds right, although I cannot accept it as written.” — Could be Support if the caveat is about implementation rather than the stance target.
- `synthetic-20261029-00190` / `unclear_language_fr_2` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne comprends pas le texte en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00192` / `unclear_ambiguity_3` / Unclear (genuine_ambiguity; precautionarily_disabled): “I can accept the change to public data, but I do not support it.” — Directly contradictory support and opposition; no stable intended label is evident.
- `synthetic-20261029-00195` / `unclear_language_fr_3` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas évaluer cette réponse écrite en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00197` / `unclear_ambiguity_4` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “More information is welcome, except for the information that should stay public.” — Semantically awkward wording can imply Support rather than Unclear.
- `synthetic-20261029-00200` / `unclear_language_fr_4` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne sais pas lire les commentaires en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00202` / `unclear_ambiguity_1` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “I support more openness, but publishing contracts would go too far.” — Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.
- `synthetic-20261029-00205` / `unclear_language_fr_1` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas lire ce commentaire en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00207` / `unclear_ambiguity_2` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “The proposal sounds right, although I cannot accept it as written.” — Could be Support if the caveat is about implementation rather than the stance target.
- `synthetic-20261029-00210` / `unclear_language_fr_2` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne comprends pas le texte en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00212` / `unclear_ambiguity_3` / Unclear (genuine_ambiguity; precautionarily_disabled): “I can accept the change to public data, but I do not support it.” — Directly contradictory support and opposition; no stable intended label is evident.
- `synthetic-20261029-00215` / `unclear_language_fr_3` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas évaluer cette réponse écrite en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00217` / `unclear_ambiguity_4` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “More information is welcome, except for the information that should stay public.” — Semantically awkward wording can imply Support rather than Unclear.
- `synthetic-20261029-00220` / `unclear_language_fr_4` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne sais pas lire les commentaires en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00222` / `unclear_ambiguity_1` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “I support more openness, but publishing meeting minutes would go too far.” — Explicitly combines support for openness with rejection of publishing records; this may be Mixed under the guideline.
- `synthetic-20261029-00225` / `unclear_language_fr_1` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas lire ce commentaire en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00227` / `unclear_ambiguity_2` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “The proposal sounds right, although I cannot accept it as written.” — Could be Support if the caveat is about implementation rather than the stance target.
- `synthetic-20261029-00230` / `unclear_language_fr_2` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne comprends pas le texte en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00232` / `unclear_ambiguity_3` / Unclear (genuine_ambiguity; precautionarily_disabled): “I can accept the change to the policy review, but I do not support it.” — Directly contradictory support and opposition; no stable intended label is evident.
- `synthetic-20261029-00235` / `unclear_language_fr_3` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne peux pas évaluer cette réponse écrite en français.” — Annotator reading ability is not visible in the comment text.
- `synthetic-20261029-00237` / `unclear_ambiguity_4` / Unclear (genuine_ambiguity; flagged_in_v4_audit): “More information is welcome, except for the information that should stay public.” — Semantically awkward wording can imply Support rather than Unclear.
- `synthetic-20261029-00240` / `unclear_language_fr_4` / Unclear (language_comprehension_difficulty; flagged_in_v4_audit): “Je ne sais pas lire les commentaires en français.” — Annotator reading ability is not visible in the comment text.

Automated quality checks require complete provenance, the three-class schema, balanced configured counts, unique normalized text, and no explicit Support/Oppose contradiction among Unclear rows. Original reference labels and source mappings were not read or modified by corpus generation.
