"""
Inspection script: sample pilot records from the three government consultation datasets.
Prints representative text content from the selected policy-relevant free-text fields.
"""
import pandas as pd
import random

random.seed(42)

# ─── Load datasets ────────────────────────────────────────────────────────────
ind = pd.read_csv(
    "data/processed/individual_comments_2016_18_source_clean.csv",
    dtype=str, keep_default_na=False,
)
comp = pd.read_csv(
    "data/processed/consultation_compilation_2017_18_source_clean.csv",
    dtype=str, keep_default_na=False,
)
giq = pd.read_csv(
    "data/processed/get_involved_questionnaire_2017_18_source_clean.csv",
    dtype=str, keep_default_na=False,
)

# ─── Exclude confirmed all-blank rows ─────────────────────────────────────────
ind_clean = ind[ind["row_rang_e"].str.strip() != ""].copy()
print(f"individual_comments non-blank rows: {len(ind_clean)}")

# ─── Check compilation free-text columns ──────────────────────────────────────
comp_text_col = "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part"
comp_ctx_col = "context_prompt_contexte_message_guide"
comp_theme_col = "theme_th_me"
print(f"\nconsultation_compilation non-empty comment: {comp[comp[comp_text_col].str.strip() != ''].shape[0]}")
print(f"consultation_compilation non-empty context: {comp[comp[comp_ctx_col].str.strip() != ''].shape[0]}")

# ─── Check questionnaire free-text columns ────────────────────────────────────
q8_col = "q8_did_we_miss_anything_if_the_previous_questions_did_not_speak_to_you_please_tell_us_about_the_important_aspects_of_open_government_that_drive_your_interest_if_you_have_more_than_one_additional_item_please_list_them_in_priority"
q14_col = "q14_please_tell_us_why"
q18_col = "q18_if_you_have_knowledge_or_experience_with_a_open_government_topic_that_you_would_be_willing_to_share_please_tell_us_about_it"

for col, label in [(q8_col, "Q8"), (q14_col, "Q14"), (q18_col, "Q18")]:
    if col in giq.columns:
        non_empty = giq[giq[col].str.strip() != ""].shape[0]
        print(f"get_involved_questionnaire {label}: {non_empty} non-empty")

# ─── Load duplicate text entries to get the list of duplicate source row indices ──
dupe_entries = pd.read_csv(
    "data/interim/government_duplicate_text_entries.csv",
    dtype=str, keep_default_na=False,
)
print(f"\nDuplicate text entries total: {len(dupe_entries)}")
print("Columns:", dupe_entries.columns.tolist())
print(dupe_entries.head(3))
