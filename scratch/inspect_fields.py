import pandas as pd

# Check individual comments column names
ind = pd.read_csv(
    'data/processed/individual_comments_2016_18_source_clean.csv',
    dtype=str,
    keep_default_na=False,
)
print("=== individual_comments columns ===")
for col in ind.columns:
    print(f"  {repr(col)}")
print()

# Count blank vs non-blank rows
blank = ind[ind["row_rang_e"].str.strip() == ""].shape[0]
non_blank = ind[ind["row_rang_e"].str.strip() != ""].shape[0]
print(f"Total rows: {len(ind)}, blank: {blank}, non-blank: {non_blank}")
print()

# Policy-relevant free-text columns
policy_cols = [
    "row_rang_e",
    "event",
    "sector",
    "source_language",
    "public_engagement_question_also_in_compilation_question_sur_la_mobilisation_du",
    "1_practical_idea_also_in_compilation_une_id_e_pratique_toujours_dans_compilation",
    "og_is_important_because_il_est_important_d_avoir_un_gouvernement_ouvert_parce_que",
    "improvements_am_liorations",
    "other_comments_autres_commentaires",
]
print("Available policy columns in individual_comments:")
for c in policy_cols:
    if c in ind.columns:
        non_empty = ind[ind[c].str.strip() != ""].shape[0]
        print(f"  {c}: {non_empty} non-empty")
    else:
        print(f"  {c}: MISSING")
