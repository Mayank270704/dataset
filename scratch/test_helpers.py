import pandas as pd
import re
from difflib import SequenceMatcher
from pathlib import Path

PROCESSED = Path(r"c:\Users\MAYANK SWAROOP\OneDrive\Desktop\dataset\data\processed")

ind = pd.read_csv(PROCESSED / "individual_comments_2016_18_source_clean.csv", keep_default_na=False, dtype=str)
comp = pd.read_csv(PROCESSED / "consultation_compilation_2017_18_source_clean.csv", keep_default_na=False, dtype=str)

ind_text_columns = [
    "public_engagement_question_also_in_compilation_question_sur_la_mobilisation_du",
    "1_practical_idea_also_in_compilation_une_id_e_pratique_toujours_dans_compilation",
    "og_is_important_because_il_est_important_d_avoir_un_gouvernement_ouvert_parce_que",
    "improvements_am_liorations",
    "other_comments_autres_commentaires",
]
comp_text_columns = [
    "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part",
    "comment_question_or_idea_translation_if_available_traduction_commentaire_question_ou_id_e_si_disponible",
    "subject_objet",
]

id_matches = ind.merge(
    comp,
    left_on="row_rang_e",
    right_on="row_ligne",
    suffixes=("_individual", "_compilation"),
)

def text_similarity(a, b) -> float:
    left = re.sub(r"\s+", " ", str(a).strip().casefold())
    right = re.sub(r"\s+", " ", str(b).strip().casefold())
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()

overlap_rows = []
for _, row in id_matches.iterrows():
    best_similarity = 0.0
    best_ind_col = ""
    best_comp_col = ""
    best_ind_text = ""
    best_comp_text = ""
    for ic in ind_text_columns:
        it = row.get(ic, "")
        if not str(it).strip():
            continue
        for cc in comp_text_columns:
            ct = row.get(cc, "")
            if not str(ct).strip():
                continue
            sim = text_similarity(it, ct)
            if sim > best_similarity:
                best_similarity = sim
                best_ind_col = ic
                best_comp_col = cc
                best_ind_text = it
                best_comp_text = ct

    ind_ev = row.get("event", "")
    comp_ev = row.get("event_v_nement", "")
    ind_src = row.get("source_individual", "")
    comp_src = row.get("source_compilation", "")
    ind_lang = row.get("source_language_individual", "")
    comp_lang = row.get("source_language_compilation", "")
    event_same = bool(ind_ev and comp_ev and ind_ev.strip().lower() == comp_ev.strip().lower())
    source_same = bool(ind_src and comp_src and ind_src.strip().lower() == comp_src.strip().lower())
    lang_same = bool(ind_lang and comp_lang and ind_lang.strip().lower() == comp_lang.strip().lower())

    if best_similarity >= 0.95 and (event_same or source_same):
        category = "confirmed_same_record"
        limitation = "High text similarity and matching event/source; requires human confirmation before merge."
    elif best_similarity >= 0.90:
        category = "same_text_different_record"
        limitation = "High text similarity but event/source differ; likely repeated phrase across distinct contexts."
    elif best_similarity >= 0.50:
        category = "ambiguous_or_partial_similarity"
        limitation = "Partial text overlap (similarity between 0.50 and 0.90); distinct events and sources."
    else:
        category = "unrelated_different_event_and_text"
        limitation = "Independent sequential integer row numbering; text similarity < 0.50, events and sources differ."

    overlap_rows.append({
        "overlap_pair_id": f"row_id_overlap_{len(overlap_rows) + 1:05d}",
        "shared_row_id": row.get("row_rang_e", ""),
        "individual_source_row_number": row.get("source_row_number_individual", ""),
        "compilation_source_row_number": row.get("source_row_number_compilation", ""),
        "individual_event": ind_ev,
        "compilation_event": comp_ev,
        "individual_source": ind_src,
        "compilation_source": comp_src,
        "individual_language": ind_lang,
        "compilation_language": comp_lang,
        "event_same": event_same,
        "source_same": source_same,
        "language_same": lang_same,
        "best_text_similarity": round(best_similarity, 3),
        "best_individual_column": best_ind_col,
        "best_compilation_column": best_comp_col,
        "individual_text_original": best_ind_text,
        "compilation_text_original": best_comp_text,
        "evidence_category": category,
        "evidence_summary": f"Shared ID={row.get('row_rang_e', '')}; best text similarity={best_similarity:.3f}; event_same={event_same}; source_same={source_same}; language_same={lang_same}.",
        "limitations": limitation,
        "review_status": "",
        "review_decision": "",
        "review_notes": "",
        "reviewer": "",
    })

df_overlap = pd.DataFrame(overlap_rows)
print("Overlap rows count:", len(df_overlap))
print("Categories:")
print(df_overlap["evidence_category"].value_counts())
