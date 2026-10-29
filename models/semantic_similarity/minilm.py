"""
models/semantic_similarity/minilm.py
====================================
PolicyLens semantic-discovery and evidence-retrieval module using
sentence-transformers/all-MiniLM-L6-v2:
  1. Mean pooling with attention mask + L2 normalization (MiniLM compatible).
  2. Substantive consultation corpus assembly from the 3 processed datasets.
  3. Embedding computation and caching with row-to-embedding alignment.
  4. Cosine-similarity retrieval (text queries and record-to-record retrieval with self-exclusion).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.common.config import DEFAULT_SENTENCE_MODEL

MODEL_ID_DEFAULT = DEFAULT_SENTENCE_MODEL
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 64
MIN_TEXT_LEN = 20

GENERIC_FILLERS = {
    "none", "n/a", "na", "no", "nil", "nothing", "not applicable",
    "no comment", "no comments", "none at this time", "none at present",
    "rien", "aucun", "aucune", "sans objet", "non", "pas de commentaire",
}

# Dataset column definitions
INDIVIDUAL_FIELDS = [
    (
        "public_engagement_question",
        "public_engagement_question_also_in_compilation_question_sur_la_mobilisation_du",
        "Public engagement question / Question sur la mobilisation du public",
    ),
    (
        "practical_idea",
        "1_practical_idea_also_in_compilation_une_id_e_pratique_toujours_dans_compilation",
        "1 practical idea / 1 idée pratique",
    ),
    (
        "og_is_important_because",
        "og_is_important_because_il_est_important_d_avoir_un_gouvernement_ouvert_parce_que",
        "Open Government is important because... / Il est important d'avoir un gouvernement ouvert parce que...",
    ),
    (
        "improvements",
        "improvements_am_liorations",
        "Improvements? / Améliorations",
    ),
    (
        "other_comments",
        "other_comments_autres_commentaires",
        "Other comments / Autres commentaires",
    ),
]

COMPILATION_TEXT_COL = (
    "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part"
)
COMPILATION_CTX_COL = "context_prompt_contexte_message_guide"
COMPILATION_THEME_COL = "theme_th_me"

QUESTIONNAIRE_FIELDS = [
    (
        "q8",
        "q8_did_we_miss_anything_if_the_previous_questions_did_not_speak_to_you_please_tell_us_about_the_important_aspects_of_open_government_that_drive_your_interest_if_you_have_more_than_one_additional_item_please_list_them_in_priority",
        "Q8. Did we miss anything? If the previous questions did not speak to you, please tell us about the important aspects of open government that drive your interest.",
    ),
    (
        "q14",
        "q14_please_tell_us_why",
        "Q14. Please tell us why (regarding satisfaction with open.canada.ca):",
    ),
    (
        "q18",
        "q18_if_you_have_knowledge_or_experience_with_a_open_government_topic_that_you_would_be_willing_to_share_please_tell_us_about_it",
        "Q18. If you have knowledge or experience with an open government topic that you would be willing to share, please tell us about it:",
    ),
]


def is_eligible(text: str, exclusion_reason: str = "") -> bool:
    """Return True when a consultation text comment meets substantive criteria."""
    t = text.strip()
    if not t or len(t) < MIN_TEXT_LEN:
        return False
    if exclusion_reason.strip():
        return False
    if t.casefold() in GENERIC_FILLERS:
        return False
    return True


def mean_pool_and_normalize(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode texts into L2-normalized 384-dimensional sentence embeddings using
    attention-mask-aware mean pooling.

    Steps:
      1. Tokenize batch with attention mask, padding, truncation.
      2. Model forward pass in torch.no_grad().
      3. Compute mean of token embeddings weighted by attention mask.
      4. Apply L2 normalization so dot products equal cosine similarities.
      5. Validate finiteness (no NaN, no Inf) and 384-d dimensionality.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model.eval()
    model.to(device)
    embeddings_list: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, 384)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()  # (batch_size, seq_len, 1)

        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # L2 normalization
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        emb_np = normalized.cpu().numpy().astype(np.float32)

        if not np.isfinite(emb_np).all():
            raise ValueError(f"Non-finite values encountered in batch {i // batch_size}")

        embeddings_list.append(emb_np)

    all_embeddings = np.vstack(embeddings_list)

    if all_embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings from MiniLM, got {all_embeddings.shape[1]}"
        )

    return all_embeddings


def build_corpus(
    processed_dir: Path,
    sample_per_dataset: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Assemble eligible consultation records from all three source-clean datasets.

    Preserves:
      - corpus_id: 0-indexed int strictly aligned with embedding matrix rows
      - source_dataset, source_record_id, source_file, source_row_number
      - source_field, question_slug, question_wording, question_context
      - duplicate_text_group_id
      - language: reliable language tag ('English', 'French', etc.)
      - stakeholder_group: sector / participant role where available
      - original_text: unmodified verbatim text
    """
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []

    # 1. individual_comments_2016_18
    ind_path = processed_dir / "individual_comments_2016_18_source_clean.csv"
    if ind_path.exists():
        ind_df = pd.read_csv(ind_path, dtype=str, keep_default_na=False)
        ind_df = ind_df[ind_df["row_rang_e"].str.strip() != ""].copy()

        ind_candidates = []
        for slug, col, wording in INDIVIDUAL_FIELDS:
            if col not in ind_df.columns:
                continue
            for _, row in ind_df.iterrows():
                text = row.get(col, "")
                excl = row.get("exclusion_reason", "")
                if not is_eligible(text, excl):
                    continue
                lang_raw = row.get("source_language", "").strip()
                lang = "English" if lang_raw.lower() == "english" else ("French" if lang_raw.lower() == "french" else (lang_raw or "Unknown"))
                ind_candidates.append({
                    "source_dataset": "individual_comments_2016_18",
                    "source_record_id": str(row.get("source_row_number", "")),
                    "source_file": row.get("source_file", "ap2016-18_individual_comments.csv"),
                    "source_row_number": str(row.get("source_row_number", "")),
                    "source_field": col,
                    "question_slug": slug,
                    "question_wording": wording,
                    "question_context": f"Event: {row.get('event', '')} | Sector: {row.get('sector', '')}",
                    "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                    "language": lang,
                    "stakeholder_group": row.get("sector", "").strip(),
                    "original_text": text.strip(),
                })

        if sample_per_dataset is not None and len(ind_candidates) > sample_per_dataset:
            indices = rng.choice(len(ind_candidates), size=sample_per_dataset, replace=False)
            ind_candidates = [ind_candidates[idx] for idx in sorted(indices)]
        records.extend(ind_candidates)

    # 2. consultation_compilation_2017_18
    comp_path = processed_dir / "consultation_compilation_2017_18_source_clean.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path, dtype=str, keep_default_na=False)
        comp_candidates = []
        for _, row in comp_df.iterrows():
            text = row.get(COMPILATION_TEXT_COL, "")
            excl = row.get("exclusion_reason", "")
            if not is_eligible(text, excl):
                continue
            lang_raw = row.get("source_language", "").strip()
            comp_candidates.append({
                "source_dataset": "consultation_compilation_2017_18",
                "source_record_id": str(row.get("source_row_number", "")),
                "source_file": row.get("source_file", "ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv"),
                "source_row_number": str(row.get("source_row_number", "")),
                "source_field": COMPILATION_TEXT_COL,
                "question_slug": "comment_question_or_idea",
                "question_wording": "Comment, Question or Idea (In source language) / Commentaire, question ou idée (dans la langue de départ)",
                "question_context": f"Event: {row.get('event_v_nement', '')} | Source: {row.get('source', '')} | Theme: {row.get(COMPILATION_THEME_COL, '')} | Prompt: {row.get(COMPILATION_CTX_COL, '')}",
                "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                "language": lang_raw or "Unknown",
                "stakeholder_group": row.get("source", "").strip(),
                "original_text": text.strip(),
            })

        if sample_per_dataset is not None and len(comp_candidates) > sample_per_dataset:
            indices = rng.choice(len(comp_candidates), size=sample_per_dataset, replace=False)
            comp_candidates = [comp_candidates[idx] for idx in sorted(indices)]
        records.extend(comp_candidates)

    # 3. get_involved_questionnaire_2017_18
    quest_path = processed_dir / "get_involved_questionnaire_2017_18_source_clean.csv"
    if quest_path.exists():
        quest_df = pd.read_csv(quest_path, dtype=str, keep_default_na=False)
        quest_candidates = []
        for slug, col, wording in QUESTIONNAIRE_FIELDS:
            if col not in quest_df.columns:
                continue
            for _, row in quest_df.iterrows():
                text = row.get(col, "")
                excl = row.get("exclusion_reason", "")
                if not is_eligible(text, excl):
                    continue
                lang_raw = row.get("language", "").strip()
                lang = "French" if "fran" in lang_raw.lower() else ("English" if "anglais" in lang_raw.lower() or "eng" in lang_raw.lower() else (lang_raw or "Unknown"))
                sector_raw = row.get("q19_the_sector_or_group_that_i_identify_most_closely_with_is", "").strip()
                quest_candidates.append({
                    "source_dataset": "get_involved_questionnaire_2017_18",
                    "source_record_id": str(row.get("source_row_number", "")),
                    "source_file": row.get("source_file", "Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv"),
                    "source_row_number": str(row.get("source_row_number", "")),
                    "source_field": col,
                    "question_slug": slug,
                    "question_wording": wording,
                    "question_context": f"Question: {slug.upper()} | Sector: {sector_raw}",
                    "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                    "language": lang,
                    "stakeholder_group": sector_raw,
                    "original_text": text.strip(),
                })

        if sample_per_dataset is not None and len(quest_candidates) > sample_per_dataset:
            indices = rng.choice(len(quest_candidates), size=sample_per_dataset, replace=False)
            quest_candidates = [quest_candidates[idx] for idx in sorted(indices)]
        records.extend(quest_candidates)

    df = pd.DataFrame(records)
    # Assign deterministic corpus_id 0..N-1
    df.insert(0, "corpus_id", range(len(df)))
    return df


def compute_corpus_hash(df: pd.DataFrame, model_id: str, max_length: int) -> str:
    """Stable SHA-256 fingerprint of corpus texts and encoding configuration."""
    content = {
        "model_id": model_id,
        "max_length": max_length,
        "n_records": len(df),
        "texts_sha256": hashlib.sha256("||".join(df["original_text"].tolist()).encode("utf-8")).hexdigest(),
        "record_ids": hashlib.sha256("||".join(df["source_record_id"].tolist()).encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def embed_and_save_corpus(
    corpus_df: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    embeddings_dir: Path,
    model_id: str = MODEL_ID_DEFAULT,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    force_recompute: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate or load embeddings from cache with strict row-to-embedding alignment.
    Returns (embeddings_array, metadata_df).
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    emb_file = embeddings_dir / "minilm_embeddings.npy"
    meta_file = embeddings_dir / "corpus_metadata.csv"
    hash_file = embeddings_dir / "corpus_hash.txt"

    expected_hash = compute_corpus_hash(corpus_df, model_id, max_length)

    cache_valid = (
        not force_recompute
        and emb_file.exists()
        and meta_file.exists()
        and hash_file.exists()
        and hash_file.read_text(encoding="utf-8").strip() == expected_hash
    )

    if cache_valid:
        embeddings = np.load(emb_file)
        meta_df = pd.read_csv(meta_file, dtype=str, keep_default_na=False)
        meta_df["corpus_id"] = meta_df["corpus_id"].astype(int)
        if embeddings.shape[0] == len(meta_df) and embeddings.shape[1] == 384:
            return embeddings, meta_df

    # Compute embeddings
    texts = corpus_df["original_text"].tolist()
    embeddings = mean_pool_and_normalize(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    # Validate alignment
    if embeddings.shape[0] != len(corpus_df):
        raise ValueError(f"Alignment error: {embeddings.shape[0]} embeddings for {len(corpus_df)} records")

    np.save(emb_file, embeddings)
    corpus_df.to_csv(meta_file, index=False, encoding="utf-8")
    hash_file.write_text(expected_hash, encoding="utf-8")

    meta_df = corpus_df.copy()
    meta_df["corpus_id"] = meta_df["corpus_id"].astype(int)
    return embeddings, meta_df


def retrieve_similar(
    query: str | int,
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    tokenizer: Any = None,
    model: Any = None,
    top_k: int = 5,
    exclude_query_record: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Retrieve top-k most semantically similar consultation comments.

    Parameters:
      query: Either a query string (encoded via model) OR an integer corpus_id.
      embeddings: (N, 384) L2-normalized float32 numpy array.
      meta_df: Metadata DataFrame aligned 1-to-1 with embeddings rows.
      tokenizer: Hugging Face tokenizer (required if query is str).
      model: Hugging Face model (required if query is str).
      top_k: Number of nearest comments to return.
      exclude_query_record: If True, exclude the query itself when querying by
                            an existing corpus_id or exact matching text.

    Returns:
      pd.DataFrame containing rank, cosine_similarity, corpus_id, source_dataset,
      source_record_id, source_field, language, stakeholder_group, original_text,
      question_context.
    """
    if len(meta_df) == 0 or embeddings.shape[0] == 0:
        return pd.DataFrame()

    exclude_idx: int | None = None

    if isinstance(query, (int, np.integer)):
        query_corpus_id = int(query)
        match_idx = meta_df.index[meta_df["corpus_id"] == query_corpus_id]
        if len(match_idx) == 0:
            raise ValueError(f"corpus_id {query_corpus_id} not found in metadata.")
        row_pos = match_idx[0]
        query_vec = embeddings[row_pos].reshape(1, -1)
        if exclude_query_record:
            exclude_idx = row_pos
    else:
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model must be provided when query is a text string.")
        query_vec = mean_pool_and_normalize([str(query)], tokenizer, model, device=device)
        # Check if this text exactly equals any existing corpus text
        if exclude_query_record:
            exact_matches = meta_df.index[meta_df["original_text"] == str(query)]
            if len(exact_matches) > 0:
                exclude_idx = exact_matches[0]

    # Cosine similarities = dot product since both are L2-normalized
    similarities = (embeddings @ query_vec.T).squeeze()
    if np.ndim(similarities) == 0:
        similarities = np.array([similarities])

    results_df = meta_df.copy()
    results_df["cosine_similarity"] = similarities.astype(float)

    if exclude_idx is not None:
        results_df = results_df[results_df.index != exclude_idx]

    top_results = results_df.nlargest(top_k, "cosine_similarity").reset_index(drop=True)
    top_results.insert(0, "rank", range(1, len(top_results) + 1))

    output_cols = [
        "rank",
        "cosine_similarity",
        "corpus_id",
        "source_dataset",
        "source_record_id",
        "source_field",
        "question_slug",
        "language",
        "stakeholder_group",
        "original_text",
        "question_context",
    ]
    return top_results[[c for c in output_cols if c in top_results.columns]]
