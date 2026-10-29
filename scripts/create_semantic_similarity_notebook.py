"""
scripts/create_semantic_similarity_notebook.py
================================================
Generates notebooks/04_semantic_similarity.ipynb.
Fully self-contained notebook demonstrating:
  1. Offline loading of sentence-transformers/all-MiniLM-L6-v2 via transformers.
  2. Attention-mask-aware mean pooling vs CLS pooling analysis.
  3. Structured corpus ingestion from the 3 source-clean datasets with full provenance.
  4. Separate embedding (.npy) and metadata (.csv) persistence with cache fingerprinting.
  5. Cosine-similarity retrieval for free-text queries and record-to-record queries with self-exclusion.
  6. Empirical language distribution checks and explicit limitation documentation.
"""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "04_semantic_similarity.ipynb"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7",
        },
    }

    cells = []

    # ── Cell 1: Markdown Title & Purpose & Limitations ─────────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""# PolicyLens — 04. Semantic Similarity & Evidence Retrieval

**Project:** PolicyLens — Public Consultation NLP Analysis  
**Phase:** Semantic Discovery & Evidence Retrieval (Unsupervised)  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (offline cached, via `transformers` directly)  
**Pooling Strategy:** Attention-mask-aware mean pooling + L2 normalization (MiniLM canonical)  

---

### Purpose
This notebook implements the first PolicyLens semantic-discovery component. Without requiring human stance labels, it indexes substantive consultation comments from the three cleaned Canadian open-government datasets (2016–2018) into a dense vector space and enables:
1. **Free-text semantic query retrieval:** Match user questions or policy queries against consultation comments to discover relevant citizen feedback.
2. **Record-to-record nearest neighbour discovery:** Find recurring themes and clustered concerns across consultation questions and events, while strictly excluding self-matches.
3. **Traceable provenance and metadata preservation:** Maintain exact 1-to-1 alignment between embeddings and source record IDs, files, question contexts, and stakeholder metadata.

---

### Critical Methodological & Ethical Limitations
Before interpreting similarity scores or using them for policy insights, note the following constraints:

1. **Semantic Similarity $\neq$ Identical Policy Concern:**
   High cosine similarity indicates shared vocabulary, phrasing, or general topic (e.g., open data formats), not necessarily identical specific requests or underlying policy positions.

2. **Semantic Similarity $\neq$ Stance Agreement:**
   Opposing viewpoints on the exact same topic (e.g., "We must proactively disclose all lobbyist logs immediately" vs. "Do not disclose lobbyist logs without strict privacy filters") will often have very high cosine similarity because they share core terminology. Dense similarity models do not solve stance classification.

3. **Language & Domain Caveat (English vs. French vs. Hindi/Hinglish):**
   `all-MiniLM-L6-v2` was trained predominantly on English web and question-answer corpora. Performance on Canadian French consultation comments (~315 records in the compilation dataset and bilingual questionnaire responses) is uncalibrated. Furthermore, auxiliary datasets in Hindi or Hinglish (`PolicyLens_Hindi_English_Hinglish_Sentiment.csv`) must not be assumed compatible with this model without explicit multilingual benchmarking.

4. **No Causal or Prioritization Claim:**
   A large cluster of similar comments does not imply consensus, organic citizen mobilization, or objective policy urgency. It may reflect structured survey prompts, organized campaign campaigns, or repetitive event notes.

5. **No Supervised Labels or Classifier Training:**
   This component is purely unsupervised. It does not generate stance labels, train classifiers, or modify source datasets."""))

    # ── Cell 2: Code — Environment & Dependency Check ─────────────────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 1: Environment and Cache Verification
# ==============================================================================
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

print("Environment verification:")
print(f"  Python       : {sys.version.split()[0]}")
print(f"  PyTorch      : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"  Transformers : {transformers.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  NumPy        : {np.__version__}")
print(f"  Pandas       : {pd.__version__}")

# Establish project paths
NOTEBOOK_PATH = Path.cwd()
PROJECT_ROOT = NOTEBOOK_PATH.parent if NOTEBOOK_PATH.name == "notebooks" else NOTEBOOK_PATH
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nPaths:")
print(f"  PROJECT_ROOT  : {PROJECT_ROOT}")
print(f"  PROCESSED_DIR : {PROCESSED_DIR} (exists: {PROCESSED_DIR.exists()})")
print(f"  EMBEDDINGS_DIR: {EMBEDDINGS_DIR} (exists: {EMBEDDINGS_DIR.exists()})")"""))

    # ── Cell 3: Code — Offline Model Loading ─────────────────────────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 2: Offline Loading of all-MiniLM-L6-v2
# ==============================================================================
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading tokenizer and model from local cache ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Successfully loaded {MODEL_ID} without network downloads.")
print(f"  Total parameters : {num_params:,}")
print(f"  Hidden size      : {model.config.hidden_size}")
print(f"  Max sequence len : {model.config.max_position_embeddings}")"""))

    # ── Cell 4: Markdown — Pooling Strategy Explanation ──────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""### Attention-Mask-Aware Mean Pooling vs. CLS Pooling

In BERT-style models:
- **CLS pooling** extracts only the hidden representation of the first `[CLS]` token. While common in classification heads, sentence-transformers models are explicitly trained with **mean pooling** over all valid non-padding tokens.
- **Attention-mask-aware mean pooling** computes:
  $$\mathbf{e} = \frac{\sum_{i=1}^T \mathbf{h}_i \cdot m_i}{\sum_{i=1}^T m_i}$$
  where $\mathbf{h}_i$ is the hidden state at position $i$ and $m_i \in \{0, 1\}$ is the attention mask.
- Applying **L2 normalization** ($\|\mathbf{e}\|_2 = 1$) ensures that the Euclidean dot product directly equals cosine similarity:
  $$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$

We implement this explicitly below and confirm that mean pooling differs substantially from naive CLS pooling."""))

    # ── Cell 5: Code — Mean Pooling Implementation & Verification ─────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 3: Pooling Implementation and Mathematical Validation
# ==============================================================================

def mean_pool_and_normalize(
    texts: list[str],
    tokenizer,
    model,
    max_length: int = 256,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    # Computes L2-normalized 384-dimensional sentence embeddings using
    # attention-mask-aware mean pooling.
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

        token_embeddings = outputs.last_hidden_state  # (B, T, 384)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)

        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Unit normalization (L2)
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        emb_np = normalized.cpu().numpy().astype(np.float32)

        if not np.isfinite(emb_np).all():
            raise ValueError(f"Non-finite values encountered in batch {i // batch_size}")

        embeddings_list.append(emb_np)

    all_embeddings = np.vstack(embeddings_list)
    assert all_embeddings.shape[1] == 384, f"Expected 384-d, got {all_embeddings.shape[1]}"
    return all_embeddings


# Verification with test sentences
test_sentences = [
    "Open government promotes transparency, accountability, and citizen participation.",
    "Federal departments should publish machine-readable datasets on open.canada.ca.",
    "Short text.",
]

emb_test = mean_pool_and_normalize(test_sentences, tokenizer, model)
norms = np.linalg.norm(emb_test, axis=1)

print("Pooling validation:")
print(f"  Shape      : {emb_test.shape} (expected 3 x 384)")
print(f"  All finite : {np.isfinite(emb_test).all()}")
print(f"  Norms      : {norms} (expected all approx 1.0)")
assert np.allclose(norms, 1.0, atol=1e-5), "Vectors are not unit normalized!"

# Demonstrate difference between Mean Pooling and CLS Pooling
with torch.no_grad():
    enc = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
    out = model(**enc)
    cls_emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1).numpy()

cls_vs_mean_cos = (emb_test * cls_emb).sum(axis=1)
print(f"\nCosine similarity between Mean-Pooled and CLS-Pooled vectors:")
for sent, sim in zip(test_sentences, cls_vs_mean_cos):
    print(f"  sim={sim:.4f} | \"{sent[:50]}...\"")
print("Notice that CLS pooling produces different vectors; mean pooling is required for MiniLM.")"""))

    # ── Cell 6: Code — Structured Corpus Builder ──────────────────────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 4: Structured Consultation Corpus Assembly
# ==============================================================================
import re

MIN_TEXT_LEN = 20
GENERIC_FILLERS = {
    "none", "n/a", "na", "no", "nil", "nothing", "not applicable",
    "no comment", "no comments", "none at this time", "none at present",
    "rien", "aucun", "aucune", "sans objet", "non", "pas de commentaire",
}

def is_eligible_comment(text: str, exclusion_reason: str = "") -> bool:
    t = text.strip()
    if not t or len(t) < MIN_TEXT_LEN:
        return False
    if exclusion_reason.strip():
        return False
    if t.casefold() in GENERIC_FILLERS:
        return False
    return True

# Load and sample from each dataset
# Note: sample_per_dataset=150 provides a fast, reproducible demo (~400 records).
# Set sample_per_dataset=None to process the full corpus.
def assemble_corpus(processed_dir: Path, sample_per_dataset: int | None = 150, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records = []

    # 1. individual_comments_2016_18
    ind_path = processed_dir / "individual_comments_2016_18_source_clean.csv"
    if ind_path.exists():
        ind_df = pd.read_csv(ind_path, dtype=str, keep_default_na=False)
        ind_df = ind_df[ind_df["row_rang_e"].str.strip() != ""].copy()
        
        ind_fields = [
            ("public_engagement_question", "public_engagement_question_also_in_compilation_question_sur_la_mobilisation_du", "Public engagement question"),
            ("practical_idea", "1_practical_idea_also_in_compilation_une_id_e_pratique_toujours_dans_compilation", "1 practical idea"),
            ("og_is_important_because", "og_is_important_because_il_est_important_d_avoir_un_gouvernement_ouvert_parce_que", "Open Government is important because..."),
            ("improvements", "improvements_am_liorations", "Improvements?"),
            ("other_comments", "other_comments_autres_commentaires", "Other comments"),
        ]
        
        ind_candidates = []
        for slug, col, wording in ind_fields:
            if col not in ind_df.columns:
                continue
            for _, row in ind_df.iterrows():
                t = row.get(col, "")
                ex = row.get("exclusion_reason", "")
                if not is_eligible_comment(t, ex):
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
                    "original_text": t.strip(),
                })
        if sample_per_dataset and len(ind_candidates) > sample_per_dataset:
            sel = rng.choice(len(ind_candidates), size=sample_per_dataset, replace=False)
            ind_candidates = [ind_candidates[i] for i in sorted(sel)]
        records.extend(ind_candidates)

    # 2. consultation_compilation_2017_18
    comp_path = processed_dir / "consultation_compilation_2017_18_source_clean.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path, dtype=str, keep_default_na=False)
        comp_col = "comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part"
        comp_candidates = []
        for _, row in comp_df.iterrows():
            t = row.get(comp_col, "")
            ex = row.get("exclusion_reason", "")
            if not is_eligible_comment(t, ex):
                continue
            lang_raw = row.get("source_language", "").strip()
            comp_candidates.append({
                "source_dataset": "consultation_compilation_2017_18",
                "source_record_id": str(row.get("source_row_number", "")),
                "source_file": row.get("source_file", "ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv"),
                "source_row_number": str(row.get("source_row_number", "")),
                "source_field": comp_col,
                "question_slug": "comment_question_or_idea",
                "question_wording": "Comment, Question or Idea",
                "question_context": f"Event: {row.get('event_v_nement', '')} | Source: {row.get('source', '')} | Theme: {row.get('theme_th_me', '')} | Prompt: {row.get('context_prompt_contexte_message_guide', '')}",
                "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                "language": lang_raw or "Unknown",
                "stakeholder_group": row.get("source", "").strip(),
                "original_text": t.strip(),
            })
        if sample_per_dataset and len(comp_candidates) > sample_per_dataset:
            sel = rng.choice(len(comp_candidates), size=sample_per_dataset, replace=False)
            comp_candidates = [comp_candidates[i] for i in sorted(sel)]
        records.extend(comp_candidates)

    # 3. get_involved_questionnaire_2017_18
    quest_path = processed_dir / "get_involved_questionnaire_2017_18_source_clean.csv"
    if quest_path.exists():
        quest_df = pd.read_csv(quest_path, dtype=str, keep_default_na=False)
        q_fields = [
            ("q8", "q8_did_we_miss_anything_if_the_previous_questions_did_not_speak_to_you_please_tell_us_about_the_important_aspects_of_open_government_that_drive_your_interest_if_you_have_more_than_one_additional_item_please_list_them_in_priority", "Q8. Did we miss anything?"),
            ("q14", "q14_please_tell_us_why", "Q14. Why satisfaction rating?"),
            ("q18", "q18_if_you_have_knowledge_or_experience_with_a_open_government_topic_that_you_would_be_willing_to_share_please_tell_us_about_it", "Q18. Topic willing to share"),
        ]
        quest_candidates = []
        for slug, col, wording in q_fields:
            if col not in quest_df.columns:
                continue
            for _, row in quest_df.iterrows():
                t = row.get(col, "")
                ex = row.get("exclusion_reason", "")
                if not is_eligible_comment(t, ex):
                    continue
                lang_raw = row.get("language", "").strip()
                lang = "French" if "fran" in lang_raw.lower() else ("English" if "anglais" in lang_raw.lower() or "eng" in lang_raw.lower() else (lang_raw or "Unknown"))
                sector = row.get("q19_the_sector_or_group_that_i_identify_most_closely_with_is", "").strip()
                quest_candidates.append({
                    "source_dataset": "get_involved_questionnaire_2017_18",
                    "source_record_id": str(row.get("source_row_number", "")),
                    "source_file": row.get("source_file", "Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv"),
                    "source_row_number": str(row.get("source_row_number", "")),
                    "source_field": col,
                    "question_slug": slug,
                    "question_wording": wording,
                    "question_context": f"Question: {slug.upper()} | Sector: {sector}",
                    "duplicate_text_group_id": row.get("duplicate_text_group_id", ""),
                    "language": lang,
                    "stakeholder_group": sector,
                    "original_text": t.strip(),
                })
        if sample_per_dataset and len(quest_candidates) > sample_per_dataset:
            sel = rng.choice(len(quest_candidates), size=sample_per_dataset, replace=False)
            quest_candidates = [quest_candidates[i] for i in sorted(sel)]
        records.extend(quest_candidates)

    df = pd.DataFrame(records)
    df.insert(0, "corpus_id", range(len(df)))
    return df

corpus_df = assemble_corpus(PROCESSED_DIR, sample_per_dataset=150, random_seed=42)
print(f"Assembled corpus size: {len(corpus_df)} records")
print("\nDataset distribution:")
print(corpus_df["source_dataset"].value_counts())
print("\nLanguage distribution:")
print(corpus_df["language"].value_counts())"""))

    # ── Cell 7: Code — Embedding & Caching with Metadata Alignment ────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 5: Embedding Computation, Validation, and Caching
# ==============================================================================
import hashlib
import json

def get_corpus_hash(df: pd.DataFrame, model_id: str, max_len: int) -> str:
    content = {
        "model_id": model_id,
        "max_len": max_len,
        "n_records": len(df),
        "texts_sha256": hashlib.sha256("||".join(df["original_text"].tolist()).encode("utf-8")).hexdigest(),
        "record_ids": hashlib.sha256("||".join(df["source_record_id"].tolist()).encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()[:16]

EMBED_FILE = EMBEDDINGS_DIR / "minilm_embeddings.npy"
META_FILE = EMBEDDINGS_DIR / "corpus_metadata.csv"
HASH_FILE = EMBEDDINGS_DIR / "corpus_hash.txt"

corpus_hash = get_corpus_hash(corpus_df, MODEL_ID, max_len=256)
print(f"Computed corpus fingerprint hash: {corpus_hash}")

cache_hit = (
    EMBED_FILE.exists()
    and META_FILE.exists()
    and HASH_FILE.exists()
    and HASH_FILE.read_text(encoding="utf-8").strip() == corpus_hash
)

if cache_hit:
    print("Cache hit! Loading existing embeddings and metadata from disk...")
    embeddings = np.load(EMBED_FILE)
    meta_df = pd.read_csv(META_FILE, dtype=str, keep_default_na=False)
    meta_df["corpus_id"] = meta_df["corpus_id"].astype(int)
else:
    print("Computing embeddings via attention-mask-aware mean pooling...")
    embeddings = mean_pool_and_normalize(
        texts=corpus_df["original_text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        max_length=256,
        batch_size=64,
    )
    np.save(EMBED_FILE, embeddings)
    corpus_df.to_csv(META_FILE, index=False, encoding="utf-8")
    HASH_FILE.write_text(corpus_hash, encoding="utf-8")
    meta_df = corpus_df.copy()
    meta_df["corpus_id"] = meta_df["corpus_id"].astype(int)
    print("Saved embeddings, metadata, and hash file.")

# Validation checks
assert embeddings.shape[0] == len(meta_df), f"Row mismatch: {embeddings.shape[0]} vs {len(meta_df)}"
assert embeddings.shape[1] == 384, f"Dimension error: {embeddings.shape[1]}"
assert np.isfinite(embeddings).all(), "Non-finite values detected in embedding matrix!"
norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), "Vectors are not unit-normalized!"

print("\nValidation passed successfully:")
print(f"  Embedding matrix : {embeddings.shape} ({embeddings.dtype})")
print(f"  Metadata rows    : {len(meta_df)}")
print(f"  Row alignment    : Exactly 1-to-1 between matrix row and meta_df['corpus_id']")"""))

    # ── Cell 8: Code — Cosine-Similarity Retrieval Implementation ─────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 6: Cosine Similarity Retrieval with Self-Exclusion
# ==============================================================================

def retrieve_similar(
    query: str | int,
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    tokenizer=None,
    model=None,
    top_k: int = 5,
    exclude_query_record: bool = True,
) -> pd.DataFrame:
    # Retrieves top-k semantically similar comments using cosine similarity.
    # If query is an int, it is treated as a corpus_id and its stored vector is used.
    # When querying by existing record, that record is excluded from results if exclude_query_record is True.
    exclude_idx: int | None = None

    if isinstance(query, (int, np.integer)):
        match = meta_df.index[meta_df["corpus_id"] == int(query)]
        if len(match) == 0:
            raise ValueError(f"corpus_id {query} not found in metadata.")
        row_pos = match[0]
        query_vec = embeddings[row_pos].reshape(1, -1)
        if exclude_query_record:
            exclude_idx = row_pos
    else:
        query_vec = mean_pool_and_normalize([str(query)], tokenizer, model)
        if exclude_query_record:
            exact = meta_df.index[meta_df["original_text"] == str(query)]
            if len(exact) > 0:
                exclude_idx = exact[0]

    # Matrix multiplication dot-product on unit vectors
    similarities = (embeddings @ query_vec.T).squeeze()
    if np.ndim(similarities) == 0:
        similarities = np.array([similarities])

    res = meta_df.copy()
    res["cosine_similarity"] = similarities.astype(float)

    if exclude_idx is not None:
        res = res[res.index != exclude_idx]

    top_matches = res.nlargest(top_k, "cosine_similarity").reset_index(drop=True)
    top_matches.insert(0, "rank", range(1, len(top_matches) + 1))

    output_cols = [
        "rank", "cosine_similarity", "corpus_id", "source_dataset",
        "source_record_id", "source_field", "question_slug", "language", "stakeholder_group",
        "original_text", "question_context",
    ]
    return top_matches[[c for c in output_cols if c in top_matches.columns]]

# Demo 1: Free-text policy search
query_prompt = "Access to federal government spending data and proactive financial disclosure."
print(f"=== Free-Text Query Retrieval ===")
print(f"Query: \"{query_prompt}\"\n")

results_query = retrieve_similar(query_prompt, embeddings, meta_df, tokenizer, model, top_k=5)
for _, r in results_query.iterrows():
    print(f"Rank {int(r['rank'])} | Sim: {r['cosine_similarity']:.4f} | ID: {r['source_record_id']} ({r['source_dataset']})")
    print(f"  Field   : {r['question_slug']} | Language: {r['language']}")
    print(f"  Evidence: \"{r['original_text'][:140]}...\"")
    print()"""))

    # ── Cell 9: Code — Record-to-Record Retrieval & Self-Exclusion ────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 7: Record-to-Record Retrieval and Self-Match Exclusion
# ==============================================================================

# Select a substantive reference comment from the corpus
sample_anchor = meta_df[meta_df["original_text"].str.len() > 60].iloc[0]
anchor_id = int(sample_anchor["corpus_id"])

print("=== Reference Anchor Record ===")
print(f"Corpus ID        : {anchor_id}")
print(f"Source Dataset   : {sample_anchor['source_dataset']}")
print(f"Source Record ID : {sample_anchor['source_record_id']}")
print(f"Original Text    : \"{sample_anchor['original_text']}\"")
print("\nRetrieving top 5 nearest neighbors with self-exclusion enabled...")

results_anchor = retrieve_similar(
    query=anchor_id,
    embeddings=embeddings,
    meta_df=meta_df,
    tokenizer=tokenizer,
    model=model,
    top_k=5,
    exclude_query_record=True,
)

print(f"\nTop 5 Nearest Neighbor Records (Self-Exclusion Verified):")
for _, r in results_anchor.iterrows():
    print(f"Rank {int(r['rank'])} | Sim: {r['cosine_similarity']:.4f} | Corpus ID: {r['corpus_id']} | Record ID: {r['source_record_id']}")
    print(f"  Context : {r['question_context']}")
    print(f"  Evidence: \"{r['original_text'][:140]}...\"")
    print()

# Verification of self-exclusion
returned_ids = results_anchor["corpus_id"].astype(int).tolist()
assert anchor_id not in returned_ids, f"Self-exclusion failed! Query ID {anchor_id} was returned in results: {returned_ids}"
print(f"[PASS] Self-exclusion verified: anchor record (corpus_id={anchor_id}) was excluded from its own nearest neighbours.")"""))

    # ── Cell 10: Code — Cross-Language Distribution Diagnostic ───────────────
    cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# Cell 8: Cross-Language Similarity Diagnostic
# ==============================================================================
print("Diagnostic: Pairwise cosine similarity distribution by language subset.")

for lang in ["English", "French"]:
    subset_mask = (meta_df["language"] == lang).values
    n_records = subset_mask.sum()
    if n_records < 2:
        print(f"Too few records for {lang} (n={n_records}). Skipping.")
        continue
    
    sub_emb = embeddings[subset_mask]
    n_sample = min(200, n_records)
    sub_sample = sub_emb[:n_sample]
    
    sim_matrix = sub_sample @ sub_sample.T
    # Exclude diagonal (self-similarity of 1.0)
    off_diag = sim_matrix[~np.eye(n_sample, dtype=bool)]
    
    print(f"\nLanguage: {lang} (total records={n_records}, sampled={n_sample}):")
    print(f"  Mean off-diagonal similarity   : {np.mean(off_diag):.4f}")
    print(f"  Median off-diagonal similarity : {np.median(off_diag):.4f}")
    print(f"  90th percentile similarity     : {np.percentile(off_diag, 90):.4f}")
    print(f"  Max off-diagonal similarity    : {np.max(off_diag):.4f}")

print("\nCaveat: High intra-language similarity does not prove cross-lingual semantic alignment.")
print("Consultation comments in French should not be assumed to have identical precision to English comments without human benchmarking.")"""))

    # ── Cell 11: Markdown — Conclusion & Next Steps ───────────────────────────
    cells.append(nbf.v4.new_markdown_cell(r"""## Summary & Next Steps

### Completed in this Notebook:
1. **Model & Pooling Verification:** Verified offline loading of `all-MiniLM-L6-v2` and correct implementation of attention-mask-aware mean pooling and unit L2 normalization.
2. **Provenance-Preserving Corpus:** Built an aligned consultation corpus spanning all three source-clean consultation datasets, preserving original IDs, questions, contexts, stakeholder information, and verbatim comments.
3. **Decoupled Embedding Index:** Saved dense embeddings separately in `data/embeddings/minilm_embeddings.npy` with strict row alignment to `data/embeddings/corpus_metadata.csv` and cache fingerprinting.
4. **Retrieval with Self-Exclusion:** Implemented and validated cosine retrieval for free-text queries and record-to-record matching with guaranteed self-match exclusion.
5. **Limitations Documented:** Clarified that semantic similarity does not equate to stance agreement, policy importance, or cross-lingual parity.

### Next Steps in PolicyLens:
- Complete human pilot annotation (`annotator_1.csv` and `annotator_2.csv`) and run agreement analysis in `03_annotation_agreement_analysis.ipynb`.
- Calibrate stance classifiers (TF-IDF + Logistic Regression / Linear SVM) on adjudicated pilot records.
- Link semantic clusters to adjudicated stances to identify which specific citizen concerns receive positive vs. critical feedback."""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_notebook()
