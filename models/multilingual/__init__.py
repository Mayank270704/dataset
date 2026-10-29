"""
models/multilingual
===================
PolicyLens Cross-Lingual & Multilingual Representation Subpackage.

Status: Planned / Pending Integration
-------------------------------------
PolicyLens handles consultation and auxiliary datasets spanning multiple languages:
  1. English (primary federal consultations & Sentiment140)
  2. French (Canadian bilingual open government consultations)
  3. Hindi & Hinglish (HinEmo-9 emotion corpus, SentiMix code-mixed benchmark)

Planned Components:
-------------------
1. MuRIL (Multilingual Representations for Indian Languages):
   - Hugging Face Model ID: `google/muril-base-cased`
   - Purpose: Cross-lingual representations across English, Hindi, and Hinglish.
   - Status: Planned for domain transfer experiments from auxiliary datasets.

2. Multilingual Sentence Transformers:
   - Target Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - Purpose: Bridging English and French consultation submissions into a single
     co-aligned embedding space. (Note: Current MiniLM `all-MiniLM-L6-v2` naturally
     clusters French and English separately; multilingual MiniLM unifies themes).
"""
MURIL_MODEL_ID = "google/muril-base-cased"
MULTILINGUAL_MINILM_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
