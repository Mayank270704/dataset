"""
models
======
PolicyLens Machine Learning Models & Pipelines.

All subpackages are loaded lazily via PEP 562 ``__getattr__``.  Importing ``models``
alone does NOT trigger UMAP, torch, or any other heavy dependency; those are only
loaded when you explicitly import the relevant subpackage.

Subpackages (lazy-loaded)
--------------------------
  - models.common               : Shared filesystem paths and model configurations.
  - models.semantic_similarity  : MiniLM sentence embeddings, cosine retrieval, corpus builder.
  - models.concern_clustering   : UMAP + HDBSCAN clustering, c-TF-IDF keyword extraction.
  - models.concern_discovery    : Duplicate handling, cluster diagnostics, summaries, review.
  - models.stance_classification : Supervised stance classification (linsvm / logreg).
  - models.multilingual         : Multilingual & cross-lingual representations (MuRIL, planned).

Inference modules (lazy-loaded)
---------------------------------
  - models.new_comment_inference  : NewCommentPredictor — 3-label stance + 3-label sentiment.
  - models.stance_inference_v2    : FinalStancePredictor — heuristic 4-label pipeline
                                    (Support / Oppose / Neutral / Unclear).
                                    NOTE: Oppose is a heuristic label, NOT a validated classifier output.
"""
import importlib

__subpackages__ = {
    "common",
    "concern_clustering",
    "concern_discovery",
    "multilingual",
    "semantic_similarity",
    "stance_classification",
    "stance_inference_v2",
}

__all__ = sorted(__subpackages__)


def __getattr__(name: str):
    if name in __subpackages__:
        return importlib.import_module(f"models.{name}")
    raise AttributeError(f"module 'models' has no attribute '{name}'")
