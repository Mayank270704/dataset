"""
app
===
PolicyLens application layer. Currently: the read-only, manifest-verified results-access layer
(app/results_access.py) that the future dashboard will consume. See app/README.md.
"""
from app.results_access import (
    COMPONENTS,
    SCOPE_NOTE,
    ArtifactPaths,
    ArtifactUnavailable,
    ResultsStore,
)

__all__ = ["COMPONENTS", "SCOPE_NOTE", "ArtifactPaths", "ArtifactUnavailable", "ResultsStore"]
