"""Canonical five-class stance schema for new staging work."""

STANCE_LABELS = ("Support", "Neutral", "Oppose", "Mixed", "Unclear")
# Only these labels currently have enough original validation records for OOF metrics.
ORIGINAL_EVALUABLE_LABELS = ("Support", "Neutral", "Unclear")
