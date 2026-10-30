"""Shared feature construction for the context-aware three-class staging model."""
from __future__ import annotations

from dataclasses import dataclass


MIN_COMMENT_CHARS = 3
MAX_COMMENT_CHARS = 5000
MAX_QUESTION_CHARS = 2000
MAX_CONTEXT_CHARS = 5000
SUPPORTED_LABELS = ("Support", "Neutral", "Unclear")
MISSING_QUESTION = "[NO_QUESTION]"
MISSING_CONTEXT = "[NO_CONTEXT]"
MISSING_COMMENT = "[NO_COMMENT]"


def _clean(value: object, missing: str, limit: int) -> str:
    if value is None:
        return missing
    text = " ".join(str(value).split())
    return text[:limit] if text else missing


def build_context_feature(question: object, context: object, comment: object) -> str:
    """Build the exact training/inference text; language and metadata are excluded."""
    return (
        f"[CONTEXT] {_clean(context, MISSING_CONTEXT, MAX_CONTEXT_CHARS)} "
        f"[QUESTION] {_clean(question, MISSING_QUESTION, MAX_QUESTION_CHARS)} "
        f"[COMMENT] {_clean(comment, MISSING_COMMENT, MAX_COMMENT_CHARS)}"
    )


@dataclass(frozen=True)
class ContextPredictionInput:
    question: str | None
    context: str | None
    comment: str | None
    language: str | None = None

    def validate(self) -> None:
        for name, value, maximum in (
            ("question", self.question, MAX_QUESTION_CHARS),
            ("context", self.context, MAX_CONTEXT_CHARS),
            ("comment", self.comment, MAX_COMMENT_CHARS),
            ("language", self.language, 100),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
            if isinstance(value, str) and len(value) > maximum:
                raise ValueError(f"{name} must be at most {maximum} characters")

    @property
    def comment_is_usable(self) -> bool:
        return isinstance(self.comment, str) and len(self.comment.strip()) >= MIN_COMMENT_CHARS

    def feature_text(self) -> str:
        self.validate()
        return build_context_feature(self.question, self.context, self.comment)
