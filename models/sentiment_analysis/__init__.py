"""
models/sentiment_analysis
=========================
Sentiment Analysis baseline models and streaming pipelines for auxiliary corpora.

Components:
- streaming_baseline.py: Memory-efficient HashingVectorizer + SGDClassifier baseline
  for large datasets (Sentiment140 1.6M records) using incremental partial_fit.
"""
from models.sentiment_analysis.sentimix_baseline import (
    SENTIMIX_INT_TO_LABEL,
    SENTIMIX_LABEL_TO_INT,
    SENTIMIX_TARGET_NAMES,
    SentiMixClassifier,
    evaluate_sentimix_model,
    train_sentimix_baseline,
)
from models.sentiment_analysis.streaming_baseline import (
    LABEL_MAPPING,
    MemoryPressureError,
    Sentiment140StreamingClassifier,
    evaluate_sentiment_model,
    train_streaming_sentiment140,
)

__all__ = [
    "LABEL_MAPPING",
    "MemoryPressureError",
    "Sentiment140StreamingClassifier",
    "evaluate_sentiment_model",
    "train_streaming_sentiment140",
    "SENTIMIX_LABEL_TO_INT",
    "SENTIMIX_INT_TO_LABEL",
    "SENTIMIX_TARGET_NAMES",
    "SentiMixClassifier",
    "evaluate_sentimix_model",
    "train_sentimix_baseline",
]
