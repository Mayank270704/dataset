"""
models/stance_classification/baselines.py
=========================================
Reproducible TF-IDF baselines for consultation stance classification:
  1. TF-IDF + Logistic Regression (L2 penalty)
  2. TF-IDF + Linear Support Vector Classifier (LinearSVC)

Design & Safety Guardrails:
  - Splitting before fitting TF-IDF: The vectorizer is fit strictly on the
    training split to prevent vocabulary and document frequency leakage.
  - Group isolation: Duplicate-text groups (duplicate_text_group_id) are kept
    strictly within a single split (using GroupKFold / GroupShuffleSplit) to
    prevent verbatim text leakage between train and test.
  - Stratified evaluation: Macro-F1, per-class precision and recall, and
    confusion matrices are computed across permitted stance labels.
  - Strict human label validation: Training refuses to run if verified labels
    are absent, preventing auto-filling or accidental unlabelled training.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

VALID_STANCES = ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]
MIN_VERIFIED_SAMPLES_FOR_TRAINING = 10


def validate_labeled_data(df: pd.DataFrame, label_col: str = "stance") -> pd.DataFrame:
    """
    Validates that the DataFrame contains verified, non-empty human stance labels.
    Raises ValueError if insufficient labels exist.
    """
    if label_col not in df.columns:
        raise ValueError(f"Required label column '{label_col}' not found in DataFrame.")

    # Filter to non-blank, valid stance labels
    s = df[label_col].dropna().astype(str).str.strip()
    valid_mask = s.isin(VALID_STANCES)
    labeled_df = df[valid_mask].copy()

    if len(labeled_df) == 0:
        raise ValueError(
            "Insufficient verified human stance labels: 0 valid labels found. "
            "Human annotation is pending. Supervised training cannot proceed."
        )

    if len(labeled_df) < MIN_VERIFIED_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Insufficient verified human stance labels: found {len(labeled_df)} labels, "
            f"but at least {MIN_VERIFIED_SAMPLES_FOR_TRAINING} are required for reproducible evaluation."
        )

    # Check that at least 2 distinct classes exist
    unique_classes = labeled_df[label_col].nunique()
    if unique_classes < 2:
        raise ValueError(
            f"Insufficient class diversity: found only 1 unique stance class ('{labeled_df[label_col].iloc[0]}'). "
            "At least 2 classes are required for classification."
        )

    return labeled_df


def train_test_split_without_leakage(
    df: pd.DataFrame,
    text_col: str = "original_text",
    label_col: str = "stance",
    group_col: str = "duplicate_text_group_id",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits dataset into train and test while guaranteeing:
      1. Duplicate-text groups do not cross the train-test boundary.
      2. If group_col is empty or unassigned, uses record index as distinct group.
    """
    # Create group identifier: use duplicate_text_group_id if present and non-empty,
    # otherwise fallback to unique row index so unique rows are freely split
    groups = df[group_col].fillna("").astype(str) if group_col in df.columns else pd.Series([""] * len(df))
    # Where group is empty, assign unique row id to avoid grouping distinct texts together
    groups = np.where(groups.str.strip() != "", groups, df.index.astype(str))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return (
        train_df[text_col],
        test_df[text_col],
        train_df[label_col],
        test_df[label_col],
    )


class TfidfStanceClassifier:
    """Base TF-IDF classifier enforcing leakage prevention and reproducible metrics."""

    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        max_features: int = 5_000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{2,}\b",
        )

        if classifier_type == "logistic_regression":
            self.clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
            )
        elif classifier_type == "linear_svm":
            self.clf = LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        self.classes_: Optional[np.ndarray] = None

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        text_col: str = "original_text",
        label_col: str = "stance",
        group_col: str = "duplicate_text_group_id",
        test_size: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Validates labels, splits data without leakage, fits vectorizer on train only,
        and evaluates on held-out test data.
        """
        labeled_df = validate_labeled_data(df, label_col=label_col)

        X_train, X_test, y_train, y_test = train_test_split_without_leakage(
            labeled_df,
            text_col=text_col,
            label_col=label_col,
            group_col=group_col,
            test_size=test_size,
            random_state=self.random_state,
        )

        # ── Leakage prevention: FIT on train ONLY ───────────────────────────
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.clf.fit(X_train_vec, y_train)
        self.classes_ = np.unique(np.concatenate([y_train.values, y_test.values]))

        y_pred = self.clf.predict(X_test_vec)

        # Metrics computation
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_).tolist()

        return {
            "model_type": self.classifier_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "classes": self.classes_.tolist(),
            "macro_f1": round(macro_f1, 4),
            "classification_report": report_dict,
            "confusion_matrix": cm,
        }

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predicts stance for new consultation texts."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
        X_vec = self.vectorizer.transform(texts)
        return self.clf.predict(X_vec)
