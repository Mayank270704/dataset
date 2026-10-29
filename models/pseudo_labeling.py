"""
models/pseudo_labeling.py
=========================
Rule for assigning provisional (pseudo) stance labels to out-of-domain text (Sentiment140
tweets) using the existing verified stance model (Support/Neutral/Unclear) and the existing
verified Sentiment140 sentiment model (negative/positive).

This module does NOT retrain or load models by itself; callers load the two verified
pipelines (see models.new_comment_inference.load_verified_pipeline and
models.stance_inference_v2._load_verified_sentiment_model) and pass their raw predictions in.

Rule (fixed by the project owner on 2026-10-29, after being shown that the project's own
in-domain check — data/sentiment_evaluation/consultation_domain_check_v1/results.json —
found that of 73 AI-labelled negative-sentiment records in the 400-record pool, true stance
was Neutral 43 / Support 19 / Unclear 10 / Oppose 1. The heuristic below is therefore known,
even in-domain, to produce far more "Oppose" candidates than genuine Oppose cases; it is kept
because the project owner explicitly chose to proceed with it, on the condition that every
Oppose pseudo-label is marked heuristic/unverified and this risk is reported, never hidden.

    stance_raw == "Unclear"          -> pseudo_label = "Unclear"   (kept separate; never forced
                                         into Support/Neutral/Oppose, regardless of sentiment)
    sentiment == "negative":
        stance_raw == "Support"      -> pseudo_label = "Support"   (heuristic does NOT override Support)
        stance_raw == "Neutral"      -> pseudo_label = "Oppose"    (heuristic_oppose; UNVERIFIED, high
                                         known false-positive risk — see module docstring above)
    sentiment == "positive":
        stance_raw in {Support, Neutral} -> pseudo_label = stance_raw directly (stance model's own
                                         provisional output; positive sentiment is NEVER assumed to be Support)

Every row keeps: the raw stance prediction, the raw sentiment prediction, the rule that was
applied, and a disclaimer. Nothing here claims human-verified ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CHARS = 3  # same threshold as models/new_comment_inference.py and models/stance_inference_v2.py


def decision_margin(pipe, texts: list[str]) -> np.ndarray:
    """Top-1 minus top-2 decision_function score (or |distance from hyperplane| for binary);
    identical computation to the _margin() helpers in new_comment_inference.py / stance_inference_v2.py."""
    dec = pipe.decision_function(texts)
    dec = dec.reshape(len(texts), -1)
    if dec.shape[1] == 1:
        return np.abs(dec[:, 0])
    s = np.sort(dec, axis=1)
    return s[:, -1] - s[:, -2]

PSEUDO_LABEL_DISCLAIMER = (
    "PSEUDO-LABEL, NOT GROUND TRUTH. Generated from two out-of-domain-prone models: a stance "
    "model trained on 239-399 AI-generated working labels for 'stance toward Open Government' "
    "(not generic policy stance), and a Twitter sentiment model (not a stance model). The "
    "'Oppose' pseudo-label is produced by an unverified heuristic (negative sentiment + "
    "non-Support stance) that the project's own in-domain check found to be a ~98% "
    "false-positive rule against the few genuine Oppose examples available (1 real Oppose vs "
    "53 heuristic-Oppose candidates among 73 negative-sentiment pool400 records; see "
    "data/sentiment_evaluation/consultation_domain_check_v1/results.json). Sentiment140 tweets "
    "are general social-media text, not government-policy consultation submissions; stance "
    "pseudo-labels on them must not be presented as validated or human-verified."
)

PSEUDO_LABEL_DISCLAIMER_INDOMAIN_CORPUS = (
    "PSEUDO-LABEL, NOT GROUND TRUTH. This record IS genuine Canadian Open Government consultation "
    "text (same domain and same source datasets as the stance model's own 400-record AI-working-label "
    "training pool) -- unlike the Sentiment140 pseudo-labels, there is no cross-domain extrapolation "
    "for the stance prediction itself. However: (1) this is a form of self-training -- the stance model "
    "used to label this record was trained on a sample FROM THE SAME three source datasets, so it may "
    "simply reinforce its own existing biases/errors rather than add genuinely new signal; (2) the "
    "sentiment signal used by the Oppose heuristic still comes from a Twitter-trained model, not a "
    "consultation-domain one; (3) the Oppose heuristic (negative sentiment + non-Support stance) is "
    "exactly as unverified here as in the Sentiment140 case -- the project's own in-domain check found a "
    "~98% false-positive rate for it (1 real Oppose vs 53 heuristic-Oppose candidates among 73 "
    "negative-sentiment pool400 records; see data/sentiment_evaluation/consultation_domain_check_v1/"
    "results.json). This label must not be presented as human-verified or as independent of the "
    "stance model's own existing biases."
)

RULE_UNCLEAR_KEPT_SEPARATE = "stance_model_direct_unclear_kept_separate"
RULE_SUPPORT_DIRECT = "stance_model_direct_support"
RULE_HEURISTIC_OPPOSE = "heuristic_oppose_unverified"
RULE_NEUTRAL_DIRECT_POSITIVE_SENTIMENT = "stance_model_direct_neutral_positive_sentiment"
RULE_SUPPORT_DIRECT_POSITIVE_SENTIMENT = "stance_model_direct_support_positive_sentiment"

PSEUDO_LABELS = ("Support", "Neutral", "Oppose", "Unclear")


@dataclass(frozen=True)
class PseudoLabelResult:
    pseudo_label: str
    rule: str


def assign_pseudo_label(sentiment: str, stance_raw: str) -> PseudoLabelResult:
    """
    Apply the fixed labeling rule to one (sentiment, stance_raw) pair.

    Parameters
    ----------
    sentiment : {"negative", "positive"}
        Output of the verified Sentiment140StreamingClassifier.
    stance_raw : {"Support", "Neutral", "Unclear"}
        Output of the verified stance pipeline (run_pool400_ai_v1_3class never predicts Oppose).

    Returns
    -------
    PseudoLabelResult
    """
    if stance_raw not in ("Support", "Neutral", "Unclear"):
        raise ValueError(f"unexpected stance_raw {stance_raw!r}; expected Support/Neutral/Unclear")
    if sentiment not in ("negative", "positive"):
        raise ValueError(f"unexpected sentiment {sentiment!r}; expected negative/positive")

    if stance_raw == "Unclear":
        return PseudoLabelResult("Unclear", RULE_UNCLEAR_KEPT_SEPARATE)

    if sentiment == "negative":
        if stance_raw == "Support":
            return PseudoLabelResult("Support", RULE_SUPPORT_DIRECT)
        return PseudoLabelResult("Oppose", RULE_HEURISTIC_OPPOSE)

    # sentiment == "positive"
    if stance_raw == "Support":
        return PseudoLabelResult("Support", RULE_SUPPORT_DIRECT_POSITIVE_SENTIMENT)
    return PseudoLabelResult("Neutral", RULE_NEUTRAL_DIRECT_POSITIVE_SENTIMENT)


def assign_pseudo_labels_batch(sentiments: np.ndarray, stances_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised convenience wrapper over assign_pseudo_label for equal-length arrays."""
    if len(sentiments) != len(stances_raw):
        raise ValueError("sentiments and stances_raw must have the same length")
    labels = np.empty(len(sentiments), dtype=object)
    rules = np.empty(len(sentiments), dtype=object)
    for i, (s, st) in enumerate(zip(sentiments, stances_raw)):
        r = assign_pseudo_label(str(s), str(st))
        labels[i] = r.pseudo_label
        rules[i] = r.rule
    return labels, rules
