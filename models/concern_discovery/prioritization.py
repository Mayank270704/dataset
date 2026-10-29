"""
models/concern_discovery/prioritization.py
==========================================
Exploratory, evidence-linked prioritization of discovered concern groups.

What the score is
    A transparent weighted sum of auditable signals computed from the existing concern groups and their
    original source records. It orders groups for INSPECTION. It is not a validated measure of public
    importance, urgency, policy impact or representativeness, and it is not fitted to any human judgment
    (none exists).

Counting units (kept separate; never conflated)
    entries          raw occurrences: non-blank designed free-text responses (one record may contribute several)
    unique_texts     normalised distinct texts
    independent_texts distinct near-duplicate groups: repeated or templated text counts once
    source_records   distinct (dataset, record id) pairs. A record is a questionnaire submission, a compilation
                     row or an index card; it is NOT a respondent, and the sources carry no respondent id.

Signals used in the score (each scaled to [0, 1] across the ranked groups)
    independent_volume   log(1 + independent_texts) / log(1 + max)        volume without campaign inflation
    independence         1 - duplicate_influence                          (duplicate_influence = 1 - independent/entries)
    dataset_diversity    normalised Shannon entropy of independent texts over source datasets
    field_diversity      normalised Shannon entropy over consultation fields (dataset|question)
    reliability          mean best-match Jaccard of the group across UMAP seeds (from concern discovery)
    tier_quality         ordinal map of the diagnostic tier (configurable)
    evidence             verified traceable supporting comments / evidence target

Context signals (reported, never scored): language mix, noise entries whose nearest group this is, raw entries,
unique texts, source records.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: dict[str, float] = {
    "independent_volume": 0.40,
    "independence": 0.10,
    "dataset_diversity": 0.10,
    "field_diversity": 0.10,
    "reliability": 0.15,
    "tier_quality": 0.05,
    "evidence": 0.10,
}
SIGNALS = tuple(DEFAULT_WEIGHTS)
DEFAULT_TIER_QUALITY = {"candidate": 1.0, "needs_review": 0.5, "campaign_or_template": 0.5}
RANKED_TIERS = tuple(DEFAULT_TIER_QUALITY)
NOT_RANKED_REASON = {
    "non_concern_pattern": "response pattern, not a policy concern (no-opinion/awareness or promotional)",
    "unreliable": "too few distinct texts, unstable across seeds, or markup-driven",
}
EVIDENCE_TARGET = 3                 # traceable supporting comments that give full evidence credit
MIN_TEMPLATE_CHARS = 20            # collective-demand candidates need >= 20 characters of non-markup text


class PrioritizationInputError(ValueError):
    pass


# ── configuration ─────────────────────────────────────────────────────────────

def validate_config(weights: dict[str, float], tier_quality: dict[str, float]) -> dict[str, float]:
    """Return weights normalised to sum 1. Rejects unknown/missing signals, negative or non-finite weights."""
    unknown = sorted(set(weights) - set(SIGNALS))
    missing = sorted(set(SIGNALS) - set(weights))
    if unknown or missing:
        raise PrioritizationInputError(f"weights: unknown {unknown}, missing {missing}; signals are {list(SIGNALS)}")
    vals = {k: float(v) for k, v in weights.items()}
    if any(not math.isfinite(v) or v < 0 for v in vals.values()):
        raise PrioritizationInputError("weights must be finite and non-negative")
    total = sum(vals.values())
    if total <= 0:
        raise PrioritizationInputError("at least one weight must be positive")
    if set(tier_quality) != set(RANKED_TIERS) or any(not 0 <= float(v) <= 1 for v in tier_quality.values()):
        raise PrioritizationInputError(f"tier_quality must map exactly {list(RANKED_TIERS)} to values in [0, 1]")
    return {k: vals[k] / total for k in SIGNALS}


# ── raw signals ───────────────────────────────────────────────────────────────

def normalised_entropy(counts: pd.Series, n_categories: int) -> float:
    """Shannon entropy of a count vector divided by log(n_categories); 0 for a single category."""
    c = counts[counts > 0].astype(float)
    if n_categories <= 1 or len(c) <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(n_categories))


def compute_signals(entries: pd.DataFrame, groups_table: pd.DataFrame, evidence_counts: dict[int, int]) -> pd.DataFrame:
    """Per-cluster raw signals from entry-level assignments (one row per original entry).

    ``entries`` needs: corpus_id, unique_id, source_dataset, source_record_id, question_slug, language, cluster_id,
    state, dup_group_id, nearest_cluster_if_noise. ``groups_table`` needs: cluster_id, tier, stability_jaccard_mean,
    short_label, flags.
    """
    e = entries.copy()
    e["cluster_id"] = e["cluster_id"].astype(int)
    e["field"] = e["source_dataset"] + "|" + e["question_slug"]
    clustered = e[e["state"] == "clustered"]
    n_ds = int(e["source_dataset"].nunique())
    n_fields = int(e["field"].nunique())
    noise = e[e["state"] == "noise"]
    nearby_noise = noise["nearest_cluster_if_noise"].astype(int).value_counts()
    g = groups_table.set_index("cluster_id")
    rows = []
    for cid, m in clustered.groupby("cluster_id"):
        # one row per independent text (near-duplicate group); its first entry supplies dataset/field
        indep = m.sort_values("corpus_id").drop_duplicates("dup_group_id")
        n_entries, n_indep = len(m), len(indep)
        lang = m["language"].where(m["language"].isin(["English", "French"]), "Other/Unknown").value_counts()
        rows.append({
            "cluster_id": int(cid),
            "tier": g.loc[cid, "tier"],
            "short_label": g.loc[cid, "short_label"],
            "flags": g.loc[cid, "flags"],
            "entries": n_entries,
            "unique_texts": int(m["unique_id"].nunique()),
            "independent_texts": n_indep,
            "source_records": int((m["source_dataset"] + "#" + m["source_record_id"]).nunique()),
            "duplicate_influence": round(1 - n_indep / n_entries, 4),
            "largest_repeat_share": round(float(m["dup_group_id"].value_counts().iloc[0] / n_entries), 4),
            "datasets_present": int(indep["source_dataset"].nunique()),
            "fields_present": int(indep["field"].nunique()),
            "dataset_entropy": round(normalised_entropy(indep["source_dataset"].value_counts(), n_ds), 4),
            "field_entropy": round(normalised_entropy(indep["field"].value_counts(), n_fields), 4),
            "dataset_counts": json.dumps(indep["source_dataset"].value_counts().to_dict()),
            "stability": float(g.loc[cid, "stability_jaccard_mean"]),
            "verified_evidence": int(evidence_counts.get(int(cid), 0)),
            "english": int(lang.get("English", 0)), "french": int(lang.get("French", 0)),
            "other_language": int(lang.get("Other/Unknown", 0)),
            "nearby_noise_entries": int(nearby_noise.get(int(cid), 0)),
        })
    out = pd.DataFrame(rows)
    out["ranked"] = out["tier"].isin(RANKED_TIERS)
    out["not_ranked_reason"] = out["tier"].map(NOT_RANKED_REASON).fillna("")
    no_ev = out["ranked"] & (out["verified_evidence"] == 0)
    out.loc[no_ev, "ranked"] = False
    out.loc[no_ev, "not_ranked_reason"] = "no traceable supporting comment could be verified"
    return out.sort_values("cluster_id").reset_index(drop=True)


# ── scoring ───────────────────────────────────────────────────────────────────

def scaled_signals(sig: pd.DataFrame, tier_quality: dict[str, float]) -> pd.DataFrame:
    """Signals in [0, 1] for the ranked groups (scaling depends only on the ranked set)."""
    r = sig[sig["ranked"]].copy()
    mx = float(np.log1p(r["independent_texts"]).max()) if len(r) else 1.0
    s = pd.DataFrame(index=r.index)
    s["independent_volume"] = np.log1p(r["independent_texts"]) / (mx if mx > 0 else 1.0)
    s["independence"] = 1 - r["duplicate_influence"]
    s["dataset_diversity"] = r["dataset_entropy"]
    s["field_diversity"] = r["field_entropy"]
    s["reliability"] = r["stability"].clip(0, 1)
    s["tier_quality"] = r["tier"].map(tier_quality).astype(float)
    s["evidence"] = (r["verified_evidence"].clip(upper=EVIDENCE_TARGET) / EVIDENCE_TARGET).astype(float)
    if s.isna().any().any():
        raise PrioritizationInputError(f"missing signal values: {s.columns[s.isna().any()].tolist()}")
    return s.round(6)


def score(sig: pd.DataFrame, weights: dict[str, float], tier_quality: dict[str, float]) -> pd.DataFrame:
    """Ranked table with per-signal contributions (weight x scaled value) that sum exactly to the score."""
    w = validate_config(weights, tier_quality)
    s = scaled_signals(sig, tier_quality)
    r = sig.loc[s.index, ["cluster_id", "tier", "short_label", "entries", "unique_texts", "independent_texts",
                          "source_records"]].copy()
    for k in SIGNALS:
        r[f"value_{k}"] = s[k]
        r[f"contrib_{k}"] = (w[k] * s[k]).round(6)
    r["score"] = r[[f"contrib_{k}" for k in SIGNALS]].sum(axis=1).round(6)
    # deterministic order: score, then independent texts, then cluster id
    r = r.sort_values(["score", "independent_texts", "cluster_id"], ascending=[False, False, True]).reset_index(drop=True)
    r["rank_multi"] = np.arange(1, len(r) + 1)
    base = r.sort_values(["entries", "unique_texts", "cluster_id"], ascending=[False, False, True])
    r["rank_frequency"] = r["cluster_id"].map(dict(zip(base["cluster_id"], range(1, len(base) + 1))))
    uq = r.sort_values(["unique_texts", "entries", "cluster_id"], ascending=[False, False, True])
    r["rank_unique"] = r["cluster_id"].map(dict(zip(uq["cluster_id"], range(1, len(uq) + 1))))
    r["rank_change_vs_frequency"] = r["rank_frequency"] - r["rank_multi"]        # positive = moved up
    return r


def explain(ranked: pd.DataFrame, top: int = 2) -> list[str]:
    """Why a group's multi-signal rank differs from its frequency rank, from contribution deviations."""
    contrib = ranked[[f"contrib_{k}" for k in SIGNALS]]
    dev = contrib - contrib.mean()
    out = []
    for i, r in ranked.iterrows():
        d = dev.loc[i].rename(lambda c: c.replace("contrib_", ""))
        ups = [f"{k} {d[k]:+.3f}" for k in d.sort_values(ascending=False).index[:top] if d[k] > 0.005]
        downs = [f"{k} {d[k]:+.3f}" for k in d.sort_values().index[:top] if d[k] < -0.005]
        move = int(r["rank_change_vs_frequency"])
        head = "unchanged" if move == 0 else (f"up {move}" if move > 0 else f"down {-move}")
        out.append(f"{head} vs frequency rank; above-average contributions: {', '.join(ups) or 'none'}; "
                   f"below-average: {', '.join(downs) or 'none'}")
    return out


def rank_correlations(a: pd.Series, b: pd.Series) -> dict[str, float]:
    from scipy.stats import kendalltau, spearmanr
    return {"spearman": round(float(spearmanr(a, b)[0]), 4), "kendall_tau": round(float(kendalltau(a, b)[0]), 4)}


def compare_baseline(ranked: pd.DataFrame, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, Any]:
    out: dict[str, Any] = {"n_ranked": int(len(ranked)),
                           "multi_vs_frequency": rank_correlations(ranked["rank_multi"], ranked["rank_frequency"]),
                           "multi_vs_unique": rank_correlations(ranked["rank_multi"], ranked["rank_unique"]),
                           "frequency_vs_unique": rank_correlations(ranked["rank_frequency"], ranked["rank_unique"])}
    for k in ks:
        a = set(ranked.loc[ranked["rank_multi"] <= k, "cluster_id"])
        b = set(ranked.loc[ranked["rank_frequency"] <= k, "cluster_id"])
        out[f"top{k}_overlap"] = {"shared": len(a & b), "entered": sorted(a - b), "left": sorted(b - a)}
    ch = ranked["rank_change_vs_frequency"].abs()
    out["mean_abs_rank_change"] = round(float(ch.mean()), 3)
    out["max_abs_rank_change"] = int(ch.max()) if len(ch) else 0
    return out


def sensitivity(sig: pd.DataFrame, weights: dict[str, float], tier_quality: dict[str, float]) -> dict[str, Any]:
    """Rank stability under weight perturbations: each signal dropped, halved and doubled (renormalised)."""
    base = score(sig, weights, tier_quality).set_index("cluster_id")["rank_multi"]
    variants: dict[str, pd.Series] = {}
    for k in SIGNALS:
        for name, f in (("drop", 0.0), ("half", 0.5), ("double", 2.0)):
            w = dict(weights)
            w[k] = weights[k] * f
            if sum(w.values()) <= 0:
                continue
            variants[f"{name}_{k}"] = score(sig, w, tier_quality).set_index("cluster_id")["rank_multi"].reindex(base.index)
    per = {name: rank_correlations(base, v)["kendall_tau"] for name, v in variants.items()}
    mat = pd.DataFrame(variants)
    span = pd.DataFrame({"rank_default": base, "rank_min": mat.min(axis=1), "rank_max": mat.max(axis=1)})
    top10 = set(base[base <= 10].index)
    always = [int(c) for c in top10 if (mat.loc[c] <= 10).all()]
    return {"kendall_tau_vs_default": per, "min_kendall_tau": round(min(per.values()), 4) if per else None,
            "top10_in_every_variant": sorted(always),
            "rank_range": {int(c): [int(r.rank_min), int(r.rank_max)] for c, r in span.iterrows()}}


# ── evidence ──────────────────────────────────────────────────────────────────

def verify_evidence(examples: pd.DataFrame, entries: pd.DataFrame, corpus_text: dict[int, str]) -> tuple[pd.DataFrame, list[str]]:
    """Keep only examples whose corpus_id belongs to the cluster and whose quote/full text are verbatim source text."""
    member = dict(zip(entries["corpus_id"].astype(int), entries["cluster_id"].astype(int)))
    ok, problems = [], []
    for r in examples.itertuples():
        cid, clu = int(r.corpus_id), int(r.cluster_id)
        src = corpus_text.get(cid)
        if member.get(cid) != clu:
            problems.append(f"cluster {clu}: corpus_id {cid} is not a member")
        elif src is None or r.full_text != src or r.quote not in src:
            problems.append(f"cluster {clu}: corpus_id {cid} text is not verbatim source text")
        else:
            ok.append(r.Index)
    return examples.loc[ok].copy(), problems


# ── collective-demand candidates ──────────────────────────────────────────────

CANDIDATE_COLUMNS = ["dup_group_id", "decision", "entries", "unique_texts", "source_records", "source_datasets",
                     "assigned_clusters", "states", "lead_corpus_id", "lead_text"]
CANDIDATE_EVIDENCE_COLUMNS = ["dup_group_id", "corpus_id", "source_dataset", "source_record_id", "source_row_number",
                              "source_field", "language", "state", "cluster_id", "text"]


def collective_demand_candidates(entries: pd.DataFrame, dup_groups: pd.DataFrame, corpus_text: dict[int, str],
                                 non_substantive: re.Pattern[str], min_entries: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated-text groups (>= min_entries entries) that state something substantive.

    These are identical or near-identical texts submitted repeatedly (possibly a coordinated campaign). Their
    entry count is NOT a count of people. Returns (candidates with inclusion decision, member evidence rows).
    """
    from models.concern_discovery.duplicates import strip_social
    g = dup_groups[dup_groups["group_entries"].astype(int) >= min_entries]
    rows, ev = [], []
    for gid, m in g.groupby("dup_group_id"):
        uids = m["unique_id"].astype(int).tolist()
        mem = entries[entries["unique_id"].astype(int).isin(uids)].sort_values("corpus_id")
        lead = mem.iloc[0]
        text = corpus_text[int(lead["corpus_id"])]
        content = strip_social(text)
        if len(content) < MIN_TEMPLATE_CHARS:
            decision = "excluded: no substantive text once links, mentions and hashtags are removed"
        elif non_substantive.search(content.replace("’", "'")):
            decision = "excluded: no-opinion / awareness or promotional wording"
        else:
            decision = "included"
        states = mem["state"].value_counts().to_dict()
        clusters = mem.loc[mem["state"] == "clustered", "cluster_id"].astype(int).value_counts().to_dict()
        rows.append({"dup_group_id": int(gid), "decision": decision, "entries": int(len(mem)),
                     "unique_texts": int(len(uids)), "source_records": int((mem["source_dataset"] + "#" + mem["source_record_id"]).nunique()),
                     "source_datasets": json.dumps(mem["source_dataset"].value_counts().to_dict()),
                     "assigned_clusters": json.dumps({str(k): v for k, v in clusters.items()}),
                     "states": json.dumps(states), "lead_corpus_id": int(lead["corpus_id"]), "lead_text": text})
        if decision == "included":
            for r in mem.itertuples():
                ev.append({"dup_group_id": int(gid), "corpus_id": int(r.corpus_id), "source_dataset": r.source_dataset,
                           "source_record_id": r.source_record_id, "source_row_number": r.source_row_number,
                           "source_field": r.source_field, "language": r.language, "state": r.state,
                           "cluster_id": int(r.cluster_id), "text": corpus_text[int(r.corpus_id)]})
    cand = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if len(cand):
        cand = cand.sort_values(["decision", "entries", "dup_group_id"], ascending=[False, False, True]).reset_index(drop=True)
    return cand, pd.DataFrame(ev, columns=CANDIDATE_EVIDENCE_COLUMNS)


# ── output schema validation ──────────────────────────────────────────────────

RANKING_COLUMNS = ["rank_multi", "rank_frequency", "rank_unique", "rank_change_vs_frequency", "cluster_id", "tier",
                   "short_label", "entries", "unique_texts", "independent_texts", "source_records", "score"] + \
                  [f"value_{k}" for k in SIGNALS] + [f"contrib_{k}" for k in SIGNALS]


def validate_outputs(ranked: pd.DataFrame, evidence: pd.DataFrame, weights: dict[str, float]) -> list[str]:
    problems = []
    missing = [c for c in RANKING_COLUMNS if c not in ranked.columns]
    if missing:
        return [f"ranking missing columns {missing}"]
    n = len(ranked)
    for col in ("rank_multi", "rank_frequency", "rank_unique"):
        if sorted(ranked[col].tolist()) != list(range(1, n + 1)):
            problems.append(f"{col} is not a permutation of 1..{n}")
    s = ranked[[f"contrib_{k}" for k in SIGNALS]].sum(axis=1)
    if (s - ranked["score"]).abs().max() > 1e-5:
        problems.append("contributions do not sum to the score")
    vals = ranked[[f"value_{k}" for k in SIGNALS]]
    if ((vals < -1e-9) | (vals > 1 + 1e-9)).any().any():
        problems.append("scaled signal outside [0, 1]")
    if (ranked["unique_texts"] > ranked["entries"]).any() or (ranked["independent_texts"] > ranked["unique_texts"]).any():
        problems.append("count hierarchy violated (independent <= unique <= entries)")
    if abs(sum(weights.values()) - 1) > 1e-9:
        problems.append("normalised weights do not sum to 1")
    lacking = sorted(set(ranked["cluster_id"]) - set(evidence["cluster_id"]))
    if lacking:
        problems.append(f"ranked clusters without verified evidence: {lacking}")
    return problems
