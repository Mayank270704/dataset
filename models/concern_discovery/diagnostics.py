"""
models/concern_discovery/diagnostics.py
=======================================
Internal (label-free) diagnostics for concern clusters.

Everything here is an INTERNAL proxy: geometry of an embedding space, agreement between
random seeds, and association with metadata that should NOT drive a concern grouping
(language, instrument, question field, channel, duplicate templates). None of these
measures shows that a cluster is a coherent policy concern; only human judgments can.

Unit conventions
----------------
* ``labels`` are unique-text-level integer arrays (one per unique text). ``>= 0`` = cluster id,
  ``-1`` = noise, ``< -1`` = excluded by preprocessing (never counted as noise).
* Entry-level statistics weight unique texts by multiplicity via the entry table.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from models.concern_discovery.duplicates import strip_social

# ── thresholds (all recorded in the run configuration) ────────────────────────
THRESHOLDS: dict[str, Any] = {
    "undersized_min_distinct_groups": 15,       # < 15 distinct near-duplicate groups -> undersized
    "template_top_group_share": 0.30,           # one near-dup group >= 30% of entries (and >= 5 entries)
    "template_repeated_share": 0.50,            # >= 50% of entries in repeated-template groups
    "non_concern_pattern_share": 0.40,          # >= 40% no-opinion/awareness OR promotional texts
    "social_markup_share": 0.50,                # markup counts as promotional only together with one of:
    "promotion_with_markup_share": 0.20,        #   >= 20% promotional phrasing, or
    "content_free_share": 0.30,                 #   >= 30% texts with < 20 characters left after stripping markup
    "channel_share": 0.70, "channel_min_entries": 20,
    "mixed_negative_margin_share": 0.25,        # >= 25% of members closer to another centroid
    "mixed_keyword_support": 0.15,              # mean support of top-3 keywords below this
    "weak_cohesion_quantile": 0.25,             # cohesion at/below the configuration's 25th percentile
    "broad_share": 0.08,
    "unstable_jaccard": 0.50, "moderate_stability_jaccard": 0.70,
    "french_majority_share": 0.50,
    "single_field_share": 0.90, "single_instrument_share": 0.95,
}

TIER_ORDER = ["unreliable", "campaign_or_template", "non_concern_pattern", "needs_review", "candidate"]
TIER_DESCRIPTIONS = {
    "candidate": "no diagnostic flag; a candidate concern group pending human review",
    "needs_review": "plausible group with a quality warning (mixed, weak, broad, moderately unstable, "
                    "language- or channel-concentrated)",
    "campaign_or_template": "dominated by a repeated template/campaign; the concern may be real but rests on "
                            "few independent texts",
    "non_concern_pattern": "coherent response pattern that is not a policy concern (no-opinion/awareness "
                           "answers, social-media promotion or event logistics)",
    "unreliable": "too few distinct texts, unstable across random seeds, or (raw-text clusterings) grouped by "
                  "social-media markup rather than content; not summarised as a concern",
}

# Transparent lexicons for response patterns that are not policy concerns (EN/FR).
NO_OPINION_RE = re.compile(
    r"\b(?:i\s+)?(?:don'?t|do not|didn'?t|did not)\s+(?:really\s+)?know\b|\bnot sure\b|\bno (?:comment|opinion|idea)\b"
    r"|\b(?:haven'?t|have not|hasn'?t|has not|never)\s+(?:yet\s+|really\s+|actually\s+)?(?:looked|visited|used|seen|"
    r"checked|been|heard|accessed|explored|gone|had)\b|\bnot (?:yet\s+)?(?:very\s+|really\s+)?familiar\b|\bunfamiliar\b"
    r"|\bnot (?:really\s+)?aware\b|\bfirst (?:time )?i(?:'ve| have)? heard\b|\bno experience\b|\bnothing to add\b"
    r"|\bje ne (?:le |la |les )?(?:sais|connais) pas\b|\bne (?:le |la )?connais pas\b|\baucune? (?:idée|commentaire|opinion)\b"
    r"|\bsans opinion\b|\bpas d'opinion\b|\bje n'(?:y )?ai (?:pas|jamais)\b|\bpas (?:encore )?(?:assez |bien )?informée?s?\b"
    r"|\bpas (?:assez )?familiers?\b|\bjamais (?:visité|utilisé|entendu)\b",
    re.I)
PROMOTION_RE = re.compile(
    r"\bthanks?\b|\bthank you\b|\bmerci\b|\bjoin us\b|\bregister\b|\brsvp\b|\blooking forward\b|\bgreat (?:session|event|"
    r"discussion|presentation)\b|\bsee you\b|\bdon'?t forget\b|\binscri(?:vez|ption)\b|\bdaily\b.*\bstories\b|\bthe latest\b",
    re.I)


def _norm(t: str) -> str:
    return t.replace("’", "'").replace("‘", "'")


def pattern_share(texts: list[str], pattern: re.Pattern[str]) -> float:
    if not texts:
        return 0.0
    return float(np.mean([pattern.search(_norm(t)) is not None for t in texts]))


# ── geometry ──────────────────────────────────────────────────────────────────

def centroids(emb: np.ndarray, labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    ids = sorted(int(c) for c in np.unique(labels[labels >= 0]))
    cen = np.vstack([emb[labels == c].mean(axis=0) for c in ids]) if ids else np.zeros((0, emb.shape[1]))
    norms = np.linalg.norm(cen, axis=1, keepdims=True)
    return ids, cen / np.maximum(norms, 1e-12)


def cluster_geometry(emb: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cluster cohesion/separation and per-point margins.

    margin = cos(own centroid) - max cos(other centroids). Negative margin = the point lies closer to another
    cluster's centroid (an overlap/mixing signal). Noise points get their nearest centroid for error analysis.
    """
    ids, cen = centroids(emb, labels)
    pos = {c: i for i, c in enumerate(ids)}
    sims = emb @ cen.T if len(ids) else np.zeros((len(emb), 0))
    pts = []
    for u in range(len(labels)):
        lab = int(labels[u])
        if lab < -1 or not len(ids):
            continue
        row = sims[u]
        if lab >= 0:
            own = float(row[pos[lab]])
            other = np.delete(row, pos[lab])
            j = int(np.argmax(other)) if len(other) else -1
            best_other_id = [c for c in ids if c != lab][j] if j >= 0 else -1
            best_other = float(other[j]) if j >= 0 else np.nan
            pts.append((u, lab, own, best_other_id, best_other, own - best_other if j >= 0 else np.nan))
        else:
            j = int(np.argmax(row))
            pts.append((u, -1, np.nan, ids[j], float(row[j]), np.nan))
    points = pd.DataFrame(pts, columns=["unique_id", "cluster_id", "cos_own_centroid", "nearest_other_cluster",
                                        "cos_nearest_other", "margin"])
    rows = []
    cc = cen @ cen.T if len(ids) else np.zeros((0, 0))
    for c in ids:
        p = points[points["cluster_id"] == c]
        i = pos[c]
        others = np.delete(cc[i], i)
        rows.append({
            "cluster_id": c,
            "cohesion_mean": round(float(p["cos_own_centroid"].mean()), 4),
            "cohesion_p10": round(float(p["cos_own_centroid"].quantile(0.10)), 4),
            "margin_mean": round(float(p["margin"].mean()), 4) if p["margin"].notna().any() else np.nan,
            "negative_margin_share": round(float((p["margin"] < 0).mean()), 4) if p["margin"].notna().any() else np.nan,
            "nearest_cluster": int([x for x in ids if x != c][int(np.argmax(others))]) if len(others) else -1,
            "centroid_cos_nearest": round(float(others.max()), 4) if len(others) else np.nan,
        })
    return pd.DataFrame(rows), points


def simplified_silhouette(points: pd.DataFrame) -> float | None:
    """Mean of (a - b) / max(a, b) on cosine DISTANCES to centroids (clustered points only); a cheap silhouette proxy."""
    p = points[points["cluster_id"] >= 0].dropna(subset=["cos_nearest_other"])
    if p.empty:
        return None
    a = 1 - p["cos_own_centroid"].to_numpy()
    b = 1 - p["cos_nearest_other"].to_numpy()
    return round(float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12))), 4)


# ── stability ─────────────────────────────────────────────────────────────────

def cluster_stability(ref: np.ndarray, others: list[np.ndarray]) -> pd.DataFrame:
    """Best-match Jaccard of each reference cluster's member set against every alternative partition.

    Only points that are in the clustering population in both runs (label >= -1) are compared, so excluded
    texts never count as disagreement. Returns mean and minimum over the alternative runs.
    """
    ids = sorted(int(c) for c in np.unique(ref[ref >= 0]))
    out = []
    for c in ids:
        vals = []
        for alt in others:
            pop = (ref >= -1) & (alt >= -1)
            a = set(np.where((ref == c) & pop)[0].tolist())
            best = 0.0
            for d in np.unique(alt[np.isin(np.arange(len(alt)), list(a)) & (alt >= 0)]):
                b = set(np.where((alt == d) & pop)[0].tolist())
                best = max(best, len(a & b) / max(len(a | b), 1))
            vals.append(best)
        out.append({"cluster_id": c, "stability_jaccard_mean": round(float(np.mean(vals)), 4) if vals else np.nan,
                    "stability_jaccard_min": round(float(np.min(vals)), 4) if vals else np.nan,
                    "stability_runs": len(vals)})
    return pd.DataFrame(out, columns=["cluster_id", "stability_jaccard_mean", "stability_jaccard_min", "stability_runs"])


# ── association with metadata (confounds) ─────────────────────────────────────

def _codes(x: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(x).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def cramers_v_corrected(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013). 0 = no association, 1 = perfect association."""
    ca, ka = _codes(a)
    cb, kb = _codes(b)
    n = len(ca)
    if n < 2 or ka < 2 or kb < 2:
        return 0.0
    table = np.bincount(ca * kb + cb, minlength=ka * kb).reshape(ka, kb).astype(float)
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (ka - 1) * (kb - 1) / (n - 1))
    kac = ka - (ka - 1) ** 2 / (n - 1)
    kbc = kb - (kb - 1) ** 2 / (n - 1)
    denom = min(kac - 1, kbc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def association(labels: np.ndarray, attr: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict[str, Any]:
    """Cramer's V and NMI between cluster labels and a metadata attribute, with a permutation null."""
    from sklearn.metrics import normalized_mutual_info_score
    labels = np.asarray(labels)
    attr = np.asarray(attr).astype(str)
    if len(labels) != len(attr):
        raise ValueError("labels and attribute are not aligned")
    if len(labels) < 2 or len(np.unique(attr)) < 2 or len(np.unique(labels)) < 2:
        return {"n": int(len(labels)), "cramers_v": None, "nmi": None, "perm_p": None, "note": "fewer than two categories"}
    v = cramers_v_corrected(labels, attr)
    rng = np.random.default_rng(seed)
    null = np.array([cramers_v_corrected(labels, rng.permutation(attr)) for _ in range(n_perm)])
    return {"n": int(len(labels)), "categories": int(len(np.unique(attr))),
            "cramers_v": round(v, 4), "null_v_mean": round(float(null.mean()), 4),
            "null_v_p95": round(float(np.percentile(null, 95)), 4),
            "perm_p": round(float((np.sum(null >= v) + 1) / (n_perm + 1)), 4),
            "nmi": round(float(normalized_mutual_info_score(attr, labels)), 4)}


def lang_group(lang: str) -> str:
    return lang if lang in ("English", "French") else "Other/Unknown"


def confound_report(entry_labels: np.ndarray, entries: pd.DataFrame, repeated_template: np.ndarray,
                    n_perm: int = 200, seed: int = 0) -> dict[str, Any]:
    """Partition-level association of clustered entries with attributes that should not drive concern groups."""
    m = entry_labels >= 0
    lab = entry_labels[m]
    e = entries[m]
    out: dict[str, Any] = {
        "language": association(lab, e["language"].map(lang_group).to_numpy(), n_perm, seed),
        "source_dataset": association(lab, e["source_dataset"].to_numpy(), n_perm, seed),
        "question_field": association(lab, (e["source_dataset"] + "|" + e["question_slug"]).to_numpy(), n_perm, seed),
        "repeated_template_membership": association(lab, np.asarray(repeated_template)[m].astype(int), n_perm, seed),
    }
    comp = (e["source_dataset"] == "consultation_compilation_2017_18").to_numpy()
    out["compilation_channel"] = association(lab[comp], e["stakeholder_group"].to_numpy()[comp], n_perm, seed)
    # cross-lingual mixing: among clusters where >= 5 French entries are expected, how many contain French?
    fr = (e["language"] == "French").to_numpy()
    base = fr.mean() if len(fr) else 0.0
    sizes = pd.Series(lab).value_counts()
    fr_counts = pd.Series(lab[fr]).value_counts().reindex(sizes.index, fill_value=0)
    eligible = sizes[sizes * base >= 5].index
    out["cross_lingual_mixing"] = {
        "french_share_of_clustered_entries": round(float(base), 4),
        "clusters_expected_ge5_french": int(len(eligible)),
        "of_which_contain_french": int((fr_counts[eligible] > 0).sum()),
        "of_which_french_below_quarter_of_expected": int((fr_counts[eligible] < 0.25 * sizes[eligible] * base).sum()),
        "clusters_french_majority": int((fr_counts / sizes >= 0.5).sum()),
    }
    return out


# ── per-cluster assessment & tiering ──────────────────────────────────────────

def keyword_support(texts: list[str], keywords: list[str]) -> dict[str, float]:
    """Share of member texts containing each keyword (word-boundary match; phrases by substring)."""
    low = [_norm(t).lower() for t in texts]
    out = {}
    for kw in keywords:
        k = kw.lower().strip()
        if not k:
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)")
        out[kw] = round(sum(pat.search(t) is not None for t in low) / max(len(low), 1), 4)
    return out


def assess_clusters(config_id: str, labels: np.ndarray, entries: pd.DataFrame, unique: pd.DataFrame,
                    emb: np.ndarray, groups: pd.DataFrame, keywords: dict[int, list[str]],
                    seed_alternatives: list[np.ndarray] | None = None, cluster_texts: list[str] | None = None,
                    thresholds: dict[str, Any] | None = None, markup_in_input: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cluster diagnostics, flags, tier and reasons.

    ``cluster_texts`` are the texts the clustering saw (stripped for strip variants); keyword support and
    pattern shares are measured on them. ``markup_in_input`` says whether URLs/mentions/hashtags were part of the
    embedded text; only then can a markup-dominated cluster be markup-driven. Returns (cluster table, point table).
    """
    th = {**THRESHOLDS, **(thresholds or {})}
    texts = cluster_texts if cluster_texts is not None else unique["original_text"].tolist()
    geom, points = cluster_geometry(emb, labels)
    stab = cluster_stability(labels, seed_alternatives) if seed_alternatives else pd.DataFrame(
        columns=["cluster_id", "stability_jaccard_mean", "stability_jaccard_min", "stability_runs"])
    e_uid = entries["unique_id"].to_numpy()
    e_lab = labels[e_uid]
    n_entries = len(entries)
    grp = groups.set_index("unique_id")
    rows = []
    for c in sorted(int(x) for x in np.unique(labels[labels >= 0])):
        members = np.where(labels == c)[0]
        ent = entries[e_lab == c]
        mtexts = [texts[i] for i in members]
        g = grp.loc[members]
        group_entries = ent["unique_id"].map(grp["dup_group_id"]).value_counts()
        rep_share = float(ent["unique_id"].map(grp["repeated_template_group"]).mean())
        lang = ent["language"].map(lang_group).value_counts()
        fields = (ent["source_dataset"] + "|" + ent["question_slug"]).value_counts(normalize=True)
        comp = ent[ent["source_dataset"] == "consultation_compilation_2017_18"]["stakeholder_group"].value_counts()
        kws = keywords.get(c, [])
        sup = keyword_support(mtexts, kws[:10])
        top3 = [sup[k] for k in kws[:3] if k in sup]
        social = float(np.mean([bool(re.search(r"https?://|www\.|@\w|#\w", t)) for t in ent["original_text"]]))
        content_free = float(np.mean([len(strip_social(t)) < 20 for t in ent["original_text"]]))
        rows.append({
            "config_id": config_id, "cluster_id": c,
            "n_entries": int(len(ent)), "n_unique_texts": int(len(members)),
            "share_of_eligible_entries": round(len(ent) / n_entries, 5),
            "distinct_near_dup_groups": int(g["dup_group_id"].nunique()),
            "top_dup_group_entries": int(group_entries.iloc[0]),
            "top_dup_group_share": round(float(group_entries.iloc[0] / len(ent)), 4),
            "repeated_template_entry_share": round(rep_share, 4),
            "exact_repeat_entry_share": round(float(ent["unique_id"].map(grp["multiplicity"]).gt(1).mean()), 4),
            "english": int(lang.get("English", 0)), "french": int(lang.get("French", 0)),
            "other_unknown": int(lang.get("Other/Unknown", 0)),
            "french_share": round(float(lang.get("French", 0) / len(ent)), 4),
            "dataset_counts": ent["source_dataset"].value_counts().to_dict(),
            "top_dataset_share": round(float(ent["source_dataset"].value_counts(normalize=True).iloc[0]), 4),
            "top_field": fields.index[0], "top_field_share": round(float(fields.iloc[0]), 4),
            "compilation_entries": int(comp.sum()),
            "top_channel": comp.index[0] if len(comp) else None,
            "top_channel_share": round(float(comp.iloc[0] / comp.sum()), 4) if len(comp) else None,
            "social_markup_share": round(social, 4),
            "content_free_share": round(content_free, 4),
            "no_opinion_share": round(pattern_share(mtexts, NO_OPINION_RE), 4),
            "promotion_share": round(pattern_share(mtexts, PROMOTION_RE), 4),
            "median_chars": int(ent["original_text"].str.len().median()),
            "keywords": kws, "keyword_support": sup,
            "keyword_support_top3_mean": round(float(np.mean(top3)), 4) if top3 else 0.0,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table, points
    table = table.merge(geom, on="cluster_id", how="left").merge(stab, on="cluster_id", how="left")
    q = float(table["cohesion_mean"].quantile(th["weak_cohesion_quantile"]))
    flags_all, tiers, reasons_all = [], [], []
    for r in table.itertuples():
        f: dict[str, bool] = {
            "undersized": r.distinct_near_dup_groups < th["undersized_min_distinct_groups"],
            "markup_driven": bool(markup_in_input and r.social_markup_share >= th["social_markup_share"]),
            "unstable": bool(pd.notna(r.stability_jaccard_mean) and r.stability_jaccard_mean < th["unstable_jaccard"]),
            "stability_not_assessed": bool(pd.isna(r.stability_jaccard_mean)),
            "moderately_unstable": bool(pd.notna(r.stability_jaccard_mean)
                                        and th["unstable_jaccard"] <= r.stability_jaccard_mean < th["moderate_stability_jaccard"]),
            "template_driven": bool((r.top_dup_group_entries >= 5 and r.top_dup_group_share >= th["template_top_group_share"])
                                    or r.repeated_template_entry_share >= th["template_repeated_share"]),
            "no_opinion_pattern": r.no_opinion_share >= th["non_concern_pattern_share"],
            "promotional_pattern": bool(r.promotion_share >= th["non_concern_pattern_share"]
                                        or (r.social_markup_share >= th["social_markup_share"]
                                            and (r.promotion_share >= th["promotion_with_markup_share"]
                                                 or r.content_free_share >= th["content_free_share"]))),
            "channel_concentrated": bool(r.compilation_entries >= th["channel_min_entries"]
                                         and (r.top_channel_share or 0) >= th["channel_share"]),
            "mixed": bool((pd.notna(r.negative_margin_share) and r.negative_margin_share >= th["mixed_negative_margin_share"])
                          or r.keyword_support_top3_mean < th["mixed_keyword_support"]),
            "weak_cohesion": r.cohesion_mean <= q,
            "broad": r.share_of_eligible_entries >= th["broad_share"],
            "french_majority": r.french_share >= th["french_majority_share"],
            "single_field": r.top_field_share >= th["single_field_share"],
            "single_instrument": r.top_dataset_share >= th["single_instrument_share"],
        }
        if f["undersized"] or f["unstable"] or f["markup_driven"]:
            tier = "unreliable"
        elif f["template_driven"]:
            tier = "campaign_or_template"
        elif f["no_opinion_pattern"] or f["promotional_pattern"]:
            tier = "non_concern_pattern"
        elif (f["mixed"] or f["weak_cohesion"] or f["broad"] or f["moderately_unstable"] or f["channel_concentrated"]
              or f["french_majority"] or f["stability_not_assessed"]):
            tier = "needs_review"
        else:
            tier = "candidate"
        flags_all.append(f)
        tiers.append(tier)
        reasons_all.append([k for k, v in f.items() if v])
    for k in flags_all[0]:
        table[f"flag_{k}"] = [f[k] for f in flags_all]
    table["tier"] = tiers
    table["flags"] = reasons_all
    table["weak_cohesion_threshold"] = round(q, 4)
    return table, points


def partition_summary(table: pd.DataFrame, labels: np.ndarray, entries: pd.DataFrame,
                      core_unique: np.ndarray | None = None) -> dict[str, Any]:
    """Coverage, noise and tier composition at entry level (and on the comparable core if given)."""
    e_lab = labels[entries["unique_id"].to_numpy()]
    in_pop = e_lab >= -1
    tier_of = dict(zip(table["cluster_id"], table["tier"])) if len(table) else {}
    tiers = np.array([tier_of.get(int(x), "noise" if x == -1 else "excluded") for x in e_lab])
    out: dict[str, Any] = {
        "entries": int(len(entries)), "clusters": int(len(table)),
        "excluded_by_preprocessing": int((~in_pop).sum()),
        "noise_entries": int((e_lab == -1).sum()),
        "noise_share_of_population": round(float((e_lab == -1).sum() / max(in_pop.sum(), 1)), 4),
        "coverage_all_entries": round(float((e_lab >= 0).mean()), 4),
        "tier_clusters": table["tier"].value_counts().reindex(TIER_ORDER, fill_value=0).to_dict() if len(table) else {},
        "tier_entries": {t: int((tiers == t).sum()) for t in TIER_ORDER},
    }
    usable = np.isin(tiers, ["candidate", "needs_review", "campaign_or_template"])
    broad_ids = set(table.loc[table["flag_broad"], "cluster_id"]) if len(table) and "flag_broad" in table else set()
    usable_nb = usable & ~np.isin(e_lab, list(broad_ids))
    cand = tiers == "candidate"
    out["usable_concern_coverage_all"] = round(float(usable.mean()), 4)
    out["usable_non_broad_coverage_all"] = round(float(usable_nb.mean()), 4)
    out["candidate_coverage_all"] = round(float(cand.mean()), 4)
    if core_unique is not None:
        core = np.asarray(core_unique)[entries["unique_id"].to_numpy()]
        out["core_entries"] = int(core.sum())
        out["usable_concern_coverage_core"] = round(float(usable[core].mean()), 4)
        out["usable_non_broad_coverage_core"] = round(float(usable_nb[core].mean()), 4)
        out["candidate_coverage_core"] = round(float(cand[core].mean()), 4)
        out["noise_share_core"] = round(float((e_lab[core] == -1).mean()), 4)
    return out
