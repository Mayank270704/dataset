"""
scripts/run_clustering_experiments_v2.py
========================================
Controlled, reproducible UNSUPERVISED clustering experiments on the existing full-corpus run
(data/full_corpus/run_v1). Nothing from run_v1 or the 450-comment baseline is modified.

Population
----------
All experiments start from the same 10,108 eligible entries / 10,043 unique texts and the
cached multilingual embeddings of run_v1. Variants may EXCLUDE texts from clustering (never silently):
    raw          text as embedded in run_v1 (reference; reproduces run_v1 exactly)
    strip        URLs, @mentions and #hashtags removed before embedding; texts left with < 20
                 characters are "content-free after preprocessing" and not clustered
    strip_dedup  strip + identical stripped texts collapsed to one representative (templates);
                 the other entries inherit the representative's cluster
    no_short     strip + only texts with >= 40 characters are clustered
Comparable core C = unique texts with >= 40 characters after stripping; every variant is also
reported on C so that populations are comparable.

Algorithms: UMAP(5-D, exact cosine kNN) + HDBSCAN (min_cluster_size 25/50, EOM or leaf),
PCA(20-D)+HDBSCAN (direct, no UMAP), spherical k-means (direct), English-only / French-only runs.
Stability: UMAP seeds 42/43/44. No human labels are used; every metric is a proxy. No cluster is a
validated concern.

Commands:
    python scripts/run_clustering_experiments_v2.py run
    python scripts/run_clustering_experiments_v2.py report
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:  # torch must be imported before psutil/pandas here: the reverse order makes torch's
    import torch  # noqa: F401  # storage code crash with an access violation on this machine
except ImportError:  # pragma: no cover - torch is optional for non-embedding stages
    pass
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import run_full_corpus_clustering as fc  # noqa: E402

RUN_V1 = fc.RUN_ROOT
OUT_ROOT = ROOT / "data" / "full_corpus" / "experiments_v2"
REPORT_PATH = ROOT / "reports" / "clustering_experiments_v2_report.md"
SEEDS = (42, 43, 44)
CORE_MIN_CHARS = 40
BROAD_CLUSTER_SHARE = 0.08          # a cluster holding >= 8% of eligible entries is flagged as broad
CONTENT_MIN_CHARS = 20

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")

UMAP_BASE = {"n_components": 5, "n_neighbors": 15, "min_dist": 0.0, "metric": "cosine"}
ALGOS: dict[str, dict[str, Any]] = {
    "umap_m25s5": {"kind": "umap_hdbscan", "umap": UMAP_BASE, "hdb": {"min_cluster_size": 25, "min_samples": 5, "selection": "eom"}},
    "umap_m50s10": {"kind": "umap_hdbscan", "umap": UMAP_BASE, "hdb": {"min_cluster_size": 50, "min_samples": 10, "selection": "eom"}},
    "umap_m25s5_leaf": {"kind": "umap_hdbscan", "umap": UMAP_BASE, "hdb": {"min_cluster_size": 25, "min_samples": 5, "selection": "leaf"}},
    "umap_m10s5": {"kind": "umap_hdbscan", "umap": UMAP_BASE, "hdb": {"min_cluster_size": 10, "min_samples": 5, "selection": "eom"}},
    "pca20_hdb_m25s5": {"kind": "pca_hdbscan", "pca_components": 20, "hdb": {"min_cluster_size": 25, "min_samples": 5, "selection": "eom"}},
    "kmeans_k40": {"kind": "kmeans", "k": 40},
    "kmeans_k80": {"kind": "kmeans", "k": 80},
}


def _exp(variant: str, algo: str, seeds=(42,), subset: str | None = None) -> dict[str, Any]:
    sub = f"_{subset.lower()}only" if subset else ""
    return {"id": f"{variant}__{algo}{sub}", "variant": variant, "algo": algo, "seeds": list(seeds), "subset": subset}


EXPERIMENTS: list[dict[str, Any]] = (
    [_exp(v, a, SEEDS) for v in ("raw", "strip") for a in ("umap_m25s5", "umap_m50s10")]
    + [_exp(v, a) for v in ("strip_dedup", "no_short") for a in ("umap_m25s5", "umap_m50s10")]
    + [_exp("strip", "umap_m25s5_leaf"), _exp("raw", "pca20_hdb_m25s5"), _exp("strip", "pca20_hdb_m25s5"),
       _exp("raw", "kmeans_k40"), _exp("strip", "kmeans_k40"), _exp("strip", "kmeans_k80"),
       _exp("strip", "umap_m25s5", subset="English"), _exp("strip", "umap_m10s5", subset="French")]
)
DIRECT_ALGOS = {"pca20_hdb_m25s5", "kmeans_k40", "kmeans_k80"}

# state codes in unique-level label arrays
NOISE, CONTENT_FREE, SHORT_EXCLUDED, OUT_OF_SUBSET = -1, -2, -3, -4
STATE_NAMES = {NOISE: "noise", CONTENT_FREE: "content_free_after_preprocessing", SHORT_EXCLUDED: "below_min_length",
               OUT_OF_SUBSET: "outside_language_subset"}


# ── preprocessing ─────────────────────────────────────────────────────────────

def strip_social(text: str) -> str:
    """Remove URLs, @mentions and #hashtags; collapse whitespace. Pure and deterministic."""
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def entry_features(entries: pd.DataFrame) -> pd.DataFrame:
    e = entries.copy()
    t = e["original_text"]
    e["has_url"] = t.str.contains(URL_RE)
    e["has_mention"] = t.str.contains(MENTION_RE)
    e["has_hashtag"] = t.str.contains(HASHTAG_RE)
    e["social_markup"] = e["has_url"] | e["has_mention"] | e["has_hashtag"]
    e["text_len"] = t.str.len()
    e["skey"] = t.map(strip_social).map(fc.normalise_text)
    e["skey_count"] = e.groupby("skey")["corpus_id"].transform("size")
    e["exact_dup_count"] = e.groupby("text_key")["corpus_id"].transform("size")
    return e


class Variant:
    """Texts, vectors and population masks for one preprocessing variant (aligned to unique_id)."""

    def __init__(self, name: str, text: list[str], emb: np.ndarray, state: np.ndarray, rep_of: np.ndarray):
        self.name, self.text, self.emb, self.state, self.rep_of = name, text, emb, state, rep_of

    @property
    def pop_idx(self) -> np.ndarray:
        return np.where((self.state == 0) & (self.rep_of == np.arange(len(self.state))))[0]


def build_variants(unique: pd.DataFrame, emb_raw: np.ndarray,
                   encode: Callable[[list[str]], np.ndarray] | None,
                   cache_dir: Path | None = None) -> dict[str, Variant]:
    n = len(unique)
    raw_text = unique["original_text"].tolist()
    stripped = [strip_social(t) for t in raw_text]
    ident = np.arange(n)

    changed = np.array([i for i in range(n) if stripped[i] != raw_text[i] and len(stripped[i]) >= CONTENT_MIN_CHARS], dtype=int)
    emb_strip = emb_raw.copy()
    if len(changed):
        cached = None
        if cache_dir is not None and (cache_dir / "strip_embeddings.npz").exists():
            z = np.load(cache_dir / "strip_embeddings.npz")
            if np.array_equal(z["idx"], changed):
                cached = z["emb"]
        if cached is None:
            if encode is None:
                raise RuntimeError("Stripped-text embeddings are not cached and no encoder was supplied.")
            cached = np.vstack([encode([stripped[i] for i in changed[k:k + 128]]) for k in range(0, len(changed), 128)]).astype(np.float32)
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                fc.write_new(cache_dir / "strip_embeddings.npz", lambda f: np.savez(f, idx=changed, emb=cached), binary=True)
        emb_strip[changed] = cached

    lens = np.array([len(s) for s in stripped])
    state_raw = np.zeros(n, dtype=int)
    state_strip = np.where(lens < CONTENT_MIN_CHARS, CONTENT_FREE, 0)
    state_short = np.where(state_strip != 0, state_strip, np.where(lens < CORE_MIN_CHARS, SHORT_EXCLUDED, 0))
    # template collapse among stripped texts that are clustered
    key_first: dict[str, int] = {}
    rep = ident.copy()
    for i in range(n):
        if state_strip[i] != 0:
            continue
        k = fc.normalise_text(stripped[i])
        rep[i] = key_first.setdefault(k, i)
    return {
        "raw": Variant("raw", raw_text, emb_raw, state_raw, ident),
        "strip": Variant("strip", stripped, emb_strip, state_strip, ident),
        "strip_dedup": Variant("strip_dedup", stripped, emb_strip, state_strip, rep),
        "no_short": Variant("no_short", stripped, emb_strip, state_short, ident),
    }


def core_mask(variants: dict[str, Variant]) -> np.ndarray:
    """Comparable core C: >= 40 characters after stripping (identical for all variants)."""
    return np.array([len(t) >= CORE_MIN_CHARS for t in variants["strip"].text])


# ── clustering ────────────────────────────────────────────────────────────────

def run_algo(emb: np.ndarray, algo: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    kind = algo["kind"]
    if kind in ("umap_hdbscan", "pca_hdbscan"):
        import hdbscan
        if kind == "umap_hdbscan":
            red = fc.umap_reduce_exact_knn(emb, random_state=seed, **algo["umap"])
        else:
            from sklearn.decomposition import PCA
            red = PCA(n_components=algo["pca_components"], random_state=seed).fit_transform(emb).astype(np.float32)
        h = algo["hdb"]
        cl = hdbscan.HDBSCAN(min_cluster_size=h["min_cluster_size"], min_samples=h["min_samples"], metric="euclidean",
                             cluster_selection_method=h["selection"], gen_min_span_tree=True)
        labels = cl.fit_predict(red)
        try:
            rv = float(cl.relative_validity_)
        except Exception:  # noqa: BLE001
            rv = None
        return labels.astype(int), cl.probabilities_.astype(np.float32), rv
    if kind == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=algo["k"], n_init=3, random_state=seed)
        return km.fit_predict(emb).astype(int), None, None
    raise ValueError(kind)


def cluster_experiment(exp: dict[str, Any], variants: dict[str, Variant], unique: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (unique-level labels with state codes, probabilities (nan where unavailable), relative validity)."""
    v = variants[exp["variant"]]
    n = len(unique)
    pop = v.pop_idx
    if exp["subset"]:
        lang = unique["language"].to_numpy()
        keep = lang[pop] == exp["subset"]
        pop = pop[keep]
    labels = np.full(n, OUT_OF_SUBSET if exp["subset"] else NOISE, dtype=int)
    probs = np.full(n, np.nan, dtype=np.float32)
    in_pop_state = v.state.copy()
    if exp["subset"]:
        labels[:] = np.where(in_pop_state != 0, in_pop_state, OUT_OF_SUBSET)
    else:
        labels[:] = np.where(in_pop_state != 0, in_pop_state, NOISE)
    lab_pop, prob_pop, rv = run_algo(v.emb[pop], ALGOS[exp["algo"]], seed)
    labels[pop] = lab_pop
    if prob_pop is not None:
        probs[pop] = prob_pop
    # template-collapsed entries inherit the representative's cluster
    if exp["variant"] == "strip_dedup":
        inherit = (v.rep_of != np.arange(n)) & (v.state == 0)
        labels[inherit] = labels[v.rep_of[inherit]]
        probs[inherit] = probs[v.rep_of[inherit]]
    return labels, probs, rv


# ── evaluation ────────────────────────────────────────────────────────────────

def keyword_support(texts: list[str], keywords: list[str]) -> dict[str, float]:
    low = [t.lower() for t in texts]
    out = {}
    for kw in keywords:
        k = kw.lower()
        pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)") if " " not in k else None
        hits = sum((pat.search(t) is not None) if pat else (k in t) for t in low)
        out[kw] = round(hits / max(len(low), 1), 3)
    return out


def descriptive_label(keywords: list[str]) -> str:
    top = [k for k in keywords[:3]]
    return ("Keyword group (descriptive only, not a validated concern): " + ", ".join(top)) if top else "No keywords extracted"


def build_cluster_table(config_id: str, labels_u: np.ndarray, probs_u: np.ndarray, entries: pd.DataFrame,
                        unique: pd.DataFrame, emb: np.ndarray, variant_text: list[str],
                        npmi_docs: list[str]) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    from models.concern_clustering.clustering import extract_cluster_topics
    n_entries = len(entries)
    e = entries.copy()
    uid = e["unique_id"].to_numpy()
    e["cluster"] = labels_u[uid]
    clustered_ids = sorted(int(c) for c in set(labels_u[labels_u >= 0]))
    meta = pd.DataFrame({"original_text": variant_text, "language": unique["language"].to_numpy(),
                         "source_dataset": unique["source_dataset"].to_numpy()})
    topics = extract_cluster_topics(meta, np.where(labels_u >= 0, labels_u, -1)) if clustered_ids else pd.DataFrame()
    kw_map = {int(r.cluster_id): [t.strip() for t in r.top_keywords.split(",") if t.strip()]
              for r in topics.itertuples() if r.cluster_id != -1} if len(topics) else {}
    npmi = fc.npmi_coherence(npmi_docs, kw_map) if kw_map else {}
    rows = []
    for cid in clustered_ids:
        g = e[e["cluster"] == cid]
        members = np.where(labels_u == cid)[0]
        cen = emb[members].mean(axis=0)
        cen /= max(np.linalg.norm(cen), 1e-12)
        sims = emb[members] @ cen
        order = np.argsort(-sims)
        pick = list(order[:3]) + [o for o in order[::-1][:1] if o not in order[:3]]
        reps = []
        for o in pick:
            r = unique.iloc[int(members[o])]
            reps.append({"corpus_id": int(r["corpus_id"]), "source_dataset": r["source_dataset"],
                         "source_record_id": r["source_record_id"], "question_slug": r["question_slug"],
                         "language": r["language"], "cosine_to_centroid": round(float(sims[o]), 4),
                         "kind": "central" if o in order[:3] else "peripheral", "text": r["original_text"]})
        lang = g["language"].value_counts()
        comp = g[g["source_dataset"] == "consultation_compilation_2017_18"]["stakeholder_group"]
        top_chan = comp.value_counts()
        tmpl = g["skey"].value_counts()
        kws = kw_map.get(cid, [])
        sup = keyword_support([variant_text[i] for i in members], kws[:5])
        sup_top3 = float(np.mean([sup[k] for k in kws[:3]])) if kws else 0.0
        probs = probs_u[members]
        row = {
            "config_id": config_id, "cluster_id": cid, "n_entries": int(len(g)), "n_unique_texts": int(len(members)),
            "share_of_eligible_entries": round(len(g) / n_entries, 5), "state": "clustered",
            "mean_probability": None if np.isnan(probs).all() else round(float(np.nanmean(probs)), 3),
            "cohesion_mean_cosine_to_centroid": round(float(sims.mean()), 4),
            "cohesion_p10": round(float(np.percentile(sims, 10)), 4),
            "english": int((g["language"] == "English").sum()), "french": int((g["language"] == "French").sum()),
            "other_unknown": int((~g["language"].isin(["English", "French"])).sum()),
            "french_share": round(float((g["language"] == "French").mean()), 3),
            "dataset_counts": json.dumps(g["source_dataset"].value_counts().to_dict()),
            "field_counts": json.dumps(g["question_slug"].value_counts().to_dict()),
            "top_dataset_share": round(float(g["source_dataset"].value_counts(normalize=True).iloc[0]), 3),
            "exact_duplicate_entry_share": round(float((g["exact_dup_count"] > 1).mean()), 3),
            "url_share": round(float(g["has_url"].mean()), 3), "hashtag_share": round(float(g["has_hashtag"].mean()), 3),
            "mention_share": round(float(g["has_mention"].mean()), 3),
            "social_markup_share": round(float(g["social_markup"].mean()), 3),
            "top_template_count": int(tmpl.iloc[0]), "top_template_share": round(float(tmpl.iloc[0] / len(g)), 3),
            "short_text_share": round(float((g["text_len"] < 40).mean()), 3), "median_chars": int(g["text_len"].median()),
            "compilation_entries": int(len(comp)),
            "top_channel": None if not len(top_chan) else top_chan.index[0],
            "top_channel_share": None if not len(top_chan) else round(float(top_chan.iloc[0] / len(comp)), 3),
            "keywords": ", ".join(kws), "keyword_support": json.dumps(sup, ensure_ascii=False),
            "keyword_support_top3_mean": round(sup_top3, 3),
            "npmi": None if npmi.get(cid) is None else round(float(npmi[cid]), 3),
            "label": descriptive_label(kws),
            "representatives": json.dumps(reps, ensure_ascii=False),
        }
        row["flag_social_markup"] = bool(row["social_markup_share"] >= 0.5)
        row["flag_template_repeat"] = bool(row["top_template_count"] >= 5 and row["top_template_share"] >= 0.3)
        row["flag_channel_concentrated"] = bool(row["compilation_entries"] >= 20 and (row["top_channel_share"] or 0) >= 0.7)
        row["flag_short_responses"] = bool(row["median_chars"] < 40)
        row["flag_single_instrument"] = bool(row["top_dataset_share"] >= 0.95)
        row["flag_broad_cluster"] = bool(row["share_of_eligible_entries"] >= BROAD_CLUSTER_SHARE)
        row["likely_channel_template_driven"] = bool(row["flag_social_markup"] or row["flag_template_repeat"]
                                                      or row["flag_channel_concentrated"])
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        q25 = float(table["cohesion_mean_cosine_to_centroid"].quantile(0.25))
        low = (table["likely_channel_template_driven"] | table["flag_broad_cluster"]
               | (table["keyword_support_top3_mean"] < 0.15) | (table["cohesion_mean_cosine_to_centroid"] <= q25))
        table["proxy_coherence"] = np.where(low, "low_or_mixed", "moderate")
    return table, kw_map


def comparable_metrics(labels_u: np.ndarray, probs_u: np.ndarray, entries: pd.DataFrame, table: pd.DataFrame,
                       core_u: np.ndarray, emb: np.ndarray, rv: float | None, subset: str | None = None) -> dict[str, Any]:
    e = entries
    uid = e["unique_id"].to_numpy()
    lab = labels_u[uid]
    in_core = core_u[uid]
    if subset:                                   # subset runs are evaluated within their own language
        in_core = in_core & (e["language"].to_numpy() == subset)
    clustered = lab >= 0
    driven_ids = set(table.loc[table["likely_channel_template_driven"], "cluster_id"]) if len(table) else set()
    low_ids = set(table.loc[table["proxy_coherence"] == "low_or_mixed", "cluster_id"]) if len(table) else set()
    cc = clustered & in_core
    lang = e["language"].to_numpy()
    out: dict[str, Any] = {
        "entries": int(len(e)), "n_clusters": int(len(set(lab[clustered]))),
        "coverage_all_entries": round(float(clustered.mean()), 4),
        "core_entries": int(in_core.sum()),
        "coverage_core": round(float(cc.sum() / max(in_core.sum(), 1)), 4),
        "noise_core": round(float(((lab == NOISE) & in_core).sum() / max(in_core.sum(), 1)), 4),
        "states": {STATE_NAMES.get(s, str(s)): int((lab == s).sum()) for s in sorted(set(lab[lab < 0]))},
        "size_unique": fc.size_stats(pd.Series(labels_u[labels_u >= 0]).value_counts().tolist()),
        "core_entries_in_driven_clusters_share": round(float(np.isin(lab[cc], list(driven_ids)).mean()) if cc.any() else 0.0, 4),
        "core_entries_in_low_proxy_clusters_share": round(float(np.isin(lab[cc], list(low_ids)).mean()) if cc.any() else 0.0, 4),
        "clusters_driven": int(len(driven_ids)), "clusters_low_or_mixed": int(len(low_ids)),
        "effective_substantive_coverage_core": round(float(cc.sum() / max(in_core.sum(), 1)) *
                                                     (1 - (float(np.isin(lab[cc], list(driven_ids)).mean()) if cc.any() else 0.0)), 4),
        "cohesion_entry_weighted_mean": None if not len(table) else round(float(np.average(
            table["cohesion_mean_cosine_to_centroid"], weights=table["n_entries"])), 4),
        "mean_npmi": None if not len(table) or table["npmi"].isna().all() else round(float(table["npmi"].mean()), 4),
        "hdbscan_relative_validity": None if rv is None else round(rv, 4),
        "clusters_ge90pct_single_language": int(((table[["english", "french", "other_unknown"]].max(axis=1) /
                                                  table["n_entries"]) >= 0.9).sum()) if len(table) else 0,
        "clusters_ge90pct_french": int((table["french_share"] >= 0.9).sum()) if len(table) else 0,
    }
    for lg in ("English", "French"):
        m = (lang == lg) & in_core
        out[f"coverage_core_{lg.lower()}"] = round(float(((lab >= 0) & m).sum() / max(m.sum(), 1)), 4)
        out[f"core_entries_{lg.lower()}"] = int(m.sum())
    return out


def partition_ari(a_u: np.ndarray, b_u: np.ndarray, core_u: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score
    both = (a_u >= 0) & (b_u >= 0) & core_u
    if both.sum() < 2:
        return {"ari": None, "n": int(both.sum())}
    return {"ari": round(float(adjusted_rand_score(a_u[both], b_u[both])), 4), "n": int(both.sum()),
            "share_of_core": round(float(both.sum() / max(core_u.sum(), 1)), 4)}


# ── orchestration ─────────────────────────────────────────────────────────────

def load_base(run_v1: Path = RUN_V1):
    entries = pd.read_csv(run_v1 / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    entries["unique_id"] = entries["unique_id"].astype(int)
    entries["corpus_id"] = entries["corpus_id"].astype(int)
    unique = pd.read_csv(run_v1 / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    emb = np.load(run_v1 / "embeddings" / "multilingual" / "embeddings.npy")
    if len(unique) != len(emb):
        raise ValueError("Embeddings are not aligned with unique_texts.csv")
    return entry_features(entries), unique, emb


def default_encoder() -> Callable[[list[str]], np.ndarray]:
    import torch
    from models.semantic_similarity.minilm import mean_pool_and_normalize
    # NOTE: torch.set_num_threads must be called AFTER the weights are read; calling it first makes
    # safetensors' get_tensor crash with an access violation on this machine/torch build.
    tok, model, _ = fc.load_encoder_low_memory(fc.MODELS["multilingual"])
    torch.set_num_threads(6)
    return lambda batch: mean_pool_and_normalize(batch, tok, model, max_length=fc.MAX_LENGTH, batch_size=fc.BATCH_SIZE, device="cpu")


def preprocess_stage(out_root: Path = OUT_ROOT, run_v1: Path = RUN_V1,
                     encode: Callable[[list[str]], np.ndarray] | None = None) -> dict[str, Any]:
    """Embed only the texts changed by URL/mention/hashtag stripping, in a minimal-memory process.

    Kept separate from clustering so the encoder weights are loaded before any large objects exist
    (loading crashed with an access violation when attempted inside the heavy process on this machine).
    """
    cache = out_root / "preprocess" / "strip_embeddings.npz"
    unique = pd.read_csv(run_v1 / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    raw = unique["original_text"].tolist()
    stripped = [strip_social(t) for t in raw]
    changed = np.array([i for i in range(len(raw)) if stripped[i] != raw[i] and len(stripped[i]) >= CONTENT_MIN_CHARS], dtype=int)
    if cache.exists():
        z = np.load(cache)
        return {"status": "cached", "changed_texts": int(len(z["idx"])), "matches": bool(np.array_equal(z["idx"], changed))}
    t0 = time.perf_counter()
    # equivalence probe: unchanged texts whose full-model vectors already exist in run_v1
    unchanged = [i for i in range(len(raw)) if stripped[i] == raw[i]][::max(1, len(raw) // 64)][:64]
    to_encode = [stripped[i] for i in changed] + [raw[i] for i in unchanged]
    enc = encode or default_encoder()
    vectors = np.vstack([enc(to_encode[k:k + 64]) for k in range(0, len(to_encode), 64)]).astype(np.float32)
    probe = vectors[len(changed):]
    ref = np.load(run_v1 / "embeddings" / "multilingual" / "embeddings.npy")[unchanged] if unchanged else probe
    max_diff = float(np.abs(probe - ref).max()) if len(unchanged) else None
    fc.write_new(cache, lambda f: np.savez(f, idx=changed, emb=vectors[:len(changed)]), binary=True)
    return {"status": "computed", "changed_texts": int(len(changed)), "equivalence_probe_texts": len(unchanged),
            "equivalence_probe_max_abs_diff_vs_run_v1": max_diff,
            "elapsed_s": round(time.perf_counter() - t0, 1), "peak_rss_mb": fc.peak_rss_mb()}


def run_experiments(out_root: Path = OUT_ROOT, run_v1: Path = RUN_V1, experiments: list[dict[str, Any]] | None = None,
                    encode: Callable[[list[str]], np.ndarray] | None = None) -> dict[str, Any]:
    experiments = experiments or EXPERIMENTS
    t_start = time.perf_counter()
    entries, unique, emb_raw = load_base(run_v1)
    cache_dir = out_root / "preprocess"
    needs_encode = not (cache_dir / "strip_embeddings.npz").exists()
    if needs_encode and encode is None:
        raise RuntimeError("Stripped-text embeddings are not cached. Run `python scripts/run_clustering_experiments_v2.py preprocess` "
                           "first (a separate low-memory process) before `run`.")
    variants = build_variants(unique, emb_raw, encode, cache_dir)
    encode = None          # release the encoder (and its weights) before clustering
    import gc
    gc.collect()
    core_u = core_mask(variants)
    docs_by_variant = {k: v.text for k, v in variants.items()}
    summary: dict[str, Any] = {"experiments": {}, "core_unique_texts": int(core_u.sum()),
                               "variant_states": {k: {STATE_NAMES.get(s, "clustered"): int((v.state == s).sum())
                                                      for s in sorted(set(v.state))} for k, v in variants.items()},
                               "strip_changed_texts": int(sum(a != b for a, b in zip(variants["strip"].text, variants["raw"].text))),
                               "template_collapsed_unique_texts": int((variants["strip_dedup"].rep_of != np.arange(len(unique))).sum())}
    labels_cache: dict[tuple[str, int], np.ndarray] = {}
    for exp in experiments:
        edir = out_root / exp["id"]
        mpath = edir / "metrics.json"
        if mpath.exists():
            summary["experiments"][exp["id"]] = json.loads(mpath.read_text(encoding="utf-8"))
            for s in exp["seeds"]:
                p = edir / f"labels_seed{s}.npy"
                if p.exists():
                    labels_cache[(exp["id"], s)] = np.load(p)
            continue
        t0 = time.perf_counter()
        v = variants[exp["variant"]]
        results = {}
        for s in exp["seeds"]:
            lab, prob, rv = cluster_experiment(exp, variants, unique, s)
            results[s] = (lab, prob, rv)
            labels_cache[(exp["id"], s)] = lab
        lab0, prob0, rv0 = results[exp["seeds"][0]]
        table, kw_map = build_cluster_table(exp["id"], lab0, prob0, entries, unique, v.emb, v.text, docs_by_variant[exp["variant"]])
        metrics = comparable_metrics(lab0, prob0, entries, table, core_u, v.emb, rv0, exp["subset"])
        seed_pairs = {}
        for a, b in itertools.combinations(exp["seeds"], 2):
            seed_pairs[f"{a}-{b}"] = partition_ari(results[a][0], results[b][0], core_u)
        metrics["seed_stability"] = seed_pairs
        aris = [p["ari"] for p in seed_pairs.values() if p["ari"] is not None]
        metrics["seed_ari_mean"] = None if not aris else round(float(np.mean(aris)), 4)
        metrics["seed_ari_min"] = None if not aris else round(float(np.min(aris)), 4)
        metrics["seed_cluster_counts"] = {str(s): int(len(set(results[s][0][results[s][0] >= 0]))) for s in exp["seeds"]}
        metrics["experiment"] = exp
        metrics["runtime_s"] = round(time.perf_counter() - t0, 1)
        metrics["peak_rss_mb"] = fc.peak_rss_mb()
        ea = entries[["corpus_id", "unique_id", "source_dataset", "source_record_id", "question_slug", "language"]].copy()
        ea["cluster_id"] = lab0[ea["unique_id"].to_numpy()]
        ea["state"] = [STATE_NAMES.get(int(c), "clustered") if c < 0 else "clustered" for c in ea["cluster_id"]]
        ea["probability"] = prob0[ea["unique_id"].to_numpy()].round(4)
        fc.write_new(edir / "entry_assignments.csv", lambda f: ea.to_csv(f, index=False, lineterminator="\n"))
        fc.write_new(edir / "cluster_table.csv", lambda f: table.to_csv(f, index=False, lineterminator="\n"))
        for s, (lab, _, _) in results.items():
            fc.write_new(edir / f"labels_seed{s}.npy", lambda f, lab=lab: np.save(f, lab), binary=True)
        fc.write_json_new(mpath, metrics)
        summary["experiments"][exp["id"]] = metrics
    # cross-experiment agreement on the comparable core
    def lab42(i):
        return labels_cache.get((i, 42))
    pairs = {}
    for a, b in (("raw__umap_m25s5", "strip__umap_m25s5"), ("raw__umap_m50s10", "strip__umap_m50s10"),
                 ("strip__umap_m25s5", "strip_dedup__umap_m25s5"), ("strip__umap_m25s5", "no_short__umap_m25s5"),
                 ("strip__umap_m25s5", "strip__kmeans_k40"), ("strip__umap_m25s5", "strip__pca20_hdb_m25s5")):
        if lab42(a) is not None and lab42(b) is not None:
            pairs[f"{a} vs {b}"] = partition_ari(lab42(a), lab42(b), core_u)
    # language: joint vs separate, restricted to the language's own core texts
    lang = unique["language"].to_numpy()
    for sub, other in (("English", "strip__umap_m25s5_englishonly"), ("French", "strip__umap_m10s5_frenchonly")):
        a, b = lab42("strip__umap_m25s5" if sub == "English" else "strip__umap_m25s5"), lab42(other)
        if a is not None and b is not None:
            pairs[f"joint strip__umap_m25s5 vs {other} ({sub} texts only)"] = partition_ari(a, b, core_u & (lang == sub))
    summary["cross_experiment_ari_on_core"] = pairs
    # regression: raw variant must reproduce run_v1 exactly
    repro = {}
    for cfg, v1cfg in (("raw__umap_m50s10", "u15_m50s10"), ("raw__umap_m25s5", "u15_m25s5")):
        p = run_v1 / "clustering" / "multilingual" / v1cfg / "entry_assignments.csv"
        if p.exists() and lab42(cfg) is not None:
            old = pd.read_csv(p, usecols=["unique_id", "cluster_id"]).drop_duplicates("unique_id").sort_values("unique_id")
            new = lab42(cfg)[old["unique_id"].to_numpy()]
            repro[cfg] = {"vs": f"run_v1/{v1cfg}", "identical_labels": bool(np.array_equal(new, old["cluster_id"].to_numpy()))}
    summary["reproduces_run_v1"] = repro
    summary["runtime_total_s"] = round(time.perf_counter() - t_start, 1)
    summary["peak_rss_mb"] = fc.peak_rss_mb()
    summary["system"] = fc.system_snapshot()
    sp = out_root / "experiments_summary.json"
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    fc.write_json_new(sp, summary)
    return summary


# ── shortlist, summaries, report ──────────────────────────────────────────────

def shortlist(summary: dict[str, Any], min_seed_ari: float = 0.6, top: int = 2) -> list[str]:
    cand = []
    for eid, m in summary["experiments"].items():
        if m["experiment"]["algo"] in DIRECT_ALGOS or m["experiment"]["subset"]:
            continue
        if m.get("seed_ari_mean") is None or m["seed_ari_mean"] < min_seed_ari:
            continue
        cand.append((m["effective_substantive_coverage_core"], m["seed_ari_mean"], eid))
    return [c[2] for c in sorted(cand, reverse=True)[:top]]


def evidence_summaries(table: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for r in table.itertuples():
        reps = json.loads(r.representatives)
        notes = []
        if r.likely_channel_template_driven:
            why = [n for n, f in (("social-media markup", r.flag_social_markup), ("repeated template text", r.flag_template_repeat),
                                  ("one source channel dominates", r.flag_channel_concentrated)) if f]
            notes.append("likely channel/template-driven: " + ", ".join(why))
        if r.keyword_support_top3_mean < 0.15:
            notes.append("top keywords appear in few member texts (mean support %.2f)" % r.keyword_support_top3_mean)
        if getattr(r, "flag_broad_cluster", False):
            notes.append("very large cluster (>= %d%% of eligible entries); large clusters tend to be generic" % int(BROAD_CLUSTER_SHARE * 100))
        if r.flag_short_responses:
            notes.append("median response is under 40 characters")
        if r.flag_single_instrument:
            notes.append("almost entirely from a single instrument")
        if r.proxy_coherence == "low_or_mixed" and not notes:
            notes.append("cohesion among the lowest in this configuration")
        quotes = "; ".join(f"[corpus_id {q['corpus_id']}] “{q['text'][:160]}{'…' if len(q['text']) > 160 else ''}”" for q in reps if q["kind"] == "central")
        text = (f"Cluster {r.cluster_id}: {r.n_entries} entries ({100 * r.share_of_eligible_entries:.1f}% of eligible). "
                f"Frequent terms: {r.keywords}. Most central verbatim examples: {quotes}. "
                f"Language: {r.english} English / {r.french} French / {r.other_unknown} other. "
                f"Proxy coherence: {r.proxy_coherence} (descriptive; not validated)."
                + (" Uncertainty: " + "; ".join(notes) + "." if notes else ""))
        out.append({"cluster_id": int(r.cluster_id), "n_entries": int(r.n_entries), "label": r.label, "summary": text,
                    "proxy_coherence": r.proxy_coherence, "uncertainty_flags": notes,
                    "supporting_corpus_ids": [q["corpus_id"] for q in reps], "keywords": r.keywords})
    return out


def write_report(out_root: Path = OUT_ROOT, report_path: Path = REPORT_PATH) -> str:
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    summary = json.loads((out_root / "experiments_summary.json").read_text(encoding="utf-8"))
    ex = summary["experiments"]
    picks = shortlist(summary)
    ref_id = "raw__umap_m50s10"
    summaries_dir = out_root / "summaries"
    for eid in dict.fromkeys(picks + ([ref_id] if ref_id in ex else [])):
        table = pd.read_csv(out_root / eid / "cluster_table.csv", keep_default_na=False)
        for col in ("likely_channel_template_driven",):
            table[col] = table[col].astype(str).str.lower() == "true"
        fc.write_json_new(summaries_dir / f"{eid}.json", {"config": eid, "unvalidated": True,
                                                         "clusters": evidence_summaries(table.replace({"": None}).assign(
                                                             flag_social_markup=lambda d: d["flag_social_markup"].astype(str).str.lower() == "true",
                                                             flag_template_repeat=lambda d: d["flag_template_repeat"].astype(str).str.lower() == "true",
                                                             flag_channel_concentrated=lambda d: d["flag_channel_concentrated"].astype(str).str.lower() == "true",
                                                             flag_short_responses=lambda d: d["flag_short_responses"].astype(str).str.lower() == "true",
                                                             flag_single_instrument=lambda d: d["flag_single_instrument"].astype(str).str.lower() == "true"))})
    L: list[str] = []
    A = L.append
    A("# PolicyLens clustering experiments v2\n")
    A("**Exploratory, unsupervised, NOT human-validated.** All figures are proxies computed without labels; no configuration is declared the winner and no cluster is a validated concern. Silhouette is not used for selection.\n")
    A("## 1. Design\n")
    A(f"- **Population:** the 10,108 eligible entries / 10,043 unique texts of `data/full_corpus/run_v1` (multilingual embeddings), for every experiment.")
    A(f"- **Variants:** `raw` (reference), `strip` (URLs/@mentions/#hashtags removed; {summary['variant_states']['strip'].get('content_free_after_preprocessing', 0)} texts content-free and not clustered), "
      f"`strip_dedup` ({summary['template_collapsed_unique_texts']} unique texts collapsed into template representatives; entries inherit the cluster), "
      f"`no_short` (texts under {CORE_MIN_CHARS} characters not clustered: {summary['variant_states']['no_short'].get('below_min_length', 0)}).")
    A(f"- **Comparable core C:** {summary['core_unique_texts']:,} unique texts (≥ {CORE_MIN_CHARS} characters after stripping). Coverage and flag shares below are computed on C so that all variants are compared on the same texts.")
    A(f"- Texts changed by stripping: {summary['strip_changed_texts']:,}; stripped embeddings were computed only for those, all others reuse run_v1 vectors.")
    rep = summary.get("reproduces_run_v1", {})
    A("- **Regression:** " + "; ".join(f"`{k}` {'reproduces' if v['identical_labels'] else 'DIFFERS FROM'} {v['vs']}" for k, v in rep.items()) + ".\n")
    A("## 2. Results (seed 42; proxies only)\n")
    A("| Experiment | Clusters | Coverage (all) | Coverage (core) | Noise (core) | Driven-cluster share | Low/mixed share | Effective substantive coverage | Cohesion | Seed ARI mean (min) | EN coverage | FR coverage | NPMI | Runtime s |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for eid, m in ex.items():
        seed = "—" if m["seed_ari_mean"] is None else f"{m['seed_ari_mean']:.2f} ({m['seed_ari_min']:.2f})"
        A(f"| `{eid}` | {m['n_clusters']} | {100 * m['coverage_all_entries']:.0f}% | {100 * m['coverage_core']:.0f}% | {100 * m['noise_core']:.0f}% | "
          f"{100 * m['core_entries_in_driven_clusters_share']:.0f}% | {100 * m['core_entries_in_low_proxy_clusters_share']:.0f}% | "
          f"{100 * m['effective_substantive_coverage_core']:.0f}% | {m['cohesion_entry_weighted_mean']} | {seed} | "
          f"{100 * m['coverage_core_english']:.0f}% | {100 * m['coverage_core_french']:.0f}% | {m['mean_npmi']} | {m['runtime_s']} |")
    A("\nColumns: *Driven-cluster share* = share of clustered core entries sitting in clusters flagged channel/template-driven (social markup ≥50%, a repeated "
      "template ≥5 and ≥30%, or one compilation channel ≥70%); *Effective substantive coverage* = coverage × (1 − driven share), a heuristic that rewards covering the core "
      "without relying on flagged clusters; *Cohesion* = entry-weighted mean cosine to cluster centroid; *Seed ARI* = pairwise ARI over UMAP seeds 42/43/44 on core points clustered under both. "
      "Direct baselines (PCA/k-means) have no seed ARI (single seed) and k-means assigns every point, so its noise is 0 by construction.\n")
    A("## 3. Cross-experiment agreement (ARI on core points clustered in both)\n")
    A("| Comparison | ARI | n | share of core |\n|---|---:|---:|---:|")
    for k, v in summary["cross_experiment_ari_on_core"].items():
        A(f"| {k} | {v['ari']} | {v['n']} | {v.get('share_of_core', '')} |")
    ref_dir = out_root / ref_id
    if ref_dir.exists():
        tab = pd.read_csv(ref_dir / "cluster_table.csv", keep_default_na=False)
        ea = pd.read_csv(ref_dir / "entry_assignments.csv", keep_default_na=False)
        noise = ea[ea["cluster_id"].astype(int) == -1]
        A(f"\n## 3b. Cluster-level diagnostics of the run_v1 reference configuration (`{ref_id}`)\n")
        A(f"Noise (unassigned): {len(noise):,} of {len(ea):,} entries ({100 * len(noise) / len(ea):.1f}%); language "
          f"{noise['language'].value_counts().to_dict()}; instrument {noise['source_dataset'].value_counts().to_dict()}.\n")
        A("| Cluster | Entries | Share | EN / FR / other | Top dataset | Exact-dup | URL / # / @ | Template | Short | Top channel | Probability | Cohesion | Flags | Proxy coherence | Descriptive label |")
        A("|---:|---:|---:|---|---|---:|---|---:|---:|---|---:|---:|---|---|---|")
        for r in tab.sort_values("n_entries", ascending=False).itertuples():
            flags = [n for n, f in (("social", r.flag_social_markup), ("template", r.flag_template_repeat),
                                    ("channel", r.flag_channel_concentrated), ("short", r.flag_short_responses),
                                    ("one-instrument", r.flag_single_instrument), ("broad", r.flag_broad_cluster)) if str(f) == "True"]
            ds = json.loads(r.dataset_counts)
            top_ds = max(ds, key=ds.get).replace("_2017_18", "").replace("_2016_18", "")
            kw = ", ".join(str(r.keywords).split(", ")[:3])
            A(f"| {r.cluster_id} | {r.n_entries} | {100 * r.share_of_eligible_entries:.1f}% | {r.english} / {r.french} / {r.other_unknown} | "
              f"{top_ds} {100 * r.top_dataset_share:.0f}% | {100 * r.exact_duplicate_entry_share:.0f}% | "
              f"{100 * r.url_share:.0f}% / {100 * r.hashtag_share:.0f}% / {100 * r.mention_share:.0f}% | {100 * r.top_template_share:.0f}% | "
              f"{100 * r.short_text_share:.0f}% | {r.top_channel or '—'} {'' if not r.top_channel_share else str(round(100 * float(r.top_channel_share))) + '%'} | "
              f"{r.mean_probability if r.mean_probability != '' else '—'} | {r.cohesion_mean_cosine_to_centroid} | {', '.join(flags) or '—'} | "
              f"{r.proxy_coherence} | {kw} |")
        A("\nVerbatim representatives with `corpus_id` for every cluster of every experiment are in each experiment's `cluster_table.csv` "
          "(`representatives` column). Descriptive labels are keyword lists only and are not policy concerns.\n")
    A("\n## 4. Reading the results\n")
    base = ex.get("raw__umap_m25s5")
    strip = ex.get("strip__umap_m25s5")
    if base and strip:
        A(f"- **Stripping URLs/mentions/hashtags** (`umap_m25s5`): driven-cluster share {100 * base['core_entries_in_driven_clusters_share']:.0f}% → "
          f"{100 * strip['core_entries_in_driven_clusters_share']:.0f}%, core noise {100 * base['noise_core']:.0f}% → {100 * strip['noise_core']:.0f}%, "
          f"clusters {base['n_clusters']} → {strip['n_clusters']}, seed ARI {base['seed_ari_mean']} → {strip['seed_ari_mean']}.")
    sd, ns = ex.get("strip_dedup__umap_m25s5"), ex.get("no_short__umap_m25s5")
    if strip and sd and ns:
        A(f"- **Template collapse** (`strip_dedup` vs `strip`, m25s5): clusters {strip['n_clusters']} → {sd['n_clusters']}, driven share "
          f"{100 * strip['core_entries_in_driven_clusters_share']:.0f}% → {100 * sd['core_entries_in_driven_clusters_share']:.0f}%, "
          f"core coverage {100 * strip['coverage_core']:.0f}% → {100 * sd['coverage_core']:.0f}%.")
        A(f"- **Excluding short texts** (`no_short` vs `strip`, m25s5): clusters {strip['n_clusters']} → {ns['n_clusters']}, driven share "
          f"{100 * strip['core_entries_in_driven_clusters_share']:.0f}% → {100 * ns['core_entries_in_driven_clusters_share']:.0f}%, "
          f"core noise {100 * strip['noise_core']:.0f}% → {100 * ns['noise_core']:.0f}% (all-entry coverage {100 * strip['coverage_all_entries']:.0f}% → "
          f"{100 * ns['coverage_all_entries']:.0f}% because short texts are not clustered).")
    pca, km = ex.get("strip__pca20_hdb_m25s5"), ex.get("strip__kmeans_k40")
    if pca and km:
        A(f"- **Direct-embedding baselines:** PCA(20-D)+HDBSCAN leaves {100 * pca['noise_core']:.0f}% of the core unassigned with {pca['n_clusters']} clusters "
          f"(density clustering in the full embedding space is not usable here); k-means (k=40) assigns every text, so its coverage is 100% by "
          f"construction and its clusters include texts a density method treats as noise; its agreement with the UMAP partition is "
          f"ARI {summary['cross_experiment_ari_on_core'].get('strip__umap_m25s5 vs strip__kmeans_k40', {}).get('ari')}.")
    en, fr = ex.get("strip__umap_m25s5_englishonly"), ex.get("strip__umap_m10s5_frenchonly")
    if en and fr:
        A(f"- **Language-separate runs:** English-only {en['n_clusters']} clusters (core coverage of English {100 * en['coverage_core_english']:.0f}%); "
          f"French-only {fr['n_clusters']} clusters (core coverage of French {100 * fr['coverage_core_french']:.0f}%). "
          f"Joint run coverage: English {100 * strip['coverage_core_english']:.0f}%, French {100 * strip['coverage_core_french']:.0f}%." if strip else "")
    A(f"- **Shortlist for human review** (non-direct, non-subset configurations with mean seed ARI ≥ 0.6, ranked by effective substantive coverage): "
      + (", ".join(f"`{p}`" for p in picks) if picks else "none met the stability threshold") + ". This is a review shortlist, not a validated selection.\n")
    A("## 5. Evidence-linked summaries\n")
    A(f"Extractive summaries (frequent terms, verbatim central examples with `corpus_id`, language counts, uncertainty flags) for the shortlisted configurations and the run_v1 reference are in "
      f"`data/full_corpus/experiments_v2/summaries/*.json`; full per-cluster diagnostics are in `<experiment>/cluster_table.csv`. "
      "Each summary is assembled from computed fields only; examples are verbatim source texts.\n")
    for eid in dict.fromkeys(picks[:2]):
        data = json.loads((summaries_dir / f"{eid}.json").read_text(encoding="utf-8"))
        A(f"### Sample from `{eid}` (largest 6 clusters)\n")
        for c in sorted(data["clusters"], key=lambda x: -x["n_entries"])[:6]:
            A(f"- **{c['label']}** — {c['summary']}")
        A("")
    A("## 6. Resources\n")
    A(f"Total runtime {summary['runtime_total_s']} s; peak RSS {summary['peak_rss_mb']} MB; machine: {summary['system']['logical_cpus']} CPUs, "
      f"{summary['system']['ram_total_mb']} MB RAM (~{summary['system']['ram_available_mb']} MB free), {summary['system']['disk_free_gb']} GB disk. "
      "UMAP uses exact chunked kNN (see run_v1 report); no new model weights beyond the cached multilingual encoder were loaded, and only changed texts were re-embedded.\n")
    A("## 7. Limitations\n")
    for x in [
        "No human labels exist; every figure is a proxy and none shows that a cluster is a coherent policy concern.",
        "Flags are heuristic thresholds (social markup ≥50%, template ≥5 repeats and ≥30%, channel ≥70%, short median <40 chars, broad ≥8% of entries); they identify clusters needing review, not invalid ones.",
        "'Channel-concentrated' can also describe a legitimately distinct engagement activity; it is flagged, not discarded.",
        "Stripping removes information (hashtags can carry topic); stripped texts shorter than 20 characters are excluded from clustering in the `strip*` and `no_short` variants.",
        "Template collapse uses a URL/mention/hashtag-stripped exact key; near-duplicates are not collapsed.",
        "Language subsets use source metadata of each unique text's first occurrence; Arabic/Japanese/'S' entries are not in either language run.",
        "Only the multilingual encoder was varied in preprocessing; model sensitivity was evaluated in run_v1.",
        "Seed ARI covers the UMAP seed only; HDBSCAN and k-means are deterministic given the reduced space/seed.",
    ]:
        A(f"- {x}")
    A("\n## 8. Reproducibility\n")
    A("```powershell\npython scripts\\run_clustering_experiments_v2.py run\npython scripts\\run_clustering_experiments_v2.py report\n```")
    A("\nOutputs: `data/full_corpus/experiments_v2/` (one folder per experiment: `entry_assignments.csv`, `cluster_table.csv`, `metrics.json`, `labels_seed*.npy`), "
      "`experiments_summary.json`, `summaries/`, `preprocess/`.")
    text = "\n".join(L) + "\n"
    fc.write_new(report_path, lambda f: f.write(text))
    manifest = {"experiments": {eid: {"clusters": m["n_clusters"], "coverage_core": m["coverage_core"],
                                      "assignments_sha256": fc.sha256_file(out_root / eid / "entry_assignments.csv")}
                                for eid, m in ex.items()},
                "shortlist": picks, "human_validated": False, "report": report_path.name,
                "run_v1_manifest_sha256": fc.sha256_file(RUN_V1 / "run_manifest.json") if (RUN_V1 / "run_manifest.json").exists() else None}
    fc.write_json_new(out_root / "run_manifest.json", manifest)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("stage", choices=["preprocess", "run", "report"])
    a = p.parse_args(argv)
    try:
        if a.stage == "preprocess":
            print(json.dumps(preprocess_stage(), indent=2))
        elif a.stage == "run":
            s = run_experiments()
            for eid, m in s["experiments"].items():
                print(f"{eid}: clusters={m['n_clusters']} cov_core={m['coverage_core']} driven={m['core_entries_in_driven_clusters_share']} "
                      f"seedARI={m['seed_ari_mean']} t={m['runtime_s']}s")
            print("runtime", s["runtime_total_s"], "s; peak RSS", s["peak_rss_mb"], "MB")
        else:
            write_report()
            print(f"written {REPORT_PATH}")
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
