"""
scripts/run_concern_discovery.py
================================
End-to-end, reproducible concern discovery on the existing full-corpus artifacts.

It REUSES (never recomputes or modifies):
    data/full_corpus/run_v1/            eligible corpus (10,108 entries / 10,043 unique texts), embeddings
    data/full_corpus/experiments_v2/    cluster labels for seeds 42/43/44, keywords, stripped-text embeddings
No embedding model is loaded and nothing is re-clustered. All outputs are new and write-once.

Stages
    python scripts/run_concern_discovery.py run              # diagnostics, tiers, summaries, error analysis
    python scripts/run_concern_discovery.py review-package   # blinded two-reviewer package (+ withheld key)
    python scripts/run_concern_discovery.py analyze-reviews  # only when completed reviewer files exist
    python scripts/run_concern_discovery.py report           # reports/concern_discovery_report_v1.md
    python scripts/run_concern_discovery.py all              # run + review-package + report

Scope: Canadian federal Open Government consultations (2016-18). Nothing here validates performance on
any other consultation (for example Indian MCA consultations). Clusters are not ground-truth labels;
sentiment and stance are not inferred.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.concern_discovery import diagnostics as dg  # noqa: E402
from models.concern_discovery import duplicates as dup  # noqa: E402
from models.concern_discovery import review as rv  # noqa: E402
from models.concern_discovery import summaries as sm  # noqa: E402
from models.concern_discovery.schema import (CheckLog, CorpusIntegrityError, sha256_file, sha256_text,  # noqa: E402
                                             validate_corpus, validate_labels)

RUN_V1 = ROOT / "data" / "full_corpus" / "run_v1"
EXP_V2 = ROOT / "data" / "full_corpus" / "experiments_v2"
OUT = ROOT / "data" / "full_corpus" / "concern_discovery_v1"
REPORT = ROOT / "reports" / "concern_discovery_report_v1.md"
SEEDS = (42, 43, 44)
CONTENT_MIN_CHARS = 20          # experiments v2: stripped texts shorter than this were not clustered
CORE_MIN_CHARS = 40             # experiments v2 comparable core
REVIEW_SEED = 20261013
CANDIDATE_CONFIGS = ["raw__umap_m25s5", "raw__umap_m50s10", "strip__umap_m25s5", "strip__umap_m50s10"]
SELECTION_RULE = ("Primary configuration = among CANDIDATE_CONFIGS (the four UMAP+HDBSCAN configurations of experiments v2 "
                  "that have three-seed runs) those with minimum pairwise seed ARI >= 0.85; choose the highest share of "
                  "comparable-core entries in candidate-tier clusters (stable groups with no diagnostic warning); ties "
                  "broken by mean seed ARI. It selects what is summarised and reviewed; it does not validate it.")
SELECTION_RULE_REVISION = ("Revised once, after a development run and before the reported run: the first rule maximised 'usable "
                           "coverage' (candidate + needs_review + campaign_or_template). It rewarded configurations whose largest "
                           "clusters are broad catch-all groups (one cluster of 12.8% of entries keyed on 'canada, canadians'), which "
                           "are not inspectable concern groups. Outcomes under both rules are reported side by side.")
SELECTION_KEYS = {"candidate_coverage_core (current rule)": "candidate_coverage_core",
                  "usable_concern_coverage_core (first rule)": "usable_concern_coverage_core",
                  "usable_non_broad_coverage_core (sensitivity)": "usable_non_broad_coverage_core"}
MIN_SEED_ARI = 0.85
INTERPRETATION = [
    "Corpus: Canadian federal Open Government consultations 2016-18 (English/French). Results do not transfer to, "
    "or validate performance on, Indian MCA or any other consultation.",
    "Clusters are unsupervised groupings of semantically similar texts, not ground-truth concern labels.",
    "Every metric in this run is an internal proxy (geometry, seed agreement, metadata association); none is "
    "evidence of semantic or policy validity. Only completed independent human reviews are such evidence.",
    "Descriptions are deterministic and extractive: computed term shares and verbatim quotes with source ids. "
    "They are not validated because they read plausibly.",
    "Sentiment (tone) and stance (position) are separate constructs and are not inferred here.",
]


# ── io helpers ────────────────────────────────────────────────────────────────

def write_new(path: Path, writer, binary: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb" if binary else "x", **({} if binary else {"encoding": "utf-8", "newline": ""})) as f:
        writer(f)


def write_json(path: Path, obj: Any) -> None:
    write_new(path, lambda f: (json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default), f.write("\n")))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    write_new(path, lambda f: df.to_csv(f, index=False, lineterminator="\n"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── loading ───────────────────────────────────────────────────────────────────

def load_inputs(run_v1: Path = RUN_V1, exp_v2: Path = EXP_V2, log: CheckLog | None = None) -> dict[str, Any]:
    log = log or CheckLog()
    for p in (run_v1 / "corpus" / "corpus_entries.csv", run_v1 / "corpus" / "unique_texts.csv",
              run_v1 / "corpus" / "corpus_manifest.json", run_v1 / "embeddings" / "multilingual" / "embeddings.npy",
              exp_v2 / "preprocess" / "strip_embeddings.npz"):
        if not p.is_file():
            log.fail("input files present", f"missing {p}")
    log.ok("input files present", "run_v1 corpus/embeddings and experiments_v2 preprocessing cache found")
    entries = pd.read_csv(run_v1 / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    unique = pd.read_csv(run_v1 / "corpus" / "unique_texts.csv", dtype=str, keep_default_na=False)
    manifest = json.loads((run_v1 / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run_v1 / "corpus" / "corpus_entries.csv") != manifest.get("corpus_entries_sha256"):
        log.fail("corpus_entries.csv hash", "differs from run_v1 corpus manifest")
    if sha256_file(run_v1 / "corpus" / "unique_texts.csv") != manifest.get("unique_texts_sha256"):
        log.fail("unique_texts.csv hash", "differs from run_v1 corpus manifest")
    log.ok("corpus file hashes", "corpus_entries.csv and unique_texts.csv match the run_v1 manifest")
    emb = np.load(run_v1 / "embeddings" / "multilingual" / "embeddings.npy")
    validate_corpus(entries, unique, emb, manifest, log)
    entries["unique_id"] = entries["unique_id"].astype(int)
    entries["corpus_id"] = entries["corpus_id"].astype(int)
    unique["multiplicity"] = unique["multiplicity"].astype(int)
    unique["corpus_id"] = unique["corpus_id"].astype(int)
    raw = unique["original_text"].tolist()
    stripped = [dup.strip_social(t) for t in raw]
    changed = np.array([i for i, (a, b) in enumerate(zip(raw, stripped)) if a != b and len(b) >= CONTENT_MIN_CHARS], dtype=int)
    z = np.load(exp_v2 / "preprocess" / "strip_embeddings.npz")
    if not np.array_equal(z["idx"], changed):
        log.fail("stripped-embedding cache alignment", "cached indices differ from the recomputed stripped-text set")
    emb_strip = emb.copy()
    emb_strip[changed] = z["emb"]
    if not np.isfinite(emb_strip).all() or np.abs(np.linalg.norm(emb_strip, axis=1) - 1).max() > 1e-3:
        log.fail("stripped embeddings", "non-finite or non-normalised vectors")
    log.ok("stripped embeddings", f"{len(changed):,} texts re-embedded in experiments v2 overlaid on run_v1 vectors")
    content_free = np.array([len(s) < CONTENT_MIN_CHARS for s in stripped])
    core = np.array([len(s) >= CORE_MIN_CHARS for s in stripped])
    return {"entries": entries, "unique": unique, "emb": {"raw": emb, "strip": emb_strip},
            "texts": {"raw": raw, "strip": stripped}, "content_free": content_free, "core": core,
            "manifest": manifest, "log": log}


def load_config_labels(config_id: str, n_unique: int, log: CheckLog, exp_v2: Path = EXP_V2) -> dict[str, Any]:
    d = exp_v2 / config_id
    labels = {}
    for s in SEEDS:
        p = d / f"labels_seed{s}.npy"
        if p.is_file():
            labels[s] = np.load(p)
            validate_labels(labels[s], n_unique, f"{config_id} seed {s}", log)
    if 42 not in labels:
        log.fail(f"labels {config_id}", "seed 42 labels missing")
    table = pd.read_csv(d / "cluster_table.csv", keep_default_na=False)
    kw = {int(r.cluster_id): [k.strip() for k in str(r.keywords).split(",") if k.strip()] for r in table.itertuples()}
    if set(kw) != set(int(c) for c in np.unique(labels[42][labels[42] >= 0])):
        log.fail(f"keywords {config_id}", "cluster_table.csv clusters differ from labels_seed42.npy")
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return {"labels": labels, "keywords": kw, "metrics": metrics,
            "input_sha256": {f"{config_id}/labels_seed{s}.npy": sha256_file(d / f"labels_seed{s}.npy") for s in labels}}


# ── run stage ─────────────────────────────────────────────────────────────────

def summarise_config(config_id: str, data: dict[str, Any], groups: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    variant = config_id.split("__")[0]
    emb = data["emb"]["raw" if variant == "raw" else "strip"]
    texts = data["texts"]["raw" if variant == "raw" else "strip"]
    labels = cfg["labels"][42]
    alts = [cfg["labels"][s] for s in SEEDS if s != 42 and s in cfg["labels"]]
    table, points = dg.assess_clusters(config_id, labels, data["entries"], data["unique"], emb, groups,
                                       cfg["keywords"], alts, texts, markup_in_input=(variant == "raw"))
    part = dg.partition_summary(table, labels, data["entries"], data["core"])
    e_lab = labels[data["entries"]["unique_id"].to_numpy()]
    rep = data["entries"]["unique_id"].map(groups.set_index("unique_id")["repeated_template_group"]).to_numpy()
    part["confounds"] = dg.confound_report(e_lab, data["entries"], rep)
    part["simplified_silhouette_cosine"] = dg.simplified_silhouette(points)
    part["mean_cluster_stability_jaccard"] = round(float(table["stability_jaccard_mean"].mean()), 4) if len(table) else None
    part["seed_ari_mean"] = cfg["metrics"].get("seed_ari_mean")
    part["seed_ari_min"] = cfg["metrics"].get("seed_ari_min")
    part["entry_weighted_cohesion"] = round(float(np.average(table["cohesion_mean"], weights=table["n_entries"])), 4)
    return {"table": table, "points": points, "partition": part, "labels": labels, "emb": emb, "texts": texts}


def build_summaries(config_id: str, res: dict[str, Any], data: dict[str, Any], groups: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    unique = data["unique"]
    labels, emb, texts = res["labels"], res["emb"], res["texts"]
    dg_arr = groups["dup_group_id"].to_numpy()
    langs = unique["language"].tolist()
    text_by_cid = dict(zip(data["entries"]["corpus_id"], data["entries"]["original_text"]))
    out, problems, rows = [], [], []
    population = [texts[i] for i in np.where(labels >= -1)[0]]          # texts the clustering saw
    base = sm.CorpusSupport(population)
    for row in res["table"].to_dict("records"):
        c = row["cluster_id"]
        members = np.where(labels == c)[0]
        reps = sm.select_representatives(members, emb, dg_arr, unique["original_text"].tolist(), langs,
                                         deprioritise=data.get("deprioritise"))
        mtexts = [texts[i] for i in members]
        s = sm.describe_cluster(row, mtexts, reps, unique, base)
        member_cids = set(data["entries"].loc[data["entries"]["unique_id"].isin(members), "corpus_id"].astype(int))
        problems += sm.verify_grounding(s, member_cids, text_by_cid, mtexts, corpus_texts=population)
        out.append(s)
        for ev in s["evidence"]:
            rows.append({"config_id": config_id, "cluster_id": c, "tier": s["tier"], "rank": ev["rank"],
                         "corpus_id": ev["corpus_id"], "unique_id": ev["unique_id"], "multiplicity": ev["multiplicity"],
                         "source_dataset": ev["source_dataset"], "source_record_id": ev["source_record_id"],
                         "question_slug": ev["question_slug"], "language": ev["language"],
                         "cosine_to_centroid": ev["cosine_to_centroid"],
                         "max_cosine_to_other_examples": ev["max_cosine_to_other_examples"],
                         "selection_reason": ev["selection_reason"], "quote": ev["quote"],
                         "full_text": ev["full_text"], "text_sha256": sha256_text(ev["full_text"])})
    if problems:
        raise sm.GroundingError("Ungrounded summary content: " + "; ".join(problems[:10]))
    return out, pd.DataFrame(rows)


def model_comparison(data: dict[str, Any], run_v1: Path = RUN_V1) -> dict[str, Any]:
    """Language/instrument association of the run_v1 partitions for the English and multilingual encoders."""
    out: dict[str, Any] = {}
    entries = data["entries"]
    for model in ("baseline", "multilingual"):
        for cfg in ("u15_m25s5", "u15_m50s10"):
            p = run_v1 / "clustering" / model / cfg / "entry_assignments.csv"
            if not p.is_file():
                continue
            a = pd.read_csv(p, usecols=["corpus_id", "cluster_id"])
            lab = entries["corpus_id"].map(dict(zip(a["corpus_id"], a["cluster_id"]))).to_numpy()
            if np.isnan(lab.astype(float)).any():
                raise CorpusIntegrityError(f"{p} does not cover every corpus entry")
            lab = lab.astype(int)
            m = lab >= 0
            sizes = pd.Series(lab[m]).value_counts()
            fr = pd.Series(lab[m & (entries["language"] == "French").to_numpy()]).value_counts().reindex(sizes.index, fill_value=0)
            out[f"{model}/{cfg}"] = {
                "model": {"baseline": "sentence-transformers/all-MiniLM-L6-v2 (English)",
                          "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}[model],
                "clusters": int(len(sizes)), "noise_share": round(float((lab == -1).mean()), 4),
                "clusters_french_majority": int((fr / sizes >= 0.5).sum()),
                "clusters_ge90_french": int((fr / sizes >= 0.9).sum()),
                "language_association": dg.association(lab[m], entries["language"].map(dg.lang_group).to_numpy()[m], 100),
                "instrument_association": dg.association(lab[m], entries["source_dataset"].to_numpy()[m], 100),
            }
    return out


def error_analysis(res: dict[str, Any], summaries: list[dict], data: dict[str, Any], groups: pd.DataFrame,
                   n: int = 5) -> dict[str, Any]:
    unique, table, points = data["unique"], res["table"], res["points"]
    cid = unique["corpus_id"].to_numpy()

    def ex(u: int, **extra: Any) -> dict[str, Any]:
        r = unique.iloc[int(u)]
        return {"corpus_id": int(r["corpus_id"]), "source_dataset": r["source_dataset"], "source_record_id": r["source_record_id"],
                "question_slug": r["question_slug"], "language": r["language"], "text": r["original_text"], **extra}

    label_of = {s["cluster_id"]: (s["short_label"] or "(no label)") for s in summaries}
    out: dict[str, Any] = {}
    by_tier = {}
    for tier in dg.TIER_ORDER:
        rows = [s for s in summaries if s["tier"] == tier]
        rows = sorted(rows, key=lambda s: -s["n_entries"])[:2]
        by_tier[tier] = [{"cluster_id": s["cluster_id"], "n_entries": s["n_entries"], "flags": s["flags"],
                          "short_label": s["short_label"], "description": s["description"] or s["not_summarised_reason"],
                          "examples": [{"corpus_id": e["corpus_id"], "text": e["full_text"]} for e in s["evidence"][:3]]}
                         for s in rows]
    out["tier_examples"] = by_tier
    clustered = points[points["cluster_id"] >= 0].sort_values("margin")
    out["members_closer_to_another_cluster"] = [
        ex(r.unique_id, assigned_cluster=int(r.cluster_id), assigned_label=label_of.get(int(r.cluster_id)),
           nearer_cluster=int(r.nearest_other_cluster), nearer_label=label_of.get(int(r.nearest_other_cluster)),
           margin=round(float(r.margin), 4))
        for r in clustered.head(n).itertuples()]
    noise = points[points["cluster_id"] == -1].sort_values("cos_nearest_other", ascending=False)
    out["noise_close_to_a_cluster"] = [ex(r.unique_id, nearest_cluster=int(r.nearest_other_cluster),
                                          nearest_label=label_of.get(int(r.nearest_other_cluster)),
                                          cosine_to_centroid=round(float(r.cos_nearest_other), 4))
                                       for r in noise.head(n).itertuples()]
    out["noise_far_from_all_clusters"] = [ex(r.unique_id, nearest_cluster=int(r.nearest_other_cluster),
                                             cosine_to_centroid=round(float(r.cos_nearest_other), 4))
                                          for r in noise.tail(n).itertuples()]
    out["noise_texts_by_length"] = {
        "noise_unique_texts": int(len(noise)),
        "median_chars_noise": int(np.median([len(unique.iloc[int(u)]["original_text"]) for u in noise["unique_id"]])) if len(noise) else None,
        "median_chars_clustered": int(np.median([len(unique.iloc[int(u)]["original_text"]) for u in clustered["unique_id"]])) if len(clustered) else None,
    }
    least = table.sort_values("stability_jaccard_mean").head(3)
    out["least_stable_clusters"] = [{"cluster_id": int(r.cluster_id), "stability_jaccard_mean": r.stability_jaccard_mean,
                                     "n_entries": int(r.n_entries), "label": label_of.get(int(r.cluster_id)), "tier": r.tier}
                                    for r in least.itertuples()]
    g = groups[groups["repeated_template_group"]].drop_duplicates("dup_group_id").sort_values("group_entries", ascending=False)
    lab = res["labels"]
    camp = []
    for r in g.head(8).itertuples():
        mem = groups.loc[groups["dup_group_id"] == r.dup_group_id, "unique_id"].to_numpy()
        dist = pd.Series(lab[mem]).value_counts().to_dict()
        camp.append({**ex(r.dup_group_id), "group_entries": int(r.group_entries), "group_unique_texts": int(r.group_unique_texts),
                     "assigned_states": {str(k): int(v) for k, v in dist.items()}})
    out["repeated_template_groups"] = camp
    # cross-lingual: in usable clusters with French members, the most central English and French member
    xl = []
    for s in summaries:
        if s["tier"] in ("candidate", "needs_review") and s["language_counts"]["French"] >= 3 and s["language_counts"]["English"] >= 3:
            mem = np.where(lab == s["cluster_id"])[0]
            cen = res["emb"][mem].mean(axis=0)
            sims = res["emb"][mem] @ (cen / np.linalg.norm(cen))
            pair = {}
            for lg in ("English", "French"):
                ok = [i for i in np.argsort(-sims) if unique.iloc[int(mem[i])]["language"] == lg]
                if ok:
                    pair[lg] = ex(mem[ok[0]])
            xl.append({"cluster_id": s["cluster_id"], "label": s["short_label"], "french_share": round(s["language_counts"]["French"] / s["n_entries"], 3),
                       "most_central_by_language": pair})
    out["cross_lingual_groupings"] = sorted(xl, key=lambda x: -x["french_share"])[:4]
    out["note"] = "Examples are selected by rule (most negative margins, nearest/farthest noise, least stable clusters); they are illustrations for human inspection, not error rates."
    _ = cid
    return out


def stage_run(out: Path = OUT, run_v1: Path = RUN_V1, exp_v2: Path = EXP_V2) -> dict[str, Any]:
    if (out / "run_manifest.json").exists():
        raise FileExistsError(f"Run already exists: {out}")
    t0 = time.perf_counter()
    log = CheckLog()
    data = load_inputs(run_v1, exp_v2, log)
    unique = data["unique"]
    groups, edges = dup.near_duplicate_groups(data["texts"]["raw"], data["emb"]["raw"], unique["multiplicity"].to_numpy())
    groups.insert(1, "corpus_id", unique["corpus_id"].to_numpy())
    # representatives: repeated-template and markup-only texts are used only when nothing else is available
    data["deprioritise"] = (groups["repeated_template_group"].to_numpy()
                            | np.array([len(s) < CONTENT_MIN_CHARS for s in data["texts"]["strip"]]))
    results: dict[str, Any] = {}
    cfg_inputs: dict[str, str] = {}
    for config_id in CANDIDATE_CONFIGS:
        cfg = load_config_labels(config_id, len(unique), log, exp_v2)
        cfg_inputs.update(cfg["input_sha256"])
        results[config_id] = summarise_config(config_id, data, groups, cfg)
    elig = {k: v for k, v in results.items() if (v["partition"]["seed_ari_min"] or 0) >= MIN_SEED_ARI}
    if not elig:
        raise RuntimeError("No configuration meets the seed-stability requirement")
    rule_outcomes = {name: max(elig, key=lambda k, key=key: (elig[k]["partition"][key], elig[k]["partition"]["seed_ari_mean"]))
                     for name, key in SELECTION_KEYS.items()}
    selected = rule_outcomes["candidate_coverage_core (current rule)"]
    all_summaries: dict[str, list[dict]] = {}
    for config_id, res in results.items():
        summaries, reps = build_summaries(config_id, res, data, groups)
        all_summaries[config_id] = summaries
        d = out / "configurations" / config_id
        t = res["table"].copy()
        for col in ("keywords", "keyword_support", "flags", "dataset_counts"):
            t[col] = t[col].map(lambda x: json.dumps(x, ensure_ascii=False))
        write_csv(d / "cluster_diagnostics.csv", t)
        write_json(d / "concern_summaries.json", {"config_id": config_id, "human_validated": False,
                                                  "summary_method": "deterministic_extractive_v1", "clusters": summaries})
        write_csv(d / "representative_examples.csv", reps)
        write_json(d / "partition_metrics.json", res["partition"])
    res = results[selected]
    # final machine-readable outputs for the selected configuration
    ent = data["entries"]
    lab = res["labels"][ent["unique_id"].to_numpy()]
    tier_of = dict(zip(res["table"]["cluster_id"], res["table"]["tier"]))
    label_of = {s["cluster_id"]: s["short_label"] for s in all_summaries[selected]}
    pts = res["points"].set_index("unique_id")
    gi = groups.set_index("unique_id")
    ea = ent[["corpus_id", "unique_id", "source_dataset", "source_record_id", "source_row_number", "source_field",
              "question_slug", "language"]].copy()
    ea["cluster_id"] = lab
    ea["state"] = np.where(lab >= 0, "clustered", np.where(lab == -1, "noise", "excluded_content_free_after_stripping"))
    ea["tier"] = [tier_of.get(int(x), "") for x in lab]
    ea["short_label"] = [label_of.get(int(x)) or "" for x in lab]
    ea["margin_to_nearest_other_cluster"] = ea["unique_id"].map(pts["margin"]).round(4)
    ea["nearest_cluster_if_noise"] = np.where(lab == -1, ea["unique_id"].map(pts["nearest_other_cluster"]).fillna(-1).astype(int), -1)
    ea["dup_group_id"] = ea["unique_id"].map(gi["dup_group_id"])
    ea["repeated_template_group"] = ea["unique_id"].map(gi["repeated_template_group"])
    groups_table = res["table"][["cluster_id", "tier", "n_entries", "n_unique_texts", "distinct_near_dup_groups",
                                 "share_of_eligible_entries", "english", "french", "other_unknown", "top_dataset_share",
                                 "top_field", "top_field_share", "cohesion_mean", "margin_mean", "negative_margin_share",
                                 "centroid_cos_nearest", "stability_jaccard_mean", "stability_jaccard_min",
                                 "keyword_support_top3_mean", "top_dup_group_share", "no_opinion_share", "promotion_share",
                                 "social_markup_share"]].copy()
    groups_table["flags"] = res["table"]["flags"].map(lambda x: ";".join(x))
    sdict = {s["cluster_id"]: s for s in all_summaries[selected]}
    groups_table["short_label"] = groups_table["cluster_id"].map(lambda c: sdict[c]["short_label"] or "")
    groups_table["description"] = groups_table["cluster_id"].map(lambda c: sdict[c]["description"] or sdict[c]["not_summarised_reason"])
    groups_table["evidence_corpus_ids"] = groups_table["cluster_id"].map(lambda c: ";".join(str(e["corpus_id"]) for e in sdict[c]["evidence"]))
    sel_dir = out / "selected"
    write_csv(sel_dir / "entry_assignments.csv", ea)
    write_csv(sel_dir / "concern_groups.csv", groups_table.sort_values("n_entries", ascending=False))
    errors = error_analysis(res, all_summaries[selected], data, groups)
    write_json(sel_dir / "error_analysis.json", errors)
    write_csv(sel_dir / "point_geometry.csv", res["points"])
    write_csv(out / "duplicates" / "near_duplicate_groups.csv", groups)
    write_csv(out / "duplicates" / "near_duplicate_edges.csv", edges)
    mc = model_comparison(data, run_v1)
    write_json(out / "model_comparison.json", mc)
    comparison = {k: v["partition"] for k, v in results.items()}
    write_json(out / "configuration_comparison.json", {"selection_rule": SELECTION_RULE, "selection_rule_revision": SELECTION_RULE_REVISION,
                                                       "min_seed_ari": MIN_SEED_ARI, "eligible": sorted(elig), "selected": selected,
                                                       "outcome_by_rule": rule_outcomes, "configurations": comparison})
    config = {"thresholds": dg.THRESHOLDS, "near_duplicates": dup.DEFAULTS, "tiers": dg.TIER_DESCRIPTIONS,
              "candidate_configs": CANDIDATE_CONFIGS, "seeds": SEEDS, "content_min_chars": CONTENT_MIN_CHARS,
              "core_min_chars": CORE_MIN_CHARS, "summary": {"method": "deterministic_extractive_v1",
                                                             "min_term_support": sm.MIN_TERM_SUPPORT,
                                                             "max_quote_chars": sm.MAX_QUOTE_CHARS,
                                                             "representatives": "MMR lambda=0.7, k=5, one per near-dup group, "
                                                                                "texts >= 40 chars preferred, language coverage >= 15%"},
              "summariser_availability": "No local abstractive summariser with weights is available (facebook/bart-large-cnn: "
                                         "config.json only in the HF cache). Extractive fallback used by design.",
              "interpretation": INTERPRETATION}
    write_json(out / "config.json", config)
    write_json(out / "input_checks.json", log.checks)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selected_configuration": selected,
        "human_validated": False, "grounding_verification": "PASSED (every quote verbatim, every citation a member, "
                                                            "every term share recomputed)",
        "inputs_sha256": {"run_v1/corpus/corpus_entries.csv": data["manifest"]["corpus_entries_sha256"],
                          "run_v1/embeddings/multilingual/embeddings.npy": sha256_file(run_v1 / "embeddings" / "multilingual" / "embeddings.npy"),
                          "experiments_v2/preprocess/strip_embeddings.npz": sha256_file(exp_v2 / "preprocess" / "strip_embeddings.npz"),
                          **{f"experiments_v2/{k}": v for k, v in cfg_inputs.items()}},
        "outputs_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in files},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    write_json(out / "run_manifest.json", manifest)
    return {"selected": selected, "comparison": comparison, "runtime_s": manifest["runtime_s"],
            "near_duplicates": dup.summarise_groups(groups)}


# ── review stages ─────────────────────────────────────────────────────────────

def _load_selected(out: Path) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    sel = man["selected_configuration"]
    t = pd.read_csv(out / "configurations" / sel / "cluster_diagnostics.csv", keep_default_na=False)
    t["flags"] = t["flags"].map(json.loads)
    summ = json.loads((out / "configurations" / sel / "concern_summaries.json").read_text(encoding="utf-8"))
    return sel, t, {s["cluster_id"]: s for s in summ["clusters"]}


def stage_review_package(out: Path = OUT, run_v1: Path = RUN_V1, exp_v2: Path = EXP_V2) -> dict[str, Any]:
    sel, table, summaries = _load_selected(out)
    data = load_inputs(run_v1, exp_v2)
    variant = sel.split("__")[0]
    labels = np.load(exp_v2 / sel / "labels_seed42.npy")
    groups = pd.read_csv(out / "duplicates" / "near_duplicate_groups.csv")["dup_group_id"].to_numpy()
    a, ak, b, bk = rv.build_review_items(table, labels, data["unique"], data["emb"]["raw" if variant == "raw" else "strip"],
                                         groups, summaries, REVIEW_SEED)
    return rv.write_package(out / "human_review", a, ak, b, bk, REVIEW_SEED,
                            {"configuration": sel, "scope": "Canadian federal Open Government consultations 2016-18"})


def stage_analyze_reviews(out: Path = OUT) -> dict[str, Any]:
    sel, table, _ = _load_selected(out)
    return rv.analyze_reviews(out / "human_review", out / "human_review_analysis", table)


# ── report ────────────────────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def stage_report(out: Path = OUT, report_path: Path = REPORT) -> str:
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    import run_research_evaluation as rre
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    comp = json.loads((out / "configuration_comparison.json").read_text(encoding="utf-8"))
    mc = json.loads((out / "model_comparison.json").read_text(encoding="utf-8"))
    checks = json.loads((out / "input_checks.json").read_text(encoding="utf-8"))
    sel = comp["selected"]
    part = comp["configurations"][sel]
    summ = json.loads((out / "configurations" / sel / "concern_summaries.json").read_text(encoding="utf-8"))["clusters"]
    groups_tab = pd.read_csv(out / "selected" / "concern_groups.csv", keep_default_na=False)
    errors = json.loads((out / "selected" / "error_analysis.json").read_text(encoding="utf-8"))
    dupg = pd.read_csv(out / "duplicates" / "near_duplicate_groups.csv", keep_default_na=False)
    dsum = dup.summarise_groups(dupg.assign(in_near_dup_group=dupg["in_near_dup_group"].astype(str) == "True",
                                            is_exact_repeat=dupg["is_exact_repeat"].astype(str) == "True",
                                            repeated_template_group=dupg["repeated_template_group"].astype(str) == "True"))
    pkg = out / "human_review" / "package_manifest.json"
    pkg_m = json.loads(pkg.read_text(encoding="utf-8")) if pkg.exists() else None
    metrics_path = out / "human_review_analysis" / "review_metrics.json"
    if metrics_path.exists():
        review_res = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:                                    # read-only status; the report never opens the answer key
        review_res = rv.review_status(out / "human_review")
        if review_res["status"] == "READY":
            review_res = {"status": "READY", "problems": ["completed files present; run the analyze-reviews stage"]}
    ann = rre.check_annotation_readiness()

    L: list[str] = []
    A = L.append
    A("# PolicyLens concern discovery report (v1)\n")
    validated = bool(review_res and review_res.get("status") == "COMPLETE")
    A(f"**Status:** pipeline implemented and run; outputs are **{'partly human-evaluated (see section 8)' if validated else 'NOT human-validated'}**. "
      f"Generated {man['created_utc']} from `data/full_corpus/concern_discovery_v1/`.\n")
    A("## 1. Scope and interpretation rules\n")
    for x in INTERPRETATION:
        A(f"- {x}")
    A("\n## 2. Inputs and integrity\n")
    A("The pipeline consumes the existing eligible corpus and cached artifacts; it loads no encoder and re-clusters nothing.\n")
    A("| Check | Status | Detail |\n|---|---|---|")
    for c in checks:
        A(f"| {c['check']} | {c['status']} | {c['detail']} |")
    A(f"\nCorpus definition unchanged: {part['entries']:,} eligible entries from the run_v1 manifest "
      "(the eligibility rule is documented in `reports/full_corpus_clustering_report_v1.md`). Source ids, original text, language, "
      "source dataset and consultation field are carried through to every output row.\n")
    A("## 3. Methods\n")
    A("1. **Duplicates and templates.** Unique texts are linked when their template key matches (URLs, mentions, hashtags, "
      "punctuation and case removed) or when they are among each other's 10 nearest neighbours with cosine ≥ 0.92 *and* "
      "character 5-gram Jaccard ≥ 0.5 (or containment ≥ 0.9). The lexical condition keeps paraphrases and translations out "
      "of duplicate groups. Groups with ≥ 5 entries are *repeated template groups*. Identical text never implies the same author.")
    A("2. **Candidate configurations.** The four UMAP+HDBSCAN configurations of experiments v2 with three-seed runs "
      "(`raw`/`strip` × `m25s5`/`m50s10`) are re-assessed from their saved labels. Noise is kept as noise: no comment is forced into a cluster.")
    A("3. **Per-cluster diagnostics.** Cohesion (cosine to centroid), separation (margin to the nearest other centroid, share of members "
      "closer to another centroid, centroid-to-centroid cosine), seed stability (best-match Jaccard of member sets across UMAP seeds "
      "42/43/44), distinct near-duplicate groups, template share, language/instrument/field/channel composition, keyword support, "
      "and transparent EN/FR lexicons for no-opinion and promotional responses.")
    A("4. **Tiers.** " + "; ".join(f"`{k}`: {v}" for k, v in dg.TIER_DESCRIPTIONS.items()) + ". Precedence: unreliable > "
      "campaign_or_template > non_concern_pattern > needs_review > candidate. Thresholds are in `config.json`.")
    A(f"5. **Selection.** {SELECTION_RULE} {SELECTION_RULE_REVISION}")
    A("6. **Confounds.** Bias-corrected Cramér's V and NMI between cluster labels and language, instrument, question field, "
      "compilation channel and repeated-template membership, each against a 200-permutation null.")
    A("7. **Descriptions.** Deterministic and extractive. No local abstractive summariser with weights is available "
      "(the cache holds only `facebook/bart-large-cnn/config.json`), so the fallback is used by design. Each description states "
      "the share of member texts containing up to four c-TF-IDF terms (only terms in ≥ 10% of members) and quotes the first "
      "maximal-marginal-relevance representative verbatim with its `corpus_id`. Representatives: MMR (λ = 0.7), one per "
      "near-duplicate group, ≥ 40 characters preferred, a language with ≥ 15% of members guaranteed one example. "
      "`verify_grounding` re-checks every quote, citation and percentage; the run aborts on any violation. "
      f"Result: **{man['grounding_verification']}**.")
    A("8. **Human review.** A blinded two-reviewer package (cluster intrusion + coherence; then summary support) with a withheld key. See section 8.\n")

    A("## 4. Duplicates, near-duplicates and templates\n")
    A("| Measure | Value |\n|---|---:|")
    for k, v in dsum.items():
        A(f"| {k.replace('_', ' ')} | {v:,} |")
    A("\nLargest repeated template groups and where the selected configuration put them (cluster id: unique texts; -1 noise, -2 excluded):\n")
    for g in errors["repeated_template_groups"][:6]:
        A(f"- {g['group_entries']} entries / {g['group_unique_texts']} texts, {g['source_dataset']}: “{sm.quotable_span(g['text'], 110)}” → {g['assigned_states']}")
    A("\n## 5. Configuration comparison (internal proxies only)\n")
    A("| Configuration | Clusters | Noise (pop.) | Seed ARI mean / min | Mean cluster stability | Cohesion | Simplified silhouette | "
      "Candidate | Needs review | Campaign | Non-concern | Unreliable | Candidate coverage (core) | Usable coverage (core) | Language V (null p95) | Instrument V | Field V |")
    A("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for k, p in comp["configurations"].items():
        tc = p["tier_clusters"]
        cf = p["confounds"]
        A(f"| `{k}`{' **(selected)**' if k == sel else ''} | {p['clusters']} | {_pct(p['noise_share_of_population'])} | "
          f"{p['seed_ari_mean']} / {p['seed_ari_min']} | {p['mean_cluster_stability_jaccard']} | {p['entry_weighted_cohesion']} | "
          f"{p['simplified_silhouette_cosine']} | {tc.get('candidate', 0)} | {tc.get('needs_review', 0)} | {tc.get('campaign_or_template', 0)} | "
          f"{tc.get('non_concern_pattern', 0)} | {tc.get('unreliable', 0)} | {_pct(p['candidate_coverage_core'])} | {_pct(p['usable_concern_coverage_core'])} | "
          f"{cf['language']['cramers_v']} ({cf['language']['null_v_p95']}) | {cf['source_dataset']['cramers_v']} | {cf['question_field']['cramers_v']} |")
    A(f"\nEligible under the stability requirement (min seed ARI ≥ {MIN_SEED_ARI}): {', '.join(f'`{e}`' for e in comp['eligible'])}. "
      f"Selected: **`{sel}`**. Usable coverage counts entries in candidate, needs_review and campaign_or_template clusters. "
      "These numbers rank configurations by internal criteria only.\n")
    A("Outcome under each selection criterion (the revision is described in section 3):\n")
    A("| Criterion | Configuration chosen |\n|---|---|")
    for k, v in comp.get("outcome_by_rule", {}).items():
        A(f"| {k} | `{v}` |")
    A("")
    cf = part["confounds"]
    A("### Confound analysis of the selected configuration\n")
    A("| Attribute | n | Cramér's V | Null mean / p95 | Permutation p | NMI |\n|---|---:|---:|---|---:|---:|")
    for k, v in cf.items():
        if k == "cross_lingual_mixing":
            continue
        A(f"| {k.replace('_', ' ')} | {v['n']:,} | {v['cramers_v']} | {v.get('null_v_mean')} / {v.get('null_v_p95')} | {v.get('perm_p')} | {v['nmi']} |")
    x = cf["cross_lingual_mixing"]
    A(f"\nCross-lingual mixing: French is {_pct(x['french_share_of_clustered_entries'])} of clustered entries. Of the "
      f"{x['clusters_expected_ge5_french']} clusters where ≥ 5 French entries would be expected at that base rate, "
      f"{x['of_which_contain_french']} contain French entries and {x['of_which_french_below_quarter_of_expected']} hold less than a quarter "
      f"of the expected French count; {x['clusters_french_majority']} clusters are French-majority. Associations with the permutation p at "
      "its floor are statistically distinguishable from chance; a non-zero V with instrument or field is expected because questions differ in "
      "topic, so V measures how strongly the partition follows the instrument, not whether that is wrong.\n")
    A("### Embedding-model comparison (run_v1 partitions)\n")
    A("| Partition | Model | Clusters | Noise | French-majority clusters | ≥ 90% French | Language V | Instrument V |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for k, v in mc.items():
        A(f"| `{k}` | {v['model']} | {v['clusters']} | {_pct(v['noise_share'])} | {v['clusters_french_majority']} | {v['clusters_ge90_french']} | "
          f"{v['language_association']['cramers_v']} | {v['instrument_association']['cramers_v']} |")
    A("\nThe English-only encoder separates French comments by language more than the multilingual encoder does; this is the reason the "
      "multilingual encoder is used. Agreement between the two models' partitions (ARI 0.27–0.51, run_v1 report) is low, so the grouping is "
      "model-dependent.\n")

    A(f"## 6. Concern groups of `{sel}`\n")
    tierc = part["tier_clusters"]
    tiere = part["tier_entries"]
    A("| Tier | Clusters | Entries |\n|---|---:|---:|")
    for t in dg.TIER_ORDER:
        A(f"| {t} | {tierc.get(t, 0)} | {tiere.get(t, 0):,} |")
    A(f"| noise (unassigned) | — | {tiere.get('noise', part['noise_entries']):,} |")
    A(f"| excluded (content-free after stripping) | — | {part['excluded_by_preprocessing']:,} |")
    A("\nAll groups, largest first (stability = mean best-match Jaccard across seeds; margin = mean cosine margin to nearest other centroid):\n")
    A("| Cluster | Tier | Entries | Distinct texts | EN/FR | Stability | Cohesion | Margin | Label (top supported terms) | Flags |")
    A("|---:|---|---:|---:|---|---:|---:|---:|---|---|")
    for r in groups_tab.itertuples():
        A(f"| {r.cluster_id} | {r.tier} | {r.n_entries} | {r.distinct_near_dup_groups} | {r.english}/{r.french} | {r.stability_jaccard_mean} | "
          f"{r.cohesion_mean} | {r.margin_mean} | {r.short_label or '—'} | {r.flags.replace(';', ', ') or '—'} |")
    A("\n### Evidence-backed descriptions (largest candidate and needs-review groups)\n")
    shown = [s for s in sorted(summ, key=lambda s: -s["n_entries"]) if s["tier"] in ("candidate", "needs_review")][:10]
    for s in shown:
        A(f"- **Cluster {s['cluster_id']} — {s['short_label'] or 'no supported term'}** ({s['tier']}, {s['n_entries']} entries). {s['description']}")
        for e in s["evidence"][1:3]:
            A(f"  - [corpus_id {e['corpus_id']}, {e['source_dataset']}, record {e['source_record_id']}, {e['question_slug']}, {e['language']}] “{e['quote']}”")
    A("\nFull descriptions and up to five source-linked examples per cluster: `selected/concern_groups.csv`, "
      f"`configurations/{sel}/concern_summaries.json`, `configurations/{sel}/representative_examples.csv` (with `text_sha256`).\n")

    A("## 7. Qualitative error analysis\n")
    A(errors["note"] + "\n")
    for tier in ("campaign_or_template", "non_concern_pattern", "unreliable"):
        for c in errors["tier_examples"].get(tier, [])[:1]:
            A(f"- **{tier}** cluster {c['cluster_id']} ({c['n_entries']} entries; flags {', '.join(c['flags'])}): "
              + "; ".join(f"[{e['corpus_id']}] “{sm.quotable_span(e['text'], 100)}”" for e in c["examples"][:2]))
    A("\n**Members closer to another cluster's centroid** (possible mixing or boundary cases):\n")
    for e in errors["members_closer_to_another_cluster"]:
        A(f"- [{e['corpus_id']}] in cluster {e['assigned_cluster']} ({e['assigned_label']}), nearer to {e['nearer_cluster']} "
          f"({e['nearer_label']}), margin {e['margin']}: “{sm.quotable_span(e['text'], 120)}”")
    A("\n**Noise texts closest to a cluster** (candidates for missed membership):\n")
    for e in errors["noise_close_to_a_cluster"]:
        A(f"- [{e['corpus_id']}] nearest cluster {e['nearest_cluster']} ({e['nearest_label']}), cosine {e['cosine_to_centroid']}: “{sm.quotable_span(e['text'], 120)}”")
    A("\n**Noise texts farthest from every cluster** (genuine outliers or idiosyncratic comments):\n")
    for e in errors["noise_far_from_all_clusters"]:
        A(f"- [{e['corpus_id']}] cosine {e['cosine_to_centroid']}: “{sm.quotable_span(e['text'], 120)}”")
    nl = errors["noise_texts_by_length"]
    A(f"\nNoise texts have median length {nl['median_chars_noise']} characters versus {nl['median_chars_clustered']} for clustered texts.\n")
    A("**Least stable clusters across seeds:** " + "; ".join(f"cluster {c['cluster_id']} ({c['label']}, {c['tier']}) Jaccard {c['stability_jaccard_mean']}"
                                                             for c in errors["least_stable_clusters"]) + ".\n")
    A("**Cross-lingual groupings** (most central English and French member of the same group; language is source metadata, "
      "so a questionnaire answer written in French in the English-language survey is listed as English):\n")
    for c in errors["cross_lingual_groupings"]:
        p = c["most_central_by_language"]
        A(f"- cluster {c['cluster_id']} ({c['label']}, French share {c['french_share']}): EN [{p.get('English', {}).get('corpus_id')}] "
          f"“{sm.quotable_span(p.get('English', {}).get('text', ''), 90)}” / FR [{p.get('French', {}).get('corpus_id')}] "
          f"“{sm.quotable_span(p.get('French', {}).get('text', ''), 90)}”")

    A("\n## 8. Human evaluation status\n")
    A("**Internal metrics versus semantic evidence.** Sections 4–7 contain internal evidence only: geometry, seed agreement, metadata "
      "association, lexical support and rule-selected examples. They show that the groups are reproducible and where they are confounded. "
      "They do not show that any group is a coherent policy concern or that any description is faithful. That requires the human judgments below.\n")
    if pkg_m:
        A(f"- **Concern-cluster review package (new):** `data/full_corpus/concern_discovery_v1/human_review/`. Task A: {pkg_m['task_a_items']} intrusion/coherence items "
          f"(strata {pkg_m['strata_task_a']}; tiers {pkg_m['tiers_task_a']}). Task B: {pkg_m['task_b_items']} summary-support items. "
          f"Leakage check: {pkg_m['leakage_check']}. Templates contain comment text only; rows are shuffled independently per reviewer; "
          "the key is in `_key_do_not_share/` and is read only after both reviewers' files validate against the templates.")
    if review_res is None or review_res.get("status") != "COMPLETE":
        st = review_res or {"status": "BLOCKED", "missing": ["package not built"]}
        A(f"- **Result: {st['status']}.** {('Missing: ' + ', '.join(st.get('missing', []))) if st.get('missing') else '; '.join(st.get('problems', []))}. "
          "No reviewer judgments exist, so **no intrusion accuracy, agreement statistic, coherence rating or summary-faithfulness result is reported**. "
          "This is a research limitation, not a result.")
    else:
        A(f"- **Result: COMPLETE.** Chance rate {review_res['chance_rate_task_a']}.")
        for r, m in review_res["per_reviewer"].items():
            A(f"  - {r}: intrusion accuracy {m['intrusion_accuracy']} (95% CI {m['intrusion_ci95']}), suitable {m['intrusion_accuracy_suitable']}, control {m['intrusion_accuracy_control']}")
        A(f"  - Agreement task A: {review_res['agreement_task_a']}; task B: {review_res['agreement_task_b']}")
        A(f"  - Coherence by tier: {review_res['coherence_by_tier']}")
        A(f"  - Internal metric vs human coherence (Spearman): {review_res.get('internal_metric_vs_human_coherence')}")
    for a in ann:
        A(f"- **{a['name']}** ({a['section'].split('. ', 1)[-1]}): {a['status']} — {a['detail']}")
    A("- Stance annotation and concern-cluster review measure different constructs; neither is used as a label for the other, and no stance or sentiment is inferred.\n")

    A("## 9. Limitations\n")
    for x in [
        "No completed human review exists; tiers and descriptions are unvalidated proxies.",
        "Noise is substantial (about a third of the clustering population) and noise texts are longer than clustered texts, so long, multi-issue comments are under-represented in the concern groups.",
        "All groupings depend on one multilingual MiniLM encoder (384-D; texts beyond its maximum length, 0.8% per the run_v1 report, are truncated) and on UMAP/HDBSCAN settings; agreement between encoders is low.",
        "Seed stability covers UMAP seeds only (three runs). Per-cluster Jaccard with three seeds is a coarse estimate.",
        "Thresholds for tiers, near-duplicates and lexicons are heuristic. They were set and revised during development on this same corpus (the promotional-pattern rule, a markup-driven flag for raw-text clusterings and the selection criterion were changed after inspecting development runs; see section 3). They were not tuned against human judgments, because none exist. The review package includes control clusters so the flags can be tested against people.",
        "The no-opinion and promotional lexicons are English/French keyword rules; they miss paraphrases and can match substantive comments that quote such phrases.",
        "c-TF-IDF terms come from experiments v2 and are bag-of-words; a term's share in members says it is frequent, not that it states the concern.",
        "Questionnaire language is the survey language selected, not necessarily the comment's language; 6 Other/Unknown entries are kept.",
        "Entries are not respondents: one respondent may contribute up to three questionnaire entries, and the sources carry no respondent id.",
        "The corpus is Canadian federal Open Government consultation data (2016-18). Nothing here validates use on Indian MCA consultations or other domains.",
    ]:
        A(f"- {x}")
    A("\n## 10. Reproduction\n")
    A("```powershell\npython scripts\\run_concern_discovery.py run              # ~1-2 min, no model loading\n"
      "python scripts\\run_concern_discovery.py review-package\npython scripts\\run_concern_discovery.py report\n"
      "# after reviewers save completed/reviewer_{1,2}_task_{a,b}.csv:\npython scripts\\run_concern_discovery.py analyze-reviews\n"
      "python -m pytest tests\\test_concern_discovery.py -q\n```\n")
    A(f"Outputs are write-once; `run_manifest.json` records SHA-256 of every input and output. Runtime of this run: {man['runtime_s']} s.\n")
    if (ROOT / "data" / "reviewer_handoff" / "REVIEWER_HANDOFF_GUIDE.md").exists() and out.resolve() == OUT.resolve():
        A(handoff_section())
    text = "\n".join(L) + "\n"
    write_new(report_path, lambda f: f.write(text))
    return text


EVIDENCE_TABLE = """### Evidence status: automated verification versus human validation

| Kind of evidence | Status | What it shows |
|---|---|---|
| Automated verification (tests, integrity hashes, grounding checks, leak audits, intake validation) | See `run_research_evaluation.py` | The software, files and packets behave as specified |
| Internal proxy metrics (sections 4-7) | Computed | Reproducibility and confounding; **not** concern coherence |
| Human-validated findings | Only for tracks whose analysis ran on accepted completed files | Until then each analysis reports BLOCKED |
"""


def handoff_section(root: Path = ROOT) -> str:
    """Section 11: exact remaining human-evaluation requirements, rendered from the live intake status."""
    import human_review_handoff as hh
    checks = hh.status(root)
    ready = hh.track_readiness(checks)
    L = ["## 11. Human-evaluation handoff and remaining requirements\n",
         f"Status checked {datetime.now(timezone.utc).isoformat(timespec='seconds')}. **No completed reviewer file exists; no part of "
         "PolicyLens is human-validated.** Every human-evaluation result below is pending.\n" if all(c.state == "MISSING" for c in checks) else
         f"Status checked {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Results exist only for tracks whose analysis has been run on valid completed files.\n",
         "Reviewer-facing guide: `data/reviewer_handoff/REVIEWER_HANDOFF_GUIDE.md`. Stage folders: `data/reviewer_handoff/packets/stage_N/person_N/` "
         "(built by `scripts/human_review_handoff.py`; allow-list only, leak-audited, no answer keys).\n",
         EVIDENCE_TABLE,
         "### Remaining requirements\n",
         "| Track | What is needed | Completed files (exact paths) | Current status |", "|---|---|---|---|"]
    need = {"stance_pilot": "2 independent annotators label 50 pilot comments (stance, concern, requested action)",
            "pair_review_44": "2 independent bilingual reviewers rate 44 comment pairs (0-3 or U)",
            "cluster_task_a": "2 independent bilingual reviewers: intruder + coherence on 32 items",
            "cluster_task_b": "same reviewers, after task A is accepted: summary support on 28 items",
            "stance_v2": "after pilot calibration (kappa target 0.60) and all concern files: 350 additional comments"}
    for t in ("stance_pilot", "pair_review_44", "cluster_task_a", "cluster_task_b", "stance_v2"):
        paths = "<br>".join(f"`{hh.TRACKS[t].completed[n].as_posix()}`" for n in (1, 2))
        L.append(f"| {t} | {need[t]} | {paths} | {ready[t]} |")
    L += ["\n### Order of work (prevents cross-task contamination)\n",
          "1. Stage 1, in parallel and independently: stance pilot, 44-pair review, comment-group task A.",
          "2. Stage 2: comment-group task B is released per reviewer only after that reviewer's task A has been accepted "
          "(SHA-256 receipt). Five task-A intruder comments reappear in task B next to their own group, so task A must be locked first.",
          "3. Stage 3: only after every stage-1/2 file of both people is accepted, the two annotators may discuss pilot disagreements "
          "and calibrate the stance guidelines; then the 350-record stance round is released. Machine-written descriptions (task B) are "
          "therefore seen only after each person's own concern judgments are locked; the stance round still follows task B, so its "
          "free-text concern field may be anchored by those descriptions (documented residual risk).\n",
          "### Intake and analysis commands\n", "```powershell",
          "python scripts\\human_review_handoff.py status          # read-only: MISSING / VALID / INVALID per file, readiness per track",
          "python scripts\\human_review_handoff.py accept --track <track> --person <1|2> --file <returned.csv>",
          "python scripts\\human_review_handoff.py accept --track <track> --person <1|2> --file <its completed/ path>   # receipt in place",
          "python scripts\\human_review_handoff.py build-packets --stage 2 --person <1|2>   # after that person's task A is accepted",
          "python scripts\\human_review_handoff.py build-packets --stage 3                 # after all stage-1/2 files + pilot analysis",
          "python scripts\\analyze_annotation_agreement.py --round pilot --check-only      # then without --check-only",
          "python scripts\\analyze_concern_review_agreement.py",
          "python scripts\\run_concern_discovery.py analyze-reviews",
          "python scripts\\analyze_annotation_agreement.py --round expansion_v2 --check-only   # then without --check-only",
          "python scripts\\run_research_evaluation.py --no-write",
          "```\n",
          "### Completed-file requirements and what each analysis does when they are not met\n",
          "| Analysis | Requires | Missing reviewer | Invalid file |", "|---|---|---|---|",
          "| Stance (`analyze_annotation_agreement.py`) | both annotators' files; columns, row ids and order unchanged; `annotator_A`/`annotator_B`; stance in Support/Oppose/Mixed/Neutral/Unclear; verbatim evidence spans | status `missing_files`, exit 2, nothing written | exit 1, errors listed, nothing written |",
          "| 44-pair (`analyze_concern_review_agreement.py`) | both reviewers' files; 44 pair ids; values 0-3/U; pair text and source references unchanged (checked against the catalog since this round) | `no_completed_files`, exit 1, nothing written | `validation_failed`, exit 1, nothing written. Blank ratings are tolerated and reported as `either_blank` (established behaviour); intake rejects them first |",
          "| Comment-group (`run_concern_discovery.py analyze-reviews`) | both reviewers' task A and task B files; columns/items/text unchanged (line-ending differences tolerated); valid values, no blanks | `BLOCKED`, exit 0, answer key not opened, nothing written | `INVALID`, exit 1, answer key not opened |",
          "| Intake (`human_review_handoff.py`) | one file at a time against its own blank template; never opens a key, catalog or model output | `MISSING` | `INVALID` with the cells to fix; `accept` refuses to store it |",
          "\n### Reviewer instructions verified\n",
          "- IDs: every template has unique, non-blank ids; comment-group ids are opaque random tokens; the two reviewers' templates hold the same items in independent orders (comment-group) or the same order (stance, 44-pair).",
          "- No leakage: the leak audit finds no cluster ids, tiers, strata, model names, scores or expected answers in any template or packet; "
          "`pair_catalog.csv`, `selection_manifest.json`, `review_package_report.md`, `package_manifest.json` and `_key_do_not_share/` are excluded by allow-list.",
          "- Bilingual handling: reviewers judge each comment in its own language without machine translation; language metadata is the survey language for questionnaire rows; "
          "uninterpretable comments get `U` (review tracks) or `Unclear` with `non-English: <language>` (stance), and notes use the tags `[language]`, `[context]`, `[ambiguous]`, `[not a concern]`.",
          "- Context: 44-pair and stance templates include the question prompt; comment-group items deliberately show text only (a prompt would reveal question-driven grouping), so reviewers mark answers that need their prompt as uninterpretable.",
          "- File safety: 8 comment-group and 6 other template comments begin with `@`, and 8 comment-group and 12 other comments contain line breaks; the guide gives spreadsheet import settings that keep them as text, and intake detects formula corruption.",
          "\n### Known limitations of the human-evaluation design\n",
          "- In task A the intruder is the longest of the six comments in 10 of 32 items (about 5 expected by chance). Intrusion accuracy should be reported alongside "
          "accuracy on items where the intruder is not the longest; this is disclosed here and not to reviewers.",
          "- 4 to 12 comments overlap between pairs of tracks; seeing a comment twice is unavoidable with one corpus and is mitigated only by the stage order.",
          "- Two reviewers per track give agreement estimates with wide confidence intervals; results describe these samples, not the population.\n"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PolicyLens concern discovery pipeline")
    p.add_argument("stage", choices=["run", "review-package", "analyze-reviews", "report", "all"])
    p.add_argument("--out", type=Path, default=OUT, help="output directory (default: data/full_corpus/concern_discovery_v1)")
    p.add_argument("--report", type=Path, default=REPORT)
    a = p.parse_args(argv)
    try:
        if a.stage in ("run", "all"):
            r = stage_run(a.out)
            print(json.dumps({"selected": r["selected"], "runtime_s": r["runtime_s"], "near_duplicates": r["near_duplicates"],
                              "configs": {k: {"clusters": v["clusters"], "tiers": v["tier_clusters"],
                                              "usable_core": v["usable_concern_coverage_core"]} for k, v in r["comparison"].items()}},
                             indent=2, default=_json_default))
        if a.stage in ("review-package", "all"):
            m = stage_review_package(a.out)
            print(f"review package: {m['task_a_items']} task-A items, {m['task_b_items']} task-B items, leakage check {m['leakage_check']}")
        if a.stage == "analyze-reviews":
            r = stage_analyze_reviews(a.out)
            print(json.dumps(r, indent=2, default=_json_default))
            return 0 if r["status"] in ("COMPLETE", "BLOCKED") else 1
        if a.stage in ("report", "all"):
            stage_report(a.out, a.report)
            print(f"written {a.report}")
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    except (CorpusIntegrityError, sm.GroundingError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
