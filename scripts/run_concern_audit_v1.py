"""
scripts/run_concern_audit_v1.py
===============================
Audit of the 74 selected concern-discovery groups (strip__umap_m25s5, seed 42), a validation of nearest-centroid
assignment for NEW comments, and one alternative configuration (question-field centering). Read-only on all
existing artifacts; outputs go to new write-once directories.

Part 1  group audit: sizes, noise, cohesion, stability, generic/mixed/question-driven flags, language effects.
Part 2  configuration sensitivity: best-match Jaccard of every selected group in other seeds/configurations.
Part 3  nearest-centroid assignment validation (5-fold, by group). Acceptance criteria are FIXED here, before results:
        accepted-item agreement >= 0.80 AND share of HDBSCAN-noise texts accepted <= 0.50. If either fails, new
        comments must NOT be assigned to existing groups.
Part 4  alternative: subtract the per-question-field mean embedding (field centering), re-normalise, run the same
        UMAP+HDBSCAN (m25s5) for seeds 42/43/44; compare question-field association, question-driven groups, noise
        and seed stability with the selected configuration. Saved separately; the selected run is not replaced.

    python scripts/run_concern_audit_v1.py            # all parts
    python scripts/run_concern_audit_v1.py --skip-alternative
Outputs: data/full_corpus/concern_audit_v1/, data/full_corpus/experiments_field_centered_v1/
Every figure is an internal proxy; no group is human-validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import run_clustering_experiments_v2 as ce  # noqa: E402

FC = ROOT / "data" / "full_corpus"
CD = FC / "concern_discovery_v1"
EXP = FC / "experiments_v2"
OUT = FC / "concern_audit_v1"
ALT = FC / "experiments_field_centered_v1"
SELECTED = "strip__umap_m25s5"
CRITERIA = {"accepted_agreement_min": 0.80, "noise_acceptance_max": 0.50, "threshold_rule": "10th percentile of training members' cosine to their own centroid (per fold)"}
QUESTION_DRIVEN_SHARE = 0.80
GENERIC = {"open", "government", "data", "information", "canada", "canadian", "canadians", "public", "citizens", "people",
           "gouvernement", "ouvert", "ouverte", "données", "need", "good", "important", "think", "make", "help"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    t = pd.crosstab(a, b).to_numpy().astype(float)
    n = t.sum()
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = ((t - e) ** 2 / np.where(e == 0, 1, e)).sum()
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def entry_cluster_table(labels_u: np.ndarray, entries: pd.DataFrame) -> pd.DataFrame:
    e = entries[["corpus_id", "unique_id", "question_slug", "language", "source_dataset"]].copy()
    e["cl"] = labels_u[e.unique_id.to_numpy()]
    return e


def question_profile(e: pd.DataFrame) -> pd.DataFrame:
    c = e[e.cl >= 0]
    g = c.groupby("cl")
    top = g.question_slug.agg(lambda s: s.value_counts().index[0])
    share = g.question_slug.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return pd.DataFrame({"top_question": top, "top_question_share": share.round(4), "n_entries": g.size()})


def best_match(sel: np.ndarray, other: np.ndarray, mask: np.ndarray) -> dict[int, tuple[float, int]]:
    out = {}
    o_sets = {c: set(np.where((other == c) & mask)[0]) for c in np.unique(other[mask]) if c >= 0}
    for c in np.unique(sel[sel >= 0]):
        s = set(np.where((sel == c) & mask)[0])
        best, bid = 0.0, -1
        for oc, os_ in o_sets.items():
            inter = len(s & os_)
            if inter:
                j = inter / len(s | os_)
                if j > best:
                    best, bid = j, int(oc)
        out[int(c)] = (round(best, 4), bid)
    return out


def part3_assignment(emb: np.ndarray, labels_u: np.ndarray, pop: np.ndarray, seed: int = 42) -> dict:
    members = pop[labels_u[pop] >= 0]
    noise = pop[labels_u[pop] == -1]
    y = labels_u[members]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred, sim, acc_flag = np.empty(len(members), int), np.empty(len(members)), np.empty(len(members), bool)
    taus = []
    for tr, te in skf.split(members, y):
        cl = np.unique(y[tr])
        cent = np.vstack([emb[members[tr][y[tr] == c]].mean(0) for c in cl])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        own = (emb[members[tr]] * cent[np.searchsorted(cl, y[tr])]).sum(1)
        tau = float(np.percentile(own, 10))
        taus.append(tau)
        s = emb[members[te]] @ cent.T
        pred[te], sim[te] = cl[s.argmax(1)], s.max(1)
        acc_flag[te] = sim[te] >= tau
    cl = np.unique(y)
    cent = np.vstack([emb[members[y == c]].mean(0) for c in cl])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    tau_all = float(np.mean(taus))
    ns = (emb[noise] @ cent.T).max(1)
    agree = pred == y
    res = {"criteria_fixed_before_run": CRITERIA, "members": int(len(members)), "noise_texts": int(len(noise)),
           "top1_agreement_all_heldout": round(float(agree.mean()), 4),
           "threshold_mean": round(tau_all, 4), "accepted_share_of_heldout_members": round(float(acc_flag.mean()), 4),
           "agreement_among_accepted": round(float(agree[acc_flag].mean()), 4),
           "noise_acceptance_share": round(float((ns >= tau_all).mean()), 4),
           "per_group_agreement_min": round(float(pd.Series(agree).groupby(y).mean().min()), 4),
           "per_group_agreement_median": round(float(pd.Series(agree).groupby(y).mean().median()), 4)}
    res["valid_for_new_comment_assignment"] = bool(res["agreement_among_accepted"] >= CRITERIA["accepted_agreement_min"]
                                                   and res["noise_acceptance_share"] <= CRITERIA["noise_acceptance_max"])
    res["centroids"], res["cluster_ids"] = cent, cl
    return res


def run_alternative(variants, unique, entries, core, sel_labels, out: Path) -> dict:
    v = variants["strip"]
    pop = v.pop_idx
    q = entries.groupby("unique_id").question_slug.agg(lambda s: s.value_counts().index[0]).reindex(range(len(unique))).to_numpy()
    emb = v.emb.copy()
    for f in np.unique(q[pop]):
        idx = pop[q[pop] == f]
        emb[idx] = emb[idx] - emb[idx].mean(0)
    emb[pop] /= np.linalg.norm(emb[pop], axis=1, keepdims=True)
    labs, info = {}, {}
    for s in ce.SEEDS:
        t0 = time.time()
        lab = np.where(v.state != 0, v.state, -1).astype(int)
        lp, _, rv = ce.run_algo(emb[pop], ce.ALGOS["umap_m25s5"], s)
        lab[pop] = lp
        labs[s] = lab
        info[s] = {"clusters": int(len(np.unique(lp[lp >= 0]))), "noise_share_pop": round(float((lp == -1).mean()), 4),
                   "relative_validity": rv, "runtime_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True)
    for s, lab in labs.items():
        np.save(out / f"labels_seed{s}.npy", lab)
    ari = {f"{a}-{b}": ce.partition_ari(labs[a], labs[b], core) for a, b in ((42, 43), (42, 44), (43, 44))}
    e_alt, e_sel = entry_cluster_table(labs[42], entries), entry_cluster_table(sel_labels, entries)
    qa, qs = question_profile(e_alt), question_profile(e_sel)
    ca, cs = e_alt[e_alt.cl >= 0], e_sel[e_sel.cl >= 0]
    cmp = {"selected": {"clusters": int(len(qs)), "noise_share_pop": round(float((sel_labels[pop] == -1).mean()), 4),
                        "question_driven_groups": int((qs.top_question_share >= QUESTION_DRIVEN_SHARE).sum()),
                        "cramers_v_question_field_entries": round(cramers_v(cs.cl.to_numpy(), cs.question_slug.to_numpy()), 4),
                        "cramers_v_language_entries": round(cramers_v(cs.cl.to_numpy(), cs.language.to_numpy()), 4)},
           "field_centered": {"clusters": int(len(qa)), "noise_share_pop": info[42]["noise_share_pop"],
                              "question_driven_groups": int((qa.top_question_share >= QUESTION_DRIVEN_SHARE).sum()),
                              "cramers_v_question_field_entries": round(cramers_v(ca.cl.to_numpy(), ca.question_slug.to_numpy()), 4),
                              "cramers_v_language_entries": round(cramers_v(ca.cl.to_numpy(), ca.language.to_numpy()), 4),
                              "seed_ari": ari, "seed_ari_min": min(x["ari"] for x in ari.values()), "per_seed": info},
           "selected_vs_field_centered_ari_core": ce.partition_ari(sel_labels, labs[42], core)}
    qa.to_csv(out / "cluster_question_profile.csv", lineterminator="\n")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": "per-question-field mean embedding subtracted from strip-variant embeddings, re-normalised; UMAP(5d, nn15, min_dist 0, cosine, exact kNN) + HDBSCAN(25/5 EOM)",
           "status": "alternative experiment only; NOT adopted, NOT summarised, NOT human-reviewed", "comparison": cmp}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=ce._json_default if hasattr(ce, "_json_default") else str) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir())}}, indent=2) + "\n", encoding="utf-8")
    return cmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-alternative", action="store_true")
    a = ap.parse_args(argv)
    if OUT.exists():
        print(f"[STOP] {OUT} exists", file=sys.stderr)
        return 2
    entries, unique, emb_raw = ce.load_base()
    variants = ce.build_variants(unique, emb_raw, None, EXP / "preprocess")
    core = ce.core_mask(variants)
    sel = np.load(EXP / SELECTED / "labels_seed42.npy")
    groups = pd.read_csv(CD / "selected" / "concern_groups.csv", keep_default_na=False)
    ea = pd.read_csv(CD / "selected" / "entry_assignments.csv", keep_default_na=False)
    # alignment checks between saved labels and the published group table
    assert set(groups.cluster_id) == set(np.unique(sel[sel >= 0])), "concern_groups ids differ from saved labels"
    cl_ea = ea[ea.cluster_id >= 0]
    assert (sel[cl_ea.unique_id.to_numpy()] == cl_ea.cluster_id.to_numpy()).all(), "entry assignments differ from saved labels"
    pop = variants["strip"].pop_idx

    # ── Part 1 ──
    e = entry_cluster_table(sel, entries)
    qp = question_profile(e)
    g = groups.set_index("cluster_id")
    aud = g[["tier", "n_entries", "n_unique_texts", "distinct_near_dup_groups", "english", "french", "cohesion_mean", "negative_margin_share",
             "stability_jaccard_mean", "keyword_support_top3_mean", "no_opinion_share", "flags", "short_label"]].copy()
    aud["french_share"] = (aud.french / aud.n_entries).round(4)
    aud = aud.join(qp[["top_question", "top_question_share"]])
    aud["question_driven"] = aud.top_question_share >= QUESTION_DRIVEN_SHARE
    terms = aud.short_label.str.lower().str.split(r"\s*/\s*")
    aud["generic_label"] = terms.map(lambda t: all(x.strip() in GENERIC for x in t if x.strip()))
    aud["mixed_flag"] = aud["flags"].astype(str).str.contains("mixed")
    aud["weak_cohesion"] = aud.cohesion_mean < aud.cohesion_mean.quantile(0.25)
    lang = entries.language.where(entries.language.isin(["English", "French"]), "other").to_numpy()
    st = ea.state.to_numpy()
    noise_by = pd.crosstab(lang, st, normalize="index").round(4)
    p1 = {"groups": int(len(aud)), "tiers": aud.tier.value_counts().to_dict(),
          "size_entries": aud.n_entries.describe().round(1).to_dict(), "largest_group_share_of_clustered": round(float(aud.n_entries.max() / aud.n_entries.sum()), 4),
          "entry_states": ea.state.value_counts().to_dict(), "state_share_by_language": noise_by.to_dict(),
          "question_driven_groups": int(aud.question_driven.sum()),
          "question_driven_by_question": aud[aud.question_driven].top_question.value_counts().to_dict(),
          "generic_label_groups": int(aud.generic_label.sum()), "mixed_flag_groups": int(aud.mixed_flag.sum()),
          "weak_cohesion_groups": int(aud.weak_cohesion.sum()),
          "groups_french_share_ge_0.5": int((aud.french_share >= 0.5).sum()), "groups_with_no_french": int((aud.french == 0).sum()),
          "flag_counts": pd.Series([f for x in aud["flags"] for f in str(x).split(";") if f]).value_counts().to_dict(),
          "cohesion_quantiles": aud.cohesion_mean.quantile([.1, .5, .9]).round(4).to_dict()}

    # ── Part 2 ──
    others = {f"seed{s}": EXP / SELECTED / f"labels_seed{s}.npy" for s in (43, 44)}
    others |= {c: EXP / c / "labels_seed42.npy" for c in ("strip__umap_m50s10", "strip_dedup__umap_m25s5", "no_short__umap_m25s5",
                                                          "strip__umap_m25s5_leaf", "strip__kmeans_k80", "raw__umap_m25s5")}
    sens = {}
    for name, p in others.items():
        if not p.is_file():
            continue
        o = np.load(p)
        bm = best_match(sel, o, core)
        aud[f"jac_{name}"] = aud.index.map(lambda c: bm[c][0])
        sens[name] = {"ari_core": ce.partition_ari(sel, o, core), "groups_best_jaccard_ge_0.5": int((aud[f"jac_{name}"] >= 0.5).sum()),
                      "median_best_jaccard": round(float(aud[f"jac_{name}"].median()), 4)}
    jcols = [c for c in aud.columns if c.startswith("jac_")]
    aud["robust_across_configs"] = (aud[jcols] >= 0.5).sum(1)
    p2 = {"comparisons": sens, "groups_matched_ge_0.5_in_all": int((aud.robust_across_configs == len(jcols)).sum()),
          "groups_matched_in_none": int((aud.robust_across_configs == 0).sum())}

    # ── Part 3 ──
    t0 = time.time()
    p3 = part3_assignment(variants["strip"].emb, sel, pop)
    cent, cids = p3.pop("centroids"), p3.pop("cluster_ids")
    p3["runtime_s"] = round(time.time() - t0, 1)

    OUT.mkdir(parents=True)
    aud.to_csv(OUT / "group_audit.csv", lineterminator="\n")
    if p3["valid_for_new_comment_assignment"]:
        np.savez(OUT / "nearest_centroid_model.npz", centroids=cent, cluster_ids=cids, threshold=p3["threshold_mean"])
    p4 = None
    if not a.skip_alternative:
        if ALT.exists():
            print(f"[SKIP] {ALT} exists")
        else:
            p4 = run_alternative(variants, unique, entries, core, sel, ALT)
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selected_configuration": f"{SELECTED} seed 42",
           "status": "internal-proxy audit; no group is human-validated",
           "part1_group_audit": p1, "part2_configuration_sensitivity": p2, "part3_nearest_centroid_assignment": p3,
           "part4_field_centered_alternative": p4 if p4 is not None else "not run (see experiments_field_centered_v1 if it exists)",
           "inputs_sha256": {"experiments_v2/strip__umap_m25s5/labels_seed42.npy": sha(EXP / SELECTED / "labels_seed42.npy"),
                             "concern_discovery_v1/selected/concern_groups.csv": sha(CD / "selected" / "concern_groups.csv"),
                             "concern_discovery_v1/selected/entry_assignments.csv": sha(CD / "selected" / "entry_assignments.csv")}}
    (OUT / "audit_results.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"outputs_sha256": {f.name: sha(f) for f in sorted(OUT.iterdir())}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"p1": {k: p1[k] for k in ("tiers", "question_driven_groups", "generic_label_groups", "mixed_flag_groups", "weak_cohesion_groups", "groups_french_share_ge_0.5")},
                      "p2": p2, "p3": p3, "p4": p4}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
