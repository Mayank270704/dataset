"""
scripts/run_prioritization_audit_v1.py
======================================
Audit of the evidence-linked concern prioritization (data/full_corpus/concern_prioritization_v1): 55 ranked and
19 not-ranked groups. Read-only on all inputs; writes data/full_corpus/prioritization_audit_v1/ (write-once).

1. Reproducibility: recompute signals and scores with the saved default weights and compare with ranking.csv.
2. Ranking stability: 200 subsamples of 80 % of SOURCE RECORDS (without replacement; cluster structure, tiers,
   stability values and evidence counts held fixed). Per group: median rank, 5-95 % rank interval, top-5/top-10 frequency;
   Kendall tau of each replicate vs the full ranking. Adjacent score gaps of the full ranking.
3. Contribution explanation: weight x scaled value per signal (sums to the score); mean share of each signal; the
   signals that separate each top-10 group from the ranked mean.
4. Not-ranked groups: reasons.
5. Evidence verification (independent of the existing ledger): every ranking_evidence and representative_examples quote
   is a substring of its full text; full text equals corpus_entries text for that corpus_id; source dataset / record /
   row / field agree with the corpus; the corpus entry is assigned to the stated cluster; text SHA-256 matches.
   The existing raw-file lineage check (scripts/verify_evidence_traceability.py --check-only) is run and its result recorded.

No impact or urgency labels are created. Counts are entries / texts / source records, never people.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.stats import kendalltau  # noqa: E402

from models.concern_discovery import prioritization as pr  # noqa: E402
import run_concern_prioritization as rcp  # noqa: E402

PZ = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"
OUT = ROOT / "data" / "full_corpus" / "prioritization_audit_v1"
N_BOOT, FRAC, SEED = 200, 0.8, 42


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        print(f"[STOP] {OUT} exists", file=sys.stderr)
        return 2
    d = rcp.load_inputs()
    entries, groups = d["entries"], d["groups"]
    cfg = json.loads((PZ / "config.json").read_text(encoding="utf-8"))
    w, tq = cfg["weights_normalised"], cfg["tier_quality"]
    saved_sig = pd.read_csv(PZ / "concern_signals.csv", keep_default_na=False)
    ev_counts = dict(zip(saved_sig.cluster_id.astype(int), saved_sig.verified_evidence.astype(int)))
    saved_rank = pd.read_csv(PZ / "ranking.csv", keep_default_na=False)

    # 1 reproducibility
    sig = pr.compute_signals(entries, groups, ev_counts)
    full = pr.score(sig, w, tq)
    repro = {"same_order": full.cluster_id.tolist() == saved_rank.cluster_id.astype(int).tolist(),
             "max_abs_score_diff": round(float(np.abs(full.score.to_numpy() - saved_rank.score.astype(float).to_numpy()).max()), 8),
             "ranked": int(len(full)), "not_ranked": int((~sig.ranked).sum())}

    # 2 stability
    rec = (entries.source_dataset + "#" + entries.source_record_id).to_numpy()
    urec = np.unique(rec)
    rng = np.random.default_rng(SEED)
    ranks, taus = [], []
    base_rank = dict(zip(full.cluster_id, full.rank_multi))
    for _ in range(N_BOOT):
        keep = set(rng.choice(urec, size=int(FRAC * len(urec)), replace=False))
        sub = entries[np.isin(rec, list(keep))]
        s = pr.compute_signals(sub, groups, ev_counts)
        s = s[s.cluster_id.isin(full.cluster_id)]
        r = pr.score(s, w, tq)
        rk = dict(zip(r.cluster_id, r.rank_multi))
        ranks.append(rk)
        common = [c for c in full.cluster_id if c in rk]
        taus.append(kendalltau([base_rank[c] for c in common], [rk[c] for c in common])[0])
    R = pd.DataFrame(ranks)[full.cluster_id.tolist()]
    stab = pd.DataFrame({"cluster_id": full.cluster_id, "short_label": full.short_label, "tier": full.tier,
                         "rank_full": full.rank_multi, "score": full.score,
                         "rank_median": R.median().values, "rank_p5": R.quantile(0.05).values, "rank_p95": R.quantile(0.95).values,
                         "top5_freq": (R <= 5).mean().values.round(3), "top10_freq": (R <= 10).mean().values.round(3),
                         "present_in_replicates": R.notna().mean().values.round(3)})
    stab["rank_interval_width"] = stab.rank_p95 - stab.rank_p5
    gaps = -np.diff(full.score.to_numpy())

    # 3 contributions
    contrib = full[[f"contrib_{k}" for k in pr.SIGNALS]]
    share = contrib.div(full.score, axis=0)
    mean_c = contrib.mean()
    top_drivers = []
    for i in range(min(10, len(full))):
        dev = (contrib.iloc[i] - mean_c).sort_values()
        top_drivers.append({"rank": int(full.rank_multi.iloc[i]), "cluster_id": int(full.cluster_id.iloc[i]), "short_label": full.short_label.iloc[i],
                            "score": float(full.score.iloc[i]),
                            "largest_positive": {k.replace("contrib_", ""): round(float(v), 4) for k, v in dev[::-1].head(2).items()},
                            "largest_negative": {k.replace("contrib_", ""): round(float(v), 4) for k, v in dev.head(1).items()}})

    # 4 not ranked
    nr = sig[~sig.ranked]
    not_ranked = {"count": int(len(nr)), "reasons": nr.not_ranked_reason.value_counts().to_dict(), "by_tier": nr.tier.value_counts().to_dict()}

    # 5 evidence verification
    corpus = pd.read_csv(ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv", dtype=str, keep_default_na=False)
    corpus = corpus.set_index(corpus.corpus_id.astype(int))
    clus = dict(zip(entries.corpus_id, entries.cluster_id))
    ev = pd.read_csv(PZ / "ranking_evidence.csv", dtype=str, keep_default_na=False)
    rep = pd.read_csv(ROOT / "data" / "full_corpus" / "concern_discovery_v1" / "configurations" / d["selected"] / "representative_examples.csv", dtype=str, keep_default_na=False)
    rank_to_cluster = dict(zip(full.rank_multi.astype(str), full.cluster_id))
    checks = []
    for kind, df in (("ranking_evidence", ev), ("representative_examples", rep)):
        for r in df.itertuples():
            cid = int(r.corpus_id)
            c = corpus.loc[cid] if cid in corpus.index else None
            exp_cluster = int(r.cluster_id)
            row = {"kind": kind, "cluster_id": exp_cluster, "corpus_id": cid,
                   "quote_in_full_text": r.quote.rstrip("…").rstrip(".") in r.full_text or r.quote in r.full_text,
                   "full_text_equals_corpus": c is not None and c.original_text == r.full_text,
                   "source_ids_agree": c is not None and c.source_dataset == r.source_dataset and c.source_record_id == r.source_record_id,
                   "entry_in_stated_cluster": clus.get(cid) == exp_cluster,
                   "text_sha256_ok": hashlib.sha256(r.full_text.encode("utf-8")).hexdigest() == r.text_sha256}
            if kind == "ranking_evidence":
                row["rank_matches_cluster"] = rank_to_cluster.get(r.rank_multi) == exp_cluster
                row["row_and_field_agree"] = c is not None and c.source_row_number == r.source_row_number and c.source_field == r.source_field
            checks.append(row)
    ck = pd.DataFrame(checks)
    bool_cols = [c for c in ck.columns if c not in ("kind", "cluster_id", "corpus_id")]
    ev_summary = {k: {c: int((~g[c].astype(bool)).sum()) for c in bool_cols if g[c].notna().all()} | {"rows": int(len(g))}
                  for k, g in ck.groupby("kind")}
    ev_summary["groups_with_representatives"] = int(rep.cluster_id.nunique())
    ev_summary["ranked_groups_with_evidence"] = int(ev.cluster_id.nunique())
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_evidence_traceability.py"), "--check-only"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    ev_summary["raw_lineage_check_only_exit_code"] = proc.returncode
    ev_summary["raw_lineage_check_only_tail"] = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]

    OUT.mkdir(parents=True)
    stab.to_csv(OUT / "rank_stability.csv", index=False, lineterminator="\n")
    ck.to_csv(OUT / "evidence_checks.csv", index=False, lineterminator="\n")
    share.assign(cluster_id=full.cluster_id, rank=full.rank_multi).to_csv(OUT / "contribution_shares.csv", index=False, lineterminator="\n")
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "status": "audit of an exploratory ranking; not a measure of importance, impact or urgency; no human validation exists",
           "reproducibility": repro,
           "stability": {"replicates": N_BOOT, "subsample": "80% of source records without replacement", "seed": SEED,
                         "kendall_tau_vs_full": {"mean": round(float(np.mean(taus)), 4), "p5": round(float(np.percentile(taus, 5)), 4), "min": round(float(np.min(taus)), 4)},
                         "median_rank_interval_width": float(stab.rank_interval_width.median()),
                         "groups_top10_in_ge_90pct": stab[stab.top10_freq >= 0.9].cluster_id.tolist(),
                         "groups_top5_in_ge_90pct": stab[stab.top5_freq >= 0.9].cluster_id.tolist(),
                         "top10_full_with_top10_freq_lt_0.5": stab[(stab.rank_full <= 10) & (stab.top10_freq < 0.5)].cluster_id.tolist(),
                         "adjacent_score_gap": {"median": round(float(np.median(gaps)), 5), "min": round(float(gaps.min()), 6), "max": round(float(gaps.max()), 5)},
                         "weight_sensitivity_existing": json.loads((PZ / "sensitivity.json").read_text(encoding="utf-8"))["min_kendall_tau"]},
           "contributions": {"weights": w, "mean_share_of_score": share.mean().round(4).rename(lambda k: k.replace("contrib_", "")).to_dict(),
                             "top10_drivers": top_drivers},
           "not_ranked": not_ranked, "evidence_verification": ev_summary,
           "inputs_sha256": d["inputs_sha256"] | {f"concern_prioritization_v1/{f}": sha(PZ / f) for f in ("ranking.csv", "concern_signals.csv", "config.json", "ranking_evidence.csv")}}
    (OUT / "audit_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"outputs_sha256": {f.name: sha(f) for f in sorted(OUT.iterdir())}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("reproducibility", "stability", "not_ranked", "evidence_verification")}, indent=1, default=str, ensure_ascii=False))
    print(json.dumps(res["contributions"]["mean_share_of_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
