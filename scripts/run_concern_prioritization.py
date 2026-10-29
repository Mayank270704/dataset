"""
scripts/run_concern_prioritization.py
=====================================
Exploratory, evidence-linked prioritization of the concern groups found by scripts/run_concern_discovery.py.

Reuses (read-only, hash-verified against their manifests):
    data/full_corpus/run_v1/corpus/corpus_entries.csv           original texts and source ids
    data/full_corpus/concern_discovery_v1/selected/*            entry assignments and concern groups
    data/full_corpus/concern_discovery_v1/configurations/<selected>/representative_examples.csv
    data/full_corpus/concern_discovery_v1/duplicates/near_duplicate_groups.csv
No embedding, clustering or model is run. Outputs are new and write-once.

    python scripts/run_concern_prioritization.py run [--weights weights.json]
    python scripts/run_concern_prioritization.py report
    python scripts/run_concern_prioritization.py all

weights.json (optional): {"weights": {<signal>: w, ...}, "tier_quality": {"candidate": 1, "needs_review": 0.5,
"campaign_or_template": 0.5}}. Weights are renormalised to sum 1. They are documented choices, not fitted values.

The ranking orders groups for inspection. It is not a validated measure of public importance and says nothing about
respondents, stakeholders, urgency, policy impact or representativeness.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.concern_discovery import prioritization as pr  # noqa: E402
from models.concern_discovery.diagnostics import NO_OPINION_RE, PROMOTION_RE  # noqa: E402
from models.concern_discovery.schema import sha256_file, sha256_text  # noqa: E402
from models.concern_discovery.summaries import quotable_span  # noqa: E402

CD = ROOT / "data" / "full_corpus" / "concern_discovery_v1"
RUN_V1 = ROOT / "data" / "full_corpus" / "run_v1"
OUT = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"
REPORT = ROOT / "reports" / "concern_prioritization_report_v1.md"
NON_SUBSTANTIVE = re.compile(f"(?:{NO_OPINION_RE.pattern})|(?:{PROMOTION_RE.pattern})", re.I)
ENTRY_COLS = ["corpus_id", "unique_id", "source_dataset", "source_record_id", "source_row_number", "source_field",
              "question_slug", "language", "cluster_id", "state", "tier", "dup_group_id", "nearest_cluster_if_noise"]
GROUP_COLS = ["cluster_id", "tier", "n_entries", "n_unique_texts", "stability_jaccard_mean", "short_label", "flags"]
EXAMPLE_COLS = ["cluster_id", "rank", "corpus_id", "quote", "full_text", "text_sha256"]
DUP_COLS = ["unique_id", "dup_group_id", "group_entries", "group_unique_texts"]
INTERPRETATION = [
    "Exploratory prioritization: the score orders concern groups for human inspection. It is not a validated measure of "
    "public importance, urgency, policy impact or representativeness, and it was not fitted to any human judgment (none exists).",
    "Counts are of entries (raw occurrences), unique texts, independent texts (near-duplicate groups) and source records. "
    "None of them is a count of respondents; repeated or templated comments are not independent people.",
    "Concern groups are unsupervised clusters, not ground-truth labels; their descriptions are extractive and not human-validated.",
    "Corpus: Canadian federal Open Government consultations 2016-18. Nothing here transfers to Indian MCA or other consultations.",
]


class InputError(ValueError):
    pass


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(text)


def write_json(path: Path, obj: Any) -> None:
    write_new(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    write_new(path, df.to_csv(index=False, lineterminator="\n"))


# ── inputs ────────────────────────────────────────────────────────────────────

def _need(df: pd.DataFrame, cols: list[str], name: str, checks: list[dict]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise InputError(f"{name}: missing columns {missing}")
    checks.append({"check": f"{name} schema", "status": "PASSED", "detail": f"{len(cols)} required columns present, {len(df):,} rows"})


def load_inputs(cd: Path = CD, run_v1: Path = RUN_V1) -> dict[str, Any]:
    checks: list[dict] = []
    man_p = cd / "run_manifest.json"
    if not man_p.is_file():
        raise InputError(f"concern discovery run not found: {man_p}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    sel = man["selected_configuration"]
    used = ["selected/entry_assignments.csv", "selected/concern_groups.csv",
            f"configurations/{sel}/representative_examples.csv", "duplicates/near_duplicate_groups.csv"]
    for rel in used:
        p = cd / rel
        if not p.is_file() or sha256_file(p) != man["outputs_sha256"].get(rel):
            raise InputError(f"{rel} is missing or differs from the concern-discovery run manifest")
    checks.append({"check": "concern-discovery inputs match run_manifest.json", "status": "PASSED",
                   "detail": f"{len(used)} files; selected configuration {sel}"})
    cm = json.loads((run_v1 / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8"))
    cpath = run_v1 / "corpus" / "corpus_entries.csv"
    if sha256_file(cpath) != cm["corpus_entries_sha256"]:
        raise InputError("corpus_entries.csv differs from the run_v1 corpus manifest")
    checks.append({"check": "original corpus matches run_v1 manifest", "status": "PASSED", "detail": f"{cm['entries']:,} entries"})
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    entries, groups = rd(cd / used[0]), rd(cd / used[1])
    examples, dups, corpus = rd(cd / used[2]), rd(cd / used[3]), rd(cpath)
    _need(entries, ENTRY_COLS, "entry_assignments", checks)
    _need(groups, GROUP_COLS, "concern_groups", checks)
    _need(examples, EXAMPLE_COLS, "representative_examples", checks)
    _need(dups, DUP_COLS, "near_duplicate_groups", checks)
    for c in ("corpus_id", "cluster_id", "unique_id", "dup_group_id", "nearest_cluster_if_noise"):
        entries[c] = pd.to_numeric(entries[c], errors="raise").astype(int)
    groups["cluster_id"] = groups["cluster_id"].astype(int)
    groups["n_entries"] = groups["n_entries"].astype(int)
    groups["stability_jaccard_mean"] = pd.to_numeric(groups["stability_jaccard_mean"], errors="coerce")
    examples["cluster_id"] = examples["cluster_id"].astype(int)
    examples["corpus_id"] = examples["corpus_id"].astype(int)
    examples["rank"] = examples["rank"].astype(int)
    if len(entries) != cm["entries"] or entries["corpus_id"].duplicated().any():
        raise InputError("entry_assignments does not cover every corpus entry exactly once")
    if set(entries["corpus_id"]) != set(corpus["corpus_id"].astype(int)):
        raise InputError("entry_assignments and corpus_entries cover different corpus ids")
    sizes = entries[entries["state"] == "clustered"].groupby("cluster_id").size()
    if sizes.to_dict() != dict(zip(groups["cluster_id"], groups["n_entries"])):
        raise InputError("cluster sizes in entry_assignments disagree with concern_groups")
    if groups["stability_jaccard_mean"].isna().any():
        raise InputError("concern_groups has clusters without a stability value")
    bad_state = sorted(set(entries["state"]) - {"clustered", "noise", "excluded_content_free_after_stripping"})
    if bad_state:
        raise InputError(f"unknown entry states {bad_state}")
    checks.append({"check": "assignment consistency", "status": "PASSED",
                   "detail": f"{len(groups)} clusters; sizes agree; states {entries['state'].value_counts().to_dict()}"})
    corpus_text = dict(zip(corpus["corpus_id"].astype(int), corpus["original_text"]))
    src = corpus.set_index(corpus["corpus_id"].astype(int))
    for c in ("source_dataset", "source_record_id", "question_slug", "language"):
        if not (entries.set_index("corpus_id")[c] == src.loc[entries["corpus_id"], c].values).all():
            raise InputError(f"entry_assignments.{c} disagrees with corpus_entries")
    checks.append({"check": "source ids and metadata agree with the original corpus", "status": "PASSED", "detail": "dataset, record id, field, language"})
    return {"entries": entries, "groups": groups, "examples": examples, "dups": dups, "corpus_text": corpus_text,
            "selected": sel, "checks": checks,
            "inputs_sha256": {**{f"concern_discovery_v1/{r}": man["outputs_sha256"][r] for r in used},
                              "run_v1/corpus/corpus_entries.csv": cm["corpus_entries_sha256"]}}


def load_weights(path: Path | None) -> tuple[dict[str, float], dict[str, float]]:
    if path is None:
        return dict(pr.DEFAULT_WEIGHTS), dict(pr.DEFAULT_TIER_QUALITY)
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(cfg.get("weights", pr.DEFAULT_WEIGHTS)), dict(cfg.get("tier_quality", pr.DEFAULT_TIER_QUALITY))


# ── run ───────────────────────────────────────────────────────────────────────

def stage_run(out: Path = OUT, weights_path: Path | None = None, cd: Path = CD, run_v1: Path = RUN_V1) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already exists: {out}")
    t0 = time.perf_counter()
    d = load_inputs(cd, run_v1)
    weights, tier_q = load_weights(weights_path)
    norm_w = pr.validate_config(weights, tier_q)
    entries = d["entries"]
    ex_ok, ex_problems = pr.verify_evidence(d["examples"], entries, d["corpus_text"])
    if ex_problems:
        raise InputError("evidence verification failed: " + "; ".join(ex_problems[:5]))
    meta = entries.set_index("corpus_id")
    ev = ex_ok.copy()
    for c in ("source_dataset", "source_record_id", "source_row_number", "source_field", "question_slug", "language"):
        ev[c] = ev["corpus_id"].map(meta[c])
    ev["text_sha256_verified"] = [sha256_text(d["corpus_text"][c]) == h for c, h in zip(ev["corpus_id"], ev["text_sha256"])]
    if not ev["text_sha256_verified"].all():
        raise InputError("an example's text hash does not match the original corpus text")
    sig = pr.compute_signals(entries, d["groups"], ev.groupby("cluster_id").size().to_dict())
    ranked = pr.score(sig, weights, tier_q)
    ranked["explanation"] = pr.explain(ranked)
    evidence = ev[ev["cluster_id"].isin(ranked["cluster_id"])].merge(ranked[["cluster_id", "rank_multi"]], on="cluster_id")
    evidence = evidence.sort_values(["rank_multi", "rank"])[
        ["rank_multi", "cluster_id", "rank", "corpus_id", "source_dataset", "source_record_id", "source_row_number",
         "source_field", "question_slug", "language", "quote", "full_text", "text_sha256"]].rename(columns={"rank": "example_rank"})
    problems = pr.validate_outputs(ranked, evidence, norm_w)
    if problems:
        raise InputError("output validation failed: " + "; ".join(problems))
    comparison = pr.compare_baseline(ranked)
    sens = pr.sensitivity(sig, weights, tier_q)
    cand, cand_ev = pr.collective_demand_candidates(entries, d["dups"], d["corpus_text"], NON_SUBSTANTIVE)
    if len(cand_ev):
        bad = [c for c, t in zip(cand_ev["corpus_id"], cand_ev["text"]) if d["corpus_text"][int(c)] != t]
        if bad:
            raise InputError(f"collective-demand evidence not verbatim: {bad[:3]}")
    totals = {"entries": int(len(entries)), "unique_texts": int(entries["unique_id"].nunique()),
              "independent_texts": int(entries["dup_group_id"].nunique()),
              "source_records": int((entries["source_dataset"] + "#" + entries["source_record_id"]).nunique()),
              "states": entries["state"].value_counts().to_dict(),
              "clusters": int(len(sig)), "ranked_clusters": int(sig["ranked"].sum()),
              "ranked_entries": int(sig.loc[sig["ranked"], "entries"].sum()),
              "ranked_independent_texts": int(sig.loc[sig["ranked"], "independent_texts"].sum())}
    ranked_cols = [*pr.RANKING_COLUMNS, "explanation"]
    write_csv(out / "concern_signals.csv", sig)
    write_csv(out / "ranking.csv", ranked[ranked_cols])
    write_csv(out / "ranking_evidence.csv", evidence)
    write_csv(out / "not_ranked.csv", sig.loc[~sig["ranked"], ["cluster_id", "tier", "short_label", "entries",
                                                                 "independent_texts", "not_ranked_reason"]])
    write_csv(out / "collective_demand_candidates.csv", cand)
    write_csv(out / "collective_demand_evidence.csv", cand_ev)
    write_json(out / "baseline_comparison.json", comparison)
    write_json(out / "sensitivity.json", sens)
    write_json(out / "config.json", {
        "weights_as_given": weights, "weights_normalised": norm_w, "tier_quality": tier_q,
        "weights_source": str(weights_path) if weights_path else "defaults (models/concern_discovery/prioritization.py)",
        "signals": list(pr.SIGNALS), "evidence_target": pr.EVIDENCE_TARGET, "ranked_tiers": list(pr.RANKED_TIERS),
        "frequency_baseline": "rank by raw entries (ties: unique texts, cluster id)",
        "collective_demand_min_entries": 5, "selected_configuration": d["selected"],
        "interpretation": INTERPRETATION, "human_validated": False, "fitted_to_human_judgments": False})
    write_json(out / "input_checks.json", d["checks"])
    write_json(out / "totals.json", totals)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "human_validated": False,
                "inputs_sha256": d["inputs_sha256"], "output_validation": "PASSED",
                "evidence_verification": f"PASSED ({len(evidence)} ranked-concern examples, {len(cand_ev)} collective-demand entries verbatim)",
                "outputs_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in files},
                "runtime_s": round(time.perf_counter() - t0, 2)}
    write_json(out / "run_manifest.json", manifest)
    return {"ranked": int(len(ranked)), "comparison": comparison, "min_kendall_tau": sens["min_kendall_tau"],
            "collective_included": int((cand["decision"] == "included").sum()) if len(cand) else 0,
            "runtime_s": manifest["runtime_s"]}


# ── report ────────────────────────────────────────────────────────────────────

def _q(text: str, n: int = 160) -> str:
    """Verbatim span in curly quotes; an ellipsis OUTSIDE the quotes marks truncation."""
    span = quotable_span(text, n)
    more = "…" if len(span) < len(text.strip()) else ""
    return f"“{span}”{more}".replace("|", "/").replace("\n", " ")


def stage_report(out: Path = OUT, report_path: Path = REPORT) -> str:
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    rd = lambda n: pd.read_csv(out / n, keep_default_na=False)  # noqa: E731
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    comp = json.loads((out / "baseline_comparison.json").read_text(encoding="utf-8"))
    sens = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))
    tot = json.loads((out / "totals.json").read_text(encoding="utf-8"))
    checks = json.loads((out / "input_checks.json").read_text(encoding="utf-8"))
    ranked, ev, sig = rd("ranking.csv"), rd("ranking_evidence.csv"), rd("concern_signals.csv")
    notr, cand, cand_ev = rd("not_ranked.csv"), rd("collective_demand_candidates.csv"), rd("collective_demand_evidence.csv")
    L: list[str] = []
    A = L.append
    A("# PolicyLens exploratory concern prioritization (v1)\n")
    A(f"**Status:** computed and automatically verified ({man['output_validation']} output validation; evidence verification "
      f"{man['evidence_verification']}). **Not human-validated, and not a measure of public importance.** Generated {man['created_utc']} "
      "from `data/full_corpus/concern_prioritization_v1/`.\n")
    A("## 1. Interpretation rules\n")
    for x in INTERPRETATION:
        A(f"- {x}")
    A("\n## 2. Inputs and integrity\n")
    A("Inputs are the concern-discovery outputs and the original corpus, read-only and hash-checked. No embedding, clustering or model was run.\n")
    A("| Check | Status | Detail |\n|---|---|---|")
    for c in checks:
        A(f"| {c['check']} | {c['status']} | {c['detail']} |")
    A("\n## 3. Counting units\n")
    A("| Unit | What it counts | Corpus total |\n|---|---|---:|")
    A(f"| Entries | raw occurrences: non-blank designed free-text responses | {tot['entries']:,} |")
    A(f"| Unique texts | normalised distinct texts | {tot['unique_texts']:,} |")
    A(f"| Independent texts | near-duplicate groups; a repeated or templated text counts once | {tot['independent_texts']:,} |")
    A(f"| Source records | distinct dataset + record id; a record is a submission, row or card, not a person | {tot['source_records']:,} |")
    A(f"\nEntry states: {tot['states']}. Noise and excluded entries are never ranked; a group's `nearby_noise_entries` "
      "(noise whose nearest group it is) is reported as context only, because their membership is uncertain.\n")
    A("## 4. Method\n")
    A("A group is ranked if its tier is candidate, needs_review or campaign_or_template and at least one supporting comment "
      "verifies verbatim against the original corpus. Each signal is scaled to [0, 1] across the ranked groups; the score is "
      "Σ weight × scaled signal, and each term is stored as a separate contribution column in `ranking.csv`.\n")
    desc = {"independent_volume": "log(1 + independent texts) / log(1 + largest ranked value)",
            "independence": "1 − duplicate influence; duplicate influence = 1 − independent texts / entries",
            "dataset_diversity": "normalised Shannon entropy of independent texts over the 3 source datasets",
            "field_diversity": "normalised Shannon entropy over consultation fields (dataset + question)",
            "reliability": "mean best-match Jaccard of the group across UMAP seeds 42/43/44",
            "tier_quality": "candidate 1.0, needs_review 0.5, campaign_or_template 0.5",
            "evidence": f"verified supporting comments, capped at {cfg['evidence_target']}, divided by {cfg['evidence_target']}"}
    A("| Signal | Definition | Weight |\n|---|---|---:|")
    for k in pr.SIGNALS:
        A(f"| `{k}` | {desc[k]} | {cfg['weights_normalised'][k]:.2f} |")
    A(f"\nWeights source: {cfg['weights_source']}. They are documented judgement calls set before inspecting the ranking and "
      "are not fitted to any outcome; volume carries the largest weight but is measured in independent texts so that "
      "repetition cannot dominate. Baseline: rank by raw entries (ties: unique texts).\n")
    sc = ["`" + k + "`" for k in pr.SIGNALS]
    A("## 5. Multi-signal ranking (top 15)\n")
    A("| Rank | Freq. rank | Cluster | Tier | Label | Entries | Unique | Independent | Records | Score | " + " | ".join(sc) + " |")
    A("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|" + "---:|" * len(sc))
    for r in ranked.head(15).itertuples():
        A(f"| {r.rank_multi} | {r.rank_frequency} | {r.cluster_id} | {r.tier} | {r.short_label or '—'} | {r.entries} | {r.unique_texts} | "
          f"{r.independent_texts} | {r.source_records} | {r.score:.3f} | " + " | ".join(f"{getattr(r, 'contrib_' + k):.3f}" for k in pr.SIGNALS) + " |")
    A("\nSignal columns show each contribution (weight × scaled value); they sum to the score. All ranked groups: `ranking.csv`.\n")
    A("## 6. Frequency-only baseline (top 15)\n")
    A("| Freq. rank | Multi rank | Cluster | Label | Entries | Independent | Duplicate influence | Stability |\n|---:|---:|---:|---|---:|---:|---:|---:|")
    s_idx = sig.set_index("cluster_id")
    for r in ranked.sort_values("rank_frequency").head(15).itertuples():
        A(f"| {r.rank_frequency} | {r.rank_multi} | {r.cluster_id} | {r.short_label or '—'} | {r.entries} | {r.independent_texts} | "
          f"{s_idx.loc[r.cluster_id, 'duplicate_influence']:.2f} | {s_idx.loc[r.cluster_id, 'stability']:.2f} |")
    A("\n## 7. How the rankings differ and why\n")
    mf = comp["multi_vs_frequency"]
    A(f"- {comp['n_ranked']} groups ranked. Rank agreement with the frequency baseline: Spearman {mf['spearman']}, Kendall τ {mf['kendall_tau']}. "
      f"Mean absolute rank change {comp['mean_abs_rank_change']}; largest {comp['max_abs_rank_change']}.")
    for k in (5, 10, 20):
        o = comp[f"top{k}_overlap"]
        A(f"- Top {k}: {o['shared']} groups shared; entered {o['entered'] or 'none'}; left {o['left'] or 'none'}.")
    fu = comp["frequency_vs_unique"]
    rs = sig[sig["ranked"].astype(str) == "True"]
    top_dup = rs.loc[rs["duplicate_influence"].idxmax()]
    vol = ranked.set_index("cluster_id")["value_independent_volume"]
    A(f"- Raw entries versus unique texts: Spearman {fu['spearman']}. Among ranked groups the median duplicate influence is "
      f"{rs['duplicate_influence'].median():.3f} (largest {top_dup['duplicate_influence']:.2f}, cluster {int(top_dup['cluster_id'])}), so "
      "removing repeats changes few positions here; most differences come from reliability, diversity, tier and evidence.")
    A(f"- Volume is log-scaled: the largest ranked group gets a volume value of {vol.max():.2f} and the smallest {vol.min():.2f}, "
      f"although they differ {ranked['independent_texts'].max() / ranked['independent_texts'].min():.0f}-fold in independent texts. "
      "This deliberately limits how far size alone decides the order, which is what allows small but stable and diverse groups to move up.\n")
    A("Largest moves (explanations are computed from each group's contributions relative to the ranked-group mean):\n")
    movers = pd.concat([ranked.nlargest(4, "rank_change_vs_frequency"), ranked.nsmallest(4, "rank_change_vs_frequency")])
    for r in movers.itertuples():
        st = s_idx.loc[r.cluster_id]
        A(f"- **Cluster {r.cluster_id} ({r.short_label or 'no label'}, {r.tier})**: frequency rank {r.rank_frequency} → multi rank {r.rank_multi}. "
          f"{r.explanation}. Raw values: {r.entries} entries, {r.independent_texts} independent texts, stability {st['stability']:.2f}, "
          f"{st['fields_present']} fields, {st['datasets_present']} datasets, duplicate influence {st['duplicate_influence']:.2f}.")
    A("\n## 8. Ranking examples with source evidence\n")
    A("Quotes are verbatim spans of the cited original comment (verified against `run_v1/corpus/corpus_entries.csv`, with SHA-256). "
      "The group label is a list of frequent terms, not a validated concern statement.\n")
    for r in ranked.head(6).itertuples():
        A(f"**{r.rank_multi}. Cluster {r.cluster_id} — {r.short_label or 'no label'}** ({r.tier}; {r.entries} entries, "
          f"{r.independent_texts} independent texts, {r.source_records} source records; score {r.score:.3f})\n")
        for e in ev[ev["cluster_id"] == r.cluster_id].head(3).itertuples():
            A(f"- [corpus_id {e.corpus_id} · {e.source_dataset} · record {e.source_record_id} · {e.question_slug} · {e.language}] {_q(e.full_text)}")
        A("")
    const = [k for k in pr.SIGNALS if ranked[f"value_{k}"].nunique() <= 1]
    if const:
        A("Signals that take the same value for every ranked group in this run, and therefore add a constant without changing "
          f"the order: {', '.join('`' + k + '`' for k in const)}. Here they act only through the eligibility rule.\n")
    A("## 9. Robustness to the weights\n")
    A(f"Each weight was dropped, halved and doubled (renormalised): {len(sens['kendall_tau_vs_default'])} variants. Minimum Kendall τ "
      f"against the default ranking: {sens['min_kendall_tau']}. Groups in the default top 10 that stay in the top 10 under every "
      f"variant: {sens['top10_in_every_variant'] or 'none'}.\n")
    worst = sorted(sens["kendall_tau_vs_default"].items(), key=lambda x: x[1])[:4]
    A("Most influential perturbations: " + "; ".join(f"`{k}` τ = {v}" for k, v in worst) + ". "
      "Positions that change under plausible weights should not be read as meaningful differences.\n")
    A("## 10. Collective-demand candidates (repeated text)\n")
    inc = cand[cand["decision"] == "included"] if len(cand) else cand
    A(f"Near-duplicate groups with at least 5 entries were screened: {len(cand)} groups; {len(inc)} state something substantive. "
      "Their entry counts measure how often the same wording was submitted. They may reflect a coordinated campaign and are "
      "**not** counts of people.\n")
    if len(inc):
        A("| Group | Entries | Unique texts | Source records | Datasets | Assigned clusters | Lead text (verbatim) |\n|---:|---:|---:|---:|---|---|---|")
        for r in inc.itertuples():
            A(f"| {r.dup_group_id} | {r.entries} | {r.unique_texts} | {r.source_records} | {r.source_datasets} | {r.assigned_clusters} | "
              f"[corpus_id {r.lead_corpus_id}] {_q(r.lead_text, 140)} |")
        A(f"\nEvery member entry, with source ids and verbatim text, is in `collective_demand_evidence.csv` ({len(cand_ev)} rows).\n")
    exc = cand[cand["decision"] != "included"] if len(cand) else cand
    if len(exc):
        A("Screened out: " + "; ".join(f"group {r.dup_group_id} ({r.entries} entries) — {r.decision.replace('excluded: ', '')}" for r in exc.itertuples()) + ".\n")
    A("## 11. Groups not ranked\n")
    A(f"{len(notr)} of {tot['clusters']} groups are not ranked: " + "; ".join(
        f"{t} ({len(g)} groups, {int(g['entries'].sum()):,} entries)" for t, g in notr.groupby("tier")) + ". Reasons per group are in `not_ranked.csv`.\n")
    A("## 12. Limitations\n")
    for x in [
        "No human relevance or importance judgments exist; the ranking is not validated and was not tuned to any outcome. Weights are judgement calls; section 9 shows how much positions depend on them.",
        "Counts are entries, texts and records, never people. A questionnaire submission can add up to three entries, and duplicates across instruments cannot be traced to a person.",
        "Volume reflects which consultation channels and questions collected text, not how widely a concern is held in the population; the instruments differ in size and design.",
        "Group boundaries come from one embedding model and one clustering configuration; agreement between encoders is low (concern-discovery report, section 5), so a different configuration would give different groups and ranks.",
        "Reliability uses only UMAP-seed stability; tiers and flags are heuristic diagnostics, not quality judgments.",
        "Diversity is measured over datasets and question fields only; no stakeholder identity, demographic or urgency information is used or inferred.",
        "Broad, generic groups can rank high on volume and diversity; their labels are frequent terms and should be read with the evidence comments.",
        "Collective-demand screening uses keyword rules for no-opinion and promotional wording, and the 5-entry threshold is a convention.",
    ]:
        A(f"- {x}")
    A("\n## 13. Reproduction\n")
    A("```powershell\npython scripts\\run_concern_prioritization.py run                       # default weights\n"
      "python scripts\\run_concern_prioritization.py run --weights my_weights.json --out data\\full_corpus\\concern_prioritization_alt\n"
      "python scripts\\run_concern_prioritization.py report\npython -m pytest tests\\test_concern_prioritization.py -q -p no:cacheprovider\n```\n")
    A(f"Outputs are write-once; `run_manifest.json` records SHA-256 of every input and output. Runtime {man['runtime_s']} s.\n")
    text = "\n".join(L) + "\n"
    write_new(report_path, text)
    return text


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description="PolicyLens exploratory concern prioritization")
    p.add_argument("stage", choices=["run", "report", "all"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--weights", type=Path)
    p.add_argument("--report", type=Path, default=REPORT)
    a = p.parse_args(argv)
    try:
        if a.stage in ("run", "all"):
            print(json.dumps(stage_run(a.out, a.weights), indent=2, default=str))
        if a.stage in ("report", "all"):
            stage_report(a.out, a.report)
            print(f"written {a.report}")
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    except (InputError, pr.PrioritizationInputError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
