"""
app/dashboard.py
================
PolicyLens research-results dashboard (Streamlit). Read-only browser over PRECOMPUTED, manifest-verified results.

    streamlit run app/dashboard.py

All data comes from app.results_access.ResultsStore; this file never opens experiment files itself. Set the
environment variable POLICYLENS_ROOT to point the store at another project tree (used by the tests).

The dashboard does not analyse new comments, serves no stance classifier, and shows exploratory, unsupervised
results that are not human-validated. The corpus is Canadian (2016-18) and supports no conclusion about Indian citizens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.results_access import ArtifactPaths, ArtifactUnavailable, ResultsStore  # noqa: E402

PAGES = ("Overview", "Concern groups", "Exploratory prioritization", "Keywords by language", "Evaluation & limitations")
BANNER = ("**Exploratory research results, not human-validated.** Groups, scores and keywords come from unsupervised "
          "analysis; scores are not a measure of public importance, and counts are comments, not people. This dashboard "
          "shows precomputed results only; it does not analyse new comments and offers no stance classifier.")
SCOPE = ("**Corpus scope:** Canadian federal Open Government consultations, 2016-18 (English and French). These results "
         "cannot support conclusions about Indian citizens or Indian government consultations.")
STATUS_HELP = {"MISSING": "the results or their manifest are absent", "MISMATCH": "files no longer match their recorded checksums",
               "MALFORMED": "a file could not be read or lacks required fields",
               "INCONSISTENT": "evidence does not match the original source records"}


@st.cache_resource(show_spinner=False)
def get_store(root: str | None) -> ResultsStore:
    return ResultsStore(ArtifactPaths(root=Path(root))) if root else ResultsStore()


def unavailable(exc: ArtifactUnavailable) -> None:
    st.error(f"**{exc.component}** is unavailable: **{exc.status}** ({STATUS_HELP.get(exc.status, 'verification failed')}). "
             "Nothing from this component is shown.")
    for p in exc.problems[:5]:
        st.caption(f"- {p}")


def caveats(payload: dict) -> None:
    for c in payload.get("caveats", []):
        if c:
            st.caption(c)
    prov = payload.get("provenance", {})
    if prov:
        st.caption(f"Verified against `{Path(prov['manifest']).name}` (SHA-256 {prov['manifest_sha256'][:12]}…) at {prov['verified_utc']}")


# ── pages ─────────────────────────────────────────────────────────────────────

def page_overview(store: ResultsStore) -> None:
    st.header("Overview")
    rep = store.verification_report()
    rows = [{"component": c, "status": v["status"], "verified (UTC)": v["verified_utc"],
             "manifest": Path(v["manifest"]).name if v["manifest"] else ""} for c, v in rep["components"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if rep["all_verified"]:
        st.success("All components verified against their manifests.")
    for c, v in rep["components"].items():
        if v["status"] != "VERIFIED":
            st.warning(f"**{c}: {v['status']}** ({STATUS_HELP.get(v['status'], '')}). Pages using it show a warning instead of data.")
            for p in v["problems"][:3]:
                st.caption(f"- {p}")
    cols = st.columns(3)
    try:
        g = store.concern_groups()["data"]
        cols[0].metric("Concern groups", len(g))
        cols[1].metric("Candidate-tier groups", sum(x["tier"] == "candidate" for x in g))
    except ArtifactUnavailable:
        cols[0].metric("Concern groups", "unavailable")
    try:
        cols[2].metric("Groups in exploratory ranking", len(store.prioritized_concerns()["data"]))
    except ArtifactUnavailable:
        cols[2].metric("Groups in exploratory ranking", "unavailable")


def page_groups(store: ResultsStore) -> None:
    st.header("Concern groups")
    try:
        payload = store.concern_groups()
    except ArtifactUnavailable as exc:
        return unavailable(exc)
    groups = payload["data"]
    tiers = sorted({g["tier"] for g in groups})
    chosen = st.multiselect("Tier", tiers, default=tiers, key="tier_filter")
    shown = [g for g in groups if g["tier"] in chosen]
    st.dataframe(pd.DataFrame([{"cluster": g["cluster_id"], "tier": g["tier"], "label (frequent terms)": g["short_label"],
                                "entries": g["entries"], "independent texts": g["independent_texts"],
                                "English": g["language_counts"]["English"], "French": g["language_counts"]["French"],
                                "seed stability": g["seed_stability"], "flags": ", ".join(g["flags"])} for g in shown]),
                 hide_index=True, use_container_width=True)
    caveats(payload)
    if not shown:
        return st.info("No groups match the selected tiers.")
    cid = st.selectbox("Inspect group", [g["cluster_id"] for g in shown],
                       format_func=lambda c: f"{c}: {next(g['short_label'] for g in shown if g['cluster_id'] == c) or 'no label'}",
                       key="group_select")
    try:
        d = store.concern_group(cid)["data"]
    except ArtifactUnavailable as exc:
        return unavailable(exc)
    st.subheader(f"Group {d['cluster_id']} ({d['tier']})")
    st.write(d["description"] or d.get("not_summarised_reason") or "No description.")
    st.caption("Flags: " + (", ".join(d["flags"]) or "none"))
    st.markdown("**Verbatim evidence with source IDs**")
    st.dataframe(pd.DataFrame([{"corpus_id": e["corpus_id"], "source dataset": e["source_dataset"],
                                "source record": e["source_record_id"], "question": e["question_slug"],
                                "language": e["language"], "quote": e["quote"]} for e in d["evidence"]]),
                 hide_index=True, use_container_width=True)


def page_prioritization(store: ResultsStore) -> None:
    st.header("Exploratory prioritization")
    st.info("The **exploratory score** orders groups for inspection. It is not a validated measure of public importance; "
            "weights are documented judgement calls and positions change when they change.")
    try:
        payload = store.prioritized_concerns()
        base = store.frequency_baseline()
    except ArtifactUnavailable as exc:
        return unavailable(exc)
    rows = payload["data"]
    signals = list(payload["weights"])
    st.caption("Weights: " + ", ".join(f"{k} {v:.2f}" for k, v in payload["weights"].items()))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Multi-signal exploratory ranking**")
        st.dataframe(pd.DataFrame([{"rank": r["rank_multi_signal"], "cluster": r["cluster_id"], "label": r["short_label"],
                                    "tier": r["tier"], "exploratory score": round(r["exploratory_score"], 3),
                                    "frequency rank": r["rank_frequency"],
                                    "change vs frequency": r["rank_change_vs_frequency"]} for r in rows]),
                     hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Raw-frequency baseline**")
        st.dataframe(pd.DataFrame([{"rank": r["rank_frequency"], "cluster": r["cluster_id"], "label": r["short_label"],
                                    "entries": r["entries"], "independent texts": r["independent_texts"],
                                    "multi-signal rank": r["rank_multi_signal"]} for r in base["data"]]),
                     hide_index=True, use_container_width=True)
    n = st.slider("Groups in contribution chart", 5, max(5, len(rows)), min(15, max(5, len(rows))), key="n_chart")
    chart = pd.DataFrame([{**r["contributions"], "group": f"{r['rank_multi_signal']:02d} · {r['cluster_id']}"} for r in rows[:n]]).set_index("group")
    st.markdown("**Per-signal contributions** (stacked; they sum to the exploratory score)")
    st.bar_chart(chart[signals])
    cid = st.selectbox("Explain one group", [r["cluster_id"] for r in rows], key="rank_select")
    r = next(x for x in rows if x["cluster_id"] == cid)
    st.write(r["explanation"])
    st.dataframe(pd.DataFrame({"scaled value": r["scaled_signals"], "contribution": r["contributions"]}), use_container_width=True)
    try:
        ev = store.ranking_evidence(cid)["data"]
        st.markdown("**Evidence (verbatim, with source IDs)**")
        st.dataframe(pd.DataFrame([{"corpus_id": e["corpus_id"], "source dataset": e["source_dataset"],
                                    "source record": e["source_record_id"], "language": e["language"], "quote": e["quote"]}
                                   for e in ev]), hide_index=True, use_container_width=True)
    except (ArtifactUnavailable, KeyError) as exc:
        st.warning(str(exc))
    if payload.get("not_ranked"):
        with st.expander(f"Groups not ranked ({len(payload['not_ranked'])})"):
            st.dataframe(pd.DataFrame(payload["not_ranked"]), hide_index=True, use_container_width=True)
    caveats(payload)


def page_keywords(store: ResultsStore) -> None:
    st.header("Keywords by language")
    try:
        inv_p = store.subgroup_eligibility()
        kw_p = store.language_keywords()
    except ArtifactUnavailable as exc:
        return unavailable(exc)
    st.caption(f"A (group, language) subgroup needs at least {inv_p.get('min_independent_texts')} independent texts. "
               "Keywords are frequent, distinctive wording, not validated concerns.")
    lang = st.radio("Language", ["English", "French"], horizontal=True, key="kw_lang")
    inv = [r for r in inv_p["data"] if r["language"] == lang]
    eligible = [r["cluster_id"] for r in inv if r["eligible"]]
    c1, c2 = st.columns(2)
    c1.metric("Eligible subgroups", len(eligible))
    c2.metric("Excluded subgroups", sum(not r["eligible"] for r in inv))
    if eligible:
        cid = st.selectbox("Group", eligible, key="kw_cluster")
        kws = [k for k in kw_p["data"] if k["cluster_id"] == cid and k["language"] == lang]
        st.dataframe(pd.DataFrame([{"rank": k["rank"], "term": k["term"], "c-TF-IDF weight": round(k["ctfidf_weight"], 4),
                                    "share of texts": k["support_share_unique_texts"],
                                    "independent texts": k["support_independent_texts"],
                                    "single-text term": k["single_text_term"],
                                    "spans two texts": k["spans_text_boundary"]} for k in kws]),
                     hide_index=True, use_container_width=True)
    else:
        st.info(f"No eligible {lang} subgroups.")
    with st.expander("Subgroup eligibility and exclusion reasons"):
        st.dataframe(pd.DataFrame([{"cluster": r["cluster_id"], "independent texts": r["independent_texts"],
                                    "eligible": r["eligible"], "exclusion reason": r["exclusion_reason"] or ""} for r in inv]),
                     hide_index=True, use_container_width=True)
    caveats(kw_p)


def page_evaluation(store: ResultsStore) -> None:
    st.header("Evaluation & limitations")
    st.markdown("**Provenance**")
    rep = store.verification_report()
    st.dataframe(pd.DataFrame([{"component": c, "status": v["status"], "manifest": v["manifest"],
                                "manifest SHA-256": v["manifest_sha256"]} for c, v in rep["components"].items()]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Research evaluation** (read-only; re-hashes protected files, so it takes a few seconds)")
    if st.button("Run research evaluation", key="run_eval"):
        st.session_state["evaluation"] = store.evaluation_status()
    ev = st.session_state.get("evaluation")
    if ev:
        st.write(", ".join(f"{k}: {v}" for k, v in sorted(ev["summary"].items())))
        if ev["failed"]:
            st.error(f"{len(ev['failed'])} check(s) FAILED")
        for title, key in (("Blocked (waiting for human evaluation)", "blocked"), ("Not run", "not_run"), ("Failed", "failed")):
            if ev[key]:
                with st.expander(f"{title}: {len(ev[key])}"):
                    st.dataframe(pd.DataFrame(ev[key])[["section", "name", "detail"]], hide_index=True, use_container_width=True)
    st.markdown("**Limitations**")
    lim = store.limitations()["data"]
    st.markdown(f"- {lim['scope']}")
    st.markdown(f"- **Stance:** {lim['stance']}")
    for rule in lim["general"]:
        st.markdown(f"- {rule}")
    for comp in ("concern_discovery", "prioritization", "language_keywords"):
        val = lim.get(comp)
        for x in (val if isinstance(val, list) else [val]):
            if x:
                st.markdown(f"- *{comp}:* {x}")


def main() -> None:
    st.set_page_config(page_title="PolicyLens results", layout="wide")
    st.title("PolicyLens: consultation analysis results")
    st.warning(BANNER)
    st.info(SCOPE)
    store = get_store(os.environ.get("POLICYLENS_ROOT"))
    page = st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.caption("Read-only view of verified, precomputed results.")
    {"Overview": page_overview, "Concern groups": page_groups, "Exploratory prioritization": page_prioritization,
     "Keywords by language": page_keywords, "Evaluation & limitations": page_evaluation}[page](store)


main()
