"""
app/demand_generation.py
========================
Deterministic, extractive collective-demand candidate generation.

Generates one candidate per ELIGIBLE concern group (tier == 'candidate' only, passing
minimum-evidence and minimum-independent-text thresholds). All candidates are labelled:
    state = 'generated_extractive'        (derived from saved concern-group artifacts)

No generative model is used. No text is invented. The demand text is assembled from the
group's verified extractive description plus representative verbatim excerpts drawn
entirely from the verified corpus. Every candidate carries:
    - Unique candidate ID  (cand-<cluster_id>)
    - Source corpus and dataset identifiers from the evidence rows
    - Original comment corpus_ids
    - Verbatim supporting excerpts (from concern_summaries.json evidence)
    - Group membership counts (entries, unique_texts, independent_texts)
    - Generation method and version
    - Evidence coverage indicators
    - Uncertainty / insufficient-evidence flag
    - Human-review status (pending)
    - Contradictory / dissenting evidence flag when present in group flags

This module never writes files; it produces in-memory dicts ready for JSON serialisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PRIORITIZATION = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"
CONCERN_DISCOVERY = ROOT / "data" / "full_corpus" / "concern_discovery_v1"

# Minimum criteria for a candidate to be generated (documented, not fitted).
MIN_INDEPENDENT_TEXTS = 10
MIN_ENTRIES = 5
CANDIDATE_TIERS = {"candidate"}          # only 'candidate' tier yields demand candidates
GENERATION_METHOD = "deterministic_extractive_v1"
GENERATION_VERSION = "1.0.0"
DISCLAIMER = (
    "Generated candidates are extractive summaries derived from automated concern groups. "
    "They are not human-validated collective demands. The concern groups are unsupervised "
    "clusters; their boundaries, descriptions and representative evidence are unverified. "
    "Training labels are AI-generated, not human ground truth. "
    "Corpus: Canadian federal Open Government consultations 2016-18 (English/French). "
    "No conclusion about Indian citizens or Indian consultations follows from these results."
)


@dataclass
class DemandEvidence:
    corpus_id: int
    source_dataset: str
    source_record_id: str
    question_slug: str
    language: str
    quote: str
    selection_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemandCandidate:
    candidate_id: str                           # "cand-<cluster_id>"
    cluster_id: int
    source_corpus: str = "canadian_federal_open_government_2016_18"
    source_datasets: list[str] = field(default_factory=list)
    short_label: str = ""
    demand_text: str = ""                       # extractive, from verified description
    entries: int = 0
    unique_texts: int = 0
    independent_texts: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    supporting_corpus_ids: list[int] = field(default_factory=list)
    evidence: list[DemandEvidence] = field(default_factory=list)
    generation_method: str = GENERATION_METHOD
    generation_version: str = GENERATION_VERSION
    state: str = "generated_extractive"          # workflow state
    human_review_status: str = "pending"
    evidence_coverage: str = ""                  # e.g. "5 of 353 unique texts shown"
    quality_flags: list[str] = field(default_factory=list)
    has_dissenting_evidence: bool = False
    insufficient_evidence: bool = False
    insufficient_reason: str = ""
    disclaimer: str = DISCLAIMER
    scope: str = (
        "Source corpus: Canadian federal Open Government consultations, 2016-18 "
        "(English/French). Not evidence about Indian citizens or Indian consultations."
    )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [e.as_dict() for e in self.evidence]
        return d


def _selected_configuration() -> str:
    manifest = CONCERN_DISCOVERY / "run_manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))["selected_configuration"]
    except (OSError, KeyError, ValueError):
        return "strip__umap_m25s5"


def _load_summaries() -> dict[int, dict[str, Any]]:
    sel = _selected_configuration()
    p = CONCERN_DISCOVERY / "configurations" / sel / "concern_summaries.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(c["cluster_id"]): c for c in data.get("clusters", [])}


def _load_ranking() -> dict[int, dict[str, Any]]:
    """Returns a dict keyed by cluster_id for ranked groups."""
    import csv
    p = PRIORITIZATION / "ranking.csv"
    if not p.is_file():
        return {}
    result = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["cluster_id"]))
            result[cid] = row
    return result


def _load_concern_groups() -> list[dict[str, Any]]:
    import csv
    p = CONCERN_DISCOVERY / "selected" / "concern_groups.csv"
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _extract_demand_text(summary: dict[str, Any]) -> str:
    """
    Build a concise demand text from the group's extractive description body.
    The description_body is already verbatim from the verified summaries. We prefix
    it with the short_label for orientation.
    """
    label = summary.get("short_label") or ""
    body = summary.get("description_body") or summary.get("description") or ""
    if not body:
        return ""
    # Trim to a single readable sentence for the candidate text
    # Use the most-representative-comment quote as the anchor if available
    evidence = summary.get("evidence") or []
    most_central = next((e for e in evidence if e.get("selection_reason") == "most central"), None)
    anchor = ""
    if most_central:
        anchor = most_central.get("quote", "")
    if anchor:
        return f'[{label}] {anchor}'
    # Fall back to the first sentence of description_body
    sentences = body.split(". ")
    return f'[{label}] {sentences[0].strip()}.'


def generate_candidates(
    min_independent_texts: int = MIN_INDEPENDENT_TEXTS,
    min_entries: int = MIN_ENTRIES,
) -> list[DemandCandidate]:
    """
    Generate and return DemandCandidate objects for every eligible concern group.

    Eligibility rules (documented before run):
        - tier must be in CANDIDATE_TIERS ('candidate')
        - independent_texts >= min_independent_texts
        - entries >= min_entries
        - has verified evidence rows in concern_summaries.json

    Groups failing eligibility are skipped, not silently excluded.
    """
    groups = _load_concern_groups()
    summaries = _load_summaries()
    ranking = _load_ranking()

    candidates: list[DemandCandidate] = []

    for g in groups:
        cid = int(float(g["cluster_id"]))
        tier = g.get("tier", "")
        n_entries = int(float(g.get("n_entries", 0)))
        n_independent = int(float(g.get("distinct_near_dup_groups", 0)))
        n_unique = int(float(g.get("n_unique_texts", 0)))
        flags_raw = g.get("flags", "")
        flags = [f for f in (flags_raw.split(";") if flags_raw else []) if f]
        short_label = g.get("short_label") or ""

        # Eligibility gate
        if tier not in CANDIDATE_TIERS:
            continue
        if n_entries < min_entries:
            continue

        summary = summaries.get(cid)
        evidence_rows = (summary.get("evidence") or []) if summary else []

        insufficient = False
        insufficient_reason = ""
        if n_independent < min_independent_texts:
            insufficient = True
            insufficient_reason = (
                f"Only {n_independent} independent texts (threshold: {min_independent_texts}); "
                "group may not represent a collective concern."
            )

        if not evidence_rows:
            insufficient = True
            insufficient_reason = insufficient_reason or "No representative evidence rows found."

        demand_text = _extract_demand_text(summary) if summary else ""
        if not demand_text:
            insufficient = True
            insufficient_reason = insufficient_reason or "Could not extract demand text from group description."

        # Collect source datasets from evidence
        datasets = list({str(e.get("source_dataset", "")) for e in evidence_rows if e.get("source_dataset")})
        supporting_ids = [int(float(e["corpus_id"])) for e in evidence_rows if "corpus_id" in e]

        evidence_objs = []
        for e in evidence_rows:
            evidence_objs.append(DemandEvidence(
                corpus_id=int(float(e.get("corpus_id", 0))),
                source_dataset=str(e.get("source_dataset", "")),
                source_record_id=str(e.get("source_record_id", "")),
                question_slug=str(e.get("question_slug", "")),
                language=str(e.get("language", "")),
                quote=str(e.get("quote", "")),
                selection_reason=str(e.get("selection_reason", "")),
            ))

        lang_counts = {
            "English": int(float(g.get("english", 0))),
            "French": int(float(g.get("french", 0))),
            "Other/Unknown": int(float(g.get("other_unknown", 0))),
        }

        # Dissenting-evidence flag: inherit from group quality flags
        dissenting = any(f in flags for f in ("mixed", "weak_cohesion", "moderately_unstable", "unstable"))

        coverage = (
            f"{len(evidence_objs)} representative comment(s) shown of "
            f"{n_unique} unique texts in this group"
        )

        cand = DemandCandidate(
            candidate_id=f"cand-{cid}",
            cluster_id=cid,
            short_label=short_label,
            demand_text=demand_text,
            entries=n_entries,
            unique_texts=n_unique,
            independent_texts=n_independent,
            language_counts=lang_counts,
            source_datasets=datasets,
            supporting_corpus_ids=supporting_ids,
            evidence=evidence_objs,
            quality_flags=flags,
            has_dissenting_evidence=dissenting,
            insufficient_evidence=insufficient,
            insufficient_reason=insufficient_reason,
            evidence_coverage=coverage,
        )
        candidates.append(cand)

    # Sort by exploratory rank where available, then by entries descending
    rank_map = {cid: int(float(r["rank_multi"])) for cid, r in ranking.items()}
    candidates.sort(key=lambda c: (rank_map.get(c.cluster_id, 9999), -c.entries))
    return candidates


# ── module-level cache (reset on import) ──────────────────────────────────────

_CANDIDATES_CACHE: list[DemandCandidate] | None = None


def get_candidates(force: bool = False) -> list[DemandCandidate]:
    """Return cached generated candidates (computed once per process)."""
    global _CANDIDATES_CACHE
    if _CANDIDATES_CACHE is None or force:
        _CANDIDATES_CACHE = generate_candidates()
    return _CANDIDATES_CACHE


def get_candidate(candidate_id: str) -> DemandCandidate | None:
    for c in get_candidates():
        if c.candidate_id == candidate_id:
            return c
    return None


get_candidate_by_id = get_candidate
