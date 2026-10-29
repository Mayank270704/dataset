"""
app/results_access.py
=====================
Read-only, manifest-verified access to the PolicyLens research results, for the future dashboard.

Every component is verified with the EXISTING checks of scripts/run_research_evaluation.py before any of its
data is returned. Files are only ever opened for reading.

Components
    corpus              data/full_corpus/run_v1/corpus/corpus_entries.csv (hash in corpus_manifest.json)
    concern_discovery   data/full_corpus/concern_discovery_v1/            (run_manifest.json)
    prioritization      data/full_corpus/concern_prioritization_v1/       (run_manifest.json)
    language_keywords   data/full_corpus/language_keywords_v1/            (run_manifest.json)
    traceability        concern_prioritization_v1/evidence_traceability_manifest.json

Status of a component
    VERIFIED      manifest checks passed and the files parsed with the expected schema
    MISSING       manifest (or required file) absent
    MISMATCH      a recorded checksum or an existing evaluation check failed
    MALFORMED     a file could not be parsed or lacks required columns/keys
    INCONSISTENT  served evidence does not match the original corpus (source ids or verbatim text)
Accessors raise ArtifactUnavailable unless the component is VERIFIED. Verification is cached per store and
repeated automatically when any served file's size or modification time changes.

Every payload is a JSON-ready dict:
    {"component", "kind", "human_validated": False, "scope", "caveats": [...], "provenance": {...}, "data": ...}
Scores are exploratory; nothing here is a validated measure of public importance, and the corpus is Canadian
(2016-18): no conclusion about Indian citizens or Indian consultations follows from it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SCOPE_NOTE = ("Source corpus: Canadian federal Open Government consultations, 2016-18 (English/French). These results "
              "are not evidence about Indian citizens or Indian government consultations.")
EXPLORATORY_NOTE = ("Exploratory, unsupervised results; not human-validated and not a measure of public importance. "
                    "Counts are entries, texts and records, not respondents.")
STATUSES = ("VERIFIED", "MISSING", "MISMATCH", "MALFORMED", "INCONSISTENT")
COMPONENTS = ("corpus", "concern_discovery", "prioritization", "language_keywords", "traceability")

SCHEMAS: dict[str, dict[str, list[str]]] = {
    "concern_discovery": {
        "selected/concern_groups.csv": ["cluster_id", "tier", "n_entries", "n_unique_texts", "distinct_near_dup_groups",
                                        "english", "french", "other_unknown", "stability_jaccard_mean", "cohesion_mean",
                                        "flags", "short_label", "description", "evidence_corpus_ids"],
    },
    "prioritization": {
        "ranking.csv": ["rank_multi", "rank_frequency", "rank_unique", "rank_change_vs_frequency", "cluster_id", "tier",
                        "short_label", "entries", "unique_texts", "independent_texts", "source_records", "score",
                        "explanation"],
        "ranking_evidence.csv": ["rank_multi", "cluster_id", "example_rank", "corpus_id", "source_dataset",
                                 "source_record_id", "source_row_number", "source_field", "question_slug", "language",
                                 "quote", "full_text", "text_sha256"],
        "not_ranked.csv": ["cluster_id", "tier", "short_label", "entries", "independent_texts", "not_ranked_reason"],
    },
    "language_keywords": {
        "subgroup_inventory.csv": ["cluster_id", "language", "unique_texts", "independent_texts", "entries", "eligible",
                                   "exclusion_reason", "tier", "short_label"],
        "language_keywords.csv": ["cluster_id", "language", "rank", "term", "ctfidf_weight", "support_share_unique_texts",
                                  "support_independent_texts", "single_text_term", "spans_text_boundary"],
    },
    "corpus": {"corpus_entries.csv": ["corpus_id", "source_dataset", "source_record_id", "original_text"]},
}
EVIDENCE_FIELDS = ("corpus_id", "source_dataset", "source_record_id", "quote")


class ArtifactUnavailable(RuntimeError):
    """Raised when a component is not VERIFIED; carries the component, its status and the problems found."""

    def __init__(self, component: str, status: str, problems: list[str]):
        super().__init__(f"{component} is {status}: {'; '.join(problems[:3])}")
        self.component, self.status, self.problems = component, status, problems


@dataclass
class ArtifactPaths:
    """Locations of the authoritative artifacts (defaults: the real project). Tests pass temporary trees."""
    root: Path = ROOT
    run_v1: Path | None = None
    concern_discovery: Path | None = None
    prioritization: Path | None = None
    language_keywords: Path | None = None
    traceability_manifest: Path | None = None

    def __post_init__(self) -> None:
        fc = self.root / "data" / "full_corpus"
        self.run_v1 = self.run_v1 or fc / "run_v1"
        self.concern_discovery = self.concern_discovery or fc / "concern_discovery_v1"
        self.prioritization = self.prioritization or fc / "concern_prioritization_v1"
        self.language_keywords = self.language_keywords or fc / "language_keywords_v1"
        self.traceability_manifest = self.traceability_manifest or self.prioritization / "evidence_traceability_manifest.json"


@dataclass
class Verification:
    component: str
    status: str
    problems: list[str] = field(default_factory=list)
    checks: list[dict[str, str]] = field(default_factory=list)
    manifest: str = ""
    manifest_sha256: str = ""
    verified_utc: str = ""
    snapshot: dict[str, tuple[int, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"component": self.component, "status": self.status, "problems": self.problems, "checks": self.checks,
                "manifest": self.manifest, "manifest_sha256": self.manifest_sha256, "verified_utc": self.verified_utc}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    """Read a CSV as text rows; raise ValueError if unreadable or a required column is absent."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        return list(reader)


def _to_int(v: str) -> int:
    return int(float(v))


class ResultsStore:
    """Read-only, verified access to PolicyLens results. See the module docstring for the contract."""

    def __init__(self, paths: ArtifactPaths | None = None):
        self.paths = paths or ArtifactPaths()
        self._ver: dict[str, Verification] = {}
        self._cache: dict[str, Any] = {}

    # ── verification ──────────────────────────────────────────────────────────

    def _files(self, component: str) -> list[Path]:
        p = self.paths
        if component == "corpus":
            return [p.run_v1 / "corpus" / "corpus_manifest.json", p.run_v1 / "corpus" / "corpus_entries.csv"]
        if component == "concern_discovery":
            sel = self._selected_configuration()
            return [p.concern_discovery / "run_manifest.json", p.concern_discovery / "selected" / "concern_groups.csv",
                    p.concern_discovery / "configurations" / sel / "concern_summaries.json"]
        if component == "prioritization":
            return [p.prioritization / "run_manifest.json", p.prioritization / "config.json",
                    *(p.prioritization / n for n in SCHEMAS["prioritization"])]
        if component == "language_keywords":
            return [p.language_keywords / "run_manifest.json", p.language_keywords / "config.json",
                    *(p.language_keywords / n for n in SCHEMAS["language_keywords"])]
        if component == "traceability":
            return [p.traceability_manifest, p.prioritization / "evidence_traceability_ledger.csv"]
        raise KeyError(f"unknown component {component!r}; choose from {COMPONENTS}")

    def _selected_configuration(self) -> str:
        try:
            return json.loads((self.paths.concern_discovery / "run_manifest.json").read_text(encoding="utf-8"))["selected_configuration"]
        except (OSError, ValueError, KeyError):
            return "__unknown__"

    @staticmethod
    def _stat(files: list[Path]) -> dict[str, tuple[int, int]]:
        out = {}
        for f in files:
            try:
                s = f.stat()
                out[str(f)] = (s.st_size, s.st_mtime_ns)
            except OSError:
                out[str(f)] = (-1, -1)
        return out

    def _existing_checks(self, component: str) -> list[dict[str, str]]:
        import run_research_evaluation as ev
        p = self.paths
        if component == "concern_discovery":
            return [c for c in ev.check_concern_discovery(p.concern_discovery)
                    if c["name"] in ("concern discovery outputs match run_manifest.json", "summary grounding verification")]
        if component == "prioritization":
            return ev.check_prioritization(p.prioritization)
        if component == "language_keywords":
            return ev.check_language_keywords(p.language_keywords, p.language_keywords.parent)
        if component == "traceability":
            return ev.check_evidence_traceability(p.traceability_manifest, p.root)
        return []

    def verify(self, component: str, force: bool = False) -> dict[str, Any]:
        """Verify one component (cached; repeated if forced or if a served file changed). Never raises for bad data."""
        files = self._files(component)
        cached = self._ver.get(component)
        if cached and not force and cached.snapshot == self._stat(files):
            return cached.as_dict()
        self._cache = {k: v for k, v in self._cache.items() if not k.startswith(component + ":")}
        v = Verification(component, "VERIFIED", verified_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        manifest = files[0]
        v.manifest = str(manifest)
        if not manifest.is_file():
            v.status, v.problems = "MISSING", [f"manifest not found: {manifest}"]
        else:
            v.manifest_sha256 = _sha(manifest)
            try:
                if component == "corpus":
                    cm = json.loads(manifest.read_text(encoding="utf-8"))
                    ce = files[1]
                    if not ce.is_file():
                        v.status, v.problems = "MISSING", ["corpus_entries.csv not found"]
                    elif _sha(ce) != cm.get("corpus_entries_sha256"):
                        v.status, v.problems = "MISMATCH", ["corpus_entries.csv differs from corpus_manifest.json"]
                else:
                    v.checks = self._existing_checks(component)
                    if any(c["status"] == "NOT RUN" for c in v.checks):
                        v.status = "MISSING"
                    elif any(c["status"] == "FAILED" for c in v.checks):
                        v.status = "MISMATCH"
                    v.problems = [f"{c['name']}: {c['detail']}" for c in v.checks if c["status"] in ("FAILED", "NOT RUN")]
                if v.status == "VERIFIED":
                    self._load(component)                          # schema validation
                    if component in ("concern_discovery", "prioritization"):
                        bad = self._evidence_problems(component)
                        if bad:
                            v.status, v.problems = "INCONSISTENT", bad
            except ArtifactUnavailable as exc:
                v.status, v.problems = exc.status, exc.problems
            except (OSError, ValueError, KeyError, TypeError) as exc:
                v.status, v.problems = "MALFORMED", [str(exc)]
        if v.status != "VERIFIED":
            self._cache = {k: val for k, val in self._cache.items() if not k.startswith(component + ":")}
        v.snapshot = self._stat(files)
        self._ver[component] = v
        return v.as_dict()

    def verification_report(self) -> dict[str, Any]:
        """Status of every component, without raising. The dashboard should call this first."""
        comps = {c: self.verify(c) for c in COMPONENTS}
        return {"all_verified": all(v["status"] == "VERIFIED" for v in comps.values()),
                "components": comps, "scope": SCOPE_NOTE, "human_validated": False}

    def _require(self, component: str) -> Verification:
        self.verify(component)
        v = self._ver[component]
        if v.status != "VERIFIED":
            raise ArtifactUnavailable(component, v.status, v.problems)
        return v

    # ── loading (schema-checked, cached per verification) ────────────────────

    def _load(self, component: str) -> dict[str, Any]:
        key = component + ":data"
        if key in self._cache:
            return self._cache[key]
        p = self.paths
        if component == "corpus":
            rows = _read_csv(p.run_v1 / "corpus" / "corpus_entries.csv", SCHEMAS["corpus"]["corpus_entries.csv"])
            data = {"by_id": {_to_int(r["corpus_id"]): r for r in rows}}
        elif component == "concern_discovery":
            groups = _read_csv(p.concern_discovery / "selected" / "concern_groups.csv",
                               SCHEMAS["concern_discovery"]["selected/concern_groups.csv"])
            sel = self._selected_configuration()
            summ = json.loads((p.concern_discovery / "configurations" / sel / "concern_summaries.json").read_text(encoding="utf-8"))
            if "clusters" not in summ:
                raise ValueError("concern_summaries.json: missing 'clusters'")
            by_cluster = {}
            for c in summ["clusters"]:
                for k in ("cluster_id", "tier", "evidence"):
                    if k not in c:
                        raise ValueError(f"concern_summaries.json: cluster entry missing '{k}'")
                for e in c["evidence"]:
                    missing = [f for f in EVIDENCE_FIELDS if f not in e]
                    if missing:
                        raise ValueError(f"concern_summaries.json: evidence missing {missing}")
                by_cluster[int(c["cluster_id"])] = c
            ids = {_to_int(g["cluster_id"]) for g in groups}
            if ids != set(by_cluster):
                raise ValueError("concern_groups.csv and concern_summaries.json list different clusters")
            data = {"groups": groups, "summaries": by_cluster, "selected_configuration": sel,
                    "summary_method": summ.get("summary_method")}
        elif component == "prioritization":
            cfg = json.loads((p.prioritization / "config.json").read_text(encoding="utf-8"))
            if "weights_normalised" not in cfg or "signals" not in cfg:
                raise ValueError("prioritization config.json: missing weights_normalised or signals")
            req = SCHEMAS["prioritization"]["ranking.csv"] + [f"{a}_{s}" for s in cfg["signals"] for a in ("value", "contrib")]
            data = {"config": cfg, "ranking": _read_csv(p.prioritization / "ranking.csv", req),
                    "evidence": _read_csv(p.prioritization / "ranking_evidence.csv", SCHEMAS["prioritization"]["ranking_evidence.csv"]),
                    "not_ranked": _read_csv(p.prioritization / "not_ranked.csv", SCHEMAS["prioritization"]["not_ranked.csv"])}
        elif component == "language_keywords":
            cfg = json.loads((p.language_keywords / "config.json").read_text(encoding="utf-8"))
            data = {"config": cfg,
                    "inventory": _read_csv(p.language_keywords / "subgroup_inventory.csv", SCHEMAS["language_keywords"]["subgroup_inventory.csv"]),
                    "keywords": _read_csv(p.language_keywords / "language_keywords.csv", SCHEMAS["language_keywords"]["language_keywords.csv"])}
        elif component == "traceability":
            data = {"manifest": json.loads(p.traceability_manifest.read_text(encoding="utf-8"))}
        else:
            raise KeyError(component)
        self._cache[key] = data
        return data

    def _evidence_problems(self, component: str) -> list[str]:
        """Every served evidence row must match the original corpus: source ids, verbatim quote and full text."""
        cv = self.verify("corpus")
        if cv["status"] != "VERIFIED":
            return [f"original corpus unavailable for evidence checks ({cv['status']})"]
        corpus = self._load("corpus")["by_id"]
        d = self._load(component)
        rows = ([dict(e, cluster_id=cid) for cid, s in d["summaries"].items() for e in s["evidence"]]
                if component == "concern_discovery" else d["evidence"])
        problems = []
        for e in rows:
            cid = _to_int(e["corpus_id"])
            src = corpus.get(cid)
            tag = f"cluster {e['cluster_id']} corpus_id {cid}"
            if src is None:
                problems.append(f"{tag}: not in the original corpus")
            elif src["source_dataset"] != str(e["source_dataset"]) or src["source_record_id"] != str(e["source_record_id"]):
                problems.append(f"{tag}: source ids differ from the original corpus")
            elif e["quote"] not in src["original_text"] or e.get("full_text", src["original_text"]) != src["original_text"]:
                problems.append(f"{tag}: quote or text is not verbatim source text")
            elif e.get("text_sha256") and hashlib.sha256(src["original_text"].encode("utf-8")).hexdigest() != e["text_sha256"]:
                problems.append(f"{tag}: text_sha256 differs from the original text")
        return problems

    # ── payloads ──────────────────────────────────────────────────────────────

    def _payload(self, component: str, kind: str, data: Any, caveats: list[str], **extra: Any) -> dict[str, Any]:
        v = self._ver[component]
        return {"component": component, "kind": kind, "human_validated": False, "scope": SCOPE_NOTE,
                "caveats": caveats, "provenance": {"manifest": v.manifest, "manifest_sha256": v.manifest_sha256,
                                                   "verified_utc": v.verified_utc}, "data": data, **extra}

    @staticmethod
    def _group_row(g: dict[str, str]) -> dict[str, Any]:
        return {"cluster_id": _to_int(g["cluster_id"]), "tier": g["tier"], "short_label": g["short_label"] or None,
                "description": g["description"], "flags": [f for f in g["flags"].split(";") if f],
                "entries": _to_int(g["n_entries"]), "unique_texts": _to_int(g["n_unique_texts"]),
                "independent_texts": _to_int(g["distinct_near_dup_groups"]),
                "language_counts": {"English": _to_int(g["english"]), "French": _to_int(g["french"]),
                                    "Other/Unknown": _to_int(g["other_unknown"])},
                "seed_stability": float(g["stability_jaccard_mean"]), "cohesion": float(g["cohesion_mean"]),
                "evidence_corpus_ids": [_to_int(x) for x in g["evidence_corpus_ids"].split(";") if x]}

    def concern_groups(self, tier: str | None = None) -> dict[str, Any]:
        """All concern groups of the selected configuration (largest first), optionally filtered by tier."""
        self._require("concern_discovery")
        d = self._load("concern_discovery")
        rows = [self._group_row(g) for g in d["groups"] if tier is None or g["tier"] == tier]
        return self._payload("concern_discovery", "exploratory", rows,
                             [EXPLORATORY_NOTE, "Tiers and descriptions are automated diagnostics, not human judgments."],
                             selected_configuration=d["selected_configuration"])

    def concern_group(self, cluster_id: int) -> dict[str, Any]:
        """One group with its extractive description and verbatim, source-linked evidence."""
        self._require("concern_discovery")
        d = self._load("concern_discovery")
        g = next((x for x in d["groups"] if _to_int(x["cluster_id"]) == int(cluster_id)), None)
        if g is None:
            raise KeyError(f"cluster {cluster_id} not found")
        s = d["summaries"][int(cluster_id)]
        evidence = [{k: e.get(k) for k in ("rank", "corpus_id", "source_dataset", "source_record_id", "question_slug",
                                          "language", "quote", "full_text")} for e in s["evidence"]]
        row = {**self._group_row(g), "description_body": s.get("description_body"),
               "not_summarised_reason": s.get("not_summarised_reason"), "term_support": s.get("term_support", {}),
               "summary_method": s.get("summary_method"), "evidence": evidence}
        return self._payload("concern_discovery", "exploratory", row,
                             [EXPLORATORY_NOTE, "Quotes are verbatim source text, checked against the original corpus."])

    def _ranking_rows(self, order_key: str, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._require("prioritization")
        d = self._load("prioritization")
        signals = d["config"]["signals"]
        rows = []
        for r in sorted(d["ranking"], key=lambda x: _to_int(x[order_key])):
            rows.append({"cluster_id": _to_int(r["cluster_id"]), "tier": r["tier"], "short_label": r["short_label"] or None,
                         "rank_multi_signal": _to_int(r["rank_multi"]), "rank_frequency": _to_int(r["rank_frequency"]),
                         "rank_unique_texts": _to_int(r["rank_unique"]),
                         "rank_change_vs_frequency": _to_int(r["rank_change_vs_frequency"]),
                         "entries": _to_int(r["entries"]), "unique_texts": _to_int(r["unique_texts"]),
                         "independent_texts": _to_int(r["independent_texts"]), "source_records": _to_int(r["source_records"]),
                         "exploratory_score": float(r["score"]),
                         "contributions": {s: float(r[f"contrib_{s}"]) for s in signals},
                         "scaled_signals": {s: float(r[f"value_{s}"]) for s in signals},
                         "explanation": r["explanation"]})
        return rows[:limit] if limit else rows, d

    def prioritized_concerns(self, limit: int | None = None) -> dict[str, Any]:
        """Exploratory multi-signal ranking with per-signal contributions (sum = exploratory_score)."""
        rows, d = self._ranking_rows("rank_multi", limit)
        return self._payload("prioritization", "exploratory", rows,
                             [EXPLORATORY_NOTE, "The score orders groups for inspection; weights are documented judgement "
                              "calls, not fitted to any outcome, and rank positions are sensitive to them."],
                             weights=d["config"]["weights_normalised"], weights_source=d["config"].get("weights_source"),
                             not_ranked=[{"cluster_id": _to_int(r["cluster_id"]), "tier": r["tier"],
                                          "reason": r["not_ranked_reason"]} for r in d["not_ranked"]])

    def frequency_baseline(self, limit: int | None = None) -> dict[str, Any]:
        """The same groups ordered by raw entry count (the frequency-only baseline)."""
        rows, d = self._ranking_rows("rank_frequency", limit)
        rows = [{k: r[k] for k in ("cluster_id", "tier", "short_label", "rank_frequency", "rank_multi_signal",
                                   "entries", "unique_texts", "independent_texts")} for r in rows]
        return self._payload("prioritization", "baseline", rows,
                             [EXPLORATORY_NOTE, d["config"].get("frequency_baseline", "rank by raw entries")])

    def ranking_evidence(self, cluster_id: int) -> dict[str, Any]:
        """Verbatim evidence rows supporting one ranked group, with source ids."""
        self._require("prioritization")
        rows = [{k: (r[k] if k not in ("corpus_id", "example_rank", "rank_multi") else _to_int(r[k]))
                 for k in SCHEMAS["prioritization"]["ranking_evidence.csv"]}
                for r in self._load("prioritization")["evidence"] if _to_int(r["cluster_id"]) == int(cluster_id)]
        if not rows:
            raise KeyError(f"cluster {cluster_id} is not ranked or has no evidence")
        return self._payload("prioritization", "evidence", rows,
                             ["Quotes are verbatim source text, checked against the original corpus."])

    def subgroup_eligibility(self, cluster_id: int | None = None) -> dict[str, Any]:
        """(cluster, language) subgroups with counts, eligibility and exclusion reasons."""
        self._require("language_keywords")
        d = self._load("language_keywords")
        rows = [{"cluster_id": _to_int(r["cluster_id"]), "language": r["language"], "unique_texts": _to_int(r["unique_texts"]),
                 "independent_texts": _to_int(r["independent_texts"]), "eligible": r["eligible"] == "True",
                 "exclusion_reason": r["exclusion_reason"] or None}
                for r in d["inventory"] if cluster_id is None or _to_int(r["cluster_id"]) == int(cluster_id)]
        return self._payload("language_keywords", "descriptive", rows, [d["config"].get("disclaimer", "")],
                             min_independent_texts=d["config"].get("min_independent_texts"))

    def language_keywords(self, cluster_id: int | None = None, language: str | None = None) -> dict[str, Any]:
        """Ranked c-TF-IDF terms of eligible subgroups only, with support and artefact flags."""
        self._require("language_keywords")
        d = self._load("language_keywords")
        rows = [{"cluster_id": _to_int(r["cluster_id"]), "language": r["language"], "rank": _to_int(r["rank"]),
                 "term": r["term"], "ctfidf_weight": float(r["ctfidf_weight"]),
                 "support_share_unique_texts": float(r["support_share_unique_texts"]),
                 "support_independent_texts": _to_int(r["support_independent_texts"]),
                 "single_text_term": r["single_text_term"] == "True", "spans_text_boundary": r["spans_text_boundary"] == "True"}
                for r in d["keywords"] if (cluster_id is None or _to_int(r["cluster_id"]) == int(cluster_id))
                and (language is None or r["language"] == language)]
        return self._payload("language_keywords", "descriptive", rows,
                             [d["config"].get("disclaimer", ""), "Terms flagged spans_text_boundary occur in no single comment."])

    # ── status and limitations ────────────────────────────────────────────────

    def evaluation_status(self, run: Callable[[], list[dict[str, str]]] | None = None) -> dict[str, Any]:
        """Live, read-only research-evaluation result (several seconds: it re-hashes protected files)."""
        import run_research_evaluation as ev
        items = (run or ev.run_all)()
        counts: dict[str, int] = {}
        for i in items:
            counts[i["status"]] = counts.get(i["status"], 0) + 1
        return {"component": "research_evaluation", "kind": "status", "human_validated": False, "scope": SCOPE_NOTE,
                "caveats": list(ev.INTERPRETATION_RULES), "summary": counts,
                "blocked": [i for i in items if i["status"] == "BLOCKED"],
                "not_run": [i for i in items if i["status"] == "NOT RUN"],
                "failed": [i for i in items if i["status"] == "FAILED"], "data": items}

    def limitations(self) -> dict[str, Any]:
        """Documented limitations and disclaimers, collected from the artifacts themselves where available."""
        import run_research_evaluation as ev
        out: dict[str, Any] = {"scope": SCOPE_NOTE, "general": list(ev.INTERPRETATION_RULES),
                               "stance": ("No stance model trained on human labels exists (human annotation BLOCKED). The "
                                          "AI-assisted working-label baseline measures agreement with unverified AI labels "
                                          "on 13 held-out records and is not human-validated stance accuracy; it is not served "
                                          "by this access layer.")}
        for comp in ("prioritization", "language_keywords"):
            if self.verify(comp)["status"] == "VERIFIED":
                cfg = self._load(comp)["config"]
                out[comp] = cfg.get("interpretation") or cfg.get("disclaimer")
            else:
                out[comp] = f"unavailable ({self._ver[comp].status})"
        out["concern_discovery"] = [EXPLORATORY_NOTE, "Cluster descriptions are extractive and unvalidated; human review is BLOCKED."]
        return {"component": "limitations", "kind": "status", "human_validated": False, "scope": SCOPE_NOTE, "data": out}
