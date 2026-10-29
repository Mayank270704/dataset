"""
models/stance_classification/labels.py
======================================
Gold stance labels from GENUINE completed double annotation, reusing the existing schema and validators
(scripts/analyze_annotation_agreement.py, reports/annotation_guidelines_v1.1.md). No modelling imports.

Inputs per round (pilot, expansion_v2), never modified:
    reference pool                       data/annotation/pilot_annotation_50.csv | expansion_pool_400_v2.csv
    completed annotator files            data/annotation/completed/<round>_annotator_{A,B}_completed.csv
    optional human adjudication          data/annotation/adjudicated/<round>_adjudicated.csv
                                         (columns pilot_id, adjudicated_stance, adjudication_rationale; a filled copy
                                          of the disagreements.csv written by the agreement analysis)

Gold rule for one record (stance only; sentiment and topic are never used):
    1. a valid human adjudicated_stance with a rationale            -> gold, label_source = "adjudicated"
    2. both annotators gave the same valid stance                   -> gold, label_source = "agreement"
       unless either note says the annotator could not read the language (guideline §7: such an `Unclear`
       carries no stance information)                               -> excluded "language_inability"
    3. annotators disagree and no adjudication                      -> excluded "unresolved_disagreement"
    4. either annotation blank                                      -> excluded "missing_annotation"
    pilot_* rows inside the v2 file are labelled in the pilot round  -> excluded "pilot_record_in_v2_file"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                     # scripts/ is imported as a namespace package from the project root
    sys.path.insert(0, str(ROOT))
import scripts.analyze_annotation_agreement as aaa  # noqa: E402
SCHEMA_LABELS = list(aaa.PERMITTED_STANCES)            # Support, Oppose, Mixed, Neutral, Unclear (guidelines v1.1)
ADJUDICATED_DIR = ROOT / "data" / "annotation" / "adjudicated"
LANGUAGE_INABILITY_RE = re.compile(r"non[- ]?english", re.I)
ROUND_ORDER = ("pilot", "expansion_v2")


@dataclass
class RoundStatus:
    round: str
    state: str                                  # missing | invalid | ok
    detail: str = ""
    errors: list[str] = field(default_factory=list)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def adjudication_path(round_name: str, base: Path = ADJUDICATED_DIR) -> Path:
    return base / f"{round_name}_adjudicated.csv"


def load_round(round_name: str, rounds: dict[str, dict[str, Any]] | None = None,
               adjudicated_dir: Path = ADJUDICATED_DIR) -> tuple[RoundStatus, pd.DataFrame | None]:
    """Validate one round's completed files with the existing validator and return per-record annotations."""
    cfg = (rounds or aaa.ROUNDS)[round_name]
    a_path, b_path = Path(cfg["completed_a"]), Path(cfg["completed_b"])
    missing = [p.name for p in (a_path, b_path) if not p.is_file()]
    if missing:
        return RoundStatus(round_name, "missing", f"completed annotator file(s) not found: {missing}"), None
    ref, a, b = _read(Path(cfg["reference"])), _read(a_path), _read(b_path)
    errors, _warnings = aaa.validate_completed_pair(a, b, ref, cfg["segments"])
    for p, df in ((a_path, a), (b_path, b)):            # the shared validator only audits labels; training must reject them
        s = df["stance"].str.strip() if "stance" in df.columns else pd.Series(dtype=str)
        bad = sorted(set(s[(s != "") & ~s.isin(SCHEMA_LABELS)]))
        if bad:
            errors.append(f"{p.name}: stance values outside the schema {bad}; allowed {SCHEMA_LABELS}")
    for p, df in ((a_path, a), (b_path, b)):
        if df["stance"].str.strip().eq("").all():
            errors.append(f"{p.name} contains no stance labels (blank template?)")
    adj = None
    ap = adjudication_path(round_name, adjudicated_dir)
    if ap.is_file():
        adj = _read(ap)
        need = {"pilot_id", "adjudicated_stance", "adjudication_rationale"}
        if not need <= set(adj.columns):
            errors.append(f"{ap.name}: missing columns {sorted(need - set(adj.columns))}")
        else:
            s = adj["adjudicated_stance"].str.strip()
            bad = sorted(set(s[(s != "") & ~s.isin(SCHEMA_LABELS)]))
            if bad:
                errors.append(f"{ap.name}: invalid adjudicated_stance values {bad}")
            if ((s != "") & (adj["adjudication_rationale"].str.strip() == "")).any():
                errors.append(f"{ap.name}: every adjudicated_stance needs an adjudication_rationale")
            unknown = sorted(set(adj["pilot_id"]) - set(ref["pilot_id"]))
            if unknown:
                errors.append(f"{ap.name}: unknown pilot_id {unknown[:5]}")
            if adj["pilot_id"].duplicated().any():
                errors.append(f"{ap.name}: duplicated pilot_id")
    if errors:
        return RoundStatus(round_name, "invalid", "; ".join(errors[:3]), errors), None
    out = ref.drop(columns=[c for c in aaa.ANNOTATION_COLUMNS if c in ref.columns]).copy()
    out["round"] = round_name
    out["stance_A"], out["stance_B"] = a["stance"].str.strip().values, b["stance"].str.strip().values
    out["notes_A"], out["notes_B"] = a["review_notes"].values, b["review_notes"].values
    adj_map = {} if adj is None else {p: (s.strip(), r) for p, s, r in zip(adj["pilot_id"], adj["adjudicated_stance"], adj["adjudication_rationale"]) if s.strip()}
    out["adjudicated_stance"] = out["pilot_id"].map(lambda p: adj_map.get(p, ("", ""))[0])
    out["adjudication_rationale"] = out["pilot_id"].map(lambda p: adj_map.get(p, ("", ""))[1])
    detail = f"{len(out)} records; adjudication file {'present' if adj is not None else 'absent'}"
    return RoundStatus(round_name, "ok", detail), out


def gold_label(row: pd.Series) -> tuple[str, str, str]:
    """Return (gold_stance, label_source, excluded_reason) for one annotated record."""
    if row["round"] == "expansion_v2" and str(row["pilot_id"]).startswith("pilot_"):
        return "", "", "pilot_record_in_v2_file"
    a, b, adj = row["stance_A"], row["stance_B"], row["adjudicated_stance"]
    if adj:
        return adj, "adjudicated", ""
    if not a or not b:
        return "", "", "missing_annotation"
    if a != b:
        return "", "", "unresolved_disagreement"
    if LANGUAGE_INABILITY_RE.search(f"{row['notes_A']} {row['notes_B']}"):
        return "", "", "language_inability"
    return a, "agreement", ""


def build_gold(rounds: tuple[str, ...] = ROUND_ORDER, round_cfg: dict[str, dict[str, Any]] | None = None,
               adjudicated_dir: Path = ADJUDICATED_DIR) -> tuple[list[RoundStatus], pd.DataFrame]:
    """All annotated records of the available rounds with gold label, source and exclusion reason."""
    statuses, frames = [], []
    for r in rounds:
        st, df = load_round(r, round_cfg, adjudicated_dir)
        statuses.append(st)
        if df is not None:
            frames.append(df)
    if not frames:
        return statuses, pd.DataFrame()
    allr = pd.concat(frames, ignore_index=True)
    res = allr.apply(gold_label, axis=1, result_type="expand")
    allr["gold_stance"], allr["label_source"], allr["excluded_reason"] = res[0], res[1], res[2]
    dup = allr["pilot_id"].duplicated(keep="first") & (allr["excluded_reason"] == "")
    allr.loc[dup, ["gold_stance", "label_source", "excluded_reason"]] = ["", "", "duplicate_record_id"]
    return statuses, allr


def readiness(rounds: tuple[str, ...] = ROUND_ORDER, round_cfg: dict[str, dict[str, Any]] | None = None,
              adjudicated_dir: Path = ADJUDICATED_DIR, min_gold: int = 40) -> dict[str, Any]:
    """Light-weight status for reports: BLOCKED (no completed files), INVALID, INSUFFICIENT or READY."""
    statuses, gold = build_gold(rounds, round_cfg, adjudicated_dir)
    rs = {s.round: {"state": s.state, "detail": s.detail} for s in statuses}
    if any(s.state == "invalid" for s in statuses):
        return {"status": "INVALID", "rounds": rs, "gold": 0}
    if all(s.state == "missing" for s in statuses):
        return {"status": "BLOCKED", "rounds": rs, "gold": 0,
                "reason": "no completed human stance annotation exists; training needs both annotators' completed files"}
    n = int((gold["gold_stance"] != "").sum())
    classes = gold.loc[gold["gold_stance"] != "", "gold_stance"].value_counts().to_dict()
    ok = n >= min_gold and len(classes) >= 2
    return {"status": "READY" if ok else "INSUFFICIENT", "rounds": rs, "gold": n, "classes": classes,
            "excluded": gold.loc[gold["excluded_reason"] != "", "excluded_reason"].value_counts().to_dict()}


# ── AI-generated working labels (separate input mode; never mixed with human gold) ──────────────

AI_LABEL_SOURCE = "ai_working_label_unverified"
AI_DISCLAIMER = ("Labels are AI-generated working labels, not human annotations. Metrics measure agreement with these "
                 "AI labels on held-out records; they are not human-verified stance accuracy, gold-standard performance "
                 "or inter-annotator agreement, and do not support generalization claims.")
PROTECTED_LABEL_DIRS = (ROOT / "data" / "annotation" / "completed", ADJUDICATED_DIR)


def load_ai_working_labels(path: Path, round_name: str = "pilot",
                           rounds: dict[str, dict[str, Any]] | None = None) -> tuple[RoundStatus, pd.DataFrame | None]:
    """One AI-assisted draft as provisional working labels, validated against the round's reference pool.

    The same rules as human gold apply wherever they are label-independent: schema labels only, unchanged record ids,
    order and source columns, no duplicate ids, and the guideline §7 language-inability exclusion. Exactly one draft is
    used; nothing is merged or voted. label_source is always AI_LABEL_SOURCE.
    """
    path = Path(path)
    if any(d.resolve() in path.resolve().parents for d in PROTECTED_LABEL_DIRS):
        return RoundStatus(round_name, "invalid", "AI working labels must not be read from the human annotation folders"), None
    if not path.is_file():
        return RoundStatus(round_name, "missing", f"AI working-label file not found: {path}"), None
    cfg = (rounds or aaa.ROUNDS)[round_name]
    ref, d = _read(Path(cfg["reference"])), _read(path)
    errors = []
    if list(d.columns) != list(ref.columns):
        errors.append("columns differ from the reference pool")
    elif d["pilot_id"].tolist() != ref["pilot_id"].tolist():
        errors.append("record ids or order differ from the reference pool")
    else:
        fixed = [c for c in ref.columns if c not in aaa.ANNOTATION_COLUMNS]
        if not d[fixed].equals(ref[fixed]):
            errors.append("source columns or text differ from the reference pool")
    if not errors:
        s = d["stance"].str.strip()
        bad = sorted(set(s[(s != "") & ~s.isin(SCHEMA_LABELS)]))
        if bad:
            errors.append(f"stance values outside the schema {bad}")
        if d["pilot_id"].duplicated().any():
            errors.append("duplicated pilot_id")
    if errors:
        return RoundStatus(round_name, "invalid", "; ".join(errors), errors), None
    out = ref.drop(columns=[c for c in aaa.ANNOTATION_COLUMNS if c in ref.columns]).copy()
    out["round"] = round_name
    out["ai_stance"] = d["stance"].str.strip().values
    out["ai_review_notes"] = d["review_notes"].values
    lang = out["ai_review_notes"].str.contains(LANGUAGE_INABILITY_RE)
    out["gold_stance"] = out["ai_stance"].where((out["ai_stance"] != "") & ~lang, "")
    out["label_source"] = out["gold_stance"].map(lambda s: AI_LABEL_SOURCE if s else "")
    out["excluded_reason"] = ""
    out.loc[out["ai_stance"] == "", "excluded_reason"] = "missing_annotation"
    out.loc[lang & (out["ai_stance"] != ""), "excluded_reason"] = "language_inability"
    return RoundStatus(round_name, "ok", f"{len(out)} records from one AI-assisted draft (unverified)"), out
