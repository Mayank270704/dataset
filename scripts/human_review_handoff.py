"""
scripts/human_review_handoff.py
===============================
Coordinates the three independent human-evaluation tracks without changing their schemas:

    stance_pilot / stance_v2      data/annotation/                         (annotator_A / annotator_B)
    pair_review_44                data/comparisons/multilingual_embedding_450/human_review/  (reviewer_1 / reviewer_2)
    cluster_task_a / _task_b      data/full_corpus/concern_discovery_v1/human_review/        (reviewer_1 / reviewer_2)

Commands
    python scripts/human_review_handoff.py status                         read-only intake check of every expected file
    python scripts/human_review_handoff.py accept --track T --person N --file F
                                                                          validate a returned file, store it under its
                                                                          required name (never overwrites), record receipt
    python scripts/human_review_handoff.py build-packets --stage S [--person N]
                                                                          stage-gated reviewer folders (allow-list only)
    python scripts/human_review_handoff.py audit                          leak audit of templates and built packets

Guarantees
    * Never opens an answer key, pair_catalog.csv or any model output; computes no agreement statistic.
    * Never edits templates or completed files; packets and receipts are write-once / append-only.
    * Creates no ratings: an intake check only reports MISSING, VALID or INVALID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = Path("data/reviewer_handoff")
GUIDE = HANDOFF / "REVIEWER_HANDOFF_GUIDE.md"
RECEIPTS = HANDOFF / "receipts.jsonl"
PACKETS = HANDOFF / "packets"
ANN = Path("data/annotation")
P44 = Path("data/comparisons/multilingual_embedding_450/human_review")
CL = Path("data/full_corpus/concern_discovery_v1/human_review")
GUIDELINES = Path("reports/annotation_guidelines_v1.1.md")
PILOT_ANALYSIS = ANN / "agreement_analysis" / "pilot_v1"

PERMITTED_STANCES = {"Support", "Oppose", "Mixed", "Neutral", "Unclear"}
CONFIDENCE = {"high", "medium", "low"}
EVIDENCE_MAX_CHARS = 200

# name fragments and content tokens that must never reach a reviewer
FORBIDDEN_NAME_PARTS = ("_key_do_not_share", "pair_catalog", "selection_manifest", "package_manifest",
                        "review_package_report", "agreement_analysis", "human_review_analysis", "task_a_key", "task_b_key")
FORBIDDEN_TOKENS = ("strip__umap", "raw__umap", "likely_similar", "likely_dissimilar", "intruder_cluster",
                    "selection_tier", "selection_rationale", "tiers_task_a", "strata_task_a", "candidate_count",
                    "cluster_id")


@dataclass(frozen=True)
class Track:
    name: str
    kind: str                     # stance | pair44 | cluster_a | cluster_b
    id_col: str
    templates: dict[int, Path]
    completed: dict[int, Path]
    expected_ids: dict[int, str]  # stance annotator ids (empty for review tracks)
    docs: tuple[tuple[Path, str], ...]
    fill: tuple[str, ...]
    analysis: str
    rows_required: str = ""       # stance: prefix of rows that must be annotated ("" = all)


TRACKS: dict[str, Track] = {
    "stance_pilot": Track(
        "stance_pilot", "stance", "pilot_id",
        {1: ANN / "annotator_1.csv", 2: ANN / "annotator_2.csv"},
        {1: ANN / "completed" / "pilot_annotator_A_completed.csv", 2: ANN / "completed" / "pilot_annotator_B_completed.csv"},
        {1: "annotator_A", 2: "annotator_B"},
        ((GUIDELINES, "annotation_guidelines_v1.1.md"), (ANN / "README_annotators.md", "README_annotators.md")),
        ("stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"),
        "python scripts\\analyze_annotation_agreement.py --round pilot --check-only; then without --check-only"),
    "stance_v2": Track(
        "stance_v2", "stance", "pilot_id",
        {1: ANN / "expansion_v2_annotator_1.csv", 2: ANN / "expansion_v2_annotator_2.csv"},
        {1: ANN / "completed" / "expansion_v2_annotator_A_completed.csv", 2: ANN / "completed" / "expansion_v2_annotator_B_completed.csv"},
        {1: "annotator_A", 2: "annotator_B"},
        ((GUIDELINES, "annotation_guidelines_v1.1.md"), (ANN / "README_expansion_v2_annotators.md", "README_expansion_v2_annotators.md")),
        ("stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"),
        "python scripts\\analyze_annotation_agreement.py --round expansion_v2 --check-only; then without --check-only",
        rows_required="expansion_v2_"),
    "pair_review_44": Track(
        "pair_review_44", "pair44", "pair_id",
        {1: P44 / "reviewer_1_template.csv", 2: P44 / "reviewer_2_template.csv"},
        {1: P44 / "completed" / "reviewer_1_completed.csv", 2: P44 / "completed" / "reviewer_2_completed.csv"},
        {},
        ((P44 / "README_reviewers.md", "README_reviewers.md"), (P44 / "review_guide.md", "review_guide.md")),
        ("reviewer_id", "concern_relation", "confidence", "shared_concern_note", "evidence_note"),
        "python scripts\\analyze_concern_review_agreement.py"),
    "cluster_task_a": Track(
        "cluster_task_a", "cluster_a", "item_id",
        {1: CL / "reviewer_1_task_a.csv", 2: CL / "reviewer_2_task_a.csv"},
        {1: CL / "completed" / "reviewer_1_task_a.csv", 2: CL / "completed" / "reviewer_2_task_a.csv"},
        {},
        ((CL / "review_guide.md", "cluster_review_guide.md"),),
        ("reviewer_id", "intruder_position", "coherence_of_others", "shared_concern_note", "confidence"),
        "python scripts\\run_concern_discovery.py analyze-reviews   (needs task A and task B from both reviewers)"),
    "cluster_task_b": Track(
        "cluster_task_b", "cluster_b", "item_id",
        {1: CL / "reviewer_1_task_b.csv", 2: CL / "reviewer_2_task_b.csv"},
        {1: CL / "completed" / "reviewer_1_task_b.csv", 2: CL / "completed" / "reviewer_2_task_b.csv"},
        {},
        ((CL / "review_guide.md", "cluster_review_guide.md"),),
        ("reviewer_id", "summary_support", "misleading_claim", "note", "confidence"),
        "python scripts\\run_concern_discovery.py analyze-reviews   (needs task A and task B from both reviewers)"),
}
STAGES = {1: ("stance_pilot", "pair_review_44", "cluster_task_a"), 2: ("cluster_task_b",), 3: ("stance_v2",)}


# ── reading and normalisation ─────────────────────────────────────────────────

def read_csv(path: Path) -> pd.DataFrame:
    """UTF-8 with or without BOM; every cell as text; empty cells stay empty strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def norm(v: Any) -> str:
    return str(v).replace("\r\n", "\n").replace("\r", "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Check:
    track: str
    person: int
    path: str
    state: str = "MISSING"          # MISSING | VALID | INVALID
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reviewer_id: str = ""
    sha256: str = ""


# ── per-file validation (template vs returned file; no key, no catalog) ──────

def _value_problems(df: pd.DataFrame, col: str, allowed: set[str], lower: bool = False, ids: pd.Series | None = None,
                    rows: pd.Series | None = None) -> list[str]:
    v = df[col].str.strip()
    v = v.str.lower() if lower else v
    m = rows if rows is not None else pd.Series(True, index=df.index)
    out = []
    blank = m & (v == "")
    if blank.any():
        out.append(f"{col}: {int(blank.sum())} blank cell(s) (use U / Unclear instead), e.g. {list((ids if ids is not None else df.index)[blank][:3])}")
    bad = sorted(set(v[m & (v != "") & ~v.isin(allowed)]))
    if bad:
        out.append(f"{col}: invalid value(s) {bad[:5]}; allowed {sorted(allowed)}")
    return out


OTHER_PERSON_RE = re.compile(r"(?:reviewer|annotator)[ _-]?([12ab])\b", re.I)


def _names_other_person(value: str, person: int) -> bool:
    m = OTHER_PERSON_RE.search(value or "")
    return bool(m) and m.group(1).lower() not in ({"1", "a"} if person == 1 else {"2", "b"})


def validate_file(track: Track, person: int, completed: pd.DataFrame, template: pd.DataFrame,
                  other_template: pd.DataFrame | None = None) -> tuple[list[str], list[str], str]:
    """Return (problems, warnings, reviewer_id). Problems make the file INVALID (return it to the reviewer)."""
    P: list[str] = []
    W: list[str] = []
    idc = track.id_col
    if list(completed.columns) != list(template.columns):
        missing = [c for c in template.columns if c not in completed.columns]
        extra = [c for c in completed.columns if c not in template.columns]
        return [f"columns differ from the template (missing={missing}, extra={extra}; names and order must be unchanged)"], W, ""
    ids = completed[idc]
    if ids.duplicated().any():
        P.append(f"duplicated {idc}: {sorted(set(ids[ids.duplicated()]))[:5]}")
    if set(ids) != set(template[idc]) or len(ids) != len(template):
        P.append(f"rows differ from the template ({len(ids)} vs {len(template)}; missing {sorted(set(template[idc]) - set(ids))[:3]}, "
                 f"extra {sorted(set(ids) - set(template[idc]))[:3]})")
        return P, W, ""
    if ids.tolist() != template[idc].tolist():
        other = 2 if person == 1 else 1
        if (other_template is not None and track.kind in ("cluster_a", "cluster_b")
                and ids.tolist() == other_template[idc].tolist()):
            P.append(f"this is person {other}'s file (its row order matches reviewer_{other}'s template); accept it under --person {other}")
        else:
            P.append("row order differs from the template; rows must not be sorted or moved")
    fixed = [c for c in template.columns if c not in track.fill and c != idc]
    a = completed.set_index(idc).loc[template[idc], fixed].map(norm)
    b = template.set_index(idc)[fixed].map(norm)
    diff = a.values != b.values
    if diff.any():
        cells = [(i, c) for r, i in enumerate(b.index) for k, c in enumerate(fixed) if diff[r, k]]
        hint = ""
        if any(re.match(r"^(#NAME\?|#VALUE!|#REF!|=)", str(a.loc[i, c])) for i, c in cells):
            hint = " A spreadsheet turned text into a formula; reopen the blank template with all columns as text."
        P.append(f"comment text or source columns modified in {len(cells)} cell(s), e.g. {[f'{i}/{c}' for i, c in cells[:4]]}.{hint}")
    c = completed.set_index(idc).loc[template[idc]].reset_index()
    ids = c[idc]
    rid = ""
    if track.kind in ("pair44", "cluster_a", "cluster_b"):
        r = c["reviewer_id"].str.strip()
        if (r == "").any():
            P.append(f"reviewer_id: {int((r == '').sum())} blank cell(s)")
        vals = sorted(set(r) - {""})
        if len(vals) > 1:
            P.append(f"reviewer_id: several values {vals}; use one id on every row")
        rid = vals[0] if vals else ""
        if rid and _names_other_person(rid, person):
            P.append(f"reviewer_id '{rid}' names the other reviewer; this file does not belong to person {person}")
    if track.kind == "pair44":
        P += _value_problems(c, "concern_relation", {"0", "1", "2", "3", "U"}, ids=ids)
        P += _value_problems(c, "confidence", CONFIDENCE, lower=True, ids=ids)
        need = c["concern_relation"].str.strip().isin({"2", "3"}) & (c["shared_concern_note"].str.strip() == "")
        if need.any():
            W.append(f"shared_concern_note empty on {int(need.sum())} pair(s) rated 2 or 3")
    elif track.kind == "cluster_a":
        P += _value_problems(c, "intruder_position", {"1", "2", "3", "4", "5", "6", "U"}, ids=ids)
        P += _value_problems(c, "coherence_of_others", {"0", "1", "2", "3", "U"}, ids=ids)
        P += _value_problems(c, "confidence", CONFIDENCE, lower=True, ids=ids)
        need = c["coherence_of_others"].str.strip().isin({"2", "3"}) & (c["shared_concern_note"].str.strip() == "")
        if need.any():
            W.append(f"shared_concern_note empty on {int(need.sum())} item(s) rated 2 or 3")
    elif track.kind == "cluster_b":
        P += _value_problems(c, "summary_support", {"0", "1", "2", "U"}, ids=ids)
        P += _value_problems(c, "misleading_claim", {"yes", "no"}, lower=True, ids=ids)
        P += _value_problems(c, "confidence", CONFIDENCE, lower=True, ids=ids)
    else:                                                    # stance
        required = ids.str.startswith(track.rows_required) if track.rows_required else pd.Series(True, index=c.index)
        P += _value_problems(c, "stance", PERMITTED_STANCES, ids=ids, rows=required)
        exp = track.expected_ids[person]
        annotated = (c[["stance", "concern", "requested_action", "evidence_span", "review_notes"]].apply(lambda s: s.str.strip() != "")).any(axis=1)
        a_ids = c["annotator_id"].str.strip()
        if ((a_ids == "") & (required | annotated)).any():
            P.append(f"annotator_id blank on {int(((a_ids == '') & (required | annotated)).sum())} annotated/required row(s); use '{exp}'")
        wrong = sorted(set(a_ids[a_ids != ""]) - {exp})
        if wrong:
            P.append(f"annotator_id must be '{exp}' for this file, found {wrong}")
        for pid, span, text in zip(ids, c["evidence_span"], c["original_text"]):
            s = span.strip()
            if s and (s not in text or len(span) > EVIDENCE_MAX_CHARS):
                P.append(f"{pid}: evidence_span must be a verbatim excerpt of the comment, at most {EVIDENCE_MAX_CHARS} characters")
        optional = (~required) & annotated
        if optional.any():
            W.append(f"{int(optional.sum())} optional pilot_* row(s) annotated; they are analysed separately and never counted among the 350")
        rid = exp
    return P, W, rid


def receipts(root: Path) -> dict[str, str]:
    p = root / RECEIPTS
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["path"]] = r["sha256"]
    return out


def check_one(root: Path, track: Track, person: int, rec: dict[str, str] | None = None, path: Path | None = None) -> Check:
    rel = path or track.completed[person]
    p = rel if rel.is_absolute() else root / rel
    ck = Check(track.name, person, rel.as_posix())
    if not p.is_file():
        if path is None and (rec or {}).get(track.completed[person].as_posix()):
            ck.state, ck.problems = "INVALID", ["accepted file is missing (a receipt exists); restore it from the reviewer's original submission"]
        return ck
    ck.sha256 = sha256(p)
    try:
        df = read_csv(p)
    except UnicodeDecodeError:
        ck.state, ck.problems = "INVALID", ["file is not UTF-8; save it as 'CSV UTF-8 (comma delimited)'"]
        return ck
    except Exception as exc:  # noqa: BLE001  (malformed CSV)
        ck.state, ck.problems = "INVALID", [f"cannot be read as CSV: {exc}"]
        return ck
    tmpl = read_csv(root / track.templates[person])
    other = read_csv(root / track.templates[2 if person == 1 else 1])
    ck.problems, ck.warnings, ck.reviewer_id = validate_file(track, person, df, tmpl, other)
    stored = (rec or {}).get(track.completed[person].as_posix())
    if path is None and stored and stored != ck.sha256:
        ck.problems.append("file changed after it was accepted (SHA-256 differs from the receipt)")
    if path is None and not stored and rec is not None:
        ck.warnings.append("present but not accepted (no receipt); run the accept command on it")
    ck.state = "INVALID" if ck.problems else "VALID"
    return ck


def status(root: Path = ROOT) -> list[Check]:
    rec = receipts(root)
    out = [check_one(root, t, n, rec) for t in TRACKS.values() for n in (1, 2)]
    by = {(c.track, c.person): c for c in out}
    for t in TRACKS:                                         # cross-file rule: two different people per track
        a, b = by[(t, 1)], by[(t, 2)]
        if a.state == b.state == "VALID" and a.reviewer_id and a.reviewer_id == b.reviewer_id:
            b.warnings.append(f"both files use reviewer id '{a.reviewer_id}'; confirm they come from two different people")
        if a.state == b.state == "VALID":
            tr = TRACKS[t]
            ja, jb = (read_csv(root / tr.completed[n]).set_index(tr.id_col).sort_index() for n in (1, 2))
            cols = [c for c in tr.fill if c not in ("reviewer_id", "annotator_id")]
            if ja[cols].equals(jb[cols]):
                b.warnings.append("judgments (including notes) are identical to the other person's file; "
                                  "confirm with both people that they worked independently before analysis")
    for n in (1, 2):                                         # task B must not be returned before task A was accepted
        if by[("cluster_task_b", n)].state != "MISSING" and TRACKS["cluster_task_a"].completed[n].as_posix() not in rec:
            by[("cluster_task_b", n)].warnings.append("task B present but task A was never accepted; check the stage order")
    return out


def track_readiness(checks: list[Check]) -> dict[str, str]:
    by: dict[str, list[Check]] = {}
    for c in checks:
        by.setdefault(c.track, []).append(c)
    out = {}
    for t, cs in by.items():
        states = [c.state for c in cs]
        if all(s == "VALID" for s in states):
            out[t] = "READY: " + TRACKS[t].analysis
        elif "INVALID" in states:
            out[t] = "FIX: return invalid file(s) to the reviewer"
        else:
            out[t] = f"BLOCKED: {states.count('MISSING')} of 2 completed file(s) missing"
    return out


# ── accept ────────────────────────────────────────────────────────────────────

def accept(root: Path, track_name: str, person: int, file: Path) -> Check:
    """Validate a returned file, store it under its required name and append a SHA-256 receipt.

    If ``file`` already IS the required completed path (some track READMEs tell reviewers to save there), it is
    validated and receipted in place. Refuses: invalid files, an existing different file, a second receipt for the
    same path, and a file byte-identical to any other accepted submission.
    """
    track = TRACKS[track_name]
    rec = receipts(root)
    dest = root / track.completed[person]
    ck = check_one(root, track, person, path=file.resolve())
    if ck.state != "VALID":
        return ck
    key = track.completed[person].as_posix()
    if key in rec:
        ck.state, ck.problems = "INVALID", [f"a submission for {key} was already accepted (receipt exists); accepted files are never replaced"]
        return ck
    dup = [p for p, h in rec.items() if h == ck.sha256]
    if dup:
        ck.state, ck.problems = "INVALID", [f"byte-identical to the already accepted {dup[0]}; a duplicate cannot count as a second review"]
        return ck
    in_place = dest.exists() and dest.resolve() == file.resolve()
    if dest.exists() and not in_place:
        ck.state, ck.problems = "INVALID", [f"{key} already exists; it is never overwritten (accept it in place with --file {key})"]
        return ck
    if not in_place:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "rb") as fi, open(dest, "xb") as fo:
            fo.write(fi.read())
    if sha256(dest) != ck.sha256:
        raise OSError("copy verification failed")
    other = root / track.completed[2 if person == 1 else 1]
    if other.exists():
        o = read_csv(other).set_index(track.id_col).sort_index()
        m = read_csv(dest).set_index(track.id_col).sort_index()
        cols = [c for c in track.fill if c not in ("reviewer_id", "annotator_id")]
        if o[cols].equals(m[cols]):
            ck.warnings.append("judgments (including notes) are identical to the other person's file; confirm independent work")
    rp = root / RECEIPTS
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"path": track.completed[person].as_posix(), "sha256": ck.sha256, "track": track_name,
                            "person": person, "reviewer_id": ck.reviewer_id,
                            "accepted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    ck.path = track.completed[person].as_posix()
    return ck


# ── packets ───────────────────────────────────────────────────────────────────

def stage_gate(root: Path, stage: int, person: int | None) -> list[str]:
    """Return unmet conditions for releasing a stage (empty = may release)."""
    if stage == 1:
        return []
    checks = {(c.track, c.person): c for c in status(root)}
    rec = receipts(root)
    unmet = []
    if stage == 2:
        for n in ([person] if person else [1, 2]):
            a = checks[("cluster_task_a", n)]
            if a.state != "VALID" or TRACKS["cluster_task_a"].completed[n].as_posix() not in rec:
                unmet.append(f"person {n}: task A must be returned and accepted first (now {a.state})")
        return unmet
    for t in ("stance_pilot", "pair_review_44", "cluster_task_a", "cluster_task_b"):
        for n in (1, 2):
            if checks[(t, n)].state != "VALID":
                unmet.append(f"{t} person {n}: {checks[(t, n)].state}")
    if not (root / PILOT_ANALYSIS).exists():
        unmet.append(f"pilot agreement analysis not run yet ({PILOT_ANALYSIS.as_posix()} missing); calibrate the guidelines first")
    return unmet


def audit_dir(root: Path, d: Path, allowed_sources: dict[Path, Path] | None = None) -> list[str]:
    """Leak audit of a reviewer-facing folder: forbidden names, forbidden tokens, and (optionally) byte-identity."""
    problems = []
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        if any(part in rel for part in FORBIDDEN_NAME_PARTS) or "key" in p.name.lower():
            problems.append(f"forbidden file {rel}")
            continue
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        hits = [t for t in FORBIDDEN_TOKENS if t in text]
        if hits:
            problems.append(f"{rel} contains {hits}")
        if allowed_sources is not None:
            src = allowed_sources.get(p)
            if src is None:
                problems.append(f"{rel} is not on the allow-list")
            elif sha256(p) != sha256(src):
                problems.append(f"{rel} differs from its source {src.relative_to(root).as_posix()}")
    return problems


def build_packets(root: Path, stage: int, person: int | None = None) -> dict[str, Any]:
    unmet = stage_gate(root, stage, person)
    if unmet:
        return {"status": "BLOCKED", "stage": stage, "unmet": unmet}
    people = [person] if person else [1, 2]
    stage_dir = root / PACKETS / f"stage_{stage}"
    targets = [stage_dir / f"person_{n}" for n in people]
    existing = [t for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f"Packet already exists: {existing[0]}")
    plan: dict[Path, Path] = {}
    for n in people:
        base = stage_dir / f"person_{n}"
        plan[base / "00_REVIEWER_HANDOFF_GUIDE.md"] = root / GUIDE
        for t in STAGES[stage]:
            tr = TRACKS[t]
            plan[base / t / tr.templates[n].name] = root / tr.templates[n]
            for src, name in tr.docs:
                plan[base / t / name] = root / src
    missing = [s for s in plan.values() if not s.is_file()]
    if missing:
        raise FileNotFoundError(f"source files missing: {[m.relative_to(root).as_posix() for m in missing]}")
    for dst, src in plan.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as fi, open(dst, "xb") as fo:
            fo.write(fi.read())
    problems = []
    for t in targets:
        problems += audit_dir(root, t, {d: s for d, s in plan.items() if t in d.parents})
    if problems:
        for t in targets:
            shutil.rmtree(t)
        raise ValueError(f"Leak audit failed; packet removed: {problems}")
    manifest = {"stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "people": people, "leak_audit": "PASSED",
                "files": {d.relative_to(root).as_posix(): {"source": s.relative_to(root).as_posix(), "sha256": sha256(d)}
                          for d, s in plan.items()}}
    mp = stage_dir / ("packet_manifest.json" if not person else f"packet_manifest_person_{person}.json")
    with open(mp, "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return {"status": "BUILT", "stage": stage, "folders": [t.relative_to(root).as_posix() for t in targets],
            "files_per_person": len(plan) // len(people), "leak_audit": "PASSED"}


def audit(root: Path = ROOT) -> list[str]:
    """Templates must be blank with opaque ids and no leaked columns; built packets must pass the folder audit."""
    problems = []
    for t in TRACKS.values():
        for n, rel in t.templates.items():
            df = read_csv(root / rel)
            filled = [c for c in t.fill if c in df.columns and (df[c].str.strip() != "").any()]
            if filled:
                problems.append(f"{rel.as_posix()}: reviewer columns pre-filled {filled}")
            if any(tok in " ".join(df.columns) for tok in FORBIDDEN_TOKENS) or any(
                    re.search(r"cluster|tier|cosine|similar|model|stratum|selection|intruder_(?!position)", c, re.I) for c in df.columns):
                problems.append(f"{rel.as_posix()}: leaked column names")
            if df[t.id_col].duplicated().any() or (df[t.id_col].str.strip() == "").any():
                problems.append(f"{rel.as_posix()}: ids blank or duplicated")
    for t in TRACKS.values():                                 # the two reviewers' templates must hold the same items
        a, b = read_csv(root / t.templates[1]), read_csv(root / t.templates[2])
        if set(a[t.id_col]) != set(b[t.id_col]):
            problems.append(f"{t.name}: reviewer templates hold different items")
    pk = root / PACKETS
    if pk.exists():
        for d in sorted(pk.glob("stage_*/person_*")):
            if d.is_dir():
                problems += [f"{d.relative_to(root).as_posix()}: {p}" for p in audit_dir(root, d)]
                problems += [f"{d.relative_to(root).as_posix()}: {p}" for p in audit_packet_against_manifest(root, d)]
            elif d.suffix.lower() == ".zip":
                problems += [f"{d.relative_to(root).as_posix()}: {p}" for p in audit_packet_zip(root, d)]
            else:
                problems.append(f"{d.relative_to(root).as_posix()}: unexpected file in the packets folder")
    return problems


def _manifest_files(stage_dir: Path) -> dict[str, str]:
    listed: dict[str, str] = {}
    for mp in sorted(stage_dir.glob("packet_manifest*.json")):
        for rel, info in json.loads(mp.read_text(encoding="utf-8")).get("files", {}).items():
            listed[rel] = info["sha256"]
    return listed


def audit_packet_zip(root: Path, zpath: Path) -> list[str]:
    """A packet archive must contain exactly its folder's manifest files, byte-identical, with blank judgments."""
    import io
    import zipfile
    stage_dir = zpath.parent
    stage_rel = stage_dir.relative_to(root).as_posix() + "/"
    prefix = stage_rel + zpath.stem + "/"
    expected = {rel: h for rel, h in _manifest_files(stage_dir).items() if rel.startswith(prefix)}
    if not expected:
        return ["no packet manifest lists the folder this archive should contain"]
    problems = []
    try:
        z = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        return ["not a readable zip archive"]
    with z:
        members = {stage_rel + i.filename: i for i in z.infolist() if not i.is_dir()}
        for rel in sorted(set(members) - set(expected)):
            problems.append(f"archive member not in the packet manifest (do not send): {rel[len(stage_rel):]}")
        for rel in sorted(set(expected) - set(members)):
            problems.append(f"archive is missing manifest file: {rel[len(stage_rel):]}")
        for rel in sorted(set(expected) & set(members)):
            data = z.read(members[rel])
            if hashlib.sha256(data).hexdigest() != expected[rel]:
                problems.append(f"archive member differs from the built file: {rel[len(stage_rel):]}")
            name = rel.lower()
            if any(part in name for part in FORBIDDEN_NAME_PARTS) or any(t.encode() in data for t in FORBIDDEN_TOKENS):
                problems.append(f"archive member contains forbidden name or content: {rel[len(stage_rel):]}")
        for rel, info in sorted(members.items()):
            if rel.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(z.read(info)), dtype=str, keep_default_na=False, encoding="utf-8-sig")
                filled = [c for c in JUDGMENT_COLUMNS if c in df.columns and (df[c].str.strip() != "").any()]
                if filled:
                    problems.append(f"pre-filled judgment columns {filled} in archive member {rel[len(stage_rel):]}")
    return problems


JUDGMENT_COLUMNS = sorted({c for t in TRACKS.values() for c in t.fill})


def audit_packet_against_manifest(root: Path, person_dir: Path) -> list[str]:
    """A built packet must hold exactly the files its build manifest lists, unchanged, with blank judgment columns.

    Read-only: reports extra, missing and modified files and any CSV with pre-filled judgment cells
    (for example draft labels added after the build); it never edits or removes anything.
    """
    stage_dir = person_dir.parent
    listed: dict[str, str] = {}
    for mp in sorted(stage_dir.glob("packet_manifest*.json")):
        for rel, info in json.loads(mp.read_text(encoding="utf-8")).get("files", {}).items():
            listed[rel] = info["sha256"]
    prefix = person_dir.relative_to(root).as_posix() + "/"
    expected = {rel: h for rel, h in listed.items() if rel.startswith(prefix)}
    if not expected:
        return ["no packet manifest lists this folder; it was not built by build-packets"]
    problems = []
    on_disk = {p.relative_to(root).as_posix(): p for p in person_dir.rglob("*") if p.is_file()}
    for rel in sorted(set(on_disk) - set(expected)):
        problems.append(f"file not in the packet manifest (added after the build; do not send): {rel[len(prefix):]}")
    for rel in sorted(set(expected) - set(on_disk)):
        problems.append(f"file listed in the packet manifest is missing: {rel[len(prefix):]}")
    for rel in sorted(set(expected) & set(on_disk)):
        if sha256(on_disk[rel]) != expected[rel]:
            problems.append(f"file changed after the build: {rel[len(prefix):]}")
    for rel, p in sorted(on_disk.items()):
        if p.suffix.lower() != ".csv":
            continue
        try:
            df = read_csv(p)
        except Exception:  # noqa: BLE001
            problems.append(f"unreadable CSV: {rel[len(prefix):]}")
            continue
        filled = [c for c in JUDGMENT_COLUMNS if c in df.columns and (df[c].astype(str).str.strip() != "").any()]
        if filled:
            problems.append(f"pre-filled judgment columns {filled} in {rel[len(prefix):]}; reviewers must receive blank templates")
    return problems


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_status(checks: list[Check]) -> None:
    for c in checks:
        print(f"{c.state:8} {c.track:15} person {c.person}  {c.path}")
        for p in c.problems:
            print(f"           problem: {p}")
        for w in c.warnings:
            print(f"           warning: {w}")
    print()
    for t, r in track_readiness(checks).items():
        print(f"{t:15} {r}")


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="PolicyLens human-review handoff and intake")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("audit")
    a = sub.add_parser("accept")
    a.add_argument("--track", required=True, choices=sorted(TRACKS))
    a.add_argument("--person", required=True, type=int, choices=[1, 2])
    a.add_argument("--file", required=True, type=Path)
    b = sub.add_parser("build-packets")
    b.add_argument("--stage", required=True, type=int, choices=sorted(STAGES))
    b.add_argument("--person", type=int, choices=[1, 2])
    args = ap.parse_args(argv)
    if args.cmd == "status":
        checks = status(ROOT)
        _print_status(checks)
        return 1 if any(c.state == "INVALID" for c in checks) else 0
    if args.cmd == "audit":
        probs = audit(ROOT)
        print("\n".join(probs) if probs else "audit PASSED: templates blank and opaque; packets contain allow-listed files only")
        return 1 if probs else 0
    if args.cmd == "accept":
        ck = accept(ROOT, args.track, args.person, args.file)
        _print_status([ck])
        return 0 if ck.state == "VALID" else 1
    try:
        res = build_packets(ROOT, args.stage, args.person)
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "BUILT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
