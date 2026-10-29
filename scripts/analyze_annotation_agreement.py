"""
scripts/analyze_annotation_agreement.py
=======================================
Reusable two-annotator agreement analysis for BOTH annotation rounds:

    --round pilot          50-row pilot (reference: data/annotation/pilot_annotation_50.csv)
    --round expansion_v2   v2 pool (reference: data/annotation/expansion_pool_400_v2.csv)

Reads ONLY completed files from data/annotation/completed/ (never the blank templates):
    pilot:         pilot_annotator_A_completed.csv, pilot_annotator_B_completed.csv
    expansion_v2:  expansion_v2_annotator_A_completed.csv, expansion_v2_annotator_B_completed.csv

Segments are analysed separately and never pooled:
    pilot round         -> segment "pilot" (50 rows, required)
    expansion_v2 round  -> segment "expansion" (350 additional records, required) and
                           segment "pilot_segment" (the 50 pilot rows inside the v2 file; optional,
                           NOT counted among the 350)

Validation (blocking): exact column names/order, row IDs and order, unchanged provenance/source
columns, annotator_id equal to annotator_A / annotator_B (and distinct), evidence_span a verbatim
substring of original_text and at most 200 characters.
Reported but non-blocking: blank or invalid stance labels, concern > 15 words, action > 20 words.

Outputs go to a NEW directory (default data/annotation/agreement_analysis/<round>_v1); an existing
directory is never overwritten. Nothing is written when the files are missing, invalid, or contain
no stance labels yet. Disagreement tables carry blank adjudication columns; labels are never
resolved or fabricated.

Agreement is computed on stratified samples; it describes annotator reliability on those records,
not population prevalence. Stance is not sentiment; concern and requested_action are free text and
are described with exact-match and token-overlap rates only (no kappa).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION = ROOT / "data" / "annotation"
COMPLETED = ANNOTATION / "completed"
OUTPUT_BASE = ANNOTATION / "agreement_analysis"

PERMITTED_STANCES = ["Support", "Oppose", "Mixed", "Neutral", "Unclear"]
ANNOTATION_COLUMNS = ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]
EXPECTED_IDS = {"A": "annotator_A", "B": "annotator_B"}
EVIDENCE_MAX_CHARS = 200
WORD_LIMITS = {"concern": 15, "requested_action": 20}
MIN_PAIRS_FOR_KAPPA = 5
KAPPA_TARGET = 0.60

ROUNDS: dict[str, dict[str, Any]] = {
    "pilot": {
        "label": "PILOT ANALYSIS (50-row pilot)",
        "reference": ANNOTATION / "pilot_annotation_50.csv",
        "completed_a": COMPLETED / "pilot_annotator_A_completed.csv",
        "completed_b": COMPLETED / "pilot_annotator_B_completed.csv",
        "templates": [ANNOTATION / "annotator_1.csv", ANNOTATION / "annotator_2.csv"],
        "segments": [{"name": "pilot", "prefix": "", "required": True,
                      "description": "50 pilot records"}],
    },
    "expansion_v2": {
        "label": "EXPANSION ANALYSIS (v2 pool: 350 additional records)",
        "reference": ANNOTATION / "expansion_pool_400_v2.csv",
        "completed_a": COMPLETED / "expansion_v2_annotator_A_completed.csv",
        "completed_b": COMPLETED / "expansion_v2_annotator_B_completed.csv",
        "templates": [ANNOTATION / "expansion_v2_annotator_1.csv", ANNOTATION / "expansion_v2_annotator_2.csv"],
        "segments": [
            {"name": "expansion", "prefix": "expansion_v2_", "required": True,
             "description": "350 additional records (primary result)"},
            {"name": "pilot_segment", "prefix": "pilot_", "required": False,
             "description": "the 50 pilot rows carried inside the v2 file; optional, NOT part of the 350"},
        ],
    },
}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_jaccard(a: str, b: str) -> float:
    t1 = set(re.findall(r"\w+", a.lower()))
    t2 = set(re.findall(r"\w+", b.lower()))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def cohen_kappa(y1: list[str], y2: list[str], categories: list[str] = PERMITTED_STANCES) -> dict[str, Any]:
    """Same guards as notebooks/03: withheld if N < 5 or Pe = 1 (zero marginal variation)."""
    n = len(y1)
    if n != len(y2):
        raise ValueError("Label lists must have identical length.")
    if n < MIN_PAIRS_FOR_KAPPA:
        return {"status": "insufficient_data", "n": n, "kappa": None, "po": None, "pe": None,
                "message": f"Only {n} valid pairs (< {MIN_PAIRS_FOR_KAPPA}); kappa withheld."}
    idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    for a, b in zip(y1, y2):
        matrix[idx[a], idx[b]] += 1
    po = float(np.trace(matrix) / n)
    pe = float(np.sum(matrix.sum(axis=1) * matrix.sum(axis=0)) / (n ** 2))
    if np.isclose(1.0 - pe, 0.0):
        return {"status": "zero_variance", "n": n, "kappa": None, "po": po, "pe": pe,
                "message": "Expected agreement Pe = 1.0 (zero marginal variation); kappa undefined."}
    kappa = float((po - pe) / (1.0 - pe))
    return {"status": "computed", "n": n, "kappa": kappa, "po": po, "pe": pe, "target_met": kappa >= KAPPA_TARGET}


def segment_masks(ref: pd.DataFrame, segments: list[dict[str, Any]]) -> dict[str, pd.Series]:
    return {s["name"]: ref["pilot_id"].str.startswith(s["prefix"]) for s in segments}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_completed_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame, ref: pd.DataFrame, segments: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Return (blocking_errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    masks = segment_masks(ref, segments)
    required = pd.Series(False, index=ref.index)
    for s in segments:
        if s["required"]:
            required |= masks[s["name"]]
    non_ann = [c for c in ref.columns if c not in ANNOTATION_COLUMNS]

    for key, df in (("A", df_a), ("B", df_b)):
        label = f"Annotator {key}"
        if list(df.columns) != list(ref.columns):
            missing = [c for c in ref.columns if c not in df.columns]
            extra = [c for c in df.columns if c not in ref.columns]
            errors.append(f"{label}: columns differ from reference (missing={missing}, extra={extra}; "
                          "names and order must be unchanged).")
            continue
        if len(df) != len(ref) or df["pilot_id"].tolist() != ref["pilot_id"].tolist():
            errors.append(f"{label}: row IDs/order differ from reference ({len(df)} vs {len(ref)} rows).")
            continue
        changed = [c for c in non_ann if not df[c].equals(ref[c])]
        if changed:
            errors.append(f"{label}: source/provenance columns modified: {changed}.")
        expected = EXPECTED_IDS[key]
        ids = df["annotator_id"].str.strip()
        has_any = (df[[c for c in ANNOTATION_COLUMNS if c != "annotator_id"]].apply(lambda s: s.str.strip() != "")).any(axis=1)
        need_id = required | has_any
        blank_ids = int(((ids == "") & need_id).sum())
        if blank_ids:
            errors.append(f"{label}: annotator_id blank on {blank_ids} row(s) that must carry '{expected}'.")
        wrong = sorted(set(ids[ids != ""]) - {expected})
        if wrong:
            errors.append(f"{label}: annotator_id must be '{expected}', found {wrong}.")
        for pid, span, text in zip(df["pilot_id"], df["evidence_span"], df["original_text"]):
            if span.strip() == "":
                continue
            if span.strip() not in text:
                errors.append(f"{label}: {pid} evidence_span is not a verbatim substring of original_text.")
            if len(span) > EVIDENCE_MAX_CHARS:
                errors.append(f"{label}: {pid} evidence_span has {len(span)} characters (maximum {EVIDENCE_MAX_CHARS}).")
        for col, limit in WORD_LIMITS.items():
            for pid, value in zip(df["pilot_id"], df[col]):
                n_words = len(value.split())
                if n_words > limit:
                    warnings.append(f"{label}: {pid} {col} has {n_words} words (preferred maximum {limit}).")
    ids_a = set(df_a["annotator_id"].str.strip()) - {""} if "annotator_id" in df_a.columns else set()
    ids_b = set(df_b["annotator_id"].str.strip()) - {""} if "annotator_id" in df_b.columns else set()
    if ids_a and ids_a == ids_b:
        errors.append(f"Both files use the same annotator_id {sorted(ids_a)}; IDs must be distinct.")
    return errors, warnings


# ── Metrics ───────────────────────────────────────────────────────────────────

def _label_audit(stances: pd.Series) -> dict[str, Any]:
    s = stances.str.strip()
    valid = s.isin(PERMITTED_STANCES)
    blank = s == ""
    return {
        "rows": int(len(s)),
        "valid": int(valid.sum()),
        "blank": int(blank.sum()),
        "invalid": int((~valid & ~blank).sum()),
        "invalid_values": sorted(set(s[~valid & ~blank])),
        "distribution": {k: int(v) for k, v in s[valid].value_counts().items()},
    }


def segment_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    a, b = df_a[mask].reset_index(drop=True), df_b[mask].reset_index(drop=True)
    sa, sb = a["stance"].str.strip(), b["stance"].str.strip()
    va, vb = sa.isin(PERMITTED_STANCES), sb.isin(PERMITTED_STANCES)
    both = va & vb
    excluded = {
        "rows_in_segment": int(len(a)),
        "paired_valid": int(both.sum()),
        "excluded_total": int((~both).sum()),
        "excluded_a_blank": int((sa == "").sum()),
        "excluded_b_blank": int((sb == "").sum()),
        "excluded_a_invalid": int((~va & (sa != "")).sum()),
        "excluded_b_invalid": int((~vb & (sb != "")).sum()),
    }
    agreement = cohen_kappa(sa[both].tolist(), sb[both].tolist())
    cm = pd.DataFrame(0, index=PERMITTED_STANCES, columns=PERMITTED_STANCES)
    for x, y in zip(sa[both], sb[both]):
        cm.loc[x, y] += 1
    conc = (a["concern"].str.strip() != "") & (b["concern"].str.strip() != "")
    act = (a["requested_action"].str.strip() != "") & (b["requested_action"].str.strip() != "")
    free = {
        "concern_pairs": int(conc.sum()),
        "concern_exact_match": int((a.loc[conc, "concern"].str.strip() == b.loc[conc, "concern"].str.strip()).sum()),
        "concern_mean_jaccard": float(np.mean([token_jaccard(x, y) for x, y in zip(a.loc[conc, "concern"], b.loc[conc, "concern"])])) if conc.any() else None,
        "action_pairs": int(act.sum()),
        "action_exact_match": int((a.loc[act, "requested_action"].str.strip() == b.loc[act, "requested_action"].str.strip()).sum()),
        "action_mean_jaccard": float(np.mean([token_jaccard(x, y) for x, y in zip(a.loc[act, "requested_action"], b.loc[act, "requested_action"])])) if act.any() else None,
    }
    disagreements = pd.DataFrame({
        "pilot_id": a["pilot_id"], "source_dataset": a["source_dataset"], "question_slug": a["question_slug"],
        "original_text": a["original_text"],
        "annotator_A_stance": sa.where(sa != "", "(blank)"), "annotator_B_stance": sb.where(sb != "", "(blank)"),
        "annotator_A_notes": a["review_notes"], "annotator_B_notes": b["review_notes"],
    })[sa != sb]
    disagreements = disagreements.assign(adjudicated_stance="", adjudication_rationale="")
    return {
        "annotator_A_labels": _label_audit(a["stance"]),
        "annotator_B_labels": _label_audit(b["stance"]),
        "sample_sizes": excluded,
        "stance_agreement": agreement,
        "raw_agreement_rate": agreement.get("po"),
        "confusion_matrix_rows_A_cols_B": {r: {c: int(cm.loc[r, c]) for c in cm.columns} for r in cm.index},
        "free_text": free,
        "_disagreements": disagreements,
    }


def analyze(
    round_name: str,
    completed_a: Path | None = None,
    completed_b: Path | None = None,
    reference: Path | None = None,
    templates: list[Path] | None = None,
) -> dict[str, Any]:
    """Validate and analyse one round in memory. Writes nothing."""
    cfg = ROUNDS[round_name]
    completed_a = Path(completed_a or cfg["completed_a"])
    completed_b = Path(completed_b or cfg["completed_b"])
    reference = Path(reference or cfg["reference"])
    templates = [Path(t) for t in (cfg["templates"] if templates is None else templates)]
    result: dict[str, Any] = {"round": round_name, "label": cfg["label"], "errors": [], "warnings": [], "segments": {}}

    missing = [str(p) for p in (completed_a, completed_b) if not p.exists()]
    if missing:
        result["status"] = "missing_files"
        result["message"] = ("Completed annotation file(s) not found: " + "; ".join(missing)
                             + ". Finish annotation first; this tool never falls back to blank templates.")
        return result
    for p in (completed_a, completed_b):
        if p.resolve() in {t.resolve() for t in templates if t.exists()}:
            result["status"] = "invalid"
            result["errors"].append(f"{p} is a blank template, not a completed file.")
            return result

    ref, df_a, df_b = _read(reference), _read(completed_a), _read(completed_b)
    segments = cfg["segments"]
    errors, warnings = validate_completed_pair(df_a, df_b, ref, segments)
    result["errors"], result["warnings"] = errors, warnings
    result["inputs_sha256"] = {"reference": _sha(reference), "completed_a": _sha(completed_a), "completed_b": _sha(completed_b)}
    if errors:
        result["status"] = "invalid"
        return result

    masks = segment_masks(ref, segments)
    any_label = False
    for seg in segments:
        m = segment_metrics(df_a, df_b, masks[seg["name"]])
        m["required"] = seg["required"]
        m["description"] = seg["description"]
        any_label |= m["annotator_A_labels"]["valid"] + m["annotator_B_labels"]["valid"] > 0
        result["segments"][seg["name"]] = m
    if not any_label:
        result["status"] = "pending_no_annotations"
        result["message"] = "Completed files contain no valid stance labels yet; no agreement was computed."
        return result
    result["status"] = "ok"
    return result


# ── Output ────────────────────────────────────────────────────────────────────

def _report(result: dict[str, Any]) -> str:
    L = [f"# {result['label']}", ""]
    L.append("Reliability of two independent annotators on a **stratified sample**. These figures describe annotator "
             "agreement on the selected records; they are not population prevalence or representative estimates. "
             "Stance is not sentiment. `concern` and `requested_action` are free text and get no kappa.")
    L.append("")
    if result["round"] == "expansion_v2":
        L.append("The expansion segment (350 additional records) and the pilot segment (50 pilot rows) are reported "
                 "separately and are never pooled or summed.")
        L.append("")
    for name, m in result["segments"].items():
        s = m["sample_sizes"]
        k = m["stance_agreement"]
        L += [f"## Segment `{name}` — {m['description']}", ""]
        L.append(f"- Rows in segment: {s['rows_in_segment']}")
        L.append(f"- Valid stance labels: A={m['annotator_A_labels']['valid']}, B={m['annotator_B_labels']['valid']}; "
                 f"blank A={m['annotator_A_labels']['blank']}, B={m['annotator_B_labels']['blank']}; "
                 f"invalid A={m['annotator_A_labels']['invalid']} {m['annotator_A_labels']['invalid_values']}, "
                 f"B={m['annotator_B_labels']['invalid']} {m['annotator_B_labels']['invalid_values']}")
        L.append(f"- Paired valid records used for agreement (N): **{s['paired_valid']}**; excluded: {s['excluded_total']}")
        if k["status"] == "computed":
            L.append(f"- Observed agreement Po = {k['po']:.3f}; expected Pe = {k['pe']:.3f}; Cohen's kappa = **{k['kappa']:.3f}** "
                     f"(calibration target ≥ {KAPPA_TARGET}: {'met' if k['target_met'] else 'not met'})")
        else:
            L.append(f"- Cohen's kappa: not computed — {k['message']}")
        f = m["free_text"]
        L.append(f"- Free text (descriptive only): concern exact {f['concern_exact_match']}/{f['concern_pairs']}, "
                 f"action exact {f['action_exact_match']}/{f['action_pairs']}")
        L += ["", "Confusion matrix (rows = annotator_A, columns = annotator_B):", "",
              "| A \\ B | " + " | ".join(PERMITTED_STANCES) + " |", "|---|" + "---:|" * len(PERMITTED_STANCES)]
        for r in PERMITTED_STANCES:
            L.append(f"| {r} | " + " | ".join(str(m['confusion_matrix_rows_A_cols_B'][r][c]) for c in PERMITTED_STANCES) + " |")
        L.append("")
    if result["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"][:50]]
        if len(result["warnings"]) > 50:
            L.append(f"- … {len(result['warnings']) - 50} more (see validation_issues.csv)")
        L.append("")
    L += ["## Notes", "",
          "- Disagreements are listed for human adjudication with blank `adjudicated_stance`; none were resolved automatically.",
          "- Small N makes kappa unstable; interpret it together with the confusion matrix.",
          "- Cluster assignments and model outputs played no part in this analysis."]
    return "\n".join(L) + "\n"


def write_outputs(result: dict[str, Any], output_dir: Path) -> list[Path]:
    if result.get("status") != "ok":
        raise ValueError(f"Nothing to write: status is {result.get('status')!r}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)          # refuses an existing directory
    serialisable = json.loads(json.dumps(
        {**result, "segments": {k: {kk: vv for kk, vv in v.items() if kk != "_disagreements"} for k, v in result["segments"].items()}},
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    paths = [output_dir / n for n in ("agreement_report.md", "agreement_metrics.json", "disagreements.csv", "validation_issues.csv")]
    with open(paths[0], "x", encoding="utf-8", newline="") as f:
        f.write(_report(result))
    with open(paths[1], "x", encoding="utf-8", newline="") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
        f.write("\n")
    dis = pd.concat([m["_disagreements"].assign(segment=name) for name, m in result["segments"].items()], ignore_index=True)
    with open(paths[2], "x", encoding="utf-8", newline="") as f:
        dis.to_csv(f, index=False, lineterminator="\n")
    issues = pd.DataFrame([{"severity": "error", "message": e} for e in result["errors"]]
                          + [{"severity": "warning", "message": w} for w in result["warnings"]],
                          columns=["severity", "message"])
    with open(paths[3], "x", encoding="utf-8", newline="") as f:
        issues.to_csv(f, index=False, lineterminator="\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-annotator agreement analysis (pilot or v2 expansion).")
    parser.add_argument("--round", choices=sorted(ROUNDS), required=True)
    parser.add_argument("--completed-a", type=Path)
    parser.add_argument("--completed-b", type=Path)
    parser.add_argument("--reference", type=Path, help="Override the round's reference file (testing).")
    parser.add_argument("--output-dir", type=Path, help="New directory for outputs (must not exist).")
    parser.add_argument("--check-only", action="store_true", help="Validate and print a summary; write nothing.")
    args = parser.parse_args(argv)
    result = analyze(args.round, args.completed_a, args.completed_b, args.reference)
    print(result["label"])
    status = result["status"]
    if status == "missing_files":
        print("[BLOCKED] " + result["message"], file=sys.stderr)
        return 2
    for e in result["errors"]:
        print("[ERROR]", e, file=sys.stderr)
    for w in result["warnings"][:10]:
        print("[WARN]", w)
    if status == "invalid":
        return 1
    if status == "pending_no_annotations":
        print("[PENDING] " + result["message"])
        return 0
    for name, m in result["segments"].items():
        k = m["stance_agreement"]
        print(f"  segment {name}: paired N={m['sample_sizes']['paired_valid']}/{m['sample_sizes']['rows_in_segment']}, "
              f"kappa={'n/a' if k['kappa'] is None else round(k['kappa'], 3)} ({k['status']})")
    if args.check_only:
        print("check-only: nothing written")
        return 0
    out = args.output_dir or OUTPUT_BASE / f"{args.round}_v1"
    try:
        paths = write_outputs(result, out)
    except FileExistsError:
        print(f"[STOP] Output directory already exists: {out}. Pass a new --output-dir.", file=sys.stderr)
        return 1
    print("written:", ", ".join(p.name for p in paths), "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
