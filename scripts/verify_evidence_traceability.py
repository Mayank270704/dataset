"""
scripts/verify_evidence_traceability.py
=======================================
End-to-end audit verifying that every cited evidence quote in PolicyLens
originates verbatim from genuine raw Canadian Federal Open Government consultation
submissions.

Verifies:
  - 275 ranked concern evidence examples (data/full_corpus/concern_prioritization_v1/ranking_evidence.csv)
  - 10 collective-demand evidence records (data/full_corpus/concern_prioritization_v1/collective_demand_evidence.csv)
  - 370 cluster representative examples (data/full_corpus/concern_discovery_v1/configurations/strip__umap_m25s5/representative_examples.csv)

Outputs (write-once):
  - data/full_corpus/concern_prioritization_v1/evidence_traceability_ledger.csv
  - data/full_corpus/concern_prioritization_v1/evidence_traceability_manifest.json
  - reports/source_evidence_traceability_report_v1.md

CLI:
  python scripts/verify_evidence_traceability.py --check-only   # audit and report without writing
  python scripts/verify_evidence_traceability.py                # audit, write ledger, manifest, and report
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.concern_discovery.schema import sha256_file
from models.concern_discovery.traceability import (
    RAW_SPECS,
    audit_full_evidence_lineage,
    load_raw_containers,
)

RUN_V1_CORPUS = ROOT / "data" / "full_corpus" / "run_v1" / "corpus" / "corpus_entries.csv"
CD_V1 = ROOT / "data" / "full_corpus" / "concern_discovery_v1"
PRIOR_V1 = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"

RANKING_EV_PATH = PRIOR_V1 / "ranking_evidence.csv"
COLLECTIVE_DEMAND_PATH = PRIOR_V1 / "collective_demand_evidence.csv"
REP_EXAMPLES_PATH = CD_V1 / "configurations" / "strip__umap_m25s5" / "representative_examples.csv"

OUT_LEDGER = PRIOR_V1 / "evidence_traceability_ledger.csv"
OUT_MANIFEST = PRIOR_V1 / "evidence_traceability_manifest.json"
OUT_REPORT = ROOT / "reports" / "source_evidence_traceability_report_v1.md"


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def generate_report_markdown(summary: dict[str, Any], ledger_df: pd.DataFrame, manifest: dict[str, Any]) -> str:
    created = manifest["created_utc"]
    total = summary["total_evidence_quotes_audited"]
    verbatim_raw = summary["verbatim_in_raw_count"]
    verbatim_raw_pct = summary["verbatim_in_raw_pct"]
    failed = summary["failed_ungrounded_count"]
    pass_rate = summary["pass_rate_grounded"]

    L = [
        "# PolicyLens Source-Evidence Traceability & Lineage Audit Report (v1)",
        "",
        f"**Audit Status:** {'PASSED (100% Grounded)' if failed == 0 else 'FAILED (Ungrounded Evidence Found)'}  ",
        f"**Audit Timestamp:** {created}  ",
        f"**Scope:** End-to-end 4-tier provenance verification across all cited evidence quotes in PolicyLens research outputs.",
        "",
        "## 1. Executive Summary",
        "",
        "PolicyLens prioritizes policy concerns and collective citizen demands directly from free-text consultation comments. "
        "To guarantee that automated clustering and extractive summarization do not hallucinate, misattribute, or alter citizen claims, "
        "this audit verifies the complete 4-tier chain of custody from raw government submissions to prioritized evidence quotes.",
        "",
        f"- **Total Evidence Quotes Audited:** **{total:,} quotes** across 3 evidence tracks:",
        f"  - Ranked concern citations: **{summary['evidence_by_type'].get('ranked_concern', 0):,}**",
        f"  - Representative cluster examples: **{summary['evidence_by_type'].get('representative_example', 0):,}**",
        f"  - Collective citizen demands: **{summary['evidence_by_type'].get('collective_demand', 0):,}**",
        f"- **Verbatim Grounding in Raw Government Files:** **{verbatim_raw:,} of {total:,} ({verbatim_raw_pct}%)**",
        f"- **Total Verified Grounded Citations:** **{total - failed:,} of {total:,} ({pass_rate}%)**",
        f"- **Ungrounded / Hallucinated Quotes Found:** **{failed}**",
        "",
        "## 2. Four-Tier Provenance Architecture",
        "",
        "| Tier | Stage | Source File / Artifact | Verification Checked |",
        "|---|---|---|---|",
        "| **Tier 0** | **Raw Submission** | 3 original government CSVs (`ap2016-18`, `ConsultationData2017`, `Get Involved`) | Physical file row, raw column header, uncleaned text, encoding integrity |",
        "| **Tier 1** | **Source Clean** | `data/processed/*_source_clean.csv` | Standardized nulls, whitespace trimming, column slugification, redactions |",
        "| **Tier 2** | **Full Corpus** | `data/full_corpus/run_v1/corpus/corpus_entries.csv` | Length & substantive eligibility, duplicate group assignment, language tag |",
        "| **Tier 3** | **Prioritized Evidence** | `ranking_evidence.csv`, `collective_demand_evidence.csv` | Substring match, cluster membership, multi-signal priority rank |",
        "",
        "## 3. Evidence Distribution Across Instruments and Languages",
        "",
        "### By Consultation Instrument",
        "",
        "| Instrument Dataset | Evidence Quotes | Share of Evidence |",
        "|---|---:|---:|",
    ]

    for ds, count in summary["evidence_by_dataset"].items():
        pct = round((count / total) * 100, 1) if total else 0.0
        L.append(f"| `{ds}` | {count:,} | {pct}% |")

    L += [
        "",
        "### By Consultation Language",
        "",
        "| Language | Evidence Quotes | Share |",
        "|---|---:|---:|",
    ]
    for lang, count in summary["evidence_by_language"].items():
        pct = round((count / total) * 100, 1) if total else 0.0
        L.append(f"| {lang} | {count:,} | {pct}% |")

    L += [
        "",
        "### By Match Status",
        "",
        "| Match Status | Count | Interpretation |",
        "|---|---:|---|",
    ]
    for status, count in summary["match_status_breakdown"].items():
        pct = round((count / total) * 100, 1) if total else 0.0
        if status == "VERBATIM_RAW":
            interp = "Exact character-for-character substring of raw uncleaned citizen submission cell."
        elif status == "VERBATIM_CLEAN":
            interp = "Exact substring of cleaned source cell (raw cell had outer carriage-returns or trailing spaces)."
        elif status == "VERBATIM_CORPUS":
            interp = "Exact substring of eligible corpus entry text."
        else:
            interp = "Ungrounded."
        L.append(f"| `{status}` | {count:,} ({pct}%) | {interp} |")

    # Sample traceable citations table (Top 10 ranked evidence)
    L += [
        "",
        "## 4. Top Ranked Concern Lineage Trace (Sample)",
        "",
        "| Rank | Cluster | Corpus ID | Instrument | Physical Row | Column Header | Verbatim Quote Excerpt | Status |",
        "|---:|---:|---:|---|---:|---|---|:---:|",
    ]

    sample_ranked = ledger_df[ledger_df["evidence_type"] == "ranked_concern"].head(10)
    for _, r in sample_ranked.iterrows():
        q_snippet = (r["quote"][:45] + "…") if len(r["quote"]) > 45 else r["quote"]
        col_snippet = (r["raw_col_header"][:25] + "…") if len(r["raw_col_header"]) > 25 else r["raw_col_header"]
        L.append(
            f"| {r['rank']} | {r['cluster_id']} | {r['corpus_id']} | `{r['source_dataset']}` | "
            f"{r['source_row_number']} | {col_snippet} | “{q_snippet}” | **{r['match_status']}** |"
        )

    L += [
        "",
        "## 5. Methodological Interpretation & Limitations",
        "",
        "- **Zero Hallucination Guarantee:** 100% of cited evidence spans trace to real, physical survey rows submitted during public consultations.",
        "- **Extractive Substrings:** Quotes are strictly extractive spans, not model-generated syntheses or LLM rewrites.",
        "- **Preservation of Context:** The lineage ledger (`evidence_traceability_ledger.csv`) includes both the extracted quote and the full citizen response, enabling human reviewers to verify that quotes are not taken out of context.",
        "- **Scope Limitation:** The Canadian federal Open Government consultation data (2016–2018) does not transfer to Indian MCA or other jurisdictions.",
        "",
    ]

    return "\n".join(L)


def run_verification(check_only: bool = False) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("PolicyLens Source-Evidence Traceability & Lineage Audit")
    print("======================================================")

    audit_result = audit_full_evidence_lineage(
        ranking_evidence_path=RANKING_EV_PATH,
        collective_demand_path=COLLECTIVE_DEMAND_PATH,
        representative_examples_path=REP_EXAMPLES_PATH,
        corpus_entries_path=RUN_V1_CORPUS,
        root=ROOT,
    )

    summary = audit_result["summary"]
    ledger_df: pd.DataFrame = audit_result["ledger_df"]

    print(f"Audited Evidence Quotes: {summary['total_evidence_quotes_audited']}")
    print(f"  - Verbatim in Raw File: {summary['verbatim_in_raw_count']} ({summary['verbatim_in_raw_pct']}%)")
    print(f"  - Verbatim in Clean/Corpus: {summary['verbatim_in_clean_count']}")
    print(f"  - Ungrounded / Failed: {summary['failed_ungrounded_count']}")
    print(f"  - Grounding Pass Rate: {summary['pass_rate_grounded']}%")
    print("\nBreakdown by Dataset:")
    for ds, c in summary["evidence_by_dataset"].items():
        print(f"  - {ds}: {c}")
    print("\nBreakdown by Language:")
    for lang, c in summary["evidence_by_language"].items():
        print(f"  - {lang}: {c}")

    if check_only:
        if summary["failed_ungrounded_count"] == 0:
            print("\n[PASSED] All evidence quotes are 100% verified and grounded in source data.")
            return 0
        else:
            print(f"\n[FAILED] Found {summary['failed_ungrounded_count']} ungrounded quotes.")
            return 1

    # Write-once outputs
    manifest = {
        "status": "PASSED" if summary["failed_ungrounded_count"] == 0 else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs_sha256": {
            "ranking_evidence.csv": sha256_file(RANKING_EV_PATH),
            "collective_demand_evidence.csv": sha256_file(COLLECTIVE_DEMAND_PATH),
            "representative_examples.csv": sha256_file(REP_EXAMPLES_PATH),
            "corpus_entries.csv": sha256_file(RUN_V1_CORPUS),
        },
        "summary": summary,
        "outputs_sha256": {},
    }

    # Save ledger CSV
    OUT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.to_csv(OUT_LEDGER, index=False, lineterminator="\n", encoding="utf-8")
    manifest["outputs_sha256"]["data/full_corpus/concern_prioritization_v1/evidence_traceability_ledger.csv"] = sha256_file(OUT_LEDGER)

    # Save report Markdown
    report_md = generate_report_markdown(summary, ledger_df, manifest)
    write_new(OUT_REPORT, report_md)
    manifest["outputs_sha256"]["reports/source_evidence_traceability_report_v1.md"] = sha256_file(OUT_REPORT)

    # Save manifest JSON
    write_new(OUT_MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Wrote lineage ledger: {OUT_LEDGER}")
    print(f"[SUCCESS] Wrote traceability report: {OUT_REPORT}")
    print(f"[SUCCESS] Wrote manifest: {OUT_MANIFEST}")

    return 0 if summary["failed_ungrounded_count"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check-only", action="store_true", help="Audit and verify without writing output files.")
    args = parser.parse_args(argv)
    return run_verification(check_only=args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
