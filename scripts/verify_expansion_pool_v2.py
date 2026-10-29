"""
scripts/verify_expansion_pool_v2.py
===================================
Read-only integrity and data-quality verification of the v2 annotation pool.

Usage:
    python scripts/verify_expansion_pool_v2.py            # exits 1 if any check FAILS

Checks (each returns PASS / FAIL / WARN):
    manifest hashes and row counts; pilot rows preserved verbatim and first; annotation columns blank;
    source-record uniqueness; no pilot-record overlap; exact-text duplicate policy; non-blank
    duplicate_text_group_id uniqueness; verbatim source-text provenance and eligibility; ledger
    completeness; language metadata; stratum allocation; annotator templates (if created).

Nothing is written. ``duplicate_text_group_id`` is a row-level text key built by the data audit; it
identifies identical response text, NOT the same respondent, and the pool contains no respondent IDs.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_annotation_expansion_pool_v2 as builder  # noqa: E402

ROOT = builder.ROOT
ANN_COLS = builder.ANNOTATION_COLUMNS


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _res(name: str, ok: bool, detail: str = "", warn: bool = False) -> dict[str, str]:
    return {"check": name, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": detail}


def run_checks(
    pool_path: Path = builder.POOL_PATH,
    ledger_path: Path = builder.LEDGER_PATH,
    manifest_path: Path = builder.MANIFEST_PATH,
    pilot_path: Path = builder.PILOT_PATH,
    processed_dir: Path = builder.PROCESSED,
    template_paths: tuple[Path, ...] | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    missing = [str(p) for p in (pool_path, ledger_path, manifest_path, pilot_path) if not p.exists()]
    if missing:
        return [_res("artifacts_exist", False, "missing: " + ", ".join(missing))]
    out.append(_res("artifacts_exist", True, "pool, ledger, manifest and pilot present"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool, ledger, pilot = _read(pool_path), _read(ledger_path), _read(pilot_path)

    # Manifest hashes and counts
    mism = [n for n, h in manifest.get("outputs_sha256", {}).items()
            if n in (pool_path.name, ledger_path.name) and _sha(pool_path.parent / n) != h]
    out.append(_res("manifest_output_hashes", not mism and len(manifest.get("outputs_sha256", {})) >= 2,
                    "pool and ledger SHA-256 match the manifest" if not mism else f"mismatch: {mism}"))
    in_mism = []
    for name, h in manifest.get("inputs_sha256", {}).items():
        p = pilot_path if name == pilot_path.name else processed_dir / name
        if not p.exists() or _sha(p) != h:
            in_mism.append(name)
    out.append(_res("manifest_input_hashes", not in_mism,
                    "pilot and source_clean inputs unchanged since the build" if not in_mism else f"changed/missing: {in_mism}"))
    stats = manifest.get("stats", {})
    counts_ok = (len(pool) == stats.get("pool_rows") == len(pilot) + sum(manifest.get("quotas", {}).values())
                 and len(ledger) == stats.get("ledger_rows"))
    out.append(_res("row_counts", counts_ok,
                    f"pool={len(pool)}, ledger={len(ledger)}, manifest pool={stats.get('pool_rows')}, ledger={stats.get('ledger_rows')}"))

    # Structure, pilot segment, blank labels
    n_pilot = len(pilot)
    head = pool.iloc[:n_pilot]
    pilot_ok = list(pool.columns) == list(pilot.columns) + builder.OUTPUT_EXTRA_COLUMNS and \
        head[list(pilot.columns)].reset_index(drop=True).equals(pilot.reset_index(drop=True))
    out.append(_res("pilot_rows_verbatim_and_first", pilot_ok, f"first {n_pilot} rows equal the pilot file; 2 appended columns"))
    seg_ok = pool.pilot_id.iloc[:n_pilot].str.startswith("pilot_").all() and \
        pool.pilot_id.iloc[n_pilot:].str.startswith("expansion_v2_").all() and pool.pilot_id.is_unique
    out.append(_res("segments_labelled_by_id", seg_ok, f"rows 1-{n_pilot} are pilot_*, remaining are expansion_v2_*"))
    blank = int((pool[ANN_COLS] != "").sum().sum())
    out.append(_res("annotation_columns_blank", blank == 0, f"non-blank annotation cells: {blank}"))

    # Uniqueness and overlap
    dup_rec = int(pool.duplicated(["source_dataset", "source_row_number"]).sum())
    out.append(_res("source_record_uniqueness", dup_rec == 0, f"repeated (source_dataset, source_row_number): {dup_rec}"))
    new = pool.iloc[n_pilot:]
    overlap = set(zip(new.source_dataset, new.source_row_number)) & set(zip(pilot.source_dataset, pilot.source_row_number))
    out.append(_res("no_pilot_record_overlap", not overlap, f"new rows sharing a pilot source record: {len(overlap)}"))
    keys = pool.original_text.map(_key)
    out.append(_res("exact_text_duplicate_policy", keys.is_unique, f"repeated normalised texts in pool: {int(keys.duplicated().sum())}"))
    groups = pool.duplicate_text_group_id.str.strip()
    out.append(_res("duplicate_group_unique", groups[groups != ""].is_unique,
                    "non-blank duplicate_text_group_id appears at most once (an identical-text key, not a respondent ID)"))

    # Provenance, eligibility
    sources = {ds: _read(processed_dir / name) for ds, name in builder.SOURCE_FILES.items()}
    cols = {
        builder.DS_IND: {s: c for s, c, _ in builder.INDIVIDUAL_FIELDS},
        builder.DS_CMP: {"comment_question_or_idea": builder.COMPILATION_TEXT_COL},
        builder.DS_QST: {s: c for s, c, _ in builder.QUESTIONNAIRE_FIELDS},
    }
    index = {ds: df.set_index("source_row_number") for ds, df in sources.items()}
    prov_bad = elig_bad = 0
    for _, r in pool.iterrows():
        try:
            src = index[r.source_dataset].loc[r.source_row_number]
            cell = src[cols[r.source_dataset][r.question_slug]]
        except KeyError:
            prov_bad += 1
            continue
        if r.original_text != cell.strip():
            prov_bad += 1
        if len(r.original_text) < builder.MIN_TEXT_LEN or _key(r.original_text) in builder.GENERIC_FILLERS \
                or src["exclusion_reason"].strip():
            elig_bad += 1
    out.append(_res("source_text_provenance_verbatim", prov_bad == 0, f"rows not matching source cell (trimmed): {prov_bad}"))
    out.append(_res("eligibility_rules", elig_bad == 0, f"rows violating length/filler/exclusion rules: {elig_bad}"))

    # Ledger completeness
    nonblank = 0
    for ds, cmap in cols.items():
        for col in cmap.values():
            nonblank += int((sources[ds][col].str.strip() != "").sum())
    ledger_ok = (len(ledger) == nonblank
                 and not ledger.duplicated(["source_dataset", "source_row_number", "question_slug"]).any()
                 and set(ledger.reason_code) <= set(builder.REASONS)
                 and (ledger.reason_code == "selected").sum() == len(new)
                 and set(ledger[ledger.reason_code == "selected"].pool_id) == set(new.pilot_id))
    out.append(_res("ledger_completeness", ledger_ok,
                    f"ledger={len(ledger)} vs non-blank designed cells={nonblank}; selected={(ledger.reason_code == 'selected').sum()}"))

    # Language metadata
    lang_ok = set(pool.source_language) <= {"English", "French", "Unknown"} and (pool.language_basis != "").all()
    qbasis = pool[pool.source_dataset == builder.DS_QST].language_basis.str.contains("survey language").all()
    out.append(_res("language_metadata", bool(lang_ok and qbasis),
                    f"values={sorted(set(pool.source_language))}; questionnaire basis states survey language: {bool(qbasis)}"))
    oos = ledger[ledger.reason_code == "language_out_of_scope"]
    out.append(_res("language_scope_exclusions_ledgered",
                    set(oos.source_language) <= {"Arabic", "Japanese"} and
                    not (set(zip(oos.source_dataset, oos.source_row_number)) & set(zip(pool.source_dataset, pool.source_row_number))),
                    f"out-of-scope entries ledgered: {len(oos)}"))

    # Stratum allocation
    alloc = new.groupby("source_dataset").size().to_dict()
    quota_ok = alloc == manifest.get("quotas")
    out.append(_res("dataset_quotas", quota_ok, f"achieved={alloc}; quotas={manifest.get('quotas')}"))
    strata_ok = True
    detail = []
    for ds in (builder.DS_IND, builder.DS_QST):
        final = manifest["selection_log"][ds]["final_quota"]
        got = new[new.source_dataset == ds].question_slug.value_counts().to_dict()
        ok = {k: v for k, v in final.items() if v} == got
        strata_ok &= ok
        detail.append(f"{ds}: {'ok' if ok else 'MISMATCH'}")
    cmp_final = manifest["selection_log"][builder.DS_CMP]["final_quota"]
    themes = (new[new.source_dataset == builder.DS_CMP].context.str.extract(r"Theme: (.*)$")[0].str.strip()
              .replace({"": "Blank/unspecified", "Blank": "Blank/unspecified"}).value_counts().to_dict())
    ok = {k: v for k, v in cmp_final.items() if v} == themes
    strata_ok &= ok
    detail.append(f"compilation themes: {'ok' if ok else 'MISMATCH'}")
    out.append(_res("stratum_allocation_matches_manifest", strata_ok, "; ".join(detail)))
    ev = [e for log in manifest["selection_log"].values() for e in log.get("events", [])]
    out.append(_res("allocation_shortfall_events", not ev, "no shortfall/redistribution events" if not ev else f"{len(ev)} logged event(s)",
                    warn=True))

    # Annotator templates (optional)
    if template_paths is None:
        template_paths = (pool_path.parent / "expansion_v2_annotator_1.csv", pool_path.parent / "expansion_v2_annotator_2.csv")
    existing = [p for p in template_paths if p.exists()]
    if existing:
        same = all(_read(p).equals(pool) and list(_read(p).columns) == list(pool.columns) for p in existing)
        out.append(_res("annotator_templates_match_pool", same and len(existing) == len(template_paths),
                        f"{len(existing)}/{len(template_paths)} templates present; each equals the pool (header, rows, blank labels)"))
    else:
        out.append(_res("annotator_templates_match_pool", True,
                        "templates not created yet (run scripts/create_expansion_v2_annotator_templates.py)", warn=True))
    return out


def main() -> int:
    results = run_checks()
    width = max(len(r["check"]) for r in results)
    for r in results:
        print(f"[{r['status']:<4}] {r['check']:<{width}}  {r['detail']}")
    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed or warned; {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
