"""
scripts/create_annotation_expansion_pool_v2.py
==============================================
Builds a NEW, versioned 400-record annotation pool (50 verbatim pilot rows + 350 new
source records). It replaces the use of ``expansion_pool_400.csv``, which is left untouched.

Outputs (never overwritten; the builder refuses to run if any already exists):
    data/annotation/expansion_pool_400_v2.csv            pool (pilot columns + source_language + language_basis)
    data/annotation/expansion_pool_400_v2_ledger.csv     disposition of every non-blank candidate text
    data/annotation/expansion_pool_400_v2_manifest.json  seed, quotas, input/output SHA-256, counts
    reports/expansion_pool_v2_report.md                  allocation, provenance, language, limitations

Rules
-----
1. The 50 pilot rows are copied verbatim and first, in their original order.
2. Unit of uniqueness is the SOURCE RECORD ``(source_dataset, source_row_number)``. At most one
   field per record is selected. The sources contain no respondent identifier, so this does
   NOT establish that different records come from different people.
3. Records overlapping the pilot (same source-record key) are excluded.
4. Duplicate policy A: among eligible entries sharing a normalised exact-text key, one is
   retained by a deterministic rule (lowest (dataset name, numeric row number, field order));
   identical text does not prove a shared respondent. The same rule keeps at most one entry
   per non-blank ``duplicate_text_group_id``.
5. Arabic and Japanese records are excluded (English/French pool) and recorded in the ledger.
   Missing or unrecognised language is ``Unknown``; language is never inferred from text.
6. Quotas per dataset are fixed (70 / 160 / 120). Within a dataset, strata quotas are equal
   splits (compilation: capped "Not Applicable" and "Blank/unspecified" strata). Shortfalls are
   redistributed inside the same dataset and logged; if a dataset quota cannot be met the
   builder stops with an error rather than relaxing a constraint.
7. No annotation labels are created: all human annotation columns are blank strings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ANNOTATION = ROOT / "data" / "annotation"
REPORTS = ROOT / "reports"

PILOT_PATH = ANNOTATION / "pilot_annotation_50.csv"
POOL_PATH = ANNOTATION / "expansion_pool_400_v2.csv"
LEDGER_PATH = ANNOTATION / "expansion_pool_400_v2_ledger.csv"
MANIFEST_PATH = ANNOTATION / "expansion_pool_400_v2_manifest.json"
REPORT_PATH = REPORTS / "expansion_pool_v2_report.md"

BUILDER_VERSION = "2.0"
SEED = 42
MIN_TEXT_LEN = 20

DS_IND = "individual_comments_2016_18"
DS_CMP = "consultation_compilation_2017_18"
DS_QST = "get_involved_questionnaire_2017_18"
QUOTAS = {DS_IND: 70, DS_CMP: 160, DS_QST: 120}

SOURCE_FILES = {
    DS_IND: "individual_comments_2016_18_source_clean.csv",
    DS_CMP: "consultation_compilation_2017_18_source_clean.csv",
    DS_QST: "get_involved_questionnaire_2017_18_source_clean.csv",
}
NATIVE_ID_COLUMN = {DS_IND: "row_rang_e", DS_CMP: "id", DS_QST: "unique_identifier"}

GENERIC_FILLERS = frozenset({
    "none", "n/a", "na", ".", "thanks", "thank you",
    "no comment", "all of the above", "same as above",
    "non", "aucun", "aucune",
})
OUT_OF_SCOPE_LANGUAGES = {"arabic": "Arabic", "japanese": "Japanese"}

COMPILATION_WORDING = (
    "Comment, Question or Idea (In source language) / "
    "Commentaire, question ou idée (dans la langue de départ)"
)
CMP_CAPPED_STRATA = frozenset({"Not Applicable", "Blank/unspecified"})
Q19_COL = "q19_the_sector_or_group_that_i_identify_most_closely_with_is"

OUTPUT_EXTRA_COLUMNS = ["source_language", "language_basis"]
ANNOTATION_COLUMNS = ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]

REASONS = {
    "selected": "Selected for the expansion pool.",
    "pilot_item": "This exact record and field is one of the 50 pilot rows (kept verbatim at the top of the pool).",
    "pilot_record_overlap": "Same source record as a pilot row; at most one field per source record is annotated.",
    "pilot_text_overlap": "Normalised text equals a pilot row's text.",
    "pilot_group_overlap": "Row shares a non-blank duplicate_text_group_id with a pilot row.",
    "source_exclusion_reason": "Source row carries a non-blank exclusion_reason.",
    "blank_row_marker": "Individual-comments row has a blank row_rang_e marker (trailing blank-row artefact).",
    "language_out_of_scope": "Source language is outside the English/French annotation scope.",
    "too_short": "Text shorter than the minimum length after trimming.",
    "generic_filler": "Generic filler text.",
    "duplicate_text_key_not_retained": "Normalised exact-text key already retained from another entry (duplicate policy A).",
    "duplicate_group_not_retained": "Non-blank duplicate_text_group_id already retained from another entry (duplicate policy A).",
    "same_record_other_field_selected": "Another field of the same source record was selected.",
    "eligible_not_drawn": "Eligible but not drawn: stratum quota already filled.",
}


class AllocationError(RuntimeError):
    """A dataset quota cannot be met without relaxing a constraint."""


# ── Imports from the pilot builder (single source of truth for field metadata) ───

def _load_pilot_module():
    spec = importlib.util.spec_from_file_location("_policylens_pilot_builder", ROOT / "scripts" / "create_annotation_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PILOT = _load_pilot_module()
INDIVIDUAL_FIELDS = list(_PILOT.INDIVIDUAL_FIELDS)                     # (slug, column, wording)
QUESTIONNAIRE_FIELDS = [f for f in _PILOT.QUESTIONNAIRE_FIELDS if f[0] in ("q8", "q14", "q18")]
COMPILATION_TEXT_COL = _PILOT.COMPILATION_TEXT_COL
COMPILATION_CTX_COL = _PILOT.COMPILATION_CTX_COL
COMPILATION_THEME_COL = _PILOT.COMPILATION_THEME_COL
FIELD_ORDER = {
    DS_IND: [f[0] for f in INDIVIDUAL_FIELDS],
    DS_CMP: ["comment_question_or_idea"],
    DS_QST: [f[0] for f in QUESTIONNAIRE_FIELDS],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_key(text: str) -> str:
    """NFC, trim, casefold, collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_language(raw: str, dataset: str) -> tuple[str, str]:
    """Return (language label, language_basis) from SOURCE METADATA only; never from text."""
    column = "language" if dataset == DS_QST else "source_language"
    base = f"source column `{column}`"
    if dataset == DS_QST:
        base += " (questionnaire/survey language; not necessarily the language of the comment)"
    value = raw.strip()
    low = value.casefold()
    if low == "":
        return "Unknown", f"{base}; raw value blank -> Unknown"
    if low.startswith("english"):
        return "English", base
    if low.startswith("french"):
        return "French", base
    if low in OUT_OF_SCOPE_LANGUAGES:
        return OUT_OF_SCOPE_LANGUAGES[low], base
    return "Unknown", f"{base}; raw value {value!r} not recognised -> Unknown"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _row_num(value: str) -> int:
    return int(value)


def context_for(dataset: str, row: pd.Series) -> str:
    """Context strings identical in format to scripts/create_annotation_pilot.py."""
    if dataset == DS_IND:
        return f"Event: {row.get('event', '')} | Sector: {row.get('sector', '')}"
    if dataset == DS_CMP:
        return (
            f"Event: {row.get('event_v_nement', '')} | "
            f"Source: {row.get('source', '')} | "
            f"Prompt: {row.get(COMPILATION_CTX_COL, '')} | "
            f"Theme: {row.get(COMPILATION_THEME_COL, '')}"
        )
    return f"Language: {row.get('language', '')} | Sector: {row.get(Q19_COL, '')}"


def build_entries(sources: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[str, str]], Counter]:
    """All non-blank candidate texts in the designed fields, plus row-level language and blank-cell counts."""
    entries: list[dict[str, Any]] = []
    row_language: dict[tuple[str, str], tuple[str, str]] = {}
    blank_cells: Counter = Counter()

    def add(dataset, row, slug, column, wording, stratum, extra_reason=""):
        text = row[column]
        lang, basis = row_language[(dataset, row["source_row_number"])]
        if text.strip() == "":
            blank_cells[(dataset, slug)] += 1
            return
        entries.append({
            "source_dataset": dataset,
            "source_row_number": row["source_row_number"],
            "question_slug": slug,
            "question_wording": wording,
            "raw_text": text,
            "native_id_column": NATIVE_ID_COLUMN[dataset],
            "native_id": row.get(NATIVE_ID_COLUMN[dataset], ""),
            "stratum": stratum,
            "source_language": lang,
            "language_basis": basis,
            "duplicate_text_group_id": row.get("duplicate_text_group_id", "").strip(),
            "exclusion_reason": row.get("exclusion_reason", "").strip(),
            "source_file": row.get("source_file", ""),
            "context": context_for(dataset, row),
            "extra_reason": extra_reason,
        })

    ind = sources[DS_IND]
    for _, row in ind.iterrows():
        row_language[(DS_IND, row["source_row_number"])] = normalize_language(row.get("source_language", ""), DS_IND)
    for _, row in ind.iterrows():
        marker = "blank_row_marker" if row.get("row_rang_e", "").strip() == "" else ""
        for slug, column, wording in INDIVIDUAL_FIELDS:
            add(DS_IND, row, slug, column, wording, slug, marker)

    cmp_ = sources[DS_CMP]
    for _, row in cmp_.iterrows():
        row_language[(DS_CMP, row["source_row_number"])] = normalize_language(row.get("source_language", ""), DS_CMP)
    for _, row in cmp_.iterrows():
        theme = row.get(COMPILATION_THEME_COL, "").strip()
        stratum = "Blank/unspecified" if theme in ("", "Blank") else theme
        add(DS_CMP, row, "comment_question_or_idea", COMPILATION_TEXT_COL, COMPILATION_WORDING, stratum)

    qst = sources[DS_QST]
    for _, row in qst.iterrows():
        row_language[(DS_QST, row["source_row_number"])] = normalize_language(row.get("language", ""), DS_QST)
    for _, row in qst.iterrows():
        for slug, column, wording in QUESTIONNAIRE_FIELDS:
            add(DS_QST, row, slug, column, wording, slug)

    return entries, row_language, blank_cells


# ── Quota allocation and one-field-per-record selection ───────────────────────

def allocate_quotas(total: int, available: dict[str, int], capped: frozenset[str] = frozenset()) -> dict[str, int]:
    """Equal split; remainder (and any availability shortfall) goes round-robin to non-capped strata by availability."""
    strata = sorted(available)
    if not strata:
        raise AllocationError("No strata available.")
    base = total // len(strata)
    quota = {s: min(base, available[s]) for s in strata}
    remaining = total - sum(quota.values())
    while remaining > 0:
        eligible = sorted((s for s in strata if s not in capped and quota[s] < available[s]),
                          key=lambda s: (-available[s], s))
        if not eligible:
            raise AllocationError(
                f"Cannot allocate {total} records: strata availability {available} (capped: {sorted(capped)}) is insufficient."
            )
        for s in eligible:
            if remaining == 0:
                break
            quota[s] += 1
            remaining -= 1
    return quota


def _try_assign(rec, visited, options, assigned, quota, where):
    for s in options[rec]:
        if s in visited:
            continue
        visited.add(s)
        if len(assigned[s]) < quota.get(s, 0):
            assigned[s].append(rec)
            where[rec] = s
            return True
        for other in list(assigned[s]):
            if _try_assign(other, visited, options, assigned, quota, where):
                assigned[s].remove(other)
                assigned[s].append(rec)
                where[rec] = s
                return True
    return False


def match_records(order: list, options: dict, quota: dict[str, int]) -> dict[str, list]:
    """Capacity-constrained bipartite matching: each record to at most one of its candidate strata."""
    assigned: dict[str, list] = {s: [] for s in quota}
    where: dict = {}
    target = sum(quota.values())
    filled = 0
    for rec in order:
        if _try_assign(rec, set(), options, assigned, quota, where):
            filled += 1
            if filled >= target:
                break
    return assigned


def select_one_field_per_record(
    options: dict,
    total: int,
    available_entries: dict[str, int],
    capped: frozenset[str],
    rng: random.Random,
) -> tuple[dict, dict[str, Any]]:
    """Return ({record: stratum}, allocation log). Stops with AllocationError instead of relaxing constraints."""
    order = sorted(options)
    rng.shuffle(order)
    initial = allocate_quotas(total, available_entries, capped)
    quota = dict(initial)
    equal_share = total // len(available_entries)
    remainder = total - equal_share * len(available_entries)
    events: list[str] = []
    for stratum, count in sorted(available_entries.items()):
        if count < equal_share:
            events.append(
                f"stratum {stratum!r} has {count} eligible entries, below its equal share {equal_share}; "
                f"shortfall {equal_share - count} redistributed within the dataset"
            )
    for _ in range(20):
        assigned = match_records(order, options, quota)
        filled = {s: len(assigned[s]) for s in quota}
        if sum(filled.values()) >= total:
            break
        short = {s: quota[s] - filled[s] for s in quota if filled[s] < quota[s]}
        shortfall = sum(short.values())
        for s in short:
            quota[s] = filled[s]
        spare = {s: available_entries[s] - quota[s] for s in quota if s not in capped and available_entries[s] > quota[s]}
        events.append(f"record-level shortfall {short}; redistributing {shortfall} within the dataset")
        moved = 0
        while moved < shortfall and spare:
            for s in sorted(spare, key=lambda x: (-spare[x], x)):
                if moved >= shortfall:
                    break
                if spare[s] > 0:
                    quota[s] += 1
                    spare[s] -= 1
                    moved += 1
            spare = {s: v for s, v in spare.items() if v > 0}
        if moved < shortfall:
            raise AllocationError(f"Cannot fill {total} records; unresolved shortfall {shortfall - moved}. Events: {events}")
    else:
        raise AllocationError(f"Quota redistribution did not converge: {events}")
    assigned = match_records(order, options, quota)
    if sum(len(v) for v in assigned.values()) != total:
        raise AllocationError(f"Selected {sum(len(v) for v in assigned.values())} records, expected {total}. Events: {events}")
    selection = {rec: s for s, recs in assigned.items() for rec in recs}
    return selection, {
        "equal_share": equal_share,
        "remainder_distributed": remainder,
        "initial_quota": initial,
        "final_quota": quota,
        "events": events,
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_expansion_pool_v2(
    processed_dir: Path = PROCESSED,
    pilot_path: Path = PILOT_PATH,
    pool_path: Path = POOL_PATH,
    ledger_path: Path = LEDGER_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    quotas: dict[str, int] | None = None,
    seed: int = SEED,
    dry_run: bool = False,
    expected_pilot_rows: int | None = 50,
) -> dict[str, Any]:
    quotas = dict(QUOTAS if quotas is None else quotas)
    outputs = [pool_path, ledger_path, manifest_path, report_path]
    if not dry_run:
        existing = [str(p) for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output(s): " + ", ".join(existing)
                + ". Choose a new version name or remove them deliberately."
            )

    pilot = _read(pilot_path)
    if expected_pilot_rows is not None and len(pilot) != expected_pilot_rows:
        raise ValueError(f"Pilot has {len(pilot)} rows, expected {expected_pilot_rows}.")
    sources = {ds: _read(processed_dir / name) for ds, name in SOURCE_FILES.items()}
    for ds, df in sources.items():
        if not df["source_row_number"].is_unique:
            raise ValueError(f"{ds}: source_row_number is not unique.")

    entries, row_language, blank_cells = build_entries(sources)
    entry_index = {(e["source_dataset"], e["source_row_number"], e["question_slug"]): e for e in entries}

    # Pilot provenance check and pilot key sets.
    pilot_items, pilot_records, pilot_keys, pilot_groups = set(), set(), set(), set()
    for _, p in pilot.iterrows():
        item = (p["source_dataset"], p["source_row_number"], p["question_slug"])
        src = entry_index.get(item)
        if src is None or src["raw_text"].strip() != p["original_text"]:
            raise ValueError(f"Pilot row {p['pilot_id']} does not match its source field verbatim.")
        pilot_items.add(item)
        pilot_records.add((p["source_dataset"], p["source_row_number"]))
        pilot_keys.add(normalize_key(p["original_text"]))
        if p["duplicate_text_group_id"].strip():
            pilot_groups.add(p["duplicate_text_group_id"].strip())
    wording_by_slug = {(d, s): w for d, s, w in zip(pilot["source_dataset"], pilot["question_slug"], pilot["question_wording"])}
    for e in entries:
        w = wording_by_slug.get((e["source_dataset"], e["question_slug"]))
        if w is not None and w != e["question_wording"]:
            raise ValueError(f"Question wording for {(e['source_dataset'], e['question_slug'])} differs from the pilot's.")

    field_rank = {(ds, slug): i for ds, order in FIELD_ORDER.items() for i, slug in enumerate(order)}
    for e in entries:
        e["text"] = e["raw_text"].strip()
        e["key"] = normalize_key(e["raw_text"])
        e["order"] = (e["source_dataset"], _row_num(e["source_row_number"]), field_rank[(e["source_dataset"], e["question_slug"])])
        e["disposition"] = ""
        e["reason_code"] = ""
        e["retained_ref"] = ""
    entries.sort(key=lambda e: e["order"])

    def mark(e, code, ref=""):
        e["disposition"] = "selected" if code == "selected" else "excluded" if not code.startswith("eligible") else "not_drawn"
        e["reason_code"] = code
        e["retained_ref"] = ref

    # Stage 1-5: source/language/eligibility/pilot rules (first applicable rule wins).
    for e in entries:
        item = (e["source_dataset"], e["source_row_number"], e["question_slug"])
        record = (e["source_dataset"], e["source_row_number"])
        if e["exclusion_reason"]:
            mark(e, "source_exclusion_reason")
        elif e["extra_reason"] == "blank_row_marker":
            mark(e, "blank_row_marker")
        elif e["source_language"] in OUT_OF_SCOPE_LANGUAGES.values():
            mark(e, "language_out_of_scope")
        elif e["key"] in GENERIC_FILLERS:
            mark(e, "generic_filler")
        elif len(e["text"]) < MIN_TEXT_LEN:
            mark(e, "too_short")
        elif item in pilot_items:
            mark(e, "pilot_item")
        elif record in pilot_records:
            mark(e, "pilot_record_overlap")
        elif e["key"] in pilot_keys:
            mark(e, "pilot_text_overlap")
        elif e["duplicate_text_group_id"] and e["duplicate_text_group_id"] in pilot_groups:
            mark(e, "pilot_group_overlap")

    # Stage 6: duplicate policy A (deterministic retention: first by (dataset, numeric row, field order)).
    retained_key: dict[str, dict] = {}
    for e in entries:
        if e["reason_code"]:
            continue
        first = retained_key.setdefault(e["key"], e)
        if first is not e:
            mark(e, "duplicate_text_key_not_retained", f"{first['source_dataset']}|{first['source_row_number']}|{first['question_slug']}")
    retained_group: dict[str, dict] = {}
    for e in entries:
        if e["reason_code"] or not e["duplicate_text_group_id"]:
            continue
        first = retained_group.setdefault(e["duplicate_text_group_id"], e)
        if first is not e:
            mark(e, "duplicate_group_not_retained", f"{first['source_dataset']}|{first['source_row_number']}|{first['question_slug']}")

    # Stage 7: per-dataset stratified, one-field-per-record selection.
    rng = random.Random(seed)
    eligible = [e for e in entries if not e["reason_code"]]
    selection_log: dict[str, Any] = {}
    selected_entries: list[dict] = []
    preflight: dict[str, Any] = {}
    for ds in (DS_CMP, DS_QST, DS_IND):  # alphabetical, deterministic
        ds_entries = [e for e in eligible if e["source_dataset"] == ds]
        options: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e in ds_entries:
            options[(ds, e["source_row_number"])].append(e["stratum"])
        available = Counter(e["stratum"] for e in ds_entries)
        if ds in (DS_IND, DS_QST):
            for slug in FIELD_ORDER[ds]:
                available.setdefault(slug, 0)
        capped = CMP_CAPPED_STRATA if ds == DS_CMP else frozenset()
        preflight[ds] = {
            "eligible_entries": len(ds_entries),
            "eligible_records": len(options),
            "quota": quotas[ds],
            "stratum_entry_availability": dict(sorted(available.items())),
        }
        if len(options) < quotas[ds]:
            raise AllocationError(f"{ds}: only {len(options)} eligible records for quota {quotas[ds]}.")
        for rec in options:
            options[rec] = sorted(set(options[rec]))
        sel, log = select_one_field_per_record(dict(options), quotas[ds], dict(available), capped, rng)
        selection_log[ds] = log
        for e in ds_entries:
            rec = (ds, e["source_row_number"])
            if rec in sel and sel[rec] == e["stratum"]:
                selected_entries.append(e)
                mark(e, "selected")
        for e in ds_entries:
            if e["reason_code"] == "selected":
                continue
            rec = (ds, e["source_row_number"])
            mark(e, "same_record_other_field_selected" if rec in sel else "eligible_not_drawn")
            if e["reason_code"] == "eligible_not_drawn":
                e["disposition"] = "not_drawn"
            else:
                e["disposition"] = "excluded"

    if not dry_run and len({(e["source_dataset"], e["source_row_number"]) for e in selected_entries}) != len(selected_entries):
        raise AssertionError("Internal error: duplicate source record selected.")

    # Assemble the pool.
    selected_entries.sort(key=lambda e: (e["source_dataset"], _row_num(e["source_row_number"])))
    new_rows = []
    for i, e in enumerate(selected_entries, start=1):
        e["selected_id"] = f"expansion_v2_{i:03d}"
        new_rows.append({
            "pilot_id": e["selected_id"],
            "source_dataset": e["source_dataset"],
            "source_record_id": e["source_row_number"],
            "source_file": e["source_file"],
            "source_row_number": e["source_row_number"],
            "source_field": e["question_slug"],
            "question_slug": e["question_slug"],
            "question_wording": e["question_wording"],
            "question_context": e["context"],
            "context": e["context"],
            "duplicate_text_group_id": e["duplicate_text_group_id"],
            "original_text": e["text"],
            **{c: "" for c in ANNOTATION_COLUMNS},
            "source_language": e["source_language"],
            "language_basis": e["language_basis"],
        })
    pilot_cols = list(pilot.columns)
    pilot_out = pilot.copy()
    langs = [row_language[(d, r)] for d, r in zip(pilot["source_dataset"], pilot["source_row_number"])]
    pilot_out["source_language"] = [l[0] for l in langs]
    pilot_out["language_basis"] = [l[1] for l in langs]
    pool = pd.concat([pilot_out, pd.DataFrame(new_rows, columns=pilot_cols + OUTPUT_EXTRA_COLUMNS)], ignore_index=True)
    expected_total = len(pilot) + sum(quotas.values())
    if len(pool) != expected_total:
        raise AssertionError(f"Pool has {len(pool)} rows, expected {expected_total}.")

    # Ledger.
    ledger_rows = []
    pilot_id_by_item = {(p["source_dataset"], p["source_row_number"], p["question_slug"]): p["pilot_id"] for _, p in pilot.iterrows()}
    for e in entries:
        item = (e["source_dataset"], e["source_row_number"], e["question_slug"])
        pool_id = e.get("selected_id", "") if e["reason_code"] == "selected" else pilot_id_by_item.get(item, "")
        ledger_rows.append({
            "source_dataset": e["source_dataset"],
            "source_row_number": e["source_row_number"],
            "question_slug": e["question_slug"],
            "native_id_column": e["native_id_column"],
            "native_id": e["native_id"],
            "stratum": e["stratum"],
            "source_language": e["source_language"],
            "text_length": len(e["text"]),
            "text_key_sha256_16": key_digest(e["key"]),
            "duplicate_text_group_id": e["duplicate_text_group_id"],
            "disposition": e["disposition"] or "excluded",
            "reason_code": e["reason_code"],
            "reason": REASONS[e["reason_code"]],
            "retained_entry": e["retained_ref"],
            "pool_id": pool_id,
        })
    ledger = pd.DataFrame(ledger_rows)

    stats = _summarise(entries, pool, ledger, preflight, selection_log, blank_cells, quotas, seed, sources, pilot)
    result = {"pool": pool, "ledger": ledger, "stats": stats, "selection_log": selection_log}
    if dry_run:
        return result

    pool_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "x", encoding="utf-8", newline="") as f:
        pool.to_csv(f, index=False, lineterminator="\n")
    with open(ledger_path, "x", encoding="utf-8", newline="") as f:
        ledger.to_csv(f, index=False, lineterminator="\n")
    manifest = {
        "builder": "scripts/create_annotation_expansion_pool_v2.py",
        "builder_version": BUILDER_VERSION,
        "seed": seed,
        "quotas": quotas,
        "min_text_length": MIN_TEXT_LEN,
        "duplicate_policy": "A: retain one entry per normalised exact-text key and per non-blank duplicate_text_group_id; "
                            "rule = lowest (dataset name, numeric source_row_number, field order)",
        "text_key": "NFC + trim + casefold + collapse whitespace",
        "uniqueness_unit": "(source_dataset, source_row_number); not respondent-level",
        "inputs_sha256": {
            "pilot_annotation_50.csv": sha256_file(pilot_path),
            **{name: sha256_file(processed_dir / name) for name in SOURCE_FILES.values()},
        },
        "outputs_sha256": {pool_path.name: sha256_file(pool_path), ledger_path.name: sha256_file(ledger_path)},
        "stats": stats,
        "selection_log": selection_log,
    }
    with open(manifest_path, "x", encoding="utf-8", newline="") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    with open(report_path, "x", encoding="utf-8", newline="") as f:
        f.write(_report(stats, selection_log, pool_path, ledger_path, manifest_path))
    return result


def _summarise(entries, pool, ledger, preflight, selection_log, blank_cells, quotas, seed, sources, pilot) -> dict[str, Any]:
    new = pool.iloc[len(pilot):]
    stage = ledger.groupby(["source_dataset", "reason_code"]).size().unstack(fill_value=0)
    return {
        "seed": seed,
        "pool_rows": int(len(pool)),
        "pilot_rows": int(len(pilot)),
        "new_rows": int(len(new)),
        "quota": quotas,
        "achieved": {k: int(v) for k, v in new.groupby("source_dataset").size().items()},
        "preflight": preflight,
        "ledger_rows": int(len(ledger)),
        "ledger_by_dataset_reason": {ds: {r: int(c) for r, c in row.items() if c} for ds, row in stage.iterrows()},
        "blank_cells_not_in_ledger": {f"{d}|{s}": int(c) for (d, s), c in sorted(blank_cells.items())},
        "new_rows_by_stratum": {
            ds: {k: int(v) for k, v in grp.groupby("question_slug").size().items()}
            for ds, grp in new.groupby("source_dataset")
        },
        "new_rows_by_theme": {
            k: int(v) for k, v in
            new[new["source_dataset"] == DS_CMP]["context"].str.extract(r"Theme: (.*)$")[0].str.strip().replace({"": "Blank/unspecified", "Blank": "Blank/unspecified"}).value_counts().items()
        },
        "language_new_rows": {ds: {k: int(v) for k, v in grp["source_language"].value_counts().items()}
                              for ds, grp in new.groupby("source_dataset")},
        "language_pilot_rows": {k: int(v) for k, v in pool.iloc[:len(pilot)]["source_language"].value_counts().items()},
        "source_rows": {ds: int(len(df)) for ds, df in sources.items()},
    }


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _report(stats, selection_log, pool_path, ledger_path, manifest_path) -> str:
    L: list[str] = []
    L.append("# PolicyLens expansion pool v2 report\n")
    L.append("Status: **annotation pool prepared; no human labels exist**. All annotation columns are blank.\n")
    L.append("## Files\n")
    L.append(f"- Pool: `{_rel(pool_path)}` ({stats['pool_rows']} rows = {stats['pilot_rows']} pilot + {stats['new_rows']} new)")
    L.append(f"- Ledger: `{_rel(ledger_path)}` ({stats['ledger_rows']} entries)")
    L.append(f"- Manifest: `{_rel(manifest_path)}` (seed, quotas, SHA-256 of inputs/outputs)")
    L.append("- The original `data/annotation/expansion_pool_400.csv` and its annotator copies are unchanged and superseded by this pool.\n")
    L.append("## Allocation\n")
    L.append("| Dataset | Quota | Achieved | Eligible records | Eligible entries |")
    L.append("|---|---:|---:|---:|---:|")
    for ds, q in stats["quota"].items():
        pf = stats["preflight"][ds]
        L.append(f"| `{ds}` | {q} | {stats['achieved'].get(ds, 0)} | {pf['eligible_records']} | {pf['eligible_entries']} |")
    L.append("\nWithin-dataset strata (quota = equal split; compilation caps \"Not Applicable\" and \"Blank/unspecified\" at the base share, "
             "and gives the remainder to the largest named themes):\n")
    for ds, log in selection_log.items():
        L.append(f"**`{ds}`**\n")
        L.append(f"Equal share per stratum: {log['equal_share']}; remainder distributed to non-capped strata: {log['remainder_distributed']}.\n")
        L.append("| Stratum | Eligible entries | Initial quota | Final quota |")
        L.append("|---|---:|---:|---:|")
        avail = stats["preflight"][ds]["stratum_entry_availability"]
        for s in sorted(avail):
            L.append(f"| {s} | {avail[s]} | {log['initial_quota'].get(s, 0)} | {log['final_quota'].get(s, 0)} |")
        if log["events"]:
            L.append("\nShortfall / redistribution events:")
            L.extend(f"- {ev}" for ev in log["events"])
        else:
            L.append("\nNo shortfall; no redistribution needed.")
        L.append("")
    L.append("## Exclusion ledger summary (entries by reason)\n")
    L.append("| Dataset | Reason | Entries |")
    L.append("|---|---|---:|")
    for ds, reasons in stats["ledger_by_dataset_reason"].items():
        for r, c in sorted(reasons.items()):
            L.append(f"| `{ds}` | {r} | {c} |")
    L.append("\nBlank cells in the designed fields are not ledgered row by row; their counts are in the manifest "
             "(`stats.blank_cells_not_in_ledger`).\n")
    L.append("## Language\n")
    L.append("`source_language` comes from source metadata only; text language is never inferred, and unrecognised or blank values are `Unknown`. "
             "`language_basis` records the source column.\n")
    L.append("| Dataset | New-row language counts |")
    L.append("|---|---|")
    for ds, d in stats["language_new_rows"].items():
        L.append(f"| `{ds}` | {json.dumps(d)} |")
    L.append(f"\nPilot rows (50): {json.dumps(stats['language_pilot_rows'])}\n")
    L.append("## Provenance and context\n")
    L.append("- Pilot rows are copied verbatim and first; new rows use the pilot's context format and question wording.")
    L.append("- `original_text` is the source cell trimmed of leading/trailing whitespace, as in the pilot.")
    L.append("- `source_record_id` equals `source_row_number`; native IDs (`row_rang_e`, `id`, `unique_identifier`) are in the ledger.\n")
    L.append("## Limitations\n")
    L.append("- **Not respondent-unique.** The sources contain no respondent identifiers. Uniqueness is only at the source-record level "
             "`(source_dataset, source_row_number)`. The same person may have submitted several records, and one card may appear in both the "
             "individual-comments and compilation files under different text.")
    L.append("- **Text duplicates:** matching normalised text does not prove a shared respondent. One occurrence per key is retained by a "
             "deterministic rule; others are in the ledger.")
    L.append("- **Questionnaire language** is the survey language the respondent used to complete the form, not necessarily the language of the comment.")
    L.append("- **Stratified, not representative.** Strata quotas are equal splits chosen for coverage, so the pool does not estimate "
             "population frequencies without weights; \"Not Applicable\" and blank themes are deliberately capped.")
    L.append("- **French coverage is incidental.** No language quota was set. French counts follow each dataset's availability "
             "(individual comments have very few French records).")
    L.append("- Arabic and Japanese records are excluded from this English/French pool, not from the corpus.")
    L.append("- Near-duplicate (non-exact) texts are not removed.\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true", help="Compute and print counts; write nothing.")
    args = parser.parse_args(argv)
    try:
        result = build_expansion_pool_v2(dry_run=args.dry_run)
    except (FileExistsError, AllocationError, ValueError) as exc:
        print(f"[STOP] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    s = result["stats"]
    print(json.dumps({k: s[k] for k in ("pool_rows", "pilot_rows", "new_rows", "quota", "achieved", "ledger_rows")}, indent=2))
    print(json.dumps(s["ledger_by_dataset_reason"], indent=2))
    print("dry run: nothing written" if args.dry_run else f"written: {POOL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
