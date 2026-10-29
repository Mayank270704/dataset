"""
tests/test_expansion_pool_v2.py
===============================
Regression tests for scripts/create_annotation_expansion_pool_v2.py and its outputs.

Part A (fixtures): the real builder runs against small SYNTHETIC source files in a temp
directory. Synthetic texts are test data only and are never written to the repository.
Part B (real outputs): invariants of data/annotation/expansion_pool_400_v2.csv and its ledger,
checked against the real source files. Skipped if the v2 outputs have not been built.

Defects guarded against (found in the audit of expansion_pool_400.csv):
  * the same (source_dataset, source_row_number) selected through several fields
  * overlap with the pilot's source records
  * repeated normalised text inside the pool
  * silent shortfalls / dropped records without a ledger entry
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION = ROOT / "data" / "annotation"
PROCESSED = ROOT / "data" / "processed"

_spec = importlib.util.spec_from_file_location(
    "expansion_v2_builder", ROOT / "scripts" / "create_annotation_expansion_pool_v2.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

DS_IND, DS_CMP, DS_QST = mod.DS_IND, mod.DS_CMP, mod.DS_QST
ANN_COLS = ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]
PILOT_COLS = [
    "pilot_id", "source_dataset", "source_record_id", "source_file", "source_row_number", "source_field",
    "question_slug", "question_wording", "question_context", "context", "duplicate_text_group_id",
    "original_text", *ANN_COLS,
]


def key_of(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


# ═══════════════════════════ Part A: synthetic fixtures ═══════════════════════════

IND_SLUGS = [f[0] for f in mod.INDIVIDUAL_FIELDS]
IND_COL = {f[0]: f[1] for f in mod.INDIVIDUAL_FIELDS}
QST_COL = {f[0]: f[1] for f in mod.QUESTIONNAIRE_FIELDS}
Q19 = mod.Q19_COL
SECTOR = "Business / Milieu des affaires"
QLANG_EN = "English /\nAnglais"
QLANG_FR = "French / français"


def ind_row(n, texts, lang="English", group="", excl="", marker=True):
    row = {
        "source_dataset": DS_IND, "source_file": "ap2016-18_individual_comments.csv",
        "source_row_number": str(n), "duplicate_text_group_id": group, "exclusion_reason": excl,
        "row_rang_e": str(100 + n) if marker else "", "event": f"Event {n}", "sector": "Academia",
        "source_language": lang,
    }
    for slug in IND_SLUGS:
        row[IND_COL[slug]] = texts.get(slug, "")
    return row


def cmp_row(n, text, theme="Open Data", lang="English", group=""):
    return {
        "source_dataset": DS_CMP, "source_file": "Consultation file.csv", "source_row_number": str(n),
        "duplicate_text_group_id": group, "exclusion_reason": "", "id": f"0.{n:03d}",
        "event_v_nement": f"Event {n}", "source": "Twitter", mod.COMPILATION_CTX_COL: f"Prompt {n}",
        mod.COMPILATION_THEME_COL: theme, "source_language": lang, mod.COMPILATION_TEXT_COL: text,
    }


def qst_row(n, q8="", q14="", q18="", lang=QLANG_EN):
    return {
        "source_dataset": DS_QST, "source_file": "Questionnaire file.csv", "source_row_number": str(n),
        "duplicate_text_group_id": "", "exclusion_reason": "", "unique_identifier": f"1100{n:05d}",
        "language": lang, Q19: SECTOR, QST_COL["q8"]: q8, QST_COL["q14"]: q14, QST_COL["q18"]: q18,
    }


def default_sources():
    ind = [ind_row(1, {"public_engagement_question": "Pilot individual text on open data portals",
                       "practical_idea": "Another field of the pilot record that must be skipped"})]
    for n in (2, 3, 4):
        ind.append(ind_row(n, {s: f"individual record {n} field {s} text about government openness" for s in IND_SLUGS}))
    cmp_ = [
        cmp_row(1, "Pilot compilation text about open data portals", "Open Data"),
        cmp_row(2, "Publish every dataset in machine readable formats", "Open Data"),
        cmp_row(3, "Publish every dataset in  machine readable FORMATS", "Open Data"),   # duplicate key of row 2
        cmp_row(4, "Open data should include legislative committee records", "Open Data"),
        cmp_row(5, "Enseigner la littératie des données dans les écoles", "Education", lang="French"),
        cmp_row(6, "Offer public workshops on how to use government datasets", "Education"),
        cmp_row(7, "Consultation sessions should publish their agendas earlier", "Not Applicable"),
        cmp_row(8, "Arabic language comment long enough to be considered", "Not Applicable", lang="Arabic"),
        cmp_row(9, "Comment from a record with unrecognised language value", "Education", lang="S"),
        cmp_row(10, "too short text", "Open Data"),
        cmp_row(11, "none", "Not Applicable"),
        cmp_row(12, "A comment whose theme cell is blank in the source", ""),
        cmp_row(13, "Another comment whose theme is the literal word Blank", "Blank"),
    ]
    qst = [
        qst_row(1, "Respondent one text eight about transparency", "Respondent one text fourteen about reasons", "Respondent one text eighteen about experience", QLANG_FR),
        qst_row(2, "Respondent two text eight about participation", "Respondent two text fourteen about satisfaction"),
        qst_row(3, "Respondent three text eight about accountability"),
        qst_row(4, q14="Texte avec accents é à ç\nsecond line kept verbatim"),
        qst_row(5, q18="Respondent five text eighteen about expertise"),
        qst_row(6, "Respondent six text eight on priorities", "Respondent six text fourteen on usability", "Respondent six text eighteen on skills"),
    ]
    pilot_specs = [(DS_IND, "1", "public_engagement_question"), (DS_CMP, "1", "comment_question_or_idea")]
    return ind, cmp_, qst, pilot_specs


DEFAULT_QUOTAS = {DS_IND: 3, DS_CMP: 6, DS_QST: 3}


def _pilot_rows(specs, ind, cmp_):
    rows = []
    for i, (ds, num, slug) in enumerate(specs, start=1):
        if ds == DS_IND:
            src = next(r for r in ind if r["source_row_number"] == num)
            text, ctx = src[IND_COL[slug]], f"Event: {src['event']} | Sector: {src['sector']}"
            wording = next(w for s, _, w in mod.INDIVIDUAL_FIELDS if s == slug)
        else:
            src = next(r for r in cmp_ if r["source_row_number"] == num)
            text = src[mod.COMPILATION_TEXT_COL]
            ctx = (f"Event: {src['event_v_nement']} | Source: {src['source']} | "
                   f"Prompt: {src[mod.COMPILATION_CTX_COL]} | Theme: {src[mod.COMPILATION_THEME_COL]}")
            wording = mod.COMPILATION_WORDING
        row = {c: "" for c in PILOT_COLS}
        row.update({
            "pilot_id": f"pilot_{i:03d}", "source_dataset": ds, "source_record_id": num, "source_file": src["source_file"],
            "source_row_number": num, "source_field": slug, "question_slug": slug, "question_wording": wording,
            "question_context": ctx, "context": ctx, "duplicate_text_group_id": src["duplicate_text_group_id"],
            "original_text": text.strip(),
        })
        rows.append(row)
    return rows


def write_fixture(tmp_path, ind, cmp_, qst, pilot_specs):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    cols = {
        DS_IND: ["source_dataset", "source_file", "source_row_number", "duplicate_text_group_id", "exclusion_reason",
                 "row_rang_e", "event", "sector", "source_language", *[IND_COL[s] for s in IND_SLUGS]],
        DS_CMP: ["source_dataset", "source_file", "source_row_number", "duplicate_text_group_id", "exclusion_reason",
                 "id", "event_v_nement", "source", mod.COMPILATION_CTX_COL, mod.COMPILATION_THEME_COL,
                 "source_language", mod.COMPILATION_TEXT_COL],
        DS_QST: ["source_dataset", "source_file", "source_row_number", "duplicate_text_group_id", "exclusion_reason",
                 "unique_identifier", "language", Q19, *[QST_COL[s] for s in ("q8", "q14", "q18")]],
    }
    for ds, rows in ((DS_IND, ind), (DS_CMP, cmp_), (DS_QST, qst)):
        pd.DataFrame(rows).reindex(columns=cols[ds], fill_value="").to_csv(
            processed / mod.SOURCE_FILES[ds], index=False, encoding="utf-8", lineterminator="\n")
    pilot_path = tmp_path / "pilot.csv"
    pd.DataFrame(_pilot_rows(pilot_specs, ind, cmp_), columns=PILOT_COLS).to_csv(
        pilot_path, index=False, encoding="utf-8", lineterminator="\n")
    return processed, pilot_path


def run_builder(tmp_path, sources=None, quotas=None, out="out", **kwargs):
    ind, cmp_, qst, specs = sources or default_sources()
    processed, pilot_path = write_fixture(tmp_path, ind, cmp_, qst, specs)
    out_dir = tmp_path / out
    paths = {
        "pool_path": out_dir / "pool_v2.csv", "ledger_path": out_dir / "ledger_v2.csv",
        "manifest_path": out_dir / "manifest_v2.json", "report_path": out_dir / "report_v2.md",
    }
    result = mod.build_expansion_pool_v2(
        processed_dir=processed, pilot_path=pilot_path, quotas=quotas or DEFAULT_QUOTAS,
        expected_pilot_rows=None, **paths, **kwargs,
    )
    return result, paths, (ind, cmp_, qst, specs)


@pytest.fixture
def built(tmp_path):
    result, paths, sources = run_builder(tmp_path)
    return result, paths, sources, tmp_path


def _new_rows(pool, n_pilot=2):
    return pool.iloc[n_pilot:]


# ── the old multi-field defect ────────────────────────────────────────────────

def _legacy_select(sources, n, seed=42):
    """Behaviour of the old builder: candidates per (record, field), shuffle, take n; no record-level key."""
    ind = sources[0]
    candidates = [(r["source_row_number"], s) for r in ind if r["source_row_number"] != "1" for s in IND_SLUGS]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def test_fixture_fails_under_old_multi_field_logic_and_passes_under_v2(tmp_path):
    sources = default_sources()
    legacy = _legacy_select(sources, n=4)               # 4 picks from only 3 eligible records (pigeonhole)
    records = [rec for rec, _ in legacy]
    assert len(set(records)) < len(records), "fixture must make the legacy logic repeat a record"

    result, _, _ = run_builder(tmp_path, sources=sources, quotas={DS_IND: 3, DS_CMP: 6, DS_QST: 3})
    ind_new = _new_rows(result["pool"])
    ind_new = ind_new[ind_new["source_dataset"] == DS_IND]
    assert len(ind_new) == 3
    assert ind_new["source_row_number"].is_unique


def test_quota_exceeding_eligible_records_stops_instead_of_relaxing(tmp_path):
    with pytest.raises(mod.AllocationError, match="eligible records"):
        run_builder(tmp_path, quotas={DS_IND: 4, DS_CMP: 6, DS_QST: 3})   # only 3 eligible individual records
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


# ── record uniqueness, pilot overlap, duplicates ──────────────────────────────

def test_pool_record_uniqueness_and_one_field_per_record(built):
    result, *_ = built
    pool = result["pool"]
    assert not pool.duplicated(["source_dataset", "source_row_number"]).any()
    assert len(pool) == 14 and len(_new_rows(pool)) == 12


def test_pilot_rows_preserved_verbatim_and_first(built):
    result, paths, (ind, cmp_, qst, specs), tmp_path = built
    pilot = pd.read_csv(tmp_path / "pilot.csv", dtype=str, keep_default_na=False)
    head = result["pool"].iloc[:len(pilot)][PILOT_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, pilot[PILOT_COLS], check_dtype=False)
    assert list(result["pool"].columns) == PILOT_COLS + ["source_language", "language_basis"]


def test_pilot_source_record_overlap_excluded_and_ledgered(built):
    result, *_ = built
    pool, ledger = result["pool"], result["ledger"]
    new = _new_rows(pool)
    assert not ((new["source_dataset"] == DS_IND) & (new["source_row_number"] == "1")).any()
    assert not ((new["source_dataset"] == DS_CMP) & (new["source_row_number"] == "1")).any()
    row = lambda ds, n, slug: ledger[(ledger.source_dataset == ds) & (ledger.source_row_number == n) & (ledger.question_slug == slug)].iloc[0]
    assert row(DS_IND, "1", "public_engagement_question").reason_code == "pilot_item"
    assert row(DS_IND, "1", "practical_idea").reason_code == "pilot_record_overlap"


def test_normalised_text_duplicates_policy_a_deterministic_and_ledgered(built):
    result, *_ = built
    pool, ledger = result["pool"], result["ledger"]
    keys = pool["original_text"].map(key_of)
    assert keys.is_unique
    dup = ledger[(ledger.source_dataset == DS_CMP) & (ledger.source_row_number == "3")].iloc[0]
    assert dup.reason_code == "duplicate_text_key_not_retained"
    assert dup.retained_entry == f"{DS_CMP}|2|comment_question_or_idea"   # lowest row number retained
    assert dup.text_key_sha256_16 == ledger[(ledger.source_dataset == DS_CMP) & (ledger.source_row_number == "2")].iloc[0].text_key_sha256_16


def test_duplicate_group_id_retains_one_per_group(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    cmp_[3]["duplicate_text_group_id"] = "dup_000500"   # row 4
    cmp_[5]["duplicate_text_group_id"] = "dup_000500"   # row 6 (different text, same row-level group)
    result, _, _ = run_builder(tmp_path, sources=(ind, cmp_, qst, specs), quotas={DS_IND: 3, DS_CMP: 5, DS_QST: 3})
    ledger, pool = result["ledger"], result["pool"]
    codes = ledger[ledger.duplicate_text_group_id == "dup_000500"].reason_code.tolist()
    assert sorted(codes).count("duplicate_group_not_retained") == 1
    assert (_new_rows(pool)["duplicate_text_group_id"] == "dup_000500").sum() <= 1


def test_pilot_text_or_group_overlap_is_excluded(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    cmp_[1]["duplicate_text_group_id"] = "dup_000900"
    cmp_[0]["duplicate_text_group_id"] = "dup_000900"                        # pilot row's group
    cmp_[6][mod.COMPILATION_TEXT_COL] = "  Pilot compilation   TEXT about open data portals "   # equals pilot text
    result, _, _ = run_builder(tmp_path, sources=(ind, cmp_, qst, specs), quotas={DS_IND: 3, DS_CMP: 4, DS_QST: 3})
    led = result["ledger"]
    lookup = lambda n: led[(led.source_dataset == DS_CMP) & (led.source_row_number == n)].iloc[0].reason_code
    assert lookup("2") == "pilot_group_overlap"
    assert lookup("7") == "pilot_text_overlap"


# ── provenance, eligibility, language, context ────────────────────────────────

def test_source_text_provenance_is_verbatim(built):
    result, paths, (ind, cmp_, qst, specs), tmp_path = built
    new = _new_rows(result["pool"])
    cell = {}
    for r in ind:
        for s in IND_SLUGS:
            cell[(DS_IND, r["source_row_number"], s)] = r[IND_COL[s]]
    for r in cmp_:
        cell[(DS_CMP, r["source_row_number"], "comment_question_or_idea")] = r[mod.COMPILATION_TEXT_COL]
    for r in qst:
        for s in ("q8", "q14", "q18"):
            cell[(DS_QST, r["source_row_number"], s)] = r[QST_COL[s]]
    for _, row in new.iterrows():
        assert row.original_text == cell[(row.source_dataset, row.source_row_number, row.question_slug)].strip()


def test_selected_rows_satisfy_eligibility_rules(built):
    result, *_ = built
    new = _new_rows(result["pool"])
    assert (new.original_text.str.len() >= mod.MIN_TEXT_LEN).all()
    assert not new.original_text.map(key_of).isin(mod.GENERIC_FILLERS).any()
    assert set(new.source_language) <= {"English", "French", "Unknown"}
    assert not new.original_text.str.contains("Arabic language comment").any()
    assert (new[ANN_COLS] == "").all().all()


def test_language_metadata_rules_and_questionnaire_caveat(built):
    result, *_ = built
    pool, ledger = result["pool"], result["ledger"]
    led = lambda n: ledger[(ledger.source_dataset == DS_CMP) & (ledger.source_row_number == n)].iloc[0]
    assert led("8").reason_code == "language_out_of_scope" and led("8").source_language == "Arabic"
    assert led("9").source_language == "Unknown"
    assert led("10").reason_code == "too_short" and led("11").reason_code == "generic_filler"
    lang, basis = mod.normalize_language("S", DS_IND)
    assert lang == "Unknown" and "not recognised" in basis
    lang, basis = mod.normalize_language("", DS_CMP)
    assert lang == "Unknown" and "blank" in basis
    lang, basis = mod.normalize_language(QLANG_FR, DS_QST)
    assert lang == "French" and "survey language" in basis and "not necessarily the language of the comment" in basis
    q = pool[pool.source_dataset == DS_QST]
    assert q.language_basis.str.contains("survey language").all()
    assert (pool.language_basis != "").all()


def test_language_is_never_inferred_from_text(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    cmp_[4]["source_language"] = ""              # French text, blank metadata
    result, _, _ = run_builder(tmp_path, sources=(ind, cmp_, qst, specs), quotas={DS_IND: 3, DS_CMP: 6, DS_QST: 3})
    led = result["ledger"]
    row = led[(led.source_dataset == DS_CMP) & (led.source_row_number == "5")].iloc[0]
    assert row.source_language == "Unknown"


def test_context_and_question_wording_match_pilot_format(built):
    result, paths, (ind, cmp_, qst, specs), tmp_path = built
    new = _new_rows(result["pool"])
    by = lambda rows: {r["source_row_number"]: r for r in rows}
    for _, row in new.iterrows():
        n = row.source_row_number
        if row.source_dataset == DS_IND:
            src = by(ind)[n]
            expected = f"Event: {src['event']} | Sector: {src['sector']}"
            wording = next(w for s, _, w in mod.INDIVIDUAL_FIELDS if s == row.question_slug)
        elif row.source_dataset == DS_CMP:
            src = by(cmp_)[n]
            expected = (f"Event: {src['event_v_nement']} | Source: {src['source']} | "
                        f"Prompt: {src[mod.COMPILATION_CTX_COL]} | Theme: {src[mod.COMPILATION_THEME_COL]}")
            wording = mod.COMPILATION_WORDING
        else:
            src = by(qst)[n]
            expected = f"Language: {src['language']} | Sector: {src[Q19]}"
            wording = next(w for s, _, w in mod.QUESTIONNAIRE_FIELDS if s == row.question_slug)
        assert row.context == expected and row.question_context == expected
        assert row.question_wording == wording
        assert row.source_record_id == row.source_row_number and row.source_field == row.question_slug


# ── allocation, strata, shortfall ─────────────────────────────────────────────

def test_stratum_allocation_and_dataset_quotas(built):
    result, *_ = built
    new = _new_rows(result["pool"])
    assert new.groupby("source_dataset").size().to_dict() == DEFAULT_QUOTAS
    assert new[new.source_dataset == DS_IND].question_slug.value_counts().max() == 1   # 3 over 5 fields, 1 each
    q = new[new.source_dataset == DS_QST].question_slug.value_counts().to_dict()
    assert q == {"q8": 1, "q14": 1, "q18": 1}
    log = result["selection_log"][DS_CMP]
    assert sum(log["final_quota"].values()) == 6
    assert log["final_quota"]["Not Applicable"] <= log["equal_share"]       # capped strata get no remainder
    assert log["final_quota"]["Blank/unspecified"] <= log["equal_share"]


def test_allocate_quotas_caps_and_errors():
    avail = {"A": 50, "B": 40, "NA": 100}
    q = mod.allocate_quotas(10, avail, capped=frozenset({"NA"}))
    assert sum(q.values()) == 10 and q["NA"] == 3            # equal share 10//3 = 3; remainder skips capped NA
    assert q["A"] + q["B"] == 7
    with pytest.raises(mod.AllocationError):
        mod.allocate_quotas(10, {"A": 2, "B": 2}, capped=frozenset())
    with pytest.raises(mod.AllocationError):
        mod.allocate_quotas(10, {"A": 100, "NA": 3}, capped=frozenset({"A"}))


def test_stratum_shortfall_is_redistributed_and_logged(tmp_path):
    result, _, _ = run_builder(tmp_path, quotas={DS_IND: 3, DS_CMP: 8, DS_QST: 3})
    log = result["selection_log"][DS_CMP]
    assert log["equal_share"] == 2
    assert any("'Not Applicable'" in e and "shortfall" in e for e in log["events"])
    assert log["final_quota"]["Not Applicable"] == 1
    assert sum(log["final_quota"].values()) == 8
    assert len(_new_rows(result["pool"])[lambda d: d.source_dataset == DS_CMP]) == 8


# ── native IDs, ledger completeness ───────────────────────────────────────────

def test_native_id_consistency(built):
    result, paths, (ind, cmp_, qst, specs), tmp_path = built
    ledger = result["ledger"]
    expected = {}
    for r in ind:
        expected[(DS_IND, r["source_row_number"])] = r["row_rang_e"]
    for r in cmp_:
        expected[(DS_CMP, r["source_row_number"])] = r["id"]
    for r in qst:
        expected[(DS_QST, r["source_row_number"])] = r["unique_identifier"]
    for _, row in ledger.iterrows():
        assert row.native_id == expected[(row.source_dataset, row.source_row_number)]
    assert ledger.groupby("source_dataset").apply(
        lambda g: g.drop_duplicates("source_row_number").native_id.is_unique, include_groups=False).all()


def test_source_row_numbers_must_be_unique(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    ind.append(dict(ind[1]))                      # same source_row_number twice
    with pytest.raises(ValueError, match="not unique"):
        run_builder(tmp_path, sources=(ind, cmp_, qst, specs))


def test_pilot_row_must_match_source_verbatim(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    processed, pilot_path = write_fixture(tmp_path, ind, cmp_, qst, specs)
    pilot = pd.read_csv(pilot_path, dtype=str, keep_default_na=False)
    pilot.loc[0, "original_text"] = "tampered text"
    pilot.to_csv(pilot_path, index=False)
    with pytest.raises(ValueError, match="does not match its source"):
        mod.build_expansion_pool_v2(processed_dir=processed, pilot_path=pilot_path, quotas=DEFAULT_QUOTAS,
                                    expected_pilot_rows=None, dry_run=True)


def test_ledger_accounts_for_every_nonblank_candidate_exactly_once(built):
    result, paths, (ind, cmp_, qst, specs), tmp_path = built
    ledger, pool = result["ledger"], result["pool"]
    nonblank = 0
    for r in ind:
        nonblank += sum(r[IND_COL[s]].strip() != "" for s in IND_SLUGS)
    nonblank += sum(r[mod.COMPILATION_TEXT_COL].strip() != "" for r in cmp_)
    nonblank += sum(r[QST_COL[s]].strip() != "" for r in qst for s in ("q8", "q14", "q18"))
    assert len(ledger) == nonblank
    assert not ledger.duplicated(["source_dataset", "source_row_number", "question_slug"]).any()
    assert set(ledger.reason_code) <= set(mod.REASONS)
    assert (ledger.reason != "").all()
    selected = ledger[ledger.reason_code == "selected"]
    assert len(selected) == len(_new_rows(pool))
    assert set(selected.pool_id) == set(_new_rows(pool).pilot_id)
    assert (ledger[ledger.reason_code == "pilot_item"].pool_id.str.startswith("pilot_")).all()
    # exactly one of: selected / excluded / not_drawn per entry
    assert set(ledger.disposition) <= {"selected", "excluded", "not_drawn"}


def test_blank_cells_are_counted_in_stats_not_ledger(built):
    result, *_ = built
    blanks = result["stats"]["blank_cells_not_in_ledger"]
    assert blanks.get(f"{DS_CMP}|comment_question_or_idea", 0) == 0     # every fixture compilation row has text
    assert blanks[f"{DS_QST}|q18"] == 3                                  # q18 blank on 3 of 6 fixture rows
    assert blanks[f"{DS_IND}|other_comments"] == 1                       # only the pilot record row lacks it


# ── overwrite protection ──────────────────────────────────────────────────────

def test_builder_refuses_to_overwrite_and_leaves_outputs_untouched(tmp_path):
    result, paths, sources = run_builder(tmp_path)
    before = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    with pytest.raises(FileExistsError):
        run_builder(tmp_path)
    after = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    assert before == after


def test_builder_refuses_when_any_single_output_exists(tmp_path):
    ind, cmp_, qst, specs = default_sources()
    processed, pilot_path = write_fixture(tmp_path, ind, cmp_, qst, specs)
    out = tmp_path / "out"
    out.mkdir()
    (out / "report_v2.md").write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        mod.build_expansion_pool_v2(
            processed_dir=processed, pilot_path=pilot_path, quotas=DEFAULT_QUOTAS, expected_pilot_rows=None,
            pool_path=out / "pool_v2.csv", ledger_path=out / "ledger_v2.csv",
            manifest_path=out / "manifest_v2.json", report_path=out / "report_v2.md")
    assert (out / "report_v2.md").read_text(encoding="utf-8") == "pre-existing"
    assert not (out / "pool_v2.csv").exists()


def test_dry_run_writes_nothing(tmp_path):
    result, paths, _ = run_builder(tmp_path, dry_run=True)
    assert not any(p.exists() for p in paths.values())
    assert len(result["pool"]) == 14


def test_default_output_paths_do_not_collide_with_original_pool():
    originals = {ANNOTATION / "expansion_pool_400.csv", ANNOTATION / "expansion_annotator_1.csv",
                 ANNOTATION / "expansion_annotator_2.csv", ANNOTATION / "pilot_annotation_50.csv"}
    outputs = {mod.POOL_PATH, mod.LEDGER_PATH, mod.MANIFEST_PATH, mod.REPORT_PATH}
    assert not (originals & outputs)
    assert mod.POOL_PATH.name == "expansion_pool_400_v2.csv"


# ═══════════════════════════ Part B: real v2 outputs ═══════════════════════════

REAL_POOL = ANNOTATION / "expansion_pool_400_v2.csv"
REAL_LEDGER = ANNOTATION / "expansion_pool_400_v2_ledger.csv"
REAL_MANIFEST = ANNOTATION / "expansion_pool_400_v2_manifest.json"
REAL_REPORT = ROOT / "reports" / "expansion_pool_v2_report.md"
real = pytest.mark.skipif(
    not (REAL_POOL.exists() and REAL_LEDGER.exists() and REAL_MANIFEST.exists()),
    reason="expansion_pool_400_v2 outputs have not been built",
)

PROTECTED_SHA256 = {
    "expansion_pool_400.csv": "f399885e3f4f406d6fcc5150de962347469ec16fce56e8361c42127e0d570cad",
    "expansion_annotator_1.csv": "f399885e3f4f406d6fcc5150de962347469ec16fce56e8361c42127e0d570cad",
    "expansion_annotator_2.csv": "f399885e3f4f406d6fcc5150de962347469ec16fce56e8361c42127e0d570cad",
    "pilot_annotation_50.csv": "e800b747e23196a1f168e8ae84a62ab56275b6d3a224d971730f1a125cae7f59",
}


def _rd(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def rpool():
    return _rd(REAL_POOL)


@pytest.fixture(scope="module")
def rledger():
    return _rd(REAL_LEDGER)


@pytest.fixture(scope="module")
def rpilot():
    return _rd(ANNOTATION / "pilot_annotation_50.csv")


@pytest.fixture(scope="module")
def rsources():
    return {ds: _rd(PROCESSED / name) for ds, name in mod.SOURCE_FILES.items()}


@real
def test_real_original_pool_pilot_and_annotator_copies_untouched():
    for name, digest in PROTECTED_SHA256.items():
        assert hashlib.sha256((ANNOTATION / name).read_bytes()).hexdigest() == digest, name


@real
def test_real_structure_and_pilot_preserved(rpool, rpilot):
    assert len(rpool) == 400
    assert list(rpool.columns) == list(rpilot.columns) + ["source_language", "language_basis"]
    pd.testing.assert_frame_equal(rpool.iloc[:50][rpilot.columns].reset_index(drop=True), rpilot, check_dtype=False)
    assert rpool.pilot_id.is_unique
    assert rpool.iloc[50:].pilot_id.str.startswith("expansion_v2_").all()
    assert (rpool[ANN_COLS] == "").all().all()


@real
def test_real_record_uniqueness_and_pilot_overlap(rpool, rpilot):
    assert not rpool.duplicated(["source_dataset", "source_row_number"]).any()
    new = rpool.iloc[50:]
    pilot_records = set(zip(rpilot.source_dataset, rpilot.source_row_number))
    assert not (set(zip(new.source_dataset, new.source_row_number)) & pilot_records)


@real
def test_real_no_repeated_normalised_text_or_group(rpool, rpilot):
    keys = rpool.original_text.map(key_of)
    assert keys.is_unique
    groups = rpool.duplicate_text_group_id.str.strip()
    nonblank = groups[groups != ""]
    assert nonblank.is_unique
    assert not (set(rpool.iloc[50:].duplicate_text_group_id.str.strip()) - {""}) & (set(rpilot.duplicate_text_group_id.str.strip()) - {""})


@real
def test_real_dataset_and_stratum_allocation(rpool):
    new = rpool.iloc[50:]
    assert new.groupby("source_dataset").size().to_dict() == mod.QUOTAS
    ind = new[new.source_dataset == DS_IND].question_slug.value_counts().to_dict()
    assert set(ind.values()) == {14} and len(ind) == 5
    q = new[new.source_dataset == DS_QST].question_slug.value_counts().to_dict()
    assert q == {"q8": 40, "q14": 40, "q18": 40}
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    cmp_final = manifest["selection_log"][DS_CMP]["final_quota"]
    assert sum(cmp_final.values()) == 160
    assert cmp_final["Not Applicable"] <= manifest["selection_log"][DS_CMP]["equal_share"]
    themes = (new[new.source_dataset == DS_CMP].context.str.extract(r"Theme: (.*)$")[0].str.strip()
              .replace({"": "Blank/unspecified", "Blank": "Blank/unspecified"}).value_counts().to_dict())
    assert themes == {k: v for k, v in cmp_final.items() if v}


@real
def test_real_provenance_text_is_verbatim_and_eligible(rpool, rsources):
    cols = {DS_IND: {s: IND_COL[s] for s in IND_SLUGS}, DS_CMP: {"comment_question_or_idea": mod.COMPILATION_TEXT_COL},
            DS_QST: {s: QST_COL[s] for s in ("q8", "q14", "q18")}}
    index = {ds: df.set_index("source_row_number") for ds, df in rsources.items()}
    for _, row in rpool.iloc[50:].iterrows():
        src = index[row.source_dataset].loc[row.source_row_number]
        cell = src[cols[row.source_dataset][row.question_slug]]
        assert row.original_text == cell.strip()
        assert len(row.original_text) >= mod.MIN_TEXT_LEN
        assert key_of(row.original_text) not in mod.GENERIC_FILLERS
        assert src["exclusion_reason"].strip() == ""
        assert row.source_file == src["source_file"]
        assert row.duplicate_text_group_id == src["duplicate_text_group_id"].strip()


@real
def test_real_native_ids_match_sources_and_are_unique(rledger, rsources):
    sel = rledger[rledger.reason_code == "selected"]
    for ds, col in mod.NATIVE_ID_COLUMN.items():
        sub = sel[sel.source_dataset == ds]
        src = rsources[ds].set_index("source_row_number")
        assert (sub.native_id.values == src.loc[sub.source_row_number, col].values).all()
        assert sub.native_id.is_unique
        assert (sub.native_id_column == col).all()
        assert rsources[ds][col].replace("", pd.NA).dropna().is_unique


@real
def test_real_language_and_context_fields(rpool, rsources):
    assert (rpool.language_basis != "").all()
    assert set(rpool.source_language) <= {"English", "French", "Unknown"}
    q = rpool[rpool.source_dataset == DS_QST]
    assert q.language_basis.str.contains("survey language").all()
    new = rpool.iloc[50:]
    index = {ds: df.set_index("source_row_number") for ds, df in rsources.items()}
    for _, row in new.iterrows():
        s = index[row.source_dataset].loc[row.source_row_number]
        if row.source_dataset == DS_IND:
            expected = f"Event: {s['event']} | Sector: {s['sector']}"
        elif row.source_dataset == DS_CMP:
            expected = (f"Event: {s['event_v_nement']} | Source: {s['source']} | "
                        f"Prompt: {s[mod.COMPILATION_CTX_COL]} | Theme: {s[mod.COMPILATION_THEME_COL]}")
        else:
            expected = f"Language: {s['language']} | Sector: {s[Q19]}"
        assert row.context == expected == row.question_context


@real
def test_real_language_scope_exclusions_ledgered(rpool, rledger):
    out = rledger[rledger.reason_code == "language_out_of_scope"]
    assert set(out.source_language) <= {"Arabic", "Japanese"} and len(out) >= 1
    assert not set(zip(out.source_dataset, out.source_row_number)) & set(zip(rpool.source_dataset, rpool.source_row_number))


@real
def test_real_ledger_complete(rledger, rpool, rsources):
    nonblank = 0
    for slug in IND_SLUGS:
        nonblank += int((rsources[DS_IND][IND_COL[slug]].str.strip() != "").sum())
    nonblank += int((rsources[DS_CMP][mod.COMPILATION_TEXT_COL].str.strip() != "").sum())
    for slug in ("q8", "q14", "q18"):
        nonblank += int((rsources[DS_QST][QST_COL[slug]].str.strip() != "").sum())
    assert len(rledger) == nonblank
    assert not rledger.duplicated(["source_dataset", "source_row_number", "question_slug"]).any()
    assert set(rledger.reason_code) <= set(mod.REASONS)
    assert (rledger.reason_code == "selected").sum() == 350
    assert set(rledger[rledger.reason_code == "selected"].pool_id) == set(rpool.iloc[50:].pilot_id)
    assert (rledger.reason_code == "pilot_item").sum() == 50


@real
def test_real_manifest_hashes_match_files_and_report_states_limitations():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["outputs_sha256"][REAL_POOL.name] == hashlib.sha256(REAL_POOL.read_bytes()).hexdigest()
    assert manifest["outputs_sha256"][REAL_LEDGER.name] == hashlib.sha256(REAL_LEDGER.read_bytes()).hexdigest()
    assert manifest["inputs_sha256"]["pilot_annotation_50.csv"] == PROTECTED_SHA256["pilot_annotation_50.csv"]
    for name in mod.SOURCE_FILES.values():
        assert manifest["inputs_sha256"][name] == hashlib.sha256((PROCESSED / name).read_bytes()).hexdigest()
    assert "respondent" in manifest["uniqueness_unit"]
    report = REAL_REPORT.read_text(encoding="utf-8")
    assert "Not respondent-unique" in report
    assert "survey language" in report
    assert "no human labels exist" in report
