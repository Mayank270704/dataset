"""
scripts/build_ai_working_labels_v1.py
=====================================
Assemble AI-generated WORKING labels (stance + separate sentiment) for the 400-record v2 annotation pool.

Inputs (read-only)
    data/annotation/expansion_pool_400_v2.csv                                   official pool (never modified)
    data/reviewer_handoff/ai_assisted_drafts_review_only/person_1/...draft.csv  existing primary AI draft, 50 pilot stance labels
    data/ai_labels/stance_sentiment_ai_v1/raw/ai_labels_expansion_v2_350.csv    new AI stance+sentiment labels, 350 expansion rows
    data/ai_labels/stance_sentiment_ai_v1/raw/ai_sentiment_pilot_50.csv         new AI sentiment labels, 50 pilot rows

Outputs (data/ai_labels/stance_sentiment_ai_v1/)
    ai_working_labels_pool400.csv   same columns, ids, order and source text as the official pool; stance + review_notes filled
    ai_sentiment_labels_pool400.csv pilot_id, ai_sentiment, label_source, original_text (sentiment kept separate from stance)
    provenance.json, manifest.json  (SHA-256 of every input and output)

Provenance: every label here is AI-generated and unverified. None is a human annotation, gold label or adjudication.
Write-once: refuses to run if any output exists.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ai_labels" / "stance_sentiment_ai_v1"
POOL = ROOT / "data" / "annotation" / "expansion_pool_400_v2.csv"
DRAFT1 = (ROOT / "data" / "reviewer_handoff" / "ai_assisted_drafts_review_only" / "person_1"
          / "policylens_annotator_1_AI_assisted_draft.csv")
RAW_EXP = OUT / "raw" / "ai_labels_expansion_v2_350.csv"
RAW_PSENT = OUT / "raw" / "ai_sentiment_pilot_50.csv"
STANCE = {"Support", "Oppose", "Mixed", "Neutral", "Unclear"}
SENT = {"Positive", "Negative", "Neutral", "Mixed"}
ANNOT = ["stance", "concern", "requested_action", "evidence_span", "review_notes", "annotator_id"]
SOURCE_TAG = "ai_working_label_unverified"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def main() -> int:
    targets = [OUT / "ai_working_labels_pool400.csv", OUT / "ai_sentiment_labels_pool400.csv",
               OUT / "provenance.json", OUT / "manifest.json"]
    if any(t.exists() for t in targets):
        print("outputs already exist; refusing to overwrite", file=sys.stderr)
        return 2
    pool = pd.read_csv(POOL, dtype=str, keep_default_na=False)
    d1 = pd.read_csv(DRAFT1, dtype=str, keep_default_na=False)
    exp = pd.read_csv(RAW_EXP, dtype=str, keep_default_na=False)
    ps = pd.read_csv(RAW_PSENT, dtype=str, keep_default_na=False)

    pilot = pool[pool.pilot_id.str.startswith("pilot_")]
    assert d1.pilot_id.tolist() == pilot.pilot_id.tolist(), "draft-1 ids differ from pool pilot rows"
    assert d1.original_text.tolist() == pilot.original_text.tolist(), "draft-1 text differs from pool pilot rows"
    assert exp.pilot_id.tolist() == pool[pool.pilot_id.str.startswith("expansion_v2_")].pilot_id.tolist()
    assert ps.pilot_id.tolist() == pilot.pilot_id.tolist()
    assert set(exp.ai_stance) <= STANCE and set(d1.stance) <= STANCE
    assert set(exp.ai_sentiment) <= SENT and set(ps.ai_sentiment) <= SENT

    fixed = [c for c in pool.columns if c not in ANNOT]
    work = pool.copy()
    for c in ANNOT:
        work[c] = ""
    st = dict(zip(d1.pilot_id, d1.stance)) | dict(zip(exp.pilot_id, exp.ai_stance))
    nt = {p: f"AI draft 1 (person_1, pre-existing): {n}" for p, n in zip(d1.pilot_id, d1.review_notes)}
    nt |= {p: (f"AI v1 (2026-10-14): {n}" if n else "AI v1 (2026-10-14)") for p, n in zip(exp.pilot_id, exp.ai_note)}
    work["stance"] = work.pilot_id.map(st)
    work["review_notes"] = work.pilot_id.map(nt)
    work["annotator_id"] = SOURCE_TAG
    assert work[fixed].equals(pool[fixed])
    assert work.stance.notna().all() and work.stance.ne("").all()

    sent = pd.DataFrame({"pilot_id": pool.pilot_id})
    sm = dict(zip(ps.pilot_id, ps.ai_sentiment)) | dict(zip(exp.pilot_id, exp.ai_sentiment))
    sent["ai_sentiment"] = sent.pilot_id.map(sm)
    sent["label_source"] = SOURCE_TAG
    sent["source_language"] = pool.source_language
    sent["original_text"] = pool.original_text
    assert sent.ai_sentiment.notna().all()

    OUT.mkdir(parents=True, exist_ok=True)
    work.to_csv(targets[0], index=False, lineterminator="\n")
    sent.to_csv(targets[1], index=False, lineterminator="\n")
    prov = {
        "label_set": "stance_sentiment_ai_v1",
        "status": ("AI-generated working labels. NOT human-verified, NOT gold, NOT adjudicated. Authorised by the project "
                   "owner as the working supervision source while human annotations are unavailable."),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": int(len(work)),
        "stance": {
            "guidelines": ("reports/annotation_guidelines_v1.1.md (stance toward Open Government; criticism alone never "
                           "Oppose; French labelled with the standard rules by a French-reading labeller, section 7)"),
            "pilot_50": {"source": rel(DRAFT1),
                         "method": "pre-existing AI-assisted draft designated primary; reused unchanged (no merge, no vote)"},
            "expansion_350": {
                "source": rel(RAW_EXP),
                "method": ("single-pass AI labelling by Claude (model id claude-opus-5-5) inside a Claude Code session, "
                           "reading each text with its question slug and source language and applying the v1.1 decision "
                           "order (understandable? -> off-topic -> position). One label per record, no second pass, "
                           "no human review."),
                "prompt_version": ("guidelines v1.1 decision order + section 5 special cases; session instruction "
                                   "'AI-label the 350 expansion_v2 records under guidelines v1.1' (2026-10-14)"),
                "consistency_adjustments": ("expansion_v2_283 and expansion_v2_345 (session praise) set to Neutral to "
                                            "match draft 1's treatment of thanks/session praise"),
            },
            "counts_all_400": work.stance.value_counts().to_dict(),
        },
        "sentiment": {
            "construct": ("tone of the text (Positive / Negative / Neutral / Mixed); independent of stance. Negative tone "
                          "does not imply Oppose."),
            "method": ("same labeller and pass as the expansion stance labels; pilot sentiment labelled in the same "
                       "session (no prior sentiment draft existed)"),
            "counts_all_400": sent.ai_sentiment.value_counts().to_dict(),
        },
        "known_limitations": [
            "single AI labeller per record; no inter-annotator agreement exists or can be computed from these labels",
            "pilot stance labels and expansion labels come from different AI passes",
            "Oppose has very few records and Mixed none; these classes cannot be trained or evaluated",
            "labels are judgements about guideline application and may encode systematic AI bias",
        ],
    }
    targets[2].write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    man = {"inputs_sha256": {rel(p): sha(p) for p in (POOL, DRAFT1, RAW_EXP, RAW_PSENT)},
           "outputs_sha256": {rel(p): sha(p) for p in targets[:3]}}
    targets[3].write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stance": prov["stance"]["counts_all_400"], "sentiment": prov["sentiment"]["counts_all_400"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
