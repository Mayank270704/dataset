"""
scripts/run_research_evaluation.py
==================================
Consolidated, NON-DESTRUCTIVE research evaluation entry point.

    python scripts/run_research_evaluation.py                 # run checks, print, write a new report folder
    python scripts/run_research_evaluation.py --no-write      # print only
    python scripts/run_research_evaluation.py --init-baseline # one-time: record SHA-256 of protected artifacts

It reads existing artifacts only. It does NOT re-embed or re-cluster, train any model, create labels,
or touch source data. Every item is reported with one of:

    PASSED   a check that ran and passed
    FAILED   a check that ran and failed (exit code 1)
    WARN     ran, but something needs attention
    BLOCKED  cannot run until human annotations/reviews exist
    NOT RUN  intentionally not executed in this entry point

Interpretation rules printed with every report:
  * Cluster assignments and silhouette scores are unsupervised and NOT human-validated.
  * Sentiment, stance and concern are different constructs; none is inferred from another.
  * Strata quotas are fixed, so pooled counts do not represent the population without weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_annotation_agreement as agreement  # noqa: E402
import verify_expansion_pool_v2 as verify  # noqa: E402
ANNOTATION = ROOT / "data" / "annotation"
REVIEW_DIR = ROOT / "data" / "comparisons" / "multilingual_embedding_450" / "human_review"
COMPARISON_DIR = ROOT / "data" / "comparisons" / "multilingual_embedding_450"
BASELINE_PATH = ROOT / "reports" / "protected_artifacts_baseline.json"
OUTPUT_BASE = ROOT / "reports" / "research_evaluation"

PROTECTED_FILES = [
    "data/annotation/pilot_annotation_50.csv", "data/annotation/annotator_1.csv", "data/annotation/annotator_2.csv",
    "data/annotation/expansion_pool_400.csv", "data/annotation/expansion_annotator_1.csv",
    "data/annotation/expansion_annotator_2.csv", "data/annotation/pilot_sampling_log.md",
    "data/annotation/expansion_sampling_log.md",
    "data/annotation/expansion_pool_400_v2.csv", "data/annotation/expansion_pool_400_v2_ledger.csv",
    "data/annotation/expansion_pool_400_v2_manifest.json", "reports/expansion_pool_v2_report.md",
    "data/annotation/expansion_v2_annotator_1.csv", "data/annotation/expansion_v2_annotator_2.csv",
    "data/annotation/expansion_v2_templates_manifest.json",
    "data/embeddings/corpus_metadata.csv", "data/embeddings/minilm_embeddings.npy", "data/embeddings/corpus_hash.txt",
    "data/clustering/cluster_labels.csv", "data/clustering/cluster_topics.csv",
    "data/clustering/clustering_config.json", "data/clustering/umap_2d.npy",
    "ap2016-18_individual_comments.csv",
    "ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv",
    "Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv",
    "data/processed/individual_comments_2016_18_source_clean.csv",
    "data/processed/consultation_compilation_2017_18_source_clean.csv",
    "data/processed/get_involved_questionnaire_2017_18_source_clean.csv",
]
PROTECTED_DIRS = [  # every file, except completed/ and analysis outputs that are written later
    "data/comparisons/multilingual_embedding_450",
]
EXCLUDED_PARTS = {"completed", "agreement_analysis"}

INTERPRETATION_RULES = [
    "Cluster assignments, silhouette and noise ratios are unsupervised descriptors and are NOT human-validated concerns.",
    "Sentiment (tone), stance (position toward Open Government) and concern (issue raised) are distinct; none is inferred from another.",
    "Sentiment corpora (Sentiment140, SentiMix, HinEmo-9, PolicyLens sentiment) are auxiliary and are not ground truth for consultation stance.",
    "The annotation pools are stratified with fixed quotas; they do not estimate population frequencies without design weights.",
    "Identical-text or duplicate-group rules prove identical text only, never that two comments came from the same respondent.",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_paths(root: Path = ROOT) -> list[Path]:
    paths = {root / p for p in PROTECTED_FILES if (root / p).exists()}
    for d in PROTECTED_DIRS:
        base = root / d
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and not (set(p.relative_to(base).parts) & EXCLUDED_PARTS):
                    paths.add(p)
    return sorted(paths)


def init_baseline(root: Path = ROOT, baseline_path: Path = BASELINE_PATH) -> dict[str, Any]:
    if baseline_path.exists():
        raise FileExistsError(f"Baseline already exists: {baseline_path}")
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in protected_paths(root)}
    data = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "SHA-256 of protected artifacts when the baseline was initialised. Changes are reported as FAILED.",
            "files": files}
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "x", encoding="utf-8", newline="") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return data


def item(section: str, name: str, status: str, detail: str = "") -> dict[str, str]:
    return {"section": section, "name": name, "status": status, "detail": detail}


# ── Sections ──────────────────────────────────────────────────────────────────

def check_protected(root: Path = ROOT, baseline_path: Path = BASELINE_PATH) -> list[dict[str, str]]:
    sec = "1. Protected artifact integrity"
    if not baseline_path.exists():
        return [item(sec, "baseline", "WARN", "no baseline yet; run with --init-baseline once to record SHA-256 values")]
    base = json.loads(baseline_path.read_text(encoding="utf-8"))["files"]
    changed, missing = [], []
    for rel, digest in base.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != digest:
            changed.append(rel)
    new_unrecorded = [p.relative_to(root).as_posix() for p in protected_paths(root) if p.relative_to(root).as_posix() not in base]
    out = [item(sec, "protected files unchanged", "FAILED" if (changed or missing) else "PASSED",
                f"{len(base)} recorded; changed={changed}; missing={missing}" if (changed or missing)
                else f"{len(base)} recorded files match the baseline")]
    if new_unrecorded:
        out.append(item(sec, "files not in baseline", "WARN", f"{len(new_unrecorded)} protected-path file(s) not recorded: {new_unrecorded[:5]}"))
    return out


def check_v2_pool(root: Path = ROOT) -> list[dict[str, str]]:
    sec = "2. Expansion pool v2 verification"
    out = []
    for r in verify.run_checks():
        status = {"PASS": "PASSED", "FAIL": "FAILED", "WARN": "WARN"}[r["status"]]
        out.append(item(sec, r["check"], status, r["detail"]))
    return out


def check_baseline_450(comparison_dir: Path = COMPARISON_DIR, root: Path = ROOT) -> list[dict[str, str]]:
    sec = "3. Existing 450-comment baseline (UNSUPERVISED, not human-validated)"
    out: list[dict[str, str]] = []
    sel = REVIEW_DIR / "selection_manifest.json"
    if sel.exists():
        fp = json.loads(sel.read_text(encoding="utf-8"))["input_sha256"]
        mapping = {
            "sample_metadata": root / "data" / "embeddings" / "corpus_metadata.csv",
            "baseline_embeddings": comparison_dir / "all_minilm_l6_v2" / "embeddings.npy",
            "multilingual_embeddings": comparison_dir / "paraphrase_multilingual_minilm_l12_v2" / "embeddings.npy",
            "baseline_cluster_labels": comparison_dir / "all_minilm_l6_v2" / "cluster_labels.csv",
            "multilingual_cluster_labels": comparison_dir / "paraphrase_multilingual_minilm_l12_v2" / "cluster_labels.csv",
            "comparison_report": comparison_dir / "comparison_report.md",
        }
        bad = [k for k, p in mapping.items() if not p.exists() or sha256_file(p) != fp.get(k)]
        out.append(item(sec, "artifacts match the 44-pair review fingerprints", "FAILED" if bad else "PASSED",
                        f"mismatch/missing: {bad}" if bad else "6 embedding/cluster/report fingerprints match"))
    metrics = comparison_dir / "comparison_metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        for name, info in m.get("models", {}).items():
            q = info.get("quality_metrics", {})
            out.append(item(sec, f"stored comparison: {name}", "PASSED",
                            f"clusters={q.get('n_clusters')}, noise_fraction={q.get('noise_fraction')}, "
                            f"silhouette={q.get('silhouette_score')} (geometric descriptor; not concern validity)"))
        out.append(item(sec, "interpretation", "PASSED", str(m.get("interpretation_note", ""))[:200]))
    else:
        out.append(item(sec, "stored comparison metrics", "WARN", "comparison_metrics.json not found"))
    return out


def _nonblank_cells(path: Path, columns: list[str]) -> int:
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return int((df[columns] != "").sum().sum())


def check_annotation_readiness(root: Path = ROOT) -> list[dict[str, str]]:
    sec = "4. Annotation readiness and human-dependent analyses"
    out: list[dict[str, str]] = []
    ann_cols = agreement.ANNOTATION_COLUMNS
    for name, paths in (("pilot templates", [ANNOTATION / "annotator_1.csv", ANNOTATION / "annotator_2.csv"]),
                        ("v2 templates", [ANNOTATION / "expansion_v2_annotator_1.csv", ANNOTATION / "expansion_v2_annotator_2.csv"])):
        if all(p.exists() for p in paths):
            n = sum(_nonblank_cells(p, ann_cols) for p in paths)
            out.append(item(sec, f"{name} present and blank", "PASSED" if n == 0 else "FAILED", f"non-blank annotation cells: {n}"))
        else:
            out.append(item(sec, f"{name} present", "BLOCKED" if "v2" in name else "FAILED",
                            "run scripts/create_expansion_v2_annotator_templates.py" if "v2" in name else "missing"))
    for rnd in ("pilot", "expansion_v2"):
        res = agreement.analyze(rnd)
        label = f"{rnd} agreement analysis"
        status = res["status"]
        if status == "missing_files":
            out.append(item(sec, label, "BLOCKED", "awaiting human annotation: completed annotator_A/B files not present"))
        elif status == "pending_no_annotations":
            out.append(item(sec, label, "BLOCKED", "completed files exist but contain no valid stance labels"))
        elif status == "invalid":
            out.append(item(sec, label, "FAILED", "; ".join(res["errors"][:3])))
        else:
            parts = []
            for seg, m in res["segments"].items():
                k = m["stance_agreement"]
                parts.append(f"{seg}: N={m['sample_sizes']['paired_valid']}/{m['sample_sizes']['rows_in_segment']}, "
                             f"kappa={'n/a' if k['kappa'] is None else round(k['kappa'], 3)} ({k['status']})")
            out.append(item(sec, label, "PASSED", "; ".join(parts) + " [stratified sample; not population estimates]"))
    review_done = all((REVIEW_DIR / "completed" / f"reviewer_{i}_completed.csv").exists() for i in (1, 2))
    if review_done:
        out.append(item(sec, "44-pair concern-coherence review", "WARN",
                        "completed files present; run scripts/analyze_concern_review_agreement.py (not run here: it writes outputs)"))
    else:
        out.append(item(sec, "44-pair concern-coherence review", "BLOCKED",
                        "awaiting two independent bilingual reviewers (human_review/completed/ missing)"))
    return out


STANCE_RUNS = ROOT / "data" / "stance_training"


def stance_baseline_item(sec: str) -> dict[str, str]:
    """Supervised stance baselines: PASSED only for a saved run whose artifacts match their recorded SHA-256."""
    name = "supervised stance baselines (TF-IDF + LR/SVM)"
    runs = sorted(STANCE_RUNS.glob("run_*/metadata.json")) if STANCE_RUNS.exists() else []
    for mp in reversed(runs):
        m = json.loads(mp.read_text(encoding="utf-8"))
        bad = [n for n, d in m.get("artifacts_sha256", {}).items() if not (mp.parent / n).exists() or sha256_file(mp.parent / n) != d]
        if m.get("status") == "trained_and_evaluated":
            if bad:
                return item(sec, name, "FAILED", f"{mp.parent.name}: artifacts changed {bad[:3]}")
            met = json.loads((mp.parent / "metrics.json").read_text(encoding="utf-8"))
            f1 = {k: v["test"]["macro_f1"] for k, v in met["models"].items()}
            return item(sec, name, "PASSED", f"{mp.parent.name}: trained on {m['gold_records_used']} genuine human labels; "
                        f"held-out macro-F1 {f1} (majority baseline {met['majority_class_baseline']['macro_f1']}); stratified sample")
    from models.stance_classification.labels import readiness
    r = readiness()
    if r["status"] == "INVALID":
        return item(sec, name, "FAILED", f"annotation files invalid: {r['rounds']}")
    hint = "run `python scripts\\train_stance_baselines.py --check-only`"
    if r["status"] == "BLOCKED":
        return item(sec, name, "BLOCKED", f"pipeline implemented, not fitted: {r['reason']}; {hint}")
    return item(sec, name, "BLOCKED" if r["status"] == "INSUFFICIENT" else "WARN",
                f"{r['status']}: {r['gold']} gold labels {r.get('classes')}; {hint}")


def check_supervised_blockers() -> list[dict[str, str]]:
    sec = "5. Supervised models and sentiment/stance separation"
    return [
        stance_baseline_item(sec),
        item(sec, "concern-priority scoring", "NOT RUN", "validated priority scoring needs human-assessed concern validity; "
             "only an exploratory, unvalidated ranking exists (section 10)"),
        item(sec, "external sentiment baselines", "NOT RUN", "auxiliary only; polarity is not stance toward Open Government"),
    ]


FULL_CORPUS_MANIFEST = ROOT / "data" / "full_corpus" / "run_v1" / "run_manifest.json"


def check_full_corpus_run(manifest_path: Path = FULL_CORPUS_MANIFEST) -> list[dict[str, str]]:
    sec = "6. Full-corpus unsupervised run v1 (exploratory; NOT human-validated)"
    if not manifest_path.exists():
        return [item(sec, "full-corpus embedding and clustering (run_v1)", "NOT RUN",
                     "run_manifest.json not found; see scripts/run_full_corpus_clustering.py")]
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    bad: list[str] = []
    entries_path = run_dir / "corpus" / "corpus_entries.csv"
    if not entries_path.exists() or sha256_file(entries_path) != m["corpus"]["corpus_entries_sha256"]:
        bad.append("corpus_entries.csv")
    for key, e in m.get("embeddings", {}).items():
        p = run_dir / "embeddings" / key / "embeddings.npy"
        if not p.exists() or sha256_file(p) != e["embeddings_sha256"]:
            bad.append(f"embeddings/{key}")
    for rel, digest in m.get("assignment_sha256", {}).items():
        k, c = rel.split("/")
        p = run_dir / "clustering" / k / c / "entry_assignments.csv"
        if not p.exists() or sha256_file(p) != digest:
            bad.append(rel)
    out = [item(sec, "run_v1 artifacts match run_manifest.json", "FAILED" if bad else "PASSED",
                f"changed/missing: {bad[:5]}" if bad else
                f"{m['corpus']['entries']:,} entries, {m['corpus']['unique_texts']:,} unique texts; "
                f"{len(m.get('assignment_sha256', {}))} cluster assignment files verified")]
    out.append(item(sec, "interpretation", "PASSED",
                    f"clusters are unsupervised (human_validated={m.get('human_validated')}); "
                    "no concern, stance or sentiment label is implied"))
    if m.get("reporting_configuration"):
        out.append(item(sec, "reporting configuration", "PASSED",
                        f"{m['reporting_configuration']} (chosen by a pre-declared readability rule, not by validity)"))
    return out


EXPERIMENTS_V2_MANIFEST = ROOT / "data" / "full_corpus" / "experiments_v2" / "run_manifest.json"


def check_experiments_v2(manifest_path: Path = EXPERIMENTS_V2_MANIFEST) -> list[dict[str, str]]:
    sec = "7. Clustering experiments v2 (exploratory; proxies only; NOT human-validated)"
    if not manifest_path.exists():
        return [item(sec, "clustering experiments v2", "NOT RUN", "run_manifest.json not found; see scripts/run_clustering_experiments_v2.py")]
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    bad = [eid for eid, e in m.get("experiments", {}).items()
           if not (base / eid / "entry_assignments.csv").exists()
           or sha256_file(base / eid / "entry_assignments.csv") != e["assignments_sha256"]]
    out = [item(sec, "experiment assignments match run_manifest.json", "FAILED" if bad else "PASSED",
                f"changed/missing: {bad[:5]}" if bad else f"{len(m.get('experiments', {}))} experiments verified")]
    v1 = ROOT / "data" / "full_corpus" / "run_v1" / "run_manifest.json"
    if m.get("run_v1_manifest_sha256") and v1.exists():
        same = sha256_file(v1) == m["run_v1_manifest_sha256"]
        out.append(item(sec, "run_v1 manifest unchanged since the experiments", "PASSED" if same else "FAILED",
                        "run_v1 artifacts were not modified" if same else "run_v1/run_manifest.json differs"))
    out.append(item(sec, "interpretation", "PASSED",
                    f"human_validated={m.get('human_validated')}; shortlist for human review: {m.get('shortlist')}; "
                    "proxy metrics only, no cluster is a validated concern"))
    return out


CONCERN_DISCOVERY_DIR = ROOT / "data" / "full_corpus" / "concern_discovery_v1"


def check_concern_discovery(base: Path = CONCERN_DISCOVERY_DIR) -> list[dict[str, str]]:
    sec = "8. Concern discovery v1 (exploratory; NOT human-validated; Canadian consultations only)"
    mpath = base / "run_manifest.json"
    if not mpath.exists():
        return [item(sec, "concern discovery pipeline", "NOT RUN", "run_manifest.json not found; see scripts/run_concern_discovery.py")]
    m = json.loads(mpath.read_text(encoding="utf-8"))
    bad = [rel for rel, digest in m.get("outputs_sha256", {}).items()
           if not (base / rel).exists() or sha256_file(base / rel) != digest]
    out = [item(sec, "concern discovery outputs match run_manifest.json", "FAILED" if bad else "PASSED",
                f"changed/missing: {bad[:5]}" if bad else f"{len(m.get('outputs_sha256', {}))} output files verified; "
                f"selected configuration {m.get('selected_configuration')}")]
    out.append(item(sec, "summary grounding verification", "PASSED" if str(m.get("grounding_verification", "")).startswith("PASSED") else "FAILED",
                    str(m.get("grounding_verification"))))
    pkg = base / "human_review" / "package_manifest.json"
    if not pkg.exists():
        out.append(item(sec, "concern-cluster human review", "BLOCKED", "review package not built (run the review-package stage)"))
    else:
        pm = json.loads(pkg.read_text(encoding="utf-8"))
        changed = [n for n, d in pm.get("template_sha256", {}).items()
                   if not (base / "human_review" / n).exists() or sha256_file(base / "human_review" / n) != d]
        out.append(item(sec, "blank reviewer templates unchanged", "FAILED" if changed else "PASSED",
                        f"changed/missing: {changed}" if changed else f"{len(pm.get('template_sha256', {}))} templates; leakage check {pm.get('leakage_check')}"))
        done = [f"reviewer_{i}_task_{t}.csv" for i in (1, 2) for t in ("a", "b")]
        present = [n for n in done if (base / "human_review" / "completed" / n).exists()]
        metrics = base / "human_review_analysis" / "review_metrics.json"
        if metrics.exists():
            r = json.loads(metrics.read_text(encoding="utf-8"))
            out.append(item(sec, "concern-cluster human review", "PASSED",
                            f"analysis complete ({r.get('task_a_items')} intrusion items); see human_review_analysis/review_metrics.json"))
        elif len(present) == len(done):
            out.append(item(sec, "concern-cluster human review", "WARN",
                            "completed files present; run `python scripts/run_concern_discovery.py analyze-reviews`"))
        else:
            out.append(item(sec, "concern-cluster human review", "BLOCKED",
                            f"awaiting two independent reviewers ({len(present)}/{len(done)} completed files present)"))
    out.append(item(sec, "interpretation", "PASSED",
                    f"human_validated={m.get('human_validated')}; tiers and extractive descriptions are internal proxies; "
                    "no claim about Indian MCA or other consultations"))
    return out


def check_human_review_handoff() -> list[dict[str, str]]:
    sec = "9. Human-evaluation handoff (three tracks; reviewer judgments only from completed files)"
    import human_review_handoff as hh
    if not (ROOT / hh.GUIDE).exists():
        return [item(sec, "reviewer handoff guide", "NOT RUN", f"{hh.GUIDE.as_posix()} not found")]
    probs = hh.audit(ROOT)
    out = [item(sec, "templates blank/opaque and reviewer packets free of answer keys", "FAILED" if probs else "PASSED",
                "; ".join(probs[:5]) if probs else "leak audit passed (no keys, catalogs, model or tier information)")]
    checks = hh.status(ROOT)
    bad = [f"{c.track}/person {c.person}: {c.problems[0]}" for c in checks if c.state == "INVALID"]
    if bad:
        out.append(item(sec, "returned reviewer files", "WARN", "return to reviewer: " + "; ".join(bad[:4])))
    for track, r in hh.track_readiness(checks).items():
        status = "BLOCKED" if r.startswith("BLOCKED") else ("WARN" if r.startswith(("FIX", "READY")) else "PASSED")
        out.append(item(sec, f"{track} completed files", status, r))
    return out


PRIORITIZATION_DIR = ROOT / "data" / "full_corpus" / "concern_prioritization_v1"


def check_prioritization(base: Path = PRIORITIZATION_DIR) -> list[dict[str, str]]:
    sec = "10. Exploratory concern prioritization (NOT a validated measure of public importance)"
    mpath = base / "run_manifest.json"
    if not mpath.exists():
        return [item(sec, "exploratory prioritization", "NOT RUN", "run_manifest.json not found; see scripts/run_concern_prioritization.py")]
    m = json.loads(mpath.read_text(encoding="utf-8"))
    bad = [rel for rel, d in m.get("outputs_sha256", {}).items() if not (base / rel).exists() or sha256_file(base / rel) != d]
    out = [item(sec, "prioritization outputs match run_manifest.json", "FAILED" if bad else "PASSED",
                f"changed/missing: {bad[:5]}" if bad else f"{len(m.get('outputs_sha256', {}))} files verified")]
    ok = str(m.get("output_validation")) == "PASSED" and str(m.get("evidence_verification", "")).startswith("PASSED")
    out.append(item(sec, "ranking schema, contribution sums and verbatim evidence", "PASSED" if ok else "FAILED",
                    f"{m.get('output_validation')}; {m.get('evidence_verification')}"))
    
    # 4-tier raw-to-rank source evidence traceability audit
    trace_man = base / "evidence_traceability_manifest.json"
    if trace_man.exists():
        tm = json.loads(trace_man.read_text(encoding="utf-8"))
        tm_bad = [rel for rel, d in tm.get("outputs_sha256", {}).items() if not (ROOT / rel).exists() or sha256_file(ROOT / rel) != d]
        sm = tm.get("summary", {})
        st = "PASSED" if tm.get("status") == "PASSED" and not tm_bad else "FAILED"
        out.append(item(sec, "source-evidence traceability (4-tier raw-to-rank verification)", st,
                        f"{sm.get('total_evidence_quotes_audited', 0)} quotes audited ({sm.get('verbatim_in_raw_pct', 0)}% verbatim raw); "
                        f"ungrounded={sm.get('failed_ungrounded_count', 0)}" + (f"; hash mismatch: {tm_bad}" if tm_bad else "")))
    elif base == PRIORITIZATION_DIR:
        out.append(item(sec, "source-evidence traceability (4-tier raw-to-rank verification)", "WARN",
                        "traceability manifest not found; run python scripts/verify_evidence_traceability.py"))

    out.append(item(sec, "interpretation", "PASSED" if m.get("human_validated") is False else "FAILED",
                    "exploratory ordering for inspection; counts are entries/texts/records, not respondents; not fitted to human judgments"))
    return out


TRACEABILITY_MANIFEST = PRIORITIZATION_DIR / "evidence_traceability_manifest.json"
# The manifest keys its inputs by file name only; these are the paths documented in
# scripts/verify_evidence_traceability.py (module docstring), which produced it.
TRACEABILITY_INPUT_PATHS = {
    "ranking_evidence.csv": "data/full_corpus/concern_prioritization_v1/ranking_evidence.csv",
    "collective_demand_evidence.csv": "data/full_corpus/concern_prioritization_v1/collective_demand_evidence.csv",
    "representative_examples.csv": "data/full_corpus/concern_discovery_v1/configurations/strip__umap_m25s5/representative_examples.csv",
    "corpus_entries.csv": "data/full_corpus/run_v1/corpus/corpus_entries.csv",
}


def check_evidence_traceability(manifest_path: Path = TRACEABILITY_MANIFEST, root: Path = ROOT) -> list[dict[str, str]]:
    """Read-only: the traceability ledger, its report and its inputs must still match the manifest that recorded them."""
    sec = "11. Source-evidence traceability ledger (verbatim provenance of cited quotes; NOT human-validated)"
    if not manifest_path.exists():
        return [item(sec, "evidence traceability ledger", "NOT RUN", "manifest not found; see scripts/verify_evidence_traceability.py")]
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_out = [rel for rel, d in m.get("outputs_sha256", {}).items() if not (root / rel).exists() or sha256_file(root / rel) != d]
    out = [item(sec, "ledger and report match the traceability manifest", "FAILED" if bad_out or not m.get("outputs_sha256") else "PASSED",
                f"changed/missing: {bad_out}" if bad_out else f"{len(m.get('outputs_sha256', {}))} files verified")]
    unmapped = [k for k in m.get("inputs_sha256", {}) if k not in TRACEABILITY_INPUT_PATHS]
    bad_in = [k for k, d in m.get("inputs_sha256", {}).items()
              if k in TRACEABILITY_INPUT_PATHS and (not (root / TRACEABILITY_INPUT_PATHS[k]).exists()
                                                    or sha256_file(root / TRACEABILITY_INPUT_PATHS[k]) != d)]
    if bad_in:
        out.append(item(sec, "ledger inputs unchanged since the audit", "FAILED", f"inputs changed after the ledger was written: {bad_in}"))
    elif unmapped:
        out.append(item(sec, "ledger inputs unchanged since the audit", "WARN", f"input path not documented for: {unmapped}"))
    else:
        out.append(item(sec, "ledger inputs unchanged since the audit", "PASSED", f"{len(m.get('inputs_sha256', {}))} inputs match"))
    s = m.get("summary", {})
    grounded = m.get("status") == "PASSED" and s.get("failed_ungrounded_count") == 0
    out.append(item(sec, "recorded audit result", "PASSED" if grounded else "FAILED",
                    f"{s.get('total_evidence_quotes_audited')} quotes audited; ungrounded {s.get('failed_ungrounded_count')}; "
                    "provenance check only, not a validation of concerns"))
    return out


LANGUAGE_KEYWORDS_DIR = ROOT / "data" / "full_corpus" / "language_keywords_v1"
LANGUAGE_KEYWORDS_MIN = 15          # documented rule: concern-discovery THRESHOLDS["undersized_min_distinct_groups"]
DESCRIPTIVE_ONLY_PHRASES = ("not validated policy concerns", "not evidence of public importance")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_language_keywords(base: Path = LANGUAGE_KEYWORDS_DIR, inputs_root: Path | None = None) -> list[dict[str, str]]:
    """Read-only invariants of the per-language keyword experiment; never reruns it and never judges keyword quality."""
    sec = "12. Per-language sub-cluster keywords v1 (descriptive only; NOT validated policy concerns)"
    mpath = base / "run_manifest.json"
    if not mpath.exists():
        return [item(sec, "per-language keyword extraction", "NOT RUN", "run_manifest.json not found; see scripts/run_language_keywords.py")]
    inputs_root = inputs_root or base.parent                       # manifest input keys are relative to data/full_corpus/
    m = json.loads(mpath.read_text(encoding="utf-8"))
    bad = [n for n, d in m.get("outputs_sha256", {}).items() if not (base / n).exists() or sha256_file(base / n) != d]
    flags_ok = m.get("status") == "completed" and m.get("human_validated") is False
    out = [item(sec, "keyword outputs match run_manifest.json", "PASSED" if not bad and flags_ok and m.get("outputs_sha256") else "FAILED",
                f"changed/missing: {bad}; status={m.get('status')}; human_validated={m.get('human_validated')}" if bad or not flags_ok
                else f"{len(m['outputs_sha256'])} files verified; human_validated=False")]
    bad_in = [k for k, d in m.get("inputs_sha256", {}).items()
              if not (inputs_root / k).exists() or sha256_file(inputs_root / k) != d]
    labels = [k for k in m.get("inputs_sha256", {}) if k.endswith("labels_seed42.npy")]
    out.append(item(sec, "cluster assignments and other inputs unchanged", "FAILED" if bad_in or not labels else "PASSED",
                    f"changed/missing: {bad_in}" if bad_in else
                    (f"{len(m['inputs_sha256'])} inputs match, including {labels[0]}" if labels else "cluster-assignment input not recorded")))
    problems: list[str] = []
    try:
        cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
        inv = _read_csv_rows(base / "subgroup_inventory.csv")
        kw = _read_csv_rows(base / "language_keywords.csv")
        report = (base / "report.md").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return out + [item(sec, "experiment files readable", "FAILED", str(exc))]
    if cfg.get("min_independent_texts") != LANGUAGE_KEYWORDS_MIN:
        problems.append(f"config minimum is {cfg.get('min_independent_texts')}, documented rule is {LANGUAGE_KEYWORDS_MIN}")
    eligible = set()
    for r in inv:
        is_elig = r["eligible"] == "True"
        if is_elig:
            eligible.add((r["cluster_id"], r["language"]))
            if int(r["independent_texts"]) < LANGUAGE_KEYWORDS_MIN or r["language"] not in ("English", "French") or r["exclusion_reason"]:
                problems.append(f"eligible subgroup violates the rule: cluster {r['cluster_id']} {r['language']}")
        elif not r["exclusion_reason"].strip():
            problems.append(f"excluded subgroup without reason: cluster {r['cluster_id']} {r['language']}")
    n_excl = sum(r["eligible"] != "True" for r in inv)
    out.append(item(sec, "minimum of 15 independent texts and recorded exclusion reasons", "FAILED" if problems else "PASSED",
                    "; ".join(problems[:4]) if problems else f"{len(eligible)} eligible subgroups all >= {LANGUAGE_KEYWORDS_MIN}; "
                    f"{n_excl} excluded subgroups all carry a reason"))
    kw_pairs = {(r["cluster_id"], r["language"]) for r in kw}
    top_n = int(cfg.get("ctfidf", {}).get("top_n", 10))
    kprob = []
    if kw_pairs - eligible:
        kprob.append(f"keywords for non-eligible subgroups: {sorted(kw_pairs - eligible)[:3]}")
    if eligible - kw_pairs:
        kprob.append(f"eligible subgroups without keywords: {sorted(eligible - kw_pairs)[:3]}")
    if any(int(r["rank"]) > top_n or int(r["rank"]) < 1 for r in kw):
        kprob.append(f"rank outside 1..{top_n}")
    out.append(item(sec, "keywords only for eligible subgroups", "FAILED" if kprob else "PASSED",
                    "; ".join(kprob) if kprob else f"{len(kw)} keyword rows across {len(kw_pairs)} eligible subgroups"))
    missing = [p for p in DESCRIPTIVE_ONLY_PHRASES if p not in cfg.get("disclaimer", "") or p not in report]
    out.append(item(sec, "descriptive-only limitations retained", "FAILED" if missing else "PASSED",
                    f"missing in config disclaimer or report: {missing}" if missing
                    else "config and report state the lists are not validated policy concerns and not evidence of public importance"))
    return out


def run_all() -> list[dict[str, str]]:
    return (check_protected() + check_v2_pool() + check_baseline_450()
            + check_annotation_readiness() + check_supervised_blockers() + check_full_corpus_run()
            + check_experiments_v2() + check_concern_discovery() + check_human_review_handoff() + check_prioritization()
            + check_evidence_traceability() + check_language_keywords())


# ── Rendering ─────────────────────────────────────────────────────────────────

def render(results: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    L = ["# PolicyLens research evaluation", "",
         f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
         "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "",
         "## Interpretation rules", ""] + [f"- {x}" for x in INTERPRETATION_RULES] + [""]
    section = ""
    for r in results:
        if r["section"] != section:
            section = r["section"]
            L += ["", f"## {section}", ""]
        L.append(f"- **{r['status']}** — {r['name']}: {r['detail']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--no-write", action="store_true", help="Print only; write no report folder.")
    parser.add_argument("--init-baseline", action="store_true", help="Record SHA-256 of protected artifacts (once).")
    parser.add_argument("--tag", help="Name of the report folder (default: UTC timestamp).")
    args = parser.parse_args(argv)
    if args.init_baseline:
        try:
            data = init_baseline()
        except FileExistsError as exc:
            print(f"[STOP] {exc}", file=sys.stderr)
            return 1
        print(f"Baseline recorded for {len(data['files'])} files -> {BASELINE_PATH}")
        return 0
    results = run_all()
    text = render(results)
    print(text)
    if not args.no_write:
        tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = OUTPUT_BASE / f"run_{tag}"
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"[STOP] Output folder already exists: {out}", file=sys.stderr)
            return 1
        with open(out / "evaluation_report.md", "x", encoding="utf-8", newline="") as f:
            f.write(text)
        with open(out / "evaluation_results.json", "x", encoding="utf-8", newline="") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"written: {out}")
    return 1 if any(r["status"] == "FAILED" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
