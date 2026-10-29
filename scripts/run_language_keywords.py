"""
scripts/run_language_keywords.py
================================
c-TF-IDF per-language sub-cluster keyword extraction on the EXISTING selected concern clusters
(registry next action of "c-TF-IDF Cluster Keywords"). See models/concern_clustering/language_keywords.py.

    python scripts/run_language_keywords.py                 # writes data/full_corpus/language_keywords_v1/ (write-once)
    python scripts/run_language_keywords.py --out <folder>

Reads (hash-verified, never modified): run_v1 unique texts, the selected configuration's seed-42 labels, and the
concern-discovery near-duplicate groups, cluster diagnostics and concern groups. Nothing is re-embedded or re-clustered.
Keyword lists are descriptive; they are not validated policy concerns or evidence of public importance.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.concern_clustering import language_keywords as lk  # noqa: E402
from models.concern_clustering.clustering import CTFIDF_MAX_FEATURES, CTFIDF_NGRAM_RANGE, CTFIDF_TOP_N_WORDS  # noqa: E402
from models.concern_discovery.duplicates import strip_social  # noqa: E402
from models.concern_discovery.schema import sha256_file  # noqa: E402

CD = ROOT / "data" / "full_corpus" / "concern_discovery_v1"
EXP_V2 = ROOT / "data" / "full_corpus" / "experiments_v2"
RUN_V1 = ROOT / "data" / "full_corpus" / "run_v1"
OUT = ROOT / "data" / "full_corpus" / "language_keywords_v1"
DISCLAIMER = ("Keyword lists describe frequent, distinctive wording within each language subgroup of an unsupervised "
              "cluster. They are not validated policy concerns, not human-reviewed, and not evidence of public importance.")


class InputError(ValueError):
    pass


def load_inputs(cd: Path = CD, exp_v2: Path = EXP_V2, run_v1: Path = RUN_V1) -> dict[str, Any]:
    man = json.loads((cd / "run_manifest.json").read_text(encoding="utf-8"))
    sel = man["selected_configuration"]
    checks, inputs = [], {}
    for rel in ("duplicates/near_duplicate_groups.csv", f"configurations/{sel}/cluster_diagnostics.csv", "selected/concern_groups.csv"):
        p = cd / rel
        if not p.is_file() or sha256_file(p) != man["outputs_sha256"].get(rel):
            raise InputError(f"concern_discovery_v1/{rel} missing or differs from its run manifest")
        inputs[f"concern_discovery_v1/{rel}"] = man["outputs_sha256"][rel]
    lab_rel = f"experiments_v2/{sel}/labels_seed42.npy"
    lab_p = exp_v2 / sel / "labels_seed42.npy"
    if not lab_p.is_file() or sha256_file(lab_p) != man["inputs_sha256"].get(lab_rel):
        raise InputError(f"{lab_rel} missing or differs from the concern-discovery input record")
    inputs[lab_rel] = man["inputs_sha256"][lab_rel]
    cm = json.loads((run_v1 / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8"))
    up = run_v1 / "corpus" / "unique_texts.csv"
    if sha256_file(up) != cm["unique_texts_sha256"]:
        raise InputError("run_v1 unique_texts.csv differs from the corpus manifest")
    inputs["run_v1/corpus/unique_texts.csv"] = cm["unique_texts_sha256"]
    checks.append({"check": "inputs match their manifests", "status": "PASSED", "detail": f"{len(inputs)} files; selected {sel}"})
    unique = pd.read_csv(up, dtype=str, keep_default_na=False)
    labels = np.load(lab_p)
    dups = pd.read_csv(cd / "duplicates/near_duplicate_groups.csv", dtype=str, keep_default_na=False)
    diag = pd.read_csv(cd / f"configurations/{sel}/cluster_diagnostics.csv", keep_default_na=False)
    groups = pd.read_csv(cd / "selected/concern_groups.csv", keep_default_na=False)
    if not (len(unique) == len(labels) == len(dups)) or dups["unique_id"].astype(int).tolist() != list(range(len(unique))):
        raise InputError("unique texts, labels and near-duplicate groups are not aligned")
    if set(diag["cluster_id"].astype(int)) != set(int(c) for c in np.unique(labels[labels >= 0])):
        raise InputError("cluster ids in the diagnostics differ from the labels")
    checks.append({"check": "alignment", "status": "PASSED",
                   "detail": f"{len(unique):,} unique texts; {len(diag)} clusters; labels unchanged (read-only)"})
    return {"selected": sel, "unique": unique, "labels": labels, "dup": dups["dup_group_id"].astype(int).to_numpy(),
            "diag": diag, "groups": groups, "inputs_sha256": inputs, "checks": checks}


def write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(text)


def run(out: Path = OUT, cd: Path = CD, exp_v2: Path = EXP_V2, run_v1: Path = RUN_V1) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    t0 = time.perf_counter()
    d = load_inputs(cd, exp_v2, run_v1)
    texts = [strip_social(t) for t in d["unique"]["original_text"]]           # the text variant of the selected configuration
    inv = lk.subgroup_inventory(d["unique"], d["labels"], d["dup"])
    meta = d["groups"].set_index("cluster_id")[["tier", "short_label"]]
    inv = inv.join(meta, on="cluster_id")
    kw, check = lk.language_keywords(texts, d["unique"], d["labels"], d["dup"], inv)
    if not all(v["identical_to_extract_cluster_topics"] in (True, None) for v in check.values()):
        raise AssertionError(f"selection differs from extract_cluster_topics: {check}")
    base = {int(r.cluster_id): json.loads(r.keywords) for r in d["diag"].itertuples()}
    comp = lk.compare_with_cluster_keywords(kw, base)
    out.mkdir(parents=True)
    write_new(out / "subgroup_inventory.csv", inv.to_csv(index=False, lineterminator="\n"))
    write_new(out / "language_keywords.csv", kw.to_csv(index=False, lineterminator="\n"))
    write_new(out / "cluster_comparison.csv", comp.to_csv(index=False, lineterminator="\n"))
    cfg = {"selected_configuration": d["selected"], "text_variant": "URLs, @mentions and #hashtags removed (strip_social), as for the selected configuration",
           "unit": "unique texts of one (cluster, language) subgroup; language = source metadata of the text's first occurrence",
           "languages": list(lk.LANGUAGES), "min_independent_texts": lk.MIN_INDEPENDENT_TEXTS,
           "min_rule_source": "models/concern_discovery/diagnostics.py THRESHOLDS['undersized_min_distinct_groups'] (existing)",
           "independent_texts": "distinct near-duplicate groups (concern_discovery_v1/duplicates)",
           "ctfidf": {"implementation": "models/concern_clustering/clustering.py::_ctfidf_matrix (unchanged)",
                      "top_n": CTFIDF_TOP_N_WORDS, "max_features": CTFIDF_MAX_FEATURES, "ngram_range": list(CTFIDF_NGRAM_RANGE),
                      "stop_words": "existing English + French list", "comparison_set": "eligible subgroups of the same language"},
           "disclaimer": DISCLAIMER}
    write_new(out / "config.json", json.dumps(cfg, indent=2) + "\n")
    write_new(out / "input_checks.json", json.dumps(d["checks"] + [{"check": "selection identical to extract_cluster_topics",
                                                                     "status": "PASSED", "detail": check}], indent=2, default=str) + "\n")
    summary = {lang: {"subgroups_present": int(((inv["language"] == lang) & (inv["independent_texts"] > 0)).sum()),
                      "eligible": int(((inv["language"] == lang) & inv["eligible"]).sum()),
                      "independent_texts_total": int(inv.loc[inv["language"] == lang, "independent_texts"].sum()),
                      "independent_texts_in_eligible": int(inv.loc[(inv["language"] == lang) & inv["eligible"], "independent_texts"].sum())}
               for lang in [*lk.LANGUAGES, "Other/Unknown"]}
    write_new(out / "summary.json", json.dumps({"by_language": summary, "keywords": int(len(kw)),
                                                "single_text_terms": int(kw["single_text_term"].sum()),
                                                "spans_text_boundary_terms": int(kw["spans_text_boundary"].sum()), "equivalence": check},
                                               indent=2, default=str) + "\n")
    write_report(out, cfg, inv, kw, comp, summary)
    files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "completed",
                "human_validated": False, "inputs_sha256": d["inputs_sha256"],
                "outputs_sha256": {p.name: sha256_file(p) for p in files},
                "python": platform.python_version(), "runtime_s": round(time.perf_counter() - t0, 2)}
    write_new(out / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return {"summary": summary, "keywords": int(len(kw)), "equivalence": check, "runtime_s": manifest["runtime_s"]}


def write_report(out: Path, cfg: dict, inv: pd.DataFrame, kw: pd.DataFrame, comp: pd.DataFrame, summary: dict) -> None:
    def terms(c: int, lang: str, n: int = 6) -> str:
        k = kw[(kw["cluster_id"] == c) & (kw["language"] == lang)].head(n)
        return ", ".join(f"{r.term} ({round(100 * r.support_share_unique_texts)}%{', spans texts' if r.spans_text_boundary else ''})"
                         for r in k.itertuples()) or "—"

    L = ["# Per-language sub-cluster keywords (c-TF-IDF), v1", "",
         f"**{DISCLAIMER}**", "",
         "## Method", "",
         f"- Clusters: the existing selected configuration `{cfg['selected_configuration']}` (concern discovery v1). Assignments are read, not changed.",
         f"- Unit: {cfg['unit']}. Text: {cfg['text_variant']}.",
         f"- Rule: a subgroup is eligible with at least {cfg['min_independent_texts']} independent texts (distinct near-duplicate groups), "
         "the existing concern-discovery minimum for a whole cluster. Other/Unknown language texts are never eligible.",
         f"- Keywords: existing c-TF-IDF (`_ctfidf_matrix`, top {cfg['ctfidf']['top_n']}, max {cfg['ctfidf']['max_features']} features, "
         "1-2-grams, existing English + French stop words), computed separately per language, each eligible subgroup one class document. "
         "Term selection was checked to be identical to `extract_cluster_topics`.",
         "- Support: measured with the same analyzer as c-TF-IDF (stop words removed before 2-grams are formed): share of the "
         "subgroup's unique texts containing the term, and the number of independent texts containing it. A term in exactly one "
         "independent text is flagged `single_text_term`; a term in none is flagged `spans_text_boundary` (see Limitations).", "",
         "## Coverage and exclusions", "",
         "| Language | Subgroups with texts | Eligible | Independent texts | In eligible subgroups |", "|---|---:|---:|---:|---:|"]
    for lang, s in summary.items():
        L.append(f"| {lang} | {s['subgroups_present']} | {s['eligible']} | {s['independent_texts_total']:,} | {s['independent_texts_in_eligible']:,} |")
    ex = inv[~inv["eligible"]]
    L += ["", "Excluded subgroups by reason: " + "; ".join(f"{r} ({n})" for r, n in ex["exclusion_reason"].value_counts().items()) +
          ". The full list is in `subgroup_inventory.csv`.", "",
          "## French subgroups (all eligible)", "",
          "| Cluster | Tier | Cluster label | FR independent texts | Top French terms (support) |", "|---:|---|---|---:|---|"]
    fr = inv[(inv["language"] == "French") & inv["eligible"]].sort_values("independent_texts", ascending=False)
    for r in fr.itertuples():
        L.append(f"| {r.cluster_id} | {r.tier} | {r.short_label or '—'} | {r.independent_texts} | {terms(r.cluster_id, 'French')} |")
    L += ["", "## English subgroups (largest 10 of all eligible)", "",
          "| Cluster | Tier | Cluster label | EN independent texts | Top English terms (support) |", "|---:|---|---|---:|---|"]
    for r in inv[(inv["language"] == "English") & inv["eligible"]].sort_values("independent_texts", ascending=False).head(10).itertuples():
        L.append(f"| {r.cluster_id} | {r.tier} | {r.short_label or '—'} | {r.independent_texts} | {terms(r.cluster_id, 'English')} |")
    both = comp[comp["french_terms"] > 0]
    L += ["", "## Comparison with the existing all-language cluster keywords", "",
          f"- English top terms shared with the cluster's existing keywords: mean {comp['english_overlap_with_cluster_keywords'].mean():.1f} of "
          f"{comp['english_terms'].mean():.1f} across {len(comp)} clusters.",
          f"- French top terms shared with the cluster's existing keywords: mean {both['french_overlap_with_cluster_keywords'].mean():.1f} of "
          f"{both['french_terms'].mean():.1f} across {len(both)} clusters with an eligible French subgroup." if len(both) else "- No eligible French subgroups.",
          f"- Clusters whose English and French top terms share at least one term: "
          f"{int((both['english_french_shared_terms'].fillna('') != '').sum())} of {len(both)}.",
          f"- Keyword rows: {len(kw)}; `single_text_term`: {int(kw['single_text_term'].sum())}; "
          f"`spans_text_boundary`: {int(kw['spans_text_boundary'].sum())}.", "",
          "## Limitations", "",
          "- Language is source metadata: questionnaire rows carry the survey language, so some French comments are listed as English and vice versa.",
          "- French comparison sets are small (only eligible French subgroups), so French weights are less stable than English ones; "
          "most French subgroups are excluded by the minimum rule.",
          "- c-TF-IDF is bag-of-words: a distinctive term need not state the concern, and shared stop-word lists may leave function words in bigrams.",
          "- The existing implementation joins a subgroup's texts with a space before counting, so some 2-grams combine the last word of one "
          "comment with the first word of the next (for example `thanks latest`). They occur in no single comment, are kept so that the "
          "selection stays identical to the existing method, and are flagged `spans_text_boundary`. The existing all-language keywords "
          "have the same property.",
          "- Clusters are unsupervised and not human-validated; per-language keywords inherit every limitation of the concern-discovery run.", "",
          "## Reproduction", "", "```powershell", "python scripts\\run_language_keywords.py",
          "python -m pytest tests\\test_language_keywords.py -q -p no:cacheprovider", "```", "",
          "Outputs are write-once; `run_manifest.json` records SHA-256 of every input and output (except itself)."]
    write_new(out / "report.md", "\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Per-language sub-cluster c-TF-IDF keywords")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    try:
        print(json.dumps(run(a.out), indent=2, default=str))
    except FileExistsError as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 1
    except InputError as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
