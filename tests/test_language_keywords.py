"""
tests/test_language_keywords.py
===============================
Per-language sub-cluster c-TF-IDF keywords (models/concern_clustering/language_keywords.py and
scripts/run_language_keywords.py). Synthetic data except one read-only check of the real inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from models.concern_clustering import language_keywords as lk  # noqa: E402
from models.concern_discovery.diagnostics import THRESHOLDS  # noqa: E402

EN_WORDS = ["housing", "transit", "budget", "clinics", "schools", "rivers", "farms", "ports", "courts", "museums",
            "parks", "roads", "grants", "audits", "pensions", "permits", "libraries", "bridges", "forests", "tariffs"]
FR_WORDS = ["logement", "transport", "budget", "cliniques", "écoles", "rivières", "fermes", "ports", "tribunaux", "musées",
            "parcs", "routes", "subventions", "audits", "pensions", "permis", "bibliothèques", "ponts", "forêts", "tarifs"]


def synth(n_en=(20, 20), n_fr=(15, 14), dup_fr0=0):
    """Two clusters; cluster c has n_en[c] English and n_fr[c] French texts; dup_fr0 French texts of cluster 0 share one group."""
    rows, labels = [], []
    topic_en = {0: "privacy protection", 1: "electoral reform"}
    topic_fr = {0: "vie privée protection", 1: "réforme électorale"}
    for c in (0, 1):
        for i in range(n_en[c]):
            rows.append(("English", f"{topic_en[c]} matters for {EN_WORDS[i % 20]} and {EN_WORDS[(i + 7) % 20]} {i}"))
            labels.append(c)
        for i in range(n_fr[c]):
            rows.append(("French", f"{topic_fr[c]} importante pour {FR_WORDS[i % 20]} et {FR_WORDS[(i + 3) % 20]} {i}"))
            labels.append(c)
    rows.append(("S", "other language text here"))
    labels.append(0)
    rows.append(("English", "noise text never clustered"))
    labels.append(-1)
    unique = pd.DataFrame(rows, columns=["language", "original_text"])
    unique["multiplicity"] = 1
    dup = np.arange(len(unique))
    if dup_fr0:
        fr0 = [i for i, (lg, lab) in enumerate(zip(unique["language"], labels)) if lg == "French" and lab == 0][:dup_fr0]
        dup[fr0] = fr0[0]
    return unique, np.array(labels), dup


def test_min_rule_is_the_existing_threshold():
    assert lk.MIN_INDEPENDENT_TEXTS == THRESHOLDS["undersized_min_distinct_groups"] == 15


def test_inventory_rule_counts_independent_texts():
    unique, labels, dup = synth(n_fr=(15, 14))
    inv = lk.subgroup_inventory(unique, labels, dup).set_index(["cluster_id", "language"])
    assert inv.loc[(0, "English"), "eligible"] and inv.loc[(0, "French"), "eligible"]
    assert not inv.loc[(1, "French"), "eligible"] and "fewer than 15" in inv.loc[(1, "French"), "exclusion_reason"]
    assert not inv.loc[(0, "Other/Unknown"), "eligible"] and "never eligible" in inv.loc[(0, "Other/Unknown"), "exclusion_reason"]
    assert -1 not in inv.index.get_level_values(0)
    unique, labels, dup = synth(n_fr=(16, 14), dup_fr0=3)                    # 16 unique but 14 independent French texts
    inv = lk.subgroup_inventory(unique, labels, dup).set_index(["cluster_id", "language"])
    assert inv.loc[(0, "French"), "unique_texts"] == 16 and inv.loc[(0, "French"), "independent_texts"] == 14
    assert not inv.loc[(0, "French"), "eligible"]


def test_empty_language_subgroup_is_listed_and_misalignment_rejected():
    unique, labels, dup = synth(n_fr=(15, 0))
    inv = lk.subgroup_inventory(unique, labels, dup).set_index(["cluster_id", "language"])
    assert inv.loc[(1, "French"), "independent_texts"] == 0 and inv.loc[(1, "French"), "exclusion_reason"] == "no texts in this language"
    with pytest.raises(ValueError):
        lk.subgroup_inventory(unique, labels[:-1], dup)


def test_keywords_match_existing_method_and_stay_within_language():
    unique, labels, dup = synth(n_fr=(15, 15))
    before = labels.copy()
    inv = lk.subgroup_inventory(unique, labels, dup)
    kw, check = lk.language_keywords(unique["original_text"].tolist(), unique, labels, dup, inv)
    assert check["English"]["identical_to_extract_cluster_topics"] and check["French"]["identical_to_extract_cluster_topics"]
    assert np.array_equal(labels, before)                                     # assignments untouched
    fr0 = set(kw[(kw["cluster_id"] == 0) & (kw["language"] == "French")]["term"])
    en0 = set(kw[(kw["cluster_id"] == 0) & (kw["language"] == "English")]["term"])
    assert "privée" in fr0 and "privacy" in en0 and "privacy" not in fr0
    assert set(kw["language"]) == {"English", "French"}
    assert (kw.groupby(["cluster_id", "language"])["rank"].max() <= 10).all()
    assert ((kw["support_share_unique_texts"] >= 0) & (kw["support_share_unique_texts"] <= 1)).all()


def test_ineligible_french_subgroup_gets_no_keywords():
    unique, labels, dup = synth(n_fr=(15, 14))
    inv = lk.subgroup_inventory(unique, labels, dup)
    kw, check = lk.language_keywords(unique["original_text"].tolist(), unique, labels, dup, inv)
    assert check["French"]["subgroups"] == 1
    assert kw[(kw["language"] == "French") & (kw["cluster_id"] == 1)].empty


def test_support_uses_the_ctfidf_analyzer():
    a = lk.ctfidf_analyzer()
    terms = [set(a("Difficile de trouver l'information")), set(a("rien ici"))]
    assert "difficile trouver" in terms[0]                                   # stop word removed before the 2-gram
    share, n = lk._support(terms, [1, 2], "difficile trouver")
    assert (share, n) == (0.5, 1)
    assert lk._support(terms, [1, 2], "trouver information rien") == (0.0, 0)


def test_boundary_terms_are_flagged():
    unique = pd.DataFrame({"language": ["English"] * 32, "multiplicity": 1,
                           "original_text": [f"alpha {EN_WORDS[i % 20]} thanks" if i % 2 == 0 else f"latest beta {EN_WORDS[i % 20]}"
                                             for i in range(16)] + [f"gamma delta {EN_WORDS[i % 20]} {i}" for i in range(16)]})
    labels = np.array([0] * 16 + [1] * 16)
    dup = np.arange(32)
    inv = lk.subgroup_inventory(unique, labels, dup)
    kw, _ = lk.language_keywords(unique["original_text"].tolist(), unique, labels, dup, inv, top_n=40)
    row = kw[(kw["cluster_id"] == 0) & (kw["term"] == "thanks latest")]
    assert len(row) == 1 and bool(row["spans_text_boundary"].iloc[0]) and row["support_independent_texts"].iloc[0] == 0
    assert not kw.loc[kw["support_independent_texts"] > 0, "spans_text_boundary"].any()


def test_comparison_with_cluster_keywords():
    kw = pd.DataFrame({"cluster_id": [0, 0, 0, 1], "language": ["English", "English", "French", "English"],
                       "term": ["privacy", "data", "données", "vote"]})
    c = lk.compare_with_cluster_keywords(kw, {0: ["privacy", "security"], 1: ["vote"]}).set_index("cluster_id")
    assert c.loc[0, "english_overlap_with_cluster_keywords"] == 1 and c.loc[0, "french_overlap_with_cluster_keywords"] == 0
    assert pd.isna(c.loc[1, "french_overlap_with_cluster_keywords"])


# ── end to end on the synthetic concern-discovery tree ───────────────────────

@pytest.fixture()
def tree(tmp_path):
    import run_concern_discovery as rcd
    import test_concern_discovery as tcd
    run_v1, exp = tcd._build_tree(tmp_path)
    cd = tmp_path / "cd"
    rcd.stage_run(cd, run_v1, exp)
    return cd, exp, run_v1


def test_run_end_to_end_write_once_and_manifest(tree, tmp_path):
    import run_language_keywords as rlk
    cd, exp, run_v1 = tree
    lab = exp / json.loads((cd / "run_manifest.json").read_text())["selected_configuration"] / "labels_seed42.npy"
    lab_hash = rlk.sha256_file(lab)
    out = tmp_path / "lk"
    res = rlk.run(out, cd, exp, run_v1)
    assert all(v["identical_to_extract_cluster_topics"] in (True, None) for v in res["equivalence"].values())
    for f in ("config.json", "input_checks.json", "subgroup_inventory.csv", "language_keywords.csv", "cluster_comparison.csv",
              "summary.json", "report.md", "run_manifest.json"):
        assert (out / f).exists(), f
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["human_validated"] is False
    assert all(rlk.sha256_file(out / n) == h for n, h in man["outputs_sha256"].items())
    assert rlk.sha256_file(lab) == lab_hash                                  # labels unchanged
    assert "not validated policy concerns" in (out / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        rlk.run(out, cd, exp, run_v1)


def test_tampered_inputs_are_rejected(tree, tmp_path):
    import run_language_keywords as rlk
    cd, exp, run_v1 = tree
    p = cd / "duplicates" / "near_duplicate_groups.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(rlk.InputError, match="manifest"):
        rlk.run(tmp_path / "lk", cd, exp, run_v1)
    assert not (tmp_path / "lk").exists()


@pytest.mark.skipif(not (ROOT / "data" / "full_corpus" / "concern_discovery_v1" / "run_manifest.json").exists(),
                    reason="real concern-discovery outputs not present")
def test_real_inputs_pass_integrity_checks():
    import run_language_keywords as rlk
    d = rlk.load_inputs()
    assert d["selected"] == "strip__umap_m25s5" and len(d["unique"]) == 10043
    assert all(c["status"] == "PASSED" for c in d["checks"])
