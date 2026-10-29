"""
models/concern_discovery/review.py
==================================
Independent, blinded human review of discovered concern clusters, and the analysis of
COMPLETED reviews only.

Task A (cluster coherence, word-intrusion style; Chang et al. 2009 adapted to comments)
    Each item shows six comments: five random members of one cluster (one per near-duplicate
    group) and one "intruder" from a different, not-nearby cluster in the same language when
    possible. The reviewer marks the comment that does not belong (1-6 or U) and rates how
    clearly the remaining comments share one policy concern (0-3 or U). If reviewers find the
    intruder far above chance (1/6), the cluster is distinguishable to people; ratings give a
    graded coherence judgment. Control clusters (flagged non-concern/unreliable) are mixed in
    so the pipeline's own flags can be tested against people.

Task B (summary support), done AFTER task A
    Each item shows the extractive description body (tier wording and citations removed) and
    eight random member comments; the reviewer rates whether the description is supported
    (0-2 or U) and whether it contains a misleading or unsupported claim.

Leakage prevention
    * Templates contain only comment text and blank reviewer columns. No cluster ids, tiers,
      flags, scores, keywords, model or configuration names, corpus ids or selection reasons.
    * Item ids are random tokens; rows are shuffled independently per reviewer.
    * The answer key lives in ``_key_do_not_share/`` and is read only by ``analyze_reviews``
      after BOTH reviewers' files validate. Completed files are checked against the templates,
      so edited comment text or added/removed items are rejected.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.concern_discovery.schema import sha256_file

SUITABLE_TIERS = ("candidate", "needs_review", "campaign_or_template")
CONTROL_TIERS = ("non_concern_pattern", "unreliable")
N_SHOWN_A = 6
N_SHOWN_B = 8
REVIEWERS = ("reviewer_1", "reviewer_2")
A_FILL = ["reviewer_id", "intruder_position", "coherence_of_others", "shared_concern_note", "confidence"]
B_FILL = ["reviewer_id", "summary_support", "misleading_claim", "note", "confidence"]
VALID_A_INTRUDER = {"1", "2", "3", "4", "5", "6", "U"}
VALID_COHERENCE = {"0", "1", "2", "3", "U"}
VALID_SUPPORT = {"0", "1", "2", "U"}
VALID_YES_NO = {"yes", "no"}
VALID_CONFIDENCE = {"high", "medium", "low"}
FORBIDDEN_COLUMN_RE = re.compile(r"cluster|tier|flag|score|cosine|similar|keyword|model|config|stability|corpus_id|"
                                 r"label|cohesion|probab|intruder_(?!position)", re.I)
KEY_DIR = "_key_do_not_share"


# ── package construction ──────────────────────────────────────────────────────

def _pick_members(members: np.ndarray, groups: np.ndarray, texts: list[str], rng: np.random.Generator, n: int,
                  exclude: set[int] = frozenset(), min_chars: int = 20) -> list[int]:
    """Random members, at most one per near-duplicate group, of at least ``min_chars`` characters."""
    cand = [int(u) for u in rng.permutation(members) if int(u) not in exclude and len(texts[int(u)].strip()) >= min_chars]
    out, seen = [], set()
    for u in cand:
        if int(groups[u]) in seen:
            continue
        seen.add(int(groups[u]))
        out.append(u)
        if len(out) == n:
            break
    return out


def sample_clusters(table: pd.DataFrame, rng: np.random.Generator, n_suitable: int, n_controls: int) -> pd.DataFrame:
    suit = table[table["tier"].isin(SUITABLE_TIERS)]
    ctrl = table[table["tier"].isin(CONTROL_TIERS)]
    pick_s = suit.iloc[rng.permutation(len(suit))[:n_suitable]] if len(suit) else suit
    pick_c = ctrl.iloc[rng.permutation(len(ctrl))[:n_controls]] if len(ctrl) else ctrl
    out = pd.concat([pick_s.assign(stratum="suitable"), pick_c.assign(stratum="control")])
    return out.sort_values("cluster_id").reset_index(drop=True)


def build_review_items(table: pd.DataFrame, labels: np.ndarray, unique: pd.DataFrame, emb: np.ndarray,
                       groups: np.ndarray, summaries: dict[int, dict[str, Any]], seed: int,
                       n_suitable: int = 24, n_controls: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (task A items, task A key, task B items, task B key). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    texts = unique["original_text"].tolist()
    langs = unique["language"].tolist()
    sampled = sample_clusters(table, rng, n_suitable, n_controls)
    ids = sorted(int(c) for c in np.unique(labels[labels >= 0]))
    cen = np.vstack([emb[labels == c].mean(axis=0) for c in ids])
    cen /= np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    pos = {c: i for i, c in enumerate(ids)}
    suitable_ids = set(table.loc[table["tier"].isin(SUITABLE_TIERS), "cluster_id"].astype(int))
    a_items, a_key, b_items, b_key = [], [], [], []
    used_tokens: set[str] = set()

    def token(prefix: str) -> str:
        while True:
            t = f"{prefix}-{rng.integers(0, 16 ** 6):06x}"
            if t not in used_tokens:
                used_tokens.add(t)
                return t

    for r in sampled.itertuples():
        c = int(r.cluster_id)
        members = np.where(labels == c)[0]
        shown = _pick_members(members, groups, texts, rng, N_SHOWN_A - 1)
        if len(shown) < 3:
            continue
        maj_lang = pd.Series([langs[u] for u in shown]).mode().iloc[0]
        sims = cen @ cen[pos[c]]
        far = [d for d in ids if d != c and sims[pos[d]] < np.median(np.delete(sims, pos[c])) and d in suitable_ids]
        intruder = None
        for d in rng.permutation(far) if far else []:
            dm = np.where(labels == int(d))[0]
            same = dm[np.array([langs[u] == maj_lang for u in dm])]
            pick = _pick_members(same if len(same) else dm, groups, texts, rng, 1, min_chars=20)
            if pick:
                intruder, intruder_cluster = pick[0], int(d)
                break
        if intruder is None:
            continue
        order = list(rng.permutation(len(shown) + 1))
        seq = (shown + [intruder])
        seq = [seq[i] for i in order]
        item_id = token("A")
        a_items.append({"item_id": item_id, **{f"comment_{i + 1}": texts[u] for i, u in enumerate(seq)},
                        **{f"comment_{i + 1}": "" for i in range(len(seq), N_SHOWN_A)}})
        a_key.append({"item_id": item_id, "cluster_id": c, "tier": r.tier, "stratum": r.stratum,
                      "n_shown": len(seq), "intruder_position": seq.index(intruder) + 1, "intruder_cluster_id": intruder_cluster,
                      **{f"corpus_id_{i + 1}": int(unique.iloc[u]["corpus_id"]) for i, u in enumerate(seq)}})
        s = summaries.get(c)
        if s and s.get("description_body"):
            body = re.sub(r" ?\[corpus_id \d+\]", "", s["description_body"])
            b_shown = _pick_members(members, groups, texts, rng, N_SHOWN_B)
            bid = token("B")
            b_items.append({"item_id": bid, "description": body,
                            **{f"comment_{i + 1}": texts[u] for i, u in enumerate(b_shown)},
                            **{f"comment_{i + 1}": "" for i in range(len(b_shown), N_SHOWN_B)}})
            b_key.append({"item_id": bid, "cluster_id": c, "tier": r.tier, "stratum": r.stratum,
                          **{f"corpus_id_{i + 1}": int(unique.iloc[u]["corpus_id"]) for i, u in enumerate(b_shown)}})
    a_cols = ["item_id"] + [f"comment_{i + 1}" for i in range(N_SHOWN_A)]
    b_cols = ["item_id", "description"] + [f"comment_{i + 1}" for i in range(N_SHOWN_B)]
    ak_cols = ["item_id", "cluster_id", "tier", "stratum", "n_shown", "intruder_position", "intruder_cluster_id"]
    bk_cols = ["item_id", "cluster_id", "tier", "stratum"]
    return (pd.DataFrame(a_items, columns=a_cols), pd.DataFrame(a_key).reindex(columns=ak_cols + [c for c in pd.DataFrame(a_key).columns if c not in ak_cols]),
            pd.DataFrame(b_items, columns=b_cols), pd.DataFrame(b_key).reindex(columns=bk_cols + [c for c in pd.DataFrame(b_key).columns if c not in bk_cols]))


def reviewer_template(items: pd.DataFrame, fill_cols: list[str], seed: int) -> pd.DataFrame:
    t = items.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for col in fill_cols:
        t[col] = ""
    return t


def check_no_leakage(template: pd.DataFrame, key: pd.DataFrame, fill_cols: list[str]) -> list[str]:
    """Return leakage problems in a reviewer template (empty list = OK)."""
    problems = []
    for col in template.columns:
        if col not in fill_cols and FORBIDDEN_COLUMN_RE.search(col):
            problems.append(f"forbidden column '{col}'")
    for col in fill_cols:
        if col in template.columns and (template[col].astype(str).str.strip() != "").any():
            problems.append(f"reviewer column '{col}' is pre-filled")
    if not template["item_id"].astype(str).str.fullmatch(r"[AB]-[0-9a-f]{6}").all():
        problems.append("item ids are not opaque random tokens")
    if set(template["item_id"]) != set(key["item_id"]):
        problems.append("template and key items differ")
    if "description" in template.columns and template["description"].str.contains(
            r"corpus_id|Candidate concern|unvalidated|Campaign/template|Response pattern", regex=True).any():
        problems.append("description shows citations or tier wording")
    return problems


GUIDE = """# Concern-cluster review guide (PolicyLens, Canadian Open Government consultations 2016-18)

You will see comments written by members of the public during Canadian federal Open Government
consultations (English and French). No translation is supplied. Work alone. Do not discuss items with the
other reviewer until both of you have returned both files.

## Task A (do this first): `reviewer_N_task_a.csv`

Each row shows up to six comments. Most of them were grouped together by a computer; one was not.

1. `intruder_position`: the number (1-6) of the comment that least belongs with the others, or `U` if you
   cannot tell.
2. `coherence_of_others`: looking at the remaining comments, how clearly do they share ONE policy concern?
   * `3` the same specific concern or requested change
   * `2` closely related concern (same substantive problem, different details)
   * `1` broad topical overlap only
   * `0` no shared concern
   * `U` unable to judge
   A "concern" is a substantive issue, problem, value or requested change. Shared wording, a shared
   hashtag, thanks for an event, or "I don't know" answers are NOT a shared policy concern: rate them 0 or 1.
3. `shared_concern_note`: if you rated 2 or 3, name the shared concern in a few words.
4. `confidence`: `high`, `medium` or `low`.
5. `reviewer_id`: your reviewer id on every row.

## Task B (only after finishing Task A): `reviewer_N_task_b.csv`

Each row shows an automatically produced description and eight comments from the same group.

1. `summary_support`: `2` the description is supported by the comments; `1` partly supported or too vague;
   `0` not supported; `U` unable to judge.
2. `misleading_claim`: `yes` if the description states something the comments do not support, else `no`.
3. `note`: optional explanation. `confidence` and `reviewer_id` as in Task A.

## Rules

* Fill in only the reviewer columns. Do not edit, add, delete or reorder comments or rows.
* Never leave a judgment blank; use `U`. Blank values are never converted into ratings.
* Save as CSV (UTF-8) in `completed/` as `reviewer_N_task_a.csv` and `reviewer_N_task_b.csv`.
* Do not open `_key_do_not_share/`. It contains the answers and must be withheld from reviewers.
"""


def write_package(out_dir: Path, a_items: pd.DataFrame, a_key: pd.DataFrame, b_items: pd.DataFrame,
                  b_key: pd.DataFrame, seed: int, context: dict[str, Any]) -> dict[str, Any]:
    """Write templates, key and manifest. Refuses to write into an existing package directory."""
    if out_dir.exists():
        raise FileExistsError(f"Review package already exists: {out_dir}")
    (out_dir / KEY_DIR).mkdir(parents=True)
    hashes: dict[str, str] = {}
    problems: list[str] = []
    for k, rid in enumerate(REVIEWERS, 1):
        ta = reviewer_template(a_items, A_FILL, seed + 10 * k)
        tb = reviewer_template(b_items, B_FILL, seed + 10 * k + 1)
        problems += check_no_leakage(ta, a_key, A_FILL) + check_no_leakage(tb, b_key, B_FILL)
        for name, df in ((f"{rid}_task_a.csv", ta), (f"{rid}_task_b.csv", tb)):
            p = out_dir / name
            with open(p, "x", encoding="utf-8-sig", newline="") as f:
                df.to_csv(f, index=False, lineterminator="\n")
            hashes[name] = sha256_file(p)
    if problems:
        raise ValueError(f"Leakage check failed: {problems}")
    for name, df in (("task_a_key.csv", a_key), ("task_b_key.csv", b_key)):
        with open(out_dir / KEY_DIR / name, "x", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, lineterminator="\n")
    with open(out_dir / "review_guide.md", "x", encoding="utf-8", newline="\n") as f:
        f.write(GUIDE)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
        "task_a_items": int(len(a_items)), "task_b_items": int(len(b_items)),
        "strata_task_a": a_key["stratum"].value_counts().to_dict() if len(a_key) else {},
        "tiers_task_a": a_key["tier"].value_counts().to_dict() if len(a_key) else {},
        "template_sha256": hashes, "leakage_check": "PASSED", "completed_dir": "completed/",
        "key_dir": KEY_DIR + "/ (withhold from reviewers)", "chance_rate_task_a": "1/n_shown per item",
        **context,
    }
    with open(out_dir / "package_manifest.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return manifest


# ── analysis of COMPLETED reviews ─────────────────────────────────────────────

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_completed(completed: pd.DataFrame, template: pd.DataFrame, fill_cols: list[str],
                       valid: dict[str, set[str]], name: str) -> list[str]:
    problems = []
    if list(completed.columns) != list(template.columns):
        return [f"{name}: columns differ from the template"]
    if len(completed) != len(template) or set(completed["item_id"]) != set(template["item_id"]):
        return [f"{name}: items differ from the template"]
    if completed["item_id"].duplicated().any():
        problems.append(f"{name}: duplicated item ids")
    fixed = [c for c in template.columns if c not in fill_cols and c != "item_id"]
    norm = lambda v: str(v).replace("\r\n", "\n").replace("\r", "\n")  # noqa: E731  (CRLF from spreadsheets only)
    a = completed.set_index("item_id")[fixed].sort_index().map(norm)
    b = template.set_index("item_id")[fixed].sort_index().map(norm)
    if not a.equals(b):
        changed = [f"{i}/{c}" for i, row in (a != b).iterrows() for c in fixed if row[c]]
        problems.append(f"{name}: comment/description text was modified in {len(changed)} cell(s), e.g. {changed[:3]}")
    if (completed["reviewer_id"].str.strip() == "").any():
        problems.append(f"{name}: blank reviewer_id")
    for col, ok in valid.items():
        vals = completed[col].str.strip()
        vals = vals.str.lower() if ok <= {"yes", "no", "high", "medium", "low"} else vals.str.upper()
        bad = sorted(set(vals) - ok)
        if bad:
            problems.append(f"{name}: invalid values in {col}: {bad[:5]}")
    return problems


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 4), round(float(c + h), 4)


def _kappa(a: pd.Series, b: pd.Series, weights: str | None = None) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    m = (a != "U") & (b != "U")
    if m.sum() < 2 or len(set(a[m]) | set(b[m])) < 2:
        return None
    if weights:
        return round(float(cohen_kappa_score(a[m].astype(int), b[m].astype(int), weights=weights)), 4)
    return round(float(cohen_kappa_score(a[m], b[m])), 4)


def review_status(package_dir: Path) -> dict[str, Any]:
    """Read-only status: BLOCKED (package or completed files missing) or READY. Never opens the key."""
    if not (package_dir / "package_manifest.json").is_file():
        return {"status": "BLOCKED", "reason": "review package not built", "missing": ["package_manifest.json"]}
    completed = package_dir / "completed"
    need = [completed / f"{r}_task_{t}.csv" for r in REVIEWERS for t in ("a", "b")]
    missing = [p.name for p in need if not p.is_file()]
    if missing:
        return {"status": "BLOCKED", "reason": "completed reviewer files missing", "missing": missing,
                "expected_dir": str(completed)}
    return {"status": "READY"}


def analyze_reviews(package_dir: Path, out_dir: Path, cluster_table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute review metrics from completed files only. Returns {'status': 'BLOCKED', ...} if any file is missing."""
    st = review_status(package_dir)
    if st["status"] != "READY":
        return st
    completed = package_dir / "completed"
    problems: list[str] = []
    data: dict[tuple[str, str], pd.DataFrame] = {}
    va = {"intruder_position": VALID_A_INTRUDER, "coherence_of_others": VALID_COHERENCE, "confidence": VALID_CONFIDENCE}
    vb = {"summary_support": VALID_SUPPORT, "misleading_claim": VALID_YES_NO, "confidence": VALID_CONFIDENCE}
    for r in REVIEWERS:
        for t, fill, valid in (("a", A_FILL, va), ("b", B_FILL, vb)):
            comp = _read(completed / f"{r}_task_{t}.csv")
            tmpl = _read(package_dir / f"{r}_task_{t}.csv")
            problems += validate_completed(comp, tmpl, fill, valid, f"{r}_task_{t}")
            for col in valid:
                comp[col] = comp[col].str.strip()
                comp[col] = comp[col].str.lower() if valid[col] <= {"yes", "no", "high", "medium", "low"} else comp[col].str.upper()
            data[(r, t)] = comp.set_index("item_id")
    if problems:
        return {"status": "INVALID", "problems": problems}
    if out_dir.exists():
        raise FileExistsError(f"Analysis output already exists: {out_dir}")
    key_a = _read(package_dir / KEY_DIR / "task_a_key.csv").set_index("item_id")
    key_b = _read(package_dir / KEY_DIR / "task_b_key.csv").set_index("item_id")
    r1a, r2a = data[("reviewer_1", "a")].loc[key_a.index], data[("reviewer_2", "a")].loc[key_a.index]
    r1b, r2b = data[("reviewer_1", "b")].loc[key_b.index], data[("reviewer_2", "b")].loc[key_b.index]
    truth = key_a["intruder_position"].astype(str)
    chance = (1 / key_a["n_shown"].astype(int)).mean()
    res: dict[str, Any] = {"status": "COMPLETE", "task_a_items": int(len(key_a)), "task_b_items": int(len(key_b)),
                           "chance_rate_task_a": round(float(chance), 4), "per_reviewer": {}}
    merged = key_a[["cluster_id", "tier", "stratum"]].copy()
    for name, ra in (("reviewer_1", r1a), ("reviewer_2", r2a)):
        judged = ra["intruder_position"] != "U"
        correct = (ra["intruder_position"] == truth) & judged
        merged[f"{name}_correct"] = np.where(judged, correct.astype(int).astype(str), "U")
        merged[f"{name}_coherence"] = ra["coherence_of_others"]
        k, n = int(correct.sum()), int(judged.sum())
        per = {"intrusion_accuracy": round(k / n, 4) if n else None, "intrusion_ci95": wilson(k, n), "judged": n,
               "unable": int((~judged).sum())}
        for stratum in ("suitable", "control"):
            m = (key_a["stratum"] == stratum) & judged
            per[f"intrusion_accuracy_{stratum}"] = round(float(correct[m].mean()), 4) if m.any() else None
            per[f"n_{stratum}"] = int(m.sum())
        res["per_reviewer"][name] = per
    res["agreement_task_a"] = {
        "intruder_choice_kappa": _kappa(r1a["intruder_position"], r2a["intruder_position"]),
        "intruder_choice_exact": round(float((r1a["intruder_position"] == r2a["intruder_position"]).mean()), 4),
        "coherence_quadratic_kappa": _kappa(r1a["coherence_of_others"], r2a["coherence_of_others"], "quadratic"),
        "coherence_exact": round(float((r1a["coherence_of_others"] == r2a["coherence_of_others"]).mean()), 4),
    }
    res["agreement_task_b"] = {
        "support_quadratic_kappa": _kappa(r1b["summary_support"], r2b["summary_support"], "quadratic"),
        "support_exact": round(float((r1b["summary_support"] == r2b["summary_support"]).mean()), 4),
        "both_rated_supported": int(((r1b["summary_support"] == "2") & (r2b["summary_support"] == "2")).sum()),
        "any_misleading": int(((r1b["misleading_claim"] == "yes") | (r2b["misleading_claim"] == "yes")).sum()),
        "misleading_kappa": _kappa(r1b["misleading_claim"], r2b["misleading_claim"]),
    }
    coh = pd.concat([pd.to_numeric(r1a["coherence_of_others"], errors="coerce"),
                     pd.to_numeric(r2a["coherence_of_others"], errors="coerce")], axis=1).mean(axis=1)
    merged["mean_coherence"] = coh
    res["coherence_by_tier"] = merged.groupby("tier")["mean_coherence"].agg(["mean", "count"]).round(3).to_dict("index")
    if cluster_table is not None and len(merged):
        from scipy.stats import spearmanr
        ct = cluster_table.set_index("cluster_id")
        corr = {}
        for metric in ("cohesion_mean", "stability_jaccard_mean", "keyword_support_top3_mean", "negative_margin_share"):
            x = merged["cluster_id"].astype(int).map(ct[metric]) if metric in ct.columns else None
            m = merged["mean_coherence"].notna() & (x.notna() if x is not None else False)
            if x is not None and m.sum() >= 5:
                rho, p = spearmanr(x[m], merged.loc[m, "mean_coherence"])
                corr[metric] = {"spearman_rho": round(float(rho), 4), "p": round(float(p), 4), "n": int(m.sum())}
        res["internal_metric_vs_human_coherence"] = corr
    out_dir.mkdir(parents=True)
    merged.reset_index().to_csv(out_dir / "merged_task_a.csv", index=False, lineterminator="\n")
    with open(out_dir / "review_metrics.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return res
