"""
models/stance_classification/pipeline.py
========================================
Leakage-safe supervised STANCE classification baselines (stance toward Open Government, guidelines v1.1).
Sentiment and topic are never used as labels or features.

Models (identical preprocessing, identical split, identical protocol)
    logreg   Pipeline([TfidfVectorizer, LogisticRegression(class_weight="balanced")])
    linsvm   Pipeline([TfidfVectorizer, LinearSVC(class_weight="balanced")])
Hyper-parameters are fixed in the configuration and never tuned on the held-out test set.

Leakage control
    Records are grouped by union-find over: same source record (dataset + record id), the pool's exact-duplicate
    group, the corpus near-duplicate group from concern discovery (when the record maps to the corpus), the
    template key, and lexical near-duplicates within the labelled set (character 5-gram Jaccard >= 0.5 or
    containment >= 0.9). A whole group is always on one side of the split. The split is StratifiedGroupKFold
    (first fold = test) with a fixed seed; TF-IDF is fitted inside the Pipeline on training data only.

Saving
    Artifacts are written only after both models were fitted AND evaluated on a valid held-out set; the run
    directory is write-once and every artifact is listed with its SHA-256 in metadata.json. load_trained_pipeline
    re-verifies the hash before unpickling.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from models.concern_discovery.duplicates import char_shingles, containment, jaccard, template_key
from models.stance_classification.labels import SCHEMA_LABELS

MODELS = ("logreg", "linsvm")
DEFAULT_CONFIG: dict[str, Any] = {
    "tfidf": {"lowercase": True, "ngram_range": [1, 2], "min_df": 1, "max_features": 20000, "sublinear_tf": True,
              "token_pattern": r"(?u)\b\w\w+\b", "strip_accents": None},
    "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
    "linsvm": {"C": 1.0, "max_iter": 5000, "class_weight": "balanced"},
    "test_size": 0.25, "seed": 42, "cv_folds_on_train": 5, "bootstrap": 1000,
    "min_gold": 40, "min_class_count": 5, "min_test_records": 10, "exclude_labels": [],
    "near_dup_jaccard": 0.5, "near_dup_containment": 0.9,
}


class TrainingBlocked(RuntimeError):
    """Raised (and reported) when genuine labels are missing or insufficient; nothing is fitted or saved."""


# ── pipelines ─────────────────────────────────────────────────────────────────

def build_pipeline(model: str, cfg: dict[str, Any] = DEFAULT_CONFIG) -> Pipeline:
    t = dict(cfg["tfidf"])
    t["ngram_range"] = tuple(t["ngram_range"])
    if model == "logreg":
        clf = LogisticRegression(random_state=cfg["seed"], **cfg["logreg"])
    elif model == "linsvm":
        clf = LinearSVC(random_state=cfg["seed"], **cfg["linsvm"])
    else:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    return Pipeline([("tfidf", TfidfVectorizer(**t)), ("clf", clf)])


# ── leakage groups ────────────────────────────────────────────────────────────

class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def leakage_groups(df: pd.DataFrame, corpus_group: pd.Series | None = None, jac: float = 0.5, cont: float = 0.9) -> np.ndarray:
    """Group id per row; rows that share a record, duplicate group, corpus near-dup group, template or near-dup text are joined."""
    n = len(df)
    uf = _UF(n)
    first: dict[str, int] = {}

    def link(key: str, i: int) -> None:
        j = first.setdefault(key, i)
        if j != i:
            uf.union(i, j)

    texts = df["original_text"].tolist()
    for i, r in enumerate(df.itertuples()):
        link(f"rec:{r.source_dataset}#{r.source_record_id}", i)
        if str(getattr(r, "duplicate_text_group_id", "")).strip():
            link(f"dup:{r.duplicate_text_group_id}", i)
        if corpus_group is not None and pd.notna(corpus_group.iloc[i]):
            link(f"cd:{int(corpus_group.iloc[i])}", i)
        k = template_key(texts[i])
        if k:
            link(f"tk:{k}", i)
    sh = [char_shingles(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sh[i], sh[j]) >= jac or containment(sh[i], sh[j]) >= cont:
                uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)])


def grouped_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified group split: first fold of StratifiedGroupKFold(n_splits = round(1 / test_size))."""
    k = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    return tr, te


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], groups: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):                                  # cluster (group) bootstrap of the test set
        take = np.concatenate([idx_by_g[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        boots.append(f1_score(y_true[take], y_pred[take], labels=labels, average="macro", zero_division=0))
    return {"macro_f1": round(macro, 4),
            "macro_f1_ci95_group_bootstrap": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                                "support": int(s[i])} for i, lab in enumerate(labels)},
            "confusion_matrix": {"labels": labels, "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
            "accuracy": round(float((y_true == y_pred).mean()), 4)}


def grouped_cv_on_train(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str],
                        cfg: dict[str, Any]) -> dict[str, Any]:
    """Grouped stratified CV inside the training split only (secondary estimate; no tuning)."""
    k = min(cfg["cv_folds_on_train"], int(pd.Series(y).value_counts().min()), len(np.unique(groups)))
    if k < 2:
        return {"status": "not_run", "reason": "too few records per class or groups for grouped CV"}
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    scores = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = build_pipeline(model, cfg).fit(X[tr], y[tr])
        scores.append(f1_score(y[te], pipe.predict(X[te]), labels=labels, average="macro", zero_division=0))
    return {"status": "ok", "folds": len(scores), "macro_f1_mean": round(float(np.mean(scores)), 4),
            "macro_f1_std": round(float(np.std(scores)), 4)}


# ── training orchestration ────────────────────────────────────────────────────

def prepare(gold: pd.DataFrame, cfg: dict[str, Any], corpus_group: pd.Series | None = None) -> dict[str, Any]:
    """Select usable gold records, form leakage groups and split. Raises TrainingBlocked if evaluation is impossible."""
    d = gold[gold["gold_stance"] != ""].copy().reset_index(drop=True)
    cg = None if corpus_group is None else corpus_group[gold["gold_stance"] != ""].reset_index(drop=True)
    bad = sorted(set(d["gold_stance"]) - set(SCHEMA_LABELS))
    if bad:
        raise ValueError(f"labels outside the stance schema: {bad}")
    dropped: dict[str, Any] = {}
    if cfg["exclude_labels"]:
        m = d["gold_stance"].isin(cfg["exclude_labels"])
        dropped["excluded_by_config"] = d.loc[m, "gold_stance"].value_counts().to_dict()
        d, cg = d[~m].reset_index(drop=True), (None if cg is None else cg[~m.values].reset_index(drop=True))
    counts = d["gold_stance"].value_counts()
    rare = counts[counts < cfg["min_class_count"]]
    if len(rare):
        dropped["below_min_class_count"] = rare.to_dict()
        m = d["gold_stance"].isin(rare.index)
        d, cg = d[~m].reset_index(drop=True), (None if cg is None else cg[~m.values].reset_index(drop=True))
    labels = [lab for lab in SCHEMA_LABELS if lab in set(d["gold_stance"])]
    if len(d) < cfg["min_gold"] or len(labels) < 2:
        raise TrainingBlocked(f"insufficient genuine labels: {len(d)} usable gold records in {len(labels)} class(es) "
                              f"(need >= {cfg['min_gold']} records and >= 2 classes with >= {cfg['min_class_count']} each)")
    groups = leakage_groups(d, cg, cfg["near_dup_jaccard"], cfg["near_dup_containment"])
    y = d["gold_stance"].to_numpy()
    tr, te = grouped_split(y, groups, cfg["test_size"], cfg["seed"])
    missing_in_train = sorted(set(y[te]) - set(y[tr]))
    if len(te) < cfg["min_test_records"] or len(set(y[te])) < 2 or missing_in_train:
        raise TrainingBlocked(f"no valid held-out set: {len(te)} test records, test classes {sorted(set(y[te]))}, "
                              f"classes absent from training {missing_in_train}")
    if set(groups[tr]) & set(groups[te]):
        raise AssertionError("leakage: a group appears in both train and test")
    d["leakage_group"] = groups
    d["split"] = np.where(np.isin(np.arange(len(d)), te), "test", "train")
    return {"data": d, "labels": labels, "train_idx": tr, "test_idx": te, "dropped": dropped}


def train_and_evaluate(prep: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    d, labels, tr, te = prep["data"], prep["labels"], prep["train_idx"], prep["test_idx"]
    X, y, g = d["original_text"].to_numpy(), d["gold_stance"].to_numpy(), d["leakage_group"].to_numpy()
    out: dict[str, Any] = {"models": {}, "pipelines": {}}
    maj = pd.Series(y[tr]).value_counts().idxmax()
    out["majority_class_baseline"] = {"predicts": maj, **evaluate(y[te], np.full(len(te), maj, dtype=object), labels, g[te], cfg["bootstrap"], cfg["seed"])}
    for m in MODELS:
        pipe = build_pipeline(m, cfg).fit(X[tr], y[tr])
        out["pipelines"][m] = pipe
        out["models"][m] = {"test": evaluate(y[te], pipe.predict(X[te]), labels, g[te], cfg["bootstrap"], cfg["seed"]),
                            "grouped_cv_on_train": grouped_cv_on_train(m, X[tr], y[tr], g[tr], labels, cfg),
                            "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_)}
    dist = {s: d.loc[d["split"] == s, "gold_stance"].value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
            for s in ("train", "test")}
    dist["all"] = d["gold_stance"].value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
    out["class_distribution"] = dist
    out["n_train"], out["n_test"] = int(len(tr)), int(len(te))
    out["n_groups"] = int(len(np.unique(g)))
    return out


# ── saving / loading ──────────────────────────────────────────────────────────

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_run(run_dir: Path, prep: dict[str, Any], result: dict[str, Any], cfg: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Write-once save of a run that has been fitted AND evaluated."""
    import joblib
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if set(result.get("pipelines", {})) != set(MODELS) or any(
            "macro_f1" not in result["models"][m]["test"] for m in MODELS):
        raise ValueError("refusing to save: both models must be fitted and evaluated")
    run_dir.mkdir(parents=True)
    arts = {}
    for m, pipe in result["pipelines"].items():
        p = run_dir / f"pipeline_{m}.joblib"
        joblib.dump(pipe, p)
        arts[p.name] = _sha(p)
    files = {
        "label_mapping.json": {"labels": prep["labels"], "index": {lab: i for i, lab in enumerate(prep["labels"])},
                               "schema": SCHEMA_LABELS, "construct": "stance toward the Open Government initiative/policy (guidelines v1.1); not sentiment, not topic"},
        "preprocessing_config.json": cfg,
        "metrics.json": {k: v for k, v in result.items() if k != "pipelines"} | {"dropped": prep["dropped"]},
    }
    for name, obj in files.items():
        (run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        arts[name] = _sha(run_dir / name)
    cols = ["pilot_id", "round", "source_dataset", "source_record_id", "source_row_number", "source_field", "question_slug",
            "gold_stance", "label_source", "leakage_group", "split", "original_text"]
    prep["data"][[c for c in cols if c in prep["data"].columns]].to_csv(run_dir / "split_assignments.csv", index=False, lineterminator="\n")
    arts["split_assignments.csv"] = _sha(run_dir / "split_assignments.csv")
    meta = {"status": "trained_and_evaluated", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__, "python": platform.python_version(), "models": list(MODELS),
            "artifacts_sha256": arts, "human_labels_only": True, **provenance}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return meta


def load_trained_pipeline(run_dir: Path, model: str) -> Pipeline:
    """Load a saved pipeline only if the run is complete and the file matches its recorded SHA-256."""
    import joblib
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "trained_and_evaluated":
        raise ValueError(f"{run_dir} is not a completed training run")
    p = run_dir / f"pipeline_{model}.joblib"
    if _sha(p) != meta["artifacts_sha256"].get(p.name):
        raise ValueError(f"{p.name} does not match its recorded SHA-256; refusing to unpickle")
    return joblib.load(p)
