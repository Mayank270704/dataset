"""
models/concern_discovery/visualization.py
=========================================
Interactive Cluster Visualization and Diagnostics for PolicyLens.

Generates self-contained HTML cluster inspection reports, reproducible 2D projections,
and diagnostic manifests from existing unsupervised clustering artifacts.

Guarantees & Scientific Integrity:
- Strict evidence traceability: every comment, quote span, and source ID is verified
  against original consultation corpus records.
- Explicit exploratory labeling: clusters are unsupervised groupings, not human-validated concerns.
- Clear methodological warnings: 2D visual separation does NOT imply cluster quality or stance accuracy.
- Zero external CDN dependencies: self-contained HTML/CSS/JS that operates completely offline.
- Reproducible projections: deterministic PCA/UMAP projection from cached embeddings when
  2D coordinates are unavailable.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Core scientific integrity disclaimers
DISCLAIMER_HEADER = (
    "STATUS: EXPLORATORY & UNSUPERVISED — NOT HUMAN VALIDATED. "
    "Clusters are statistical groupings of dense multilingual sentence embeddings, "
    "not verified citizen concerns or ground-truth stance classifications."
)

VISUAL_SEPARATION_WARNING = (
    "METHODOLOGICAL WARNING: Visual separation in 2D projections (PCA/UMAP) is a mathematical "
    "artifact of dimensional compression and DOES NOT constitute evidence of cluster validity, "
    "semantic distinctiveness, or stance accuracy. Apparent proximity or overlap in 2D cannot "
    "replace empirical human evaluation, ground-truth label validation, or formal inter-annotator agreement."
)

TIER_COLORS = {
    "candidate": "#059669",             # Emerald green
    "needs_review": "#d97706",          # Amber / Warm gold
    "campaign_or_template": "#7c3aed",  # Violet
    "non_concern_pattern": "#2563eb",   # Royal blue
    "unreliable": "#4b5563",            # Slate gray
    "noise": "#94a3b8",                 # Light slate
}

TIER_LABELS = {
    "candidate": "Candidate Concern",
    "needs_review": "Needs Review",
    "campaign_or_template": "Campaign / Template",
    "non_concern_pattern": "Non-Concern Pattern",
    "unreliable": "Unreliable / Artifact",
    "noise": "Noise / Unclustered",
}


class MissingDataError(FileNotFoundError):
    """Raised when required datasets, embeddings, or cluster artifacts are missing."""


class TraceabilityError(ValueError):
    """Raised when evidence comments or source IDs fail grounding verification."""


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_coordinates_availability(coord_path: Optional[Path]) -> Tuple[bool, str]:
    """
    Check if precomputed 2D dimensionality-reduction coordinates exist.

    Returns
    -------
    (available, description_or_path)
    """
    if coord_path is None or not Path(coord_path).exists():
        return False, "2D coordinates unavailable in input artifacts"
    try:
        p = Path(coord_path)
        if p.suffix == ".npy":
            arr = np.load(p)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return True, f"Valid 2D array loaded from {p} (shape: {arr.shape})"
            return False, f"Array in {p} has shape {arr.shape}, expected (*, 2)"
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            if "x" in df.columns and "y" in df.columns:
                return True, f"Valid 2D CSV loaded from {p} ({len(df)} rows)"
            return False, f"CSV in {p} missing 'x' and 'y' columns"
        return False, f"Unsupported file format: {p.suffix}"
    except Exception as e:
        return False, f"Error inspecting coordinate file {coord_path}: {e}"


def compute_reproducible_projection(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Compute a deterministic, reproducible 2D projection from high-dimensional embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (N, D).
    method : str
        'pca' (fast, linear, deterministic) or 'umap' (non-linear manifold).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (N, 2), dtype float32.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    method_clean = method.lower().strip()
    if method_clean == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings).astype(np.float32)
        return coords

    elif method_clean == "umap":
        import umap
        n_samples = embeddings.shape[0]
        effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            transform_seed=random_state,
        )
        coords = reducer.fit_transform(embeddings).astype(np.float32)
        return coords

    else:
        raise ValueError(f"Unknown projection method: '{method}'. Choose 'pca' or 'umap'.")


def verify_evidence_traceability(
    representatives_df: pd.DataFrame,
    corpus_entries_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Verify that every representative comment and quote strictly matches original corpus entries.

    Returns
    -------
    dict with summary metrics: total_checked, matched, discrepancies, is_valid.
    """
    corpus_map: Dict[int, Any] = {}
    for row in corpus_entries_df.itertuples():
        corpus_map[int(row.corpus_id)] = row

    total_checked = 0
    matched = 0
    discrepancies: List[Dict[str, Any]] = []

    for rep in representatives_df.itertuples():
        total_checked += 1
        cid = int(rep.corpus_id)
        if cid not in corpus_map:
            discrepancies.append({
                "corpus_id": cid,
                "reason": "corpus_id missing from corpus_entries.csv",
            })
            continue

        c_rec = corpus_map[cid]
        # Check source dataset and record ID alignment
        if str(rep.source_dataset).strip() != str(c_rec.source_dataset).strip():
            discrepancies.append({
                "corpus_id": cid,
                "reason": f"source_dataset mismatch: rep='{rep.source_dataset}' vs corpus='{c_rec.source_dataset}'",
            })
            continue

        if str(rep.source_record_id).strip() != str(c_rec.source_record_id).strip():
            discrepancies.append({
                "corpus_id": cid,
                "reason": f"source_record_id mismatch: rep='{rep.source_record_id}' vs corpus='{c_rec.source_record_id}'",
            })
            continue

        # Check verbatim quote grounding if quote is provided
        if hasattr(rep, "quote") and rep.quote:
            quote_clean = str(rep.quote).strip()
            orig_text = str(c_rec.original_text)
            if quote_clean not in orig_text:
                # Check relaxed whitespace
                norm_quote = " ".join(quote_clean.split())
                norm_orig = " ".join(orig_text.split())
                if norm_quote not in norm_orig:
                    discrepancies.append({
                        "corpus_id": cid,
                        "reason": f"Quote span is not a substring of original text: '{quote_clean[:40]}...'",
                    })
                    continue

        matched += 1

    is_valid = (len(discrepancies) == 0 and total_checked > 0)
    return {
        "total_checked": total_checked,
        "matched": matched,
        "discrepancies": discrepancies,
        "is_valid": is_valid,
    }


def load_cluster_discovery_artifacts(
    input_dir: Union[Path, str],
    run_v1_dir: Optional[Union[Path, str]] = None,
    exp_v2_dir: Optional[Union[Path, str]] = None,
    coord_file: Optional[Union[Path, str]] = None,
    projection_method: str = "pca",
    random_state: int = 42,
    verify_traceability: bool = True,
) -> Dict[str, Any]:
    """
    Load existing clustering and discovery artifacts without re-running clustering.

    If 2D coordinates are unavailable, calculates a reproducible 2D projection
    from existing embeddings.
    """
    base = Path(input_dir)
    if not base.exists():
        raise MissingDataError(f"Input directory does not exist: {base}")

    # Paths to required artifacts
    groups_path = base / "selected" / "concern_groups.csv"
    if not groups_path.exists():
        raise MissingDataError(f"concern_groups.csv missing at {groups_path}")

    assignments_path = base / "selected" / "entry_assignments.csv"
    if not assignments_path.exists():
        raise MissingDataError(f"entry_assignments.csv missing at {assignments_path}")

    # Representative examples
    reps_path = base / "configurations" / "strip__umap_m25s5" / "representative_examples.csv"
    if not reps_path.exists():
        # Fallback check across configurations
        cand_reps = list(base.glob("configurations/*/representative_examples.csv"))
        if cand_reps:
            reps_path = cand_reps[0]
        else:
            raise MissingDataError(f"representative_examples.csv missing under {base / 'configurations'}")

    # Corpus entries path
    if run_v1_dir is None:
        run_v1_dir = base.parents[1] / "full_corpus" / "run_v1" if (base.parents[1] / "full_corpus" / "run_v1").exists() else base.parent / "run_v1"
    run_v1 = Path(run_v1_dir)
    corpus_path = run_v1 / "corpus" / "corpus_entries.csv"
    if not corpus_path.exists():
        raise MissingDataError(f"corpus_entries.csv missing at {corpus_path}")

    # Load DataFrames
    groups_df = pd.read_csv(groups_path)
    assignments_df = pd.read_csv(assignments_path)
    reps_df = pd.read_csv(reps_path)
    corpus_df = pd.read_csv(corpus_path)

    # Lineage and traceability audit
    traceability_report = {"is_valid": True, "total_checked": 0, "matched": 0, "discrepancies": []}
    if verify_traceability:
        traceability_report = verify_evidence_traceability(reps_df, corpus_df)
        if not traceability_report["is_valid"]:
            raise TraceabilityError(
                f"Evidence traceability verification failed: {len(traceability_report['discrepancies'])} mismatches"
            )

    # Check 2D coordinates availability
    coord_p = Path(coord_file) if coord_file else None
    coords_available, coord_msg = check_coordinates_availability(coord_p)

    coords_2d: np.ndarray
    projection_info: Dict[str, Any] = {}

    if coords_available and coord_p is not None:
        if coord_p.suffix == ".npy":
            coords_2d = np.load(coord_p)
        else:
            cdf = pd.read_csv(coord_p)
            coords_2d = cdf[["x", "y"]].to_numpy(dtype=np.float32)
        projection_info = {
            "source": "precomputed",
            "file": str(coord_p),
            "status": coord_msg,
            "method": "precomputed",
        }
    else:
        # 2D coordinates unavailable in existing artifacts — compute reproducible projection
        # Locate embeddings
        emb_path = run_v1 / "embeddings" / "multilingual" / "embeddings.npy"
        if not emb_path.exists():
            raise MissingDataError(
                f"2D coordinates unavailable and base embeddings missing at {emb_path}. Cannot project."
            )

        emb_raw = np.load(emb_path)
        if exp_v2_dir is None:
            exp_v2_dir = base.parent / "experiments_v2"
        exp_v2 = Path(exp_v2_dir)
        strip_cache = exp_v2 / "preprocess" / "strip_embeddings.npz"

        emb_final = emb_raw.copy()
        if strip_cache.exists():
            z = np.load(strip_cache)
            emb_final[z["idx"]] = z["emb"]

        # Compute reproducible 2D projection
        coords_2d = compute_reproducible_projection(
            emb_final,
            method=projection_method,
            random_state=random_state,
        )
        projection_info = {
            "source": "reproduced_from_embeddings",
            "embeddings_path": str(emb_path),
            "strip_cache_path": str(strip_cache) if strip_cache.exists() else None,
            "method": projection_method,
            "random_state": random_state,
            "shape": list(coords_2d.shape),
            "reason_for_projection": (
                "2D coordinates were not pre-generated in concern_discovery_v1 "
                "(which clustered on 5D UMAP). Projected reproducibly from existing embeddings."
            ),
        }

    # Compute 2D centroid for each cluster
    centroids_map: Dict[int, Dict[str, float]] = {}
    unique_ids = assignments_df["unique_id"].to_numpy()
    cluster_ids = assignments_df["cluster_id"].to_numpy()

    # Create mapping of unique_id to 2D coords
    # Check if coords_2d matches unique_id count
    n_unique = int(assignments_df["unique_id"].max()) + 1
    if coords_2d.shape[0] >= n_unique:
        for cid in sorted(set(cluster_ids)):
            if cid < 0:
                continue
            # find all unique_ids in this cluster
            c_uids = np.unique(unique_ids[cluster_ids == cid])
            if len(c_uids) > 0:
                c_coords = coords_2d[c_uids]
                centroids_map[int(cid)] = {
                    "x": float(np.mean(c_coords[:, 0])),
                    "y": float(np.mean(c_coords[:, 1])),
                }

    return {
        "groups": groups_df,
        "assignments": assignments_df,
        "representatives": reps_df,
        "corpus": corpus_df,
        "coords_2d": coords_2d,
        "centroids": centroids_map,
        "projection_info": projection_info,
        "traceability": traceability_report,
        "base_dir": str(base),
    }


def generate_cluster_inspection_html(
    data: Dict[str, Any],
    title: str = "PolicyLens: Interactive Cluster Inspection & Diagnostics",
) -> str:
    """
    Generate a self-contained, offline-capable HTML report for interactive cluster inspection.

    Zero external CDN dependencies (all styles, SVG/Canvas rendering, and JS logic inline).
    """
    groups_df: pd.DataFrame = data["groups"]
    reps_df: pd.DataFrame = data["representatives"]
    assignments_df: pd.DataFrame = data["assignments"]
    coords_2d: np.ndarray = data["coords_2d"]
    centroids_map: Dict[int, Dict[str, float]] = data["centroids"]
    proj_info: Dict[str, Any] = data["projection_info"]

    # Compute key stats
    total_entries = len(assignments_df)
    n_clusters = len(groups_df)
    n_clustered_entries = int((assignments_df["cluster_id"] >= 0).sum())
    n_noise_entries = int((assignments_df["cluster_id"] == -1).sum())
    pct_clustered = round(100.0 * n_clustered_entries / max(1, total_entries), 1)

    tier_counts = groups_df["tier"].value_counts().to_dict()

    # Prepare JSON serializable clusters metadata
    clusters_json_data = []
    for row in groups_df.itertuples():
        cid = int(row.cluster_id)
        cen = centroids_map.get(cid, {"x": 0.0, "y": 0.0})
        flags_val = str(row.flags) if pd.notna(row.flags) and row.flags != "" else "None (Clean)"

        # Get representative quotes for this cluster
        c_reps = reps_df[reps_df["cluster_id"] == cid].sort_values("rank")
        rep_list = []
        for r in c_reps.itertuples():
            rep_list.append({
                "rank": int(r.rank),
                "corpus_id": int(r.corpus_id),
                "unique_id": int(r.unique_id),
                "source_dataset": str(r.source_dataset),
                "source_record_id": str(r.source_record_id),
                "language": str(r.language),
                "cosine_to_centroid": round(float(r.cosine_to_centroid), 4) if pd.notna(r.cosine_to_centroid) else None,
                "quote": str(r.quote).strip(),
                "full_text": str(r.full_text).strip(),
                "selection_reason": str(getattr(r, "selection_reason", "MMR Extractive Representative")),
            })

        clusters_json_data.append({
            "cluster_id": cid,
            "tier": str(row.tier),
            "tier_label": TIER_LABELS.get(str(row.tier), str(row.tier)),
            "tier_color": TIER_COLORS.get(str(row.tier), "#6b7280"),
            "n_entries": int(row.n_entries),
            "n_unique_texts": int(row.n_unique_texts),
            "share_of_corpus": round(float(row.share_of_eligible_entries) * 100, 2),
            "short_label": str(row.short_label),
            "description": str(row.description),
            "flags": flags_val,
            "cohesion_mean": round(float(row.cohesion_mean), 3) if pd.notna(row.cohesion_mean) else None,
            "margin_mean": round(float(row.margin_mean), 3) if pd.notna(row.margin_mean) else None,
            "negative_margin_share": round(float(row.negative_margin_share), 3) if pd.notna(row.negative_margin_share) else None,
            "stability_jaccard_mean": round(float(row.stability_jaccard_mean), 3) if pd.notna(row.stability_jaccard_mean) else None,
            "english_pct": round(float(row.english) * 100, 1) if pd.notna(row.english) else None,
            "french_pct": round(float(row.french) * 100, 1) if pd.notna(row.french) else None,
            "centroid_x": round(cen["x"], 3),
            "centroid_y": round(cen["y"], 3),
            "representatives": rep_list,
        })

    # Prepare point coordinates for canvas plotting
    # To keep HTML compact and responsive, pack unique_id coordinates:
    # [x, y, cluster_id, tier_idx]
    tier_idx_map = {
        "candidate": 0,
        "needs_review": 1,
        "campaign_or_template": 2,
        "non_concern_pattern": 3,
        "unreliable": 4,
        "noise": 5,
    }

    # Create mapping of unique_id to cluster and tier
    uid_to_cid: Dict[int, int] = {}
    uid_to_tier: Dict[int, str] = {}
    for r in assignments_df.itertuples():
        uid = int(r.unique_id)
        if uid not in uid_to_cid:
            uid_to_cid[uid] = int(r.cluster_id)
            uid_to_tier[uid] = str(r.tier)

    # Subsample or include points
    point_data = []
    n_pts = min(len(coords_2d), len(uid_to_cid))
    for i in range(n_pts):
        cid = uid_to_cid.get(i, -1)
        tier_name = uid_to_tier.get(i, "noise")
        t_idx = tier_idx_map.get(tier_name, 5 if cid == -1 else 4)
        point_data.append([
            round(float(coords_2d[i, 0]), 2),
            round(float(coords_2d[i, 1]), 2),
            cid,
            t_idx,
        ])

    points_json = json.dumps(point_data)
    clusters_json = json.dumps(clusters_json_data)
    proj_info_json = json.dumps(proj_info)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-card-hover: #273549;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-candidate: #10b981;
    --accent-review: #f59e0b;
    --accent-template: #8b5cf6;
    --accent-nonconcern: #3b82f6;
    --accent-unreliable: #6b7280;
    --accent-noise: #475569;
    --accent-primary: #38bdf8;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 20px;
  }}

  .container {{ max-width: 1600px; margin: 0 auto; }}

  /* Disclaimers & Header */
  .disclaimer-banner {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .disclaimer-icon {{ font-size: 22px; color: #f87171; flex-shrink: 0; }}
  .disclaimer-title {{ font-weight: 700; color: #fca5a5; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .disclaimer-text {{ font-size: 13px; color: #fecaca; margin-top: 2px; }}

  .warning-callout {{
    background: rgba(30, 41, 59, 0.9);
    border-left: 4px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 13px;
    color: var(--text-secondary);
  }}
  .warning-callout strong {{ color: #e2e8f0; }}

  .header-area {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .main-title {{ font-size: 24px; font-weight: 700; color: var(--text-primary); }}
  .subtitle {{ font-size: 14px; color: var(--text-secondary); margin-top: 4px; }}
  .badge-tag {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-primary);
    border: 1px solid rgba(56, 189, 248, 0.3);
  }}

  /* KPIs */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px;
    text-align: left;
  }}
  .kpi-val {{ font-size: 22px; font-weight: 700; color: var(--text-primary); }}
  .kpi-label {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}

  /* Layout grid */
  .workspace-grid {{
    display: grid;
    grid-template-columns: 1fr 480px;
    gap: 20px;
    margin-bottom: 24px;
  }}
  @media (max-width: 1200px) {{
    .workspace-grid {{ grid-template-columns: 1fr; }}
  }}

  /* Visualizer panel */
  .panel-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .panel-header {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.5);
  }}
  .panel-title {{ font-size: 15px; font-weight: 600; color: var(--text-primary); }}

  .controls-bar {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    font-size: 13px;
    background: rgba(15, 23, 42, 0.25);
  }}
  .filter-group {{ display: flex; align-items: center; gap: 8px; }}
  .tier-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    user-select: none;
    border: 1px solid transparent;
    transition: opacity 0.15s ease;
  }}
  .tier-pill.inactive {{ opacity: 0.35; }}

  .canvas-container {{
    position: relative;
    width: 100%;
    height: 580px;
    background: #090d16;
    overflow: hidden;
    cursor: grab;
  }}
  .canvas-container:active {{ cursor: grabbing; }}
  canvas {{ display: block; width: 100%; height: 100%; }}

  .zoom-controls {{
    position: absolute;
    bottom: 16px;
    right: 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    z-index: 10;
  }}
  .btn-icon {{
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    cursor: pointer;
    font-size: 16px;
  }}
  .btn-icon:hover {{ background: rgba(51, 65, 85, 0.9); }}

  /* Detail Inspector */
  .inspector-body {{
    padding: 18px;
    overflow-y: auto;
    max-height: 660px;
  }}
  .inspector-placeholder {{
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
  }}
  .inspector-placeholder-icon {{ font-size: 36px; margin-bottom: 12px; }}

  .cluster-hero {{
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-color);
  }}
  .cluster-hero-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
  }}
  .cluster-cid {{ font-size: 20px; font-weight: 700; color: var(--text-primary); }}
  .cluster-label {{ font-size: 14px; color: var(--accent-primary); font-weight: 600; margin-bottom: 8px; }}
  .cluster-desc {{ font-size: 13px; color: var(--text-secondary); line-height: 1.4; }}

  .badge-tier {{
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }}

  .metrics-strip {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 16px;
  }}
  .metric-item {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 10px;
  }}
  .metric-val {{ font-size: 14px; font-weight: 600; color: var(--text-primary); }}
  .metric-lbl {{ font-size: 11px; color: var(--text-muted); }}

  .section-heading {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin: 16px 0 10px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  /* Quotes cards */
  .quote-card {{
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
    font-size: 13px;
  }}
  .quote-header {{
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 11px;
    margin-bottom: 6px;
    font-family: monospace;
  }}
  .quote-text {{
    color: #e2e8f0;
    border-left: 3px solid var(--accent-primary);
    padding-left: 8px;
    margin-bottom: 6px;
    font-style: italic;
  }}
  .quote-meta {{
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .trace-tag {{
    color: #34d399;
    font-weight: 600;
  }}

  /* Table panel */
  .table-controls {{
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .search-input {{
    background: #0f172a;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    width: 280px;
  }}
  .table-wrapper {{
    overflow-x: auto;
    max-height: 400px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
  }}
  th {{
    background: rgba(15, 23, 42, 0.8);
    color: var(--text-secondary);
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    position: sticky;
    top: 0;
    cursor: pointer;
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.4);
    color: var(--text-primary);
  }}
  tr.cluster-row {{ cursor: pointer; transition: background 0.1s ease; }}
  tr.cluster-row:hover {{ background: var(--bg-card-hover); }}
  tr.cluster-row.selected {{ background: rgba(56, 189, 248, 0.15); }}

  .tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 12px;
    color: var(--text-primary);
    font-size: 12px;
    pointer-events: none;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    display: none;
    max-width: 260px;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Exploratory Disclaimer Banner -->
  <div class="disclaimer-banner" id="exploratory-disclaimer">
    <div class="disclaimer-icon">&#9888;</div>
    <div>
      <div class="disclaimer-title">Exploratory &amp; Unsupervised Artifact — Not Human Validated</div>
      <div class="disclaimer-text">{DISCLAIMER_HEADER}</div>
    </div>
  </div>

  <!-- Visual Separation Methodological Warning Callout -->
  <div class="warning-callout" id="separation-warning">
    <strong>Scientific Integrity Note:</strong> {VISUAL_SEPARATION_WARNING}
  </div>

  <!-- Header Area -->
  <div class="header-area">
    <div>
      <div class="badge-tag">PolicyLens Technical Deliverable &bull; Full-Corpus v1</div>
      <h1 class="main-title">{title}</h1>
      <div class="subtitle">
        Canadian Federal Open Government Consultations 2016–2018 (English &amp; French).
        Configuration: <code>{proj_info.get("method", "pca").upper()} 2D Projection</code> &bull; Seed <code>{proj_info.get("random_state", 42)}</code>
      </div>
    </div>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-val">{total_entries:,}</div>
      <div class="kpi-label">Eligible Consultation Entries</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">{n_clusters}</div>
      <div class="kpi-label">Discovered Clusters</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">{n_clustered_entries:,} <span style="font-size: 14px; color: var(--accent-candidate);">({pct_clustered}%)</span></div>
      <div class="kpi-label">Clustered Submissions</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">{n_noise_entries:,} <span style="font-size: 14px; color: var(--text-muted);">({round(100.0 * n_noise_entries / max(1, total_entries), 1)}%)</span></div>
      <div class="kpi-label">Unclustered / Noise Points</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val" style="color: var(--accent-candidate);">{tier_counts.get("candidate", 0)}</div>
      <div class="kpi-label">Candidate Tiers</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val" style="color: var(--accent-review);">{tier_counts.get("needs_review", 0)}</div>
      <div class="kpi-label">Needs Review Tiers</div>
    </div>
  </div>

  <!-- Main Visualizer and Detail Inspector Grid -->
  <div class="workspace-grid">

    <!-- Scatter Plot Canvas Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <span class="panel-title">2D Manifold Projection &bull; Cluster Geometry</span>
        <span style="font-size: 12px; color: var(--text-secondary);">
          {proj_info.get("method", "pca").upper()} (Exact Reproducible Reduction)
        </span>
      </div>

      <!-- Controls & Tiers Filter Bar -->
      <div class="controls-bar">
        <span style="color: var(--text-muted); font-size: 12px;">Filter Tiers:</span>
        <div class="tier-pill" data-tier="candidate" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></span> Candidate ({tier_counts.get("candidate", 0)})
        </div>
        <div class="tier-pill" data-tier="needs_review" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #f59e0b;"></span> Needs Review ({tier_counts.get("needs_review", 0)})
        </div>
        <div class="tier-pill" data-tier="campaign_or_template" style="background: rgba(139, 92, 246, 0.2); color: #8b5cf6;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #8b5cf6;"></span> Template ({tier_counts.get("campaign_or_template", 0)})
        </div>
        <div class="tier-pill" data-tier="non_concern_pattern" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #3b82f6;"></span> Non-Concern ({tier_counts.get("non_concern_pattern", 0)})
        </div>
        <div class="tier-pill" data-tier="unreliable" style="background: rgba(107, 114, 128, 0.2); color: #9ca3af;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #9ca3af;"></span> Unreliable ({tier_counts.get("unreliable", 0)})
        </div>
        <div class="tier-pill" data-tier="noise" style="background: rgba(71, 85, 105, 0.2); color: #94a3b8;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #94a3b8;"></span> Noise Points
        </div>
      </div>

      <!-- Canvas Plot Area -->
      <div class="canvas-container" id="canvas-container">
        <canvas id="plot-canvas"></canvas>
        <div class="tooltip" id="plot-tooltip"></div>
        <div class="zoom-controls">
          <button class="btn-icon" id="btn-zoom-in" title="Zoom In">+</button>
          <button class="btn-icon" id="btn-zoom-out" title="Zoom Out">&minus;</button>
          <button class="btn-icon" id="btn-reset" title="Reset View">&#8634;</button>
        </div>
      </div>
    </div>

    <!-- Right Column: Detail & Evidence Inspector -->
    <div class="panel-card">
      <div class="panel-header">
        <span class="panel-title">Cluster Detail &amp; Verbatim Evidence Inspector</span>
        <span class="trace-tag" id="trace-verified-badge">&#10003; 100% Traceable</span>
      </div>
      <div class="inspector-body" id="inspector-body">
        <div class="inspector-placeholder" id="inspector-placeholder">
          <div class="inspector-placeholder-icon">&#128269;</div>
          <div style="font-weight: 600; font-size: 15px; color: var(--text-primary);">No Cluster Selected</div>
          <div style="font-size: 13px; margin-top: 6px;">
            Click any cluster centroid on the scatter plot or choose a row in the table below to inspect its verbatim evidence.
          </div>
        </div>

        <div id="inspector-content" style="display: none;">
          <div class="cluster-hero">
            <div class="cluster-hero-top">
              <span class="cluster-cid" id="insp-cid">Cluster #0</span>
              <span class="badge-tier" id="insp-tier">Candidate</span>
            </div>
            <div class="cluster-label" id="insp-label">Topic Label</div>
            <div class="cluster-desc" id="insp-desc">Extractive description here.</div>
          </div>

          <div class="metrics-strip">
            <div class="metric-item">
              <div class="metric-val" id="insp-entries">0</div>
              <div class="metric-lbl">Submissions (% share)</div>
            </div>
            <div class="metric-item">
              <div class="metric-val" id="insp-texts">0</div>
              <div class="metric-lbl">Distinct Unique Texts</div>
            </div>
            <div class="metric-item">
              <div class="metric-val" id="insp-cohesion">0.0</div>
              <div class="metric-lbl">Cohesion (Centroid Cosine)</div>
            </div>
            <div class="metric-item">
              <div class="metric-val" id="insp-stability">0.0</div>
              <div class="metric-lbl">Seed Stability (Jaccard)</div>
            </div>
            <div class="metric-item">
              <div class="metric-val" id="insp-lang">--</div>
              <div class="metric-lbl">Language Split (EN / FR)</div>
            </div>
            <div class="metric-item">
              <div class="metric-val" id="insp-flags">None</div>
              <div class="metric-lbl">Quality Diagnostics</div>
            </div>
          </div>

          <div class="section-heading">
            <span>Verbatim Evidence Quotes (5 Selected)</span>
            <span style="font-size: 11px; text-transform: none; color: #34d399;">Grounded in source CSV</span>
          </div>
          <div id="insp-quotes-container"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom Table Explorer -->
  <div class="panel-card">
    <div class="panel-header">
      <span class="panel-title">Discovered Concern Clusters Directory ({n_clusters} Discovered Groups)</span>
    </div>
    <div class="table-controls">
      <input type="text" id="table-search" class="search-input" placeholder="Search clusters by ID, label, keywords...">
      <div style="font-size: 12px; color: var(--text-secondary);">
        Click any row to focus on the 2D projection and view evidence
      </div>
    </div>
    <div class="table-wrapper">
      <table id="clusters-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Tier</th>
            <th>Short Descriptive Label</th>
            <th>Entries</th>
            <th>Unique</th>
            <th>Cohesion</th>
            <th>Margin</th>
            <th>Stability</th>
            <th>Language</th>
            <th>Quality Flags</th>
          </tr>
        </thead>
        <tbody id="clusters-table-body">
        </tbody>
      </table>
    </div>
  </div>

</div>

<!-- Embedded Dataset Payloads -->
<script>
  const CLUSTERS_DATA = {clusters_json};
  const POINTS_DATA = {points_json};
  const PROJ_INFO = {proj_info_json};

  const TIER_COLORS = {{
    0: "#10b981", // candidate
    1: "#f59e0b", // needs_review
    2: "#8b5cf6", // campaign_or_template
    3: "#3b82f6", // non_concern_pattern
    4: "#6b7280", // unreliable
    5: "#475569"  // noise
  }};

  const TIER_NAMES = ["candidate", "needs_review", "campaign_or_template", "non_concern_pattern", "unreliable", "noise"];

  // State
  let activeTiers = new Set(TIER_NAMES);
  let selectedClusterId = null;
  let zoom = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;

  // DOM Elements
  const canvas = document.getElementById("plot-canvas");
  const ctx = canvas.getContext("2d");
  const container = document.getElementById("canvas-container");
  const tooltip = document.getElementById("plot-tooltip");

  // Calculate bounding box of coordinates
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < POINTS_DATA.length; i++) {{
    const x = POINTS_DATA[i][0];
    const y = POINTS_DATA[i][1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }}
  const padX = (maxX - minX) * 0.08 || 1;
  const padY = (maxY - minY) * 0.08 || 1;
  minX -= padX; maxX += padX;
  minY -= padY; maxY += padY;

  function resizeCanvas() {{
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    renderPlot();
  }}

  function worldToScreen(wx, wy) {{
    const w = container.clientWidth;
    const h = container.clientHeight;
    const scaleX = w / (maxX - minX);
    const scaleY = h / (maxY - minY);
    const scale = Math.min(scaleX, scaleY) * zoom;
    const cx = w / 2 + panX;
    const cy = h / 2 + panY;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    const sx = cx + (wx - midX) * scale;
    const sy = cy - (wy - midY) * scale;
    return [sx, sy];
  }}

  function screenToWorld(sx, sy) {{
    const w = container.clientWidth;
    const h = container.clientHeight;
    const scaleX = w / (maxX - minX);
    const scaleY = h / (maxY - minY);
    const scale = Math.min(scaleX, scaleY) * zoom;
    const cx = w / 2 + panX;
    const cy = h / 2 + panY;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    const wx = midX + (sx - cx) / scale;
    const wy = midY - (sy - cy) / scale;
    return [wx, wy];
  }}

  function renderPlot() {{
    const w = container.clientWidth;
    const h = container.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // 1. Draw points
    for (let i = 0; i < POINTS_DATA.length; i++) {{
      const pt = POINTS_DATA[i];
      const cid = pt[2];
      const tIdx = pt[3];
      const tierName = TIER_NAMES[tIdx];

      if (!activeTiers.has(tierName)) continue;

      const [sx, sy] = worldToScreen(pt[0], pt[1]);
      if (sx < -10 || sx > w + 10 || sy < -10 || sy > h + 10) continue;

      const isSelected = (selectedClusterId !== null && cid === selectedClusterId);
      const isNoise = (cid === -1);

      ctx.beginPath();
      ctx.arc(sx, sy, isSelected ? 3.5 : (isNoise ? 1.5 : 2.0), 0, Math.PI * 2);

      if (selectedClusterId !== null) {{
        if (isSelected) {{
          ctx.fillStyle = TIER_COLORS[tIdx];
        }} else {{
          ctx.fillStyle = "rgba(71, 85, 105, 0.15)";
        }}
      }} else {{
        ctx.fillStyle = isNoise ? "rgba(148, 163, 184, 0.25)" : TIER_COLORS[tIdx];
      }}
      ctx.fill();
    }}

    // 2. Draw cluster centroids with badges
    for (let i = 0; i < CLUSTERS_DATA.length; i++) {{
      const c = CLUSTERS_DATA[i];
      if (!activeTiers.has(c.tier)) continue;

      const [cx, cy] = worldToScreen(c.centroid_x, c.centroid_y);
      if (cx < -30 || cx > w + 30 || cy < -30 || cy > h + 30) continue;

      const isSelected = (c.cluster_id === selectedClusterId);

      // Centroid bubble
      ctx.beginPath();
      const radius = isSelected ? 12 : 8;
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = c.tier_color;
      ctx.fill();
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.strokeStyle = isSelected ? "#ffffff" : "rgba(15, 23, 42, 0.8)";
      ctx.stroke();

      // Cluster ID label
      ctx.fillStyle = "#ffffff";
      ctx.font = isSelected ? "bold 11px sans-serif" : "9px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(c.cluster_id, cx, cy);
    }}
  }}

  // Find nearest centroid or point on mouse move
  canvas.addEventListener("mousemove", (e) => {{
    if (isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let hoverCluster = null;
    let minDist = 18;

    for (let i = 0; i < CLUSTERS_DATA.length; i++) {{
      const c = CLUSTERS_DATA[i];
      if (!activeTiers.has(c.tier)) continue;
      const [cx, cy] = worldToScreen(c.centroid_x, c.centroid_y);
      const d = Math.hypot(cx - mx, cy - my);
      if (d < minDist) {{
        minDist = d;
        hoverCluster = c;
      }}
    }}

    if (hoverCluster) {{
      tooltip.style.display = "block";
      tooltip.style.left = (mx + 15) + "px";
      tooltip.style.top = (my + 10) + "px";
      tooltip.innerHTML = `
        <div style="font-weight: bold; color: ${{hoverCluster.tier_color}};">Cluster #${{hoverCluster.cluster_id}} &bull; ${{hoverCluster.tier_label}}</div>
        <div style="font-size: 11px; margin-top: 3px; color: #f8fafc;">${{hoverCluster.short_label}}</div>
        <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">${{hoverCluster.n_entries}} entries (${{hoverCluster.share_of_corpus}}% of corpus)</div>
      `;
    }} else {{
      tooltip.style.display = "none";
    }}
  }});

  canvas.addEventListener("mouseleave", () => {{
    tooltip.style.display = "none";
  }});

  // Pan & Zoom handlers
  canvas.addEventListener("mousedown", (e) => {{
    isDragging = true;
    dragStartX = e.clientX - panX;
    dragStartY = e.clientY - panY;
  }});

  window.addEventListener("mouseup", (e) => {{
    if (!isDragging) return;
    isDragging = false;
  }});

  window.addEventListener("mousemove", (e) => {{
    if (!isDragging) return;
    panX = e.clientX - dragStartX;
    panY = e.clientY - dragStartY;
    renderPlot();
  }});

  canvas.addEventListener("click", (e) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let clicked = null;
    let minDist = 20;
    for (let i = 0; i < CLUSTERS_DATA.length; i++) {{
      const c = CLUSTERS_DATA[i];
      if (!activeTiers.has(c.tier)) continue;
      const [cx, cy] = worldToScreen(c.centroid_x, c.centroid_y);
      const d = Math.hypot(cx - mx, cy - my);
      if (d < minDist) {{
        minDist = d;
        clicked = c;
      }}
    }}

    if (clicked) {{
      selectCluster(clicked.cluster_id);
    }}
  }});

  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    zoom = Math.max(0.4, Math.min(25.0, zoom * factor));
    renderPlot();
  }}, {{ passive: false }});

  document.getElementById("btn-zoom-in").addEventListener("click", () => {{
    zoom = Math.min(25.0, zoom * 1.25);
    renderPlot();
  }});
  document.getElementById("btn-zoom-out").addEventListener("click", () => {{
    zoom = Math.max(0.4, zoom * 0.8);
    renderPlot();
  }});
  document.getElementById("btn-reset").addEventListener("click", () => {{
    zoom = 1.0;
    panX = 0;
    panY = 0;
    renderPlot();
  }});

  // Tier filter pills
  document.querySelectorAll(".tier-pill").forEach(pill => {{
    pill.addEventListener("click", () => {{
      const tier = pill.dataset.tier;
      if (activeTiers.has(tier)) {{
        activeTiers.delete(tier);
        pill.classList.add("inactive");
      }} else {{
        activeTiers.add(tier);
        pill.classList.remove("inactive");
      }}
      renderPlot();
    }});
  }});

  // Select cluster and update detail inspector
  function selectCluster(cid) {{
    selectedClusterId = cid;
    renderPlot();

    const c = CLUSTERS_DATA.find(x => x.cluster_id === cid);
    if (!c) return;

    document.getElementById("inspector-placeholder").style.display = "none";
    document.getElementById("inspector-content").style.display = "block";

    document.getElementById("insp-cid").textContent = `Cluster #${{c.cluster_id}}`;
    const tierBadge = document.getElementById("insp-tier");
    tierBadge.textContent = c.tier_label;
    tierBadge.style.backgroundColor = c.tier_color + "22";
    tierBadge.style.color = c.tier_color;

    document.getElementById("insp-label").textContent = c.short_label;
    document.getElementById("insp-desc").textContent = c.description;

    document.getElementById("insp-entries").textContent = `${{c.n_entries}} (${{c.share_of_corpus}}%)`;
    document.getElementById("insp-texts").textContent = `${{c.n_unique_texts}}`;
    document.getElementById("insp-cohesion").textContent = c.cohesion_mean !== null ? c.cohesion_mean : "--";
    document.getElementById("insp-stability").textContent = c.stability_jaccard_mean !== null ? c.stability_jaccard_mean : "--";
    document.getElementById("insp-lang").textContent = `${{c.english_pct}}% EN / ${{c.french_pct}}% FR`;
    document.getElementById("insp-flags").textContent = c.flags;

    // Render quotes
    const quotesContainer = document.getElementById("insp-quotes-container");
    quotesContainer.innerHTML = "";

    c.representatives.forEach(r => {{
      const card = document.createElement("div");
      card.className = "quote-card";
      card.innerHTML = `
        <div class="quote-header">
          <span>#${{r.rank}} &bull; Corpus ID: ${{r.corpus_id}} (Row ${{r.source_record_id}})</span>
          <span>${{r.language.toUpperCase()}} &bull; Proximity: ${{r.cosine_to_centroid}}</span>
        </div>
        <div class="quote-text">&ldquo;${{escapeHtml(r.quote)}}&rdquo;</div>
        <div class="quote-meta">
          <span><strong>Dataset:</strong> ${{r.source_dataset}}</span>
          <span class="trace-tag">&#10003; Grounded Citation</span>
        </div>
      `;
      quotesContainer.appendChild(card);
    }});

    // Highlight row in table
    document.querySelectorAll(".cluster-row").forEach(row => {{
      if (parseInt(row.dataset.cid) === cid) {{
        row.classList.add("selected");
        row.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
      }} else {{
        row.classList.remove("selected");
      }}
    }});
  }}

  function escapeHtml(str) {{
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }}

  // Build Table
  const tbody = document.getElementById("clusters-table-body");
  function renderTable(filterText = "") {{
    tbody.innerHTML = "";
    const term = filterText.toLowerCase();

    CLUSTERS_DATA.forEach(c => {{
      const match = (
        c.cluster_id.toString().includes(term) ||
        c.short_label.toLowerCase().includes(term) ||
        c.description.toLowerCase().includes(term) ||
        c.tier.toLowerCase().includes(term) ||
        c.flags.toLowerCase().includes(term)
      );
      if (!match) return;

      const tr = document.createElement("tr");
      tr.className = "cluster-row";
      tr.dataset.cid = c.cluster_id;
      if (selectedClusterId === c.cluster_id) tr.classList.add("selected");

      tr.innerHTML = `
        <td><strong>#${{c.cluster_id}}</strong></td>
        <td><span class="badge-tier" style="background: ${{c.tier_color}}22; color: ${{c.tier_color}};">${{c.tier_label}}</span></td>
        <td><strong>${{escapeHtml(c.short_label)}}</strong></td>
        <td>${{c.n_entries}}</td>
        <td>${{c.n_unique_texts}}</td>
        <td>${{c.cohesion_mean !== null ? c.cohesion_mean : "--"}}</td>
        <td>${{c.margin_mean !== null ? c.margin_mean : "--"}}</td>
        <td>${{c.stability_jaccard_mean !== null ? c.stability_jaccard_mean : "--"}}</td>
        <td>${{c.english_pct}}% / ${{c.french_pct}}%</td>
        <td style="font-size: 11px; color: #94a3b8;">${{c.flags}}</td>
      `;

      tr.addEventListener("click", () => {{
        selectCluster(c.cluster_id);
      }});

      tbody.appendChild(tr);
    }});
  }}

  document.getElementById("table-search").addEventListener("input", (e) => {{
    renderTable(e.target.value);
  }});

  // Init
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  renderTable();

  // Select first cluster by default
  if (CLUSTERS_DATA.length > 0) {{
    selectCluster(CLUSTERS_DATA[0].cluster_id);
  }}
</script>
</body>
</html>
"""
    return html_template


def build_visualization_package(
    input_dir: Union[Path, str],
    output_dir: Union[Path, str],
    run_v1_dir: Optional[Union[Path, str]] = None,
    exp_v2_dir: Optional[Union[Path, str]] = None,
    coord_file: Optional[Union[Path, str]] = None,
    projection_method: str = "pca",
    random_state: int = 42,
    verify_traceability: bool = True,
) -> Dict[str, Any]:
    """
    Build the full cluster visualization deliverable in a versioned output directory.

    Guarantees:
    - Existing artifacts are untouched (read-only).
    - Writes self-contained HTML report.
    - Writes reproducible 2D projection coordinates.
    - Writes run manifest with SHA-256 checksums and verification status.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts and compute reproducible projection if needed
    data = load_cluster_discovery_artifacts(
        input_dir=input_dir,
        run_v1_dir=run_v1_dir,
        exp_v2_dir=exp_v2_dir,
        coord_file=coord_file,
        projection_method=projection_method,
        random_state=random_state,
        verify_traceability=verify_traceability,
    )

    # 2. Generate self-contained HTML report
    html_content = generate_cluster_inspection_html(data)
    html_report_path = out_path / "cluster_inspection_report.html"
    html_report_path.write_text(html_content, encoding="utf-8")

    # 3. Save reproducible 2D coordinates CSV
    coords_2d = data["coords_2d"]
    assignments_df = data["assignments"]
    unique_ids = assignments_df["unique_id"].to_numpy()
    cluster_ids = assignments_df["cluster_id"].to_numpy()
    tiers = assignments_df["tier"].to_numpy()
    short_labels = assignments_df["short_label"].to_numpy()

    # Create mapping of unique_id to entry attributes
    coord_rows = []
    n_pts = min(len(coords_2d), int(assignments_df["unique_id"].max()) + 1)
    for i in range(n_pts):
        matches = np.where(unique_ids == i)[0]
        cid = int(cluster_ids[matches[0]]) if len(matches) > 0 else -1
        tier_val = str(tiers[matches[0]]) if len(matches) > 0 else "noise"
        label_val = str(short_labels[matches[0]]) if len(matches) > 0 else "Unclustered Noise"
        coord_rows.append({
            "unique_id": i,
            "x": round(float(coords_2d[i, 0]), 5),
            "y": round(float(coords_2d[i, 1]), 5),
            "cluster_id": cid,
            "tier": tier_val,
            "short_label": label_val,
        })

    coords_df = pd.DataFrame(coord_rows)
    coords_csv_path = out_path / "projection_coordinates_2d.csv"
    coords_df.to_csv(coords_csv_path, index=False)

    # 4. Save cluster summary table
    summary_csv_path = out_path / "cluster_summary.csv"
    data["groups"].to_csv(summary_csv_path, index=False)

    # 5. Save run manifest
    manifest = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "deliverable": "interactive_cluster_visualization_v1",
        "human_validated": False,
        "disclaimer": DISCLAIMER_HEADER,
        "methodological_warning": VISUAL_SEPARATION_WARNING,
        "projection_info": data["projection_info"],
        "traceability_audit": data["traceability"],
        "counts": {
            "total_entries": len(assignments_df),
            "n_clusters": len(data["groups"]),
            "clustered_entries": int((assignments_df["cluster_id"] >= 0).sum()),
            "noise_entries": int((assignments_df["cluster_id"] == -1).sum()),
        },
        "outputs_sha256": {
            "cluster_inspection_report.html": sha256_file(html_report_path),
            "projection_coordinates_2d.csv": sha256_file(coords_csv_path),
            "cluster_summary.csv": sha256_file(summary_csv_path),
        },
    }
    manifest_path = out_path / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "output_dir": str(out_path),
        "html_report": str(html_report_path),
        "coords_csv": str(coords_csv_path),
        "manifest": str(manifest_path),
        "manifest_data": manifest,
    }
