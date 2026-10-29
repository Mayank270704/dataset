"""
scripts/generate_cluster_visualization.py
=========================================
CLI tool to generate self-contained interactive cluster visualization and diagnostics
for the PolicyLens research project.

Usage:
    python scripts/generate_cluster_visualization.py
    python scripts/generate_cluster_visualization.py --projection-method pca --seed 42
    python scripts/generate_cluster_visualization.py --output-dir data/full_corpus/cluster_visualization_v1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.concern_discovery.visualization import (
    DISCLAIMER_HEADER,
    VISUAL_SEPARATION_WARNING,
    build_visualization_package,
    check_coordinates_availability,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate interactive cluster visualization and diagnostics report."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "full_corpus" / "concern_discovery_v1",
        help="Path to concern_discovery_v1 output directory (default: data/full_corpus/concern_discovery_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "full_corpus" / "cluster_visualization_v1",
        help="Output directory for versioned visualization artifacts (default: data/full_corpus/cluster_visualization_v1)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "cluster_visualization_v1",
        help="Reports directory for HTML report copy (default: reports/cluster_visualization_v1)",
    )
    parser.add_argument(
        "--run-v1-dir",
        type=Path,
        default=ROOT / "data" / "full_corpus" / "run_v1",
        help="Path to run_v1 directory with corpus and embeddings",
    )
    parser.add_argument(
        "--exp-v2-dir",
        type=Path,
        default=ROOT / "data" / "full_corpus" / "experiments_v2",
        help="Path to experiments_v2 directory with stripped embedding cache",
    )
    parser.add_argument(
        "--coord-file",
        type=Path,
        default=None,
        help="Optional path to precomputed 2D coordinates (.npy or .csv)",
    )
    parser.add_argument(
        "--projection-method",
        type=str,
        default="pca",
        choices=["pca", "umap"],
        help="Projection method if 2D coordinates are unavailable (default: pca)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for projection reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-traceability",
        action="store_true",
        help="Skip strict evidence traceability verification",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PolicyLens: Interactive Cluster Inspection & Diagnostics Generator")
    print("=" * 80)
    print(f"Disclaimers:")
    print(f"  [1] {DISCLAIMER_HEADER}")
    print(f"  [2] {VISUAL_SEPARATION_WARNING}")
    print("-" * 80)

    # Check coordinate availability beforehand
    coords_avail, coord_msg = check_coordinates_availability(args.coord_file)
    if coords_avail:
        print(f"Coordinate status: Available ({coord_msg})")
    else:
        print(f"Coordinate status: Unavailable ({coord_msg}).")
        print(f"  Action: Computing reproducible {args.projection_method.upper()} 2D projection (seed={args.seed}) from existing embeddings.")

    t0 = time.time()
    try:
        res = build_visualization_package(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            run_v1_dir=args.run_v1_dir,
            exp_v2_dir=args.exp_v2_dir,
            coord_file=args.coord_file,
            projection_method=args.projection_method,
            random_state=args.seed,
            verify_traceability=not args.no_traceability,
        )
    except Exception as e:
        print(f"\nERROR: Failed to generate visualization: {e}", file=sys.stderr)
        return 1

    elapsed = round(time.time() - t0, 2)

    # Copy HTML report to reports/ directory as well
    if args.reports_dir:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        report_dest = args.reports_dir / "cluster_inspection_report.html"
        shutil.copy2(res["html_report"], report_dest)
        print(f"Report copy saved to: {report_dest}")

    manifest_data = res["manifest_data"]
    counts = manifest_data["counts"]
    trace = manifest_data["traceability_audit"]

    print("\nGeneration Completed Successfully:")
    print(f"  Total Entries Processed : {counts['total_entries']:,}")
    print(f"  Clusters Discovered      : {counts['n_clusters']}")
    print(f"  Clustered Submissions    : {counts['clustered_entries']:,}")
    print(f"  Noise / Unclustered      : {counts['noise_entries']:,}")
    print(f"  Traceability Verification: {trace['matched']}/{trace['total_checked']} representative quotes perfectly grounded (is_valid: {trace['is_valid']})")
    print(f"  Execution Time           : {elapsed} seconds")
    print("\nOutput Artifacts:")
    print(f"  HTML Report  : {res['html_report']}")
    print(f"  Coordinates  : {res['coords_csv']}")
    print(f"  Manifest     : {res['manifest']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
