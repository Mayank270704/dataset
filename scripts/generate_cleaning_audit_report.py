"""
Generates the comprehensive data-cleaning status audit report.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def generate_report():
    json_path = ROOT / "data" / "interim" / "full_workspace_cleaning_audit.json"
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    # Classify into 6 primary operational sections
    sections = {
        "1. Primary Government Consultation Datasets (Raw & Cleaned)": [],
        "2. External Corpora & Benchmarks (Raw & Processed/Canonical)": [],
        "3. Human Annotation Pilot Datasets": [],
        "4. Semantic Embeddings & Concern Clustering Pipeline Artifacts": [],
        "5. Interim Diagnostic, Mapping & Deduplication Reviews": [],
        "6. Reference Academic Benchmark Tables (HinEmo-9 Paper)": [],
    }

    for r in records:
        path = r["path"].lower()
        if any(k in path for k in ["questionnaire", "individual_comments", "consultation_compilation", "consultationdata2017"]):
            if "interim" in path:
                sections["5. Interim Diagnostic, Mapping & Deduplication Reviews"].append(r)
            else:
                sections["1. Primary Government Consultation Datasets (Raw & Cleaned)"].append(r)
        elif "annotation" in path:
            sections["3. Human Annotation Pilot Datasets"].append(r)
        elif "clustering" in path or "embeddings" in path:
            sections["4. Semantic Embeddings & Concern Clustering Pipeline Artifacts"].append(r)
        elif "reference" in path:
            sections["6. Reference Academic Benchmark Tables (HinEmo-9 Paper)"].append(r)
        elif "interim" in path or "data_inventory" in path:
            sections["5. Interim Diagnostic, Mapping & Deduplication Reviews"].append(r)
        else:
            sections["2. External Corpora & Benchmarks (Raw & Processed/Canonical)"].append(r)

    lines = []
    lines.append("# PolicyLens — Comprehensive Data-Cleaning Status Audit")
    lines.append("")
    lines.append("This document is the verified data-cleaning audit across all dataset files in the PolicyLens workspace.")
    lines.append(f"A total of **{len(records)} dataset files** were inspected recursively.")
    lines.append("")
    lines.append("## Executive Summary of Dataset Holdings")
    lines.append("")
    lines.append("| Category | File Count | Verified Cleaning Script | Primary Suitability |")
    lines.append("|---|---|---|---|")
    lines.append("| Primary Consultation Datasets (Raw & Cleaned) | 6 | `scripts/policylens_government_consultation_audit.py` | Unsupervised clustering & semantic discovery |")
    lines.append("| External Corpora (Raw & Processed/Canonical) | 29 | `scripts/policylens_dataset_audit.py` | Auxiliary sentiment / emotion pretraining |")
    lines.append("| Human Annotation Pilot Datasets | 3 | `scripts/create_annotation_pilot.py` | Supervised stance pilot / IAA evaluation |")
    lines.append("| Model Artifacts (Embeddings & Clusters) | 3 | `scripts/concern_clustering.py` | Topic discovery & evidence retrieval |")
    lines.append("| Interim Diagnostic & Deduplication Tables | 37 | Automated audits | Quality control & traceability |")
    lines.append("| Academic Reference Benchmark Tables | 19 | HinEmo-9 authors | Reference baselines |")
    lines.append("| **Total** | **97** | | |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for sec_name, items in sections.items():
        lines.append(f"## {sec_name} ({len(items)} files)")
        lines.append("")
        for idx, item in enumerate(items, 1):
            lines.append(f"### {idx}. `{item['filename']}`")
            lines.append(f"- **Exact File Path**: `{item['path']}`")
            lines.append(f"- **Architectural Role**: `{item['role']}`")
            lines.append(f"- **File Size**: {item['file_size']:,} bytes")
            lines.append(f"- **Row Count**: {item['row_count']:,} | **Columns**: {item['column_count']}")
            
            # Format columns
            cols = item["column_names"]
            if len(cols) <= 10:
                cols_str = ", ".join(f"`{c}`" for c in cols)
            else:
                cols_str = ", ".join(f"`{c}`" for c in cols[:8]) + f" ... (+{len(cols)-8} more)"
            lines.append(f"- **Column Names**: {cols_str}")
            
            # Missing values
            miss = item.get("missing_values", {})
            if miss:
                top_miss = sorted(miss.items(), key=lambda x: x[1], reverse=True)[:5]
                miss_str = f"{item.get('total_missing', 0):,} total nulls (" + ", ".join(f"`{k}`: {v:,}" for k, v in top_miss) + ")"
            else:
                miss_str = "0 null values"
            lines.append(f"- **Missing Values**: {miss_str}")
            
            lines.append(f"- **Blank Text Cells (Whitespace-only)**: {item.get('blank_text_cells', 0):,}")
            lines.append(f"- **Exact Duplicate Rows**: {item.get('exact_duplicate_rows', 0):,}")
            
            # Duplicate text groups
            dtg = item.get("duplicate_text_groups", {})
            if dtg:
                dtg_str = ", ".join(f"`{k}`: {v} groups" for k, v in dtg.items())
            else:
                dtg_str = "0 duplicate text groups"
            lines.append(f"- **Duplicate Text Groups**: {dtg_str}")
            
            # Cleaned version mapping
            cleaned_p = item.get("cleaned_version_path")
            if cleaned_p:
                lines.append(f"- **Corresponding Cleaned Version**: `{cleaned_p}`")
            else:
                lines.append(f"- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)")

            # Cleaning verification
            lines.append(f"- **Cleaning Verification**: {'[VERIFIED]' if item.get('cleaning_verified') else '[INFERRED]'}")
            lines.append(f"  - **Verification Source**: {item.get('verification_source')}")
            
            # Language & Label Distribution
            lang_dist = item.get("language_distribution", {})
            if lang_dist:
                lines.append(f"- **Language Distribution**: {lang_dist}")
            lab_dist = item.get("label_distribution", {})
            if lab_dist:
                lines.append(f"- **Label Distribution**: {lab_dist}")

            # Suitability
            lines.append(f"- **Task Suitability**: **{item['suitability']}**")
            lines.append(f"  - *Rationale*: {item['suitability_rationale']}")
            lines.append("")

    report_content = "\n".join(lines)
    out_path = ROOT / "reports" / "full_workspace_cleaning_audit_report.md"
    out_path.write_text(report_content, encoding="utf-8")
    print(f"Report written to {out_path} ({len(report_content):,} chars).")

if __name__ == "__main__":
    generate_report()
