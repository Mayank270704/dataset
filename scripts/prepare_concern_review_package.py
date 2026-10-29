"""Build a blinded, reproducible human-review package from the saved 450 sample."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "data" / "comparisons" / "multilingual_embedding_450"
REVIEW_DIR = COMPARISON_DIR / "human_review"
PAIR_LIMIT = 2
ORDER_SEED = 20261012
LANGUAGE_PAIRS = (("English", "English"), ("French", "French"), ("English", "French"))
MODEL_DIRS = {
    "all_minilm_l6_v2": "all_minilm_l6_v2",
    "paraphrase_multilingual_minilm_l12_v2": "paraphrase_multilingual_minilm_l12_v2",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(
    comparison_dir: Path = COMPARISON_DIR,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Load and validate the saved comparison artifacts without changing them."""
    metadata_path = PROJECT_ROOT / "data" / "embeddings" / "corpus_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Saved sample metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if len(metadata) != 450:
        raise ValueError(f"Expected 450 saved sample records; found {len(metadata)}.")
    if metadata["corpus_id"].astype(int).tolist() != list(range(450)):
        raise ValueError("Saved sample corpus IDs are not in the expected 0..449 order.")

    embeddings: dict[str, np.ndarray] = {}
    labels: dict[str, pd.DataFrame] = {}
    for model_key, model_dir in MODEL_DIRS.items():
        model_path = comparison_dir / model_dir
        embedding_path = model_path / "embeddings.npy"
        labels_path = model_path / "cluster_labels.csv"
        if not embedding_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Comparison artifacts are incomplete for {model_key}: "
                f"{embedding_path}, {labels_path}"
            )
        vectors = np.load(embedding_path)
        assignments = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        if vectors.ndim != 2 or vectors.shape[0] != len(metadata):
            raise ValueError(f"Embedding rows do not align with sample for {model_key}.")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Embeddings contain non-finite values for {model_key}.")
        if assignments["corpus_id"].astype(int).tolist() != list(range(450)):
            raise ValueError(f"Cluster labels are not aligned for {model_key}.")
        if assignments["original_text"].tolist() != metadata["original_text"].tolist():
            raise ValueError(f"Cluster label text differs from sample for {model_key}.")
        if len(assignments) != len(metadata):
            raise ValueError(f"Cluster-label count differs from sample for {model_key}.")
        embeddings[model_key] = vectors.astype(np.float64, copy=False)
        labels[model_key] = assignments
    return metadata, embeddings, labels


def _oriented_pair(
    left_index: int,
    right_index: int,
    metadata: pd.DataFrame,
) -> tuple[int, int]:
    """Keep English first for English-French pairs; otherwise order by corpus ID."""
    left_language = metadata.iloc[left_index]["language"]
    right_language = metadata.iloc[right_index]["language"]
    if left_language == "French" and right_language == "English":
        return right_index, left_index
    left_id = int(metadata.iloc[left_index]["corpus_id"])
    right_id = int(metadata.iloc[right_index]["corpus_id"])
    return (left_index, right_index) if left_id <= right_id else (right_index, left_index)


def select_model_pairs(
    model_key: str,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    pair_limit: int = PAIR_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Select top- and bottom-cosine pairs within deterministic review strata."""
    if pair_limit < 1:
        raise ValueError("pair_limit must be at least 1.")
    if len(metadata) != len(embeddings) or len(metadata) != len(assignments):
        raise ValueError("Metadata, embeddings, and assignments must align row-for-row.")

    vector_norms = np.linalg.norm(embeddings, axis=1)
    if np.any(vector_norms == 0):
        raise ValueError(f"Zero-length embeddings found for {model_key}.")
    vectors = embeddings / vector_norms[:, None]
    cluster_ids = assignments["cluster_id"].astype(int).to_numpy()
    languages = metadata["language"].tolist()

    candidates: dict[tuple[str, str, str], list[tuple[float, int, int]]] = {}
    for i in range(len(metadata)):
        if cluster_ids[i] < 0 or languages[i] not in {"English", "French"}:
            continue
        for j in range(i + 1, len(metadata)):
            if cluster_ids[j] < 0 or languages[j] not in {"English", "French"}:
                continue
            lang_pair = tuple(sorted((languages[i], languages[j])))
            if lang_pair not in LANGUAGE_PAIRS:
                continue
            relationship = "within_cluster" if cluster_ids[i] == cluster_ids[j] else "cross_cluster"
            left, right = _oriented_pair(i, j, metadata)
            score = float(np.dot(vectors[left], vectors[right]))
            candidates.setdefault((relationship, *lang_pair), []).append(
                (score, left, right)
            )

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for relationship in ("within_cluster", "cross_cluster"):
        for lang_pair in LANGUAGE_PAIRS:
            key = (relationship, *lang_pair)
            pool = candidates.get(key, [])
            ordered = sorted(
                pool,
                key=lambda item: (
                    item[0],
                    int(metadata.iloc[item[1]]["corpus_id"]),
                    int(metadata.iloc[item[2]]["corpus_id"]),
                ),
            )
            bottom = ordered[:pair_limit]
            bottom_pairs = {(item[1], item[2]) for item in bottom}
            top = [
                item for item in reversed(ordered)
                if (item[1], item[2]) not in bottom_pairs
            ][:pair_limit]
            strata = (
                ("likely_dissimilar", bottom, "ascending"),
                ("likely_similar", top, "descending"),
            )
            stratum_counts["|".join(key)] = {
                "candidate_count": len(pool),
                "selected_likely_dissimilar": len(bottom),
                "selected_likely_similar": len(top),
            }
            for tier, chosen, direction in strata:
                # Re-sort top-ranked items in descending score order after selection.
                chosen_sorted = sorted(
                    chosen,
                    key=lambda item: (
                        -item[0] if direction == "descending" else item[0],
                        int(metadata.iloc[item[1]]["corpus_id"]),
                        int(metadata.iloc[item[2]]["corpus_id"]),
                    ),
                )
                for rank, (score, left, right) in enumerate(chosen_sorted, start=1):
                    selected.append(
                        {
                            "pair_key": tuple(
                                sorted(
                                    (
                                        int(metadata.iloc[left]["corpus_id"]),
                                        int(metadata.iloc[right]["corpus_id"]),
                                    )
                                )
                            ),
                            "model_key": model_key,
                            "relationship": relationship,
                            "language_pair": "|".join(lang_pair),
                            "tier": tier,
                            "cosine": score,
                            "rank_in_tier": rank,
                            "candidate_count": len(pool),
                            "rationale": (
                                f"{tier} candidate: {rank} of {len(pool)} in the "
                                f"{relationship}, {lang_pair[0]}-{lang_pair[1]} stratum, "
                                f"ranked by this model's cosine similarity "
                                f"({direction} order; pair limit={pair_limit})."
                            ),
                        }
                    )
    return selected, stratum_counts


def _source_fields(prefix: str, row: pd.Series) -> dict[str, Any]:
    return {
        f"{prefix}_corpus_id": int(row["corpus_id"]),
        f"{prefix}_language": row["language"],
        f"{prefix}_original_text": row["original_text"],
        f"{prefix}_source_dataset": row["source_dataset"],
        f"{prefix}_source_file": row["source_file"],
        f"{prefix}_source_record_id": row["source_record_id"],
        f"{prefix}_source_row_number": row["source_row_number"],
        f"{prefix}_source_field": row["source_field"],
        f"{prefix}_question_context": row.get("question_context", ""),
    }


def build_pair_catalog(
    metadata: pd.DataFrame,
    embeddings_by_model: dict[str, np.ndarray],
    labels_by_model: dict[str, pd.DataFrame],
    pair_limit: int = PAIR_LIMIT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    stratum_summary: dict[str, Any] = {}
    for model_key in MODEL_DIRS:
        chosen, counts = select_model_pairs(
            model_key,
            metadata,
            embeddings_by_model[model_key],
            labels_by_model[model_key],
            pair_limit,
        )
        selections.extend(chosen)
        stratum_summary[model_key] = counts

    all_pairs = sorted({item["pair_key"] for item in selections})
    ids_to_indices = {
        int(row["corpus_id"]): index for index, (_, row) in enumerate(metadata.iterrows())
    }
    selection_by_pair_model = {
        (item["pair_key"], item["model_key"]): item for item in selections
    }
    assignment_maps = {
        model_key: assignments.set_index(assignments["corpus_id"].astype(int))["cluster_id"].astype(int).to_dict()
        for model_key, assignments in labels_by_model.items()
    }

    rows: list[dict[str, Any]] = []
    for pair_number, pair_key in enumerate(all_pairs, start=1):
        left_id, right_id = pair_key
        left_index = ids_to_indices[left_id]
        right_index = ids_to_indices[right_id]
        left_row = metadata.iloc[left_index]
        right_row = metadata.iloc[right_index]
        row: dict[str, Any] = {"pair_id": f"P{pair_number:04d}"}
        row.update(_source_fields("a", left_row))
        row.update(_source_fields("b", right_row))
        for model_key in MODEL_DIRS:
            display = model_key
            cluster_map = assignment_maps[model_key]
            row[f"{display}_cluster_a"] = cluster_map[left_id]
            row[f"{display}_cluster_b"] = cluster_map[right_id]
            if cluster_map[left_id] < 0 or cluster_map[right_id] < 0:
                relation = "noise_involved"
            elif cluster_map[left_id] == cluster_map[right_id]:
                relation = "within_cluster"
            else:
                relation = "cross_cluster"
            row[f"{display}_cluster_relation"] = relation
            selection = selection_by_pair_model.get((pair_key, model_key))
            row[f"{display}_selected"] = selection is not None
            for field in ("relationship", "language_pair", "tier", "cosine", "rank_in_tier", "candidate_count", "rationale"):
                row[f"{display}_selection_{field}"] = (
                    selection[field] if selection is not None else ""
                )
        rows.append(row)
    catalog = pd.DataFrame(rows)
    summary = {
        "pair_limit_per_model_stratum_tier": pair_limit,
        "model_strata": stratum_summary,
        "selected_pair_model_records": len(selections),
        "unique_review_pairs": len(catalog),
        "pairs_selected_by_both_models": sum(
            1
            for pair_key in all_pairs
            if all((pair_key, model_key) in selection_by_pair_model for model_key in MODEL_DIRS)
        ),
        "noise_records_excluded_from_pair_candidates": {
            model_key: int((assignments["cluster_id"].astype(int) == -1).sum())
            for model_key, assignments in labels_by_model.items()
        },
    }
    return catalog, summary


def make_reviewer_template(catalog: pd.DataFrame, seed: int = ORDER_SEED) -> pd.DataFrame:
    """Create a model-blinded template with empty human-judgment fields."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(catalog))
    reviewed = catalog.iloc[order].reset_index(drop=True)
    fields = [
        "pair_id",
        "a_corpus_id",
        "a_language",
        "a_original_text",
        "a_source_dataset",
        "a_source_file",
        "a_source_record_id",
        "a_source_row_number",
        "a_source_field",
        "a_question_context",
        "b_corpus_id",
        "b_language",
        "b_original_text",
        "b_source_dataset",
        "b_source_file",
        "b_source_record_id",
        "b_source_row_number",
        "b_source_field",
        "b_question_context",
    ]
    template = reviewed[fields].copy()
    template["reviewer_id"] = ""
    template["concern_relation"] = ""
    template["confidence"] = ""
    template["shared_concern_note"] = ""
    template["evidence_note"] = ""
    return template


def _write_guide(path: Path) -> None:
    path.write_text(
        """# Bilingual reviewer guide

## What to review

Review each pair's two original comments in the language supplied. Use the question context only to understand what prompted the response; do not treat the prompt itself as evidence that the comments share a concern. No machine translation is supplied. If you cannot confidently interpret a comment, choose `U` and explain briefly.

Rate whether the comments express the same or a closely related **policy concern**, not whether they have the same sentiment, use similar words, come from the same stakeholder, or appear in the same consultation. A concern is a substantive issue, problem, value, or requested change. Promotional links, general process remarks, and broad topic words may be related only at a high level.

## Rating scale

Enter exactly one value in `concern_relation`:

- `3` — Same specific concern or substantially the same requested change.
- `2` — Closely related concern: same substantive policy problem, though detail or proposed action differs.
- `1` — Broad topical overlap only; distinct concerns or requests.
- `0` — No meaningful shared policy concern.
- `U` — Unable to judge (language, ambiguity, insufficient context, or comment is not interpretable).

Enter `confidence` as `high`, `medium`, or `low`. In `shared_concern_note`, briefly state the common concern if rated 2 or 3; for 0 or 1, say why the overlap is absent or only broad. `evidence_note` may include short verbatim snippets, with a clear indication of which comment they come from. Do not add labels on behalf of the other reviewer.

## Independent review and blinding

Reviewer 1 and Reviewer 2 must fill their separate files independently and must not compare notes before both files are complete. The templates intentionally omit model names, cluster assignments, cosine scores, selection strata, and rationales. The `pair_id` links a judgment to the model-derived details in `pair_catalog.csv`; keep that catalog from reviewers until independent ratings are submitted where practical.

Do not edit pair text or source references. Save completed templates as separate copies and retain the untouched blank templates.
""",
        encoding="utf-8",
    )


def _write_report(path: Path, summary: dict[str, Any], hashes: dict[str, str]) -> None:
    lines = [
        "# PolicyLens human-review package: concern coherence",
        "",
        "## Purpose and files",
        "",
        "This package prepares independent bilingual human judgments for evaluating concern coherence in the existing two-model, 450-comment comparison. It does not create labels or modify model outputs.",
        "",
        "- `pair_catalog.csv`: unique pair texts, source provenance, both models' cluster assignments, selected strata, cosine scores, and selection rationale.",
        "- `reviewer_1_template.csv` and `reviewer_2_template.csv`: independently completed copies; contain no model assignment or sampling-stratum information and have blank judgment fields.",
        "- `review_guide.md`: rubric, workflow, and reviewer blinding guidance.",
        "- `selection_manifest.json`: input fingerprints and reproducible sampling summary.",
        "",
        "## Sampling procedure",
        "",
        f"Within each model, all eligible unordered comment pairs were enumerated. Noise-assigned records were excluded; language pairs were limited to English-English, French-French, and English-French. Non-noise pairs were split into within-cluster (same cluster ID) and cross-cluster (different cluster IDs). Within each of those six strata, the two highest cosine pairs were selected as `likely_similar`, and the two lowest as `likely_dissimilar`, with corpus IDs breaking score ties. The two model selections were then unioned and exact pairs deduplicated so reviewers rate a pair only once. Selection is deterministic and uses no random pair sampling.",
        "",
        f"Pair limit per stratum and tier: **{summary['pair_limit_per_model_stratum_tier']}**. Selected pair-model records: **{summary['selected_pair_model_records']}**. Unique pairs for review: **{summary['unique_review_pairs']}**. Pairs selected by both models: **{summary['pairs_selected_by_both_models']}**.",
        "",
        "Candidate and selected counts by model/stratum:",
        "",
        "| Model | Relationship | Language pair | Candidates | Dissimilar selected | Similar selected |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model_key, strata in summary["model_strata"].items():
        for stratum, counts in strata.items():
            relationship, language_pair = stratum.split("|", 1)
            lines.append(
                f"| {model_key} | {relationship} | {language_pair.replace('|', '-')} | "
                f"{counts['candidate_count']} | {counts['selected_likely_dissimilar']} | "
                f"{counts['selected_likely_similar']} |"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Template order uses NumPy `default_rng({ORDER_SEED})`; both reviewer files share the same order.",
            "- Similarity is cosine similarity calculated from each saved model's own embedding matrix after row-wise L2 normalization.",
            "- Scores are kept only in the catalog for auditing; the reviewer templates do not expose scores or model predictions.",
            "- Pair IDs are assigned after sorting unique `(min corpus_id, max corpus_id)` keys.",
            f"- Input SHA-256 fingerprints: `{json.dumps(hashes, sort_keys=True)}`.",
            "- Exact artifact versions, cluster assignments, pair scores, and pair rationale are recorded in `selection_manifest.json` and `pair_catalog.csv`.",
            "",
            "## Reviewer workflow",
            "",
            "Give each bilingual reviewer one separate template and the guide, but withhold the catalog until both reviewers submit. Reviewers score independently and return completed copies. Merge only by `pair_id` after both are frozen. Do not convert missing ratings to `0`; follow up or mark unresolved. Keep model-derived fields separate from judgments.",
            "",
            "## Agreement calculation after annotation",
            "",
            "For pairs where both reviewers give an ordinal score (`0`–`3`), report exact agreement and Cohen's **linearly weighted kappa** on the ordinal scale. Publish the 4×4 confusion matrix, number of jointly rated pairs, and the number/proportion excluded because either reviewer marked `U` or left a rating blank. Separately report `U` counts by reviewer and their agreement on `U` versus rateable. Do not include `U` as an ordered numeric score. If the jointly rateable sample or rating variation is too small for a stable kappa, report the matrix and agreement counts without a strong reliability claim. Break results down by model-selection strata only after unblinding, and note that overlapping pairs are rated once.",
            "",
            "## Limitations",
            "",
            "- This is a deliberately stratified, model-driven diagnostic set, not a random or population-representative sample; it cannot estimate consultation-wide concern frequency or the prevalence of coherent clusters.",
            "- Selecting score extremes tests illustrative likely-similar and likely-dissimilar examples, but does not measure general pairwise accuracy or calibration.",
            "- There are no usable duplicate-group IDs in the 450-record sample; repeated-text/campaign effects cannot be controlled.",
            "- The sample is English-dominant, so French-French comparisons are fewer and language-specific findings are uncertain.",
            "- Similarity scores, cluster membership, and terms are model outputs, not human labels or evidence of sentiment, consensus, or policy priority.",
            "- Manual judgments remain pending; this package contains blank templates only.",
            "",
            "## Safety",
            "",
            "The generator refuses to overwrite an existing review package. Keep original model artifacts unchanged and store completed reviewer copies separately from the blank templates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_review_package(output_dir: Path = REVIEW_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing review package: {output_dir}")
    metadata, embeddings, labels = load_inputs()
    catalog, summary = build_pair_catalog(metadata, embeddings, labels)
    if catalog.empty:
        raise ValueError("No eligible review pairs were found in the saved comparison.")

    input_paths = {
        "sample_metadata": PROJECT_ROOT / "data" / "embeddings" / "corpus_metadata.csv",
        "baseline_embeddings": COMPARISON_DIR / MODEL_DIRS["all_minilm_l6_v2"] / "embeddings.npy",
        "multilingual_embeddings": COMPARISON_DIR / MODEL_DIRS["paraphrase_multilingual_minilm_l12_v2"] / "embeddings.npy",
        "baseline_cluster_labels": COMPARISON_DIR / MODEL_DIRS["all_minilm_l6_v2"] / "cluster_labels.csv",
        "multilingual_cluster_labels": COMPARISON_DIR / MODEL_DIRS["paraphrase_multilingual_minilm_l12_v2"] / "cluster_labels.csv",
        "comparison_report": COMPARISON_DIR / "comparison_report.md",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required review inputs are missing: {missing}")
    hashes = {key: sha256_file(path) for key, path in input_paths.items()}

    output_dir.mkdir(parents=True)
    catalog.to_csv(output_dir / "pair_catalog.csv", index=False, encoding="utf-8")
    reviewer_template = make_reviewer_template(catalog)
    reviewer_template.to_csv(
        output_dir / "reviewer_1_template.csv", index=False, encoding="utf-8"
    )
    reviewer_template.to_csv(
        output_dir / "reviewer_2_template.csv", index=False, encoding="utf-8"
    )
    manifest = {
        "source_comparison_report": "comparison_report.md",
        "sample_size": len(metadata),
        "pair_limit_per_model_stratum_tier": PAIR_LIMIT,
        "template_order_seed": ORDER_SEED,
        "model_directories": MODEL_DIRS,
        "input_sha256": hashes,
        "numpy_version": importlib.metadata.version("numpy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "duplicate_group_ids": {
            "nonblank_rows": int(metadata["duplicate_text_group_id"].str.strip().ne("").sum()),
            "duplicate_aware_selection": False,
        },
        "summary": summary,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_guide(output_dir / "review_guide.md")
    _write_report(output_dir / "review_package_report.md", summary, hashes)
    return manifest


if __name__ == "__main__":
    result = prepare_review_package()
    print(
        f"Prepared {result['summary']['unique_review_pairs']} unique review pairs "
        f"in {REVIEW_DIR}"
    )
