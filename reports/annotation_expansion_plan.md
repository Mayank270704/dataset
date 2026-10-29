# PolicyLens — Consultation Stance Annotation Expansion Plan (400 Records)

## 1. Overview and Status
- **Current Verified Human Labels**: **0** (Pilot dataset `data/annotation/pilot_annotation_50.csv` and annotator templates `annotator_1.csv`, `annotator_2.csv` are strictly blank templates awaiting human adjudication).
- **Target Size**: **400 records** (expanded from 50 to meet the 300–500 sample threshold required for statistically reliable supervised stance modeling).
- **Expansion Candidate Pool**: [`data/annotation/expansion_pool_400.csv`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/data/annotation/expansion_pool_400.csv)
- **Independent Blind Templates**:
  - [`data/annotation/expansion_annotator_1.csv`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/data/annotation/expansion_annotator_1.csv)
  - [`data/annotation/expansion_annotator_2.csv`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/data/annotation/expansion_annotator_2.csv)

---

## 2. Stratification and Dataset Allocation

The 400-record expansion pool preserves all 50 pilot items and draws 350 additional substantive comments without replacement:

| Source Dataset | Pilot Allocation | Additional Allocation | Total Expansion Pool | Stratification Dimension |
|---|:---:|:---:|:---:|---|
| `individual_comments_2016_18` | 20 | 70 | **90** | 5 core open-ended fields across in-person workshops |
| `consultation_compilation_2017_18` | 20 | 160 | **180** | Themes (`Open Data`, `Open Dialogue`, `Open Science`, `Digital Government`) |
| `get_involved_questionnaire_2017_18` | 10 | 120 | **130** | Substantive catch-alls: Q8 (Priorities), Q14 (Rationale), Q18 (Knowledge) |
| **Total** | **50** | **350** | **400** | Bilingual (English & French) Canadian Federal Consultations |

---

## 3. Strict Quality & Anti-Leakage Constraints

1. **Exact Deduplication**: At most 1 record per `duplicate_text_group_id` is included.
2. **Participant Deduplication**: No underlying `source_row_number` is sampled more than once within the same dataset.
3. **Substantive Length**: All comments have verbatim text length $\ge 20$ characters.
4. **Filler Exclusion**: Generic boilerplate (`none`, `n/a`, `.`, `thanks`, `no comment`) is excluded.
5. **Human Independence**: Annotator 1 and Annotator 2 work on completely separate files without seeing each other's entries.
6. **No Machine Generation**: Zero synthetic, heuristic, or LLM-generated labels. All human columns (`stance`, `concern`, `requested_action`, `evidence_span`, `review_notes`, `annotator_id`) remain strictly empty strings until real human review is performed.

---

## 4. Annotation Schema & Valid Values

For each record, human annotators must complete:
1. `stance`: One of `Support`, `Oppose`, `Mixed`, `Neutral`, `Unclear`.
2. `concern`: Noun phrase ($\le 15$ words), or `none` / `unclear`.
3. `requested_action`: Verb phrase ($\le 20$ words), or `none` / `unclear`.
4. `evidence_span`: Verbatim substring from `original_text` ($\le 200$ chars).
5. `review_notes`: Optional human notes (e.g. bilingual French nuance, off-topic).
6. `annotator_id`: Human identifier (e.g. `annotator_A`, `annotator_B`).

---

## 5. Next Steps for Supervised Modeling
Once human annotators submit completed labels for the 400 records:
1. Run [`notebooks/03_annotation_agreement_analysis.ipynb`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/notebooks/03_annotation_agreement_analysis.ipynb) to calculate Cohen's Kappa, raw percent agreement, and confusion matrices between Annotator 1 and Annotator 2.
2. Adjudicate disagreements into a unified consensus set.
3. Fit the verified baseline stance models implemented in [`models/stance_classification/baselines.py`](file:///C:/Users/MAYANK%20SWAROOP/OneDrive/Desktop/dataset/models/stance_classification/baselines.py) (`TfidfStanceClassifier`).
