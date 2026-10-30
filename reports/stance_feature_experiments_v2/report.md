# Preliminary stance model comparison — development only

This preliminary experiment used repository stance reference labels. Source history is recorded in `data/ai_labels/stance_sentiment_ai_v1/provenance.json`; reference-label terminology does not imply independent human verification.

The 239 / 80 / 80 seed-42 grouped split was previously inspected and is development data. It must not be described as an untouched test set. The later comparison in `reports/stance_reference_model_comparison_v2/` creates a new grouped fold for the current fit, while noting that this reference pool was already exposed in earlier development.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority | 0.5500 | 0.2366 | 0.3903 |
| Linear SVM | 0.5750 | 0.3958 | 0.5447 |
| Logistic Regression | 0.5375 | 0.3756 | 0.5107 |
| Multilingual MiniLM + balanced Logistic Regression | 0.5125 | 0.3948 | 0.5288 |

Use the new grouped comparison report for current development metrics and reproduction instructions.
