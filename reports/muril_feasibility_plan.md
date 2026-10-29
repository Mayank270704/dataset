# MuRIL feasibility plan

**Decision (2026-10-13): do not download, fine-tune or run MuRIL now.** No stance-labelled data exists for it to learn
from or be evaluated on, the model weights are not cached, and this machine cannot hold a fine-tuning run in memory.
This plan records the evidence and the gates that must be met before any MuRIL work starts. It proposes no new
dataset and no new experiment.

## 1. What the roadmap asks of MuRIL

| Source | Statement |
|---|---|
| `models/MODEL_REGISTRY.md` | "MuRIL Representation Model": Planned; input "Multilingual Hindi, Hinglish, and English auxiliary corpora"; Not Evaluated; next action "Implement tokenization and mean-pooling module for cross-lingual representation alignment." |
| `models/multilingual/__init__.py` | `google/muril-base-cased`, "Planned for domain transfer experiments from auxiliary datasets." |
| `reports/current_project_status.md`, §5B | Blocker: no multilingual consultation stance benchmark; Hindi and Hinglish text exists only in auxiliary YouTube (HinEmo-9) and Twitter (SentiMix) corpora. |

## 2. Data availability

| Dataset (processed file) | Rows | Labels | Usable for stance? |
|---|---:|---|---|
| Canadian consultation corpus (`data/full_corpus/run_v1`) | 10,108 eligible entries | none yet; stance annotation pending | target domain, but English/French, not Hindi |
| SentiMix train / test (`sentimix_*_canonical.csv`) | 15,131 / 3,000 | sentiment: neutral, positive, negative | no: sentiment, Hinglish tweets |
| HinEmo-9 full / gold (`hinemo9_*_canonical.csv`) | 30,436 / 3,363 | 9 emotions | no: emotion, YouTube comments |
| `PolicyLens_Hindi_English_Hinglish_Sentiment.csv` | 2,500 | sentiment | no: rated `currently_unsuitable` ("Not a model training or retrieval corpus") in `reports/full_workspace_cleaning_audit_report.md` |

No file in the repository has stance labels in Hindi or Hinglish. The auxiliary corpora measure sentiment and emotion,
which the annotation guidelines (§2) separate from stance, and they come from social media, not consultations. A MuRIL
model trained on them would be a sentiment or emotion model. It could not be reported as a stance model.

MuRIL's published pre-training covers English and Indian languages, not French. For the current Canadian English/French
corpus, the multilingual MiniLM encoder already in use is the better-matched encoder. MuRIL becomes relevant only for
Hindi or Hinglish consultation text that the project does not hold.

## 3. Model and compute requirements

Computed from the cached `config.json` (BERT-base, 12 layers, hidden size 768, vocabulary 197,285):

| Quantity | Value |
|---|---:|
| Parameters | 237,556,224 (151,911,168 of them in the embedding layer) |
| fp32 weights | about 0.95 GB |
| Full fine-tuning state with AdamW (weights, gradients, two moments), before activations | about 3.8 GB |

Measured on this machine:

| Resource | Value | Consequence |
|---|---|---|
| Cached MuRIL files | `config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` | **no weights**; a download of about 1 GB is needed first |
| RAM | 7.9 GB total; 0.36-0.9 GB free across checks on 2026-10-13 | full fine-tuning does not fit; even loading the fp32 weights exceeds what is usually free |
| GPU | none (`torch 2.8.0+cpu`) | fine-tuning would be CPU-only |
| CPU / disk | 12 logical CPUs, 11 GB free disk | disk is sufficient for the weights |

The run_v1 report records that the default Hugging Face loader crashed with an access violation at about 500 MB free RAM,
and that a low-memory loader was needed for the much smaller MiniLM encoders.

## 4. Gates before any MuRIL run

All four must hold, and each is checked and recorded before the next step:

1. **Target data.** Genuine, double-annotated human stance labels exist for the text MuRIL would classify, following
   `reports/annotation_guidelines_v1.1.md`. For the current project that means completed annotation files read by
   `scripts/train_stance_baselines.py`. MuRIL is only appropriate if those texts include Hindi or Hinglish.
2. **Baseline first.** `python scripts\train_stance_baselines.py` has produced a `trained_and_evaluated` run on the same
   labels. A MuRIL model is justified only if it is compared on the identical grouped split and leakage groups.
3. **Weights.** `google/muril-base-cased` is downloaded at a pinned revision, and its file hashes and revision are recorded
   in `models/MODEL_REGISTRY.md`.
4. **Compute.** At least about 2 GB of free RAM for frozen-feature extraction on CPU. Full fine-tuning needs about 4 GB
   plus activations, which in practice means a GPU. Measure throughput on 100 texts before committing to a full run.

Read-only check of the resource gates:

```powershell
python -c "import glob,os,psutil; f=[os.path.basename(p) for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--google--muril-base-cased/snapshots/*/*'))]; print('cached files:', f); print('weights present:', any(x.endswith(('.bin','.safetensors')) for x in f)); print('free RAM MB:', psutil.virtual_memory().available // 2**20)"
```

## 5. Options once the gates are met, cheapest first

| Option | What it is | Memory | Fits the roadmap |
|---|---|---|---|
| A. Frozen embeddings + linear probe | Mean-pooled MuRIL sentence vectors, then logistic regression on the same split as the TF-IDF baselines | about 1-1.5 GB, CPU | yes: the registry's "tokenization and mean-pooling module" |
| B. Partial fine-tuning | Train only the top encoder layers and the classifier head | several GB; GPU preferred | after A shows a gain |
| C. Full fine-tuning | All 237.6 M parameters | about 3.8 GB optimizer state plus activations; GPU | only with enough labels to justify it |

Option A matches the existing next action, reuses the leakage-safe protocol and needs no new data. If option A is used on
the auxiliary SentiMix data as a representation check, it must be reported as sentiment. The existing SentiMix baseline
(hashing vectorizer + SGD, macro-F1 0.6614 on the 3,000-row official test split, `reports/sentimix_baseline_report_v1.md`)
is the comparison point. It would not change the stance blocker.

## 6. Not planned

* Fine-tuning MuRIL on sentiment or emotion data and calling it stance.
* Machine-translating English or French consultation comments into Hindi to create training data.
* Using AI-assisted or AI-generated labels as training or evaluation labels.
* Paid APIs or hosted training services.
