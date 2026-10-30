# PolicyLens results-access layer

## Next.js integration

The Next.js application in `frontend/frontend` now reads this layer through a local, read-only JSON API. Start
the API from the repository root with `python -m app.api`, then start the UI with `cd frontend/frontend; npm run dev`.
The API binds to `127.0.0.1:8000` by default. Set `NEXT_PUBLIC_POLICYLENS_API_URL` to change the URL and
`POLICYLENS_API_HOST` / `POLICYLENS_API_PORT` to change its bind address. It has no data mutation routes.

Routes: `GET /api/health`, `/api/stats`, `/api/dataset?q=&source=&limit=&offset=`, `/api/groups`,
`/api/groups/{cluster_id}`, `/api/evidence/{corpus_id}`, `/api/search?q=&source=&top=`, `/api/experiments`,
and `/api/analysis/status`.
`POST /api/inference/consultation` accepts 1–16 bounded comment texts and a `linsvm` or `logreg` selector. It
loads the verified TF-IDF pipelines lazily, returns provisional stance and sentiment suggestions against the project
reference labels, and does not persist input text. See `models/INFERENCE_API.md` for exact artifacts, source provenance,
metrics, and known failure modes. The API binds to loopback by default; inference also checks browser `Origin`.
Corpus, concern and evidence routes require the existing manifest/schema and traceability checks to pass.
Only public source records in the saved research corpus are served. API filters are bounded and read-only.
The BM25 search route lazy-loads the existing CIVIS and Lok Sabha indexes on first use, ranks each source
independently, and never compares their scores. Search provenance and the existing unverified-licence caveat
are returned with results. This search corpus is separate from the Canadian consultation-comment corpus.

The browser exposes the real Canadian corpus and saved exploratory clusters, not the previous unrelated static
policy feed. Dataset evidence preserves corpus and source record identifiers. Current concern descriptions are
extractive summaries. `collective_demand_candidates.csv` is an audit candidate ledger with screening decisions;
it does not contain generated representative demands. Demand generation and human review remain pending.

```powershell
python -m app.api
# in another terminal
cd frontend/frontend
npm run dev
```

`app/results_access.py` gives the future dashboard **read-only, manifest-verified** access to the existing research
results. It never writes, never recomputes and loads no model. The source corpus is Canadian federal Open Government
consultations, 2016–18, in English and French. Nothing served here is evidence about Indian citizens or Indian
government consultations.

## Quick start

```powershell
python -c "from app import ResultsStore; s = ResultsStore(); print(s.verification_report()['all_verified'])"
python -m pytest tests\test_results_access.py -q -p no:cacheprovider
```

## Dashboard

`app/dashboard.py` is a Streamlit browser over the same verified results. Streamlit is already installed (1.57) and
needs no frontend build. The dashboard reads data only through `ResultsStore`, shows precomputed results only, does not
analyse new comments and offers no stance classifier.

Launch from the project root in PowerShell, then open http://127.0.0.1:8501 in a desktop browser:

```powershell
python -m streamlit run app\dashboard.py --server.address 127.0.0.1
```

`--server.address 127.0.0.1` keeps the dashboard on this computer. Without it, Streamlit listens on every network
interface. Stop the server with Ctrl+C. To view another project tree, set `$env:POLICYLENS_ROOT = "<path>"` before
launching; the tests use this.

| Page | Shows | ResultsStore methods |
|---|---|---|
| Overview | verification status of each component, warnings, headline counts | `verification_report`, `concern_groups`, `prioritized_concerns` |
| Concern groups | tier filter, descriptions, flags, verbatim evidence with corpus and source record ids | `concern_groups`, `concern_group` |
| Exploratory prioritization | exploratory ranking, stacked per-signal contributions, raw-frequency baseline, per-group explanation and evidence, unranked groups | `prioritized_concerns`, `frequency_baseline`, `ranking_evidence` |
| Keywords by language | English/French switch, eligible subgroups' keywords with support flags, eligibility and exclusion reasons | `subgroup_eligibility`, `language_keywords` |
| Evaluation & limitations | manifest provenance, on-demand read-only research evaluation, every recorded limitation | `verification_report`, `evaluation_status`, `limitations` |

Every page carries the not-human-validated banner and the Canadian-corpus scope note. A component that is `MISSING`,
`MISMATCH`, `MALFORMED` or `INCONSISTENT` is flagged on the Overview page. Its own panels show an error instead of
data, and the other pages keep working.

```powershell
python -m pytest tests\test_dashboard.py tests\test_results_access.py -q -p no:cacheprovider
```

## Components and verification

| Component | Authoritative files | Verified by |
|---|---|---|
| `corpus` | `data/full_corpus/run_v1/corpus/corpus_entries.csv` | SHA-256 recorded in `corpus_manifest.json` |
| `concern_discovery` | `concern_discovery_v1/selected/concern_groups.csv`, `configurations/<selected>/concern_summaries.json` | `run_research_evaluation.check_concern_discovery` |
| `prioritization` | `concern_prioritization_v1/ranking.csv`, `ranking_evidence.csv`, `not_ranked.csv`, `config.json` | `run_research_evaluation.check_prioritization` |
| `language_keywords` | `language_keywords_v1/subgroup_inventory.csv`, `language_keywords.csv`, `config.json` | `run_research_evaluation.check_language_keywords` |
| `traceability` | `concern_prioritization_v1/evidence_traceability_manifest.json` and its ledger | `run_research_evaluation.check_evidence_traceability` |

After the manifest checks pass, each component's files are parsed and their required columns or keys are checked.
For `concern_discovery` and `prioritization`, every evidence row served must also match the verified original corpus:
the same `corpus_id`, source dataset and source record id, a verbatim quote, identical full text, and a matching text hash.

| Status | Meaning | Accessor behaviour |
|---|---|---|
| `VERIFIED` | all checks passed | data returned |
| `MISSING` | manifest or required file absent | raises `ArtifactUnavailable` |
| `MISMATCH` | a recorded checksum or existing evaluation check failed | raises |
| `MALFORMED` | unreadable file, or missing columns/keys | raises |
| `INCONSISTENT` | evidence does not match the original corpus | raises |

Results are cached per `ResultsStore`. Before each access the store compares the size and modification time of every
file it serves; if any changed, it re-verifies before serving. A failure in one component never blocks the others.

## Public interface (`ResultsStore`)

| Method | Returns |
|---|---|
| `verification_report()` | status of every component; never raises. The dashboard calls this first and greys out unavailable panels. |
| `verify(component, force=False)` | status, problems, checks and manifest provenance of one component |
| `concern_groups(tier=None)` | groups (largest first): `cluster_id`, `tier`, `short_label`, `description`, `flags`, entries / unique / independent texts, language counts, seed stability, cohesion, `evidence_corpus_ids` |
| `concern_group(cluster_id)` | one group plus `description_body`, `term_support`, and its evidence (`corpus_id`, source dataset and record id, question, language, verbatim `quote` and `full_text`) |
| `prioritized_concerns(limit=None)` | exploratory ranking: `exploratory_score`, `contributions` and `scaled_signals` per signal, all ranks, `explanation`; plus `weights` and the `not_ranked` groups with reasons |
| `frequency_baseline(limit=None)` | the same groups ordered by raw entries |
| `ranking_evidence(cluster_id)` | verbatim evidence rows of a ranked group with full source ids |
| `subgroup_eligibility(cluster_id=None)` | (cluster, language) subgroups with counts, `eligible`, `exclusion_reason`; payload carries `min_independent_texts` |
| `language_keywords(cluster_id=None, language=None)` | terms of eligible subgroups only, with weights, support and the `single_text_term` / `spans_text_boundary` flags |
| `evaluation_status()` | live, read-only research-evaluation result: counts and the BLOCKED / NOT RUN / FAILED items. Takes several seconds because it re-hashes protected files. |
| `limitations()` | interpretation rules, the scope note, the stance limitation and each component's own disclaimer |

Every data payload has this shape, so panels can render caveats and provenance uniformly:

```json
{"component": "prioritization", "kind": "exploratory", "human_validated": false,
 "scope": "Source corpus: Canadian ... not evidence about Indian citizens ...",
 "caveats": ["Exploratory, unsupervised results; not human-validated ..."],
 "provenance": {"manifest": "...run_manifest.json", "manifest_sha256": "...", "verified_utc": "..."},
 "data": [ ... ]}
```

## How the dashboard should use it

1. Create one `ResultsStore` per session. Call `verification_report()` and show any component that is not `VERIFIED`
   as unavailable, together with its status and problems.
2. Show `scope` and `caveats` next to every panel. Label scores "exploratory score", never "importance".
3. Link every displayed quote to its `corpus_id`, source dataset and source record id from the payload.
4. Show the frequency baseline next to the multi-signal ranking, with the `contributions` breakdown, so differences are
   explained rather than asserted.
5. Catch `ArtifactUnavailable` for each panel separately.

## Not served

* Stance predictions. The saved baseline is for exploratory research only and is deliberately excluded from these
  aggregate routes; its reported metrics measure agreement with the project reference labels on a small split.
* Sentiment-baseline models, which are auxiliary and not stance.
* Anything requiring a model at request time, such as embedding or clustering new comments. No fitted UMAP or HDBSCAN
  model is saved, and loading an encoder exceeds the memory usually free on this machine.
