"""Small read-only JSON API for the verified PolicyLens research artifacts.

Run with ``python -m app.api``. No models are loaded and no write routes exist.

Routes
------
GET  /api/health
GET  /api/stats
GET  /api/groups[?tier=]
GET  /api/groups/{cluster_id}
GET  /api/dataset[?q=&source=&limit=&offset=]
GET  /api/search?q=&source=&top=
GET  /api/experiments
GET  /api/analysis/status
GET  /api/evidence/{corpus_id}
GET  /api/demands[?limit=&offset=]          -- extractive collective-demand candidates
GET  /api/demands/{candidate_id}            -- one candidate with full evidence
POST /api/inference/consultation
POST /api/inference/stance                  -- direct saved three-class stance model
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.results_access import ArtifactUnavailable, ResultsStore

ROOT = Path(__file__).resolve().parent.parent
STORE = ResultsStore()
_SEARCH_INDEXES = {}
_PREDICTORS = {}


def payload(path: str, query: dict[str, list[str]]) -> tuple[int, dict]:
    if path == "/api/health":
        report = STORE.verification_report()
        return 200, {"status": "ok", "application": "PolicyLens", "research": report}
    if path == "/api/stats":
        STORE._require("corpus")
        groups = STORE.concern_groups()["data"]
        ranking = STORE.prioritized_concerns(limit=1)["data"]
        corpus = STORE._load("corpus")["by_id"]
        from app.demand_generation import get_candidates
        try:
            demand_count = len([c for c in get_candidates() if not c.insufficient_evidence])
        except Exception:
            demand_count = 0
        return 200, {"corpus_entries": len(corpus), "concern_groups": len(groups),
                     "ranked_groups": len(STORE.prioritized_concerns()["data"]),
                     "demand_candidates": demand_count,
                     "top_exploratory_group": ranking[0] if ranking else None,
                     "scope": STORE.verification_report()["scope"], "human_validated": False}
    if path == "/api/groups":
        return 200, STORE.concern_groups(query.get("tier", [None])[0])
    if path.startswith("/api/groups/"):
        try:
            return 200, STORE.concern_group(int(path.rsplit("/", 1)[1]))
        except ValueError:
            return 400, {"error": "cluster_id must be an integer"}
        except KeyError:
            return 404, {"error": "concern group not found"}
    if path == "/api/dataset":
        STORE._require("corpus")
        STORE._require("concern_discovery")
        term = query.get("q", [""])[0].strip().casefold()
        source = query.get("source", [""])[0].strip()
        try:
            limit = min(max(int(query.get("limit", ["50"])[0]), 1), 100)
            offset = max(int(query.get("offset", ["0"])[0]), 0)
        except ValueError:
            return 400, {"error": "limit and offset must be integers"}
        if len(term) > 200 or len(source) > 100:
            return 400, {"error": "query filters are too long"}
        corpus = STORE._load("corpus")["by_id"]
        assignments = {}
        p = STORE.paths.concern_discovery / "selected" / "entry_assignments.csv"
        import csv
        with p.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                assignments[row["corpus_id"]] = {"cluster_id": int(row["cluster_id"]), "state": row["state"],
                                                  "language": row["language"], "question_slug": row["question_slug"]}
        rows = []
        for cid, row in corpus.items():
            if source and row["source_dataset"] != source:
                continue
            if term and term not in row["original_text"].casefold():
                continue
            rows.append({"corpus_id": cid, "source_dataset": row["source_dataset"],
                         "source_record_id": row["source_record_id"], "text": row["original_text"],
                         **assignments.get(str(cid), {})})
        rows.sort(key=lambda x: x["corpus_id"])
        return 200, {"component": "corpus", "scope": STORE.verification_report()["scope"],
                     "total": len(rows), "offset": offset, "limit": limit, "data": rows[offset:offset + limit]}
    if path == "/api/search":
        term = query.get("q", [""])[0].strip()
        source = query.get("source", ["both"])[0]
        if not term or len(term) > 200:
            return 400, {"error": "q is required and must be at most 200 characters"}
        if source not in {"civis", "loksabha", "both"}:
            return 400, {"error": "source must be civis, loksabha, or both"}
        try:
            top = min(max(int(query.get("top", ["5"])[0]), 1), 5)
        except ValueError:
            return 400, {"error": "top must be an integer"}
        from indian.retrieval import policy_search as ps
        indexes = {"civis": ps.CivisIndex, "loksabha": ps.LokSabhaIndex}
        sources = ("civis", "loksabha") if source == "both" else (source,)
        results = {}
        for name in sources:
            if name not in _SEARCH_INDEXES:
                _SEARCH_INDEXES[name] = indexes[name]()
            results[name] = [r.__dict__ for r in _SEARCH_INDEXES[name].search(term, top)]
        return 200, {"query": term, "kind": "bm25_candidate_retrieval", "results_by_source": results,
                     "disclaimer": ps.DISCLAIMER, "scores_comparable_across_sources": False,
                     "scope_note": "These Indian policy and parliamentary sources are separate from the Canadian consultation comment corpus."}
    if path == "/api/experiments":
        p = ROOT / "data/full_corpus/experiments_v2"
        manifest = json.loads((p / "run_manifest.json").read_text(encoding="utf-8")).get("experiments", {})
        result = []
        for file in sorted(p.glob("*/metrics.json")):
            try:
                name = file.parent.name
                metrics = json.loads(file.read_text(encoding="utf-8"))
                recorded = manifest.get(name, {})
                matches = (recorded.get("clusters") == metrics.get("n_clusters") and
                           recorded.get("coverage_core") == metrics.get("coverage_core"))
                result.append({"experiment": name, "verification": "manifest fields match" if matches else "manifest mismatch or incomplete",
                               "metrics": metrics if matches else None})
            except (OSError, json.JSONDecodeError):
                result.append({"experiment": file.parent.name, "unavailable": "metrics file is malformed"})
        return 200, {"kind": "saved_experiment_metrics", "human_validated": False, "data": result}
    if path == "/api/analysis/status":
        return 200, STORE.evaluation_status()
    if path.startswith("/api/evidence/"):
        STORE._require("corpus")
        try:
            cid = int(path.rsplit("/", 1)[1])
        except ValueError:
            return 400, {"error": "corpus_id must be an integer"}
        row = STORE._load("corpus")["by_id"].get(cid)
        if row is None:
            return 404, {"error": "evidence not found"}
        return 200, {"component": "corpus", "data": {"corpus_id": cid, "source_dataset": row["source_dataset"],
                     "source_record_id": row["source_record_id"], "text": row["original_text"]},
                     "scope": STORE.verification_report()["scope"]}
    # ── Collective-demand candidate routes ────────────────────────────────
    if path == "/api/demands":
        from app.demand_generation import get_candidates, DISCLAIMER
        try:
            limit = min(max(int(query.get("limit", ["50"])[0]), 1), 200)
            offset = max(int(query.get("offset", ["0"])[0]), 0)
        except ValueError:
            return 400, {"error": "limit and offset must be integers"}
        try:
            all_cands = get_candidates()
        except Exception as exc:
            return 503, {"error": "demand generation unavailable", "detail": str(exc)}
        page = all_cands[offset: offset + limit]
        data = []
        for c in page:
            d = {
                "candidate_id": c.candidate_id,
                "cluster_id": c.cluster_id,
                "short_label": c.short_label,
                "demand_text": c.demand_text,
                "entries": c.entries,
                "unique_texts": c.unique_texts,
                "independent_texts": c.independent_texts,
                "language_counts": c.language_counts,
                "source_datasets": c.source_datasets,
                "state": c.state,
                "human_review_status": c.human_review_status,
                "generation_method": c.generation_method,
                "quality_flags": c.quality_flags,
                "has_dissenting_evidence": c.has_dissenting_evidence,
                "insufficient_evidence": c.insufficient_evidence,
                "insufficient_reason": c.insufficient_reason,
                "evidence_coverage": c.evidence_coverage,
            }
            data.append(d)
        return 200, {
            "component": "collective_demands",
            "kind": "extractive_demand_candidates",
            "human_validated": False,
            "generation_method": "deterministic_extractive_v1",
            "scope": "Source corpus: Canadian federal Open Government consultations, 2016-18 (English/French). Not evidence about Indian citizens or Indian consultations.",
            "caveats": [
                DISCLAIMER,
                "These are generated candidates, not validated collective demands.",
                "'state' is 'generated_extractive'; human review is pending for all.",
                "Groups with insufficient_evidence=true are included for completeness but should not be interpreted as demands.",
            ],
            "total": len(all_cands),
            "offset": offset,
            "limit": limit,
            "data": data,
        }
    if path.startswith("/api/demands/"):
        candidate_id = path.rsplit("/", 1)[1]
        if not candidate_id.startswith("cand-"):
            return 400, {"error": "candidate_id must have the form cand-<cluster_id>"}
        from app.demand_generation import get_candidate, DISCLAIMER
        try:
            cand = get_candidate(candidate_id)
        except Exception as exc:
            return 503, {"error": "demand generation unavailable", "detail": str(exc)}
        if cand is None:
            return 404, {"error": "demand candidate not found"}
        return 200, {
            "component": "collective_demands",
            "kind": "extractive_demand_candidate_detail",
            "human_validated": False,
            "generation_method": "deterministic_extractive_v1",
            "scope": cand.scope,
            "caveats": [
                DISCLAIMER,
                "This is a generated candidate, not a validated collective demand.",
                "'state' is 'generated_extractive'; human review is pending.",
                "Evidence quotes are verbatim from the verified corpus.",
            ],
            "data": cand.as_dict(),
        }
    return 404, {"error": "route not found"}


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, result: dict) -> None:
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", os.getenv("POLICYLENS_FRONTEND_ORIGIN", "http://localhost:3000"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            status, result = payload(parsed.path, parse_qs(parsed.query))
        except ArtifactUnavailable as exc:
            status, result = 503, {"error": str(exc), "component": exc.component,
                                   "status": exc.status, "problems": exc.problems}
        except Exception:
            status, result = 500, {"error": "research artifact request failed"}
        self._respond(status, result)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", os.getenv("POLICYLENS_FRONTEND_ORIGIN", "http://localhost:3000"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        req_path = urlparse(self.path).path
        if req_path not in {"/api/inference/consultation", "/api/inference/stance",
                             "/api/inference/stance/context-staging",
                             "/api/inference/stance/five-class-staging"}:
            self._respond(405, {"error": "read-only API; only consultation/stance inference accepts POST"})
            return
        allowed_origin = os.getenv("POLICYLENS_FRONTEND_ORIGIN", "http://localhost:3000")
        origin = self.headers.get("Origin")
        if origin and origin != allowed_origin:
            self._respond(403, {"error": "origin is not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 128_000:
                self._respond(413, {"error": "request body must be between 1 byte and 128 KB"})
                return
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("JSON body must be an object")
            if req_path == "/api/inference/stance/five-class-staging":
                from models.context_stance_features import ContextPredictionInput
                from models.five_class_stance_inference import FiveClassStancePredictor
                if set(request) != {"items"}:
                    raise ValueError("five-class staging accepts only the items field")
                entries = request.get("items")
                if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
                    raise ValueError("items must be an array containing 1 to 16 objects")
                allowed = {"question", "context", "comment"}
                items = []
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        raise ValueError(f"items[{index}] must be an object")
                    extra = set(entry) - allowed
                    if extra:
                        raise ValueError(f"items[{index}] contains unsupported fields")
                    if "comment" not in entry:
                        raise ValueError(f"items[{index}].comment is required")
                    if not isinstance(entry["comment"], str):
                        raise ValueError(f"items[{index}].comment must be a string")
                    item = ContextPredictionInput(entry.get("question"), entry.get("context"), entry["comment"])
                    item.validate()
                    items.append(item)
                cache_key = "five_class_stance_staging_v1"
                if cache_key not in _PREDICTORS:
                    _PREDICTORS[cache_key] = FiveClassStancePredictor()
                rows = _PREDICTORS[cache_key].predict(items).to_dict(orient="records")
                self._respond(200, {
                    "component": "five_class_stance_staging_inference",
                    "kind": "synthetic_candidate_provisional_suggestions",
                    "candidate": "stance_context_wordchar_5class_synth_v1",
                    "human_validated": False,
                    "labels_supported": ["Support", "Neutral", "Oppose", "Mixed", "Unclear"],
                    "caveats": [
                        "Separate staging route; production routes and artifact are unchanged.",
                        "Oppose has only one original reference record and Mixed has none; synthetic training examples do not establish real-world accuracy for these classes.",
                        "Empty or whitespace comments shorter than three characters return no stance prediction.",
                        "Missing question/context values use explicit missing-field markers.",
                        "Decision margins are diagnostics, not calibrated probabilities or confidence.",
                    ],
                    "data": rows,
                })
                return
            if req_path == "/api/inference/stance/context-staging":
                from models.context_stance_features import ContextPredictionInput
                from models.context_stance_inference import ContextArtifactError, ContextStancePredictor
                if set(request) != {"items"}:
                    raise ValueError("context staging accepts only the items field")
                raw_items = request.get("items")
                if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 16:
                    raise ValueError("items must be an array containing 1 to 16 objects")
                allowed = {"question", "context", "comment", "language"}
                items = []
                for index, entry in enumerate(raw_items):
                    if not isinstance(entry, dict):
                        raise ValueError(f"items[{index}] must be an object")
                    extra = set(entry) - allowed
                    if extra:
                        raise ValueError(f"items[{index}] contains unsupported fields")
                    if "comment" not in entry:
                        raise ValueError(f"items[{index}].comment is required")
                    if not isinstance(entry["comment"], str):
                        raise ValueError(f"items[{index}].comment must be a string")
                    item = ContextPredictionInput(
                        question=entry.get("question"), context=entry.get("context"),
                        comment=entry.get("comment"), language=entry.get("language"),
                    )
                    item.validate()
                    items.append(item)
                cache_key = "context_staging_v1"
                if cache_key not in _PREDICTORS:
                    _PREDICTORS[cache_key] = ContextStancePredictor()
                result_rows = _PREDICTORS[cache_key].predict(items).to_dict(orient="records")
                self._respond(200, {
                    "component": "context_staging_inference",
                    "kind": "staging_only_provisional_model_suggestions",
                    "candidate": "stance_context_wordchar_v1",
                    "human_validated": False,
                    "labels_supported": ["Support", "Neutral", "Unclear"],
                    "language_policy": "language metadata is accepted for compatibility but ignored by this model",
                    "caveats": [
                        "Staging prototype only; the existing production inference routes are unchanged.",
                        "Predictions are provisional agreement with repository reference labels, not human-validated accuracy.",
                        "Margins are diagnostics, not calibrated probabilities or confidence.",
                        "Empty or whitespace comments of fewer than three characters return no stance prediction.",
                        "Missing question/context values use explicit missing-field markers; context is optional.",
                    ],
                    "data": result_rows,
                })
                return
            model = request.get("model", "linsvm")
            texts = request.get("texts")
            if model not in {"linsvm", "logreg"}:
                raise ValueError("model must be linsvm or logreg")
            if not isinstance(texts, list) or not 1 <= len(texts) <= 16:
                raise ValueError("texts must be an array containing 1 to 16 strings")
            if any(not isinstance(t, str) or len(t) > 5000 for t in texts):
                raise ValueError("each text must be a string of at most 5000 characters")

            # Route to direct stance inference if requested or on /api/inference/stance
            use_v2 = (req_path == "/api/inference/stance" or
                      request.get("pipeline") in {"v2", "final"} or
                      request.get("version") in {"v2", "final"})

            if use_v2:
                from models.stance_inference_v2 import FinalStancePredictor, ArtifactError
                cache_key = f"final_stance_{model}"
                if cache_key not in _PREDICTORS:
                    _PREDICTORS[cache_key] = FinalStancePredictor(model=model)
                predictor = _PREDICTORS[cache_key]
                rows = predictor.predict(texts).to_dict(orient="records")
                self._respond(200, {
                    "component": "final_stance_inference", "kind": "provisional_model_suggestions",
                    "model": model, "human_validated": False,
                    "pipeline": "FinalStancePredictor (direct three-class stance)",
                    "label_provenance": "repository stance reference labels; no sentiment-derived stance classes",
                    "caveats": [
                        "Predictions are provisional suggestions, not ground truth or human-reviewed labels.",
                        "The available artifact recognizes Support, Neutral and Unclear only; Oppose and Mixed are not trained classes.",
                        "Decision margins are diagnostics, not probabilities or calibrated confidence.",
                        "The models were trained on a 400-record Canadian consultation pool and have limited language/domain coverage.",
                        "Input text is processed in memory for this response and is not written by this endpoint.",
                    ],
                    "data": rows,
                })
                return

            from models.new_comment_inference import ArtifactError, NewCommentPredictor
            if model not in _PREDICTORS:
                _PREDICTORS[model] = NewCommentPredictor(model=model)
            predictor = _PREDICTORS[model]
            rows = predictor.predict(texts).to_dict(orient="records")
            self._respond(200, {
                "component": "consultation_inference", "kind": "provisional_model_suggestions",
                "model": model, "human_validated": False,
                "label_provenance": "repository stance reference labels",
                "caveats": [
                    "Predictions are provisional suggestions; performance outside the repository reference set is not established.",
                    "Stance and sentiment are separate constructs; sentiment is tone, not stance.",
                    "Decision margins are not probabilities or calibrated confidence.",
                    "The models were trained on the repository consultation reference set; coverage outside it is not established.",
                    "Input text is processed in memory for this response and is not written by this endpoint.",
                ],
                "data": rows,
            })
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._respond(400, {"error": str(exc)})
        except ImportError:
            self._respond(503, {"error": "consultation inference dependencies are unavailable"})
        except Exception as exc:
            if exc.__class__.__name__ in {"ArtifactError", "ContextArtifactError", "FiveClassArtifactError"}:
                self._respond(503, {"error": "verified inference artifacts unavailable", "detail": str(exc)})
            else:
                self._respond(500, {"error": "consultation inference failed"})

    def log_message(self, fmt: str, *args: object) -> None:
        # Request query strings and POST bodies may contain submitted comment text.
        path = urlparse(self.path).path
        print(f"{self.address_string()} - {self.command} {path}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((os.getenv("POLICYLENS_API_HOST", "127.0.0.1"),
                                  int(os.getenv("POLICYLENS_API_PORT", "8000"))), Handler)
    print(f"PolicyLens read-only API listening on http://{server.server_address[0]}:{server.server_address[1]}")
    server.serve_forever()
