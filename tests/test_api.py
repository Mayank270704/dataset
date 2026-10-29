"""HTTP contract and saved-data smoke tests for the read-only API."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from app.api import Handler


def request(server, path):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    connection.request("GET", path)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def post_json(server, path, body, origin=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=60)
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    connection.request("POST", path, json.dumps(body), headers)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def test_actual_saved_research_http_workflow():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = request(server, "/api/health")
        assert status == 200 and health["status"] == "ok"

        status, rows = request(server, "/api/dataset?q=open%20government&limit=2")
        assert status == 200 and rows["total"] > 0 and len(rows["data"]) == 2
        row = rows["data"][0]
        assert {"corpus_id", "source_dataset", "source_record_id", "text"} <= row.keys()

        status, group = request(server, "/api/groups/30")
        assert status == 200 and group["data"]["cluster_id"] == 30
        assert group["data"]["evidence"][0]["corpus_id"]

        status, evidence = request(server, f"/api/evidence/{group['data']['evidence'][0]['corpus_id']}")
        assert status == 200 and evidence["data"]["source_dataset"]

        status, search = request(server, "/api/search?q=motor%20vehicles%20registration%20rules&source=civis&top=2")
        assert status == 200 and set(search["results_by_source"]) == {"civis"}
        assert search["scores_comparable_across_sources"] is False
        assert search["results_by_source"]["civis"]

        status, demands = request(server, "/api/demands?limit=5")
        assert status == 200
        assert demands["total"] > 0
        assert len(demands["data"]) == 5
        cand = demands["data"][0]
        assert cand["candidate_id"].startswith("cand-")
        assert cand["state"] == "generated_extractive"
        assert cand["demand_text"]

        cand_id = cand["candidate_id"]
        status, demand_detail = request(server, f"/api/demands/{cand_id}")
        assert status == 200
        assert demand_detail["data"]["candidate_id"] == cand_id
        assert demand_detail["data"]["evidence"]

        assert request(server, "/api/demands/cand-999999")[0] == 404
        assert request(server, "/api/demands/invalid_id")[0] == 400

        status, stats = request(server, "/api/stats")
        assert status == 200
        assert "demand_candidates" in stats
        assert stats["demand_candidates"] > 0

        assert request(server, "/api/groups/not-an-id")[0] == 400
        assert request(server, "/api/groups/999999")[0] == 404
        assert request(server, "/api/search?q=test&source=invalid")[0] == 400

        root = Path(__file__).resolve().parents[1]
        corpus = pd.read_csv(root / "data/full_corpus/run_v1/corpus/corpus_entries.csv",
                             dtype=str, keep_default_na=False)
        sample = corpus[corpus.corpus_id.isin(["479", "2826"])].sort_values("corpus_id")
        pool = pd.read_csv(root / "data/annotation/expansion_pool_400_v2.csv", dtype=str, keep_default_na=False)
        trained_records = set(pool.source_dataset + "#" + pool.source_record_id)
        assert not set(sample.source_dataset + "#" + sample.source_record_id) & trained_records
        for model in ("linsvm", "logreg"):
            status, inference = post_json(server, "/api/inference/consultation",
                                          {"model": model, "texts": sample.original_text.tolist()})
            assert status == 200
            assert inference["human_validated"] is False
            assert inference["kind"] == "provisional_model_suggestions"
            assert len(inference["data"]) == 2
            assert all(r["input_status"] == "ok" for r in inference["data"])
            assert all(r["stance_suggestion"] in {"Support", "Neutral", "Unclear"} for r in inference["data"])
            assert all(r["sentiment_suggestion"] in {"Positive", "Negative", "Neutral"} for r in inference["data"])
            assert "not probabilities" in " ".join(inference["caveats"])
        assert post_json(server, "/api/inference/consultation", {"model": "unknown", "texts": ["Hello"]})[0] == 400
        assert post_json(server, "/api/inference/consultation", {"texts": ["Hello"]}, "https://untrusted.example")[0] == 403
        assert post_json(server, "/api/inference/consultation", {"texts": ["  "]})[1]["data"][0]["input_status"] == "too_short_or_empty"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_final_stance_inference_route():
    """POST /api/inference/stance returns heuristic 4-label predictions (v2 pipeline).

    Verifies:
    - HTTP 200 with correct component / kind fields
    - human_validated is False
    - Every row exposes final_stance in {Support, Oppose, Neutral, Unclear}
    - Every row carries a disclaimer field (non-empty)
    - Oppose rows are marked prediction_method == 'heuristic_oppose'
    - Error: unknown model -> 400
    - Error: too many texts (>16) -> 400
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        texts = [
            "I fully support this open government initiative.",
            "This policy is absolutely terrible and must be stopped.",
            "The proposal seems reasonable in some respects.",
        ]
        for model in ("linsvm", "logreg"):
            status, resp = post_json(server, "/api/inference/stance", {"model": model, "texts": texts})
            assert status == 200, f"Expected 200, got {status}: {resp}"
            assert resp["component"] == "final_stance_inference"
            assert resp["kind"] == "provisional_model_suggestions"
            assert resp["human_validated"] is False
            assert "heuristic_oppose" in resp["pipeline"]
            assert len(resp["data"]) == 3
            valid_stances = {"Support", "Oppose", "Neutral", "Unclear"}
            for row in resp["data"]:
                assert row["final_stance"] in valid_stances, (
                    f"Unexpected final_stance: {row['final_stance']!r}"
                )
                assert row["disclaimer"], "disclaimer field must be non-empty"
                assert row["input_status"] == "ok"
                if row["final_stance"] == "Oppose":
                    assert row["prediction_method"] == "heuristic_oppose", (
                        "Oppose must always be heuristic_oppose"
                    )
            # caveats must mention the heuristic nature of Oppose
            caveat_text = " ".join(resp["caveats"])
            assert "heuristically" in caveat_text.lower() or "heuristic" in caveat_text.lower()

        # Bad model -> 400
        assert post_json(server, "/api/inference/stance", {"model": "badmodel", "texts": ["hi"]})[0] == 400

        # Too many texts -> 400
        assert post_json(server, "/api/inference/stance",
                         {"model": "linsvm", "texts": ["x"] * 17})[0] == 400

        # pipeline=v2 flag on /api/inference/consultation must also route to v2
        status2, resp2 = post_json(server, "/api/inference/consultation",
                                   {"model": "linsvm", "texts": ["Support open data."], "pipeline": "v2"})
        assert status2 == 200
        assert resp2["component"] == "final_stance_inference"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
