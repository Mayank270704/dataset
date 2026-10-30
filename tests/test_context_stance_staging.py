import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from app.api import Handler
from models.context_stance_features import ContextPredictionInput, build_context_feature
from models.context_stance_inference import ContextStancePredictor


def post(server, path, body):
    conn = HTTPConnection("127.0.0.1", server.server_port, timeout=60)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    result = response.status, json.loads(response.read())
    conn.close()
    return result


def test_shared_feature_builder_marks_missing_context_and_excludes_language():
    absent = build_context_feature("Question?", None, "Comment")
    present = build_context_feature("Question?", "", "Comment")
    assert absent == "[CONTEXT] [NO_CONTEXT] [QUESTION] Question? [COMMENT] Comment"
    assert "[NO_CONTEXT]" in absent and "[NO_CONTEXT]" in present
    assert "Question?" in absent and "Comment" in absent
    assert "Klingon" not in build_context_feature("Q", "C", "T")


def test_input_validation_allows_unsupported_language_but_bounds_long_text():
    item = ContextPredictionInput("Q", None, "valid comment", "Klingon")
    item.validate()
    assert item.comment_is_usable
    with pytest.raises(ValueError, match="comment must be at most"):
        ContextPredictionInput("Q", None, "x" * 5001).validate()
    assert not ContextPredictionInput("Q", None, "  ").comment_is_usable


def test_context_staging_route_realistic_inputs_and_response_contract():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        items = [
            {"question": "Should the proposal proceed?", "context": "Consultation overview.",
             "comment": "I support this practical proposal."},
            {"question": "Should the proposal proceed?", "comment": "The proposal seems reasonable."},
            {"question": "", "context": None, "comment": "Details are missing.", "language": "Klingon"},
            {"question": "Q" * 2000, "context": "C" * 5000, "comment": "L" * 5000},
            {"question": "Question?", "comment": "   "},
            {"question": "Question?", "comment": "ab"},
            {"question": "Should this proposal be adopted?", "context": "Consultation prompt.",
             "comment": "This terrible policy should be stopped."},
        ]
        status, response = post(server, "/api/inference/stance/context-staging", {"items": items})
        assert status == 200, response
        assert response["component"] == "context_staging_inference"
        assert response["candidate"] == "stance_context_wordchar_v1"
        assert response["human_validated"] is False
        assert set(response["labels_supported"]) == {"Support", "Neutral", "Unclear"}
        assert len(response["data"]) == len(items)
        assert all(row["final_stance"] in {"Support", "Neutral", "Unclear"}
                   for row in response["data"][:4])
        assert response["data"][1]["context_missing"] is True
        assert response["data"][2]["question_missing"] is True
        assert response["data"][2]["language_status"] == "ignored_not_a_model_feature"
        assert response["data"][3]["input_status"] == "ok"
        assert response["data"][4]["input_status"] == "too_short_or_empty"
        assert response["data"][4]["final_stance"] is None
        assert response["data"][5]["input_status"] == "too_short_or_empty"
        assert response["data"][6]["final_stance"] in {"Support", "Neutral", "Unclear"}
        assert response["data"][6]["final_stance"] != "Oppose"
        assert not any("sentiment" in key for key in response["data"][0])

        status, bad = post(server, "/api/inference/stance/context-staging", {
            "items": [{"comment": "Valid comment", "gold_stance": "Oppose"}],
        })
        assert status == 400 and "unsupported fields" in bad["error"]
        status, bad = post(server, "/api/inference/stance/context-staging", {
            "items": [{"comment": "Valid comment", "language": 4}],
        })
        assert status == 400 and "language must be a string" in bad["error"]
        status, bad = post(server, "/api/inference/stance/context-staging", {
            "items": [{"comment": "Valid comment", "sentiment": "Negative"}],
        })
        assert status == 400 and "unsupported fields" in bad["error"]
        status, bad = post(server, "/api/inference/stance/context-staging", {
            "items": [{"comment": None}],
        })
        assert status == 400 and "comment must be a string" in bad["error"]
        for invalid in (
            {"items": [{"comment": "Valid comment", "question": 7}]},
            {"items": [{"comment": "Valid comment", "context": []}]},
            {"items": ["not an object"]},
            {"items": [{"comment": "Valid comment"}], "sentiment": "Negative"},
        ):
            status, _ = post(server, "/api/inference/stance/context-staging", invalid)
            assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_staging_endpoint_accepts_existing_examples_for_all_three_labels():
    from models.stance_classification.labels import load_stance_reference_labels
    from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH

    status, frame = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    assert status.state == "ok" and frame is not None
    sample_ids = {"pilot_001": "Support", "pilot_003": "Neutral", "pilot_002": "Unclear"}
    examples = frame[frame.pilot_id.isin(sample_ids)].set_index("pilot_id")
    assert {key: examples.loc[key, "gold_stance"] for key in sample_ids} == sample_ids
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        items = [{"question": examples.loc[key, "question_wording"],
                  "context": examples.loc[key, "question_context"],
                  "comment": examples.loc[key, "original_text"]} for key in sample_ids]
        status_code, response = post(server, "/api/inference/stance/context-staging", {"items": items})
        assert status_code == 200
        assert [row["final_stance"] for row in response["data"]] == list(sample_ids.values())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_staging_predictor_artifact_integrity_and_only_supported_classes():
    predictor = ContextStancePredictor()
    assert set(predictor.labels) == {"Support", "Neutral", "Unclear"}
    rows = predictor.predict([ContextPredictionInput("Question?", None, "A valid comment.")])
    assert rows.iloc[0].final_stance in {"Support", "Neutral", "Unclear"}


def test_inference_passes_training_feature_order_and_format(monkeypatch):
    predictor = ContextStancePredictor()

    class Recorder:
        def __init__(self):
            self.inputs = None

        def predict(self, features):
            self.inputs = list(features)
            return ["Support"]

        def decision_function(self, features):
            return [[0.8, 0.2, -0.1]]

    recorder = Recorder()
    monkeypatch.setattr(predictor, "pipeline", recorder)
    item = ContextPredictionInput("Which change?", "Consultation on access.", "Please expand access.")
    predictor.predict([item])
    assert recorder.inputs == [build_context_feature(item.question, item.context, item.comment)]
    assert recorder.inputs[0] == (
        "[CONTEXT] Consultation on access. [QUESTION] Which change? "
        "[COMMENT] Please expand access."
    )
