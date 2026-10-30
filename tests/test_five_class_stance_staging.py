"""Five-class staging pipeline provenance and API regression checks."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from app.api import Handler, _PREDICTORS
from models.context_stance_features import build_context_feature
from models.five_class_stance_inference import FiveClassStancePredictor
from models.stance_classification.schema import STANCE_LABELS
from models.stance_classification.labels import load_stance_reference_labels
from scripts.experiment_stance_reference_models_v2 import REFERENCE_LABELS_PATH
from scripts.generate_stance_five_class_synthetic_v1 import generate

ROOT = Path(__file__).resolve().parents[1]


def _post(server, body, path="/api/inference/stance/five-class-staging"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=90)
    connection.request("POST", path,
                       json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def test_feature_order_matches_training_contract():
    feature = build_context_feature("Q", "C", "T")
    assert feature == "[CONTEXT] C [QUESTION] Q [COMMENT] T"
    assert build_context_feature(None, None, "T") == (
        "[CONTEXT] [NO_CONTEXT] [QUESTION] [NO_QUESTION] [COMMENT] T"
    )


def test_synthetic_provenance_and_training_only_boundary():
    path = ROOT / "reports/stance_five_class_synthetic_v1/synthetic_training_rows.jsonl"
    rows = pd.read_json(path, lines=True)
    assert rows.is_synthetic.all() and rows.training_only.all() and rows.model_eligible.all()
    assert rows.original_record_id.isna().all()
    assert rows.groupby("reference_stance").size().to_dict() == {"Mixed": 120, "Oppose": 120, "Unclear": 120}
    protocol = pd.read_json(path.with_name("language_ability_protocol_only.jsonl"), lines=True)
    assert not protocol.model_eligible.any()
    assert protocol.stance_label.isna().all()


def test_generator_is_seed_reproducible_and_folds_keep_synthetic_out():
    first = generate(5, 20261030)
    second = generate(5, 20261030)
    pd.testing.assert_frame_equal(first, second)
    metrics = json.loads((ROOT / "reports/stance_five_class_staging_v1/metrics.json").read_text())
    assert metrics["synthetic_source"]["evaluation_synthetic_rows"] == 0
    assert metrics["oof_repeated_n"] == 1197
    assert metrics["original_counts"] == {"Support": 140, "Neutral": 219, "Oppose": 0, "Mixed": 0, "Unclear": 40}
    assignments = pd.read_json(ROOT / "reports/stance_five_class_staging_v1/fold_assignments.jsonl", lines=True)
    assert not assignments.is_synthetic.any()
    for _, fold in assignments.groupby(["repeat", "fold"]):
        train_groups = set(fold.loc[fold.split == "train", "leakage_group"])
        valid_groups = set(fold.loc[fold.split == "validation", "leakage_group"])
        assert train_groups.isdisjoint(valid_groups)


def test_feature_contract_excludes_target_derived_fields():
    manifest = json.loads((ROOT / "models/candidates/stance_context_wordchar_5class_synth_v1/candidate_manifest.json").read_text())
    assert manifest["feature_allowlist"] == ["question_context", "question_wording", "original_text"]
    forbidden = " ".join(manifest["forbidden_feature_families"]).casefold()
    assert all(term in forbidden for term in ("labels", "reviewer", "provenance", "sentiment", "synthetic metadata"))


def test_saved_candidate_integrity_and_five_labels():
    predictor = FiveClassStancePredictor()
    assert set(predictor.labels) == set(STANCE_LABELS)
    synthetic = pd.read_json(ROOT / "reports/stance_five_class_synthetic_v1/synthetic_training_rows.jsonl", lines=True)
    _, originals = load_stance_reference_labels(REFERENCE_LABELS_PATH, "expansion_v2")
    examples = [*synthetic.itertuples(), *originals.itertuples()]
    from models.context_stance_features import ContextPredictionInput
    items = [ContextPredictionInput(row.question_wording, row.question_context, row.original_text)
             for row in examples]
    observed_outputs = set(predictor.predict(items).final_stance.dropna())
    # Smoke-check only: these are training/reference examples, not evaluation evidence.
    assert observed_outputs == set(STANCE_LABELS)


def test_five_class_staging_api_validates_schema_and_preserves_production_contract():
    old_predictors = dict(_PREDICTORS)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, response = _post(server, {"items": [{"question": "", "context": None,
                                                       "comment": "I reject this policy outright."}]})
        assert status == 200
        assert response["component"] == "five_class_stance_staging_inference"
        assert response["labels_supported"] == list(STANCE_LABELS)
        assert response["data"][0]["input_status"] == "ok"
        assert response["data"][0]["context_missing"] is True
        assert response["data"][0]["final_stance"] in STANCE_LABELS
        assert not ({"sentiment_suggestion", "sentiment_prediction"} & set(response["data"][0]))

        # Maximum accepted comment length is allowed; longer inputs are rejected.
        assert _post(server, {"items": [{"comment": "x" * 5000}]})[0] == 200
        assert _post(server, {"items": [{"comment": "x" * 5001}]})[0] == 400

        # Empty input yields an explicit no-prediction state.
        status, empty = _post(server, {"items": [{"comment": "  "}]})
        assert status == 200 and empty["data"][0]["final_stance"] is None
        assert empty["data"][0]["input_status"] == "too_short_or_empty"

        # Unknown target-like fields and invalid types are rejected.
        assert _post(server, {"items": [{"comment": "Relevant text", "reference_stance": "Oppose"}]})[0] == 400
        assert _post(server, {"items": [{"comment": "Relevant text", "sentiment": "Negative"}]})[0] == 400
        assert _post(server, {"items": [{"comment": 7}]})[0] == 400
        assert _post(server, {"items": [{"comment": "abc", "question": []}]})[0] == 400
        assert _post(server, {"items": [{"comment": "abc", "extra": "value"}]})[0] == 400
        # Existing production route contract remains its original three classes.
        prod_status, production = _post(server, {"texts": ["I support open data."]}, "/api/inference/stance")
        assert prod_status == 200
        assert production["data"][0]["final_stance"] in {"Support", "Neutral", "Unclear"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _PREDICTORS.clear()
        _PREDICTORS.update(old_predictors)
