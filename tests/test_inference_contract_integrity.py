"""
tests/test_inference_contract_integrity.py
=============================================
Step 6 (model-integration) regression tests. Three gaps identified by inspecting the existing
inference contract (models/new_comment_inference.py, models/stance_inference_v2.py, app/api.py)
against the existing test suite (test_model_phase_v1.py, test_final_stance_inference.py,
test_api.py), which already cover tamper/hash-mismatch rejection, representative inputs, all
supported output labels, and empty-input handling for both the v1 (3-class) and v2 (heuristic
4-label) pipelines:

1. No existing test confirms that a run with a DIFFERENT declared label provenance (as opposed to
   a tampered hash) is rejected by load_verified_pipeline -- i.e. nothing locks in *why* the
   baseline was selected over the larger Sentiment140-augmented runs. This directly encodes the
   Step 6 finding: run_pool400_plus_sentiment140_400k_v1 and _800k_v1 are rejected by the
   verified-loading contract because their metadata.json declares
   label_type="mixed_ai_and_pseudo_unverified" (accurate -- they do mix pseudo-labels), not
   "ai_generated_unverified" -- a provenance/contract mismatch, NOT a performance comparison.
   This is a permanent guard against ever silently swapping in an incompatible artifact.
2. No existing test exercises the ArtifactError -> HTTP 503 path in app/api.py's do_POST.
3. No existing test asserts, as a standing regression guard, that inference never trains.

No production code is modified. No existing test file is modified.
"""
from __future__ import annotations

import inspect
import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASELINE_RUN = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_ai_v1_3class"
RUN_400K = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_plus_sentiment140_400k_v1"
RUN_800K = ROOT / "data" / "stance_training_ai_working_labels" / "run_pool400_plus_sentiment140_800k_v1"


def _post_json(server, path, body):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A. Which artifact the inference contract accepts -- not a size/performance question
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not BASELINE_RUN.is_dir(), reason="baseline run artifacts not present")
@pytest.mark.parametrize("model", ["linsvm", "logreg"])
class TestBaselineIsCompatible:
    def test_baseline_passes_verified_loading(self, model):
        from models.new_comment_inference import load_verified_pipeline
        pipe, meta = load_verified_pipeline(BASELINE_RUN, model)
        assert meta["label_type"] == "ai_generated_unverified"
        assert sorted(pipe.named_steps["clf"].classes_.tolist()) == ["Neutral", "Support", "Unclear"]


@pytest.mark.parametrize("run_dir,run_name", [(RUN_400K, "400k"), (RUN_800K, "800k")])
@pytest.mark.parametrize("model", ["linsvm", "logreg"])
class TestLargerAugmentedRunsAreNotSelectedByContract:
    """Confirms the 400k/800k Sentiment140-expansion runs are rejected by the SAME verified-loading
    contract the production API uses -- for a declared-provenance reason, not merely because their
    held-out macro-F1 is lower (Steps 3-4 already established that separately)."""

    def test_rejected_run_declares_mixed_provenance_not_the_expected_type(self, run_dir, run_name, model):
        if not run_dir.is_dir():
            pytest.skip(f"{run_name} run artifacts not present")
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["label_type"] != "ai_generated_unverified", (
            f"{run_name} is expected to declare mixed AI+pseudo-label provenance "
            "(accurately -- it mixes pool400 with Sentiment140/corpus pseudo-labels), which is "
            "exactly why load_verified_pipeline must refuse it, not swap it in."
        )

    def test_larger_run_raises_artifact_error_on_verified_loading(self, run_dir, run_name, model):
        if not run_dir.is_dir():
            pytest.skip(f"{run_name} run artifacts not present")
        from models.new_comment_inference import ArtifactError, load_verified_pipeline
        with pytest.raises(ArtifactError, match="unverified AI label provenance"):
            load_verified_pipeline(run_dir, model)


# ─────────────────────────────────────────────────────────────────────────────
# B. ArtifactError during predictor construction must surface as HTTP 503, not a crash
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceErrorHandling:
    def test_consultation_endpoint_returns_503_on_artifact_error(self):
        import app.api as api_mod
        from models.new_comment_inference import ArtifactError

        saved_cache = dict(api_mod._PREDICTORS)
        api_mod._PREDICTORS.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), api_mod.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch("models.new_comment_inference.load_verified_pipeline",
                       side_effect=ArtifactError("simulated artifact failure for test")):
                status, body = _post_json(server, "/api/inference/consultation",
                                           {"model": "linsvm", "texts": ["Some comment text here"]})
            assert status == 503
            assert body["error"] == "verified inference artifacts unavailable"
            assert "simulated artifact failure" in body["detail"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            api_mod._PREDICTORS.clear()
            api_mod._PREDICTORS.update(saved_cache)

    def test_stance_endpoint_returns_503_on_artifact_error(self):
        import app.api as api_mod
        from models.new_comment_inference import ArtifactError

        saved_cache = dict(api_mod._PREDICTORS)
        api_mod._PREDICTORS.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), api_mod.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch("models.stance_inference_v2.load_verified_pipeline",
                       side_effect=ArtifactError("simulated artifact failure for test")):
                status, body = _post_json(server, "/api/inference/stance",
                                           {"model": "linsvm", "texts": ["Some comment text here"]})
            assert status == 503
            assert body["error"] == "verified inference artifacts unavailable"
            assert "simulated artifact failure" in body["detail"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            api_mod._PREDICTORS.clear()
            api_mod._PREDICTORS.update(saved_cache)


# ─────────────────────────────────────────────────────────────────────────────
# C. No training during inference -- standing regression guard
# ─────────────────────────────────────────────────────────────────────────────

class TestNoTrainingDuringInference:
    def test_new_comment_predictor_never_calls_fit(self):
        from models import new_comment_inference as mod
        src = inspect.getsource(mod)
        assert ".fit(" not in src, "new_comment_inference.py must never call .fit() -- it is inference-only"
        assert "joblib.load" in src

    def test_final_stance_predictor_never_calls_fit(self):
        from models import stance_inference_v2 as mod
        src = inspect.getsource(mod)
        assert ".fit(" not in src, "stance_inference_v2.py must never call .fit() -- it is inference-only"
