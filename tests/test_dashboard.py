"""
tests/test_dashboard.py
=======================
Smoke and behaviour tests of the Streamlit dashboard (app/dashboard.py) with streamlit.testing.v1.AppTest.
Real-project tests only read artifacts; failure-isolation tests use broken copies of a synthetic project tree.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from streamlit.testing.v1 import AppTest  # noqa: E402

from test_results_access import built  # noqa: E402,F401  (module-scoped synthetic project fixture)

APP = str(ROOT / "app" / "dashboard.py")
PAGES = ("Overview", "Concern groups", "Exploratory prioritization", "Keywords by language", "Evaluation & limitations")
REAL = (ROOT / "data/full_corpus/concern_prioritization_v1/run_manifest.json").exists()


def run_app(page: str | None = None, root: Path | None = None) -> AppTest:
    if root is None:
        os.environ.pop("POLICYLENS_ROOT", None)
    else:
        os.environ["POLICYLENS_ROOT"] = str(root)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page:
        at.sidebar.radio(key="page").set_value(page).run()
    return at


def texts(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "warning", "info", "error", "success", "caption", "header", "subheader", "title")
             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(autouse=True)
def _clean_env():
    yield
    os.environ.pop("POLICYLENS_ROOT", None)


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_real_project(page):
    at = run_app(page)
    assert not at.exception, at.exception
    t = texts(at)
    assert "not human-validated" in t and "cannot support conclusions about Indian citizens" in t
    assert "does not analyse new comments" in t


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
def test_overview_shows_verification():
    at = run_app()
    assert any("All components verified" in s.value for s in at.success)
    df = at.dataframe[0].value
    assert set(df["status"]) == {"VERIFIED"} and len(df) == 5


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
def test_concern_group_evidence_has_source_ids():
    at = run_app("Concern groups")
    at.multiselect(key="tier_filter").set_value(["candidate"]).run()
    assert set(at.dataframe[0].value["tier"]) == {"candidate"}
    at.selectbox(key="group_select").set_value(30).run()
    ev = at.dataframe[1].value
    assert {"corpus_id", "source dataset", "source record", "quote"} <= set(ev.columns) and len(ev) > 0


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
def test_prioritization_shows_baseline_and_exploratory_labels():
    at = run_app("Exploratory prioritization")
    assert not at.exception
    ranking, baseline = at.dataframe[0].value, at.dataframe[1].value
    assert "exploratory score" in ranking.columns and list(baseline["rank"]) == sorted(baseline["rank"])
    t = texts(at)
    assert "not a validated measure of public importance" in t
    assert "importance score" not in t.lower()


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
def test_keywords_language_switch():
    at = run_app("Keywords by language")
    at.radio(key="kw_lang").set_value("French").run()
    assert not at.exception
    assert at.metric[0].value == "15"


@pytest.mark.skipif(not REAL, reason="real artifacts not present")
def test_evaluation_button_runs_read_only_check():
    at = run_app("Evaluation & limitations")
    at.button(key="run_eval").click().run(timeout=300)
    assert not at.exception
    t = texts(at)
    assert "BLOCKED:" in t                                   # summary line, e.g. "BLOCKED: 10, NOT RUN: 2, PASSED: 50"
    assert not any("check(s) FAILED" in e.value for e in at.error)
    assert "AI-assisted working-label baseline" in t and "not human-validated stance accuracy" in t


def test_broken_component_is_isolated(built, tmp_path):
    proj = tmp_path / "p"
    shutil.copytree(built, proj)
    p = proj / "data/full_corpus/concern_prioritization_v1/ranking.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    at = run_app(root=proj)
    assert not at.exception
    assert any("prioritization: MISMATCH" in w.value for w in at.warning)
    at.sidebar.radio(key="page").set_value("Exploratory prioritization").run()
    assert not at.exception and any("MISMATCH" in e.value for e in at.error)
    at.sidebar.radio(key="page").set_value("Concern groups").run()
    assert not at.exception and len(at.dataframe) >= 1                  # other components still shown


def test_missing_keywords_component(built, tmp_path):
    proj = tmp_path / "p"
    shutil.copytree(built, proj)
    (proj / "data/full_corpus/language_keywords_v1/run_manifest.json").unlink()
    at = run_app("Keywords by language", root=proj)
    assert not at.exception and any("MISSING" in e.value for e in at.error)


def test_dashboard_only_uses_results_store():
    src = Path(APP).read_text(encoding="utf-8")
    for forbidden in ("read_csv(", "np.load(", "json.load(", "open(", "joblib", "stance_training_ai_working_labels"):
        assert forbidden not in src, forbidden
