"""Second-cafe generic-onboarding proof (plan section 28.7 / M30). Runs the
full graph against a different profile/data directory (Sundown Roasters,
Jeddah) and asserts the output actually reflects the new profile -- location
queries, prayer-time coordinates, report title, social handles -- with zero
application source changes (proven separately via `git diff`, see
outputs/test_evidence/second_cafe_git_diff.txt).
"""
import uuid
from datetime import date
from pathlib import Path

import pytest

from src.config.runtime_config import resolve_runtime_config
from src.context.prayer_times import compute_prayer_times
from src.graph.main_graph import build_main_graph
from src.tools.tavily_search import build_local_queries
from tests.integration.test_full_pipeline import _FakeChatModel

ROOT = Path(__file__).resolve().parent.parent
SECOND_CAFE_DIR = ROOT / "data" / "sundown_roasters"


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr("src.tools.llm_factory.get_chat_model", lambda model_name, temperature=0: _FakeChatModel())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def test_second_cafe_preflight_and_config_resolve():
    config = resolve_runtime_config(
        profile_path=SECOND_CAFE_DIR / "cafe_profile.json", data_dir=SECOND_CAFE_DIR,
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 2, 2),
    )
    assert config.raw_profile.cafe_name == "Sundown Roasters"
    assert config.raw_profile.city == "Jeddah"
    assert config.local_search_terms[0] == "Jeddah"
    assert config.social_handles == {"instagram": "@sundown.roasters", "tiktok": "@sundownroasters"}


def test_second_cafe_queries_and_prayer_times_reflect_new_profile():
    """No hardcoded location: queries and prayer times must be derived from
    THIS profile's fields/coordinates, not Qahwa's Saihat/Qatif values."""
    config = resolve_runtime_config(
        profile_path=SECOND_CAFE_DIR / "cafe_profile.json", data_dir=SECOND_CAFE_DIR,
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 2, 2),
    )
    queries = build_local_queries(config.local_search_terms, config.recommendation_period["start"], config.recommendation_period["end"])
    assert any("Jeddah" in q for q in queries)
    assert not any("Saihat" in q or "Qatif" in q for q in queries)

    coords = config.raw_profile.coordinates
    pt = compute_prayer_times(date(2026, 2, 9), coords.lat, coords.lng, config.raw_profile.timezone, config.prayer_calculation_method)
    # Jeddah is west of Saihat/Qatif; Maghrib should differ from the Eastern Province value.
    pt_qahwa = compute_prayer_times(date(2026, 2, 9), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    assert pt.maghrib != pt_qahwa.maghrib


def test_second_cafe_full_pipeline_runs_and_reflects_profile(fake_llm):
    config = resolve_runtime_config(
        profile_path=SECOND_CAFE_DIR / "cafe_profile.json", data_dir=SECOND_CAFE_DIR,
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 2, 2),
        artifact_root=ROOT / "outputs" / "artifacts",
        checkpoint_db=ROOT / "outputs" / "test_evidence" / "second_cafe_checkpoints.sqlite",
        memory_db=ROOT / "outputs" / "test_evidence" / "second_cafe_memory.sqlite",
    )
    run_id = "second_cafe_" + uuid.uuid4().hex[:8]
    graph = build_main_graph()
    thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
    initial_state = {
        "run_id": run_id, "thread_id": run_id, "config": config,
        "analysis_period": config.analysis_period, "previous_period": config.previous_period,
        "trailing_baseline_periods": config.trailing_baseline_periods,
        "recommendation_period": config.recommendation_period,
        "critic_round": 0, "content_repair_attempts": 0,
    }
    out = graph.invoke(initial_state, config=thread_config)

    dq = out["data_quality"]
    assert dq["sources_failed"] == []
    assert set(dq["sources_successful"]) == {"pos", "menu", "traffic", "staff", "inventory", "emails", "reviews"}

    html = Path(out["report"]["html_path"]).read_text(encoding="utf-8")
    assert "Sundown Roasters" in html
    assert "Qahwa" not in html
