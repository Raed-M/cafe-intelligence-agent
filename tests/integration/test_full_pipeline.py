"""End-to-end proof that the assembled main graph runs the full pipeline
described in implementation_plan_final.md section 6.1 against the real
Qahwa Saihat dataset: ingestion -> cleaning -> analysis -> critic -> context
-> content -> validation -> report -> HITL pause -> approve -> deliver ->
persist. All LLM call sites are monkeypatched to a single fake chat model that
inspects the system prompt to decide what to return, so this test proves the
graph *wiring* and every deterministic contract without spending real API
calls or depending on a live key.
"""
import json
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph
from tests.fakes import _FakeChatModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr("src.tools.llm_factory.get_chat_model", lambda model_name, temperature=0: _FakeChatModel())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield


def test_full_pipeline_reaches_human_gate_and_delivers(fake_llm):
    config = resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
        checkpoint_db=ROOT / "outputs" / "test_evidence" / "checkpoints_test.sqlite",
        memory_db=ROOT / "outputs" / "test_evidence" / "memory_test.sqlite",
    )
    run_id = "e2e_" + uuid.uuid4().hex[:8]
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

    # Graph should have paused before human_gate (no human_decision recorded yet).
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("human_gate",)
    assert out["report"]["html_path"]
    assert Path(out["report"]["html_path"]).exists()
    assert out["data_quality"]["sources_successful"]

    # Approve and resume.
    graph.update_state(thread_config, {"human_decision": "approve"})
    final_out = graph.invoke(None, config=thread_config)
    assert final_out["run_status"] == "succeeded"
    assert final_out["delivery_receipt"] is not None
