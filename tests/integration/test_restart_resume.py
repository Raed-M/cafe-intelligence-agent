"""Proves process-restart resume: a run paused at the HITL gate, persisted via
the real SQLite checkpointer, can be resumed by a *second, independent*
compiled graph instance (simulating a fresh process) using only the
on-disk checkpoint DB and thread_id -- no in-memory state carried over.
Also proves cross-session memory: a second MemoryStore instance opened fresh
can read a previous run's findings/content history.
"""
import uuid
from datetime import date
from pathlib import Path

import pytest

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph
from src.persistence.checkpointer import build_checkpointer
from src.persistence.memory_store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_llm(monkeypatch):
    from tests.integration.test_full_pipeline import _FakeChatModel

    monkeypatch.setattr("src.tools.llm_factory.get_chat_model", lambda model_name, temperature=0: _FakeChatModel())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


def test_restart_resume_across_independent_graph_instances(fake_llm, tmp_path):
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    memory_db = tmp_path / "memory.sqlite"

    config = resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
        checkpoint_db=checkpoint_db, memory_db=memory_db,
    )
    run_id = "restart_" + uuid.uuid4().hex[:8]
    thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
    initial_state = {
        "run_id": run_id, "thread_id": run_id, "config": config,
        "analysis_period": config.analysis_period, "previous_period": config.previous_period,
        "trailing_baseline_periods": config.trailing_baseline_periods,
        "recommendation_period": config.recommendation_period,
        "critic_round": 0, "content_repair_attempts": 0,
    }

    # "Process 1": run up to the HITL pause, then drop this graph/checkpointer entirely.
    checkpointer_1 = build_checkpointer(checkpoint_db)
    graph_1 = build_main_graph(checkpointer=checkpointer_1)
    out_1 = graph_1.invoke(initial_state, config=thread_config)
    snapshot_1 = graph_1.get_state(thread_config)
    assert snapshot_1.next == ("human_gate",)
    del graph_1, checkpointer_1  # simulate process exit

    # "Process 2": brand-new checkpointer + graph instance pointed at the same DB file.
    checkpointer_2 = build_checkpointer(checkpoint_db)
    graph_2 = build_main_graph(checkpointer=checkpointer_2)
    snapshot_resumed = graph_2.get_state(thread_config)
    assert snapshot_resumed.next == ("human_gate",), "a fresh process must see the same paused state"
    assert snapshot_resumed.values["report"]["html_path"] == out_1["report"]["html_path"]

    graph_2.update_state(thread_config, {"human_decision": "approve"})
    final_out = graph_2.invoke(None, config=thread_config)
    assert final_out["run_status"] == "succeeded"

    # Cross-session memory: a brand-new MemoryStore instance reads what was persisted.
    store = MemoryStore(memory_db)
    recent = store.recent_findings(config.profile_key)
    assert len(recent) >= 1
    store.close()
