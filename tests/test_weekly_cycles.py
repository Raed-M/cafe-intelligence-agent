"""Ten weekly-cycle harness (plan section 28.3): runs the full graph across
ten Monday-start weeks spanning normal, Ramadan, Eid, launch, sensor-outage
and summer contexts, and records an honest status table.

Per M18A, these ten cycles run in fixture/offline mode for Tavily (search
unavailable, disclosed via `search_status`) so results are reproducible
without depending on live network state; a separate live Tavily call is
proven in `tests/integration/test_full_pipeline.py` /
`src/context/context_builder.py` smoke testing.
"""
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph
from tests.integration.test_full_pipeline import _FakeChatModel

ROOT = Path(__file__).resolve().parent.parent

CYCLES = [
    ("2026-01-05", "First complete normal week/baseline edge"),
    ("2026-01-26", "Normal winter comparison"),
    ("2026-02-16", "Ramadan transition and Founding Day context"),
    ("2026-02-23", "Ramadan night behaviour"),
    ("2026-03-16", "Ramadan/Eid al-Fitr transition"),
    ("2026-04-06", "Mid-period Matcha launch; launch-aware tests"),
    ("2026-05-25", "Eid al-Adha period"),
    ("2026-06-08", "Sensor outage and supplier delivery delay"),
    ("2026-07-06", "Peak summer and Summer Nights event"),
    ("2026-07-20", "Latest complete supplied week"),
]


@dataclass
class CycleResult:
    week_starting: str
    purpose: str
    status: str
    sources_successful: int
    sources_partial: int
    sources_failed: int
    candidate_findings: int
    approved_findings: int
    rejected_findings: int
    final_findings: int
    content_valid: bool
    steps: int
    duration_seconds: float
    primary_failure_layer: str
    diagnosis: str


def _run_one_cycle(week_starting: str, purpose: str, monkeypatch) -> CycleResult:
    monkeypatch.setattr("src.tools.llm_factory.get_chat_model", lambda model_name, temperature=0: _FakeChatModel())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    config = resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date.fromisoformat(week_starting),
        artifact_root=ROOT / "outputs" / "artifacts",
        checkpoint_db=ROOT / "outputs" / "test_evidence" / "cycles_checkpoints.sqlite",
        memory_db=ROOT / "outputs" / "test_evidence" / "cycles_memory.sqlite",
    )
    run_id = f"cycle_{week_starting}_" + uuid.uuid4().hex[:6]
    graph = build_main_graph()
    thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 300}
    initial_state = {
        "run_id": run_id, "thread_id": run_id, "config": config,
        "analysis_period": config.analysis_period, "previous_period": config.previous_period,
        "trailing_baseline_periods": config.trailing_baseline_periods,
        "recommendation_period": config.recommendation_period,
        "critic_round": 0, "content_repair_attempts": 0,
    }

    start = time.monotonic()
    primary_failure_layer = "none"
    diagnosis = "completed normally"
    try:
        out = graph.invoke(initial_state, config=thread_config)
    except Exception as e:  # noqa: BLE001
        duration = time.monotonic() - start
        return CycleResult(
            week_starting, purpose, "failed", 0, 0, 0, 0, 0, 0, 0, False,
            0, duration, "plan", f"graph invocation raised: {type(e).__name__}: {e}",
        )
    duration = time.monotonic() - start

    dq = out.get("data_quality", {})
    critic = out.get("critic_results", {})
    content_validation = out.get("content_validation", {})

    if dq.get("sources_failed"):
        primary_failure_layer = "tool"
        diagnosis = f"source(s) failed: {dq['sources_failed']}"
    elif not out.get("final_findings"):
        primary_failure_layer = "model"
        diagnosis = "no findings survived critic approval this week"
    elif content_validation and not content_validation.get("valid", True):
        primary_failure_layer = "model"
        diagnosis = f"content validation issues: {content_validation.get('issues')}"

    return CycleResult(
        week_starting=week_starting, purpose=purpose, status=out.get("run_status", "unknown"),
        sources_successful=len(dq.get("sources_successful", [])),
        sources_partial=len(dq.get("sources_partial", [])),
        sources_failed=len(dq.get("sources_failed", [])),
        candidate_findings=len(out.get("candidate_findings", [])),
        approved_findings=len(critic.get("approved_findings", [])),
        rejected_findings=len(critic.get("rejected_findings", [])),
        final_findings=len(out.get("final_findings", [])),
        content_valid=bool(content_validation.get("valid")) if content_validation else False,
        steps=out.get("step_count", 0),
        duration_seconds=round(duration, 2),
        primary_failure_layer=primary_failure_layer,
        diagnosis=diagnosis,
    )


def test_ten_weekly_cycles(monkeypatch):
    results = [_run_one_cycle(week, purpose, monkeypatch) for week, purpose in CYCLES]

    out_dir = ROOT / "outputs" / "test_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_dir / "ten_weekly_cycles.csv", index=False)
    (out_dir / "ten_weekly_cycles.md").write_text(df.to_markdown(index=False), encoding="utf-8")

    # Every cycle must reach a defined, honest terminal status -- never crash
    # the harness and never silently claim success without evidence.
    for r in results:
        assert r.status in ("succeeded", "partial", "failed", "rejected"), r
        if r.status == "failed":
            assert r.primary_failure_layer != "none"

    # All seven sources are readable for every week in this dataset's coverage,
    # so ingestion itself should never fail across all ten cycles.
    assert all(r.sources_failed == 0 for r in results)
