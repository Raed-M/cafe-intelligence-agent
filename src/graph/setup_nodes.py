from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Send

from src.config.preflight import run_preflight
from src.context.context_builder import build_context_bundle
from src.graph.ingestion_subgraph import dispatch_ingestion
from src.state import CafeIntelligenceState

_DEFAULT_PROFILE_PATH = "data/qahwa_saihat/cafe_profile.json"
_DEFAULT_DATA_DIR = "data/qahwa_saihat"
_DEFAULT_APP_SETTINGS_PATH = "config/app_settings.yaml"
_DEFAULT_SOURCE_REGISTRY_PATH = "config/source_registry.yaml"


def resolve_config(state: CafeIntelligenceState) -> dict[str, Any]:
    """First node in the graph. Every existing caller (scripts/run_week.py,
    scheduler/run.py, the test suite) already builds a resolved
    RuntimeCafeConfig via resolve_runtime_config() and passes it as
    state["config"] before invoking -- for them this is a pure no-op
    passthrough. A caller that can't construct that object itself (LangGraph
    Studio's JSON input form) can start the graph with the plain fields
    instead (profile_path/data_dir/target_week/...), and this node resolves
    the real config + run_id + periods from those. Either way the rest of
    the graph is unchanged and identical -- this is the only node whose
    behavior differs by caller."""
    if state.get("config") is not None:
        return {}

    from src.config.runtime_config import resolve_runtime_config

    config = resolve_runtime_config(
        profile_path=Path(state.get("profile_path") or _DEFAULT_PROFILE_PATH),
        data_dir=Path(state.get("data_dir") or _DEFAULT_DATA_DIR),
        app_settings_path=Path(state.get("app_settings_path") or _DEFAULT_APP_SETTINGS_PATH),
        source_registry_path=Path(state.get("source_registry_path") or _DEFAULT_SOURCE_REGISTRY_PATH),
        target_week=date.fromisoformat(state["target_week"]) if state.get("target_week") else None,
    )
    run_id = state.get("run_id") or f"studio_{uuid.uuid4().hex[:8]}"
    return {
        "run_id": run_id, "thread_id": state.get("thread_id") or run_id, "config": config,
        "analysis_period": config.analysis_period, "previous_period": config.previous_period,
        "trailing_baseline_periods": config.trailing_baseline_periods,
        "recommendation_period": config.recommendation_period,
        "critic_round": 0, "content_repair_attempts": 0,
    }


def preflight_dataset(state: CafeIntelligenceState) -> dict[str, Any]:
    report = run_preflight(state["config"])
    if not report.ok:
        return {
            "run_status": "failed",
            "errors": [{"node": "preflight_dataset", "errors": report.errors}],
            "step_count": 1,
        }
    return {"step_count": 1, "_started_at": datetime.now(timezone.utc).isoformat()}


def route_after_preflight(state: CafeIntelligenceState) -> Literal["guard", "abort"]:
    return "abort" if state.get("run_status") == "failed" else "guard"


def guard_limits(state: CafeIntelligenceState) -> dict[str, Any]:
    limits = state["config"].app_settings.limits
    if state.get("step_count", 0) >= limits.max_graph_steps or state.get("cost_usd", 0.0) >= limits.max_cost_usd:
        return {"run_status": "failed", "errors": [{"node": "guard_limits", "reason": "step/cost limit exceeded"}]}
    return {}


def route_after_guard(state: CafeIntelligenceState) -> str | list[Send]:
    if state.get("run_status") == "failed":
        return "abort"
    return dispatch_ingestion(state)


def abort_node(state: CafeIntelligenceState) -> dict[str, Any]:
    return {"run_status": "failed", "step_count": 1}


def build_context_node(state: CafeIntelligenceState) -> dict[str, Any]:
    bundle = build_context_bundle(state["config"], state["recommendation_period"], state.get("cleaned_artifacts", {}))
    return {"context_bundle": bundle, "step_count": 1}
