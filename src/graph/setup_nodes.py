from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.types import Send

from src.config.preflight import run_preflight
from src.context.context_builder import build_context_bundle
from src.graph.ingestion_subgraph import dispatch_ingestion
from src.state import CafeIntelligenceState


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
