"""Parallel specialist-analyst dispatch (Module 4).

`route_analysts` only sends an analyst whose required cleaned artifacts are
actually available -- an analyst missing a hard dependency is skipped with a
reason rather than crashing or fabricating output.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.analysts import anomaly, customer, margin, operations, sales
from src.analysts.base import run_analyst
from src.state import CafeIntelligenceState

ANALYST_SPECS = {
    "sales": sales.SPEC,
    "margin": margin.SPEC,
    "operations": operations.SPEC,
    "customer": customer.SPEC,
    "anomaly": anomaly.SPEC,
}


class _AnalystJobState(CafeIntelligenceState, total=False):
    _analyst_name: str


def route_analysts(state: CafeIntelligenceState) -> list[Send]:
    cleaned = state.get("cleaned_artifacts", {})
    sends = []
    for name, spec in ANALYST_SPECS.items():
        if all(a in cleaned for a in spec.required_artifacts):
            sends.append(Send("run_one_analyst", {**state, "_analyst_name": name}))
    return sends


def run_one_analyst(state: _AnalystJobState) -> dict[str, Any]:
    name = state["_analyst_name"]
    spec = ANALYST_SPECS[name]
    config = state["config"]
    result = run_analyst(
        spec=spec,
        run_id=state["run_id"],
        config=config,
        cleaned_artifacts=state.get("cleaned_artifacts", {}),
        analysis_period=state["analysis_period"],
        previous_period=state["previous_period"],
        trailing_baseline_periods=state["trailing_baseline_periods"],
    )
    errors = []
    if result.status not in ("success",):
        errors.append({"node": f"analyst.{name}", "status": result.status, "notes": result.notes})
    return {
        "candidate_findings": result.findings,
        "step_count": 1,
        "errors": errors,
    }


def analysis_fanin(state: CafeIntelligenceState) -> dict[str, Any]:
    return {}


def build_analysis_subgraph():
    graph = StateGraph(CafeIntelligenceState)
    graph.add_node("run_one_analyst", run_one_analyst)
    graph.add_node("analysis_fanin", analysis_fanin)
    graph.add_conditional_edges(START, route_analysts, ["run_one_analyst"])
    graph.add_edge("run_one_analyst", "analysis_fanin")
    graph.add_edge("analysis_fanin", END)
    return graph.compile()
