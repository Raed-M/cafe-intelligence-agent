"""Full CafeIntelligenceAgent StateGraph, mirroring implementation_plan_final.md
section 6.1. Built as one flat graph (rather than nested compiled subgraphs)
since every stage shares the same CafeIntelligenceState; each Send-based fan-out
stage still has its own dispatcher + fan-in node pair, exactly as validated in
isolation in the ingestion/analysis/critic subgraph tests.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.analysis_subgraph import analysis_fanin, route_analysts, run_one_analyst
from src.graph.cross_domain_nodes import cross_domain_synthesis_node
from src.graph.content_nodes import (
    content_agent_node,
    increment_repair_attempts,
    route_after_content_validation,
    validate_content_node,
)
from src.graph.hitl import deliver, human_gate, persist_run, route_human_decision, stop_rejected
from src.graph.ingestion_subgraph import ingestion_fanin, parse_source
from src.graph.routers import (
    critic_node,
    no_evidence_node,
    rank_node,
    revision_fanin,
    route_after_critic,
    run_one_analyst_revision,
)
from src.graph.setup_nodes import (
    abort_node,
    build_context_node,
    guard_limits,
    preflight_dataset,
    resolve_config,
    route_after_guard,
    route_after_preflight,
)
from src.cleaning.cleaner import clean_and_materialise
from src.reporting.report_generator import generate_report
from src.state import CafeIntelligenceState


def build_main_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(CafeIntelligenceState)

    graph.add_node("resolve_config", resolve_config)
    graph.add_node("preflight_dataset", preflight_dataset)
    graph.add_node("guard_limits", guard_limits)
    graph.add_node("abort", abort_node)

    graph.add_node("parse_source", parse_source)
    graph.add_node("ingestion_fanin", ingestion_fanin)
    graph.add_node("clean_and_materialise", clean_and_materialise)

    graph.add_node("run_one_analyst", run_one_analyst)
    graph.add_node("analysis_fanin", analysis_fanin)
    graph.add_node("cross_domain_synthesis", cross_domain_synthesis_node)

    graph.add_node("critic", critic_node)
    graph.add_node("run_one_analyst_revision", run_one_analyst_revision)
    graph.add_node("revision_fanin", revision_fanin)
    graph.add_node("rank", rank_node)
    graph.add_node("no_evidence", no_evidence_node)

    graph.add_node("build_context", build_context_node)
    graph.add_node("content_agent", content_agent_node)
    graph.add_node("validate_content", validate_content_node)
    graph.add_node("increment_repair_attempts", increment_repair_attempts)

    graph.add_node("report_generator", generate_report)
    graph.add_node("human_gate", human_gate)
    graph.add_node("deliver", deliver)
    graph.add_node("stop_rejected", stop_rejected)
    graph.add_node("persist_run", persist_run)

    graph.add_edge(START, "resolve_config")
    graph.add_edge("resolve_config", "preflight_dataset")
    graph.add_conditional_edges("preflight_dataset", route_after_preflight, {"guard": "guard_limits", "abort": "abort"})
    graph.add_conditional_edges("guard_limits", route_after_guard, ["parse_source", "abort"])

    graph.add_edge("parse_source", "ingestion_fanin")
    graph.add_edge("ingestion_fanin", "clean_and_materialise")
    graph.add_conditional_edges("clean_and_materialise", route_analysts, ["run_one_analyst"])
    graph.add_edge("run_one_analyst", "analysis_fanin")
    # Synthesis sits between the analyst fan-in and the critic: it is the only
    # stage that sees every analyst's grounded evidence at once, and it runs
    # before validation so its output is held to exactly the same standard as
    # a single-domain finding. Revision reruns re-enter at `critic` directly,
    # so synthesis happens once per run, never once per revision round.
    graph.add_edge("analysis_fanin", "cross_domain_synthesis")
    graph.add_edge("cross_domain_synthesis", "critic")

    graph.add_conditional_edges(
        "critic", route_after_critic, ["run_one_analyst_revision", "rank", "no_evidence"]
    )
    graph.add_edge("run_one_analyst_revision", "revision_fanin")
    graph.add_edge("revision_fanin", "critic")

    graph.add_edge("rank", "build_context")
    graph.add_edge("no_evidence", "build_context")
    graph.add_edge("build_context", "content_agent")
    graph.add_edge("content_agent", "validate_content")
    graph.add_conditional_edges(
        "validate_content", route_after_content_validation,
        {"repair": "increment_repair_attempts", "report": "report_generator", "no_evidence": "report_generator"},
    )
    graph.add_edge("increment_repair_attempts", "content_agent")

    graph.add_edge("report_generator", "human_gate")
    graph.add_conditional_edges(
        "human_gate", route_human_decision, {"deliver": "deliver", "report_generator": "report_generator", "stop": "stop_rejected"}
    )
    graph.add_edge("deliver", "persist_run")
    graph.add_edge("stop_rejected", "persist_run")
    graph.add_edge("abort", "persist_run")
    graph.add_edge("persist_run", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["human_gate"])


def graph(config=None):  # noqa: ARG001 -- required factory signature, see langgraph.json
    """Factory form for `langgraph dev` / LangGraph Studio (langgraph.json
    points here directly -- no wrapper graph). `langgraph dev` calls this
    with a RunnableConfig we don't need: real per-run input arrives through
    graph state via resolve_config above, identically to every other caller
    of build_main_graph. The platform supplies its own persistence layer, so
    no checkpointer is passed here."""
    return build_main_graph()
