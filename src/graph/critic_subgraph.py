"""Critic + targeted revision + ranking subgraph (Module 5).

critic -> [revise -> revision_fanin -> critic]* (<= critic_revision_rounds) -> rank | no_evidence
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.routers import (
    critic_node,
    no_evidence_node,
    rank_node,
    revision_fanin,
    route_after_critic,
    run_one_analyst_revision,
)
from src.state import CafeIntelligenceState


def build_critic_subgraph():
    graph = StateGraph(CafeIntelligenceState)
    graph.add_node("critic", critic_node)
    graph.add_node("run_one_analyst_revision", run_one_analyst_revision)
    graph.add_node("revision_fanin", revision_fanin)
    graph.add_node("rank", rank_node)
    graph.add_node("no_evidence", no_evidence_node)

    graph.add_edge(START, "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic,
        ["run_one_analyst_revision", "rank", "no_evidence"],
    )
    graph.add_edge("run_one_analyst_revision", "revision_fanin")
    graph.add_edge("revision_fanin", "critic")
    graph.add_edge("rank", END)
    graph.add_edge("no_evidence", END)
    return graph.compile()
