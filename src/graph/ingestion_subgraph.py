"""Registry-driven parallel ingestion subgraph.

`dispatch_ingestion` fans out one `Send` per configured source; the `parse_source`
node resolves the registered parser function by name and runs it safely so one
failing source cannot crash the others. `merge_source_results` (a state reducer)
upserts by source_name at fan-in.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.parsers.base import RunContext, run_parser_safely
from src.parsers.email_parser import parse_emails
from src.parsers.inventory_parser import parse_inventory
from src.parsers.menu_parser import parse_menu
from src.parsers.pos_parser import parse_pos
from src.parsers.reviews_parser import parse_reviews
from src.parsers.staff_parser import parse_staff
from src.parsers.traffic_parser import parse_traffic
from src.state import CafeIntelligenceState

PARSER_REGISTRY = {
    "parse_pos": parse_pos,
    "parse_menu": parse_menu,
    "parse_traffic": parse_traffic,
    "parse_staff": parse_staff,
    "parse_inventory": parse_inventory,
    "parse_emails": parse_emails,
    "parse_reviews": parse_reviews,
}


class _SourceJobState(CafeIntelligenceState, total=False):
    _source_name: str


def dispatch_ingestion(state: CafeIntelligenceState) -> list[Send]:
    config = state["config"]
    sends = []
    for source in config.source_registry.sources:
        sends.append(Send("parse_source", {**state, "_source_name": source.name}))
    return sends


def parse_source(state: _SourceJobState) -> dict[str, Any]:
    config = state["config"]
    source_name = state["_source_name"]
    source = config.source_registry.by_name(source_name)
    parser_fn = PARSER_REGISTRY.get(source.parser)
    ctx = RunContext(run_id=state["run_id"], config=config)

    if parser_fn is None:
        from src.parsers.base import failed_result

        result = failed_result(source.name, ValueError(f"Unregistered parser: {source.parser}"))
    else:
        result = run_parser_safely(parser_fn, source, ctx)

    return {"source_results": [result], "step_count": 1}


def ingestion_fanin(state: CafeIntelligenceState) -> dict[str, Any]:
    """No-op fan-in node; reducer already merged source_results."""
    return {}


def build_ingestion_subgraph():
    graph = StateGraph(CafeIntelligenceState)
    graph.add_node("parse_source", parse_source)
    graph.add_node("ingestion_fanin", ingestion_fanin)
    graph.add_conditional_edges(START, dispatch_ingestion, ["parse_source"])
    graph.add_edge("parse_source", "ingestion_fanin")
    graph.add_edge("ingestion_fanin", END)
    return graph.compile()
