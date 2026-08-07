from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.reducers import (
    append_errors,
    merge_source_results,
    merge_token_usage,
    sum_float,
    sum_int,
    upsert_findings,
)
from src.schemas.artifacts import ArtifactRef
from src.schemas.content import ContentIdea
from src.schemas.context import ContextBundle
from src.schemas.findings import AnalystFinding, CriticOutput
from src.schemas.sources import DataQualitySummary, SourceResult


class Period(TypedDict):
    start: str
    end: str


class CafeIntelligenceState(TypedDict, total=False):
    run_id: str
    thread_id: str
    raw_profile: dict[str, Any]
    config: dict[str, Any]
    analysis_period: Period
    previous_period: Period
    trailing_baseline_periods: list[Period]
    recommendation_period: Period

    # Plain-JSON alternative to a pre-built `config` -- lets a caller that
    # can't construct a RuntimeCafeConfig object itself (e.g. a human typing
    # input into LangGraph Studio) start the graph with these instead; the
    # first node (setup_nodes.resolve_config) resolves them into the same
    # `config`/period fields above and is a no-op for every other caller,
    # which already builds `config` itself before invoking.
    profile_path: str
    data_dir: str
    app_settings_path: str
    source_registry_path: str
    target_week: str

    source_results: Annotated[list[SourceResult], merge_source_results]
    parsed_artifacts: dict[str, ArtifactRef]
    cleaned_artifacts: dict[str, ArtifactRef]
    data_quality: DataQualitySummary

    candidate_findings: Annotated[list[AnalystFinding], upsert_findings]
    critic_results: CriticOutput
    critic_round: int
    final_findings: list[AnalystFinding]

    context_bundle: ContextBundle
    content_ideas: list[ContentIdea]
    content_validation: dict[str, Any]
    content_repair_attempts: int
    report: dict[str, Any]

    human_decision: Literal["approve", "edit", "reject"] | None
    delivery_receipt: dict[str, Any] | None

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: Annotated[int, sum_int]
    cost_usd: Annotated[float, sum_float]
    token_usage: Annotated[dict[str, int], merge_token_usage]
    errors: Annotated[list[dict[str, Any]], append_errors]
    run_status: Literal["running", "succeeded", "partial", "failed", "rejected"]
