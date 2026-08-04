from __future__ import annotations

from typing import Literal, TypedDict


class ContextEvidence(TypedDict):
    context_id: str
    kind: Literal["event", "weather", "calendar", "prayer", "profile", "email_event"]
    title: str
    date_start: str | None
    date_end: str | None
    location: str | None
    source: str
    source_url_or_artifact: str | None
    retrieved_at: str
    summary: str


class PostingWindow(TypedDict):
    window_id: str
    post_date: str
    start_time_local: str
    end_time_local: str
    busy_metric_keys: list[str]
    demand_score: float
    prayer_relation: str | None
    event_context_ids: list[str]
    rationale: str


class ContextBundle(TypedDict):
    recommendation_period_start: str
    recommendation_period_end: str
    search_queries: list[str]
    evidence: list[ContextEvidence]
    posting_windows: list[PostingWindow]
    search_status: Literal["success", "partial", "unavailable"]
    warnings: list[str]
