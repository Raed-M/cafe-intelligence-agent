from __future__ import annotations

from typing import Literal, TypedDict


class ContentIdea(TypedDict):
    idea_id: str
    hook_ar: str
    hook_en: str
    format: Literal["reel", "carousel", "trend_audio"]
    product_sku: str
    product_name_ar: str
    product_name_en: str
    finding_id: str
    cited_metric_keys: list[str]
    local_context_ids: list[str]
    calendar_context_ids: list[str]
    posting_window_id: str
    timing_metric_keys: list[str]
    rationale_ar: str
    rationale_en: str
    post_date: str
    post_time_local: str
    timing_reason: str
    inventory_suitability: Literal["supported", "unknown", "not_applicable"]


class ContentIdeasOutput(TypedDict):
    """Wrapped in a top-level object because .with_structured_output()
    constrains one schema per call and the content agent must return exactly
    three ideas, not a bare top-level array."""
    items: list[ContentIdea]


class AlignmentCheckResult(TypedDict):
    aligned: bool
    explanation: str
