from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.schemas.artifacts import ArtifactRef


class ErrorRecord(TypedDict):
    error_type: str
    message: str
    traceback: str | None


class SourceResult(TypedDict):
    source_name: str
    status: Literal["success", "partial", "failed"]
    raw_row_count: int
    accepted_row_count: int
    rejected_row_count: int
    artifact: ArtifactRef | None
    schema_version: str
    date_min: str | None
    date_max: str | None
    warnings: list[str]
    error: ErrorRecord | None


class SourceQuality(TypedDict):
    source_name: str
    rows_in: int
    rows_accepted: int
    rows_dropped: int
    rows_repaired: int
    rows_quarantined: int
    null_counts: dict[str, int]
    issue_counts: dict[str, int]
    excluded_periods: list[dict[str, str]]
    examples: list[dict[str, Any]]


class DataQualitySummary(TypedDict):
    source_summaries: list[SourceQuality]
    sources_successful: list[str]
    sources_partial: list[str]
    sources_failed: list[str]
    critical_dependencies_missing: list[str]
    total_rows_in: int
    total_rows_dropped: int
    total_rows_repaired: int
    warnings: list[str]
