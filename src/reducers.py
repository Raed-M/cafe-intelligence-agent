"""Custom LangGraph state reducers.

`operator.add` is insufficient here: source ingestion and analyst findings must be
upserted by key so that a revision replaces (not duplicates) the prior version.
"""
from __future__ import annotations

from typing import Any

from src.schemas.findings import AnalystFinding
from src.schemas.sources import SourceResult


def merge_source_results(
    left: list[SourceResult] | None, right: list[SourceResult] | None
) -> list[SourceResult]:
    """Upsert by source_name; the most recently written result for a source wins."""
    left = left or []
    right = right or []
    by_name: dict[str, SourceResult] = {r["source_name"]: r for r in left}
    for r in right:
        by_name[r["source_name"]] = r
    return list(by_name.values())


def upsert_findings(
    left: list[AnalystFinding] | None, right: list[AnalystFinding] | None
) -> list[AnalystFinding]:
    """Upsert by finding_id; higher revision_count replaces the older version."""
    left = left or []
    right = right or []
    by_id: dict[str, AnalystFinding] = {f["finding_id"]: f for f in left}
    for f in right:
        existing = by_id.get(f["finding_id"])
        if existing is None or f["revision_count"] >= existing["revision_count"]:
            by_id[f["finding_id"]] = f
    return list(by_id.values())


def merge_token_usage(
    left: dict[str, int] | None, right: dict[str, int] | None
) -> dict[str, int]:
    """Sum token counts by key (e.g. '<node>.prompt_tokens')."""
    left = dict(left or {})
    for k, v in (right or {}).items():
        left[k] = left.get(k, 0) + v
    return left


def sum_int(left: int | None, right: int | None) -> int:
    return (left or 0) + (right or 0)


def sum_float(left: float | None, right: float | None) -> float:
    return (left or 0.0) + (right or 0.0)


def append_errors(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    return (left or []) + (right or [])
