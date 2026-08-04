"""Local-context search tool. Queries are built dynamically from the resolved
profile's location fields/coordinates -- never a hardcoded city. Degrades to
`unavailable` (not a crash) when TAVILY_API_KEY is absent or the API errors.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SearchHit:
    title: str
    source: str
    url: str | None
    published_date: str | None
    location: str | None
    snippet: str
    retrieved_at: str


def build_local_queries(
    local_search_terms: list[str], recommendation_start: str, recommendation_end: str
) -> list[str]:
    place = ", ".join(local_search_terms[:3])
    date_range = f"{recommendation_start[:10]} to {recommendation_end[:10]}"
    return [
        f"events near {place} during {date_range}",
        f"weather {local_search_terms[0]} {date_range}",
        f"seasonal events near {place} during {date_range}",
    ]


def run_local_search(
    queries: list[str], max_results: int = 5, timeout_seconds: int = 10
) -> tuple[list[SearchHit], str, list[str]]:
    """Returns (hits, status, warnings). status is 'success' | 'partial' | 'unavailable'."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return [], "unavailable", ["TAVILY_API_KEY not configured; local search skipped"]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
    except Exception as e:  # noqa: BLE001
        return [], "unavailable", [f"Tavily client init failed: {e}"]

    hits: list[SearchHit] = []
    warnings: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for q in queries:
        try:
            resp = client.search(query=q, max_results=max_results, timeout=timeout_seconds)
            for r in resp.get("results", [])[:max_results]:
                hits.append(SearchHit(
                    title=r.get("title", ""), source="tavily", url=r.get("url"),
                    published_date=r.get("published_date"), location=None,
                    snippet=(r.get("content") or "")[:500], retrieved_at=now,
                ))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"query failed: {q!r}: {e}")

    if not hits and warnings:
        return [], "unavailable", warnings
    status = "success" if not warnings else "partial"
    return hits, status, warnings
