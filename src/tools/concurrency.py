"""Bounded-concurrency helper for independent per-item LLM calls.

Two places in this codebase process a list of items with one LLM call each in
a plain sequential loop -- email fact extraction (one call per email) and the
critic's semantic review (one call per candidate finding). Measured against a
live run, neither was actually bottlenecked by the LLM_RPM_LIMIT rate limiter
(achieved throughput was ~1.5 req/min against an 8 req/min cap) -- the real
cost was per-call network/generation latency (tens of seconds) multiplied by
list length with zero concurrency. llm_factory's rate limiter is thread-safe
(a lock-guarded sliding window) and enforces the RPM ceiling centrally no
matter how many workers submit through it concurrently, so running these
loops concurrently only removes needless serialization -- it cannot exceed
the configured RPM cap.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_CONCURRENCY = 5


def bounded_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int = DEFAULT_MAX_CONCURRENCY) -> list[R]:
    """Runs fn(item) concurrently across items, returning results in the same
    order as `items` (ThreadPoolExecutor.map preserves input order regardless
    of completion order). An exception raised by fn for one item propagates
    when its result is consumed, same as a plain for-loop would -- callers
    that need per-item error isolation should catch inside fn."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))
