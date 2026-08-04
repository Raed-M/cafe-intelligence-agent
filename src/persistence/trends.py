"""Cross-run trend statements computed strictly from `metric_history` /
`content_history` rows -- never from model recollection. This is what lets the
report honestly say things like "third consecutive week of decline" or "this
idea was rejected last time it was proposed" with a verifiable basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.persistence.memory_store import MemoryStore


@dataclass
class TrendStatement:
    metric_key: str
    statement_en: str
    statement_ar: str
    consecutive_periods: int
    direction: str  # "increase" | "decrease" | "flat"
    values: list[float]


def compute_metric_streak(
    store: MemoryStore, profile_key: str, metric_key: str, current_value: float, current_period_end: str,
    lookback: int = 8,
) -> TrendStatement | None:
    """Looks at the most recent `lookback` recorded values (strictly before the
    current period) for `metric_key` under this profile and reports the
    longest consecutive same-direction streak ending at the current value."""
    rows = store.conn.execute(
        """SELECT mh.value, mh.period_end FROM metric_history mh
           JOIN weekly_runs wr ON wr.run_id = mh.run_id
           WHERE wr.profile_key = ? AND mh.metric_key = ? AND mh.period_end < ? AND mh.value IS NOT NULL
           ORDER BY mh.period_end DESC LIMIT ?""",
        (profile_key, metric_key, current_period_end, lookback),
    ).fetchall()
    if not rows:
        return None

    history = [r[0] for r in reversed(rows)] + [current_value]
    if len(history) < 2:
        return None

    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    directions = ["increase" if d > 0 else "decrease" if d < 0 else "flat" for d in diffs]

    latest_direction = directions[-1]
    streak = 1
    for d in reversed(directions[:-1]):
        if d == latest_direction and latest_direction != "flat":
            streak += 1
        else:
            break

    if streak < 2 or latest_direction == "flat":
        return None

    verb_en = "risen" if latest_direction == "increase" else "fallen"
    verb_ar = "ارتفع" if latest_direction == "increase" else "انخفض"
    return TrendStatement(
        metric_key=metric_key,
        statement_en=f"{metric_key} has {verb_en} for {streak} consecutive comparable periods.",
        statement_ar=f"{verb_ar} {metric_key} لمدة {streak} فترات متتالية قابلة للمقارنة.",
        consecutive_periods=streak,
        direction=latest_direction,
        values=history[-(streak + 1):],
    )


def content_repetition_notes(store: MemoryStore, profile_key: str, hook_en: str, limit: int = 30) -> list[str]:
    """Flags if a near-identical hook was previously proposed and what happened to it."""
    rows = store.conn.execute(
        """SELECT ch.hook_en, ch.human_decision, ch.was_posted FROM content_history ch
           JOIN weekly_runs wr ON wr.run_id = ch.run_id
           WHERE wr.profile_key = ? ORDER BY ch.id DESC LIMIT ?""",
        (profile_key, limit),
    ).fetchall()
    notes = []
    normalized = hook_en.strip().lower()
    for prior_hook, decision, was_posted in rows:
        if prior_hook and prior_hook.strip().lower() == normalized:
            status = "posted" if was_posted else (decision or "no decision recorded")
            notes.append(f"An identical hook was previously proposed; outcome: {status}.")
    return notes


def all_metric_streaks(
    store: MemoryStore, profile_key: str, current_metrics: dict[str, float], current_period_end: str,
) -> list[TrendStatement]:
    statements = []
    for metric_key, value in current_metrics.items():
        stmt = compute_metric_streak(store, profile_key, metric_key, value, current_period_end)
        if stmt:
            statements.append(stmt)
    return statements
