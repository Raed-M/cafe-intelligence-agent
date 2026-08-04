"""Rule-based dead-sensor detection for foot_traffic.csv.

A day is flagged dead when every recorded hourly door_count for that date is
zero while the immediately adjacent recorded days are non-zero -- this is a
rule, not a hardcoded date list, so it rediscovers whichever days are actually
dead in the supplied data (documented as 2026-06-08..10 for Qahwa, but must not
be hardcoded as such).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def flag_dead_sensor_days(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    daily = df.groupby("date")["door_count"].agg(["sum", "count"]).sort_index()
    dead_days: list[str] = []
    dates = list(daily.index)
    is_zero = [daily.loc[d, "sum"] == 0 for d in dates]

    # Flag maximal runs of consecutive zero-days that are bounded on at least
    # one side by a recorded non-zero day, so a multi-day sensor outage (not
    # just single isolated zero days) is fully rediscovered.
    i = 0
    while i < len(dates):
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < len(dates) and is_zero[j]:
            j += 1
        prev_nonzero = i > 0 and not is_zero[i - 1]
        next_nonzero = j < len(dates) and not is_zero[j]
        if prev_nonzero or next_nonzero:
            dead_days.extend(dates[i:j])
        i = j

    df["is_dead_sensor_day"] = df["date"].isin(dead_days)
    audit = {
        "dead_sensor_days": sorted(dead_days),
        "rows_excluded_from_denominators": int(df["is_dead_sensor_day"].sum()),
    }
    return df, audit
