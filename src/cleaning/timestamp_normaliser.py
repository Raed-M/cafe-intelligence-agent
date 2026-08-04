"""POS timestamp normalisation and business-date derivation."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from src.business_time import business_date_for, parse_pos_timestamp, resolve_interval_for_date


def normalise_pos_timestamps(
    df: pd.DataFrame, tz_name: str, opening_hours: dict[str, str]
) -> tuple[pd.DataFrame, int]:
    """Parse `timestamp_raw` into tz-aware `timestamp_local`, plus `calendar_date`,
    `hour_local`, `business_date` and `week_starting`. Rows with unparseable
    timestamps are dropped and counted (returned as the second tuple element)."""
    tz = ZoneInfo(tz_name)
    parsed: list[pd.Timestamp | None] = []
    for raw in df["timestamp_raw"]:
        try:
            naive = parse_pos_timestamp(str(raw))
            parsed.append(naive.replace(tzinfo=tz))
        except ValueError:
            parsed.append(None)

    df = df.copy()
    df["timestamp_local"] = parsed
    n_bad = df["timestamp_local"].isna().sum()
    df = df[df["timestamp_local"].notna()].copy()

    df["calendar_date"] = df["timestamp_local"].map(lambda t: t.date())
    df["hour_local"] = df["timestamp_local"].map(lambda t: t.hour)

    interval_cache: dict = {}
    for d in df["calendar_date"].unique():
        interval_cache[d] = resolve_interval_for_date(d, opening_hours)

    df["business_date"] = [
        business_date_for(ts, interval_cache[cd])
        for ts, cd in zip(df["timestamp_local"], df["calendar_date"])
    ]
    df["week_starting"] = df["business_date"].map(lambda d: d - pd.Timedelta(days=d.weekday()))

    df["timestamp_local"] = df["timestamp_local"].astype(str)
    df["calendar_date"] = df["calendar_date"].astype(str)
    df["business_date"] = df["business_date"].astype(str)
    df["week_starting"] = df["week_starting"].astype(str)

    return df, int(n_bad)
