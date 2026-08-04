"""Business-date logic supporting cross-midnight operating hours (e.g. Ramadan 14:00-01:00).

An event's `business_date` is the calendar date the shift/session "belongs to" from the
cafe's operating perspective, not the naive calendar date of the timestamp. When the
configured closing time is earlier than the opening time, the interval crosses midnight,
and events between 00:00 and the closing boundary belong to the *previous* business date.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import NamedTuple

_HOURS_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")

POS_TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M"]


class OpeningInterval(NamedTuple):
    open_time: time
    close_time: time
    crosses_midnight: bool


def parse_opening_hours(spec: str) -> OpeningInterval:
    m = _HOURS_RE.match(spec.strip())
    if not m:
        raise ValueError(f"Invalid opening_hours spec: {spec!r}")
    oh, om, ch, cm = (int(x) for x in m.groups())
    open_t = time(oh, om)
    close_t = time(ch, cm)
    crosses = close_t <= open_t
    return OpeningInterval(open_t, close_t, crosses)


def business_date_for(local_dt: datetime, interval: OpeningInterval) -> date:
    """Given a timezone-aware local datetime and the applicable opening interval,
    return the business_date it belongs to."""
    t = local_dt.time()
    if not interval.crosses_midnight:
        return local_dt.date()
    # Crosses midnight: times from 00:00 up to (and including) close_time belong to
    # the previous business date; times from open_time onward belong to today.
    if t <= interval.close_time:
        return local_dt.date() - timedelta(days=1)
    return local_dt.date()


def resolve_interval_for_date(d: date, opening_hours: dict[str, str]) -> OpeningInterval:
    """Pick the applicable named opening-hours regime for a given calendar date.

    Regime selection is delegated to `src.context.calendar.resolve_regime_for_date`,
    which computes Ramadan dates deterministically from the Hijri calendar rather
    than hardcoding any date range. Falls back to 'default' when the profile does
    not declare the resolved regime (e.g. no 'ramadan' entry).
    """
    from src.context.calendar import resolve_regime_for_date

    regime = resolve_regime_for_date(d)
    if regime in opening_hours:
        return parse_opening_hours(opening_hours[regime])
    return parse_opening_hours(opening_hours["default"])


def parse_pos_timestamp(raw: str) -> datetime:
    """Parse the two documented POS timestamp formats into a naive local datetime."""
    raw = raw.strip()
    for fmt in POS_TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised POS timestamp format: {raw!r}")
