"""Analysis / comparison / recommendation period resolution.

All periods are Monday 00:00 (inclusive) -> following Monday 00:00 (exclusive),
local to the cafe's timezone, per implementation_plan_final.md section 7.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.state import Period


def _monday_on_or_before(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_period(monday: date, tz: ZoneInfo) -> Period:
    start = datetime.combine(monday, datetime.min.time(), tzinfo=tz)
    end = start + timedelta(days=7)
    return Period(start=start.isoformat(), end=end.isoformat())


def resolve_analysis_period(
    tz_name: str,
    target_week: date | None,
    now: datetime | None = None,
) -> tuple[Period, Period, list[Period]]:
    """Return (analysis_period, previous_period, trailing_baseline_periods).

    Default: most recent *complete* Monday-Sunday week relative to `now`.
    A manually supplied `target_week` (any date in that week) is snapped to its Monday.
    """
    tz = ZoneInfo(tz_name)
    if now is None:
        now = datetime.now(tz)
    else:
        now = now.astimezone(tz)

    if target_week is not None:
        monday = _monday_on_or_before(target_week)
    else:
        this_monday = _monday_on_or_before(now.date())
        # "most recent complete week" ends at this_monday 00:00, so it started
        # the Monday before that.
        monday = this_monday - timedelta(days=7)

    analysis_period = week_period(monday, tz)
    previous_period = week_period(monday - timedelta(days=7), tz)
    trailing = [week_period(monday - timedelta(days=7 * i), tz) for i in range(1, 5)]
    return analysis_period, previous_period, trailing


def resolve_recommendation_period(analysis_period: Period, tz_name: str) -> Period:
    """Next 7 local calendar days after analysis_period end (half-open)."""
    tz = ZoneInfo(tz_name)
    end = datetime.fromisoformat(analysis_period["end"]).astimezone(tz)
    start = end
    stop = start + timedelta(days=7)
    return Period(start=start.isoformat(), end=stop.isoformat())


def is_launch_eligible(
    period: Period, launch_date: str | None, retire_date: str | None
) -> bool:
    """A product's activity period must not precede launch or follow retirement."""
    period_start = datetime.fromisoformat(period["start"])
    period_end = datetime.fromisoformat(period["end"])
    if launch_date:
        launch = datetime.fromisoformat(launch_date).replace(tzinfo=period_start.tzinfo)
        if period_end <= launch:
            return False
    if retire_date:
        retire = datetime.fromisoformat(retire_date).replace(tzinfo=period_start.tzinfo)
        if period_start >= retire:
            return False
    return True
