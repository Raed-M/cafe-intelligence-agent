from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.business_time import (
    business_date_for,
    parse_opening_hours,
    parse_pos_timestamp,
    resolve_interval_for_date,
)
from src.periods import (
    is_launch_eligible,
    resolve_analysis_period,
    resolve_recommendation_period,
)


def test_analysis_period_is_monday_to_monday_half_open():
    period, prev, trailing = resolve_analysis_period(
        "Asia/Riyadh", target_week=date(2026, 1, 7)
    )
    assert period["start"].startswith("2026-01-05")
    assert period["end"].startswith("2026-01-12")
    assert prev["start"].startswith("2025-12-29")
    assert len(trailing) == 4


def test_recommendation_period_follows_analysis_period():
    period, _, _ = resolve_analysis_period("Asia/Riyadh", target_week=date(2026, 1, 5))
    rec = resolve_recommendation_period(period, "Asia/Riyadh")
    assert rec["start"] == period["end"]
    assert rec["start"].startswith("2026-01-12")
    assert rec["end"].startswith("2026-01-19")


def test_default_run_uses_most_recent_complete_week():
    now = datetime(2026, 1, 12, 8, 0, tzinfo=ZoneInfo("Asia/Riyadh"))
    period, _, _ = resolve_analysis_period("Asia/Riyadh", target_week=None, now=now)
    assert period["start"].startswith("2026-01-05")
    assert period["end"].startswith("2026-01-12")


def test_cross_midnight_business_date():
    interval = parse_opening_hours("14:00-01:00")
    assert interval.crosses_midnight
    tz = ZoneInfo("Asia/Riyadh")
    before_close = datetime(2026, 3, 1, 0, 30, tzinfo=tz)
    assert business_date_for(before_close, interval) == date(2026, 2, 28)
    after_open = datetime(2026, 3, 1, 15, 0, tzinfo=tz)
    assert business_date_for(after_open, interval) == date(2026, 3, 1)
    after_close_before_open = datetime(2026, 3, 1, 10, 0, tzinfo=tz)
    assert business_date_for(after_close_before_open, interval) == date(2026, 3, 1)


def test_normal_hours_do_not_cross_midnight():
    interval = parse_opening_hours("07:00-23:00")
    assert not interval.crosses_midnight
    tz = ZoneInfo("Asia/Riyadh")
    dt = datetime(2026, 1, 5, 0, 30, tzinfo=tz)
    assert business_date_for(dt, interval) == date(2026, 1, 5)


def test_pos_timestamp_formats():
    assert parse_pos_timestamp("2026-01-05 07:10:00") == datetime(2026, 1, 5, 7, 10, 0)
    assert parse_pos_timestamp("05-Jan-2026 07:10") == datetime(2026, 1, 5, 7, 10)
    with pytest.raises(ValueError):
        parse_pos_timestamp("not-a-timestamp")


def test_launch_aware_eligibility():
    # Period entirely before launch: not eligible.
    period = {"start": "2026-03-23T00:00:00+03:00", "end": "2026-03-30T00:00:00+03:00"}
    assert not is_launch_eligible(period, "2026-04-06", None)
    # Period starting exactly on launch date: eligible.
    period2 = {"start": "2026-04-06T00:00:00+03:00", "end": "2026-04-13T00:00:00+03:00"}
    assert is_launch_eligible(period2, "2026-04-06", None)
