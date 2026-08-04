"""Deterministic Hijri/Gregorian calendar resolution using the `hijri-converter`
library (declared, not silently approximated). Used to resolve which named
opening-hours regime (e.g. "ramadan") applies to a given Gregorian date, and to
surface calendar context (Ramadan, Eid) for content grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from hijridate import Gregorian, Hijri


@dataclass
class HijriPeriod:
    name: str
    gregorian_start: date
    gregorian_end: date
    hijri_year: int


def _hijri_month_range(hijri_year: int, month: int, name: str) -> HijriPeriod:
    start = Hijri(hijri_year, month, 1).to_gregorian()
    # last day of the Hijri month: try day 30, fall back to 29 if invalid.
    try:
        end_h = Hijri(hijri_year, month, 30)
    except ValueError:
        end_h = Hijri(hijri_year, month, 29)
    end = end_h.to_gregorian()
    return HijriPeriod(name, date(start.year, start.month, start.day), date(end.year, end.month, end.day), hijri_year)


def ramadan_period(gregorian_year: int) -> HijriPeriod | None:
    """Return the Ramadan (Hijri month 9) period overlapping the given Gregorian year, if any."""
    for hy in _candidate_hijri_years(gregorian_year):
        period = _hijri_month_range(hy, 9, "ramadan")
        if period.gregorian_start.year == gregorian_year or period.gregorian_end.year == gregorian_year:
            return period
    return None


def eid_al_fitr_period(gregorian_year: int) -> HijriPeriod | None:
    """Eid al-Fitr: Hijri month 10, days 1-3 (Shawwal)."""
    for hy in _candidate_hijri_years(gregorian_year):
        start = Hijri(hy, 10, 1).to_gregorian()
        end = Hijri(hy, 10, 3).to_gregorian()
        s = date(start.year, start.month, start.day)
        e = date(end.year, end.month, end.day)
        if s.year == gregorian_year or e.year == gregorian_year:
            return HijriPeriod("eid_al_fitr", s, e, hy)
    return None


def eid_al_adha_period(gregorian_year: int) -> HijriPeriod | None:
    """Eid al-Adha: Hijri month 12, days 9-13 (Dhu al-Hijjah, incl. Arafah)."""
    for hy in _candidate_hijri_years(gregorian_year):
        start = Hijri(hy, 12, 9).to_gregorian()
        end = Hijri(hy, 12, 13).to_gregorian()
        s = date(start.year, start.month, start.day)
        e = date(end.year, end.month, end.day)
        if s.year == gregorian_year or e.year == gregorian_year:
            return HijriPeriod("eid_al_adha", s, e, hy)
    return None


def _candidate_hijri_years(gregorian_year: int) -> list[int]:
    approx = Gregorian(gregorian_year, 6, 15).to_hijri().year
    return [approx - 1, approx, approx + 1]


def resolve_regime_for_date(d: date) -> str:
    """Return 'ramadan' if `d` falls in Ramadan for its Gregorian year, else 'default'."""
    for year in (d.year - 1, d.year, d.year + 1):
        rp = ramadan_period(year)
        if rp and rp.gregorian_start <= d <= rp.gregorian_end:
            return "ramadan"
    return "default"
