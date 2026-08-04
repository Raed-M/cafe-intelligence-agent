"""Deterministic prayer-time calculation from date, coordinates, timezone and a
declared calculation method -- the standard solar-position algorithm used by
praytimes.org (public astronomical formulas: Julian date -> solar declination
and equation of time -> hour-angle based prayer times). No network call, no
hardcoded times; the same date/location/method always yields the same output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# (fajr_angle, isha_angle_or_minutes) per declared method. Isha given as a
# negative-degrees convention like Fajr, except Umm al-Qura which fixes a
# 90-minute interval after Maghrib (120 during Ramadan, simplified here to 90
# year-round since Ramadan-specific handling is out of scope for this bonus).
METHOD_PARAMS = {
    "umm_al_qura": {"fajr_angle": 18.5, "isha_minutes_after_maghrib": 90},
    "mwl": {"fajr_angle": 18.0, "isha_angle": 17.0},
    "isna": {"fajr_angle": 15.0, "isha_angle": 15.0},
    "egypt": {"fajr_angle": 19.5, "isha_angle": 17.5},
}

MAGHRIB_ANGLE = 0.833  # standard sunset/sunrise depression angle (refraction + solar radius)


@dataclass
class PrayerTimes:
    date: str
    fajr: str
    sunrise: str
    dhuhr: str
    asr: str
    maghrib: str
    isha: str
    method: str
    timezone: str


def _julian_date(d: date) -> float:
    y, m, day = d.year, d.month, d.day
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5


def _sun_position(jd: float) -> tuple[float, float]:
    """Returns (declination_deg, equation_of_time_hours) for the given Julian date (noon)."""
    d = jd - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    l = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)

    dec = math.asin(math.sin(e) * math.sin(l))
    ra = math.degrees(math.atan2(math.cos(e) * math.sin(l), math.cos(l))) / 15.0
    ra = ra % 24
    q_hours = q / 15.0
    eq_time = q_hours - ra
    if eq_time > 12:
        eq_time -= 24
    if eq_time < -12:
        eq_time += 24
    return math.degrees(dec), eq_time


def _hour_angle(lat_deg: float, dec_deg: float, depression_deg: float) -> float:
    """Hour angle (in hours from local solar noon) at which the sun is
    `depression_deg` degrees *below* the horizon (used for fajr/sunrise/maghrib/isha)."""
    return _hour_angle_from_altitude(lat_deg, dec_deg, -depression_deg)


def _hour_angle_from_altitude(lat_deg: float, dec_deg: float, altitude_deg: float) -> float:
    """Hour angle (in hours from local solar noon) at which the sun reaches the
    given altitude above the horizon (positive = above, negative = below)."""
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    cos_h = (math.sin(math.radians(altitude_deg)) - math.sin(lat) * math.sin(dec)) / (math.cos(lat) * math.cos(dec))
    cos_h = max(-1.0, min(1.0, cos_h))
    return math.degrees(math.acos(cos_h)) / 15.0


def compute_prayer_times(
    d: date, lat: float, lng: float, tz_name: str, method: str = "umm_al_qura"
) -> PrayerTimes:
    if method not in METHOD_PARAMS:
        raise ValueError(f"Unknown prayer calculation method: {method}")
    params = METHOD_PARAMS[method]
    tz = ZoneInfo(tz_name)
    utc_offset = datetime(d.year, d.month, d.day, tzinfo=tz).utcoffset().total_seconds() / 3600.0

    jd = _julian_date(d) - lng / 360.0
    dec, eq_time = _sun_position(jd)

    dhuhr_utc = 12.0 - lng / 15.0 - eq_time
    dhuhr_local = dhuhr_utc + utc_offset

    fajr_ha = _hour_angle(lat, dec, params["fajr_angle"])
    sunrise_ha = _hour_angle(lat, dec, MAGHRIB_ANGLE)
    maghrib_ha = sunrise_ha  # symmetric around dhuhr

    lat_r = math.radians(lat)
    dec_r = math.radians(dec)
    asr_altitude = math.degrees(math.atan(1.0 / (1.0 + math.tan(abs(lat_r - dec_r)))))
    asr_ha = _hour_angle_from_altitude(lat, dec, asr_altitude)

    fajr = dhuhr_local - fajr_ha
    sunrise = dhuhr_local - sunrise_ha
    asr = dhuhr_local + asr_ha
    maghrib = dhuhr_local + maghrib_ha

    if "isha_angle" in params:
        isha_ha = _hour_angle(lat, dec, params["isha_angle"])
        isha = dhuhr_local + isha_ha
    else:
        isha = maghrib + params["isha_minutes_after_maghrib"] / 60.0

    def _fmt(hours: float) -> str:
        hours = hours % 24
        h = int(hours)
        m = int(round((hours - h) * 60))
        if m == 60:
            m = 0
            h = (h + 1) % 24
        return f"{h:02d}:{m:02d}"

    return PrayerTimes(
        date=d.isoformat(), fajr=_fmt(fajr), sunrise=_fmt(sunrise), dhuhr=_fmt(dhuhr_local),
        asr=_fmt(asr), maghrib=_fmt(maghrib), isha=_fmt(isha), method=method, timezone=tz_name,
    )
