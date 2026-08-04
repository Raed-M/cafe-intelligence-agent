from datetime import date

import pytest

from src.context.prayer_times import compute_prayer_times


def test_prayer_times_ordered_correctly_summer():
    pt = compute_prayer_times(date(2026, 7, 15), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    times = [pt.fajr, pt.sunrise, pt.dhuhr, pt.asr, pt.maghrib, pt.isha]
    assert times == sorted(times)


def test_prayer_times_ordered_correctly_winter():
    pt = compute_prayer_times(date(2026, 1, 15), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    times = [pt.fajr, pt.sunrise, pt.dhuhr, pt.asr, pt.maghrib, pt.isha]
    assert times == sorted(times)


def test_dhuhr_near_local_solar_noon():
    pt = compute_prayer_times(date(2026, 3, 20), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    h, m = map(int, pt.dhuhr.split(":"))
    assert 11 <= h <= 12


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_prayer_times(date(2026, 1, 1), 26.465, 50.04, "Asia/Riyadh", "made_up_method")


def test_deterministic_repeatable():
    a = compute_prayer_times(date(2026, 5, 1), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    b = compute_prayer_times(date(2026, 5, 1), 26.465, 50.04, "Asia/Riyadh", "umm_al_qura")
    assert a == b
