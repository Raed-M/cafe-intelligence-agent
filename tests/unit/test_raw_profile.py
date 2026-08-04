from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.raw_profile import load_raw_profile

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "qahwa_saihat"


def test_supplied_profile_loads_unchanged():
    profile = load_raw_profile(DATA_DIR / "cafe_profile.json")
    assert profile.cafe_name == "Qahwa Saihat"
    assert profile.instagram == "@qahwa.saihat"
    assert profile.opening_hours["ramadan"] == "14:00-01:00"
    assert profile.weekend_days == ["Friday", "Saturday"]


def test_invalid_timezone_rejected(tmp_path):
    import json

    data = json.loads((DATA_DIR / "cafe_profile.json").read_text())
    data["timezone"] = "Not/AZone"
    p = tmp_path / "bad_profile.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_raw_profile(p)
