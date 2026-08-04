"""Validation for the supplied, unmodified cafe_profile.json contract.

This must accept the exact supplied schema without requiring any cafe-specific
additions. See implementation_plan_final.md section 8.1.
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

_VALID_WEEKDAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


class Coordinates(BaseModel):
    lat: float
    lng: float

    @field_validator("lat")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError("lat out of range")
        return v

    @field_validator("lng")
    @classmethod
    def _lng_range(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError("lng out of range")
        return v


class RawCafeProfile(BaseModel):
    cafe_name: str
    city: str
    governorate: str
    region: str
    country: str
    coordinates: Coordinates
    timezone: str
    seats: int = Field(gt=0)
    opened: str
    opening_hours: dict[str, str]
    instagram: str | None = None
    tiktok: str | None = None
    currency: str
    weekend_days: list[str]
    notes: str | None = None
    local_search_areas: list[str] | None = None

    @field_validator("cafe_name", "city", "governorate", "region", "country", "currency")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"invalid IANA timezone: {v}") from e
        return v

    @field_validator("weekend_days")
    @classmethod
    def _valid_weekdays(cls, v: list[str]) -> list[str]:
        bad = [d for d in v if d not in _VALID_WEEKDAYS]
        if bad:
            raise ValueError(f"invalid weekday names: {bad}")
        return v

    @model_validator(mode="after")
    def _opening_hours_present(self) -> "RawCafeProfile":
        if "default" not in self.opening_hours:
            raise ValueError("opening_hours must define a 'default' regime")
        return self


def load_raw_profile(path: Path) -> RawCafeProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RawCafeProfile.model_validate(data)
