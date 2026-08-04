"""Resolves RuntimeCafeConfig from the raw profile + CLI args + app settings + registry.

The supplied cafe_profile.json is never edited to carry operational fields. Everything
this system needs beyond the raw profile is derived here, at runtime, so a second cafe
can be onboarded purely by supplying a different profile/data-dir/config -- zero
application-code changes (plan section 8.2 / ADR-002).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from src.config.app_settings import AppSettings, load_app_settings
from src.config.raw_profile import RawCafeProfile, load_raw_profile
from src.config.source_registry import SourceRegistry, load_source_registry
from src.periods import resolve_analysis_period, resolve_recommendation_period
from src.state import Period


def _profile_key(profile: RawCafeProfile) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", profile.cafe_name.lower()).strip("-")
    digest = hashlib.sha256(
        f"{profile.cafe_name}|{profile.city}|{profile.coordinates.lat}|{profile.coordinates.lng}".encode()
    ).hexdigest()[:8]
    return f"{slug}-{digest}"


@dataclass
class RuntimeCafeConfig:
    profile_key: str
    raw_profile: RawCafeProfile
    social_handles: dict[str, str | None]
    data_dir: Path
    source_registry: SourceRegistry
    analysis_period: Period
    previous_period: Period
    trailing_baseline_periods: list[Period]
    recommendation_period: Period
    local_search_terms: list[str]
    app_settings: AppSettings
    artifact_root: Path
    checkpoint_db: Path
    memory_db: Path
    prayer_calculation_method: str
    run_id: str = field(default="")


def _local_search_terms(profile: RawCafeProfile) -> list[str]:
    if profile.local_search_areas:
        return list(profile.local_search_areas)
    return [profile.city, profile.governorate, profile.region, profile.country]


def resolve_runtime_config(
    profile_path: Path,
    data_dir: Path,
    app_settings_path: Path,
    source_registry_path: Path,
    target_week: date | None = None,
    artifact_root: Path = Path("outputs/artifacts"),
    checkpoint_db: Path = Path("db/checkpoints.sqlite"),
    memory_db: Path = Path("db/memory.sqlite"),
    now: datetime | None = None,
) -> RuntimeCafeConfig:
    raw_profile = load_raw_profile(profile_path)
    app_settings = load_app_settings(app_settings_path)
    registry = load_source_registry(source_registry_path)

    analysis_period, previous_period, trailing = resolve_analysis_period(
        raw_profile.timezone, target_week, now
    )
    recommendation_period = resolve_recommendation_period(
        analysis_period, raw_profile.timezone
    )

    return RuntimeCafeConfig(
        profile_key=_profile_key(raw_profile),
        raw_profile=raw_profile,
        social_handles={"instagram": raw_profile.instagram, "tiktok": raw_profile.tiktok},
        data_dir=Path(data_dir),
        source_registry=registry,
        analysis_period=analysis_period,
        previous_period=previous_period,
        trailing_baseline_periods=trailing,
        recommendation_period=recommendation_period,
        local_search_terms=_local_search_terms(raw_profile),
        app_settings=app_settings,
        artifact_root=Path(artifact_root),
        checkpoint_db=Path(checkpoint_db),
        memory_db=Path(memory_db),
        prayer_calculation_method=app_settings.prayer_times.calculation_method,
    )
