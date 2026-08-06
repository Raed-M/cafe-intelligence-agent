from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate(obj):
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    if isinstance(obj, str):
        m = _ENV_VAR_RE.fullmatch(obj)
        if m:
            return os.environ.get(m.group(1), "")
        return obj
    return obj


class LimitsSettings(BaseModel):
    max_graph_steps: int
    max_cost_usd: float
    analyst_code_attempts: int
    critic_revision_rounds: int
    content_repair_attempts: int
    max_candidate_findings_per_analyst: int
    max_final_findings: int
    code_timeout_seconds: int


class ScheduleSettings(BaseModel):
    day_of_week: str
    hour: int
    minute: int


class ModelSettings(BaseModel):
    analyst: str
    critic: str
    content: str
    content_validator: str
    report_summary: str
    email_extractor: str
    temperature: float


class ReportSettings(BaseModel):
    whatsapp_max_chars: int
    pdf_optional: bool
    use_llm_summary_compression: bool = False


class PrayerTimesSettings(BaseModel):
    calculation_method: str


class AppSettings(BaseModel):
    schedule: ScheduleSettings
    limits: LimitsSettings
    models: ModelSettings
    report: ReportSettings
    prayer_times: PrayerTimesSettings


def load_app_settings(path: Path) -> AppSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = _interpolate(raw)
    return AppSettings.model_validate(raw)
