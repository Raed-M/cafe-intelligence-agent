"""Assembles the ContextBundle: calendar (Hijri-derived), prayer times, local
search (Tavily, degrading gracefully), supplier-email event facts, and
deterministic posting windows built from observed busy-period metrics. All of
this is scoped to `recommendation_period`, never the historical analysis
period (plan section 7.3 / ADR-007).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.config.runtime_config import RuntimeCafeConfig
from src.context.calendar import eid_al_adha_period, eid_al_fitr_period, ramadan_period
from src.context.prayer_times import compute_prayer_times
from src.schemas.artifacts import ArtifactRef
from src.schemas.context import ContextBundle, ContextEvidence, PostingWindow
from src.tools.artifact_io import read_dataframe
from src.tools.tavily_search import build_local_queries, run_local_search


def _daterange(start: datetime, end: datetime):
    d = start.date()
    while d < end.date():
        yield d
        d += timedelta(days=1)


def _calendar_evidence(rec_start: datetime, rec_end: datetime) -> list[ContextEvidence]:
    evidence = []
    now = datetime.now(timezone.utc).isoformat()
    for year in {rec_start.year, rec_end.year}:
        for period_fn, name in ((ramadan_period, "Ramadan"), (eid_al_fitr_period, "Eid al-Fitr"), (eid_al_adha_period, "Eid al-Adha")):
            period = period_fn(year)
            if not period:
                continue
            if period.gregorian_end < rec_start.date() or period.gregorian_start > rec_end.date():
                continue
            evidence.append(ContextEvidence(
                context_id=f"cal-{name.lower().replace(' ', '-')}-{period.hijri_year}",
                kind="calendar", title=name,
                date_start=period.gregorian_start.isoformat(), date_end=period.gregorian_end.isoformat(),
                location=None, source="hijridate", source_url_or_artifact=None,
                retrieved_at=now, summary=f"{name} ({period.hijri_year}H) overlaps the recommendation period.",
            ))
    return evidence


def _prayer_evidence(rec_start: datetime, rec_end: datetime, config: RuntimeCafeConfig) -> list[ContextEvidence]:
    evidence = []
    now = datetime.now(timezone.utc).isoformat()
    coords = config.raw_profile.coordinates
    for d in _daterange(rec_start, rec_end):
        pt = compute_prayer_times(d, coords.lat, coords.lng, config.raw_profile.timezone, config.prayer_calculation_method)
        evidence.append(ContextEvidence(
            context_id=f"prayer-{d.isoformat()}", kind="prayer", title=f"Prayer times {d.isoformat()}",
            date_start=d.isoformat(), date_end=d.isoformat(), location=config.raw_profile.city,
            source=f"deterministic:{config.prayer_calculation_method}", source_url_or_artifact=None,
            retrieved_at=now,
            summary=f"Fajr {pt.fajr}, Dhuhr {pt.dhuhr}, Asr {pt.asr}, Maghrib {pt.maghrib}, Isha {pt.isha}",
        ))
    return evidence


def _email_event_evidence(cleaned_artifacts: dict[str, ArtifactRef], rec_start: datetime, rec_end: datetime) -> list[ContextEvidence]:
    if "emails" not in cleaned_artifacts:
        return []
    df = read_dataframe(cleaned_artifacts["emails"])
    events = df[df.get("category") == "event"] if "category" in df.columns else df.iloc[0:0]
    evidence = []
    now = datetime.now(timezone.utc).isoformat()
    for _, row in events.iterrows():
        evidence.append(ContextEvidence(
            context_id=f"email-event-{uuid.uuid4().hex[:8]}", kind="email_event",
            title=str(row.get("subject") or "Supplier/local event email"),
            date_start=row.get("event_start"), date_end=row.get("event_end"),
            location=row.get("location"), source=str(row.get("email_file", "")),
            source_url_or_artifact=str(row.get("email_file", "")), retrieved_at=now,
            summary=str(row.get("evidence_text", ""))[:280],
        ))
    return evidence


def _profile_fallback_evidence(config: RuntimeCafeConfig) -> ContextEvidence:
    now = datetime.now(timezone.utc).isoformat()
    return ContextEvidence(
        context_id="profile-locality", kind="profile", title="Cafe locality (profile-derived)",
        date_start=None, date_end=None, location=", ".join(config.local_search_terms),
        source="cafe_profile.json", source_url_or_artifact=None, retrieved_at=now,
        summary=f"Neighbourhood context for {config.raw_profile.cafe_name} in {config.local_search_terms[0]}, "
                f"{config.local_search_terms[1]}. {config.raw_profile.notes or ''}".strip(),
    )


def _posting_windows(
    cleaned_artifacts: dict[str, ArtifactRef], rec_start: datetime, rec_end: datetime, config: RuntimeCafeConfig
) -> list[PostingWindow]:
    windows: list[PostingWindow] = []
    opening = config.raw_profile.opening_hours.get("default", "07:00-23:00")
    open_h = int(opening.split("-")[0].split(":")[0])
    close_h = int(opening.split("-")[1].split(":")[0]) or 23

    hourly_avg: dict[tuple[int, int], float] = {}
    if "pos" in cleaned_artifacts:
        df = read_dataframe(cleaned_artifacts["pos"])
        if len(df) and "business_date" in df.columns:
            df = df.copy()
            df["_dow"] = pd.to_datetime(df["business_date"]).dt.dayofweek
            valid = df[df["quantity"] > 0]
            grouped = valid.groupby(["_dow", "hour_local"])["transaction_id"].nunique()
            hourly_avg = grouped.to_dict()

    for d in _daterange(rec_start, rec_end):
        dow = d.weekday()
        candidates = [(h, hourly_avg.get((dow, h), 0)) for h in range(open_h, close_h)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[0] if candidates else (open_h, 0)
        hour, score = top
        has_history = bool(hourly_avg)
        windows.append(PostingWindow(
            window_id=f"pw-{d.isoformat()}", post_date=d.isoformat(),
            start_time_local=f"{hour:02d}:00", end_time_local=f"{(hour + 1) % 24:02d}:00",
            busy_metric_keys=["hourly_valid_transaction_count"] if has_history else [],
            demand_score=float(score),
            prayer_relation=None,
            event_context_ids=[],
            rationale=(
                f"Historically busiest hour ({hour:02d}:00) on this weekday based on unique valid "
                f"transactions observed across the available POS history." if has_history else
                f"No sufficient transaction history; defaulted to a mid-hours opening-hour slot ({hour:02d}:00)."
            ),
        ))
    return windows


def build_context_bundle(
    config: RuntimeCafeConfig,
    recommendation_period: dict[str, str],
    cleaned_artifacts: dict[str, ArtifactRef],
) -> ContextBundle:
    rec_start = datetime.fromisoformat(recommendation_period["start"])
    rec_end = datetime.fromisoformat(recommendation_period["end"])

    queries = build_local_queries(config.local_search_terms, recommendation_period["start"], recommendation_period["end"])
    hits, search_status, search_warnings = run_local_search(queries)

    now = datetime.now(timezone.utc).isoformat()
    evidence: list[ContextEvidence] = []
    evidence.extend(_calendar_evidence(rec_start, rec_end))
    evidence.extend(_prayer_evidence(rec_start, rec_end, config))
    evidence.extend(_email_event_evidence(cleaned_artifacts, rec_start, rec_end))

    for hit in hits:
        evidence.append(ContextEvidence(
            context_id=f"tavily-{uuid.uuid4().hex[:8]}", kind="event", title=hit.title,
            date_start=hit.published_date, date_end=None, location=None, source="tavily",
            source_url_or_artifact=hit.url, retrieved_at=hit.retrieved_at, summary=hit.snippet,
        ))

    warnings = list(search_warnings)
    if search_status != "success":
        evidence.append(_profile_fallback_evidence(config))
        warnings.append("Local search degraded; profile-derived locality context used instead.")

    posting_windows = _posting_windows(cleaned_artifacts, rec_start, rec_end, config)

    return ContextBundle(
        recommendation_period_start=recommendation_period["start"],
        recommendation_period_end=recommendation_period["end"],
        search_queries=queries,
        evidence=evidence,
        posting_windows=posting_windows,
        search_status=search_status,
        warnings=warnings,
    )
