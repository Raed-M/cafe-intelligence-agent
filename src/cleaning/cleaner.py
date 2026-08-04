"""Cleaning + QA node (Module 3). Consumes parsed artifacts written by ingestion,
applies the contractual cleaning rules per source, writes canonical *cleaned*
artifacts, and produces the DataQualitySummary the report is required to show.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.cleaning.item_repair import repair_item_names
from src.cleaning.pos_dedup import dedup_double_swipes
from src.cleaning.quality_report import build_data_quality_summary
from src.cleaning.sensor_quality import flag_dead_sensor_days
from src.cleaning.timestamp_normaliser import normalise_pos_timestamps
from src.config.runtime_config import RuntimeCafeConfig
from src.parsers.base import RunContext
from src.schemas.artifacts import ArtifactRef
from src.schemas.sources import SourceQuality
from src.state import CafeIntelligenceState
from src.tools.artifact_io import artifact_dir, read_dataframe, write_dataframe

LINE_TOTAL_TOLERANCE_SAR = 0.02


def _by_name(source_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["source_name"]: r for r in source_results}


def _clean_pos(
    results: dict[str, dict[str, Any]], config: RuntimeCafeConfig, out_dir
) -> tuple[ArtifactRef | None, SourceQuality]:
    pos_res = results.get("pos")
    menu_res = results.get("menu")
    if not pos_res or pos_res["status"] == "failed" or not pos_res["artifact"]:
        return None, SourceQuality(
            source_name="pos", rows_in=0, rows_accepted=0, rows_dropped=0, rows_repaired=0,
            rows_quarantined=0, null_counts={}, issue_counts={}, excluded_periods=[], examples=[],
        )

    df = read_dataframe(pos_res["artifact"])
    rows_in = len(df)

    df, n_bad_ts = normalise_pos_timestamps(
        df, config.raw_profile.timezone, config.raw_profile.opening_hours
    )

    dedup_df, dedup_audit = dedup_double_swipes(df)

    if menu_res and menu_res["status"] != "failed" and menu_res["artifact"]:
        menu_df = read_dataframe(menu_res["artifact"])
        repaired_df, repair_audit = repair_item_names(dedup_df, menu_df)
    else:
        repaired_df = dedup_df.copy()
        repaired_df["known_sku"] = False
        repaired_df["item_name_en"] = None
        repaired_df["item_name_ar"] = None
        repaired_df["category"] = None
        repair_audit = {"unknown_sku_rows": len(repaired_df), "unknown_skus": [], "rows_with_name_repaired_from_menu": 0}

    repaired_df["is_refund"] = repaired_df["quantity"] < 0

    expected_total = (
        repaired_df["quantity"] * repaired_df["unit_price_sar"] - repaired_df["discount_sar"]
    )
    repaired_df["line_total_inconsistent"] = (
        (repaired_df["line_total_sar"] - expected_total).abs() > LINE_TOTAL_TOLERANCE_SAR
    )
    n_inconsistent = int(repaired_df["line_total_inconsistent"].sum())

    n_missing_cashier = int(repaired_df["cashier_id"].isna().sum())
    n_missing_item_name = int(repaired_df["item_name_raw"].isna().sum())

    artifact = write_dataframe(repaired_df, out_dir / "pos.parquet")

    null_counts = {
        "cashier_id": n_missing_cashier,
        "item_name_raw": n_missing_item_name,
    }
    issue_counts = {
        "unparseable_timestamp": int(n_bad_ts),
        "double_swipe_transactions": dedup_audit["duplicated_transactions"],
        "double_swipe_rows_removed": dedup_audit["rows_removed"],
        "unknown_sku_rows": repair_audit["unknown_sku_rows"],
        "line_total_inconsistent": n_inconsistent,
        "refund_rows": int(repaired_df["is_refund"].sum()),
    }
    quality = SourceQuality(
        source_name="pos",
        rows_in=rows_in,
        rows_accepted=len(repaired_df),
        rows_dropped=int(n_bad_ts) + dedup_audit["rows_removed"],
        rows_repaired=repair_audit["rows_with_name_repaired_from_menu"],
        rows_quarantined=repair_audit["unknown_sku_rows"],
        null_counts=null_counts,
        issue_counts=issue_counts,
        excluded_periods=[],
        examples=dedup_audit["examples"],
    )
    return artifact, quality


def _clean_traffic(results: dict[str, dict[str, Any]], out_dir) -> tuple[ArtifactRef | None, SourceQuality]:
    traffic_res = results.get("traffic")
    if not traffic_res or traffic_res["status"] == "failed" or not traffic_res["artifact"]:
        return None, SourceQuality(
            source_name="traffic", rows_in=0, rows_accepted=0, rows_dropped=0, rows_repaired=0,
            rows_quarantined=0, null_counts={}, issue_counts={}, excluded_periods=[], examples=[],
        )
    df = read_dataframe(traffic_res["artifact"])
    rows_in = len(df)
    df, audit = flag_dead_sensor_days(df)
    artifact = write_dataframe(df, out_dir / "traffic.parquet")
    quality = SourceQuality(
        source_name="traffic",
        rows_in=rows_in,
        rows_accepted=len(df),
        rows_dropped=0,
        rows_repaired=0,
        rows_quarantined=audit["rows_excluded_from_denominators"],
        null_counts={},
        issue_counts={"dead_sensor_days": len(audit["dead_sensor_days"])},
        excluded_periods=[{"date": d} for d in audit["dead_sensor_days"]],
        examples=[],
    )
    return artifact, quality


def _clean_staff(results: dict[str, dict[str, Any]], out_dir) -> tuple[ArtifactRef | None, SourceQuality]:
    staff_res = results.get("staff")
    if not staff_res or staff_res["status"] == "failed" or not staff_res["artifact"]:
        return None, SourceQuality(
            source_name="staff", rows_in=0, rows_accepted=0, rows_dropped=0, rows_repaired=0,
            rows_quarantined=0, null_counts={}, issue_counts={}, examples=[], excluded_periods=[],
        )
    df = read_dataframe(staff_res["artifact"])
    rows_in = len(df)

    def _duration_hours(row) -> float | None:
        try:
            start = pd.to_datetime(row["shift_start"], format="%H:%M", errors="coerce")
            end = pd.to_datetime(row["shift_end"], format="%H:%M", errors="coerce")
        except Exception:  # noqa: BLE001
            return None
        if pd.isna(start) or pd.isna(end):
            return None
        delta = (end - start).total_seconds() / 3600.0
        if delta < 0:
            delta += 24  # cross-midnight shift
        return delta

    df["computed_duration_hours"] = df.apply(_duration_hours, axis=1)
    df["hours_mismatch"] = (
        df["computed_duration_hours"].notna()
        & ((df["computed_duration_hours"] - df["hours"]).abs() > 0.25)
    )
    n_mismatch = int(df["hours_mismatch"].sum())
    df["labour_cost_sar"] = df["hours"] * df["hourly_rate_sar"]

    artifact = write_dataframe(df, out_dir / "staff.parquet")
    quality = SourceQuality(
        source_name="staff", rows_in=rows_in, rows_accepted=len(df), rows_dropped=0,
        rows_repaired=0, rows_quarantined=0, null_counts={},
        issue_counts={"shift_duration_mismatch": n_mismatch}, excluded_periods=[], examples=[],
    )
    return artifact, quality


def _clean_inventory(results: dict[str, dict[str, Any]], out_dir) -> tuple[ArtifactRef | None, SourceQuality]:
    inv_res = results.get("inventory")
    if not inv_res or inv_res["status"] == "failed" or not inv_res["artifact"]:
        return None, SourceQuality(
            source_name="inventory", rows_in=0, rows_accepted=0, rows_dropped=0, rows_repaired=0,
            rows_quarantined=0, null_counts={}, issue_counts={}, examples=[], excluded_periods=[],
        )
    df = read_dataframe(inv_res["artifact"])
    rows_in = len(df)
    n_unknown_waste = int(df["units_wasted"].isna().sum())

    has_all = df["units_wasted"].notna()
    df["estimated_remaining_units"] = np.where(
        has_all, df["units_ordered"] - df["units_sold"] - df["units_wasted"], np.nan
    )
    df["known_waste_cost_sar"] = np.where(
        has_all, df["units_wasted"] * df["unit_cost_sar"], np.nan
    )

    artifact = write_dataframe(df, out_dir / "inventory.parquet")
    quality = SourceQuality(
        source_name="inventory", rows_in=rows_in, rows_accepted=len(df), rows_dropped=0,
        rows_repaired=0, rows_quarantined=0,
        null_counts={"units_wasted": n_unknown_waste},
        issue_counts={"unknown_waste_weeks": n_unknown_waste}, excluded_periods=[], examples=[],
    )
    return artifact, quality


def _passthrough(
    source_name: str, results: dict[str, dict[str, Any]], out_dir
) -> tuple[ArtifactRef | None, SourceQuality]:
    res = results.get(source_name)
    if not res or res["status"] == "failed" or not res["artifact"]:
        return None, SourceQuality(
            source_name=source_name, rows_in=0, rows_accepted=0, rows_dropped=0, rows_repaired=0,
            rows_quarantined=0, null_counts={}, issue_counts={}, examples=[], excluded_periods=[],
        )
    df = read_dataframe(res["artifact"])
    artifact = write_dataframe(df, out_dir / f"{source_name}.parquet")
    quality = SourceQuality(
        source_name=source_name, rows_in=len(df), rows_accepted=len(df), rows_dropped=0,
        rows_repaired=0, rows_quarantined=0, null_counts={}, issue_counts={}, examples=[], excluded_periods=[],
    )
    return artifact, quality


def clean_and_materialise(state: CafeIntelligenceState) -> dict[str, Any]:
    config: RuntimeCafeConfig = state["config"]
    run_id = state["run_id"]
    results = _by_name(state.get("source_results", []))
    out_dir = artifact_dir(config.artifact_root, run_id, "cleaned")

    cleaned_artifacts: dict[str, ArtifactRef] = {}
    qualities: list[SourceQuality] = []

    pos_artifact, pos_q = _clean_pos(results, config, out_dir)
    if pos_artifact:
        cleaned_artifacts["pos"] = pos_artifact
    qualities.append(pos_q)

    traffic_artifact, traffic_q = _clean_traffic(results, out_dir)
    if traffic_artifact:
        cleaned_artifacts["traffic"] = traffic_artifact
    qualities.append(traffic_q)

    staff_artifact, staff_q = _clean_staff(results, out_dir)
    if staff_artifact:
        cleaned_artifacts["staff"] = staff_artifact
    qualities.append(staff_q)

    inv_artifact, inv_q = _clean_inventory(results, out_dir)
    if inv_artifact:
        cleaned_artifacts["inventory"] = inv_artifact
    qualities.append(inv_q)

    for name in ("menu", "emails", "reviews"):
        artifact, q = _passthrough(name, results, out_dir)
        if artifact:
            cleaned_artifacts[name] = artifact
        qualities.append(q)

    successful = [r["source_name"] for r in results.values() if r["status"] == "success"]
    partial = [r["source_name"] for r in results.values() if r["status"] == "partial"]
    failed = [r["source_name"] for r in results.values() if r["status"] == "failed"]

    critical_missing = [name for name in failed if name in ("pos", "menu")]
    warnings = [w for r in results.values() for w in r.get("warnings", [])]

    quality_summary = build_data_quality_summary(
        qualities, successful, partial, failed, critical_missing, warnings
    )

    return {
        "cleaned_artifacts": cleaned_artifacts,
        "data_quality": quality_summary,
        "step_count": 1,
    }
