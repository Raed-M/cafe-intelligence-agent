from __future__ import annotations

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe

REQUIRED_COLUMNS = [
    "date", "employee_id", "name", "role", "shift_start", "shift_end", "hours", "hourly_rate_sar",
]


def parse_staff(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    df = pd.read_csv(path, dtype=str)
    raw_row_count = len(df)
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"staff_shifts.csv missing required columns: {missing_cols}")

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    df["hourly_rate_sar"] = pd.to_numeric(df["hourly_rate_sar"], errors="coerce")

    bad = (
        df["date_parsed"] == "NaT"
    ) | df["employee_id"].isna() | df["hours"].isna() | df["hourly_rate_sar"].isna()
    rejected = df[bad]
    accepted = df[~bad].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} staff rows rejected for invalid date/employee/hours/rate")

    accepted = accepted.drop(columns=["date"]).rename(columns={"date_parsed": "date"})
    date_min = accepted["date"].min() if len(accepted) else None
    date_max = accepted["date"].max() if len(accepted) else None

    out_path = ctx.artifact_root / "parsed" / "staff.parquet"
    artifact = write_dataframe(accepted, out_path)

    return SourceResult(
        source_name="staff",
        status="success" if not len(rejected) else "partial",
        raw_row_count=raw_row_count,
        accepted_row_count=len(accepted),
        rejected_row_count=len(rejected),
        artifact=artifact,
        schema_version="1.0",
        date_min=date_min,
        date_max=date_max,
        warnings=warnings,
        error=None,
    )
