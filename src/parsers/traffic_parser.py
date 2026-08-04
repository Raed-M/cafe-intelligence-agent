from __future__ import annotations

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe

REQUIRED_COLUMNS = ["date", "hour", "door_count"]


def parse_traffic(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    df = pd.read_csv(path, dtype=str)
    raw_row_count = len(df)
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"foot_traffic.csv missing required columns: {missing_cols}")

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["door_count"] = pd.to_numeric(df["door_count"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    bad = (
        df["hour"].isna() | (df["hour"] < 0) | (df["hour"] > 23)
        | df["door_count"].isna() | (df["door_count"] < 0)
        | (df["date_parsed"] == "NaT")
    )
    rejected = df[bad]
    accepted = df[~bad].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} traffic rows rejected for invalid hour/door_count/date")

    dup_key = accepted.duplicated(subset=["date_parsed", "hour"], keep="first")
    if dup_key.any():
        warnings.append(f"{dup_key.sum()} duplicate (date, hour) rows dropped")
        accepted = accepted[~dup_key]

    accepted["hour"] = accepted["hour"].astype(int)
    accepted["door_count"] = accepted["door_count"].astype(int)
    accepted = accepted.drop(columns=["date"]).rename(columns={"date_parsed": "date"})

    date_min = accepted["date"].min() if len(accepted) else None
    date_max = accepted["date"].max() if len(accepted) else None

    out_path = ctx.artifact_root / "parsed" / "traffic.parquet"
    artifact = write_dataframe(accepted, out_path)

    return SourceResult(
        source_name="traffic",
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
