from __future__ import annotations

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe, write_json

REQUIRED_COLUMNS = [
    "week_starting", "sku", "item", "units_ordered", "units_sold", "units_wasted", "unit_cost_sar",
]

_DATE_FORMATS = ["%Y-%m-%d", "%d-%b-%Y"]


def _parse_mixed_date(raw: str) -> str | None:
    raw = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            from datetime import datetime

            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_inventory(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    sheet = source.sheet or "weekly_counts"
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    raw_row_count = len(df)
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"inventory_weekly.xlsx[{sheet}] missing required columns: {missing_cols}")

    df["week_starting_raw"] = df["week_starting"]
    df["week_starting"] = df["week_starting"].map(_parse_mixed_date)

    for col in ("units_ordered", "units_sold"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # units_wasted: blank means "not recorded" -> stays NaN/null, never coerced to 0.
    df["units_wasted"] = pd.to_numeric(df["units_wasted"], errors="coerce")
    df["unit_cost_sar"] = pd.to_numeric(df["unit_cost_sar"], errors="coerce")

    bad = (
        df["week_starting"].isna() | df["sku"].isna()
        | df["units_ordered"].isna() | df["units_sold"].isna() | df["unit_cost_sar"].isna()
    )
    rejected = df[bad]
    accepted = df[~bad].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} inventory rows rejected for invalid week/sku/quantities/cost")

    n_blank_waste = accepted["units_wasted"].isna().sum()
    if n_blank_waste:
        warnings.append(f"{n_blank_waste} rows have unknown (blank) units_wasted; treated as unrecorded, not zero")

    out_path = ctx.artifact_root / "parsed" / "inventory.parquet"
    artifact = write_dataframe(accepted, out_path)

    # Preserve README sheet as source metadata.
    try:
        readme_df = pd.read_excel(path, sheet_name="README", header=None)
        readme_notes = readme_df.iloc[1:, 0].dropna().astype(str).tolist()
    except Exception:  # noqa: BLE001
        readme_notes = []
    write_json({"notes": readme_notes}, ctx.artifact_root / "parsed" / "inventory_readme.json")

    date_min = accepted["week_starting"].min() if len(accepted) else None
    date_max = accepted["week_starting"].max() if len(accepted) else None

    return SourceResult(
        source_name="inventory",
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
