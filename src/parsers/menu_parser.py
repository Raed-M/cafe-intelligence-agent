from __future__ import annotations

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe

REQUIRED_COLUMNS = [
    "sku", "item_en", "item_ar", "category", "price_sar", "unit_cost_sar",
    "is_iced", "launch_date", "retire_date",
]


def parse_menu(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    df = pd.read_csv(path, dtype=str)
    raw_row_count = len(df)
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"menu_items.csv missing required columns: {missing_cols}")

    df["price_sar"] = pd.to_numeric(df["price_sar"], errors="coerce")
    df["unit_cost_sar"] = pd.to_numeric(df["unit_cost_sar"], errors="coerce")

    bad = df["sku"].isna() | df["price_sar"].isna() | df["unit_cost_sar"].isna()
    bad |= df["price_sar"] < 0
    bad |= df["unit_cost_sar"] < 0
    rejected = df[bad]
    accepted = df[~bad].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} menu rows rejected for invalid sku/price/cost")

    dup_sku = accepted["sku"].duplicated()
    if dup_sku.any():
        warnings.append(f"{dup_sku.sum()} duplicate SKUs found; keeping first occurrence")
        accepted = accepted[~dup_sku]

    accepted["is_iced"] = accepted["is_iced"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y"]
    )

    out_path = ctx.artifact_root / "parsed" / "menu.parquet"
    artifact = write_dataframe(accepted, out_path)

    return SourceResult(
        source_name="menu",
        status="success" if not len(rejected) else "partial",
        raw_row_count=raw_row_count,
        accepted_row_count=len(accepted),
        rejected_row_count=len(rejected),
        artifact=artifact,
        schema_version="1.0",
        date_min=None,
        date_max=None,
        warnings=warnings,
        error=None,
    )
