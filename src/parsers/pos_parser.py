"""POS ingestion (Module 2): read -> structural validation -> materialise raw-typed parquet.

Timestamp normalisation, dedup, SKU/name repair and refund handling are cleaning
concerns (Module 3, src/cleaning/) that require the menu artifact as well; this
parser's job is only to get the file safely into a canonical, typed artifact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe

REQUIRED_COLUMNS = [
    "transaction_id", "timestamp", "sku", "item_name", "quantity",
    "unit_price_sar", "discount_sar", "line_total_sar", "payment_method",
    "channel", "cashier_id",
]


def parse_pos(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    df = pd.read_csv(path, dtype=str)
    raw_row_count = len(df)
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"pos_transactions.csv missing required columns: {missing_cols}")

    df["timestamp_raw"] = df["timestamp"]
    df["item_name_raw"] = df["item_name"]
    df["cashier_id_raw"] = df["cashier_id"]

    for numeric_col in ("quantity", "unit_price_sar", "discount_sar", "line_total_sar"):
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    bad_numeric = df[["quantity", "unit_price_sar", "discount_sar", "line_total_sar"]].isna().any(axis=1)
    rejected = df[bad_numeric]
    accepted = df[~bad_numeric].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} rows rejected for non-numeric quantity/price fields")

    accepted["quantity"] = accepted["quantity"].astype(int)

    out_path = ctx.artifact_root / "parsed" / "pos.parquet"
    artifact = write_dataframe(accepted, out_path)

    return SourceResult(
        source_name="pos",
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
