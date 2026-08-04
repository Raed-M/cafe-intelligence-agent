"""Repair unreliable POS item_name via SKU join to the canonical menu, never the
reverse. Unknown SKUs (not present in menu) are quarantined from item-level
analysis but kept (flagged) in the cleaned artifact for quality-count purposes.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def repair_item_names(pos_df: pd.DataFrame, menu_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    menu_lookup = menu_df.set_index("sku")[["item_en", "item_ar", "category"]]
    df = pos_df.copy()
    df["known_sku"] = df["sku"].isin(menu_lookup.index)

    joined = df.join(menu_lookup, on="sku", rsuffix="_menu")
    df["item_name_en"] = joined["item_en"]
    df["item_name_ar"] = joined["item_ar"]
    df["category"] = joined["category"]

    n_unknown_sku = int((~df["known_sku"]).sum())
    n_repaired = int((df["known_sku"] & df["item_name_raw"].isna()).sum())

    audit = {
        "unknown_sku_rows": n_unknown_sku,
        "unknown_skus": sorted(df.loc[~df["known_sku"], "sku"].unique().tolist()),
        "rows_with_name_repaired_from_menu": n_repaired,
    }
    return df, audit
