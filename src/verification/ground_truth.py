"""Independent hand-verification of 5 metrics, computed directly from the raw
supplied files with plain pandas -- deliberately *not* calling any analyst
LLM-generated code, and re-implementing dedup/filtering logic independently
of `src/cleaning/` so this serves as a genuine cross-check, not a rerun of the
same code path (plan section 28.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class VerificationResult:
    metric_name: str
    formula: str
    filters: str
    manual_value: float
    unit: str


def _load_pos(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "pos_transactions.csv")
    def parse_ts(raw: str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M"):
            try:
                return pd.Timestamp(datetime.strptime(str(raw), fmt))
            except (ValueError, TypeError):
                continue
        return pd.NaT
    df["ts"] = df["timestamp"].map(parse_ts)
    return df


def _independent_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Re-implements double-swipe removal independently: within each
    transaction_id, if every distinct (ts, sku, quantity, unit_price_sar,
    discount_sar, line_total_sar, payment_method, channel) signature repeats
    with identical multiplicity >= 2, keep only one copy of each."""
    sig_cols = ["ts", "sku", "quantity", "unit_price_sar", "discount_sar", "line_total_sar", "payment_method", "channel"]
    df = df.copy()
    df["_sig"] = list(zip(*[df[c].astype(str) for c in sig_cols]))
    keep_idx = []
    for tid, g in df.groupby("transaction_id"):
        counts = g["_sig"].value_counts()
        if len(counts) and (counts.values == counts.values[0]).all() and counts.values[0] >= 2:
            keep_idx.extend(g.groupby("_sig").head(1).index.tolist())
        else:
            keep_idx.extend(g.index.tolist())
    return df.loc[keep_idx].drop(columns=["_sig"])


def verify_net_revenue(data_dir: Path, week_start: str, week_end: str) -> VerificationResult:
    df = _load_pos(data_dir)
    week = df[(df["ts"] >= week_start) & (df["ts"] < week_end)]
    week = _independent_dedup(week)
    value = float(week["line_total_sar"].sum())
    return VerificationResult(
        "net_revenue", "sum(line_total_sar) after independent double-swipe dedup, including refunds",
        f"timestamp in [{week_start}, {week_end})", value, "SAR",
    )


def verify_transaction_count(data_dir: Path, week_start: str, week_end: str) -> VerificationResult:
    df = _load_pos(data_dir)
    week = df[(df["ts"] >= week_start) & (df["ts"] < week_end)]
    week = _independent_dedup(week)
    basket = week.groupby("transaction_id").agg(
        has_positive=("quantity", lambda s: (s > 0).any()),
        net_total=("line_total_sar", "sum"),
    )
    valid = basket[(basket["has_positive"]) & (basket["net_total"] > 0)]
    return VerificationResult(
        "valid_transaction_count", "count(distinct transaction_id) with >=1 positive line and net basket revenue > 0",
        f"timestamp in [{week_start}, {week_end})", float(len(valid)), "transactions",
    )


def verify_conversion_rate(data_dir: Path, week_start: str, week_end: str) -> VerificationResult:
    pos_df = _load_pos(data_dir)
    week_pos = pos_df[(pos_df["ts"] >= week_start) & (pos_df["ts"] < week_end)]
    week_pos = _independent_dedup(week_pos)
    basket = week_pos.groupby("transaction_id").agg(
        has_positive=("quantity", lambda s: (s > 0).any()), net_total=("line_total_sar", "sum"),
    )
    valid_tx = int(((basket["has_positive"]) & (basket["net_total"] > 0)).sum())

    traffic = pd.read_csv(data_dir / "foot_traffic.csv")
    traffic["date"] = pd.to_datetime(traffic["date"])
    week_traffic = traffic[(traffic["date"] >= week_start[:10]) & (traffic["date"] < week_end[:10])]
    daily_sum = week_traffic.groupby(week_traffic["date"].dt.date)["door_count"].sum()
    dead_days = daily_sum[daily_sum == 0].index.tolist()
    week_traffic = week_traffic[~week_traffic["date"].dt.date.isin(dead_days)]
    footfall = int(week_traffic["door_count"].sum())

    rate = valid_tx / footfall if footfall else None
    return VerificationResult(
        "conversion_rate", "valid_transaction_count / sum(door_count) excluding full-zero (dead sensor) days",
        f"date in [{week_start[:10]}, {week_end[:10]}), dead_days_excluded={dead_days}", rate or 0.0, "ratio",
    )


def verify_sku_gross_profit(data_dir: Path, week_start: str, week_end: str, sku: str) -> VerificationResult:
    df = _load_pos(data_dir)
    menu = pd.read_csv(data_dir / "menu_items.csv")
    unit_cost = float(menu.loc[menu["sku"] == sku, "unit_cost_sar"].iloc[0])

    week = df[(df["ts"] >= week_start) & (df["ts"] < week_end) & (df["sku"] == sku)]
    week = _independent_dedup(week)
    revenue = float(week["line_total_sar"].sum())
    cogs = float((week["quantity"] * unit_cost).sum())
    gross_profit = revenue - cogs
    return VerificationResult(
        f"gross_profit[{sku}]", f"sum(line_total_sar) - sum(quantity * menu.unit_cost_sar={unit_cost})",
        f"sku={sku}, timestamp in [{week_start}, {week_end})", gross_profit, "SAR",
    )


def verify_known_waste_cost(data_dir: Path, week_starting: str) -> VerificationResult:
    inv = pd.read_excel(data_dir / "inventory_weekly.xlsx", sheet_name="weekly_counts")

    def parse_date(raw):
        for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
            try:
                return pd.Timestamp(datetime.strptime(str(raw), fmt))
            except (ValueError, TypeError):
                continue
        return pd.NaT

    inv["week_starting_parsed"] = inv["week_starting"].map(parse_date)
    week = inv[inv["week_starting_parsed"] == pd.Timestamp(week_starting)]
    known = week[week["units_wasted"].notna()]
    cost = float((known["units_wasted"] * known["unit_cost_sar"]).sum())
    return VerificationResult(
        "known_waste_cost", "sum(units_wasted * unit_cost_sar) where units_wasted is not null",
        f"week_starting={week_starting}, {len(week) - len(known)} unknown-waste rows excluded", cost, "SAR",
    )


def run_all_verifications(data_dir: Path, week_start: str, week_end: str, sku: str) -> list[VerificationResult]:
    return [
        verify_net_revenue(data_dir, week_start, week_end),
        verify_transaction_count(data_dir, week_start, week_end),
        verify_conversion_rate(data_dir, week_start, week_end),
        verify_sku_gross_profit(data_dir, week_start, week_end, sku),
        verify_known_waste_cost(data_dir, week_start[:10]),
    ]
