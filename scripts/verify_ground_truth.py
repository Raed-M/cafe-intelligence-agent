"""Runs the 5 independent hand-verification checks and cross-compares them
against this run's own cleaned-artifact pipeline output for the same week,
recording formula, filters, agent value, manual value, absolute difference
and pass/fail against a tolerance (plan section 28.4)."""
from __future__ import annotations

import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.cleaning.cleaner import clean_and_materialise
from src.config.runtime_config import resolve_runtime_config
from src.graph.ingestion_subgraph import build_ingestion_subgraph
from src.tools.artifact_io import read_dataframe
from src.verification.ground_truth import run_all_verifications

TOLERANCE_ABS = 0.05
WEEK_START, WEEK_END = "2026-01-05", "2026-01-12"
SKU = "HOT-001"


def pipeline_values(data_dir: Path) -> dict[str, float]:
    config = resolve_runtime_config(
        profile_path=data_dir / "cafe_profile.json", data_dir=data_dir,
        app_settings_path=Path("config/app_settings.yaml"),
        source_registry_path=Path("config/source_registry.yaml"),
        target_week=date(2026, 1, 5),
    )
    run_id = "verify_" + uuid.uuid4().hex[:8]
    graph = build_ingestion_subgraph()
    ingest_out = graph.invoke({"run_id": run_id, "config": config})
    clean_out = clean_and_materialise({"run_id": run_id, "config": config, "source_results": ingest_out["source_results"]})
    cleaned = clean_out["cleaned_artifacts"]

    pos = read_dataframe(cleaned["pos"])
    week = pos[(pos["business_date"] >= WEEK_START) & (pos["business_date"] < WEEK_END)]
    net_revenue = float(week["line_total_sar"].sum())

    valid_tx = week.groupby("transaction_id").agg(
        has_positive=("quantity", lambda s: (s > 0).any()), net_total=("line_total_sar", "sum"),
    )
    tx_count = int(((valid_tx["has_positive"]) & (valid_tx["net_total"] > 0)).sum())

    traffic = read_dataframe(cleaned["traffic"])
    week_traffic = traffic[(traffic["date"] >= WEEK_START) & (traffic["date"] < WEEK_END) & (~traffic["is_dead_sensor_day"])]
    footfall = int(week_traffic["door_count"].sum())
    conversion = tx_count / footfall if footfall else 0.0

    sku_week = week[week["sku"] == SKU]
    menu = read_dataframe(cleaned["menu"])
    unit_cost = float(menu.loc[menu["sku"] == SKU, "unit_cost_sar"].iloc[0])
    gross_profit = float(sku_week["line_total_sar"].sum()) - float((sku_week["quantity"] * unit_cost).sum())

    inv = read_dataframe(cleaned["inventory"])
    inv_week = inv[inv["week_starting"] == WEEK_START]
    known_waste_cost = float(inv_week["known_waste_cost_sar"].fillna(0).sum())

    return {
        "net_revenue": net_revenue, "valid_transaction_count": float(tx_count),
        "conversion_rate": conversion, f"gross_profit[{SKU}]": gross_profit,
        "known_waste_cost": known_waste_cost,
    }


def main() -> None:
    data_dir = Path("data/qahwa_saihat")
    manual = {r.metric_name: r for r in run_all_verifications(data_dir, WEEK_START, WEEK_END, SKU)}
    pipeline = pipeline_values(data_dir)

    rows = []
    for name, result in manual.items():
        agent_value = pipeline.get(name)
        diff = abs(agent_value - result.manual_value) if agent_value is not None else None
        passed = diff is not None and diff <= TOLERANCE_ABS
        rows.append({
            "metric": name, "formula": result.formula, "filters": result.filters,
            "manual_value": result.manual_value, "agent_value": agent_value,
            "abs_diff": diff, "pass": passed,
        })

    out_dir = Path("outputs/test_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ground_truth_verification.csv", index=False)

    print(df.to_string(index=False))
    all_pass = df["pass"].all()
    print(f"\nAll checks passed: {all_pass}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
