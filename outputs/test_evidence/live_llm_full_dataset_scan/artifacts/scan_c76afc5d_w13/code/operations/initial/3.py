import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
staff_df = pd.read_parquet(inputs['staff'])
inventory_df = pd.read_parquet(inputs['inventory'])

# Define periods using datetime.timezone.utc and manual offset
tz_offset = timezone(timedelta(hours=3))  # Asia/Riyadh is UTC+3
analysis_start = datetime(2026, 4, 6, 0, 0, 0, tzinfo=tz_offset)
analysis_end = datetime(2026, 4, 13, 0, 0, 0, tzinfo=tz_offset)
prev_start = datetime(2026, 3, 30, 0, 0, 0, tzinfo=tz_offset)
prev_end = datetime(2026, 4, 6, 0, 0, 0, tzinfo=tz_offset)

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz_offset)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz_offset)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz_offset)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz_offset)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz_offset)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz_offset)

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS vs Traffic)
# ============================================================================

# Filter analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()

# Count valid transactions (exclude refunds, use transaction_id)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Count valid footfall (exclude dead sensor days)
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
total_footfall_analysis = traffic_analysis_valid['door_count'].sum()

# Previous period
pos_prev = pos_df[(pos_df['timestamp'] >= prev_start) & (pos_df['timestamp'] < prev_end)].copy()
traffic_prev = traffic_df[(traffic_df['date'] >= prev_start) & (traffic_df['date'] < prev_end)].copy()

valid_transactions_prev = pos_prev[~pos_prev['is_refund']]['transaction_id'].nunique()
traffic_prev_valid = traffic_prev[~traffic_prev['is_dead_sensor_day']].copy()
total_footfall_prev = traffic_prev_valid['door_count'].sum()

# Calculate conversion rates
if total_footfall_analysis > 0:
    conversion_analysis = valid_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

if total_footfall_prev > 0:
    conversion_prev = valid_transactions_prev / total_footfall_prev
else:
    conversion_prev = None

if conversion_analysis is not None and conversion_prev is not None:
    conversion_change = conversion_analysis - conversion_prev
    
    findings.append({
        "title": "Conversion Rate Decline Week-over-Week",
        "claim": f"Conversion rate decreased from {conversion_prev:.4f} to {conversion_analysis:.4f}, a decline of {conversion_change:.4f} ({conversion_change/conversion_prev*100:.1f}%)",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "ratio",
                "numerator": valid_transactions_analysis,
                "denominator": total_footfall_analysis,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conversion_prev, 4),
                "unit": "ratio",
                "numerator": valid_transactions_prev,
                "denominator": total_footfall_prev,
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "conversion_change": {
                "value": round(conversion_change, 4),
                "unit": "ratio_points",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": valid_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {valid_transactions_analysis} valid transactions from {pos_analysis['transaction_id'].nunique()} total transactions",
            f"Traffic: {total_footfall_analysis} door counts from {len(traffic_analysis_valid)} valid hours (dead sensor days excluded)",
            f"Previous period: {valid_transactions_prev} valid transactions, {total_footfall_prev} door counts"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall denominator",
            "One transaction_id = one basket regardless of line items"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Sales Revenue
# ============================================================================

# Calculate total labour cost for analysis period
# Convert analysis_start and analysis_end to date objects for comparison with staff_df['date']
analysis_start_date = analysis_start.date()
analysis_end_date = analysis_end.date()
prev_start_date = prev_start.date()
prev_end_date = prev_end.date()

staff_analysis = staff_df[(staff_df['date'].dt.date >= analysis_start_date) & (staff_df['date'].dt.date < analysis_end_date)].copy()
total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()

# Calculate total sales revenue (net of refunds and discounts)
pos_analysis_sales = pos_analysis[~pos_analysis['is_refund']].copy()
total_revenue_analysis = pos_analysis_sales['line_total_sar'].sum()

# Previous period
staff_prev = staff_df[(staff_df['date'].dt.date >= prev_start_date) & (staff_df['date'].dt.date < prev_end_date)].copy()
total_labour_cost_prev = staff_prev['labour_cost_sar'].sum()

pos_prev_sales = pos_prev[~pos_prev['is_refund']].copy()
total_revenue_prev = pos_prev_sales['line_total_sar'].sum()

# Calculate labour cost as % of revenue
if total_revenue_analysis > 0:
    labour_ratio_analysis = total_labour_cost_analysis / total_revenue_analysis
else:
    labour_ratio_analysis = None

if total_revenue_prev > 0:
    labour_ratio_prev = total_labour_cost_prev / total_revenue_prev
else:
    labour_ratio_prev = None

if labour_ratio_analysis is not None and labour_ratio_prev is not None:
    labour_ratio_change = labour_ratio_analysis - labour_ratio_prev
    
    findings.append({
        "title": "Labour Cost Efficiency Deterioration",
        "claim": f"Labour cost as % of revenue increased from {labour_ratio_prev*100:.1f}% to {labour_ratio_analysis*100:.1f}%, indicating reduced labour efficiency",
        "finding_type": "cost_efficiency",
        "metrics": {
            "labour_cost_ratio_analysis": {
                "value": round(labour_ratio_analysis, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost_analysis, 2),
                "denominator": round(total_revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_ratio_previous": {
                "value": round(labour_ratio_prev, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost_prev, 2),
                "denominator": round(total_revenue_prev, 2),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "total_labour_cost_analysis": {
                "value": round(total_labour_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_revenue_analysis": {
                "value": round(total_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Staff records: {len(staff_analysis)} shifts in analysis period",
            f"POS records: {pos_analysis_sales['transaction_id'].nunique()} valid transactions",
            f"Labour cost calculated from computed_duration_hours × hourly_rate_sar"
        ],
        "assumptions": [
            "Labour cost includes all shifts in period",
            "Revenue excludes refunds and includes net line totals",
            "No imputation for missing staff records"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Filter inventory for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start].copy()
inv_prev = inventory_df[(inventory_df['week_starting'] >= prev_start) & (inventory_df['week_starting'] < analysis_start)].copy()

total_waste_cost_analysis = inv_analysis['known_waste_cost_sar'].sum()
total_units_sold_analysis = inv_analysis['units_sold'].sum()
total_units_wasted_analysis = inv_analysis['units_wasted'].sum()

total_waste_cost_prev = inv_prev['known_waste_cost_sar'].sum()
total_units_sold_prev = inv_prev['units_sold'].sum()
total_units_wasted_prev = inv_prev['units_wasted'].sum()

# Calculate waste rate
if total_units_sold_analysis > 0:
    waste_rate_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
else:
    waste_rate_analysis = None

if total_units_sold_prev > 0:
    waste_rate_prev = total_units_wasted_prev / (total_units_sold_prev + total_units_wasted_prev)
else:
    waste_rate_prev = None

if waste_rate_analysis is not None and waste_rate_prev is not None:
    waste_rate_change = waste_rate_analysis - waste_rate_prev
    
    findings.append({
        "title": "Increased Product Waste Rate",
        "claim": f"Known waste rate increased from {waste_rate_prev*100:.1f}% to {waste_rate_analysis*100:.1f}%, with known waste cost of {total_waste_cost_analysis:.2f} SAR",
        "finding_type": "waste_metric",
        "metrics": {
            "waste_rate_analysis": {
                "value": round(waste_rate_analysis, 4),
                "unit": "ratio",
                "numerator": total_units_wasted_analysis,
                "denominator": total_units_sold_analysis + total_units_wasted_analysis,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_rate_previous": {
                "value": round(waste_rate_prev, 4),
                "unit": "ratio",
                "numerator": total_units_wasted_prev,
                "denominator": total_units_sold_prev + total_units_wasted_prev,
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "known_waste_cost_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inv_analysis),
        "coverage_notes": [
            f"Inventory records: {len(inv_analysis)} SKU-week combinations in analysis period",
            f"Known waste only; unknown waste values preserved per schema",
            f"Waste cost calculated from units_wasted × unit_cost_sar"
        ],
        "assumptions": [
            "Inventory counts reflect weekly aggregates",
            "Known waste cost is deterministic; unknown waste excluded",
            "No real-time stock validation against Sunday counts"
        ],
        "confidence": 0.75
    })

# ============================================================================
# Output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
