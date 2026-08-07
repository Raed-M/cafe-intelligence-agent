import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone as dt_timezone

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

# Define periods using datetime.timezone (UTC+03:00 for Asia/Riyadh)
riyadh_offset = dt_timezone(timedelta(hours=3))
analysis_start = datetime(2026, 7, 13, 0, 0, 0, tzinfo=riyadh_offset)
analysis_end = datetime(2026, 7, 20, 0, 0, 0, tzinfo=riyadh_offset)
prev_start = datetime(2026, 7, 6, 0, 0, 0, tzinfo=riyadh_offset)
prev_end = datetime(2026, 7, 13, 0, 0, 0, tzinfo=riyadh_offset)

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(riyadh_offset)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(riyadh_offset)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(riyadh_offset)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(riyadh_offset)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(riyadh_offset)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(riyadh_offset)

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Filter analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()

# Filter previous period
pos_prev = pos_df[(pos_df['timestamp'] >= prev_start) & (pos_df['timestamp'] < prev_end)].copy()
traffic_prev = traffic_df[(traffic_df['date'] >= prev_start) & (traffic_df['date'] < prev_end)].copy()

# Count valid transactions (exclude refunds, count unique transaction_ids)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()
valid_transactions_prev = pos_prev[~pos_prev['is_refund']]['transaction_id'].nunique()

# Count valid footfall (exclude dead sensor days)
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
traffic_prev_valid = traffic_prev[~traffic_prev['is_dead_sensor_day']].copy()

footfall_analysis = traffic_analysis_valid['door_count'].sum()
footfall_prev = traffic_prev_valid['door_count'].sum()

# Calculate conversion rates
if footfall_analysis > 0:
    conversion_analysis = valid_transactions_analysis / footfall_analysis
else:
    conversion_analysis = None

if footfall_prev > 0:
    conversion_prev = valid_transactions_prev / footfall_prev
else:
    conversion_prev = None

if conversion_analysis is not None and conversion_prev is not None:
    conversion_change = conversion_analysis - conversion_prev
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis week (Jul 13-20) was {conversion_analysis:.4f} vs {conversion_prev:.4f} in previous week, a change of {conversion_change:+.4f}",
        "finding_type": "conversion_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(valid_transactions_analysis),
                "denominator": int(footfall_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conversion_prev, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(valid_transactions_prev),
                "denominator": int(footfall_prev),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "conversion_change": {
                "value": round(conversion_change, 4),
                "unit": "transactions_per_visitor",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(valid_transactions_analysis),
        "coverage_notes": [
            f"Analysis period: {len(traffic_analysis_valid)} valid traffic days (excluded {len(traffic_analysis) - len(traffic_analysis_valid)} dead sensor days)",
            f"Previous period: {len(traffic_prev_valid)} valid traffic days (excluded {len(traffic_prev) - len(traffic_prev_valid)} dead sensor days)",
            "Transactions counted as unique transaction_ids excluding refunds"
        ],
        "assumptions": [
            "Footfall sensor accuracy assumed for non-dead-sensor intervals",
            "Refunds excluded from transaction count per metric definition"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Demand Alignment
# ============================================================================

# Aggregate labour cost by date for analysis period
staff_analysis = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)].copy()
staff_prev = staff_df[(staff_df['date'] >= prev_start) & (staff_df['date'] < prev_end)].copy()

labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
labour_cost_prev = staff_prev['labour_cost_sar'].sum()

# Calculate average daily labour cost
analysis_days = len(staff_analysis['date'].unique())
prev_days = len(staff_prev['date'].unique())

if analysis_days > 0:
    avg_daily_labour_analysis = labour_cost_analysis / analysis_days
else:
    avg_daily_labour_analysis = None

if prev_days > 0:
    avg_daily_labour_prev = labour_cost_prev / prev_days
else:
    avg_daily_labour_prev = None

# Calculate revenue for same periods
pos_analysis_revenue = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
pos_prev_revenue = pos_prev[~pos_prev['is_refund']]['line_total_sar'].sum()

if analysis_days > 0:
    avg_daily_revenue_analysis = pos_analysis_revenue / analysis_days
else:
    avg_daily_revenue_analysis = None

if prev_days > 0:
    avg_daily_revenue_prev = pos_prev_revenue / prev_days
else:
    avg_daily_revenue_prev = None

# Labour cost as % of revenue
if avg_daily_revenue_analysis is not None and avg_daily_revenue_analysis > 0:
    labour_pct_analysis = (avg_daily_labour_analysis / avg_daily_revenue_analysis) * 100
else:
    labour_pct_analysis = None

if avg_daily_revenue_prev is not None and avg_daily_revenue_prev > 0:
    labour_pct_prev = (avg_daily_labour_prev / avg_daily_revenue_prev) * 100
else:
    labour_pct_prev = None

if labour_pct_analysis is not None and labour_pct_prev is not None:
    findings.append({
        "title": "Labour Cost as Percentage of Revenue",
        "claim": f"Average daily labour cost was {labour_pct_analysis:.1f}% of revenue in analysis week vs {labour_pct_prev:.1f}% in previous week",
        "finding_type": "labour_efficiency",
        "metrics": {
            "labour_cost_pct_analysis": {
                "value": round(labour_pct_analysis, 1),
                "unit": "percent",
                "numerator": round(avg_daily_labour_analysis, 2),
                "denominator": round(avg_daily_revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_pct_previous": {
                "value": round(labour_pct_prev, 1),
                "unit": "percent",
                "numerator": round(avg_daily_labour_prev, 2),
                "denominator": round(avg_daily_revenue_prev, 2),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "avg_daily_labour_analysis": {
                "value": round(avg_daily_labour_analysis, 2),
                "unit": "SAR",
                "numerator": round(labour_cost_analysis, 2),
                "denominator": analysis_days,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "avg_daily_revenue_analysis": {
                "value": round(avg_daily_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": round(pos_analysis_revenue, 2),
                "denominator": analysis_days,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": analysis_days,
        "coverage_notes": [
            f"Analysis period: {analysis_days} days with staff data",
            f"Previous period: {prev_days} days with staff data",
            "Labour cost includes all shifts with computed_duration_hours",
            "Revenue calculated from non-refund transactions"
        ],
        "assumptions": [
            "Staff shifts fully captured in shift_start/shift_end",
            "Hourly rates accurately reflect actual compensation",
            "Revenue excludes refunds per metric definition"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Waste Cost Analysis
# ============================================================================

# Get inventory data for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start].copy()
inv_prev = inventory_df[inventory_df['week_starting'] >= prev_start].copy()
inv_prev = inv_prev[inv_prev['week_starting'] < analysis_start]

total_waste_cost_analysis = inv_analysis['known_waste_cost_sar'].sum()
total_waste_cost_prev = inv_prev['known_waste_cost_sar'].sum()

total_revenue_analysis = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
total_revenue_prev = pos_prev[~pos_prev['is_refund']]['line_total_sar'].sum()

if total_revenue_analysis > 0:
    waste_pct_analysis = (total_waste_cost_analysis / total_revenue_analysis) * 100
else:
    waste_pct_analysis = None

if total_revenue_prev > 0:
    waste_pct_prev = (total_waste_cost_prev / total_revenue_prev) * 100
else:
    waste_pct_prev = None

if waste_pct_analysis is not None and waste_pct_prev is not None and len(inv_analysis) > 0:
    findings.append({
        "title": "Known Waste Cost as Percentage of Revenue",
        "claim": f"Known waste cost was {waste_pct_analysis:.2f}% of revenue in analysis week vs {waste_pct_prev:.2f}% in previous week",
        "finding_type": "waste_efficiency",
        "metrics": {
            "waste_cost_pct_analysis": {
                "value": round(waste_pct_analysis, 2),
                "unit": "percent",
                "numerator": round(total_waste_cost_analysis, 2),
                "denominator": round(total_revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_cost_pct_previous": {
                "value": round(waste_pct_prev, 2),
                "unit": "percent",
                "numerator": round(total_waste_cost_prev, 2),
                "denominator": round(total_revenue_prev, 2),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "total_waste_cost_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_waste_cost_previous": {
                "value": round(total_waste_cost_prev, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": len(inv_analysis),
        "coverage_notes": [
            f"Analysis period: {len(inv_analysis)} inventory records with known waste",
            f"Previous period: {len(inv_prev)} inventory records with known waste",
            "Waste cost includes only known_waste_cost_sar (unknown waste excluded)",
            "Revenue from non-refund transactions"
        ],
        "assumptions": [
            "Inventory waste data captured accurately in weekly counts",
            "Known waste cost reflects actual disposal/loss value",
            "Sunday inventory counts treated as weekly snapshot, not real-time"
        ],
        "confidence": 0.75
    })

# Write output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
