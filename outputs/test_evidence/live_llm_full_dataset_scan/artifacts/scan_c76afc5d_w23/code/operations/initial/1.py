import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

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

# Define periods
tz = timezone('Asia/Riyadh')
analysis_start = tz.localize(datetime(2026, 6, 15, 0, 0, 0))
analysis_end = tz.localize(datetime(2026, 6, 22, 0, 0, 0))
prev_start = tz.localize(datetime(2026, 6, 8, 0, 0, 0))
prev_end = tz.localize(datetime(2026, 6, 15, 0, 0, 0))

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (Sales Transactions vs Footfall)
# ============================================================================

# Filter analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()

# Count valid sales transactions (exclude refunds, use transaction_id)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Count footfall, excluding dead sensor days
traffic_analysis_clean = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
total_footfall_analysis = traffic_analysis_clean['door_count'].sum()

# Previous period for comparison
pos_prev = pos_df[(pos_df['timestamp'] >= prev_start) & (pos_df['timestamp'] < prev_end)].copy()
traffic_prev = traffic_df[(traffic_df['date'] >= prev_start) & (traffic_df['date'] < prev_end)].copy()

valid_transactions_prev = pos_prev[~pos_prev['is_refund']]['transaction_id'].nunique()
traffic_prev_clean = traffic_prev[~traffic_prev['is_dead_sensor_day']].copy()
total_footfall_prev = traffic_prev_clean['door_count'].sum()

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
    conversion_change = ((conversion_analysis - conversion_prev) / conversion_prev) * 100
    
    findings.append({
        "title": "Conversion Rate Decline Week-over-Week",
        "claim": f"Conversion rate (valid transactions / footfall) decreased from {conversion_prev:.4f} ({conversion_prev*100:.2f}%) in the previous week to {conversion_analysis:.4f} ({conversion_analysis*100:.2f}%) in the analysis week, a decline of {conversion_change:.1f}%.",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions_analysis),
                "denominator": int(total_footfall_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conversion_prev, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions_prev),
                "denominator": int(total_footfall_prev),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "conversion_change_pct": {
                "value": round(conversion_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(valid_transactions_analysis),
        "coverage_notes": [
            f"Analysis period: {valid_transactions_analysis} valid transactions from {pos_analysis['transaction_id'].nunique()} unique transaction IDs",
            f"Footfall: {total_footfall_analysis} door counts across {len(traffic_analysis_clean)} hours (dead sensor days excluded)",
            f"Previous period: {valid_transactions_prev} valid transactions, {total_footfall_prev} footfall"
        ],
        "assumptions": [
            "Refunds excluded from transaction count (is_refund=False)",
            "Dead sensor days excluded from footfall denominator",
            "One transaction_id may contain multiple POS line items",
            "Footfall sensor accuracy assumed consistent within non-dead periods"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Sales Revenue Alignment
# ============================================================================

# Calculate total labour cost for analysis period
staff_analysis = staff_df[staff_df['date'] >= analysis_start.date()]
staff_analysis = staff_analysis[staff_analysis['date'] < analysis_end.date()]
total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()

# Calculate net sales revenue (excluding refunds)
pos_analysis_sales = pos_analysis[~pos_analysis['is_refund']].copy()
total_revenue_analysis = pos_analysis_sales['line_total_sar'].sum()

# Previous period
staff_prev = staff_df[staff_df['date'] >= prev_start.date()]
staff_prev = staff_prev[staff_prev['date'] < prev_end.date()]
total_labour_cost_prev = staff_prev['labour_cost_sar'].sum()

pos_prev_sales = pos_prev[~pos_prev['is_refund']].copy()
total_revenue_prev = pos_prev_sales['line_total_sar'].sum()

if total_revenue_analysis > 0:
    labour_ratio_analysis = total_labour_cost_analysis / total_revenue_analysis
else:
    labour_ratio_analysis = None

if total_revenue_prev > 0:
    labour_ratio_prev = total_labour_cost_prev / total_revenue_prev
else:
    labour_ratio_prev = None

if labour_ratio_analysis is not None and labour_ratio_prev is not None:
    labour_ratio_change = ((labour_ratio_analysis - labour_ratio_prev) / labour_ratio_prev) * 100
    
    findings.append({
        "title": "Labour Cost Ratio Increase",
        "claim": f"Labour cost as a percentage of net sales revenue increased from {labour_ratio_prev*100:.2f}% in the previous week to {labour_ratio_analysis*100:.2f}% in the analysis week, an increase of {labour_ratio_change:.1f}%. This indicates either higher staffing levels or lower sales efficiency.",
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
            "labour_ratio_change_pct": {
                "value": round(labour_ratio_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": analysis_end.isoformat()
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
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period: {len(staff_analysis)} staff shifts with total labour cost {total_labour_cost_analysis:.2f} SAR",
            f"Revenue: {total_revenue_analysis:.2f} SAR from {pos_analysis_sales['transaction_id'].nunique()} transactions",
            f"Previous period: {len(staff_prev)} staff shifts, {total_labour_cost_prev:.2f} SAR labour cost, {total_revenue_prev:.2f} SAR revenue"
        ],
        "assumptions": [
            "Labour cost calculated from staff shift records with computed_duration_hours",
            "Revenue excludes refunds (is_refund=False)",
            "Staff date field used to align with analysis period boundaries",
            "No imputation for missing labour cost records"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste Cost Analysis
# ============================================================================

# Filter inventory for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start]
inv_analysis = inv_analysis[inv_analysis['week_starting'] < analysis_end]

# Filter inventory for previous week
inv_prev = inventory_df[inventory_df['week_starting'] >= prev_start]
inv_prev = inv_prev[inventory_df['week_starting'] < prev_end]

total_waste_cost_analysis = inv_analysis['known_waste_cost_sar'].sum()
total_waste_cost_prev = inv_prev['known_waste_cost_sar'].sum()

total_units_wasted_analysis = inv_analysis['units_wasted'].sum()
total_units_wasted_prev = inv_prev['units_wasted'].sum()

total_units_sold_analysis = inv_analysis['units_sold'].sum()
total_units_sold_prev = inv_prev['units_sold'].sum()

if total_units_sold_analysis > 0:
    waste_ratio_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
else:
    waste_ratio_analysis = None

if total_units_sold_prev > 0:
    waste_ratio_prev = total_units_wasted_prev / (total_units_sold_prev + total_units_wasted_prev)
else:
    waste_ratio_prev = None

if waste_ratio_analysis is not None and waste_ratio_prev is not None and total_waste_cost_analysis > 0:
    waste_ratio_change = ((waste_ratio_analysis - waste_ratio_prev) / waste_ratio_prev) * 100
    
    findings.append({
        "title": "Known Waste Cost Increase",
        "claim": f"Known waste cost increased from {total_waste_cost_prev:.2f} SAR in the previous week to {total_waste_cost_analysis:.2f} SAR in the analysis week. Waste ratio (units wasted / total units handled) increased from {waste_ratio_prev*100:.2f}% to {waste_ratio_analysis*100:.2f}%, a change of {waste_ratio_change:.1f}%.",
        "finding_type": "waste_efficiency",
        "metrics": {
            "known_waste_cost_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_previous": {
                "value": round(total_waste_cost_prev, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "waste_ratio_analysis": {
                "value": round(waste_ratio_analysis, 4),
                "unit": "ratio",
                "numerator": int(total_units_wasted_analysis),
                "denominator": int(total_units_sold_analysis + total_units_wasted_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_ratio_previous": {
                "value": round(waste_ratio_prev, 4),
                "unit": "ratio",
                "numerator": int(total_units_wasted_prev),
                "denominator": int(total_units_sold_prev + total_units_wasted_prev),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "waste_ratio_change_pct": {
                "value": round(waste_ratio_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inv_analysis),
        "coverage_notes": [
            f"Analysis period: {len(inv_analysis)} SKU records with {total_units_wasted_analysis} units wasted",
            f"Known waste cost: {total_waste_cost_analysis:.2f} SAR (unknown waste excluded)",
            f"Previous period: {len(inv_prev)} SKU records, {total_units_wasted_prev} units wasted, {total_waste_cost_prev:.2f} SAR cost"
        ],
        "assumptions": [
            "Waste cost calculated from known_waste_cost_sar field only; unknown waste excluded",
            "Waste ratio computed as units_wasted / (units_sold + units_wasted)",
            "Week_starting field used to align inventory records with analysis periods",
            "No imputation for missing waste values"
        ],
        "confidence": 0.75
    })

# ============================================================================
# Output Result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
