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
analysis_start = pd.Timestamp('2026-02-02T00:00:00+03:00')
analysis_end = pd.Timestamp('2026-02-09T00:00:00+03:00')
previous_start = pd.Timestamp('2026-01-26T00:00:00+03:00')
previous_end = pd.Timestamp('2026-02-02T00:00:00+03:00')

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=False).dt.tz_localize('Asia/Riyadh')
staff_df['date'] = pd.to_datetime(staff_df['date'], utc=False).dt.tz_localize('Asia/Riyadh')
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=False).dt.tz_localize('Asia/Riyadh')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=False).dt.tz_localize('Asia/Riyadh')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=False).dt.tz_localize('Asia/Riyadh')

# Convert pos timestamp to Asia/Riyadh for business_date alignment
pos_df['timestamp_riyadh'] = pos_df['timestamp'].dt.tz_convert('Asia/Riyadh')
pos_df['business_date_check'] = pos_df['timestamp_riyadh'].dt.date

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_riyadh'] >= analysis_start) & 
    (pos_df['timestamp_riyadh'] < analysis_end)
].copy()

pos_previous = pos_df[
    (pos_df['timestamp_riyadh'] >= previous_start) & 
    (pos_df['timestamp_riyadh'] < previous_end)
].copy()

# Count valid transactions (exclude refunds for conversion numerator)
valid_transactions_analysis = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
valid_transactions_previous = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()

# Filter traffic for analysis period (exclude dead sensor days)
traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start.normalize()) & 
    (traffic_df['date'] < analysis_end.normalize()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

traffic_previous = traffic_df[
    (traffic_df['date'] >= previous_start.normalize()) & 
    (traffic_df['date'] < previous_end.normalize()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

total_footfall_analysis = traffic_analysis['door_count'].sum()
total_footfall_previous = traffic_previous['door_count'].sum()

# Calculate conversion rates
if total_footfall_analysis > 0:
    conversion_analysis = valid_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

if total_footfall_previous > 0:
    conversion_previous = valid_transactions_previous / total_footfall_previous
else:
    conversion_previous = None

if conversion_analysis is not None and conversion_previous is not None:
    conversion_change = conversion_analysis - conversion_previous
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis week (Feb 2-9) was {conversion_analysis:.4f} vs {conversion_previous:.4f} in previous week, a change of {conversion_change:+.4f}",
        "finding_type": "conversion_metric",
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
                "value": round(conversion_previous, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions_previous),
                "denominator": int(total_footfall_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_change": {
                "value": round(conversion_change, 4),
                "unit": "ratio_change",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(valid_transactions_analysis),
        "coverage_notes": [
            f"Analysis period: {len(traffic_analysis)} traffic records after excluding dead sensor days",
            f"Previous period: {len(traffic_previous)} traffic records after excluding dead sensor days",
            "Conversion calculated as unique valid (non-refund) transactions / total footfall"
        ],
        "assumptions": [
            "Dead sensor days excluded from footfall denominator",
            "Refunds excluded from transaction count",
            "Each transaction_id represents one unique customer transaction"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Demand Alignment
# ============================================================================

# Filter staff for analysis period
staff_analysis = staff_df[
    (staff_df['date'] >= analysis_start.normalize()) & 
    (staff_df['date'] < analysis_end.normalize())
].copy()

staff_previous = staff_df[
    (staff_df['date'] >= previous_start.normalize()) & 
    (staff_df['date'] < previous_end.normalize())
].copy()

total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
total_labour_cost_previous = staff_previous['labour_cost_sar'].sum()

# Calculate revenue (net of refunds)
revenue_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
revenue_previous = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()

# Add refund amounts (negative)
refund_analysis = pos_analysis[pos_analysis['is_refund'] == True]['line_total_sar'].sum()
refund_previous = pos_previous[pos_previous['is_refund'] == True]['line_total_sar'].sum()

net_revenue_analysis = revenue_analysis + refund_analysis
net_revenue_previous = revenue_previous + refund_previous

# Calculate labour cost as % of revenue
if net_revenue_analysis > 0:
    labour_pct_analysis = (total_labour_cost_analysis / net_revenue_analysis) * 100
else:
    labour_pct_analysis = None

if net_revenue_previous > 0:
    labour_pct_previous = (total_labour_cost_previous / net_revenue_previous) * 100
else:
    labour_pct_previous = None

if labour_pct_analysis is not None and labour_pct_previous is not None:
    labour_pct_change = labour_pct_analysis - labour_pct_previous
    
    findings.append({
        "title": "Labour Cost Efficiency",
        "claim": f"Labour cost as % of net revenue was {labour_pct_analysis:.2f}% in analysis week vs {labour_pct_previous:.2f}% in previous week, a change of {labour_pct_change:+.2f} percentage points",
        "finding_type": "labour_efficiency",
        "metrics": {
            "labour_cost_pct_analysis": {
                "value": round(labour_pct_analysis, 2),
                "unit": "percent",
                "numerator": round(total_labour_cost_analysis, 2),
                "denominator": round(net_revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_pct_previous": {
                "value": round(labour_pct_previous, 2),
                "unit": "percent",
                "numerator": round(total_labour_cost_previous, 2),
                "denominator": round(net_revenue_previous, 2),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_pct_change": {
                "value": round(labour_pct_change, 2),
                "unit": "percentage_points",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period: {len(staff_analysis)} staff records",
            f"Previous period: {len(staff_previous)} staff records",
            "Labour cost includes all computed labour_cost_sar values",
            "Revenue calculated as net of refunds"
        ],
        "assumptions": [
            "Staff labour_cost_sar is accurate and complete",
            "Refunds are properly marked with is_refund flag",
            "Labour cost allocation is uniform across the period"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Filter inventory for analysis week
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start.normalize()) & 
    (inventory_df['week_starting'] < analysis_end.normalize())
].copy()

inventory_previous = inventory_df[
    (inventory_df['week_starting'] >= previous_start.normalize()) & 
    (inventory_df['week_starting'] < previous_end.normalize())
].copy()

# Calculate known waste metrics
total_known_waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
total_known_waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()

total_units_sold_analysis = inventory_analysis['units_sold'].sum()
total_units_sold_previous = inventory_previous['units_sold'].sum()

total_units_wasted_analysis = inventory_analysis['units_wasted'].sum()
total_units_wasted_previous = inventory_previous['units_wasted'].sum()

# Calculate waste rate
if (total_units_sold_analysis + total_units_wasted_analysis) > 0:
    waste_rate_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
else:
    waste_rate_analysis = None

if (total_units_sold_previous + total_units_wasted_previous) > 0:
    waste_rate_previous = total_units_wasted_previous / (total_units_sold_previous + total_units_wasted_previous)
else:
    waste_rate_previous = None

if waste_rate_analysis is not None and waste_rate_previous is not None:
    waste_rate_change = waste_rate_analysis - waste_rate_previous
    
    findings.append({
        "title": "Inventory Waste Rate",
        "claim": f"Waste rate (units wasted / total units) was {waste_rate_analysis:.4f} in analysis week vs {waste_rate_previous:.4f} in previous week, a change of {waste_rate_change:+.4f}",
        "finding_type": "inventory_efficiency",
        "metrics": {
            "waste_rate_analysis": {
                "value": round(waste_rate_analysis, 4),
                "unit": "ratio",
                "numerator": int(total_units_wasted_analysis),
                "denominator": int(total_units_sold_analysis + total_units_wasted_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_rate_previous": {
                "value": round(waste_rate_previous, 4),
                "unit": "ratio",
                "numerator": int(total_units_wasted_previous),
                "denominator": int(total_units_sold_previous + total_units_wasted_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "known_waste_cost_analysis": {
                "value": round(total_known_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_previous": {
                "value": round(total_known_waste_cost_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis period: {len(inventory_analysis)} inventory records",
            f"Previous period: {len(inventory_previous)} inventory records",
            "Waste rate calculated from units_wasted and units_sold columns",
            "Known waste cost from known_waste_cost_sar column; unknown waste preserved"
        ],
        "assumptions": [
            "Inventory records represent complete weekly snapshots",
            "units_wasted includes only known/measured waste",
            "Unknown waste is not included in calculations"
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
