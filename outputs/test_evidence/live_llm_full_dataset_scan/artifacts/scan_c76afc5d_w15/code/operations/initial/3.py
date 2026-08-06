import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, timedelta as td

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

# Define periods using fixed offset for Asia/Riyadh (UTC+3)
riyadh_offset = timezone(timedelta(hours=3))
analysis_start = datetime(2026, 4, 20, 0, 0, 0, tzinfo=riyadh_offset)
analysis_end = datetime(2026, 4, 27, 0, 0, 0, tzinfo=riyadh_offset)
prev_start = datetime(2026, 4, 13, 0, 0, 0, tzinfo=riyadh_offset)
prev_end = datetime(2026, 4, 20, 0, 0, 0, tzinfo=riyadh_offset)

findings = []

# Helper function to convert numpy types to Python native types
def convert_to_native(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    return obj

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Convert timestamp to timezone-aware datetime
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(riyadh_offset)
traffic_df['date_dt'] = pd.to_datetime(traffic_df['date'], utc=True).dt.tz_convert(riyadh_offset)

# Filter analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_dt'] >= analysis_start) & 
    (pos_df['timestamp_dt'] < analysis_end)
].copy()

traffic_analysis = traffic_df[
    (traffic_df['date_dt'] >= analysis_start) & 
    (traffic_df['date_dt'] < analysis_end)
].copy()

# Count valid transactions (exclude refunds, use transaction_id)
valid_transactions_analysis = int(pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique())

# Count valid footfall (exclude dead sensor days)
valid_traffic_analysis = int(traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['door_count'].sum())

# Previous period
pos_prev = pos_df[
    (pos_df['timestamp_dt'] >= prev_start) & 
    (pos_df['timestamp_dt'] < prev_end)
].copy()

traffic_prev = traffic_df[
    (traffic_df['date_dt'] >= prev_start) & 
    (traffic_df['date_dt'] < prev_end)
].copy()

valid_transactions_prev = int(pos_prev[pos_prev['is_refund'] == False]['transaction_id'].nunique())
valid_traffic_prev = int(traffic_prev[traffic_prev['is_dead_sensor_day'] == False]['door_count'].sum())

# Calculate conversion rates
if valid_traffic_analysis > 0:
    conversion_analysis = valid_transactions_analysis / valid_traffic_analysis
else:
    conversion_analysis = None

if valid_traffic_prev > 0:
    conversion_prev = valid_transactions_prev / valid_traffic_prev
else:
    conversion_prev = None

# Check if we have sufficient data
traffic_analysis_valid_days = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['date_dt'].dt.date.nunique()
traffic_prev_valid_days = traffic_prev[traffic_prev['is_dead_sensor_day'] == False]['date_dt'].dt.date.nunique()

if conversion_analysis is not None and conversion_prev is not None and traffic_analysis_valid_days >= 5:
    conversion_change = conversion_analysis - conversion_prev
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis week (Apr 20-27) was {conversion_analysis:.4f} vs {conversion_prev:.4f} in previous week, a change of {conversion_change:+.4f}",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": valid_transactions_analysis,
                "denominator": valid_traffic_analysis,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conversion_prev, 4),
                "unit": "transactions_per_visitor",
                "numerator": valid_transactions_prev,
                "denominator": valid_traffic_prev,
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
        "sample_size": valid_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {int(traffic_analysis_valid_days)} valid traffic days (dead sensor days excluded)",
            f"Previous period: {int(traffic_prev_valid_days)} valid traffic days (dead sensor days excluded)",
            f"Transactions counted using transaction_id (refunds excluded)"
        ],
        "assumptions": [
            "Dead sensor days excluded from traffic denominator",
            "Refunds excluded from transaction count",
            "Timezone: Asia/Riyadh (UTC+3)"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Revenue Analysis
# ============================================================================

# Convert staff dates
staff_df['date_dt'] = pd.to_datetime(staff_df['date'], utc=True).dt.tz_convert(riyadh_offset)

# Filter staff for analysis period
staff_analysis = staff_df[
    (staff_df['date_dt'] >= analysis_start) & 
    (staff_df['date_dt'] < analysis_end)
].copy()

staff_prev = staff_df[
    (staff_df['date_dt'] >= prev_start) & 
    (staff_df['date_dt'] < prev_end)
].copy()

# Calculate total labour cost
labour_cost_analysis = float(staff_analysis['labour_cost_sar'].sum())
labour_cost_prev = float(staff_prev['labour_cost_sar'].sum())

# Calculate revenue (net of refunds and discounts)
revenue_analysis = float(pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum())
revenue_prev = float(pos_prev[pos_prev['is_refund'] == False]['line_total_sar'].sum())

# Calculate labour cost as % of revenue
if revenue_analysis > 0:
    labour_pct_analysis = (labour_cost_analysis / revenue_analysis) * 100
else:
    labour_pct_analysis = None

if revenue_prev > 0:
    labour_pct_prev = (labour_cost_prev / revenue_prev) * 100
else:
    labour_pct_prev = None

staff_analysis_days = int(staff_analysis['date_dt'].dt.date.nunique())
staff_prev_days = int(staff_prev['date_dt'].dt.date.nunique())

if labour_pct_analysis is not None and labour_pct_prev is not None and staff_analysis_days >= 5:
    labour_pct_change = labour_pct_analysis - labour_pct_prev
    
    findings.append({
        "title": "Labour Cost Efficiency",
        "claim": f"Labour cost as % of revenue was {labour_pct_analysis:.2f}% in analysis week vs {labour_pct_prev:.2f}% in previous week, a change of {labour_pct_change:+.2f} percentage points",
        "finding_type": "cost_efficiency",
        "metrics": {
            "labour_cost_pct_analysis": {
                "value": round(labour_pct_analysis, 2),
                "unit": "percent",
                "numerator": round(labour_cost_analysis, 2),
                "denominator": round(revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_pct_previous": {
                "value": round(labour_pct_prev, 2),
                "unit": "percent",
                "numerator": round(labour_cost_prev, 2),
                "denominator": round(revenue_prev, 2),
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            },
            "labour_cost_pct_change": {
                "value": round(labour_pct_change, 2),
                "unit": "percentage_points",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": staff_analysis_days,
        "coverage_notes": [
            f"Analysis period: {staff_analysis_days} days with staff data",
            f"Previous period: {staff_prev_days} days with staff data",
            f"Revenue calculated from non-refund transactions",
            f"Labour cost from staff.labour_cost_sar"
        ],
        "assumptions": [
            "Labour cost includes computed_duration_hours * hourly_rate_sar",
            "Revenue excludes refunds and includes discounts as negative",
            "Timezone: Asia/Riyadh (UTC+3)"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Filter inventory for analysis week
inventory_df['week_starting_dt'] = pd.to_datetime(inventory_df['week_starting'], utc=True).dt.tz_convert(riyadh_offset)

inventory_analysis = inventory_df[
    (inventory_df['week_starting_dt'] >= analysis_start) & 
    (inventory_df['week_starting_dt'] < analysis_end)
].copy()

inventory_prev = inventory_df[
    (inventory_df['week_starting_dt'] >= prev_start) & 
    (inventory_df['week_starting_dt'] < prev_end)
].copy()

# Calculate waste metrics
total_waste_cost_analysis = float(inventory_analysis['known_waste_cost_sar'].sum())
total_waste_cost_prev = float(inventory_prev['known_waste_cost_sar'].sum())

total_units_sold_analysis = int(inventory_analysis['units_sold'].sum())
total_units_sold_prev = int(inventory_prev['units_sold'].sum())

total_units_wasted_analysis = int(inventory_analysis['units_wasted'].sum())
total_units_wasted_prev = int(inventory_prev['units_wasted'].sum())

# Calculate waste rate
if (total_units_sold_analysis + total_units_wasted_analysis) > 0:
    waste_rate_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
else:
    waste_rate_analysis = None

if (total_units_sold_prev + total_units_wasted_prev) > 0:
    waste_rate_prev = total_units_wasted_prev / (total_units_sold_prev + total_units_wasted_prev)
else:
    waste_rate_prev = None

inventory_analysis_skus = len(inventory_analysis)
inventory_prev_skus = len(inventory_prev)

if waste_rate_analysis is not None and waste_rate_prev is not None and inventory_analysis_skus > 0:
    waste_rate_change = waste_rate_analysis - waste_rate_prev
    
    findings.append({
        "title": "Inventory Waste Rate",
        "claim": f"Waste rate (units wasted / total units) was {waste_rate_analysis:.4f} ({waste_rate_analysis*100:.2f}%) in analysis week vs {waste_rate_prev:.4f} ({waste_rate_prev*100:.2f}%) in previous week, a change of {waste_rate_change:+.4f}",
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
            "waste_cost_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_cost_previous": {
                "value": round(total_waste_cost_prev, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": prev_start.isoformat(),
                "period_end": prev_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": inventory_analysis_skus,
        "coverage_notes": [
            f"Analysis period: {inventory_analysis_skus} SKUs tracked",
            f"Previous period: {inventory_prev_skus} SKUs tracked",
            f"Waste cost calculated from known_waste_cost_sar (unknown waste excluded)",
            f"Units wasted includes only known waste values"
        ],
        "assumptions": [
            "Inventory counts are weekly snapshots (week_starting date)",
            "Known waste cost is deterministic; unknown waste excluded per requirements",
            "Waste rate = units_wasted / (units_sold + units_wasted)"
        ],
        "confidence": 0.75
    })

# ============================================================================
# Write output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
