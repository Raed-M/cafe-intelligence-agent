import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# Define periods using ISO 8601 strings and convert to UTC
# Analysis period: 2026-05-25 to 2026-06-01 (UTC+3)
analysis_start = pd.Timestamp('2026-05-25T00:00:00+03:00')
analysis_end = pd.Timestamp('2026-06-01T00:00:00+03:00')

# Previous period: 2026-05-18 to 2026-05-25 (UTC+3)
previous_start = pd.Timestamp('2026-05-18T00:00:00+03:00')
previous_end = pd.Timestamp('2026-05-25T00:00:00+03:00')

# Convert to UTC for comparison
analysis_start_utc = analysis_start.tz_convert('UTC')
analysis_end_utc = analysis_end.tz_convert('UTC')
previous_start_utc = previous_start.tz_convert('UTC')
previous_end_utc = previous_end.tz_convert('UTC')

# Convert timestamps to timezone-aware UTC
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True)
staff_df['date'] = pd.to_datetime(staff_df['date'], utc=True)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True, format='mixed')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True, format='mixed')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)

# ============================================================================
# FINDING 1: Conversion Rate Comparison (Analysis vs Previous Period)
# ============================================================================

def calculate_conversion(pos_data, traffic_data, period_start, period_end):
    """Calculate conversion rate for a period, excluding dead sensor days."""
    
    # Filter POS to period (non-refunds only)
    pos_period = pos_data[
        (pos_data['timestamp'] >= period_start) & 
        (pos_data['timestamp'] < period_end) &
        (pos_data['is_refund'] == False)
    ]
    
    # Count unique transactions
    unique_transactions = pos_period['transaction_id'].nunique()
    
    # Filter traffic to period, exclude dead sensor days
    traffic_period = traffic_data[
        (traffic_data['date'] >= period_start) & 
        (traffic_data['date'] < period_end) &
        (traffic_data['is_dead_sensor_day'] == False)
    ]
    
    # Sum door counts
    total_footfall = traffic_period['door_count'].sum()
    
    if total_footfall == 0:
        return None, None, None
    
    conversion_rate = unique_transactions / total_footfall if total_footfall > 0 else None
    
    return conversion_rate, unique_transactions, total_footfall

# Analysis period conversion
conv_analysis, trans_analysis, foot_analysis = calculate_conversion(
    pos_df, traffic_df, analysis_start_utc, analysis_end_utc
)

# Previous period conversion
conv_previous, trans_previous, foot_previous = calculate_conversion(
    pos_df, traffic_df, previous_start_utc, previous_end_utc
)

# Calculate change
if conv_analysis and conv_previous:
    conversion_change_pct = ((conv_analysis - conv_previous) / conv_previous) * 100
else:
    conversion_change_pct = None

# ============================================================================
# FINDING 2: Labour Cost Comparison (Analysis vs Previous Period)
# ============================================================================

def calculate_labour_cost(staff_data, period_start, period_end):
    """Calculate total labour cost for a period."""
    
    staff_period = staff_data[
        (staff_data['date'] >= period_start) & 
        (staff_data['date'] < period_end)
    ]
    
    total_cost = staff_period['labour_cost_sar'].sum()
    num_shifts = len(staff_period)
    
    return total_cost, num_shifts

# Analysis period labour cost
labour_analysis, shifts_analysis = calculate_labour_cost(
    staff_df, analysis_start_utc, analysis_end_utc
)

# Previous period labour cost
labour_previous, shifts_previous = calculate_labour_cost(
    staff_df, previous_start_utc, previous_end_utc
)

# Calculate change
if labour_analysis and labour_previous:
    labour_change_pct = ((labour_analysis - labour_previous) / labour_previous) * 100
else:
    labour_change_pct = None

# ============================================================================
# FINDING 3: Known Waste Cost Comparison (Analysis vs Previous Period)
# ============================================================================

def calculate_waste_cost(inventory_data, period_start, period_end):
    """Calculate known waste cost for a period."""
    
    # Map period to weeks
    inv_period = inventory_data[
        (inventory_data['week_starting'] >= period_start) & 
        (inventory_data['week_starting'] < period_end)
    ]
    
    total_waste_cost = inv_period['known_waste_cost_sar'].sum()
    num_items = len(inv_period)
    
    return total_waste_cost, num_items

# Analysis period waste cost
waste_analysis, items_analysis = calculate_waste_cost(
    inventory_df, analysis_start_utc, analysis_end_utc
)

# Previous period waste cost
waste_previous, items_previous = calculate_waste_cost(
    inventory_df, previous_start_utc, previous_end_utc
)

# Calculate change
if waste_analysis and waste_previous:
    waste_change_pct = ((waste_analysis - waste_previous) / waste_previous) * 100
else:
    waste_change_pct = None

# ============================================================================
# Build Result JSON
# ============================================================================

findings = []

# Finding 1: Conversion Rate
if conv_analysis is not None and conv_previous is not None:
    findings.append({
        "title": "Conversion Rate Comparison: Analysis vs Previous Period",
        "claim": f"Conversion rate in analysis period (2026-05-25 to 2026-06-01) was {conv_analysis:.4f} transactions per visitor, compared to {conv_previous:.4f} in previous period (2026-05-18 to 2026-05-25), representing a {conversion_change_pct:.2f}% change.",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis_period": {
                "value": float(round(conv_analysis, 4)),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_analysis),
                "denominator": int(foot_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous_period": {
                "value": float(round(conv_previous, 4)),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_previous),
                "denominator": int(foot_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_rate_change_pct": {
                "value": float(round(conversion_change_pct, 2)),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(trans_analysis + trans_previous),
        "coverage_notes": [
            "Dead sensor days excluded from traffic denominator",
            "Refunds excluded from transaction count",
            "Analysis period: 2026-05-25 to 2026-06-01",
            "Previous period: 2026-05-18 to 2026-05-25"
        ],
        "assumptions": [
            "Unique transaction_id represents one customer visit",
            "Door count is valid footfall metric",
            "is_dead_sensor_day flag is accurate"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost
if labour_analysis is not None and labour_previous is not None:
    findings.append({
        "title": "Labour Cost Comparison: Analysis vs Previous Period",
        "claim": f"Total labour cost in analysis period (2026-05-25 to 2026-06-01) was {labour_analysis:.2f} SAR, compared to {labour_previous:.2f} SAR in previous period (2026-05-18 to 2026-05-25), representing a {labour_change_pct:.2f}% change.",
        "finding_type": "cost_metric",
        "metrics": {
            "labour_cost_analysis_period": {
                "value": float(round(labour_analysis, 2)),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_previous_period": {
                "value": float(round(labour_previous, 2)),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_change_pct": {
                "value": float(round(labour_change_pct, 2)),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff"],
        "sample_size": int(shifts_analysis + shifts_previous),
        "coverage_notes": [
            "Labour cost calculated from staff shifts",
            "Analysis period: 2026-05-25 to 2026-06-01",
            "Previous period: 2026-05-18 to 2026-05-25",
            f"Analysis shifts: {shifts_analysis}, Previous shifts: {shifts_previous}"
        ],
        "assumptions": [
            "labour_cost_sar field is accurate",
            "All shifts in period are captured"
        ],
        "confidence": 0.90
    })

# Finding 3: Known Waste Cost
if waste_analysis is not None and waste_previous is not None:
    findings.append({
        "title": "Known Waste Cost Comparison: Analysis vs Previous Period",
        "claim": f"Known waste cost in analysis period (2026-05-25 to 2026-06-01) was {waste_analysis:.2f} SAR, compared to {waste_previous:.2f} SAR in previous period (2026-05-18 to 2026-05-25), representing a {waste_change_pct:.2f}% change.",
        "finding_type": "waste_metric",
        "metrics": {
            "known_waste_cost_analysis_period": {
                "value": float(round(waste_analysis, 2)),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_previous_period": {
                "value": float(round(waste_previous, 2)),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "known_waste_cost_change_pct": {
                "value": float(round(waste_change_pct, 2)),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": int(items_analysis + items_previous),
        "coverage_notes": [
            "Known waste cost only; unknown waste excluded",
            "Analysis period: 2026-05-25 to 2026-06-01",
            "Previous period: 2026-05-18 to 2026-05-25",
            f"Analysis items: {items_analysis}, Previous items: {items_previous}"
        ],
        "assumptions": [
            "known_waste_cost_sar field is accurate",
            "Week-based inventory data aligns with analysis periods"
        ],
        "confidence": 0.75
    })

# Build output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Analysis complete. Results written to {output_path}")
