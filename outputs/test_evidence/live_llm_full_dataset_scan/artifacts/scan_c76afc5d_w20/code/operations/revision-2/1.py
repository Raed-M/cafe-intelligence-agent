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

# Define periods (all in UTC+3)
tz = timezone('Asia/Riyadh')
analysis_start = tz.localize(datetime(2026, 5, 25, 0, 0, 0))
analysis_end = tz.localize(datetime(2026, 6, 1, 0, 0, 0))
previous_start = tz.localize(datetime(2026, 5, 18, 0, 0, 0))
previous_end = tz.localize(datetime(2026, 5, 25, 0, 0, 0))

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True)
staff_df['date'] = pd.to_datetime(staff_df['date'], utc=True)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True)
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
    pos_df, traffic_df, analysis_start, analysis_end
)

# Previous period conversion
conv_previous, trans_previous, foot_previous = calculate_conversion(
    pos_df, traffic_df, previous_start, previous_end
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
    staff_df, analysis_start, analysis_end
)

# Previous period labour cost
labour_previous, shifts_previous = calculate_labour_cost(
    staff_df, previous_start, previous_end
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
    inventory_df, analysis_start, analysis_end
)

# Previous period waste cost
waste_previous, items_previous = calculate_waste_cost(
    inventory_df, previous_start, previous_end
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
        "title": "Conversion Rate Decline in Analysis Period",
        "claim": f"Conversion rate decreased from {conv_previous:.4f} (previous period) to {conv_analysis:.4f} (analysis period), a {conversion_change_pct:.2f}% change.",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conv_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_analysis),
                "denominator": int(foot_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conv_previous, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_previous),
                "denominator": int(foot_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_rate_change_pct": {
                "value": round(conversion_change_pct, 2),
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
        "title": "Labour Cost Increase in Analysis Period",
        "claim": f"Total labour cost increased from {labour_previous:.2f} SAR (previous period) to {labour_analysis:.2f} SAR (analysis period), a {labour_change_pct:.2f}% change.",
        "finding_type": "cost_metric",
        "metrics": {
            "labour_cost_analysis": {
                "value": round(labour_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_previous": {
                "value": round(labour_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_change_pct": {
                "value": round(labour_change_pct, 2),
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
        "title": "Known Waste Cost Variation Between Periods",
        "claim": f"Known waste cost was {waste_analysis:.2f} SAR (analysis period) vs {waste_previous:.2f} SAR (previous period), a {waste_change_pct:.2f}% change.",
        "finding_type": "waste_metric",
        "metrics": {
            "known_waste_cost_analysis": {
                "value": round(waste_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_previous": {
                "value": round(waste_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "known_waste_cost_change_pct": {
                "value": round(waste_change_pct, 2),
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
