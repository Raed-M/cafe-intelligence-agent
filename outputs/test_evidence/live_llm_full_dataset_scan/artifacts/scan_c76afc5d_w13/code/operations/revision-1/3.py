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

# Define periods using UTC+3 offset (no pytz)
utc_plus_3 = pd.Timedelta(hours=3)

def make_aware(dt):
    """Create timezone-aware datetime with UTC+3 offset"""
    return pd.Timestamp(dt, tz='UTC').tz_convert('UTC+03:00')

analysis_start = make_aware(datetime(2026, 4, 6, 0, 0, 0))
analysis_end = make_aware(datetime(2026, 4, 13, 0, 0, 0))
previous_start = make_aware(datetime(2026, 3, 30, 0, 0, 0))
previous_end = make_aware(datetime(2026, 4, 6, 0, 0, 0))

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('UTC+03:00')
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize('UTC+03:00')
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize('UTC+03:00')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize('UTC+03:00')

# Filter POS for analysis and previous periods
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter traffic for analysis and previous periods
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)].copy()

# Filter staff for analysis and previous periods
staff_analysis = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start) & (staff_df['date'] < previous_end)].copy()

# Filter inventory for analysis and previous periods
inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)].copy()
inventory_previous = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)].copy()

findings = []

# Helper function to convert numpy types to Python native types
def to_native(val):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        return float(val)
    if isinstance(val, bool):
        return bool(val)
    return val

# ============================================================================
# FINDING 1: Conversion Rate Analysis
# ============================================================================

# Analysis period conversion
valid_sales_analysis = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
dead_sensor_days_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == True]['date'].nunique()
traffic_analysis_valid = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False].copy()
total_footfall_analysis = traffic_analysis_valid['door_count'].sum()

if total_footfall_analysis > 0:
    conversion_analysis = valid_sales_analysis / total_footfall_analysis
else:
    conversion_analysis = None

# Previous period conversion
valid_sales_previous = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()
dead_sensor_days_previous = traffic_previous[traffic_previous['is_dead_sensor_day'] == True]['date'].nunique()
traffic_previous_valid = traffic_previous[traffic_previous['is_dead_sensor_day'] == False].copy()
total_footfall_previous = traffic_previous_valid['door_count'].sum()

if total_footfall_previous > 0:
    conversion_previous = valid_sales_previous / total_footfall_previous
else:
    conversion_previous = None

if conversion_analysis is not None and conversion_previous is not None:
    conversion_change = conversion_analysis - conversion_previous
    conversion_change_pct = (conversion_change / conversion_previous * 100) if conversion_previous > 0 else None
    
    if conversion_change_pct is not None and abs(conversion_change_pct) > 5:  # Only report if meaningful change
        findings.append({
            "title": "Conversion Rate Change Week-over-Week",
            "claim": f"Conversion rate changed from {conversion_previous:.4f} ({conversion_previous*100:.2f}%) to {conversion_analysis:.4f} ({conversion_analysis*100:.2f}%), a change of {conversion_change_pct:.1f}%.",
            "finding_type": "conversion_metric",
            "metrics": {
                "conversion_previous": {
                    "value": round(conversion_previous, 4),
                    "unit": "ratio",
                    "numerator": to_native(valid_sales_previous),
                    "denominator": to_native(total_footfall_previous),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "conversion_analysis": {
                    "value": round(conversion_analysis, 4),
                    "unit": "ratio",
                    "numerator": to_native(valid_sales_analysis),
                    "denominator": to_native(total_footfall_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "conversion_change_pct": {
                    "value": round(conversion_change_pct, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "traffic"],
            "sample_size": to_native(total_footfall_analysis),
            "coverage_notes": [
                f"Analysis period: {to_native(dead_sensor_days_analysis)} dead sensor days excluded from {len(traffic_analysis)} total traffic records",
                f"Previous period: {to_native(dead_sensor_days_previous)} dead sensor days excluded from {len(traffic_previous)} total traffic records",
                f"Valid sales transactions (non-refunds) used for numerator"
            ],
            "assumptions": [
                "Conversion = unique valid sales transactions / valid footfall (excluding dead sensor intervals)",
                "Refunds excluded from transaction count",
                "Footfall from non-dead-sensor days only"
            ],
            "confidence": 0.85
        })

# ============================================================================
# FINDING 2: Labour Cost as Percentage of Revenue
# ============================================================================

# Analysis period
revenue_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
labour_cost_pct_analysis = (labour_cost_analysis / revenue_analysis * 100) if revenue_analysis > 0 else None

# Previous period
revenue_previous = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()
labour_cost_previous = staff_previous['labour_cost_sar'].sum()
labour_cost_pct_previous = (labour_cost_previous / revenue_previous * 100) if revenue_previous > 0 else None

if labour_cost_pct_analysis is not None and labour_cost_pct_previous is not None:
    labour_cost_pct_change = labour_cost_pct_analysis - labour_cost_pct_previous
    
    # Determine direction of change
    if labour_cost_pct_change < 0:
        direction = "decreased"
    else:
        direction = "increased"
    
    findings.append({
        "title": f"Labour Cost Ratio {direction.capitalize()} Week-over-Week",
        "claim": f"Labour cost as percentage of revenue {direction} from {labour_cost_pct_previous:.1f}% to {labour_cost_pct_analysis:.1f}%, a change of {labour_cost_pct_change:.1f} percentage points.",
        "finding_type": "labour_efficiency_metric",
        "metrics": {
            "labour_cost_pct_previous": {
                "value": round(labour_cost_pct_previous, 1),
                "unit": "percent",
                "numerator": round(float(labour_cost_previous), 2),
                "denominator": round(float(revenue_previous), 2),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_pct_analysis": {
                "value": round(labour_cost_pct_analysis, 1),
                "unit": "percent",
                "numerator": round(float(labour_cost_analysis), 2),
                "denominator": round(float(revenue_analysis), 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_pct_change": {
                "value": round(labour_cost_pct_change, 1),
                "unit": "percentage_points",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": to_native(staff_analysis.shape[0]),
        "coverage_notes": [
            f"Analysis period staff records: {staff_analysis.shape[0]}",
            f"Previous period staff records: {staff_previous.shape[0]}",
            "Labour cost includes all shifts in period with computed_duration_hours",
            "Revenue calculated from non-refund POS transactions"
        ],
        "assumptions": [
            "Labour cost ratio = total labour cost / total revenue (non-refunds)",
            "No imputation for missing staff records",
            "Comparison periods may have different day-of-week distributions"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Known Waste Rate Analysis
# ============================================================================

# Analysis period
if len(inventory_analysis) > 0:
    waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
    units_sold_analysis = inventory_analysis['units_sold'].sum()
    units_wasted_analysis = inventory_analysis['units_wasted'].sum()
    
    if (units_sold_analysis + units_wasted_analysis) > 0:
        waste_rate_analysis = units_wasted_analysis / (units_sold_analysis + units_wasted_analysis)
    else:
        waste_rate_analysis = None
else:
    waste_cost_analysis = 0
    waste_rate_analysis = None

# Previous period
if len(inventory_previous) > 0:
    waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
    units_sold_previous = inventory_previous['units_sold'].sum()
    units_wasted_previous = inventory_previous['units_wasted'].sum()
    
    if (units_sold_previous + units_wasted_previous) > 0:
        waste_rate_previous = units_wasted_previous / (units_sold_previous + units_wasted_previous)
    else:
        waste_rate_previous = None
else:
    waste_cost_previous = 0
    waste_rate_previous = None

if waste_rate_analysis is not None and waste_rate_previous is not None:
    waste_rate_change = waste_rate_analysis - waste_rate_previous
    waste_rate_change_pct = (waste_rate_change / waste_rate_previous * 100) if waste_rate_previous > 0 else None
    
    # Determine direction of change
    if waste_rate_change < 0:
        direction = "decreased"
    else:
        direction = "increased"
    
    findings.append({
        "title": f"Known Product Waste Rate {direction.capitalize()}",
        "claim": f"Known waste rate {direction} from {waste_rate_previous*100:.1f}% to {waste_rate_analysis*100:.1f}%, with known waste cost of {waste_cost_analysis:.2f} SAR in analysis period.",
        "finding_type": "waste_metric",
        "metrics": {
            "waste_rate_previous": {
                "value": round(waste_rate_previous, 4),
                "unit": "ratio",
                "numerator": to_native(units_wasted_previous),
                "denominator": to_native(units_sold_previous + units_wasted_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "waste_rate_analysis": {
                "value": round(waste_rate_analysis, 4),
                "unit": "ratio",
                "numerator": to_native(units_wasted_analysis),
                "denominator": to_native(units_sold_analysis + units_wasted_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_cost_analysis": {
                "value": round(float(waste_cost_analysis), 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis period inventory records: {len(inventory_analysis)} SKUs",
            f"Previous period inventory records: {len(inventory_previous)} SKUs",
            "Waste rate = units_wasted / (units_sold + units_wasted)",
            "Known waste cost from inventory.known_waste_cost_sar column",
            "Unknown waste values preserved; only known waste included"
        ],
        "assumptions": [
            "Inventory counts reflect weekly aggregates, not real-time stock",
            "Known waste is deterministic; unknown waste excluded from rate calculation",
            "Units sold and wasted are accurate for the week"
        ],
        "confidence": 0.75
    })

# ============================================================================
# Write output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
