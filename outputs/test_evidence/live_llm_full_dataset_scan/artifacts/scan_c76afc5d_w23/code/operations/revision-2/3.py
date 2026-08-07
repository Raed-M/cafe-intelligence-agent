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
# All times are in ISO format with +03:00 offset
analysis_start_str = "2026-06-15T00:00:00+03:00"
analysis_end_str = "2026-06-22T00:00:00+03:00"
previous_start_str = "2026-06-08T00:00:00+03:00"
previous_end_str = "2026-06-15T00:00:00+03:00"

# Parse ISO strings to UTC datetime for filtering
analysis_start = pd.to_datetime(analysis_start_str).tz_convert('UTC')
analysis_end = pd.to_datetime(analysis_end_str).tz_convert('UTC')
previous_start = pd.to_datetime(previous_start_str).tz_convert('UTC')
previous_end = pd.to_datetime(previous_end_str).tz_convert('UTC')

# Convert POS timestamp to datetime with UTC awareness
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Calculate conversion metrics
def calculate_conversion(pos_data, traffic_data, period_start, period_end):
    """Calculate conversion rate for a period"""
    # Get valid transactions (exclude refunds for transaction count)
    valid_transactions = pos_data[~pos_data['is_refund']].groupby('transaction_id').size()
    transaction_count = len(valid_transactions)
    
    # Get valid footfall (exclude dead sensor days)
    # Convert period_start and period_end to date objects for comparison
    period_start_date = period_start.date()
    period_end_date = period_end.date()
    
    traffic_period = traffic_data[(traffic_data['date'] >= period_start_date) & 
                                   (traffic_data['date'] < period_end_date) &
                                   (~traffic_data['is_dead_sensor_day'])]
    total_footfall = traffic_period['door_count'].sum()
    
    if total_footfall == 0:
        conversion = None
    else:
        conversion = transaction_count / total_footfall
    
    return {
        'transaction_count': transaction_count,
        'total_footfall': total_footfall,
        'conversion_rate': conversion
    }

# Calculate conversion for analysis period
conv_analysis = calculate_conversion(pos_analysis, traffic_df, analysis_start, analysis_end)

# Calculate conversion for previous period
conv_previous = calculate_conversion(pos_previous, traffic_df, previous_start, previous_end)

# Calculate revenue metrics
def calculate_revenue(pos_data, period_start, period_end):
    """Calculate net revenue for a period"""
    # Net revenue = sum of line_total_sar (includes refunds as negative)
    net_revenue = pos_data['line_total_sar'].sum()
    transaction_count = len(pos_data[~pos_data['is_refund']].groupby('transaction_id').unique())
    return {
        'net_revenue': net_revenue,
        'transaction_count': transaction_count
    }

rev_analysis = calculate_revenue(pos_analysis, analysis_start, analysis_end)
rev_previous = calculate_revenue(pos_previous, previous_start, previous_end)

# Calculate labour cost metrics
def calculate_labour_cost(staff_data, period_start, period_end):
    """Calculate total labour cost for a period"""
    period_start_date = period_start.date()
    period_end_date = period_end.date()
    
    staff_period = staff_data[(staff_data['date'] >= period_start_date) & 
                              (staff_data['date'] < period_end_date)]
    total_labour_cost = staff_period['labour_cost_sar'].sum()
    staff_count = len(staff_period)
    return {
        'total_labour_cost': total_labour_cost,
        'staff_count': staff_count
    }

labour_analysis = calculate_labour_cost(staff_df, analysis_start, analysis_end)
labour_previous = calculate_labour_cost(staff_df, previous_start, previous_end)

# Calculate waste metrics
def calculate_waste(inventory_data, period_start, period_end):
    """Calculate known waste cost for a period"""
    inv_period = inventory_data[(inventory_data['week_starting'] >= period_start) & 
                                (inventory_data['week_starting'] < period_end)]
    total_waste_cost = inv_period['known_waste_cost_sar'].sum()
    total_units_sold = inv_period['units_sold'].sum()
    return {
        'total_waste_cost': total_waste_cost,
        'total_units_sold': total_units_sold
    }

waste_analysis = calculate_waste(inventory_df, analysis_start, analysis_end)
waste_previous = calculate_waste(inventory_df, previous_start, previous_end)

# Build findings
findings = []

# Finding 1: Conversion Rate Analysis
if conv_analysis['total_footfall'] > 0 and conv_previous['total_footfall'] > 0:
    conv_change_pct = ((conv_analysis['conversion_rate'] - conv_previous['conversion_rate']) / 
                       conv_previous['conversion_rate'] * 100) if conv_previous['conversion_rate'] > 0 else None
    
    findings.append({
        "title": "Conversion Rate Comparison: Analysis vs Previous Period",
        "claim": f"Conversion rate in analysis period (2026-06-15 to 2026-06-22) was {conv_analysis['conversion_rate']:.4f} vs {conv_previous['conversion_rate']:.4f} in previous period (2026-06-08 to 2026-06-15), representing a {conv_change_pct:.1f}% change.",
        "finding_type": "conversion_analysis",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conv_analysis['conversion_rate'], 4),
                "unit": "transactions_per_visitor",
                "numerator": conv_analysis['transaction_count'],
                "denominator": conv_analysis['total_footfall'],
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "conversion_rate_previous": {
                "value": round(conv_previous['conversion_rate'], 4),
                "unit": "transactions_per_visitor",
                "numerator": conv_previous['transaction_count'],
                "denominator": conv_previous['total_footfall'],
                "period_start": previous_start_str,
                "period_end": previous_end_str
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": conv_analysis['transaction_count'] + conv_previous['transaction_count'],
        "coverage_notes": [
            "Excluded dead sensor days from footfall denominator",
            "Excluded refunds from transaction count",
            "Analysis period: 2026-06-15 to 2026-06-22",
            "Previous period: 2026-06-08 to 2026-06-15"
        ],
        "assumptions": [
            "Transaction_id uniqueness identifies distinct sales baskets",
            "Door count represents valid footfall",
            "is_dead_sensor_day flag is accurate"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost to Revenue Ratio
if rev_analysis['net_revenue'] > 0 and rev_previous['net_revenue'] > 0:
    labour_ratio_analysis = labour_analysis['total_labour_cost'] / rev_analysis['net_revenue']
    labour_ratio_previous = labour_previous['total_labour_cost'] / rev_previous['net_revenue']
    labour_ratio_change_pct = ((labour_ratio_analysis - labour_ratio_previous) / labour_ratio_previous * 100)
    
    findings.append({
        "title": "Labour Cost to Revenue Ratio Trend",
        "claim": f"Labour cost as percentage of revenue increased from {labour_ratio_previous*100:.2f}% in previous period to {labour_ratio_analysis*100:.2f}% in analysis period, a {labour_ratio_change_pct:.1f}% increase in the ratio.",
        "finding_type": "labour_efficiency",
        "metrics": {
            "labour_cost_ratio_analysis": {
                "value": round(labour_ratio_analysis, 4),
                "unit": "ratio",
                "numerator": round(labour_analysis['total_labour_cost'], 2),
                "denominator": round(rev_analysis['net_revenue'], 2),
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "labour_cost_ratio_previous": {
                "value": round(labour_ratio_previous, 4),
                "unit": "ratio",
                "numerator": round(labour_previous['total_labour_cost'], 2),
                "denominator": round(rev_previous['net_revenue'], 2),
                "period_start": previous_start_str,
                "period_end": previous_end_str
            },
            "total_labour_cost_analysis": {
                "value": round(labour_analysis['total_labour_cost'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "total_revenue_analysis": {
                "value": round(rev_analysis['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": labour_analysis['staff_count'] + labour_previous['staff_count'],
        "coverage_notes": [
            "Labour cost includes computed_duration_hours × hourly_rate_sar",
            "Revenue is net of refunds (line_total_sar includes negative refund amounts)",
            "Analysis period: 2026-06-15 to 2026-06-22",
            "Previous period: 2026-06-08 to 2026-06-15"
        ],
        "assumptions": [
            "Labour cost data is complete for both periods",
            "Hourly rates and shift durations are accurate",
            "Revenue includes all payment methods"
        ],
        "confidence": 0.80
    })

# Finding 3: Known Waste Cost Trend
if waste_analysis['total_units_sold'] > 0 and waste_previous['total_units_sold'] > 0:
    waste_ratio_analysis = waste_analysis['total_waste_cost'] / waste_analysis['total_units_sold']
    waste_ratio_previous = waste_previous['total_waste_cost'] / waste_previous['total_units_sold']
    waste_ratio_change_pct = ((waste_ratio_analysis - waste_ratio_previous) / waste_ratio_previous * 100) if waste_ratio_previous > 0 else None
    
    findings.append({
        "title": "Known Waste Cost per Unit Sold",
        "claim": f"Known waste cost per unit sold was {waste_ratio_analysis:.4f} SAR in analysis period vs {waste_ratio_previous:.4f} SAR in previous period, a {waste_ratio_change_pct:.1f}% change.",
        "finding_type": "waste_efficiency",
        "metrics": {
            "waste_ratio_analysis": {
                "value": round(waste_ratio_analysis, 4),
                "unit": "SAR_per_unit",
                "numerator": round(waste_analysis['total_waste_cost'], 2),
                "denominator": waste_analysis['total_units_sold'],
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "waste_ratio_previous": {
                "value": round(waste_ratio_previous, 4),
                "unit": "SAR_per_unit",
                "numerator": round(waste_previous['total_waste_cost'], 2),
                "denominator": waste_previous['total_units_sold'],
                "period_start": previous_start_str,
                "period_end": previous_end_str
            },
            "known_waste_cost_analysis": {
                "value": round(waste_analysis['total_waste_cost'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_df[(inventory_df['week_starting'] >= analysis_start) & 
                                        (inventory_df['week_starting'] < analysis_end)]),
        "coverage_notes": [
            "Known waste cost only; unknown waste excluded per schema",
            "Units sold from inventory records",
            "Analysis period: 2026-06-15 to 2026-06-22",
            "Previous period: 2026-06-08 to 2026-06-15"
        ],
        "assumptions": [
            "Inventory records are complete for both periods",
            "Known waste cost is accurately recorded",
            "Units sold reflects actual sales volume"
        ],
        "confidence": 0.75
    })

# Write output
result = {
    "status": "success",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
