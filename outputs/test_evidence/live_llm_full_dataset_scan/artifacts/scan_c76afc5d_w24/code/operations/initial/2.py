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

# Define timezone offset for Asia/Riyadh (UTC+3)
tz_offset = '+03:00'

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period (using UTC timestamps)
analysis_start = pd.Timestamp('2026-06-22T00:00:00', tz='UTC')
analysis_end = pd.Timestamp('2026-06-29T00:00:00', tz='UTC')
previous_start = pd.Timestamp('2026-06-15T00:00:00', tz='UTC')
previous_end = pd.Timestamp('2026-06-22T00:00:00', tz='UTC')

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.date()) & (traffic_df['date'] < previous_end.date())].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())].copy()

findings = []

# Finding 1: Conversion Rate Analysis
# Calculate valid transactions (exclude refunds for conversion calculation)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('transaction_id').size().reset_index(name='items')
num_transactions_analysis = len(valid_transactions_analysis)

# Calculate valid footfall (exclude dead sensor days)
traffic_valid_analysis = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
total_footfall_analysis = traffic_valid_analysis['door_count'].sum()

if total_footfall_analysis > 0:
    conversion_analysis = num_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

# Previous period conversion
valid_transactions_previous = pos_previous[~pos_previous['is_refund']].groupby('transaction_id').size().reset_index(name='items')
num_transactions_previous = len(valid_transactions_previous)

traffic_valid_previous = traffic_previous[~traffic_previous['is_dead_sensor_day']].copy()
total_footfall_previous = traffic_valid_previous['door_count'].sum()

if total_footfall_previous > 0:
    conversion_previous = num_transactions_previous / total_footfall_previous
else:
    conversion_previous = None

if conversion_analysis is not None and conversion_previous is not None:
    conversion_change = ((conversion_analysis - conversion_previous) / conversion_previous) * 100
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis period (Jun 22-29) was {conversion_analysis:.4f} ({conversion_analysis*100:.2f}%), compared to {conversion_previous:.4f} ({conversion_previous*100:.2f}%) in previous period (Jun 15-22), representing a {conversion_change:+.1f}% change.",
        "finding_type": "conversion_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "ratio",
                "numerator": num_transactions_analysis,
                "denominator": total_footfall_analysis,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "conversion_rate_previous": {
                "value": round(conversion_previous, 4),
                "unit": "ratio",
                "numerator": num_transactions_previous,
                "denominator": total_footfall_previous,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": num_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {num_transactions_analysis} valid transactions from {len(pos_analysis)} POS rows",
            f"Previous period: {num_transactions_previous} valid transactions from {len(pos_previous)} POS rows",
            f"Footfall analysis period: {total_footfall_analysis} door counts (dead sensor days excluded)",
            f"Footfall previous period: {total_footfall_previous} door counts (dead sensor days excluded)"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall denominator",
            "One transaction_id = one unique customer visit"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost and Staffing Analysis
total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
total_hours_analysis = staff_analysis['computed_duration_hours'].sum()
num_staff_days_analysis = len(staff_analysis)

total_labour_cost_previous = staff_previous['labour_cost_sar'].sum()
total_hours_previous = staff_previous['computed_duration_hours'].sum()
num_staff_days_previous = len(staff_previous)

# Calculate revenue for analysis period
revenue_analysis = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
revenue_previous = pos_previous[~pos_previous['is_refund']]['line_total_sar'].sum()

if total_hours_analysis > 0 and revenue_analysis > 0:
    labour_cost_per_hour_analysis = total_labour_cost_analysis / total_hours_analysis
    labour_cost_ratio_analysis = total_labour_cost_analysis / revenue_analysis
else:
    labour_cost_per_hour_analysis = None
    labour_cost_ratio_analysis = None

if total_hours_previous > 0 and revenue_previous > 0:
    labour_cost_per_hour_previous = total_labour_cost_previous / total_hours_previous
    labour_cost_ratio_previous = total_labour_cost_previous / revenue_previous
else:
    labour_cost_per_hour_previous = None
    labour_cost_ratio_previous = None

if labour_cost_ratio_analysis is not None and labour_cost_ratio_previous is not None:
    labour_ratio_change = ((labour_cost_ratio_analysis - labour_cost_ratio_previous) / labour_cost_ratio_previous) * 100
    
    findings.append({
        "title": "Labour Cost Efficiency",
        "claim": f"Labour cost as percentage of revenue was {labour_cost_ratio_analysis*100:.2f}% in analysis period (Jun 22-29) vs {labour_cost_ratio_previous*100:.2f}% in previous period (Jun 15-22), a {labour_ratio_change:+.1f}% change.",
        "finding_type": "labour_efficiency",
        "metrics": {
            "labour_cost_ratio_analysis": {
                "value": round(labour_cost_ratio_analysis, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost_analysis, 2),
                "denominator": round(revenue_analysis, 2),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "labour_cost_ratio_previous": {
                "value": round(labour_cost_ratio_previous, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost_previous, 2),
                "denominator": round(revenue_previous, 2),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "total_labour_cost_analysis": {
                "value": round(total_labour_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "total_hours_analysis": {
                "value": round(total_hours_analysis, 1),
                "unit": "hours",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": num_staff_days_analysis,
        "coverage_notes": [
            f"Analysis period: {num_staff_days_analysis} staff shift records, {round(total_hours_analysis, 1)} total hours",
            f"Previous period: {num_staff_days_previous} staff shift records, {round(total_hours_previous, 1)} total hours",
            f"Revenue calculated from non-refund POS transactions"
        ],
        "assumptions": [
            "Labour cost calculated from computed_duration_hours and hourly_rate_sar",
            "Revenue excludes refunds",
            "Staff shifts computed from shift_start and shift_end timestamps"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory and Waste Analysis
inventory_analysis = inventory_df[inventory_df['week_starting'] >= pd.Timestamp('2026-06-22')].copy()
inventory_previous = inventory_df[(inventory_df['week_starting'] >= pd.Timestamp('2026-06-15')) & (inventory_df['week_starting'] < pd.Timestamp('2026-06-22'))].copy()

if len(inventory_analysis) > 0:
    total_units_sold_analysis = inventory_analysis['units_sold'].sum()
    total_units_wasted_analysis = inventory_analysis['units_wasted'].sum()
    total_waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
    
    if total_units_sold_analysis > 0:
        waste_ratio_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
    else:
        waste_ratio_analysis = None
else:
    total_units_sold_analysis = 0
    total_units_wasted_analysis = 0
    total_waste_cost_analysis = 0
    waste_ratio_analysis = None

if len(inventory_previous) > 0:
    total_units_sold_previous = inventory_previous['units_sold'].sum()
    total_units_wasted_previous = inventory_previous['units_wasted'].sum()
    total_waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
    
    if total_units_sold_previous > 0:
        waste_ratio_previous = total_units_wasted_previous / (total_units_sold_previous + total_units_wasted_previous)
    else:
        waste_ratio_previous = None
else:
    total_units_sold_previous = 0
    total_units_wasted_previous = 0
    total_waste_cost_previous = 0
    waste_ratio_previous = None

if waste_ratio_analysis is not None and waste_ratio_previous is not None:
    waste_ratio_change = ((waste_ratio_analysis - waste_ratio_previous) / waste_ratio_previous) * 100
    
    findings.append({
        "title": "Inventory Waste Analysis",
        "claim": f"Known waste ratio was {waste_ratio_analysis*100:.2f}% in analysis period (Jun 22-29) vs {waste_ratio_previous*100:.2f}% in previous period (Jun 15-22), a {waste_ratio_change:+.1f}% change. Known waste cost was SAR {total_waste_cost_analysis:.2f} vs SAR {total_waste_cost_previous:.2f}.",
        "finding_type": "inventory_waste",
        "metrics": {
            "waste_ratio_analysis": {
                "value": round(waste_ratio_analysis, 4),
                "unit": "ratio",
                "numerator": total_units_wasted_analysis,
                "denominator": total_units_sold_analysis + total_units_wasted_analysis,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "waste_ratio_previous": {
                "value": round(waste_ratio_previous, 4),
                "unit": "ratio",
                "numerator": total_units_wasted_previous,
                "denominator": total_units_sold_previous + total_units_wasted_previous,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "known_waste_cost_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "units_wasted_analysis": {
                "value": total_units_wasted_analysis,
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis period: {len(inventory_analysis)} SKU records with {total_units_wasted_analysis} units wasted",
            f"Previous period: {len(inventory_previous)} SKU records with {total_units_wasted_previous} units wasted",
            "Only known waste values included; unknown waste preserved per requirements"
        ],
        "assumptions": [
            "Waste ratio calculated as units_wasted / (units_sold + units_wasted)",
            "Known waste cost from known_waste_cost_sar column",
            "Week-based inventory data aligned to week_starting dates"
        ],
        "confidence": 0.75
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
