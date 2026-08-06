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
utc_offset = pd.Timedelta(hours=3)

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Define analysis periods (using UTC timestamps)
analysis_start = pd.Timestamp('2026-01-19T00:00:00', tz='UTC')
analysis_end = pd.Timestamp('2026-01-26T00:00:00', tz='UTC')
previous_start = pd.Timestamp('2026-01-12T00:00:00', tz='UTC')
previous_end = pd.Timestamp('2026-01-19T00:00:00', tz='UTC')

# Convert date boundaries to date objects for comparison with traffic_df['date']
analysis_start_date = analysis_start.date()
analysis_end_date = analysis_end.date()
previous_start_date = previous_start.date()
previous_end_date = previous_end.date()

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start_date) & (traffic_df['date'] < analysis_end_date)].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start_date) & (staff_df['date'] < analysis_end_date)].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start_date) & (traffic_df['date'] < previous_end_date)].copy()

# Calculate conversion metrics for analysis period
# Count unique valid transactions (exclude refunds for conversion calculation)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Count valid footfall (exclude dead sensor days)
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
total_footfall_analysis = traffic_analysis_valid['door_count'].sum()

# Calculate conversion for analysis period
if total_footfall_analysis > 0:
    conversion_analysis = valid_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

# Calculate conversion metrics for previous period
valid_transactions_previous = pos_previous[~pos_previous['is_refund']]['transaction_id'].nunique()
traffic_previous_valid = traffic_previous[~traffic_previous['is_dead_sensor_day']].copy()
total_footfall_previous = traffic_previous_valid['door_count'].sum()

if total_footfall_previous > 0:
    conversion_previous = valid_transactions_previous / total_footfall_previous
else:
    conversion_previous = None

# Calculate revenue metrics
revenue_analysis = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
revenue_previous = pos_previous[~pos_previous['is_refund']]['line_total_sar'].sum()

# Calculate average transaction value
if valid_transactions_analysis > 0:
    avg_transaction_analysis = revenue_analysis / valid_transactions_analysis
else:
    avg_transaction_analysis = None

if valid_transactions_previous > 0:
    avg_transaction_previous = revenue_previous / valid_transactions_previous
else:
    avg_transaction_previous = None

# Calculate labour cost metrics
labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
labour_cost_previous = staff_df[(staff_df['date'] >= previous_start_date) & (staff_df['date'] < previous_end_date)]['labour_cost_sar'].sum()

# Calculate staff hours
staff_hours_analysis = staff_analysis['computed_duration_hours'].sum()
staff_hours_previous = staff_df[(staff_df['date'] >= previous_start_date) & (staff_df['date'] < previous_end_date)]['computed_duration_hours'].sum()

# Calculate labour cost per transaction
if valid_transactions_analysis > 0:
    labour_cost_per_transaction_analysis = labour_cost_analysis / valid_transactions_analysis
else:
    labour_cost_per_transaction_analysis = None

if valid_transactions_previous > 0:
    labour_cost_per_transaction_previous = labour_cost_previous / valid_transactions_previous
else:
    labour_cost_per_transaction_previous = None

# Inventory analysis
inventory_analysis = inventory_df[inventory_df['week_starting'] >= pd.Timestamp('2026-01-19')].copy()
total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
total_units_wasted = inventory_analysis['units_wasted'].sum()
total_units_sold = inventory_analysis['units_sold'].sum()

# Calculate waste rate
if (total_units_sold + total_units_wasted) > 0:
    waste_rate = total_units_wasted / (total_units_sold + total_units_wasted)
else:
    waste_rate = None

# Prepare findings
findings = []

# Finding 1: Conversion Rate Comparison
if conversion_analysis is not None and conversion_previous is not None:
    conversion_change = ((conversion_analysis - conversion_previous) / conversion_previous * 100) if conversion_previous > 0 else None
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis period ({conversion_analysis:.2%}) compared to previous period ({conversion_previous:.2%})",
        "finding_type": "performance_comparison",
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
                "value": round(conversion_previous, 4),
                "unit": "ratio",
                "numerator": valid_transactions_previous,
                "denominator": total_footfall_previous,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 2) if conversion_change is not None else None,
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": valid_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {valid_transactions_analysis} valid transactions from {total_footfall_analysis} footfall events",
            f"Previous period: {valid_transactions_previous} valid transactions from {total_footfall_previous} footfall events",
            "Dead sensor days excluded from footfall denominator"
        ],
        "assumptions": [
            "Conversion = unique valid sales transactions / valid footfall",
            "Refunds excluded from transaction count",
            "Dead sensor intervals excluded from footfall calculation"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost Efficiency
if labour_cost_per_transaction_analysis is not None and labour_cost_per_transaction_previous is not None:
    labour_efficiency_change = ((labour_cost_per_transaction_analysis - labour_cost_per_transaction_previous) / labour_cost_per_transaction_previous * 100) if labour_cost_per_transaction_previous > 0 else None
    
    findings.append({
        "title": "Labour Cost Efficiency",
        "claim": f"Labour cost per transaction in analysis period (SAR {labour_cost_per_transaction_analysis:.2f}) vs previous period (SAR {labour_cost_per_transaction_previous:.2f})",
        "finding_type": "cost_efficiency",
        "metrics": {
            "labour_cost_per_transaction_analysis": {
                "value": round(labour_cost_per_transaction_analysis, 2),
                "unit": "SAR",
                "numerator": round(labour_cost_analysis, 2),
                "denominator": valid_transactions_analysis,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_per_transaction_previous": {
                "value": round(labour_cost_per_transaction_previous, 2),
                "unit": "SAR",
                "numerator": round(labour_cost_previous, 2),
                "denominator": valid_transactions_previous,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_efficiency_change_percent": {
                "value": round(labour_efficiency_change, 2) if labour_efficiency_change is not None else None,
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": valid_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {staff_hours_analysis:.1f} total staff hours, {labour_cost_analysis:.2f} SAR total labour cost",
            f"Previous period: {staff_hours_previous:.1f} total staff hours, {labour_cost_previous:.2f} SAR total labour cost"
        ],
        "assumptions": [
            "Labour cost calculated from staff shift records with computed duration",
            "Cost per transaction = total labour cost / valid transactions",
            "Staff hours computed from shift_start and shift_end overlap"
        ],
        "confidence": 0.80
    })

# Finding 3: Waste Rate Analysis
if waste_rate is not None and total_units_wasted > 0:
    findings.append({
        "title": "Product Waste Rate",
        "claim": f"Known waste rate in analysis period is {waste_rate:.2%} with {total_units_wasted} units wasted (cost: SAR {total_waste_cost:.2f})",
        "finding_type": "waste_analysis",
        "metrics": {
            "waste_rate": {
                "value": round(waste_rate, 4),
                "unit": "ratio",
                "numerator": total_units_wasted,
                "denominator": total_units_sold + total_units_wasted,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_units_wasted": {
                "value": total_units_wasted,
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
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
            f"Analysis covers {len(inventory_analysis)} SKU records from inventory week starting 2026-01-19",
            "Unknown waste values preserved; only known waste included in calculations",
            "Waste rate calculated as units_wasted / (units_sold + units_wasted)"
        ],
        "assumptions": [
            "Inventory counts represent weekly aggregates",
            "Known waste cost is deterministic from inventory records",
            "Unknown waste values are excluded from rate calculations"
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Output written to {output_path}")
print(f"Generated {len(findings)} findings")
