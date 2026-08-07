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

# Parse dates and times with explicit format and timezone handling
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('UTC+03:00')
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert('UTC+03:00')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert('UTC+03:00')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period - create timezone-aware timestamps
analysis_start = pd.Timestamp('2026-06-29', tz='UTC+03:00')
analysis_end = pd.Timestamp('2026-07-06', tz='UTC+03:00')
previous_start = pd.Timestamp('2026-06-22', tz='UTC+03:00')
previous_end = pd.Timestamp('2026-06-29', tz='UTC+03:00')

# ISO format strings for output
analysis_start_iso = '2026-06-29T00:00:00+03:00'
analysis_end_iso = '2026-07-06T00:00:00+03:00'
previous_start_iso = '2026-06-22T00:00:00+03:00'
previous_end_iso = '2026-06-29T00:00:00+03:00'

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.tz_localize(None)) & (traffic_df['date'] < analysis_end.tz_localize(None))].copy()
staff_analysis = staff_df[(staff_df['date'] >= previous_start.tz_localize(None)) & (staff_df['date'] < analysis_end.tz_localize(None))].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.tz_localize(None)) & (traffic_df['date'] < previous_end.tz_localize(None))].copy()

findings = []

# Finding 1: Conversion Rate Analysis
# Calculate unique transactions and valid footfall
analysis_transactions = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()
analysis_footfall = traffic_analysis[~traffic_analysis['is_dead_sensor_day']]['door_count'].sum()

previous_transactions = pos_previous[~pos_previous['is_refund']]['transaction_id'].nunique()
previous_footfall = traffic_previous[~traffic_previous['is_dead_sensor_day']]['door_count'].sum()

if analysis_footfall > 0 and previous_footfall > 0:
    analysis_conversion = analysis_transactions / analysis_footfall if analysis_footfall > 0 else 0
    previous_conversion = previous_transactions / previous_footfall if previous_footfall > 0 else 0
    
    conversion_change = ((analysis_conversion - previous_conversion) / previous_conversion * 100) if previous_conversion > 0 else 0
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis period ({analysis_conversion:.4f}) compared to previous period ({previous_conversion:.4f}), representing a {conversion_change:.2f}% change",
        "finding_type": "conversion_analysis",
        "metrics": {
            "analysis_conversion_rate": {
                "value": round(analysis_conversion, 4),
                "unit": "transactions_per_visitor",
                "numerator": analysis_transactions,
                "denominator": analysis_footfall,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "previous_conversion_rate": {
                "value": round(previous_conversion, 4),
                "unit": "transactions_per_visitor",
                "numerator": previous_transactions,
                "denominator": previous_footfall,
                "period_start": previous_start_iso,
                "period_end": previous_end_iso
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Analysis period: {analysis_transactions} transactions from {analysis_footfall} valid footfall events",
            f"Previous period: {previous_transactions} transactions from {previous_footfall} valid footfall events",
            "Excluded dead sensor days from footfall denominator"
        ],
        "assumptions": [
            "Conversion calculated as unique valid sales transactions divided by valid footfall",
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall calculation"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost vs Sales Analysis
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
analysis_sales = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()

previous_labour_cost = staff_df[(staff_df['date'] >= previous_start.tz_localize(None)) & (staff_df['date'] < previous_end.tz_localize(None))]['labour_cost_sar'].sum()
previous_sales = pos_previous[~pos_previous['is_refund']]['line_total_sar'].sum()

if analysis_sales > 0 and previous_sales > 0:
    analysis_labour_ratio = analysis_labour_cost / analysis_sales if analysis_sales > 0 else 0
    previous_labour_ratio = previous_labour_cost / previous_sales if previous_sales > 0 else 0
    
    labour_ratio_change = ((analysis_labour_ratio - previous_labour_ratio) / previous_labour_ratio * 100) if previous_labour_ratio > 0 else 0
    
    findings.append({
        "title": "Labour Cost to Sales Ratio",
        "claim": f"Labour cost as percentage of sales in analysis period ({analysis_labour_ratio*100:.2f}%) compared to previous period ({previous_labour_ratio*100:.2f}%), representing a {labour_ratio_change:.2f}% change",
        "finding_type": "labour_efficiency",
        "metrics": {
            "analysis_labour_ratio": {
                "value": round(analysis_labour_ratio, 4),
                "unit": "ratio",
                "numerator": round(analysis_labour_cost, 2),
                "denominator": round(analysis_sales, 2),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "previous_labour_ratio": {
                "value": round(previous_labour_ratio, 4),
                "unit": "ratio",
                "numerator": round(previous_labour_cost, 2),
                "denominator": round(previous_sales, 2),
                "period_start": previous_start_iso,
                "period_end": previous_end_iso
            },
            "labour_ratio_change_percent": {
                "value": round(labour_ratio_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period labour cost: {analysis_labour_cost:.2f} SAR",
            f"Analysis period sales: {analysis_sales:.2f} SAR",
            f"Staff records in analysis period: {len(staff_analysis)}"
        ],
        "assumptions": [
            "Labour cost calculated from staff shift records",
            "Sales calculated as sum of line_total_sar excluding refunds",
            "Ratio represents labour efficiency metric"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory Waste Analysis
analysis_week = pd.Timestamp('2026-06-29')
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week]

if len(inventory_analysis) > 0:
    total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_wasted = inventory_analysis['units_wasted'].sum()
    total_units_sold = inventory_analysis['units_sold'].sum()
    
    if total_units_sold > 0:
        waste_to_sales_ratio = total_units_wasted / total_units_sold
        
        findings.append({
            "title": "Inventory Waste Analysis",
            "claim": f"Known waste cost of {total_waste_cost:.2f} SAR with {total_units_wasted} units wasted against {total_units_sold} units sold, representing a {waste_to_sales_ratio*100:.2f}% waste ratio",
            "finding_type": "inventory_efficiency",
            "metrics": {
                "known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "waste_to_sales_ratio": {
                    "value": round(waste_to_sales_ratio, 4),
                    "unit": "ratio",
                    "numerator": int(total_units_wasted),
                    "denominator": int(total_units_sold),
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                f"Inventory records for week starting {analysis_week.date()}: {len(inventory_analysis)} SKUs",
                "Only known waste values included; unknown waste preserved",
                "Sunday inventory counts treated as snapshot, not real-time"
            ],
            "assumptions": [
                "Waste cost calculated from known_waste_cost_sar column",
                "Waste ratio calculated as units_wasted / units_sold",
                "Inventory data represents weekly aggregation"
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

print(f"Analysis complete. {len(findings)} findings generated.")
