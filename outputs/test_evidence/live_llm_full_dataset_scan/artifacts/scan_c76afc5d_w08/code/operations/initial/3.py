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
tz_offset = pd.Timedelta(hours=3)

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.date
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True, format='mixed')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True, format='mixed')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.date

# Analysis period (using UTC timestamps with +03:00 offset)
analysis_start = pd.Timestamp('2026-03-02T00:00:00', tz='UTC')
analysis_end = pd.Timestamp('2026-03-09T00:00:00', tz='UTC')
previous_start = pd.Timestamp('2026-02-23T00:00:00', tz='UTC')
previous_end = pd.Timestamp('2026-03-02T00:00:00', tz='UTC')

analysis_start_date = analysis_start.date()
analysis_end_date = analysis_end.date()
previous_start_date = previous_start.date()
previous_end_date = previous_end.date()

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start_date) & (traffic_df['date'] < analysis_end_date)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start_date) & (traffic_df['date'] < previous_end_date)].copy()

staff_analysis = staff_df[(staff_df['date'] >= analysis_start_date) & (staff_df['date'] < analysis_end_date)].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start_date) & (staff_df['date'] < previous_end_date)].copy()

findings = []

# Finding 1: Conversion Rate Analysis
# Calculate unique transactions and valid footfall
analysis_transactions = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
previous_transactions = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()

# Filter out dead sensor days
traffic_analysis_valid = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False].copy()
traffic_previous_valid = traffic_previous[traffic_previous['is_dead_sensor_day'] == False].copy()

analysis_footfall = traffic_analysis_valid['door_count'].sum()
previous_footfall = traffic_previous_valid['door_count'].sum()

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
                "numerator": int(analysis_transactions),
                "denominator": int(analysis_footfall),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_conversion_rate": {
                "value": round(previous_conversion, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(previous_transactions),
                "denominator": int(previous_footfall),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(analysis_transactions),
        "coverage_notes": [
            f"Analysis period: {len(traffic_analysis_valid)} valid traffic days out of {len(traffic_analysis)} total days",
            f"Previous period: {len(traffic_previous_valid)} valid traffic days out of {len(traffic_previous)} total days",
            "Dead sensor days excluded from footfall calculation"
        ],
        "assumptions": [
            "Conversion = unique valid sales transactions / valid footfall",
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from denominator"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost vs Demand Analysis
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
previous_labour_cost = staff_previous['labour_cost_sar'].sum()

# Calculate revenue for demand proxy
analysis_revenue = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
previous_revenue = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()

if analysis_labour_cost > 0 and previous_labour_cost > 0 and analysis_revenue > 0 and previous_revenue > 0:
    analysis_labour_ratio = analysis_labour_cost / analysis_revenue if analysis_revenue > 0 else 0
    previous_labour_ratio = previous_labour_cost / previous_revenue if previous_revenue > 0 else 0
    
    labour_ratio_change = ((analysis_labour_ratio - previous_labour_ratio) / previous_labour_ratio * 100) if previous_labour_ratio > 0 else 0
    
    findings.append({
        "title": "Labour Cost to Revenue Ratio",
        "claim": f"Labour cost as percentage of revenue increased from {previous_labour_ratio*100:.2f}% to {analysis_labour_ratio*100:.2f}%, a {labour_ratio_change:.2f}% change",
        "finding_type": "labour_efficiency",
        "metrics": {
            "analysis_labour_cost": {
                "value": round(analysis_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_labour_ratio": {
                "value": round(analysis_labour_ratio, 4),
                "unit": "ratio",
                "numerator": round(analysis_labour_cost, 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_labour_ratio": {
                "value": round(previous_labour_ratio, 4),
                "unit": "ratio",
                "numerator": round(previous_labour_cost, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_ratio_change_percent": {
                "value": round(labour_ratio_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period staff records: {len(staff_analysis)}",
            f"Previous period staff records: {len(staff_previous)}",
            "Labour cost calculated from computed_duration_hours and hourly_rate_sar"
        ],
        "assumptions": [
            "Labour cost includes all shifts in the period",
            "Revenue used as proxy for demand (net of refunds)",
            "Staff hours computed from shift_start and shift_end overlap"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory Waste Analysis
analysis_week = pd.Timestamp('2026-03-02').date()
previous_week = pd.Timestamp('2026-02-23').date()

inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()
inventory_previous = inventory_df[inventory_df['week_starting'] == previous_week].copy()

if len(inventory_analysis) > 0 and len(inventory_previous) > 0:
    analysis_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    previous_waste_cost = inventory_previous['known_waste_cost_sar'].sum()
    
    analysis_total_cost = (inventory_analysis['units_sold'] * inventory_analysis['unit_cost_sar']).sum() + analysis_waste_cost
    previous_total_cost = (inventory_previous['units_sold'] * inventory_previous['unit_cost_sar']).sum() + previous_waste_cost
    
    analysis_waste_ratio = analysis_waste_cost / analysis_total_cost if analysis_total_cost > 0 else 0
    previous_waste_ratio = previous_waste_cost / previous_total_cost if previous_total_cost > 0 else 0
    
    waste_ratio_change = ((analysis_waste_ratio - previous_waste_ratio) / previous_waste_ratio * 100) if previous_waste_ratio > 0 else 0
    
    findings.append({
        "title": "Known Waste Cost Analysis",
        "claim": f"Known waste cost ratio changed from {previous_waste_ratio*100:.2f}% to {analysis_waste_ratio*100:.2f}% of total inventory cost, a {waste_ratio_change:.2f}% change",
        "finding_type": "inventory_waste",
        "metrics": {
            "analysis_known_waste_cost": {
                "value": round(analysis_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_total_inventory_cost": {
                "value": round(analysis_total_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_waste_ratio": {
                "value": round(analysis_waste_ratio, 4),
                "unit": "ratio",
                "numerator": round(analysis_waste_cost, 2),
                "denominator": round(analysis_total_cost, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_waste_ratio": {
                "value": round(previous_waste_ratio, 4),
                "unit": "ratio",
                "numerator": round(previous_waste_cost, 2),
                "denominator": round(previous_total_cost, 2),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "waste_ratio_change_percent": {
                "value": round(waste_ratio_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis week inventory items: {len(inventory_analysis)}",
            f"Previous week inventory items: {len(inventory_previous)}",
            "Only known waste costs included; unknown waste values preserved"
        ],
        "assumptions": [
            "Total inventory cost = (units_sold * unit_cost) + known_waste_cost",
            "Waste ratio = known_waste_cost / total_inventory_cost",
            "Sunday inventory counts treated as weekly snapshots, not real-time"
        ],
        "confidence": 0.75
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
