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

# Define timezone
tz = timezone('Asia/Riyadh')

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Define analysis periods
analysis_start = pd.Timestamp('2026-04-27', tz=tz)
analysis_end = pd.Timestamp('2026-05-04', tz=tz)
previous_start = pd.Timestamp('2026-04-20', tz=tz)
previous_end = pd.Timestamp('2026-04-27', tz=tz)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.date()) & (traffic_df['date'] < previous_end.date())].copy()

staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())].copy()

inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)].copy()
inventory_previous = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)].copy()

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
                "numerator": analysis_transactions,
                "denominator": analysis_footfall,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_conversion_rate": {
                "value": round(previous_conversion, 4),
                "unit": "transactions_per_visitor",
                "numerator": previous_transactions,
                "denominator": previous_footfall,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Analysis period: {analysis_transactions} valid transactions from {len(pos_analysis)} POS rows",
            f"Previous period: {previous_transactions} valid transactions from {len(pos_previous)} POS rows",
            f"Analysis footfall: {analysis_footfall} from {len(traffic_analysis_valid)} valid traffic records",
            f"Previous footfall: {previous_footfall} from {len(traffic_previous_valid)} valid traffic records",
            "Dead sensor days excluded from footfall calculation"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall denominator",
            "Each transaction_id represents one unique customer visit"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost vs Revenue Analysis
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
previous_labour_cost = staff_previous['labour_cost_sar'].sum()

# Calculate revenue (excluding refunds)
analysis_revenue = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
previous_revenue = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()

if analysis_revenue > 0 and previous_revenue > 0:
    analysis_labour_ratio = analysis_labour_cost / analysis_revenue if analysis_revenue > 0 else 0
    previous_labour_ratio = previous_labour_cost / previous_revenue if previous_revenue > 0 else 0
    
    labour_ratio_change = ((analysis_labour_ratio - previous_labour_ratio) / previous_labour_ratio * 100) if previous_labour_ratio > 0 else 0
    
    findings.append({
        "title": "Labour Cost to Revenue Ratio",
        "claim": f"Labour cost as percentage of revenue in analysis period ({analysis_labour_ratio*100:.2f}%) compared to previous period ({previous_labour_ratio*100:.2f}%), representing a {labour_ratio_change:.2f}% change in ratio",
        "finding_type": "labour_efficiency",
        "metrics": {
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
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period labour cost: {len(staff_analysis)} staff records, total {round(analysis_labour_cost, 2)} SAR",
            f"Previous period labour cost: {len(staff_previous)} staff records, total {round(previous_labour_cost, 2)} SAR",
            f"Analysis period revenue: {round(analysis_revenue, 2)} SAR from {analysis_transactions} transactions",
            f"Previous period revenue: {round(previous_revenue, 2)} SAR from {previous_transactions} transactions"
        ],
        "assumptions": [
            "Labour cost calculated from staff shift records with computed_duration_hours",
            "Revenue excludes refunds",
            "All staff records have valid labour_cost_sar values"
        ],
        "confidence": 0.80
    })

# Finding 3: Waste Cost Analysis
analysis_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
previous_waste_cost = inventory_previous['known_waste_cost_sar'].sum()

analysis_units_sold = inventory_analysis['units_sold'].sum()
previous_units_sold = inventory_previous['units_sold'].sum()

if analysis_units_sold > 0 and previous_units_sold > 0:
    analysis_waste_per_unit = analysis_waste_cost / analysis_units_sold if analysis_units_sold > 0 else 0
    previous_waste_per_unit = previous_waste_cost / previous_units_sold if previous_units_sold > 0 else 0
    
    waste_per_unit_change = ((analysis_waste_per_unit - previous_waste_per_unit) / previous_waste_per_unit * 100) if previous_waste_per_unit > 0 else 0
    
    findings.append({
        "title": "Known Waste Cost per Unit Sold",
        "claim": f"Known waste cost per unit sold in analysis period ({analysis_waste_per_unit:.4f} SAR) compared to previous period ({previous_waste_per_unit:.4f} SAR), representing a {waste_per_unit_change:.2f}% change",
        "finding_type": "waste_efficiency",
        "metrics": {
            "analysis_waste_per_unit": {
                "value": round(analysis_waste_per_unit, 4),
                "unit": "SAR_per_unit",
                "numerator": round(analysis_waste_cost, 2),
                "denominator": analysis_units_sold,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_waste_per_unit": {
                "value": round(previous_waste_per_unit, 4),
                "unit": "SAR_per_unit",
                "numerator": round(previous_waste_cost, 2),
                "denominator": previous_units_sold,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "waste_per_unit_change_percent": {
                "value": round(waste_per_unit_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis period: {len(inventory_analysis)} inventory records, {analysis_units_sold} units sold, {round(analysis_waste_cost, 2)} SAR known waste cost",
            f"Previous period: {len(inventory_previous)} inventory records, {previous_units_sold} units sold, {round(previous_waste_cost, 2)} SAR known waste cost",
            "Only known waste values included; unknown waste excluded per requirements"
        ],
        "assumptions": [
            "Waste cost calculated from known_waste_cost_sar column only",
            "Units sold from inventory records represent actual sales",
            "Weekly inventory data aggregated to period level"
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

print(f"Analysis complete. {len(findings)} findings generated.")
