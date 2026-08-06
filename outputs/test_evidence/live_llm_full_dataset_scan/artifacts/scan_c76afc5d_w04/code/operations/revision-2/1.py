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

# Define periods
tz = timezone('Asia/Riyadh')
analysis_start = tz.localize(datetime(2026, 2, 2, 0, 0, 0))
analysis_end = tz.localize(datetime(2026, 2, 9, 0, 0, 0))
previous_start = tz.localize(datetime(2026, 1, 26, 0, 0, 0))
previous_end = tz.localize(datetime(2026, 2, 2, 0, 0, 0))

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)]
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)]

staff_analysis = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)]
staff_previous = staff_df[(staff_df['date'] >= previous_start) & (staff_df['date'] < previous_end)]

inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)]
inventory_previous = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)]

findings = []

# Finding 1: Conversion Rate Analysis
# Calculate unique transactions and valid footfall
analysis_transactions = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()
previous_transactions = pos_previous[~pos_previous['is_refund']]['transaction_id'].nunique()

# Filter out dead sensor days from traffic
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']]
traffic_previous_valid = traffic_previous[~traffic_previous['is_dead_sensor_day']]

analysis_footfall = traffic_analysis_valid['door_count'].sum()
previous_footfall = traffic_previous_valid['door_count'].sum()

if analysis_footfall > 0 and previous_footfall > 0:
    analysis_conversion = analysis_transactions / analysis_footfall
    previous_conversion = previous_transactions / previous_footfall
    conversion_change = ((analysis_conversion - previous_conversion) / previous_conversion) * 100
    
    findings.append({
        "title": "Conversion Rate Change",
        "claim": f"Conversion rate changed by {conversion_change:.1f}% from previous period to analysis period",
        "finding_type": "performance_metric",
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
            "conversion_rate_change_pct": {
                "value": round(conversion_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Analysis period: {analysis_transactions} valid transactions from {len(pos_analysis)} POS rows",
            f"Previous period: {previous_transactions} valid transactions from {len(pos_previous)} POS rows",
            f"Analysis footfall: {analysis_footfall} from {len(traffic_analysis_valid)} valid traffic days",
            f"Previous footfall: {previous_footfall} from {len(traffic_previous_valid)} valid traffic days",
            "Excluded dead sensor days from traffic denominators"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall calculation",
            "Each transaction_id represents one unique customer visit"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost and Staffing Analysis
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
previous_labour_cost = staff_previous['labour_cost_sar'].sum()

analysis_staff_days = staff_analysis['date'].nunique()
previous_staff_days = staff_previous['date'].nunique()

if analysis_staff_days > 0 and previous_staff_days > 0:
    analysis_daily_labour = analysis_labour_cost / analysis_staff_days
    previous_daily_labour = previous_labour_cost / previous_staff_days
    labour_cost_change = ((analysis_daily_labour - previous_daily_labour) / previous_daily_labour) * 100
    
    findings.append({
        "title": "Daily Labour Cost Change",
        "claim": f"Average daily labour cost changed by {labour_cost_change:.1f}% from previous period to analysis period",
        "finding_type": "cost_metric",
        "metrics": {
            "analysis_daily_labour_cost": {
                "value": round(analysis_daily_labour, 2),
                "unit": "SAR",
                "numerator": round(analysis_labour_cost, 2),
                "denominator": analysis_staff_days,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_daily_labour_cost": {
                "value": round(previous_daily_labour, 2),
                "unit": "SAR",
                "numerator": round(previous_labour_cost, 2),
                "denominator": previous_staff_days,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_change_pct": {
                "value": round(labour_cost_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period: {len(staff_analysis)} staff records across {analysis_staff_days} days",
            f"Previous period: {len(staff_previous)} staff records across {previous_staff_days} days",
            "Labour cost calculated from computed_duration_hours and hourly_rate_sar"
        ],
        "assumptions": [
            "Staff records represent actual shifts worked",
            "Hourly rates are consistent within each period",
            "No imputation for missing staff records"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory Waste Analysis
if len(inventory_analysis) > 0 and len(inventory_previous) > 0:
    analysis_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    previous_waste_cost = inventory_previous['known_waste_cost_sar'].sum()
    
    analysis_total_cost = (inventory_analysis['units_sold'] * inventory_analysis['unit_cost_sar']).sum() + analysis_waste_cost
    previous_total_cost = (inventory_previous['units_sold'] * inventory_previous['unit_cost_sar']).sum() + previous_waste_cost
    
    if analysis_total_cost > 0 and previous_total_cost > 0:
        analysis_waste_pct = (analysis_waste_cost / analysis_total_cost) * 100
        previous_waste_pct = (previous_waste_cost / previous_total_cost) * 100
        waste_pct_change = analysis_waste_pct - previous_waste_pct
        
        findings.append({
            "title": "Known Waste Cost Percentage",
            "claim": f"Known waste as percentage of total inventory cost changed by {waste_pct_change:.2f} percentage points",
            "finding_type": "waste_metric",
            "metrics": {
                "analysis_waste_pct": {
                    "value": round(analysis_waste_pct, 2),
                    "unit": "percent",
                    "numerator": round(analysis_waste_cost, 2),
                    "denominator": round(analysis_total_cost, 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "previous_waste_pct": {
                    "value": round(previous_waste_pct, 2),
                    "unit": "percent",
                    "numerator": round(previous_waste_cost, 2),
                    "denominator": round(previous_total_cost, 2),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "waste_pct_change": {
                    "value": round(waste_pct_change, 2),
                    "unit": "percentage_points",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                f"Analysis period: {len(inventory_analysis)} inventory records",
                f"Previous period: {len(inventory_previous)} inventory records",
                "Only known waste costs included; unknown waste values excluded per requirements"
            ],
            "assumptions": [
                "Inventory records represent complete weekly snapshots",
                "Unit costs are consistent within each period",
                "Known waste values are accurate and complete"
            ],
            "confidence": 0.75
        })

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
