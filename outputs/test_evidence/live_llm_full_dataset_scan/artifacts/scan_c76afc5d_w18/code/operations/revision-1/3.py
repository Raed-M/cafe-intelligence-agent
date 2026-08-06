import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

# Define periods using zoneinfo instead of pytz
tz = ZoneInfo('Asia/Riyadh')
analysis_start = datetime(2026, 5, 11, 0, 0, 0, tzinfo=tz)
analysis_end = datetime(2026, 5, 18, 0, 0, 0, tzinfo=tz)
previous_start = datetime(2026, 5, 4, 0, 0, 0, tzinfo=tz)
previous_end = datetime(2026, 5, 11, 0, 0, 0, tzinfo=tz)

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

# Extract business_date from pos
pos_df['business_date'] = pd.to_datetime(pos_df['business_date']).dt.tz_localize(tz)

findings = []

# FINDING 1: Conversion Rate Analysis
# Calculate conversion for analysis period
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)]

# Filter out dead sensor days
analysis_traffic_valid = analysis_traffic[analysis_traffic['is_dead_sensor_day'] == False]

# Count unique transactions (not rows)
analysis_transactions = analysis_pos[analysis_pos['is_refund'] == False]['transaction_id'].nunique()
analysis_footfall = analysis_traffic_valid['door_count'].sum()

if analysis_footfall > 0:
    analysis_conversion = analysis_transactions / analysis_footfall
else:
    analysis_conversion = None

# Calculate conversion for previous period
previous_pos = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]
previous_traffic = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)]
previous_traffic_valid = previous_traffic[previous_traffic['is_dead_sensor_day'] == False]

previous_transactions = previous_pos[previous_pos['is_refund'] == False]['transaction_id'].nunique()
previous_footfall = previous_traffic_valid['door_count'].sum()

if previous_footfall > 0:
    previous_conversion = previous_transactions / previous_footfall
else:
    previous_conversion = None

if analysis_conversion is not None and previous_conversion is not None and previous_conversion > 0:
    conversion_change_pct = ((analysis_conversion - previous_conversion) / previous_conversion) * 100
    
    findings.append({
        "title": "Conversion Rate Change Week-over-Week",
        "claim": f"Conversion rate changed by {conversion_change_pct:.2f}% from previous week to analysis week",
        "finding_type": "operational_metric",
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
            "conversion_change_pct": {
                "value": round(conversion_change_pct, 2),
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
            f"Analysis period: {analysis_transactions} valid transactions from {int(analysis_footfall)} footfall events",
            f"Previous period: {previous_transactions} valid transactions from {int(previous_footfall)} footfall events",
            f"Dead sensor days excluded from footfall denominator"
        ],
        "assumptions": [
            "Conversion = unique valid sales transactions / valid footfall",
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall calculation"
        ],
        "confidence": 0.85
    })

# FINDING 2: Labour Cost and Staffing Analysis
analysis_staff = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)]
previous_staff = staff_df[(staff_df['date'] >= previous_start) & (staff_df['date'] < previous_end)]

analysis_labour_cost = analysis_staff['labour_cost_sar'].sum()
analysis_staff_hours = analysis_staff['computed_duration_hours'].sum()
previous_labour_cost = previous_staff['labour_cost_sar'].sum()
previous_staff_hours = previous_staff['computed_duration_hours'].sum()

if previous_labour_cost > 0:
    labour_cost_change_pct = ((analysis_labour_cost - previous_labour_cost) / previous_labour_cost) * 100
    
    findings.append({
        "title": "Labour Cost Change Week-over-Week",
        "claim": f"Total labour cost changed by {labour_cost_change_pct:.2f}% from previous week to analysis week",
        "finding_type": "operational_metric",
        "metrics": {
            "analysis_labour_cost_sar": {
                "value": round(analysis_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_staff_hours": {
                "value": round(analysis_staff_hours, 2),
                "unit": "hours",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_labour_cost_sar": {
                "value": round(previous_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "previous_staff_hours": {
                "value": round(previous_staff_hours, 2),
                "unit": "hours",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_change_pct": {
                "value": round(labour_cost_change_pct, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["staff"],
        "sample_size": len(analysis_staff),
        "coverage_notes": [
            f"Analysis period: {len(analysis_staff)} staff records with total cost {analysis_labour_cost:.2f} SAR",
            f"Previous period: {len(previous_staff)} staff records with total cost {previous_labour_cost:.2f} SAR"
        ],
        "assumptions": [
            "Labour cost calculated from computed_duration_hours and hourly_rate_sar",
            "Staff shifts computed using shift_start and shift_end overlap"
        ],
        "confidence": 0.90
    })

# FINDING 3: Waste and Inventory Analysis
analysis_inventory = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)]
previous_inventory = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)]

analysis_known_waste_cost = analysis_inventory['known_waste_cost_sar'].sum()
analysis_total_units_wasted = analysis_inventory['units_wasted'].sum()
analysis_total_units_sold = analysis_inventory['units_sold'].sum()

previous_known_waste_cost = previous_inventory['known_waste_cost_sar'].sum()
previous_total_units_wasted = previous_inventory['units_wasted'].sum()
previous_total_units_sold = previous_inventory['units_sold'].sum()

if analysis_total_units_sold > 0 and previous_total_units_sold > 0:
    analysis_waste_ratio = analysis_total_units_wasted / (analysis_total_units_sold + analysis_total_units_wasted)
    previous_waste_ratio = previous_total_units_wasted / (previous_total_units_sold + previous_total_units_wasted)
    
    waste_ratio_change = ((analysis_waste_ratio - previous_waste_ratio) / previous_waste_ratio) * 100 if previous_waste_ratio > 0 else 0
    
    findings.append({
        "title": "Known Waste Cost and Ratio Analysis",
        "claim": f"Known waste cost was {analysis_known_waste_cost:.2f} SAR in analysis week vs {previous_known_waste_cost:.2f} SAR in previous week",
        "finding_type": "operational_metric",
        "metrics": {
            "analysis_known_waste_cost_sar": {
                "value": round(analysis_known_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_waste_ratio": {
                "value": round(analysis_waste_ratio, 4),
                "unit": "ratio",
                "numerator": int(analysis_total_units_wasted),
                "denominator": int(analysis_total_units_sold + analysis_total_units_wasted),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_known_waste_cost_sar": {
                "value": round(previous_known_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "previous_waste_ratio": {
                "value": round(previous_waste_ratio, 4),
                "unit": "ratio",
                "numerator": int(previous_total_units_wasted),
                "denominator": int(previous_total_units_sold + previous_total_units_wasted),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "waste_ratio_change_pct": {
                "value": round(waste_ratio_change, 2),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(analysis_inventory),
        "coverage_notes": [
            f"Analysis period: {len(analysis_inventory)} inventory records",
            f"Previous period: {len(previous_inventory)} inventory records",
            "Only known waste values included; unknown waste preserved"
        ],
        "assumptions": [
            "Waste ratio = units_wasted / (units_sold + units_wasted)",
            "Known waste cost calculated from units_wasted and unit_cost_sar",
            "Sunday inventory counts treated as weekly snapshots, not real-time"
        ],
        "confidence": 0.80
    })

# Write output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
