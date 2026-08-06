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
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period
analysis_start = pd.Timestamp('2026-05-18', tz=tz)
analysis_end = pd.Timestamp('2026-05-25', tz=tz)
previous_start = pd.Timestamp('2026-05-11', tz=tz)
previous_end = pd.Timestamp('2026-05-18', tz=tz)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.date()) & (traffic_df['date'] < previous_end.date())].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())].copy()

# Finding 1: Conversion Rate Analysis
# Calculate unique transactions and valid footfall
analysis_transactions = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
analysis_footfall = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['door_count'].sum()

previous_transactions = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()
previous_footfall = traffic_previous[traffic_previous['is_dead_sensor_day'] == False]['door_count'].sum()

# Calculate conversion rates
analysis_conversion = analysis_transactions / analysis_footfall if analysis_footfall > 0 else None
previous_conversion = previous_transactions / previous_footfall if previous_footfall > 0 else None

# Finding 2: Labour Cost vs Revenue Analysis
# Calculate total labour cost and revenue
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
analysis_revenue = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()

previous_labour_cost = staff_previous['labour_cost_sar'].sum()
previous_revenue = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()

# Calculate labour cost as percentage of revenue
analysis_labour_pct = (analysis_labour_cost / analysis_revenue * 100) if analysis_revenue > 0 else None
previous_labour_pct = (previous_labour_cost / previous_revenue * 100) if previous_revenue > 0 else None

# Finding 3: Inventory Waste Analysis
# Get inventory data for the analysis week
analysis_week_start = pd.Timestamp('2026-05-18', tz=tz)
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start.date()].copy()

# Calculate total waste cost and units
total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
total_waste_units = inventory_analysis['units_wasted'].sum()
total_units_sold = inventory_analysis['units_sold'].sum()

# Calculate waste as percentage of sold units
waste_pct = (total_waste_units / (total_units_sold + total_waste_units) * 100) if (total_units_sold + total_waste_units) > 0 else None

# Prepare findings
findings = []

# Finding 1: Conversion Rate
if analysis_conversion is not None and previous_conversion is not None:
    conversion_change = ((analysis_conversion - previous_conversion) / previous_conversion * 100) if previous_conversion > 0 else None
    
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
            "conversion_change_pct": {
                "value": round(conversion_change, 2) if conversion_change is not None else None,
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(analysis_transactions),
        "coverage_notes": [
            "Excluded dead sensor days from footfall denominator",
            "Excluded refund transactions from transaction count",
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Previous period: 2026-05-11 to 2026-05-18"
        ],
        "assumptions": [
            "Footfall sensor accuracy is consistent across periods",
            "Transaction_id uniqueness indicates distinct customer baskets",
            "Dead sensor days are correctly flagged in traffic data"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost Efficiency
if analysis_labour_pct is not None and previous_labour_pct is not None:
    labour_pct_change = analysis_labour_pct - previous_labour_pct
    
    findings.append({
        "title": "Labour Cost as Percentage of Revenue",
        "claim": f"Labour cost represented {analysis_labour_pct:.2f}% of revenue in analysis period vs {previous_labour_pct:.2f}% in previous period, a change of {labour_pct_change:.2f} percentage points",
        "finding_type": "labour_efficiency",
        "metrics": {
            "analysis_labour_pct": {
                "value": round(analysis_labour_pct, 2),
                "unit": "percent",
                "numerator": round(analysis_labour_cost, 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_labour_pct": {
                "value": round(previous_labour_pct, 2),
                "unit": "percent",
                "numerator": round(previous_labour_cost, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_pct_change": {
                "value": round(labour_pct_change, 2),
                "unit": "percentage_points",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            "Labour cost includes all staff shifts in period",
            "Revenue excludes refund transactions",
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Previous period: 2026-05-11 to 2026-05-18"
        ],
        "assumptions": [
            "Labour cost data is complete for all staff",
            "Hourly rates are accurately recorded",
            "Revenue calculation excludes refunds as per metric definition"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory Waste
if waste_pct is not None and total_waste_cost > 0:
    findings.append({
        "title": "Inventory Waste Analysis",
        "claim": f"Known waste represented {waste_pct:.2f}% of total units (waste + sold) in analysis week, with total waste cost of {total_waste_cost:.2f} SAR",
        "finding_type": "inventory_waste",
        "metrics": {
            "waste_percentage": {
                "value": round(waste_pct, 2),
                "unit": "percent",
                "numerator": int(total_waste_units),
                "denominator": int(total_waste_units + total_units_sold),
                "period_start": analysis_week_start.isoformat(),
                "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
            },
            "total_waste_cost": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_week_start.isoformat(),
                "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
            },
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_week_start.isoformat(),
                "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            "Only known waste values included",
            "Unknown waste values excluded per requirements",
            "Week starting: 2026-05-18",
            f"Number of SKUs with waste data: {len(inventory_analysis)}"
        ],
        "assumptions": [
            "Waste data is accurately recorded in inventory system",
            "Unit costs are consistent with recorded values",
            "Waste includes only known/measured waste, not estimated"
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
