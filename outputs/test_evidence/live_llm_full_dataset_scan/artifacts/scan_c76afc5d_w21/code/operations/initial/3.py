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
tz_offset = timedelta(hours=3)

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True, format='mixed')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True, format='mixed')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period (using UTC timestamps)
analysis_start = pd.Timestamp('2026-06-01T00:00:00', tz='UTC')
analysis_end = pd.Timestamp('2026-06-08T00:00:00', tz='UTC')
previous_start = pd.Timestamp('2026-05-25T00:00:00', tz='UTC')
previous_end = pd.Timestamp('2026-06-01T00:00:00', tz='UTC')

# Convert date columns to datetime for consistent comparison
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.normalize()) & (staff_df['date'] < analysis_end.normalize())].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start.normalize()) & (staff_df['date'] < previous_end.normalize())].copy()

findings = []

# Finding 1: Conversion Rate Analysis
# Calculate unique transactions and valid footfall
analysis_transactions = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
analysis_footfall = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['door_count'].sum()

previous_transactions = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()
previous_footfall = traffic_previous[traffic_previous['is_dead_sensor_day'] == False]['door_count'].sum()

if analysis_footfall > 0 and previous_footfall > 0:
    analysis_conversion = analysis_transactions / analysis_footfall if analysis_footfall > 0 else 0
    previous_conversion = previous_transactions / previous_footfall if previous_footfall > 0 else 0
    conversion_change = ((analysis_conversion - previous_conversion) / previous_conversion * 100) if previous_conversion > 0 else 0
    
    findings.append({
        "title": "Conversion Rate Comparison",
        "claim": f"Conversion rate in analysis week was {analysis_conversion:.4f} vs {previous_conversion:.4f} in previous week, a change of {conversion_change:.1f}%",
        "finding_type": "conversion_analysis",
        "metrics": {
            "analysis_conversion_rate": {
                "value": round(analysis_conversion, 4),
                "unit": "ratio",
                "numerator": analysis_transactions,
                "denominator": analysis_footfall,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_conversion_rate": {
                "value": round(previous_conversion, 4),
                "unit": "ratio",
                "numerator": previous_transactions,
                "denominator": previous_footfall,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_change_percent": {
                "value": round(conversion_change, 1),
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
            f"Previous period: {previous_transactions} valid transactions",
            f"Dead sensor days excluded from footfall calculation",
            f"Analysis footfall: {analysis_footfall}, Previous footfall: {previous_footfall}"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Dead sensor days excluded from footfall denominator",
            "Conversion = unique valid sales transactions / valid footfall"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost vs Revenue Analysis
analysis_labour_cost = staff_analysis['labour_cost_sar'].sum()
analysis_revenue = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()

previous_labour_cost = staff_previous['labour_cost_sar'].sum()
previous_revenue = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()

if analysis_revenue > 0 and previous_revenue > 0:
    analysis_labour_ratio = analysis_labour_cost / analysis_revenue if analysis_revenue > 0 else 0
    previous_labour_ratio = previous_labour_cost / previous_revenue if previous_revenue > 0 else 0
    labour_ratio_change = ((analysis_labour_ratio - previous_labour_ratio) / previous_labour_ratio * 100) if previous_labour_ratio > 0 else 0
    
    findings.append({
        "title": "Labour Cost to Revenue Ratio",
        "claim": f"Labour cost ratio was {analysis_labour_ratio:.4f} in analysis week vs {previous_labour_ratio:.4f} in previous week, a change of {labour_ratio_change:.1f}%",
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
                "value": round(labour_ratio_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period labour cost: {analysis_labour_cost:.2f} SAR from {len(staff_analysis)} staff records",
            f"Previous period labour cost: {previous_labour_cost:.2f} SAR from {len(staff_previous)} staff records",
            f"Analysis revenue: {analysis_revenue:.2f} SAR (refunds excluded)",
            f"Previous revenue: {previous_revenue:.2f} SAR (refunds excluded)"
        ],
        "assumptions": [
            "Labour cost calculated from staff shift records",
            "Revenue includes only non-refund transactions",
            "Staff records represent actual hours worked"
        ],
        "confidence": 0.80
    })

# Finding 3: Inventory Waste Analysis
analysis_week_start = pd.Timestamp('2026-06-01')
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start].copy()

if len(inventory_analysis) > 0:
    total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_wasted = inventory_analysis['units_wasted'].sum()
    total_units_sold = inventory_analysis['units_sold'].sum()
    
    if total_units_sold > 0:
        waste_to_sales_ratio = total_units_wasted / total_units_sold
        
        findings.append({
            "title": "Inventory Waste Analysis",
            "claim": f"Known waste cost was {total_waste_cost:.2f} SAR with {total_units_wasted} units wasted vs {total_units_sold} units sold (waste ratio: {waste_to_sales_ratio:.4f})",
            "finding_type": "inventory_efficiency",
            "metrics": {
                "known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_week_start.isoformat(),
                    "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                },
                "units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_week_start.isoformat(),
                    "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                },
                "waste_to_sales_ratio": {
                    "value": round(waste_to_sales_ratio, 4),
                    "unit": "ratio",
                    "numerator": int(total_units_wasted),
                    "denominator": int(total_units_sold),
                    "period_start": analysis_week_start.isoformat(),
                    "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                f"Analysis covers {len(inventory_analysis)} SKUs in week starting {analysis_week_start}",
                f"Known waste cost: {total_waste_cost:.2f} SAR",
                f"Unknown waste values preserved as per data quality notes",
                f"Units sold: {int(total_units_sold)}, Units wasted: {int(total_units_wasted)}"
            ],
            "assumptions": [
                "Inventory counts represent end-of-week snapshots",
                "Known waste values are accurate; unknown waste excluded from calculations",
                "Units sold derived from inventory records"
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
