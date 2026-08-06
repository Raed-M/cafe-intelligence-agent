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

# Define timezone using zoneinfo instead of pytz
tz = ZoneInfo('Asia/Riyadh')

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.date
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.date

# Analysis period
analysis_start = pd.Timestamp('2026-05-04', tz=tz)
analysis_end = pd.Timestamp('2026-05-11', tz=tz)
previous_start = pd.Timestamp('2026-04-27', tz=tz)
previous_end = pd.Timestamp('2026-05-04', tz=tz)

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
# Calculate conversion for analysis period
valid_traffic_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False].copy()
if len(valid_traffic_analysis) > 0:
    total_footfall_analysis = valid_traffic_analysis['door_count'].sum()
    
    # Count unique transactions (excluding refunds for conversion)
    valid_transactions_analysis = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
    
    if total_footfall_analysis > 0:
        conversion_analysis = valid_transactions_analysis / total_footfall_analysis
        
        # Calculate for previous period
        valid_traffic_previous = traffic_previous[traffic_previous['is_dead_sensor_day'] == False].copy()
        if len(valid_traffic_previous) > 0:
            total_footfall_previous = valid_traffic_previous['door_count'].sum()
            valid_transactions_previous = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()
            
            if total_footfall_previous > 0:
                conversion_previous = valid_transactions_previous / total_footfall_previous
                
                findings.append({
                    "title": "Conversion Rate Comparison",
                    "claim": f"Conversion rate in analysis period ({conversion_analysis:.2%}) compared to previous period ({conversion_previous:.2%})",
                    "finding_type": "conversion_analysis",
                    "metrics": {
                        "conversion_rate_analysis": {
                            "value": round(conversion_analysis, 4),
                            "unit": "ratio",
                            "numerator": int(valid_transactions_analysis),
                            "denominator": int(total_footfall_analysis),
                            "period_start": analysis_start.isoformat(),
                            "period_end": analysis_end.isoformat()
                        },
                        "conversion_rate_previous": {
                            "value": round(conversion_previous, 4),
                            "unit": "ratio",
                            "numerator": int(valid_transactions_previous),
                            "denominator": int(total_footfall_previous),
                            "period_start": previous_start.isoformat(),
                            "period_end": previous_end.isoformat()
                        }
                    },
                    "source_names": ["pos", "traffic"],
                    "sample_size": int(valid_transactions_analysis),
                    "coverage_notes": [
                        f"Analysis period: {len(valid_traffic_analysis)} days with valid sensor data",
                        f"Previous period: {len(valid_traffic_previous)} days with valid sensor data",
                        "Excluded dead sensor days from footfall denominator"
                    ],
                    "assumptions": [
                        "Conversion = unique non-refund transactions / valid footfall",
                        "Dead sensor days excluded from both periods",
                        "Transaction_id used for unique transaction count"
                    ],
                    "confidence": 0.85
                })

# Finding 2: Labour Cost vs Sales Analysis
if len(staff_analysis) > 0 and len(pos_analysis) > 0:
    total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
    total_sales_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
    
    if total_sales_analysis > 0:
        labour_to_sales_ratio_analysis = total_labour_cost_analysis / total_sales_analysis
        
        if len(staff_previous) > 0 and len(pos_previous) > 0:
            total_labour_cost_previous = staff_previous['labour_cost_sar'].sum()
            total_sales_previous = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()
            
            if total_sales_previous > 0:
                labour_to_sales_ratio_previous = total_labour_cost_previous / total_sales_previous
                
                findings.append({
                    "title": "Labour Cost Efficiency",
                    "claim": f"Labour cost as % of sales: {labour_to_sales_ratio_analysis:.2%} (analysis) vs {labour_to_sales_ratio_previous:.2%} (previous)",
                    "finding_type": "labour_efficiency",
                    "metrics": {
                        "labour_cost_ratio_analysis": {
                            "value": round(labour_to_sales_ratio_analysis, 4),
                            "unit": "ratio",
                            "numerator": round(total_labour_cost_analysis, 2),
                            "denominator": round(total_sales_analysis, 2),
                            "period_start": analysis_start.isoformat(),
                            "period_end": analysis_end.isoformat()
                        },
                        "labour_cost_ratio_previous": {
                            "value": round(labour_to_sales_ratio_previous, 4),
                            "unit": "ratio",
                            "numerator": round(total_labour_cost_previous, 2),
                            "denominator": round(total_sales_previous, 2),
                            "period_start": previous_start.isoformat(),
                            "period_end": previous_end.isoformat()
                        }
                    },
                    "source_names": ["staff", "pos"],
                    "sample_size": len(staff_analysis),
                    "coverage_notes": [
                        f"Analysis period: {len(staff_analysis)} staff records",
                        f"Previous period: {len(staff_previous)} staff records",
                        "Labour cost includes computed_duration_hours * hourly_rate_sar"
                    ],
                    "assumptions": [
                        "Labour cost from staff.labour_cost_sar field",
                        "Sales excludes refunds (is_refund == False)",
                        "No imputation for missing staff records"
                    ],
                    "confidence": 0.80
                })

# Finding 3: Inventory Waste Analysis
if len(inventory_df) > 0:
    # Get inventory for analysis week
    analysis_week_start = pd.Timestamp('2026-05-04', tz=tz).date()
    inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start].copy()
    
    if len(inventory_analysis) > 0:
        total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
        total_units_wasted = inventory_analysis['units_wasted'].sum()
        total_units_sold = inventory_analysis['units_sold'].sum()
        
        if total_units_sold > 0:
            waste_to_sales_ratio = total_units_wasted / (total_units_sold + total_units_wasted)
            
            findings.append({
                "title": "Inventory Waste Rate",
                "claim": f"Waste rate for analysis week: {waste_to_sales_ratio:.2%} of total units (sold + wasted)",
                "finding_type": "inventory_waste",
                "metrics": {
                    "waste_rate": {
                        "value": round(waste_to_sales_ratio, 4),
                        "unit": "ratio",
                        "numerator": int(total_units_wasted),
                        "denominator": int(total_units_sold + total_units_wasted),
                        "period_start": pd.Timestamp(analysis_week_start, tz=tz).isoformat(),
                        "period_end": (pd.Timestamp(analysis_week_start, tz=tz) + timedelta(days=7)).isoformat()
                    },
                    "waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": pd.Timestamp(analysis_week_start, tz=tz).isoformat(),
                        "period_end": (pd.Timestamp(analysis_week_start, tz=tz) + timedelta(days=7)).isoformat()
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inventory_analysis),
                "coverage_notes": [
                    f"Analysis week: {len(inventory_analysis)} SKUs tracked",
                    "Known waste cost calculated from units_wasted * unit_cost_sar",
                    "Unknown waste values preserved as per schema"
                ],
                "assumptions": [
                    "Waste rate = units_wasted / (units_sold + units_wasted)",
                    "Only known waste costs included in total",
                    "Sunday inventory counts treated as weekly snapshot, not real-time"
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
