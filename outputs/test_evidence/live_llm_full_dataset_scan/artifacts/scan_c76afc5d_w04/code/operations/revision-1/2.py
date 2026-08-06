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
analysis_start = datetime(2026, 2, 2, 0, 0, 0, tzinfo=tz)
analysis_end = datetime(2026, 2, 9, 0, 0, 0, tzinfo=tz)
previous_start = datetime(2026, 1, 26, 0, 0, 0, tzinfo=tz)
previous_end = datetime(2026, 2, 2, 0, 0, 0, tzinfo=tz)

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

# Extract business_date from pos if available, otherwise compute from timestamp
if 'business_date' not in pos_df.columns:
    pos_df['business_date'] = pos_df['timestamp'].dt.date
else:
    pos_df['business_date'] = pd.to_datetime(pos_df['business_date']).dt.date

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)].copy()

staff_analysis = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start) & (staff_df['date'] < previous_end)].copy()

inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)].copy()
inventory_previous = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)].copy()

findings = []

# Finding 1: Conversion Rate Analysis
try:
    # Analysis period conversion
    valid_sales_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('transaction_id').size().reset_index(name='count')
    unique_transactions_analysis = len(valid_sales_analysis)
    
    # Filter out dead sensor days
    valid_traffic_analysis = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
    total_footfall_analysis = valid_traffic_analysis['door_count'].sum()
    
    if total_footfall_analysis > 0:
        conversion_analysis = unique_transactions_analysis / total_footfall_analysis
    else:
        conversion_analysis = None
    
    # Previous period conversion
    valid_sales_previous = pos_previous[~pos_previous['is_refund']].groupby('transaction_id').size().reset_index(name='count')
    unique_transactions_previous = len(valid_sales_previous)
    
    valid_traffic_previous = traffic_previous[~traffic_previous['is_dead_sensor_day']].copy()
    total_footfall_previous = valid_traffic_previous['door_count'].sum()
    
    if total_footfall_previous > 0:
        conversion_previous = unique_transactions_previous / total_footfall_previous
    else:
        conversion_previous = None
    
    if conversion_analysis is not None and conversion_previous is not None:
        conversion_change = ((conversion_analysis - conversion_previous) / conversion_previous) * 100
        
        findings.append({
            "title": "Conversion Rate Change",
            "claim": f"Conversion rate changed by {conversion_change:.1f}% from {conversion_previous:.3f} to {conversion_analysis:.3f}",
            "finding_type": "performance_metric",
            "metrics": {
                "conversion_rate_analysis": {
                    "value": round(conversion_analysis, 4),
                    "unit": "transactions_per_visitor",
                    "numerator": unique_transactions_analysis,
                    "denominator": total_footfall_analysis,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "conversion_rate_previous": {
                    "value": round(conversion_previous, 4),
                    "unit": "transactions_per_visitor",
                    "numerator": unique_transactions_previous,
                    "denominator": total_footfall_previous,
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "conversion_change_pct": {
                    "value": round(conversion_change, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "traffic"],
            "sample_size": unique_transactions_analysis,
            "coverage_notes": [
                f"Analysis period: {unique_transactions_analysis} valid transactions, {total_footfall_analysis} footfall (dead sensor days excluded)",
                f"Previous period: {unique_transactions_previous} valid transactions, {total_footfall_previous} footfall (dead sensor days excluded)"
            ],
            "assumptions": [
                "Conversion = unique valid sales transactions / valid footfall",
                "Refunds excluded from transaction count",
                "Dead sensor days excluded from footfall denominator"
            ],
            "confidence": 0.85
        })
except Exception as e:
    pass

# Finding 2: Labour Cost and Staffing Analysis
try:
    # Analysis period labour cost
    labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
    staff_days_analysis = staff_analysis['date'].nunique()
    avg_daily_labour_analysis = labour_cost_analysis / staff_days_analysis if staff_days_analysis > 0 else 0
    
    # Previous period labour cost
    labour_cost_previous = staff_previous['labour_cost_sar'].sum()
    staff_days_previous = staff_previous['date'].nunique()
    avg_daily_labour_previous = labour_cost_previous / staff_days_previous if staff_days_previous > 0 else 0
    
    # Revenue analysis
    revenue_analysis = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
    revenue_previous = pos_previous[~pos_previous['is_refund']]['line_total_sar'].sum()
    
    if revenue_analysis > 0 and revenue_previous > 0:
        labour_pct_analysis = (labour_cost_analysis / revenue_analysis) * 100
        labour_pct_previous = (labour_cost_previous / revenue_previous) * 100
        labour_pct_change = labour_pct_analysis - labour_pct_previous
        
        findings.append({
            "title": "Labour Cost as Percentage of Revenue",
            "claim": f"Labour cost as % of revenue changed from {labour_pct_previous:.1f}% to {labour_pct_analysis:.1f}% (change: {labour_pct_change:+.1f} percentage points)",
            "finding_type": "cost_efficiency",
            "metrics": {
                "labour_cost_pct_analysis": {
                    "value": round(labour_pct_analysis, 1),
                    "unit": "percent",
                    "numerator": round(labour_cost_analysis, 2),
                    "denominator": round(revenue_analysis, 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "labour_cost_pct_previous": {
                    "value": round(labour_pct_previous, 1),
                    "unit": "percent",
                    "numerator": round(labour_cost_previous, 2),
                    "denominator": round(revenue_previous, 2),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "labour_cost_pct_change": {
                    "value": round(labour_pct_change, 1),
                    "unit": "percentage_points",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "avg_daily_labour_analysis": {
                    "value": round(avg_daily_labour_analysis, 2),
                    "unit": "sar",
                    "numerator": round(labour_cost_analysis, 2),
                    "denominator": staff_days_analysis,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["staff", "pos"],
            "sample_size": len(staff_analysis),
            "coverage_notes": [
                f"Analysis period: {staff_days_analysis} days with staff data, {len(staff_analysis)} staff records",
                f"Previous period: {staff_days_previous} days with staff data, {len(staff_previous)} staff records",
                f"Revenue calculated from non-refund transactions"
            ],
            "assumptions": [
                "Labour cost includes all computed hourly rates",
                "Revenue excludes refunds",
                "Staff records represent actual shifts worked"
            ],
            "confidence": 0.80
        })
except Exception as e:
    pass

# Finding 3: Inventory Waste Analysis
try:
    # Analysis period waste
    if len(inventory_analysis) > 0:
        total_waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
        total_units_sold_analysis = inventory_analysis['units_sold'].sum()
        total_units_wasted_analysis = inventory_analysis['units_wasted'].sum()
        
        # Previous period waste
        if len(inventory_previous) > 0:
            total_waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
            total_units_sold_previous = inventory_previous['units_sold'].sum()
            total_units_wasted_previous = inventory_previous['units_wasted'].sum()
            
            if total_units_sold_analysis > 0 and total_units_sold_previous > 0:
                waste_rate_analysis = (total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)) * 100 if (total_units_sold_analysis + total_units_wasted_analysis) > 0 else 0
                waste_rate_previous = (total_units_wasted_previous / (total_units_sold_previous + total_units_wasted_previous)) * 100 if (total_units_sold_previous + total_units_wasted_previous) > 0 else 0
                
                waste_rate_change = waste_rate_analysis - waste_rate_previous
                
                findings.append({
                    "title": "Known Waste Rate Change",
                    "claim": f"Known waste rate changed from {waste_rate_previous:.1f}% to {waste_rate_analysis:.1f}% (change: {waste_rate_change:+.1f} percentage points)",
                    "finding_type": "waste_efficiency",
                    "metrics": {
                        "waste_rate_analysis": {
                            "value": round(waste_rate_analysis, 1),
                            "unit": "percent",
                            "numerator": total_units_wasted_analysis,
                            "denominator": total_units_sold_analysis + total_units_wasted_analysis,
                            "period_start": analysis_start.isoformat(),
                            "period_end": analysis_end.isoformat()
                        },
                        "waste_rate_previous": {
                            "value": round(waste_rate_previous, 1),
                            "unit": "percent",
                            "numerator": total_units_wasted_previous,
                            "denominator": total_units_sold_previous + total_units_wasted_previous,
                            "period_start": previous_start.isoformat(),
                            "period_end": previous_end.isoformat()
                        },
                        "waste_rate_change": {
                            "value": round(waste_rate_change, 1),
                            "unit": "percentage_points",
                            "numerator": None,
                            "denominator": None,
                            "period_start": analysis_start.isoformat(),
                            "period_end": analysis_end.isoformat()
                        },
                        "known_waste_cost_analysis": {
                            "value": round(total_waste_cost_analysis, 2),
                            "unit": "sar",
                            "numerator": None,
                            "denominator": None,
                            "period_start": analysis_start.isoformat(),
                            "period_end": analysis_end.isoformat()
                        }
                    },
                    "source_names": ["inventory"],
                    "sample_size": len(inventory_analysis),
                    "coverage_notes": [
                        f"Analysis period: {len(inventory_analysis)} inventory records, {total_units_wasted_analysis} units wasted",
                        f"Previous period: {len(inventory_previous)} inventory records, {total_units_wasted_previous} units wasted",
                        "Only known waste values included; unknown waste excluded"
                    ],
                    "assumptions": [
                        "Waste rate = units_wasted / (units_sold + units_wasted)",
                        "Only known waste costs are included in analysis",
                        "Inventory records represent weekly snapshots"
                    ],
                    "confidence": 0.75
                })
except Exception as e:
    pass

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
