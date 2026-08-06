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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
staff_df = pd.read_parquet(inputs['staff'])
inventory_df = pd.read_parquet(inputs['inventory'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'])
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period (using UTC offsets instead of pytz)
analysis_start = pd.Timestamp('2026-02-16T00:00:00+03:00')
analysis_end = pd.Timestamp('2026-02-23T00:00:00+03:00')
previous_start = pd.Timestamp('2026-02-09T00:00:00+03:00')
previous_end = pd.Timestamp('2026-02-16T00:00:00+03:00')

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

# Count unique valid transactions (exclude refunds for conversion numerator)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Filter traffic for analysis period
traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start.date()) & 
    (traffic_df['date'] < analysis_end.date())
].copy()

# Exclude dead sensor days
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
total_footfall_analysis = traffic_analysis_valid['door_count'].sum()

# Calculate conversion rate
if total_footfall_analysis > 0:
    conversion_rate_analysis = valid_transactions_analysis / total_footfall_analysis
else:
    conversion_rate_analysis = None

# Previous period for comparison
pos_previous = pos_df[
    (pos_df['timestamp'] >= previous_start) & 
    (pos_df['timestamp'] < previous_end)
].copy()

valid_transactions_previous = pos_previous[~pos_previous['is_refund']]['transaction_id'].nunique()

traffic_previous = traffic_df[
    (traffic_df['date'] >= previous_start.date()) & 
    (traffic_df['date'] < previous_end.date())
].copy()

traffic_previous_valid = traffic_previous[~traffic_previous['is_dead_sensor_day']].copy()
total_footfall_previous = traffic_previous_valid['door_count'].sum()

if total_footfall_previous > 0:
    conversion_rate_previous = valid_transactions_previous / total_footfall_previous
else:
    conversion_rate_previous = None

# Determine if there's a meaningful change
if conversion_rate_analysis is not None and conversion_rate_previous is not None:
    conversion_change = conversion_rate_analysis - conversion_rate_previous
    conversion_pct_change = (conversion_change / conversion_rate_previous * 100) if conversion_rate_previous > 0 else None
    
    if abs(conversion_pct_change) >= 5:  # At least 5% change
        findings.append({
            "title": "Conversion Rate Change Week-over-Week",
            "claim": f"Conversion rate in analysis week (Feb 16-23) was {conversion_rate_analysis:.4f} vs {conversion_rate_previous:.4f} in previous week, a {conversion_pct_change:.1f}% change.",
            "finding_type": "performance_metric",
            "metrics": {
                "conversion_rate_analysis": {
                    "value": round(conversion_rate_analysis, 4),
                    "unit": "transactions_per_visitor",
                    "numerator": int(valid_transactions_analysis),
                    "denominator": int(total_footfall_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "conversion_rate_previous": {
                    "value": round(conversion_rate_previous, 4),
                    "unit": "transactions_per_visitor",
                    "numerator": int(valid_transactions_previous),
                    "denominator": int(total_footfall_previous),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "conversion_change_pct": {
                    "value": round(conversion_pct_change, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "traffic"],
            "sample_size": int(valid_transactions_analysis),
            "coverage_notes": [
                f"Analysis period: {int(len(traffic_analysis_valid))} valid traffic days out of {len(traffic_analysis)} total days",
                f"Previous period: {int(len(traffic_previous_valid))} valid traffic days out of {len(traffic_previous)} total days",
                "Dead sensor days excluded from footfall denominator"
            ],
            "assumptions": [
                "Conversion = unique valid sales transactions / valid footfall",
                "Refunds excluded from transaction count",
                "Dead sensor days excluded from footfall calculation"
            ],
            "confidence": 0.75
        })

# ============================================================================
# FINDING 2: Labour Cost vs Demand Alignment
# ============================================================================

# Calculate total labour cost for analysis period
staff_analysis = staff_df[
    (staff_df['date'] >= analysis_start.date()) & 
    (staff_df['date'] < analysis_end.date())
].copy()

total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
total_staff_hours_analysis = staff_analysis['computed_duration_hours'].sum()

# Calculate total revenue (net of refunds) for analysis period
pos_analysis_revenue = pos_analysis.copy()
pos_analysis_revenue['net_line_total'] = pos_analysis_revenue['line_total_sar']
total_revenue_analysis = pos_analysis_revenue['net_line_total'].sum()

# Previous period labour and revenue
staff_previous = staff_df[
    (staff_df['date'] >= previous_start.date()) & 
    (staff_df['date'] < previous_end.date())
].copy()

total_labour_cost_previous = staff_previous['labour_cost_sar'].sum()
total_staff_hours_previous = staff_previous['computed_duration_hours'].sum()

pos_previous_revenue = pos_previous.copy()
pos_previous_revenue['net_line_total'] = pos_previous_revenue['line_total_sar']
total_revenue_previous = pos_previous_revenue['net_line_total'].sum()

# Calculate labour efficiency metrics
if total_revenue_analysis > 0 and total_labour_cost_analysis > 0:
    labour_efficiency_analysis = total_revenue_analysis / total_labour_cost_analysis
else:
    labour_efficiency_analysis = None

if total_revenue_previous > 0 and total_labour_cost_previous > 0:
    labour_efficiency_previous = total_revenue_previous / total_labour_cost_previous
else:
    labour_efficiency_previous = None

# Check if there's a meaningful change in labour efficiency
if labour_efficiency_analysis is not None and labour_efficiency_previous is not None:
    efficiency_change = labour_efficiency_analysis - labour_efficiency_previous
    efficiency_pct_change = (efficiency_change / labour_efficiency_previous * 100) if labour_efficiency_previous > 0 else None
    
    if abs(efficiency_pct_change) >= 5:  # At least 5% change
        findings.append({
            "title": "Labour Efficiency Change",
            "claim": f"Labour efficiency (revenue per SAR of labour cost) in analysis week was {labour_efficiency_analysis:.2f} vs {labour_efficiency_previous:.2f} in previous week, a {efficiency_pct_change:.1f}% change.",
            "finding_type": "operational_metric",
            "metrics": {
                "labour_efficiency_analysis": {
                    "value": round(labour_efficiency_analysis, 2),
                    "unit": "sar_revenue_per_sar_labour",
                    "numerator": round(total_revenue_analysis, 2),
                    "denominator": round(total_labour_cost_analysis, 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "labour_efficiency_previous": {
                    "value": round(labour_efficiency_previous, 2),
                    "unit": "sar_revenue_per_sar_labour",
                    "numerator": round(total_revenue_previous, 2),
                    "denominator": round(total_labour_cost_previous, 2),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "efficiency_change_pct": {
                    "value": round(efficiency_pct_change, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_labour_cost_analysis": {
                    "value": round(total_labour_cost_analysis, 2),
                    "unit": "sar",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_staff_hours_analysis": {
                    "value": round(total_staff_hours_analysis, 1),
                    "unit": "hours",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "staff"],
            "sample_size": len(staff_analysis),
            "coverage_notes": [
                f"Analysis period: {len(staff_analysis)} staff records",
                f"Previous period: {len(staff_previous)} staff records",
                "Labour cost calculated from computed_duration_hours and hourly_rate_sar"
            ],
            "assumptions": [
                "Labour efficiency = total revenue / total labour cost",
                "Revenue includes all line items net of refunds",
                "Staff hours computed from shift_start and shift_end overlap"
            ],
            "confidence": 0.70
        })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Get inventory for analysis week
analysis_week_start = pd.Timestamp('2026-02-16')
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == analysis_week_start
].copy()

# Get inventory for previous week
previous_week_start = pd.Timestamp('2026-02-09')
inventory_previous = inventory_df[
    inventory_df['week_starting'] == previous_week_start
].copy()

if len(inventory_analysis) > 0 and len(inventory_previous) > 0:
    # Calculate waste metrics
    total_waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_sold_analysis = inventory_analysis['units_sold'].sum()
    total_units_wasted_analysis = inventory_analysis['units_wasted'].sum()
    
    total_waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
    total_units_sold_previous = inventory_previous['units_sold'].sum()
    total_units_wasted_previous = inventory_previous['units_wasted'].sum()
    
    # Calculate waste rate
    if total_units_sold_analysis > 0:
        waste_rate_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis)
    else:
        waste_rate_analysis = None
    
    if total_units_sold_previous > 0:
        waste_rate_previous = total_units_wasted_previous / (total_units_sold_previous + total_units_wasted_previous)
    else:
        waste_rate_previous = None
    
    # Check if there's a meaningful change
    if waste_rate_analysis is not None and waste_rate_previous is not None:
        waste_rate_change = waste_rate_analysis - waste_rate_previous
        waste_rate_pct_change = (waste_rate_change / waste_rate_previous * 100) if waste_rate_previous > 0 else None
        
        if abs(waste_rate_pct_change) >= 10:  # At least 10% change in waste rate
            findings.append({
                "title": "Inventory Waste Rate Change",
                "claim": f"Waste rate (units wasted / total units) in analysis week was {waste_rate_analysis:.4f} vs {waste_rate_previous:.4f} in previous week, a {waste_rate_pct_change:.1f}% change. Known waste cost was {total_waste_cost_analysis:.2f} SAR vs {total_waste_cost_previous:.2f} SAR.",
                "finding_type": "inventory_metric",
                "metrics": {
                    "waste_rate_analysis": {
                        "value": round(waste_rate_analysis, 4),
                        "unit": "ratio",
                        "numerator": int(total_units_wasted_analysis),
                        "denominator": int(total_units_sold_analysis + total_units_wasted_analysis),
                        "period_start": analysis_week_start.isoformat(),
                        "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                    },
                    "waste_rate_previous": {
                        "value": round(waste_rate_previous, 4),
                        "unit": "ratio",
                        "numerator": int(total_units_wasted_previous),
                        "denominator": int(total_units_sold_previous + total_units_wasted_previous),
                        "period_start": previous_week_start.isoformat(),
                        "period_end": (previous_week_start + timedelta(days=7)).isoformat()
                    },
                    "waste_rate_change_pct": {
                        "value": round(waste_rate_pct_change, 1),
                        "unit": "percent",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_week_start.isoformat(),
                        "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                    },
                    "known_waste_cost_analysis": {
                        "value": round(total_waste_cost_analysis, 2),
                        "unit": "sar",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_week_start.isoformat(),
                        "period_end": (analysis_week_start + timedelta(days=7)).isoformat()
                    },
                    "known_waste_cost_previous": {
                        "value": round(total_waste_cost_previous, 2),
                        "unit": "sar",
                        "numerator": None,
                        "denominator": None,
                        "period_start": previous_week_start.isoformat(),
                        "period_end": (previous_week_start + timedelta(days=7)).isoformat()
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inventory_analysis),
                "coverage_notes": [
                    f"Analysis week: {len(inventory_analysis)} SKU records",
                    f"Previous week: {len(inventory_previous)} SKU records",
                    "Only known waste values included; unknown waste preserved"
                ],
                "assumptions": [
                    "Waste rate = units_wasted / (units_sold + units_wasted)",
                    "Known waste cost reflects only recorded waste",
                    "Unknown waste values excluded from calculation"
                ],
                "confidence": 0.65
            })

# ============================================================================
# Output Result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
