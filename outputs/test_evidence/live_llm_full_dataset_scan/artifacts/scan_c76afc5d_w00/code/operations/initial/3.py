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

# Define periods using ISO format strings and manual timezone offset
# Asia/Riyadh is UTC+3
analysis_start_str = "2026-01-05T00:00:00+03:00"
analysis_end_str = "2026-01-12T00:00:00+03:00"
prev_start_str = "2025-12-29T00:00:00+03:00"
prev_end_str = "2026-01-05T00:00:00+03:00"

# Parse ISO format strings
analysis_start = pd.to_datetime(analysis_start_str)
analysis_end = pd.to_datetime(analysis_end_str)
prev_start = pd.to_datetime(prev_start_str)
prev_end = pd.to_datetime(prev_end_str)

# Convert to date objects for comparison with date columns
analysis_start_date = analysis_start.date()
analysis_end_date = analysis_end.date()

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Convert traffic date to datetime
traffic_df['date'] = pd.to_datetime(traffic_df['date'])

# Filter traffic for analysis period (exclude dead sensor days)
traffic_analysis = traffic_df[
    (traffic_df['date'].dt.date >= analysis_start_date) &
    (traffic_df['date'].dt.date < analysis_end_date) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

# Convert POS timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) &
    (pos_df['timestamp'] < analysis_end)
].copy()

# Count unique valid transactions (exclude refunds for conversion numerator)
valid_transactions = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()

# Sum footfall from valid traffic days
total_footfall = traffic_analysis['door_count'].sum()

if total_footfall > 0:
    conversion_rate = valid_transactions / total_footfall
    
    findings.append({
        "title": "Conversion Rate (Analysis Period)",
        "claim": f"During the analysis period (2026-01-05 to 2026-01-12), the cafe converted {conversion_rate:.2%} of foot traffic into sales transactions, with {valid_transactions} valid transactions from {total_footfall} visitors across {len(traffic_analysis)} days of valid sensor data.",
        "finding_type": "conversion_metric",
        "metrics": {
            "conversion_rate": {
                "value": round(conversion_rate, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions),
                "denominator": int(total_footfall),
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "valid_transactions": {
                "value": int(valid_transactions),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "total_footfall": {
                "value": int(total_footfall),
                "unit": "visitors",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": len(traffic_analysis),
        "coverage_notes": [
            f"Analysis period: {len(traffic_analysis)} days with valid sensor data",
            f"Excluded {len(traffic_df[(traffic_df['date'].dt.date >= analysis_start_date) & (traffic_df['date'].dt.date < analysis_end_date) & (traffic_df['is_dead_sensor_day'] == True)])} dead sensor days",
            "Conversion numerator excludes refunds"
        ],
        "assumptions": [
            "Unique transaction_id represents a basket/sale event",
            "Door count is accurate for valid sensor days",
            "Refunds are excluded from conversion numerator"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Revenue Analysis
# ============================================================================

# Filter staff for analysis period
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_analysis = staff_df[
    (staff_df['date'].dt.date >= analysis_start_date) &
    (staff_df['date'].dt.date < analysis_end_date)
].copy()

# Calculate total labour cost
total_labour_cost = staff_analysis['labour_cost_sar'].sum()

# Calculate net revenue (excluding refunds)
pos_analysis['line_total_net'] = pos_analysis['line_total_sar']
revenue_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()

if revenue_analysis > 0:
    labour_to_revenue_ratio = total_labour_cost / revenue_analysis
    
    findings.append({
        "title": "Labour Cost to Revenue Ratio",
        "claim": f"During the analysis period, labour costs were {labour_to_revenue_ratio:.2%} of gross revenue, with {total_labour_cost:.2f} SAR in labour costs against {revenue_analysis:.2f} SAR in sales revenue across {len(staff_analysis)} staff shifts.",
        "finding_type": "cost_efficiency",
        "metrics": {
            "labour_to_revenue_ratio": {
                "value": round(labour_to_revenue_ratio, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost, 2),
                "denominator": round(revenue_analysis, 2),
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "total_labour_cost_sar": {
                "value": round(total_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            },
            "gross_revenue_sar": {
                "value": round(revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_str,
                "period_end": analysis_end_str
            }
        },
        "source_names": ["staff", "pos"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Staff shifts: {len(staff_analysis)} records",
            "Labour cost includes computed_duration_hours × hourly_rate_sar",
            "Revenue excludes refunds (is_refund == False)"
        ],
        "assumptions": [
            "labour_cost_sar is accurately computed from shift duration and hourly rate",
            "All staff shifts in period are captured",
            "Revenue calculation excludes refunds"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Filter inventory for analysis period
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inventory_analysis = inventory_df[
    (inventory_df['week_starting'].dt.date >= analysis_start_date) &
    (inventory_df['week_starting'].dt.date < analysis_end_date)
].copy()

if len(inventory_analysis) > 0:
    total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_wasted = inventory_analysis['units_wasted'].sum()
    total_units_sold = inventory_analysis['units_sold'].sum()
    
    if total_units_sold > 0:
        waste_to_sales_ratio = total_units_wasted / total_units_sold
        
        findings.append({
            "title": "Known Waste to Sales Ratio",
            "claim": f"During the analysis period, known waste represented {waste_to_sales_ratio:.2%} of units sold, with {total_units_wasted} units wasted (cost: {total_waste_cost:.2f} SAR) against {total_units_sold} units sold across {len(inventory_analysis)} SKU-week records.",
            "finding_type": "waste_efficiency",
            "metrics": {
                "waste_to_sales_ratio": {
                    "value": round(waste_to_sales_ratio, 4),
                    "unit": "ratio",
                    "numerator": int(total_units_wasted),
                    "denominator": int(total_units_sold),
                    "period_start": analysis_start_str,
                    "period_end": analysis_end_str
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_str,
                    "period_end": analysis_end_str
                },
                "known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_str,
                    "period_end": analysis_end_str
                },
                "total_units_sold": {
                    "value": int(total_units_sold),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_str,
                    "period_end": analysis_end_str
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                f"Inventory records: {len(inventory_analysis)} SKU-week combinations",
                "Waste values are known/recorded only; unknown waste is excluded",
                "Analysis period week_starting dates within analysis window"
            ],
            "assumptions": [
                "units_wasted and units_sold are accurately recorded",
                "known_waste_cost_sar reflects actual waste cost",
                "Unknown waste is not included in this analysis"
            ],
            "confidence": 0.75
        })

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
