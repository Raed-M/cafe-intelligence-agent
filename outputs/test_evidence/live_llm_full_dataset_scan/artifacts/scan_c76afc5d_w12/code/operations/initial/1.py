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
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

# Define analysis periods
analysis_start = datetime(2026, 3, 30, 0, 0, 0, tzinfo=tz)
analysis_end = datetime(2026, 4, 6, 0, 0, 0, tzinfo=tz)
previous_start = datetime(2026, 3, 23, 0, 0, 0, tzinfo=tz)
previous_end = datetime(2026, 3, 30, 0, 0, 0, tzinfo=tz)

findings = []

# FINDING 1: Conversion Rate Analysis
# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)]

# Count unique transactions (not rows)
valid_transactions = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Sum footfall, excluding dead sensor days
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']]
total_footfall = traffic_analysis_valid['door_count'].sum()

if total_footfall > 0:
    conversion_rate = valid_transactions / total_footfall
    
    findings.append({
        "title": "Conversion Rate - Analysis Period",
        "claim": f"During the analysis period (Mar 30 - Apr 6, 2026), the cafe achieved a conversion rate of {conversion_rate:.2%}, with {valid_transactions} valid transactions from {total_footfall} footfall entries.",
        "finding_type": "conversion_metric",
        "metrics": {
            "conversion_rate": {
                "value": round(conversion_rate, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions),
                "denominator": int(total_footfall),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(total_footfall),
        "coverage_notes": [
            f"Excluded {len(traffic_analysis) - len(traffic_analysis_valid)} dead sensor days from denominator",
            f"Used {len(traffic_analysis_valid)} valid traffic days"
        ],
        "assumptions": [
            "Transaction ID uniqueness represents distinct customer baskets",
            "Refunds excluded from valid transaction count",
            "Dead sensor days excluded from footfall denominator"
        ],
        "confidence": 0.85
    })

# FINDING 2: Labour Cost vs Sales Analysis
# Calculate total sales for analysis period (net of refunds)
pos_analysis_sales = pos_analysis.copy()
pos_analysis_sales['net_line_total'] = pos_analysis_sales['line_total_sar'].copy()
pos_analysis_sales.loc[pos_analysis_sales['is_refund'], 'net_line_total'] = -pos_analysis_sales.loc[pos_analysis_sales['is_refund'], 'line_total_sar']
total_sales = pos_analysis_sales['net_line_total'].sum()

# Calculate total labour cost for analysis period
staff_analysis = staff_df[(staff_df['date'] >= analysis_start) & (staff_df['date'] < analysis_end)]
total_labour_cost = staff_analysis['labour_cost_sar'].sum()

if total_sales > 0 and total_labour_cost > 0:
    labour_to_sales_ratio = total_labour_cost / total_sales
    
    findings.append({
        "title": "Labour Cost to Sales Ratio",
        "claim": f"During the analysis period, total labour costs were {total_labour_cost:.2f} SAR against {total_sales:.2f} SAR in net sales, yielding a labour-to-sales ratio of {labour_to_sales_ratio:.2%}.",
        "finding_type": "labour_efficiency",
        "metrics": {
            "labour_to_sales_ratio": {
                "value": round(labour_to_sales_ratio, 4),
                "unit": "ratio",
                "numerator": round(total_labour_cost, 2),
                "denominator": round(total_sales, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_labour_cost_sar": {
                "value": round(total_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_net_sales_sar": {
                "value": round(total_sales, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Staff records: {len(staff_analysis)} entries for analysis period",
            f"POS records: {len(pos_analysis)} line items for analysis period",
            "Refunds netted against sales"
        ],
        "assumptions": [
            "Labour cost includes all computed duration-based costs",
            "Sales include all payment methods",
            "No imputation for missing cashier linkage"
        ],
        "confidence": 0.80
    })

# FINDING 3: Waste Analysis by Category
# Analyze waste by category for analysis period
inventory_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start]

# Group by category (derive from item name if available)
waste_by_item = inventory_analysis.groupby('item').agg({
    'units_wasted': 'sum',
    'known_waste_cost_sar': 'sum',
    'units_sold': 'sum'
}).reset_index()

waste_by_item = waste_by_item[waste_by_item['units_wasted'] > 0].sort_values('known_waste_cost_sar', ascending=False)

if len(waste_by_item) > 0:
    total_waste_cost = waste_by_item['known_waste_cost_sar'].sum()
    total_waste_units = waste_by_item['units_wasted'].sum()
    total_sold_units = waste_by_item['units_sold'].sum()
    
    if total_sold_units > 0:
        waste_to_sales_ratio = total_waste_units / (total_waste_units + total_sold_units)
        
        findings.append({
            "title": "Known Waste Analysis",
            "claim": f"During the analysis period, {total_waste_units:.0f} units were wasted with a known cost of {total_waste_cost:.2f} SAR. The waste-to-total-units ratio was {waste_to_sales_ratio:.2%}, indicating {waste_to_sales_ratio*100:.1f}% of inventory was wasted.",
            "finding_type": "waste_metric",
            "metrics": {
                "total_waste_units": {
                    "value": int(total_waste_units),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_to_total_ratio": {
                    "value": round(waste_to_sales_ratio, 4),
                    "unit": "ratio",
                    "numerator": int(total_waste_units),
                    "denominator": int(total_waste_units + total_sold_units),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_by_item),
            "coverage_notes": [
                "Only known waste values included",
                "Unknown waste values excluded per requirements",
                f"Analysis covers {len(waste_by_item)} items with recorded waste"
            ],
            "assumptions": [
                "Inventory week_starting dates align with analysis period",
                "Known waste cost is accurate and complete",
                "Unknown waste values are not imputed"
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
