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

# Parse analysis period
analysis_start = datetime(2026, 7, 20, 0, 0, 0)
analysis_end = datetime(2026, 7, 27, 0, 0, 0)
previous_start = datetime(2026, 7, 13, 0, 0, 0)
previous_end = datetime(2026, 7, 20, 0, 0, 0)

# Convert POS timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])

# Convert traffic date to datetime
traffic_df['date'] = pd.to_datetime(traffic_df['date'])

# Convert staff date to datetime
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'])
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'])

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

# Filter POS for analysis period (exclude refunds for transaction count)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Count unique valid transactions
valid_transactions_analysis = pos_analysis['transaction_id'].nunique()

# Filter traffic for analysis period, exclude dead sensor days
traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start.date()) & 
    (traffic_df['date'] < analysis_end.date()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

total_footfall_analysis = traffic_analysis['door_count'].sum()

# Previous period
pos_previous = pos_df[
    (pos_df['timestamp'] >= previous_start) & 
    (pos_df['timestamp'] < previous_end) &
    (pos_df['is_refund'] == False)
].copy()

valid_transactions_previous = pos_previous['transaction_id'].nunique()

traffic_previous = traffic_df[
    (traffic_df['date'] >= previous_start.date()) & 
    (traffic_df['date'] < previous_end.date()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

total_footfall_previous = traffic_previous['door_count'].sum()

# Calculate conversion rates
if total_footfall_analysis > 0:
    conversion_analysis = valid_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

if total_footfall_previous > 0:
    conversion_previous = valid_transactions_previous / total_footfall_previous
else:
    conversion_previous = None

# Only create finding if both periods have valid data
if conversion_analysis is not None and conversion_previous is not None:
    conversion_change = conversion_analysis - conversion_previous
    
    analysis_start_iso = analysis_start.isoformat() + "+03:00"
    analysis_end_iso = analysis_end.isoformat() + "+03:00"
    previous_start_iso = previous_start.isoformat() + "+03:00"
    previous_end_iso = previous_end.isoformat() + "+03:00"
    
    finding_1 = {
        "title": "Conversion Rate Comparison: Analysis Week vs Previous Week",
        "claim": f"Conversion rate (valid transactions / footfall) in analysis week (20-27 Jul) was {conversion_analysis:.4f} vs {conversion_previous:.4f} in previous week (13-20 Jul), a change of {conversion_change:+.4f}",
        "finding_type": "operational_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions_analysis),
                "denominator": int(total_footfall_analysis),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "conversion_rate_previous": {
                "value": round(conversion_previous, 4),
                "unit": "ratio",
                "numerator": int(valid_transactions_previous),
                "denominator": int(total_footfall_previous),
                "period_start": previous_start_iso,
                "period_end": previous_end_iso
            },
            "conversion_rate_change": {
                "value": round(conversion_change, 4),
                "unit": "ratio",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(valid_transactions_analysis),
        "coverage_notes": [
            f"Analysis period: {len(traffic_analysis)} traffic days with valid sensors",
            f"Previous period: {len(traffic_previous)} traffic days with valid sensors",
            "Dead sensor days excluded from footfall denominator",
            "Refunds excluded from transaction count"
        ],
        "assumptions": [
            "Each transaction_id represents one unique customer visit",
            "Traffic sensor data is accurate for non-dead-sensor days",
            "Conversion metric is valid transactions / footfall"
        ],
        "confidence": 0.85
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Labour Cost vs Demand by Day
# ============================================================================

# Calculate daily labour cost for analysis period
staff_analysis = staff_df[
    (staff_df['date'] >= analysis_start.date()) & 
    (staff_df['date'] < analysis_end.date())
].copy()

daily_labour = staff_analysis.groupby('date').agg({
    'labour_cost_sar': 'sum',
    'employee_id': 'count'
}).reset_index()
daily_labour.columns = ['date', 'total_labour_cost_sar', 'staff_count']

# Calculate daily transaction count for analysis period
pos_analysis_daily = pos_analysis.groupby('business_date').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum'
}).reset_index()
pos_analysis_daily.columns = ['date', 'transaction_count', 'revenue_sar']

# Merge labour and demand
daily_metrics = daily_labour.merge(pos_analysis_daily, on='date', how='inner')

if len(daily_metrics) > 0:
    # Calculate labour cost per transaction
    daily_metrics['labour_cost_per_transaction'] = daily_metrics['total_labour_cost_sar'] / daily_metrics['transaction_count']
    
    avg_labour_per_transaction = daily_metrics['labour_cost_per_transaction'].mean()
    avg_staff_count = daily_metrics['staff_count'].mean()
    avg_daily_transactions = daily_metrics['transaction_count'].mean()
    
    analysis_start_iso = analysis_start.isoformat() + "+03:00"
    analysis_end_iso = analysis_end.isoformat() + "+03:00"
    
    finding_2 = {
        "title": "Daily Labour Cost and Transaction Volume Alignment",
        "claim": f"During analysis week (20-27 Jul), average daily labour cost was {daily_metrics['total_labour_cost_sar'].mean():.0f} SAR with {avg_staff_count:.1f} staff on average, serving {avg_daily_transactions:.0f} transactions/day. Labour cost per transaction averaged {avg_labour_per_transaction:.2f} SAR.",
        "finding_type": "operational_metric",
        "metrics": {
            "avg_daily_labour_cost": {
                "value": round(daily_metrics['total_labour_cost_sar'].mean(), 2),
                "unit": "SAR",
                "numerator": round(daily_metrics['total_labour_cost_sar'].sum(), 2),
                "denominator": len(daily_metrics),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "avg_staff_count": {
                "value": round(avg_staff_count, 1),
                "unit": "employees",
                "numerator": int(daily_metrics['staff_count'].sum()),
                "denominator": len(daily_metrics),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "avg_daily_transactions": {
                "value": round(avg_daily_transactions, 0),
                "unit": "transactions",
                "numerator": int(daily_metrics['transaction_count'].sum()),
                "denominator": len(daily_metrics),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "labour_cost_per_transaction": {
                "value": round(avg_labour_per_transaction, 2),
                "unit": "SAR/transaction",
                "numerator": round(daily_metrics['total_labour_cost_sar'].sum(), 2),
                "denominator": int(daily_metrics['transaction_count'].sum()),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(daily_metrics),
        "coverage_notes": [
            f"Analysis covers {len(daily_metrics)} days with both staff and POS data",
            "Labour cost includes all shifts on each date",
            "Transactions exclude refunds"
        ],
        "assumptions": [
            "Staff labour_cost_sar is accurate and complete",
            "Each transaction_id is unique per visit",
            "Daily alignment uses business_date from POS and date from staff"
        ],
        "confidence": 0.80
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Waste and Ordering Patterns
# ============================================================================

# Filter inventory for analysis week
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Filter inventory for previous week
inventory_previous = inventory_df[
    (inventory_df['week_starting'] >= previous_start) & 
    (inventory_df['week_starting'] < previous_end)
].copy()

if len(inventory_analysis) > 0 and len(inventory_previous) > 0:
    # Calculate waste metrics
    analysis_total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    analysis_total_ordered = inventory_analysis['units_ordered'].sum()
    analysis_total_sold = inventory_analysis['units_sold'].sum()
    analysis_total_wasted = inventory_analysis['units_wasted'].sum()
    
    previous_total_waste_cost = inventory_previous['known_waste_cost_sar'].sum()
    previous_total_ordered = inventory_previous['units_ordered'].sum()
    previous_total_sold = inventory_previous['units_sold'].sum()
    previous_total_wasted = inventory_previous['units_wasted'].sum()
    
    # Calculate waste rate (wasted / ordered)
    if analysis_total_ordered > 0:
        analysis_waste_rate = analysis_total_wasted / analysis_total_ordered
    else:
        analysis_waste_rate = 0
    
    if previous_total_ordered > 0:
        previous_waste_rate = previous_total_wasted / previous_total_ordered
    else:
        previous_waste_rate = 0
    
    waste_rate_change = analysis_waste_rate - previous_waste_rate
    
    analysis_start_iso = analysis_start.isoformat() + "+03:00"
    analysis_end_iso = analysis_end.isoformat() + "+03:00"
    previous_start_iso = previous_start.isoformat() + "+03:00"
    previous_end_iso = previous_end.isoformat() + "+03:00"
    
    finding_3 = {
        "title": "Inventory Waste Rate Comparison",
        "claim": f"Known waste rate in analysis week (20-27 Jul) was {analysis_waste_rate:.4f} ({int(analysis_total_wasted)} units wasted of {int(analysis_total_ordered)} ordered) vs {previous_waste_rate:.4f} in previous week (13-20 Jul), a change of {waste_rate_change:+.4f}. Known waste cost was {analysis_total_waste_cost:.2f} SAR vs {previous_total_waste_cost:.2f} SAR.",
        "finding_type": "operational_metric",
        "metrics": {
            "waste_rate_analysis": {
                "value": round(analysis_waste_rate, 4),
                "unit": "ratio",
                "numerator": int(analysis_total_wasted),
                "denominator": int(analysis_total_ordered),
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "waste_rate_previous": {
                "value": round(previous_waste_rate, 4),
                "unit": "ratio",
                "numerator": int(previous_total_wasted),
                "denominator": int(previous_total_ordered),
                "period_start": previous_start_iso,
                "period_end": previous_end_iso
            },
            "waste_rate_change": {
                "value": round(waste_rate_change, 4),
                "unit": "ratio",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "known_waste_cost_analysis": {
                "value": round(analysis_total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(analysis_total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start_iso,
                "period_end": analysis_end_iso
            },
            "known_waste_cost_previous": {
                "value": round(previous_total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(previous_total_waste_cost, 2),
                "denominator": None,
                "period_start": previous_start_iso,
                "period_end": previous_end_iso
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis week inventory records: {len(inventory_analysis)} SKUs",
            f"Previous week inventory records: {len(inventory_previous)} SKUs",
            "Waste metric includes only known_waste_cost_sar (unknown waste excluded)",
            "Waste rate calculated as units_wasted / units_ordered"
        ],
        "assumptions": [
            "units_wasted and known_waste_cost_sar are accurate for recorded items",
            "Unknown waste is not included in this analysis",
            "Week-level inventory data is representative of actual waste"
        ],
        "confidence": 0.75
    }
    findings.append(finding_3)

# ============================================================================
# Output Result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
