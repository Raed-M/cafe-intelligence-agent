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

# Define periods using UTC offset for Asia/Riyadh (+03:00)
utc_offset = timedelta(hours=3)

analysis_start = datetime(2026, 1, 26, 0, 0, 0)
analysis_end = datetime(2026, 2, 2, 0, 0, 0)
prev_start = datetime(2026, 1, 19, 0, 0, 0)
prev_end = datetime(2026, 1, 26, 0, 0, 0)

# Convert timestamps to UTC-aware datetime for comparison
# Assume input timestamps are in Asia/Riyadh timezone
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True, format='mixed')
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True, format='mixed')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Create timezone-aware comparison datetimes (UTC)
analysis_start_utc = pd.Timestamp(analysis_start, tz='UTC')
analysis_end_utc = pd.Timestamp(analysis_end, tz='UTC')
prev_start_utc = pd.Timestamp(prev_start, tz='UTC')
prev_end_utc = pd.Timestamp(prev_end, tz='UTC')

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (Sales Transactions vs Footfall)
# ============================================================================

# Filter POS for analysis period (exclude refunds for transaction count)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_utc) & 
    (pos_df['timestamp'] < analysis_end_utc) &
    (pos_df['is_refund'] == False)
].copy()

# Count unique valid transactions
unique_transactions_analysis = pos_analysis['transaction_id'].nunique()

# Filter traffic for analysis period, exclude dead sensor days
traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start) & 
    (traffic_df['date'] < analysis_end) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

total_footfall_analysis = traffic_analysis['door_count'].sum()
valid_traffic_days_analysis = traffic_analysis['date'].nunique()

# Previous period for comparison
pos_prev = pos_df[
    (pos_df['timestamp'] >= prev_start_utc) & 
    (pos_df['timestamp'] < prev_end_utc) &
    (pos_df['is_refund'] == False)
].copy()

unique_transactions_prev = pos_prev['transaction_id'].nunique()

traffic_prev = traffic_df[
    (traffic_df['date'] >= prev_start) & 
    (traffic_df['date'] < prev_end) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

total_footfall_prev = traffic_prev['door_count'].sum()
valid_traffic_days_prev = traffic_prev['date'].nunique()

# Calculate conversion rates
if total_footfall_analysis > 0:
    conversion_analysis = unique_transactions_analysis / total_footfall_analysis
else:
    conversion_analysis = None

if total_footfall_prev > 0:
    conversion_prev = unique_transactions_prev / total_footfall_prev
else:
    conversion_prev = None

if conversion_analysis is not None and conversion_prev is not None:
    conversion_change = ((conversion_analysis - conversion_prev) / conversion_prev) * 100
    
    findings.append({
        "title": "Conversion Rate Decline Week-over-Week",
        "claim": f"Conversion rate decreased from {conversion_prev:.4f} (previous week) to {conversion_analysis:.4f} (analysis week), a {conversion_change:.1f}% decline. This represents {unique_transactions_analysis} transactions from {total_footfall_analysis} valid footfall events.",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": unique_transactions_analysis,
                "denominator": total_footfall_analysis,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "conversion_rate_previous": {
                "value": round(conversion_prev, 4),
                "unit": "transactions_per_visitor",
                "numerator": unique_transactions_prev,
                "denominator": total_footfall_prev,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "conversion_change_pct": {
                "value": round(conversion_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": unique_transactions_analysis,
        "coverage_notes": [
            f"Analysis period: {valid_traffic_days_analysis} days with valid traffic sensors",
            f"Previous period: {valid_traffic_days_prev} days with valid traffic sensors",
            "Excluded refunds from transaction count",
            "Excluded dead sensor days from footfall denominator"
        ],
        "assumptions": [
            "Each transaction_id represents one unique customer visit",
            "Door count accurately reflects cafe visitors",
            "Dead sensor days properly flagged in traffic data"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Demand Alignment
# ============================================================================

# Calculate hourly demand (transactions per hour) for analysis period
pos_analysis['hour_date'] = pos_analysis['timestamp'].dt.floor('h')
hourly_transactions = pos_analysis.groupby('hour_date')['transaction_id'].nunique().reset_index()
hourly_transactions.columns = ['hour_date', 'transaction_count']

# Calculate hourly footfall for analysis period
traffic_analysis['hour'] = traffic_analysis['date'].dt.floor('h')
hourly_footfall = traffic_analysis.groupby('hour')['door_count'].sum().reset_index()
hourly_footfall.columns = ['hour_date', 'footfall']

# Merge hourly data
hourly_demand = hourly_transactions.merge(hourly_footfall, on='hour_date', how='inner')

# Calculate staff hours by date for analysis period
staff_analysis = staff_df[
    (staff_df['date'] >= analysis_start) & 
    (staff_df['date'] < analysis_end)
].copy()

staff_by_date = staff_analysis.groupby('date').agg({
    'computed_duration_hours': 'sum',
    'labour_cost_sar': 'sum'
}).reset_index()
staff_by_date.columns = ['date', 'total_hours', 'total_labour_cost']

# Calculate daily transactions for analysis period
pos_analysis['date'] = pos_analysis['timestamp'].dt.date
daily_transactions = pos_analysis.groupby('date')['transaction_id'].nunique().reset_index()
daily_transactions.columns = ['date', 'daily_transactions']

# Merge daily data
daily_demand = daily_transactions.merge(staff_by_date, on='date', how='inner')

if len(daily_demand) > 0:
    # Calculate labour cost per transaction
    daily_demand['labour_cost_per_transaction'] = daily_demand['total_labour_cost'] / daily_demand['daily_transactions']
    
    avg_labour_cost_per_transaction = daily_demand['labour_cost_per_transaction'].mean()
    avg_daily_transactions = daily_demand['daily_transactions'].mean()
    avg_daily_labour_cost = daily_demand['total_labour_cost'].mean()
    
    findings.append({
        "title": "Labour Cost Efficiency During Analysis Period",
        "claim": f"Average labour cost per transaction during analysis week was {avg_labour_cost_per_transaction:.2f} SAR, with {avg_daily_transactions:.0f} average daily transactions and {avg_daily_labour_cost:.2f} SAR average daily labour cost. Staff scheduling shows {len(staff_analysis)} shifts across {len(daily_demand)} days.",
        "finding_type": "operational_metric",
        "metrics": {
            "avg_labour_cost_per_transaction": {
                "value": round(avg_labour_cost_per_transaction, 2),
                "unit": "SAR_per_transaction",
                "numerator": round(daily_demand['total_labour_cost'].sum(), 2),
                "denominator": int(daily_demand['daily_transactions'].sum()),
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "avg_daily_transactions": {
                "value": round(avg_daily_transactions, 0),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "avg_daily_labour_cost": {
                "value": round(avg_daily_labour_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "staff"],
        "sample_size": len(daily_demand),
        "coverage_notes": [
            f"Analysis covers {len(daily_demand)} days with both POS and staff data",
            f"Total staff shifts in period: {len(staff_analysis)}",
            "Labour cost includes computed_duration_hours × hourly_rate_sar"
        ],
        "assumptions": [
            "computed_duration_hours accurately reflects actual time on floor",
            "All staff shifts are properly recorded",
            "Labour cost is proportional to transaction volume"
        ],
        "confidence": 0.80
    })

# ============================================================================
# FINDING 3: Inventory Waste and Stock Movement
# ============================================================================

# Filter inventory for analysis week
inv_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Filter for previous week
inv_prev = inventory_df[
    (inventory_df['week_starting'] >= prev_start) & 
    (inventory_df['week_starting'] < prev_end)
].copy()

if len(inv_analysis) > 0:
    total_units_ordered_analysis = inv_analysis['units_ordered'].sum()
    total_units_sold_analysis = inv_analysis['units_sold'].sum()
    total_units_wasted_analysis = inv_analysis['units_wasted'].sum()
    total_known_waste_cost_analysis = inv_analysis['known_waste_cost_sar'].sum()
    
    # Calculate waste rate
    if total_units_ordered_analysis > 0:
        waste_rate_analysis = (total_units_wasted_analysis / total_units_ordered_analysis) * 100
    else:
        waste_rate_analysis = None
    
    if len(inv_prev) > 0:
        total_units_ordered_prev = inv_prev['units_ordered'].sum()
        total_units_sold_prev = inv_prev['units_sold'].sum()
        total_units_wasted_prev = inv_prev['units_wasted'].sum()
        total_known_waste_cost_prev = inv_prev['known_waste_cost_sar'].sum()
        
        if total_units_ordered_prev > 0:
            waste_rate_prev = (total_units_wasted_prev / total_units_ordered_prev) * 100
        else:
            waste_rate_prev = None
        
        if waste_rate_analysis is not None and waste_rate_prev is not None:
            findings.append({
                "title": "Inventory Waste Rate Comparison",
                "claim": f"Known waste rate in analysis week was {waste_rate_analysis:.1f}% ({total_units_wasted_analysis} units wasted from {total_units_ordered_analysis} ordered), compared to {waste_rate_prev:.1f}% in previous week. Known waste cost was {total_known_waste_cost_analysis:.2f} SAR (analysis) vs {total_known_waste_cost_prev:.2f} SAR (previous).",
                "finding_type": "inventory_metric",
                "metrics": {
                    "waste_rate_analysis": {
                        "value": round(waste_rate_analysis, 1),
                        "unit": "percent",
                        "numerator": total_units_wasted_analysis,
                        "denominator": total_units_ordered_analysis,
                        "period_start": "2026-01-26T00:00:00+03:00",
                        "period_end": "2026-02-02T00:00:00+03:00"
                    },
                    "waste_rate_previous": {
                        "value": round(waste_rate_prev, 1),
                        "unit": "percent",
                        "numerator": total_units_wasted_prev,
                        "denominator": total_units_ordered_prev,
                        "period_start": "2026-01-19T00:00:00+03:00",
                        "period_end": "2026-01-26T00:00:00+03:00"
                    },
                    "known_waste_cost_analysis": {
                        "value": round(total_known_waste_cost_analysis, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-01-26T00:00:00+03:00",
                        "period_end": "2026-02-02T00:00:00+03:00"
                    },
                    "known_waste_cost_previous": {
                        "value": round(total_known_waste_cost_prev, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-01-19T00:00:00+03:00",
                        "period_end": "2026-01-26T00:00:00+03:00"
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inv_analysis),
                "coverage_notes": [
                    f"Analysis week: {len(inv_analysis)} SKU records",
                    f"Previous week: {len(inv_prev)} SKU records",
                    "Waste values include both known and unknown waste",
                    "known_waste_cost_sar reflects only documented waste"
                ],
                "assumptions": [
                    "units_wasted accurately reflects actual waste",
                    "known_waste_cost_sar is properly calculated from unit_cost_sar",
                    "Unknown waste is preserved in units_wasted but not in cost calculations"
                ],
                "confidence": 0.75
            })

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
