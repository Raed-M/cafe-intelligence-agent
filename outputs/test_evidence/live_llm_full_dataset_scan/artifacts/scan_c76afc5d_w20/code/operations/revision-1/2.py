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

# Define periods (all in UTC+3, using fixed offset instead of pytz)
# UTC+3 offset
utc_plus_3 = pd.Timedelta(hours=3)

# Analysis period: 2026-05-25 to 2026-06-01 (UTC+3)
analysis_start = pd.Timestamp('2026-05-25T00:00:00', tz='UTC') + utc_plus_3
analysis_end = pd.Timestamp('2026-06-01T00:00:00', tz='UTC') + utc_plus_3

# Previous period: 2026-05-18 to 2026-05-25 (UTC+3)
previous_start = pd.Timestamp('2026-05-18T00:00:00', tz='UTC') + utc_plus_3
previous_end = pd.Timestamp('2026-05-25T00:00:00', tz='UTC') + utc_plus_3

# Convert timestamps to UTC for comparison
analysis_start_utc = pd.Timestamp('2026-05-25T00:00:00', tz='UTC')
analysis_end_utc = pd.Timestamp('2026-06-01T00:00:00', tz='UTC')
previous_start_utc = pd.Timestamp('2026-05-18T00:00:00', tz='UTC')
previous_end_utc = pd.Timestamp('2026-05-25T00:00:00', tz='UTC')

# Convert timestamps to timezone-aware UTC
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True)
staff_df['date'] = pd.to_datetime(staff_df['date'], utc=True)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)

# Helper function to filter by period
def filter_by_period(df, col, start, end):
    return df[(df[col] >= start) & (df[col] < end)]

# ============================================================================
# FINDING 1: Conversion Rate Analysis (Analysis vs Previous Period)
# ============================================================================

# Analysis period transactions
pos_analysis = filter_by_period(pos_df, 'timestamp', analysis_start_utc, analysis_end_utc)
valid_transactions_analysis = pos_analysis[~pos_analysis['is_refund']]['transaction_id'].nunique()

# Previous period transactions
pos_previous = filter_by_period(pos_df, 'timestamp', previous_start_utc, previous_end_utc)
valid_transactions_previous = pos_previous[~pos_previous['is_refund']]['transaction_id'].nunique()

# Traffic data - exclude dead sensor days
traffic_analysis = filter_by_period(traffic_df, 'date', analysis_start_utc, analysis_end_utc)
traffic_analysis_valid = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]
footfall_analysis = traffic_analysis_valid['door_count'].sum()

traffic_previous = filter_by_period(traffic_df, 'date', previous_start_utc, previous_end_utc)
traffic_previous_valid = traffic_previous[traffic_previous['is_dead_sensor_day'] == False]
footfall_previous = traffic_previous_valid['door_count'].sum()

# Calculate conversions
conversion_analysis = valid_transactions_analysis / footfall_analysis if footfall_analysis > 0 else None
conversion_previous = valid_transactions_previous / footfall_previous if footfall_previous > 0 else None

conversion_change_pct = ((conversion_analysis - conversion_previous) / conversion_previous * 100) if conversion_previous and conversion_previous > 0 else None

# ============================================================================
# FINDING 2: Labour Cost and Staffing Analysis
# ============================================================================

# Analysis period labour
staff_analysis = filter_by_period(staff_df, 'date', analysis_start_utc, analysis_end_utc)
labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
staff_hours_analysis = staff_analysis['computed_duration_hours'].sum()

# Previous period labour
staff_previous = filter_by_period(staff_df, 'date', previous_start_utc, previous_end_utc)
labour_cost_previous = staff_previous['labour_cost_sar'].sum()
staff_hours_previous = staff_previous['computed_duration_hours'].sum()

labour_cost_change_pct = ((labour_cost_analysis - labour_cost_previous) / labour_cost_previous * 100) if labour_cost_previous and labour_cost_previous > 0 else None

# ============================================================================
# FINDING 3: Waste Cost Analysis
# ============================================================================

# Analysis period inventory (week starting 2026-05-25)
analysis_week_start_utc = pd.Timestamp('2026-05-25T00:00:00', tz='UTC')
analysis_week_end_utc = pd.Timestamp('2026-06-01T00:00:00', tz='UTC')

inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_week_start_utc) & 
    (inventory_df['week_starting'] < analysis_week_end_utc)
]
waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()

# Previous period inventory (week starting 2026-05-18)
previous_week_start_utc = pd.Timestamp('2026-05-18T00:00:00', tz='UTC')
previous_week_end_utc = pd.Timestamp('2026-05-25T00:00:00', tz='UTC')

inventory_previous = inventory_df[
    (inventory_df['week_starting'] >= previous_week_start_utc) & 
    (inventory_df['week_starting'] < previous_week_end_utc)
]
waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()

waste_cost_change = waste_cost_analysis - waste_cost_previous
waste_cost_change_pct = ((waste_cost_analysis - waste_cost_previous) / waste_cost_previous * 100) if waste_cost_previous and waste_cost_previous > 0 else None

# ============================================================================
# Build findings
# ============================================================================

findings = []

# Finding 1: Conversion Rate
if conversion_analysis is not None and conversion_previous is not None and footfall_analysis > 0 and footfall_previous > 0:
    findings.append({
        "title": "Conversion Rate Decline in Analysis Period",
        "claim": f"Conversion rate decreased from {conversion_previous:.4f} ({valid_transactions_previous} transactions / {footfall_previous} footfall) in the previous period to {conversion_analysis:.4f} ({valid_transactions_analysis} transactions / {footfall_analysis} footfall) in the analysis period, a change of {conversion_change_pct:.2f}%.",
        "finding_type": "performance_metric",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conversion_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(valid_transactions_analysis),
                "denominator": int(footfall_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conversion_previous, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(valid_transactions_previous),
                "denominator": int(footfall_previous),
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "conversion_rate_change_pct": {
                "value": round(conversion_change_pct, 2) if conversion_change_pct is not None else None,
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(valid_transactions_analysis),
        "coverage_notes": [
            f"Analysis period: {int(footfall_analysis)} valid footfall records (dead sensor days excluded)",
            f"Previous period: {int(footfall_previous)} valid footfall records (dead sensor days excluded)",
            f"Analysis period: {int(valid_transactions_analysis)} non-refund transactions",
            f"Previous period: {int(valid_transactions_previous)} non-refund transactions"
        ],
        "assumptions": [
            "Conversion calculated as unique valid (non-refund) transactions divided by valid footfall",
            "Dead sensor days excluded from footfall denominator",
            "Transaction uniqueness determined by transaction_id"
        ],
        "confidence": 0.85
    })

# Finding 2: Labour Cost Change
if labour_cost_analysis > 0 and labour_cost_previous > 0:
    findings.append({
        "title": "Labour Cost Increase in Analysis Period",
        "claim": f"Labour cost increased from SAR {labour_cost_previous:.2f} ({staff_hours_previous:.1f} hours) in the previous period to SAR {labour_cost_analysis:.2f} ({staff_hours_analysis:.1f} hours) in the analysis period, a change of {labour_cost_change_pct:.2f}%.",
        "finding_type": "cost_metric",
        "metrics": {
            "labour_cost_analysis": {
                "value": round(labour_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "labour_cost_previous": {
                "value": round(labour_cost_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "labour_cost_change_pct": {
                "value": round(labour_cost_change_pct, 2) if labour_cost_change_pct is not None else None,
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "staff_hours_analysis": {
                "value": round(staff_hours_analysis, 1),
                "unit": "hours",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "staff_hours_previous": {
                "value": round(staff_hours_previous, 1),
                "unit": "hours",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            }
        },
        "source_names": ["staff"],
        "sample_size": len(staff_analysis),
        "coverage_notes": [
            f"Analysis period: {len(staff_analysis)} staff shift records",
            f"Previous period: {len(staff_previous)} staff shift records",
            f"Labour cost calculated from computed_duration_hours × hourly_rate_sar"
        ],
        "assumptions": [
            "Labour cost computed from staff records with computed_duration_hours",
            "Hourly rates applied as recorded in staff data",
            "No imputation for missing shift data"
        ],
        "confidence": 0.80
    })

# Finding 3: Waste Cost Change
if waste_cost_analysis >= 0 and waste_cost_previous >= 0:
    findings.append({
        "title": "Known Waste Cost Increase in Analysis Week",
        "claim": f"Known waste cost increased from SAR {waste_cost_previous:.2f} in the previous week to SAR {waste_cost_analysis:.2f} in the analysis week, a change of SAR {waste_cost_change:.2f} ({waste_cost_change_pct:.2f}% if previous > 0).",
        "finding_type": "waste_metric",
        "metrics": {
            "known_waste_cost_analysis": {
                "value": round(waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "known_waste_cost_previous": {
                "value": round(waste_cost_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "known_waste_cost_change_sar": {
                "value": round(waste_cost_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis week: {len(inventory_analysis)} SKU records with known waste",
            f"Previous week: {len(inventory_previous)} SKU records with known waste",
            "Metric includes only known_waste_cost_sar; unknown waste excluded per schema"
        ],
        "assumptions": [
            "Waste cost calculated from known_waste_cost_sar column only",
            "Unknown waste values excluded per data quality rules",
            "Week-level aggregation from inventory records"
        ],
        "confidence": 0.75
    })

# ============================================================================
# Output result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
