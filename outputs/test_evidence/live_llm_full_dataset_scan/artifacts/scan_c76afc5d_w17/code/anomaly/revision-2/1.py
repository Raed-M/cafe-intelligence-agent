import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define analysis periods
analysis_period = {
    "start": "2026-05-04T00:00:00+03:00",
    "end": "2026-05-11T00:00:00+03:00"
}
previous_period = {
    "start": "2026-04-27T00:00:00+03:00",
    "end": "2026-05-04T00:00:00+03:00"
}
trailing_baseline_periods = [
    {"start": "2026-04-27T00:00:00+03:00", "end": "2026-05-04T00:00:00+03:00"},
    {"start": "2026-04-20T00:00:00+03:00", "end": "2026-04-27T00:00:00+03:00"},
    {"start": "2026-04-13T00:00:00+03:00", "end": "2026-04-20T00:00:00+03:00"},
    {"start": "2026-04-06T00:00:00+03:00", "end": "2026-04-13T00:00:00+03:00"}
]

def parse_iso_date(iso_str):
    """Parse ISO 8601 datetime string to naive datetime."""
    return pd.to_datetime(iso_str).tz_localize(None)

analysis_start = parse_iso_date(analysis_period["start"])
analysis_end = parse_iso_date(analysis_period["end"])
baseline_starts = [parse_iso_date(p["start"]) for p in trailing_baseline_periods]
baseline_ends = [parse_iso_date(p["end"]) for p in trailing_baseline_periods]

# Read data
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baselines
pos_df['date'] = pos_df['timestamp_local'].dt.date

# Analysis period daily revenue
analysis_pos = pos_df[(pos_df['timestamp_local'] >= analysis_start) & 
                      (pos_df['timestamp_local'] < analysis_end)]
analysis_daily_revenue = analysis_pos.groupby('date')['line_total_sar'].sum()

# Baseline daily revenue (all trailing periods combined)
baseline_pos = pos_df[(pos_df['timestamp_local'] >= baseline_starts[0]) & 
                      (pos_df['timestamp_local'] < baseline_ends[-1])]
baseline_daily_revenue = baseline_pos.groupby('date')['line_total_sar'].sum()

if len(baseline_daily_revenue) > 1 and baseline_daily_revenue.std() > 0:
    baseline_mean = baseline_daily_revenue.mean()
    baseline_std = baseline_daily_revenue.std()
    
    # Find anomalies in analysis period
    for date, revenue in analysis_daily_revenue.items():
        z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # Flag if |z_score| > 2
        if abs(z_score) > 2.0:
            findings.append({
                "title": f"Unusual Daily Revenue on {date}",
                "claim": f"Daily revenue of {revenue:.2f} SAR on {date} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "observed_daily_revenue": {
                        "value": round(revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": str(date),
                        "period_end": str(date)
                    },
                    "baseline_mean_daily_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-06",
                        "period_end": "2026-05-04"
                    },
                    "z_score": {
                        "value": round(z_score, 2),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": str(date),
                        "period_end": str(date)
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenue),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: {baseline_starts[0].date()} to {baseline_ends[-1].date()} (4 weeks)",
                    f"Baseline sample size: {len(baseline_daily_revenue)} days"
                ],
                "assumptions": [
                    "Z-score threshold: |z| > 2.0 (p < 0.05)",
                    "Baseline computed from trailing 4 weeks",
                    "Revenue includes refunds as net values per schema"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = traffic_df['date'].dt.date
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Analysis period hourly traffic
analysis_traffic = traffic_clean[(traffic_clean['date'] >= analysis_start.date()) & 
                                 (traffic_clean['date'] < analysis_end.date())]

# Baseline hourly traffic
baseline_traffic = traffic_clean[(traffic_clean['date'] >= baseline_starts[0].date()) & 
                                 (traffic_clean['date'] < baseline_ends[-1].date())]

if len(baseline_traffic) > 10 and baseline_traffic['door_count'].std() > 0:
    baseline_mean_traffic = baseline_traffic['door_count'].mean()
    baseline_std_traffic = baseline_traffic['door_count'].std()
    
    # Find anomalies in analysis period
    anomaly_hours = []
    for idx, row in analysis_traffic.iterrows():
        door_count = row['door_count']
        z_score = (door_count - baseline_mean_traffic) / baseline_std_traffic if baseline_std_traffic > 0 else 0
        
        if abs(z_score) > 2.0:
            anomaly_hours.append({
                "date": row['date'],
                "hour": int(row['hour']),
                "door_count": door_count,
                "z_score": z_score
            })
    
    # Sort by magnitude and take top 1
    if anomaly_hours:
        anomaly_hours.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_anomaly = anomaly_hours[0]
        
        findings.append({
            "title": f"Unusual Hourly Traffic on {top_anomaly['date']} Hour {top_anomaly['hour']}",
            "claim": f"Door count of {top_anomaly['door_count']} at hour {top_anomaly['hour']} on {top_anomaly['date']} deviates {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.",
            "finding_type": "traffic_anomaly",
            "metrics": {
                "observed_hourly_door_count": {
                    "value": int(top_anomaly['door_count']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": f"{top_anomaly['date']}T{top_anomaly['hour']:02d}:00:00",
                    "period_end": f"{top_anomaly['date']}T{top_anomaly['hour']:02d}:59:59"
                },
                "baseline_mean_hourly_door_count": {
                    "value": round(baseline_mean_traffic, 1),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-06",
                    "period_end": "2026-05-04"
                },
                "z_score": {
                    "value": round(top_anomaly['z_score'], 2),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": f"{top_anomaly['date']}T{top_anomaly['hour']:02d}:00:00",
                    "period_end": f"{top_anomaly['date']}T{top_anomaly['hour']:02d}:59:59"
                }
            },
            "source_names": ["traffic"],
            "sample_size": len(baseline_traffic),
            "coverage_notes": [
                f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                f"Baseline: {baseline_starts[0].date()} to {baseline_ends[-1].date()} (4 weeks)",
                f"Dead sensor days excluded per is_dead_sensor_day flag",
                f"Baseline sample size: {len(baseline_traffic)} hourly observations"
            ],
            "assumptions": [
                "Z-score threshold: |z| > 2.0 (p < 0.05)",
                "Baseline computed from trailing 4 weeks, excluding dead sensor intervals",
                "Hourly observations treated as independent"
            ],
            "confidence": 0.80
        })

# ============================================================================
# ANOMALY 3: Weekly Waste Analysis
# ============================================================================

# Filter to analysis and baseline weeks
analysis_week_start = analysis_start.date()
baseline_week_starts = [p_start.date() for p_start in baseline_starts]

analysis_inventory = inventory_df[inventory_df['week_starting'].dt.date == analysis_week_start]
baseline_inventory = inventory_df[inventory_df['week_starting'].dt.date.isin(baseline_week_starts)]

# Calculate total waste per week
if len(baseline_inventory) > 1:
    baseline_waste = baseline_inventory['units_wasted'].sum()
    baseline_waste_cost = baseline_inventory['known_waste_cost_sar'].sum()
    baseline_weeks = len(baseline_inventory['week_starting'].unique())
    
    if baseline_weeks > 0:
        baseline_mean_waste = baseline_waste / baseline_weeks
        baseline_mean_waste_cost = baseline_waste_cost / baseline_weeks
        
        analysis_waste = analysis_inventory['units_wasted'].sum()
        analysis_waste_cost = analysis_inventory['known_waste_cost_sar'].sum()
        
        # Calculate z-score for waste units
        if baseline_weeks > 1:
            baseline_waste_std = baseline_inventory.groupby('week_starting')['units_wasted'].sum().std()
            
            if baseline_waste_std > 0:
                z_score_waste = (analysis_waste - baseline_mean_waste) / baseline_waste_std
                
                if abs(z_score_waste) > 2.0:
                    findings.append({
                        "title": f"Unusual Weekly Waste for Week Starting {analysis_week_start}",
                        "claim": f"Weekly waste of {analysis_waste:.0f} units ({analysis_waste_cost:.2f} SAR) deviates {abs(z_score_waste):.2f} standard deviations from baseline mean of {baseline_mean_waste:.1f} units ({baseline_mean_waste_cost:.2f} SAR).",
                        "finding_type": "waste_anomaly",
                        "metrics": {
                            "observed_weekly_units_wasted": {
                                "value": round(analysis_waste, 1),
                                "unit": "units",
                                "numerator": None,
                                "denominator": None,
                                "period_start": str(analysis_week_start),
                                "period_end": str(analysis_end.date())
                            },
                            "observed_weekly_waste_cost": {
                                "value": round(analysis_waste_cost, 2),
                                "unit": "SAR",
                                "numerator": None,
                                "denominator": None,
                                "period_start": str(analysis_week_start),
                                "period_end": str(analysis_end.date())
                            },
                            "baseline_mean_weekly_units_wasted": {
                                "value": round(baseline_mean_waste, 1),
                                "unit": "units",
                                "numerator": None,
                                "denominator": None,
                                "period_start": "2026-04-06",
                                "period_end": "2026-05-04"
                            },
                            "baseline_mean_weekly_waste_cost": {
                                "value": round(baseline_mean_waste_cost, 2),
                                "unit": "SAR",
                                "numerator": None,
                                "denominator": None,
                                "period_start": "2026-04-06",
                                "period_end": "2026-05-04"
                            },
                            "z_score": {
                                "value": round(z_score_waste, 2),
                                "unit": None,
                                "numerator": None,
                                "denominator": None,
                                "period_start": str(analysis_week_start),
                                "period_end": str(analysis_end.date())
                            }
                        },
                        "source_names": ["inventory"],
                        "sample_size": baseline_weeks,
                        "coverage_notes": [
                            f"Analysis period: week starting {analysis_week_start}",
                            f"Baseline: 4 weeks from {baseline_week_starts[0]} to {baseline_week_starts[-1]}",
                            f"Baseline sample size: {baseline_weeks} weeks",
                            "Unknown waste values excluded per schema"
                        ],
                        "assumptions": [
                            "Z-score threshold: |z| > 2.0 (p < 0.05)",
                            "Baseline computed from trailing 4 weeks",
                            "Waste cost uses known_waste_cost_sar field"
                        ],
                        "confidence": 0.75
                    })

# Sort findings by confidence and limit to 3
findings.sort(key=lambda x: x['confidence'], reverse=True)
findings = findings[:3]

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
