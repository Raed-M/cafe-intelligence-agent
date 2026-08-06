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
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define periods
analysis_start = pd.Timestamp("2026-05-04T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-05-11T00:00:00+03:00")
previous_start = pd.Timestamp("2026-04-27T00:00:00+03:00")
previous_end = pd.Timestamp("2026-05-04T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-04-27T00:00:00+03:00"), pd.Timestamp("2026-05-04T00:00:00+03:00")),
    (pd.Timestamp("2026-04-20T00:00:00+03:00"), pd.Timestamp("2026-04-27T00:00:00+03:00")),
    (pd.Timestamp("2026-04-13T00:00:00+03:00"), pd.Timestamp("2026-04-20T00:00:00+03:00")),
    (pd.Timestamp("2026-04-06T00:00:00+03:00"), pd.Timestamp("2026-04-13T00:00:00+03:00")),
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Convert timestamp to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Calculate daily revenue for analysis period
analysis_daily_revenue = pos_df[
    (pos_df['calendar_date'] >= analysis_start.date()) & 
    (pos_df['calendar_date'] < analysis_end.date())
].groupby('calendar_date')['line_total_sar'].sum()

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    period_revenue = pos_df[
        (pos_df['calendar_date'] >= period_start.date()) & 
        (pos_df['calendar_date'] < period_end.date())
    ].groupby('calendar_date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(period_revenue.values)

if len(baseline_daily_revenues) > 1 and np.std(baseline_daily_revenues) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Find anomalies in analysis period
    for date, revenue in analysis_daily_revenue.items():
        z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # Flag if |z_score| > 2
        if abs(z_score) > 2:
            findings.append({
                'title': f'Unusual Daily Revenue on {date.strftime("%Y-%m-%d")}',
                'claim': f'Daily revenue of {revenue:.2f} SAR on {date.strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'observed_daily_revenue': {
                        'value': round(revenue, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': date.isoformat(),
                        'period_end': (date + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_daily_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': None,
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Baseline computed from {len(baseline_daily_revenues)} daily observations across 4 trailing weeks',
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    'Refunds included in net revenue calculation'
                ],
                'assumptions': [
                    'Daily revenue follows approximately normal distribution',
                    'Z-score threshold of 2.0 used to flag anomalies',
                    'No known product launches or promotions in baseline or analysis periods'
                ],
                'confidence': 0.75
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Calculate hourly traffic for analysis period
analysis_hourly_traffic = traffic_clean[
    (traffic_clean['date'] >= analysis_start.date()) & 
    (traffic_clean['date'] < analysis_end.date())
].copy()

# Calculate hourly traffic for baseline periods
baseline_hourly_traffic = []
for period_start, period_end in baseline_periods:
    period_traffic = traffic_clean[
        (traffic_clean['date'] >= period_start.date()) & 
        (traffic_clean['date'] < period_end.date())
    ]
    baseline_hourly_traffic.extend(period_traffic['door_count'].values)

if len(baseline_hourly_traffic) > 10 and np.std(baseline_hourly_traffic) > 0:
    baseline_mean_traffic = np.mean(baseline_hourly_traffic)
    baseline_std_traffic = np.std(baseline_hourly_traffic)
    
    # Find anomalies in analysis period
    for idx, row in analysis_hourly_traffic.iterrows():
        door_count = row['door_count']
        z_score = (door_count - baseline_mean_traffic) / baseline_std_traffic if baseline_std_traffic > 0 else 0
        
        # Flag if |z_score| > 2.5 (stricter for traffic)
        if abs(z_score) > 2.5:
            findings.append({
                'title': f'Unusual Hourly Traffic on {row["date"].strftime("%Y-%m-%d")} Hour {int(row["hour"].hour)}',
                'claim': f'Door count of {int(door_count)} at {row["date"].strftime("%Y-%m-%d")} {int(row["hour"].hour):02d}:00 deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_door_count': {
                        'value': int(door_count),
                        'unit': 'count',
                        'numerator': None,
                        'denominator': None,
                        'period_start': analysis_start.isoformat(),
                        'period_end': analysis_end.isoformat()
                    },
                    'baseline_mean_door_count': {
                        'value': round(baseline_mean_traffic, 1),
                        'unit': 'count',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': None,
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_hourly_traffic),
                'coverage_notes': [
                    f'Baseline computed from {len(baseline_hourly_traffic)} hourly observations across 4 trailing weeks',
                    'Dead sensor days excluded from analysis',
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'
                ],
                'assumptions': [
                    'Hourly door counts follow approximately normal distribution',
                    'Z-score threshold of 2.5 used to flag anomalies',
                    'Sensor reliability consistent across baseline and analysis periods'
                ],
                'confidence': 0.70
            })

# ============================================================================
# ANOMALY 3: Weekly Waste Analysis
# ============================================================================

inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get analysis week waste
analysis_week_start = pd.Timestamp("2026-05-04T00:00:00+03:00").date()
analysis_waste = inventory_df[
    inventory_df['week_starting'].dt.date == analysis_week_start
]['units_wasted'].sum()

# Get baseline weeks waste
baseline_waste_values = []
for period_start, period_end in baseline_periods:
    week_start = period_start.date()
    week_waste = inventory_df[
        inventory_df['week_starting'].dt.date == week_start
    ]['units_wasted'].sum()
    if week_waste > 0:  # Only include weeks with known waste
        baseline_waste_values.append(week_waste)

if len(baseline_waste_values) > 1 and np.std(baseline_waste_values) > 0:
    baseline_mean_waste = np.mean(baseline_waste_values)
    baseline_std_waste = np.std(baseline_waste_values)
    
    z_score_waste = (analysis_waste - baseline_mean_waste) / baseline_std_waste if baseline_std_waste > 0 else 0
    
    if abs(z_score_waste) > 1.5:  # Lower threshold for waste due to smaller sample
        findings.append({
            'title': f'Unusual Weekly Waste for Week Starting {analysis_week_start}',
            'claim': f'Weekly waste of {int(analysis_waste)} units for week starting {analysis_week_start} deviates {abs(z_score_waste):.2f} standard deviations from baseline mean of {baseline_mean_waste:.1f} units.',
            'finding_type': 'waste_anomaly',
            'metrics': {
                'observed_units_wasted': {
                    'value': int(analysis_waste),
                    'unit': 'units',
                    'numerator': None,
                    'denominator': None,
                    'period_start': analysis_start.isoformat(),
                    'period_end': analysis_end.isoformat()
                },
                'baseline_mean_waste': {
                    'value': round(baseline_mean_waste, 1),
                    'unit': 'units',
                    'numerator': None,
                    'denominator': None,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                },
                'z_score': {
                    'value': round(z_score_waste, 2),
                    'unit': None,
                    'numerator': None,
                    'denominator': None,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                }
            },
            'source_names': ['inventory'],
            'sample_size': len(baseline_waste_values),
            'coverage_notes': [
                f'Baseline computed from {len(baseline_waste_values)} weeks with known waste values',
                'Only weeks with non-zero waste included in baseline',
                f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'
            ],
            'assumptions': [
                'Weekly waste follows approximately normal distribution',
                'Z-score threshold of 1.5 used (lower due to small sample)',
                'Waste measurement methodology consistent across periods'
            ],
            'confidence': 0.65
        })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)

# Keep top 3
findings = findings[:3]

# Prepare output
output = {
    'status': 'success' if findings else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
