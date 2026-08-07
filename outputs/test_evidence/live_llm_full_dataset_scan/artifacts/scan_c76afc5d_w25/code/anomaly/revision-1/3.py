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
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define periods
analysis_start = pd.Timestamp("2026-06-29T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-07-06T00:00:00+03:00")
previous_start = pd.Timestamp("2026-06-22T00:00:00+03:00")
previous_end = pd.Timestamp("2026-06-29T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-06-22T00:00:00+03:00"), pd.Timestamp("2026-06-29T00:00:00+03:00")),
    (pd.Timestamp("2026-06-15T00:00:00+03:00"), pd.Timestamp("2026-06-22T00:00:00+03:00")),
    (pd.Timestamp("2026-06-08T00:00:00+03:00"), pd.Timestamp("2026-06-15T00:00:00+03:00")),
    (pd.Timestamp("2026-06-01T00:00:00+03:00"), pd.Timestamp("2026-06-08T00:00:00+03:00")),
]

findings = []

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

pos_analysis['date'] = pos_analysis['timestamp'].dt.date
daily_revenue_analysis = pos_analysis.groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp'] >= period_start) & 
        (pos_df['timestamp'] < period_end)
    ].copy()
    pos_baseline['date'] = pos_baseline['timestamp'].dt.date
    daily_rev = pos_baseline.groupby('date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue_analysis) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalies in analysis period
        for idx, row in daily_revenue_analysis.iterrows():
            z_score = (row['revenue'] - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': 'Unusual Daily Revenue',
                    'claim': f"Daily revenue on {row['date']} was {row['revenue']:.2f} SAR, which is {abs(z_score):.2f} standard deviations from the baseline mean of {baseline_mean:.2f} SAR.",
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'observed_daily_revenue': {
                            'value': round(row['revenue'], 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{row['date']}T00:00:00+03:00",
                            'period_end': f"{row['date']}T23:59:59+03:00"
                        },
                        'baseline_mean_daily_revenue': {
                            'value': round(baseline_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-01T00:00:00+03:00',
                            'period_end': '2026-06-29T00:00:00+03:00'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{row['date']}T00:00:00+03:00",
                            'period_end': f"{row['date']}T23:59:59+03:00"
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_revenues),
                    'coverage_notes': [
                        f"Analysis period: 2026-06-29 to 2026-07-06",
                        f"Baseline: 4 weeks from 2026-06-01 to 2026-06-29",
                        f"Baseline sample size: {len(baseline_daily_revenues)} daily observations"
                    ],
                    'assumptions': [
                        'Daily revenue follows normal distribution',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'No known product launches or promotions in analysis period'
                    ],
                    'confidence': 0.85
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic data for analysis period
traffic_analysis = traffic_df[
    (traffic_df['date'] >= pd.Timestamp(analysis_start.date())) & 
    (traffic_df['date'] < pd.Timestamp(analysis_end.date())) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

# Filter baseline traffic
baseline_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[
        (traffic_df['date'] >= pd.Timestamp(period_start.date())) & 
        (traffic_df['date'] < pd.Timestamp(period_end.date())) &
        (traffic_df['is_dead_sensor_day'] == False)
    ]
    baseline_traffic.extend(traffic_baseline['door_count'].values)

if len(baseline_traffic) > 0 and len(traffic_analysis) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic)
    baseline_traffic_std = np.std(baseline_traffic)
    
    if baseline_traffic_std > 0:
        # Find anomalies
        for idx, row in traffic_analysis.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std
            if abs(z_score) > 2.5:  # Higher threshold for hourly data
                hour_str = f"{int(row['hour']):02d}"
                date_str = row['date'].strftime('%Y-%m-%d')
                findings.append({
                    'title': 'Unusual Hourly Traffic',
                    'claim': f"Door count on {date_str} at hour {hour_str} was {row['door_count']} visitors, which is {abs(z_score):.2f} standard deviations from the baseline mean of {baseline_traffic_mean:.1f}.",
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T{hour_str}:00:00+03:00",
                            'period_end': f"{date_str}T{hour_str}:59:59+03:00"
                        },
                        'baseline_mean_hourly_door_count': {
                            'value': round(baseline_traffic_mean, 1),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-01T00:00:00+03:00',
                            'period_end': '2026-06-29T00:00:00+03:00'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T{hour_str}:00:00+03:00",
                            'period_end': f"{date_str}T{hour_str}:59:59+03:00"
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic),
                    'coverage_notes': [
                        f"Analysis period: 2026-06-29 to 2026-07-06",
                        f"Baseline: 4 weeks from 2026-06-01 to 2026-06-29",
                        f"Excluded dead sensor days",
                        f"Baseline sample size: {len(baseline_traffic)} hourly observations"
                    ],
                    'assumptions': [
                        'Hourly door counts follow normal distribution',
                        'Z-score threshold of 2.5 (98% confidence for hourly data)',
                        'Dead sensor days properly excluded'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Weekly Waste Cost Analysis
# ============================================================================

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Filter baseline inventory
baseline_waste_costs = []
for period_start, period_end in baseline_periods:
    inv_baseline = inventory_df[
        (inventory_df['week_starting'] >= period_start) & 
        (inventory_df['week_starting'] < period_end)
    ]
    # Sum waste costs per week
    weekly_waste = inv_baseline.groupby('week_starting')['known_waste_cost_sar'].sum()
    baseline_waste_costs.extend(weekly_waste.values)

if len(baseline_waste_costs) > 0 and len(inventory_analysis) > 0:
    baseline_waste_mean = np.mean(baseline_waste_costs)
    baseline_waste_std = np.std(baseline_waste_costs)
    
    if baseline_waste_std > 0:
        # Calculate weekly waste for analysis period
        analysis_weekly_waste = inventory_analysis.groupby('week_starting')['known_waste_cost_sar'].sum()
        
        for week, waste_cost in analysis_weekly_waste.items():
            z_score = (waste_cost - baseline_waste_mean) / baseline_waste_std
            if abs(z_score) > 1.5:  # Lower threshold due to smaller sample
                week_end = week + timedelta(days=6)
                findings.append({
                    'title': 'Unusual Weekly Waste Cost',
                    'claim': f"Weekly waste cost for week starting {week.date()} was {waste_cost:.2f} SAR, which is {abs(z_score):.2f} standard deviations from the baseline mean of {baseline_waste_mean:.2f} SAR.",
                    'finding_type': 'waste_anomaly',
                    'metrics': {
                        'observed_weekly_waste_cost': {
                            'value': round(waste_cost, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{week.date()}T00:00:00+03:00",
                            'period_end': f"{week_end.date()}T23:59:59+03:00"
                        },
                        'baseline_mean_weekly_waste_cost': {
                            'value': round(baseline_waste_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-01T00:00:00+03:00',
                            'period_end': '2026-06-29T00:00:00+03:00'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{week.date()}T00:00:00+03:00",
                            'period_end': f"{week_end.date()}T23:59:59+03:00"
                        }
                    },
                    'source_names': ['inventory'],
                    'sample_size': len(baseline_waste_costs),
                    'coverage_notes': [
                        f"Analysis period: 2026-06-29 to 2026-07-06",
                        f"Baseline: 4 weeks from 2026-06-01 to 2026-06-29",
                        f"Only known waste costs included",
                        f"Baseline sample size: {len(baseline_waste_costs)} weekly observations"
                    ],
                    'assumptions': [
                        'Weekly waste costs follow normal distribution',
                        'Z-score threshold of 1.5 (87% confidence)',
                        'Unknown waste values excluded from analysis'
                    ],
                    'confidence': 0.75
                })

# Sort findings by absolute z-score (magnitude)
findings.sort(key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)

# Keep only top 3
findings = findings[:3]

# Prepare output
result = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
