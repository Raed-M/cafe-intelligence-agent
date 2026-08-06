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

# Parse timestamps
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.Timestamp('2026-03-09T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')
analysis_end = pd.Timestamp('2026-03-16T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')

baseline_periods = [
    (pd.Timestamp('2026-03-02T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-03-09T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-02-23T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-03-02T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-02-16T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-02-23T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-02-09T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-02-16T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
]

findings = []

# Ensure timezone-aware timestamps for comparison
if pos_df['timestamp'].dt.tz is None:
    pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('Asia/Riyadh')
else:
    pos_df['timestamp'] = pos_df['timestamp'].dt.tz_convert('Asia/Riyadh')

if traffic_df['hour'].dt.tz is None:
    traffic_df['hour'] = pd.to_datetime(traffic_df['hour'], utc=True).dt.tz_convert('Asia/Riyadh')
else:
    traffic_df['hour'] = traffic_df['hour'].dt.tz_convert('Asia/Riyadh')

# ============================================================================
# ANOMALY 1: Daily Revenue
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis['date'] = pos_analysis['timestamp'].dt.date

daily_revenue_analysis = pos_analysis.groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline['date'] = pos_baseline['timestamp'].dt.date
    daily_rev = pos_baseline.groupby('date')['line_total_sar'].sum().reset_index()
    baseline_daily_revenues.extend(daily_rev['line_total_sar'].values)

if len(baseline_daily_revenues) > 1 and np.std(baseline_daily_revenues) > 0:
    baseline_mean_revenue = np.mean(baseline_daily_revenues)
    baseline_std_revenue = np.std(baseline_daily_revenues)
    
    # Find anomalies in analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean_revenue) / baseline_std_revenue if baseline_std_revenue > 0 else 0
        if abs(z_score) > 2.0:  # 2-sigma threshold
            findings.append({
                'title': f'Daily Revenue Anomaly on {row["date"]}',
                'claim': f'Daily revenue of {row["revenue"]:.2f} SAR on {row["date"]} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_revenue:.2f} SAR.',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'daily_revenue': {
                        'value': round(row['revenue'], 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(row['date']),
                        'period_end': str(row['date'])
                    },
                    'baseline_mean_revenue': {
                        'value': round(baseline_mean_revenue, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-02-09',
                        'period_end': '2026-03-09'
                    },
                    'z_score_revenue': {
                        'value': round(z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(row['date']),
                        'period_end': str(row['date'])
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks prior (2026-02-09 to 2026-03-09)',
                    f'Baseline sample size: {len(baseline_daily_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Baseline periods are representative of normal operations',
                    'No known product launches or major events during baseline'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic
# ============================================================================

# Convert traffic_df['date'] to datetime if it's not already
if traffic_df['date'].dtype == 'object':
    traffic_df['date'] = pd.to_datetime(traffic_df['date'])

# Extract date from analysis period timestamps for comparison
analysis_start_date = analysis_start.date()
analysis_end_date = analysis_end.date()

traffic_analysis = traffic_df[(traffic_df['date'].dt.date >= analysis_start_date) & (traffic_df['date'].dt.date < analysis_end_date)].copy()
traffic_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]

if len(traffic_analysis) > 0:
    # Build baseline hourly traffic
    baseline_hourly_traffic = []
    for period_start, period_end in baseline_periods:
        period_start_date = period_start.date()
        period_end_date = period_end.date()
        traffic_baseline = traffic_df[(traffic_df['date'].dt.date >= period_start_date) & (traffic_df['date'].dt.date < period_end_date)].copy()
        traffic_baseline = traffic_baseline[traffic_baseline['is_dead_sensor_day'] == False]
        baseline_hourly_traffic.extend(traffic_baseline['door_count'].values)
    
    if len(baseline_hourly_traffic) > 1 and np.std(baseline_hourly_traffic) > 0:
        baseline_mean_traffic = np.mean(baseline_hourly_traffic)
        baseline_std_traffic = np.std(baseline_hourly_traffic)
        
        # Check for anomalies
        for idx, row in traffic_analysis.iterrows():
            z_score = (row['door_count'] - baseline_mean_traffic) / baseline_std_traffic if baseline_std_traffic > 0 else 0
            if abs(z_score) > 2.5:  # Slightly higher threshold for traffic
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date"].date()}',
                    'claim': f'Hourly door count of {int(row["door_count"])} on {row["date"].date()} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(row['date'].date()),
                            'period_end': str(row['date'].date())
                        },
                        'baseline_mean_traffic': {
                            'value': round(baseline_mean_traffic, 1),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-02-09',
                            'period_end': '2026-03-09'
                        },
                        'z_score_traffic': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(row['date'].date()),
                            'period_end': str(row['date'].date())
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_hourly_traffic),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: 4 weeks prior (2026-02-09 to 2026-03-09)',
                        f'Excluded dead sensor days',
                        f'Baseline sample size: {len(baseline_hourly_traffic)} hourly observations'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.5 (99% confidence)',
                        'Sensor data is reliable (dead sensor days excluded)',
                        'Traffic patterns are consistent week-to-week'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Waste Cost
# ============================================================================

inventory_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start].copy()
inventory_baseline = inventory_df[inventory_df['week_starting'] < analysis_start].copy()

if len(inventory_analysis) > 0 and len(inventory_baseline) > 0:
    baseline_waste_costs = inventory_baseline['known_waste_cost_sar'].dropna().values
    
    if len(baseline_waste_costs) > 1 and np.std(baseline_waste_costs) > 0:
        baseline_mean_waste = np.mean(baseline_waste_costs)
        baseline_std_waste = np.std(baseline_waste_costs)
        
        for idx, row in inventory_analysis.iterrows():
            if pd.notna(row['known_waste_cost_sar']) and row['known_waste_cost_sar'] > 0:
                z_score = (row['known_waste_cost_sar'] - baseline_mean_waste) / baseline_std_waste if baseline_std_waste > 0 else 0
                if abs(z_score) > 2.0:
                    week_end = row['week_starting'] + timedelta(days=6)
                    findings.append({
                        'title': f'Weekly Waste Cost Anomaly (Week of {row["week_starting"].date()})',
                        'claim': f'Weekly waste cost of {row["known_waste_cost_sar"]:.2f} SAR for week starting {row["week_starting"].date()} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_waste:.2f} SAR.',
                        'finding_type': 'waste_anomaly',
                        'metrics': {
                            'weekly_waste_cost': {
                                'value': round(row['known_waste_cost_sar'], 2),
                                'unit': 'SAR',
                                'numerator': None,
                                'denominator': None,
                                'period_start': str(row['week_starting'].date()),
                                'period_end': str(week_end.date())
                            },
                            'baseline_mean_waste': {
                                'value': round(baseline_mean_waste, 2),
                                'unit': 'SAR',
                                'numerator': None,
                                'denominator': None,
                                'period_start': '2026-02-09',
                                'period_end': '2026-03-09'
                            },
                            'z_score_waste': {
                                'value': round(z_score, 2),
                                'unit': 'std_dev',
                                'numerator': None,
                                'denominator': None,
                                'period_start': str(row['week_starting'].date()),
                                'period_end': str(week_end.date())
                            }
                        },
                        'source_names': ['inventory'],
                        'sample_size': len(baseline_waste_costs),
                        'coverage_notes': [
                            f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                            f'Baseline: weeks prior to {analysis_start.date()}',
                            f'Baseline sample size: {len(baseline_waste_costs)} weekly observations',
                            'Only known waste costs included (excludes unknown waste values)'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.0 (95% confidence)',
                            'Waste patterns are consistent week-to-week',
                            'Known waste cost is accurate and complete'
                        ],
                        'confidence': 0.75
                    })

# Sort by z-score magnitude and limit to 3
findings_sorted = sorted(findings, key=lambda x: max([abs(v.get('value', 0)) for k, v in x['metrics'].items() if 'z_score' in k], default=0), reverse=True)[:3]

# Build output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
