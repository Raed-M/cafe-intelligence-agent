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

# Parse dates
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-07-13T00:00:00+03:00').tz_localize(None)
baseline_periods = [
    (pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None))
]

findings = []

# 1. Daily Revenue Anomaly Detection
print("Analyzing daily revenue...")
pos_df['date'] = pos_df['timestamp'].dt.date
daily_revenue = pos_df[~pos_df['is_refund']].groupby('date')['line_total_sar'].sum()
daily_revenue.index = pd.to_datetime(daily_revenue.index)

# Get baseline daily revenues
baseline_revenues = []
for period_start, period_end in baseline_periods:
    period_data = daily_revenue[(daily_revenue.index >= period_start) & (daily_revenue.index < period_end)]
    baseline_revenues.extend(period_data.values)

baseline_revenues = np.array(baseline_revenues)
if len(baseline_revenues) > 0 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Check analysis period
    analysis_revenues = daily_revenue[(daily_revenue.index >= analysis_start) & (daily_revenue.index < analysis_end)]
    
    if len(analysis_revenues) > 0:
        for date, revenue in analysis_revenues.items():
            z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Daily Revenue Anomaly on {date.date()}',
                    'claim': f'Daily revenue of {revenue:.2f} SAR on {date.date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': round(revenue, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_revenues),
                    'coverage_notes': [f'Baseline from {len(baseline_periods)} weeks', f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'],
                    'assumptions': ['Normal distribution of daily revenues', 'Z-score threshold of 2.0 standard deviations'],
                    'confidence': 0.85
                })

# 2. Hourly Traffic Anomaly Detection
print("Analyzing hourly traffic...")
traffic_df['date'] = traffic_df['hour'].dt.date
traffic_df['hour_of_day'] = traffic_df['hour'].dt.hour

# Get baseline traffic by hour
baseline_traffic_by_hour = {}
for period_start, period_end in baseline_periods:
    period_data = traffic_df[(traffic_df['hour'] >= period_start) & (traffic_df['hour'] < period_end) & (~traffic_df['is_dead_sensor_day'])]
    for hour in period_data['hour_of_day'].unique():
        hour_data = period_data[period_data['hour_of_day'] == hour]['door_count'].values
        if hour not in baseline_traffic_by_hour:
            baseline_traffic_by_hour[hour] = []
        baseline_traffic_by_hour[hour].extend(hour_data)

# Analyze analysis period traffic
analysis_traffic = traffic_df[(traffic_df['hour'] >= analysis_start) & (traffic_df['hour'] < analysis_end) & (~traffic_df['is_dead_sensor_day'])]

anomalies = []
for hour in analysis_traffic['hour_of_day'].unique():
    if hour in baseline_traffic_by_hour and len(baseline_traffic_by_hour[hour]) > 2:
        baseline_values = np.array(baseline_traffic_by_hour[hour])
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        
        hour_data = analysis_traffic[analysis_traffic['hour_of_day'] == hour]
        for idx, row in hour_data.iterrows():
            if baseline_std > 0:
                z_score = (row['door_count'] - baseline_mean) / baseline_std
                if abs(z_score) > 2.0:
                    anomalies.append({
                        'date': row['date'],
                        'hour': hour,
                        'traffic': row['door_count'],
                        'baseline_mean': baseline_mean,
                        'z_score': z_score,
                        'sample_size': len(baseline_values)
                    })

# Sort by magnitude and add top anomalies
anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
for anomaly in anomalies[:1]:  # Limit to 1 traffic anomaly
    findings.append({
        'title': f'Hourly Traffic Anomaly on {anomaly["date"]} at {anomaly["hour"]:02d}:00',
        'claim': f'Door count of {anomaly["traffic"]} at {anomaly["hour"]:02d}:00 on {anomaly["date"]} is {abs(anomaly["z_score"]):.2f} standard deviations from baseline mean of {anomaly["baseline_mean"]:.1f}',
        'finding_type': 'traffic_anomaly',
        'metrics': {
            'hourly_door_count': {
                'value': int(anomaly['traffic']),
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': f'{anomaly["date"]}T{anomaly["hour"]:02d}:00:00',
                'period_end': f'{anomaly["date"]}T{anomaly["hour"]+1:02d}:00:00'
            },
            'baseline_mean': {
                'value': round(anomaly['baseline_mean'], 1),
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': baseline_periods[0][0].isoformat(),
                'period_end': baseline_periods[-1][1].isoformat()
            },
            'z_score': {
                'value': round(anomaly['z_score'], 2),
                'unit': 'std_dev',
                'numerator': None,
                'denominator': None,
                'period_start': f'{anomaly["date"]}T{anomaly["hour"]:02d}:00:00',
                'period_end': f'{anomaly["date"]}T{anomaly["hour"]+1:02d}:00:00'
            }
        },
        'source_names': ['traffic'],
        'sample_size': anomaly['sample_size'],
        'coverage_notes': ['Excluded dead sensor days', f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'],
        'assumptions': ['Normal distribution of hourly traffic', 'Z-score threshold of 2.0 standard deviations'],
        'confidence': 0.80
    })

# 3. Daily Transaction Count Anomaly
print("Analyzing daily transaction counts...")
daily_transactions = pos_df[~pos_df['is_refund']].groupby('calendar_date')['transaction_id'].nunique()

# Get baseline transaction counts
baseline_transactions = []
for period_start, period_end in baseline_periods:
    period_data = daily_transactions[(daily_transactions.index >= period_start) & (daily_transactions.index < period_end)]
    baseline_transactions.extend(period_data.values)

baseline_transactions = np.array(baseline_transactions)
if len(baseline_transactions) > 2 and np.std(baseline_transactions) > 0:
    baseline_mean = np.mean(baseline_transactions)
    baseline_std = np.std(baseline_transactions)
    
    # Check analysis period
    analysis_transactions = daily_transactions[(daily_transactions.index >= analysis_start) & (daily_transactions.index < analysis_end)]
    
    if len(analysis_transactions) > 0:
        for date, count in analysis_transactions.items():
            z_score = (count - baseline_mean) / baseline_std if baseline_std > 0 else 0
            if abs(z_score) > 2.0:
                findings.append({
                    'title': f'Daily Transaction Count Anomaly on {date.date()}',
                    'claim': f'Transaction count of {count} on {date.date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.1f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'daily_transaction_count': {
                            'value': int(count),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_mean, 1),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_transactions),
                    'coverage_notes': [f'Baseline from {len(baseline_periods)} weeks', f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'],
                    'assumptions': ['Normal distribution of daily transaction counts', 'Z-score threshold of 2.0 standard deviations'],
                    'confidence': 0.85
                })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: x['confidence'], reverse=True)
findings = findings[:3]  # Limit to 3 findings

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Found {len(findings)} anomalies.")
