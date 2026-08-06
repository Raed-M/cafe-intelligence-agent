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
analysis_start = pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-03-09T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None))
]

findings = []

# 1. Daily Revenue Anomaly Detection
print("Analyzing daily revenue...")
pos_df['date'] = pos_df['timestamp'].dt.date
daily_revenue = pos_df.groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

# Get baseline daily revenues
baseline_revenues = []
for period_start, period_end in baseline_periods:
    period_data = daily_revenue[(daily_revenue['date'] >= period_start) & 
                                (daily_revenue['date'] < period_end)]
    baseline_revenues.extend(period_data['line_total_sar'].values)

# Get analysis period revenues
analysis_revenues = daily_revenue[(daily_revenue['date'] >= analysis_start) & 
                                  (daily_revenue['date'] < analysis_end)]

if len(baseline_revenues) > 5 and len(analysis_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    if baseline_std > 0:
        for idx, row in analysis_revenues.iterrows():
            z_score = (row['line_total_sar'] - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Daily Revenue Anomaly on {row["date"].date()}',
                    'claim': f'Daily revenue of {row["line_total_sar"]:.2f} SAR on {row["date"].date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': row['line_total_sar'],
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': baseline_mean,
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': z_score,
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_revenues),
                    'coverage_notes': [f'Baseline: {len(baseline_revenues)} daily observations from 4 weeks prior',
                                      f'Analysis period: {len(analysis_revenues)} days'],
                    'assumptions': ['Normal distribution of daily revenues',
                                   'No structural breaks in business operations',
                                   'Baseline periods are representative'],
                    'confidence': min(0.95, 1.0 - (0.05 * (abs(z_score) - 2.0)))
                })

# 2. Hourly Traffic Anomaly Detection
print("Analyzing hourly traffic...")
traffic_df['date'] = traffic_df['hour'].dt.date
traffic_df['hour_of_day'] = traffic_df['hour'].dt.hour

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Get baseline hourly traffic
baseline_hourly = []
for period_start, period_end in baseline_periods:
    period_data = traffic_clean[(traffic_clean['hour'] >= period_start) & 
                                (traffic_clean['hour'] < period_end)]
    baseline_hourly.extend(period_data['door_count'].values)

# Get analysis period hourly traffic
analysis_hourly = traffic_clean[(traffic_clean['hour'] >= analysis_start) & 
                                (traffic_clean['hour'] < analysis_end)]

if len(baseline_hourly) > 10 and len(analysis_hourly) > 0:
    baseline_mean_traffic = np.mean(baseline_hourly)
    baseline_std_traffic = np.std(baseline_hourly)
    
    if baseline_std_traffic > 0:
        for idx, row in analysis_hourly.iterrows():
            z_score = (row['door_count'] - baseline_mean_traffic) / baseline_std_traffic
            if abs(z_score) > 2.5:  # 2.5 standard deviations for traffic
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["hour"]}',
                    'claim': f'Door count of {row["door_count"]} at {row["hour"]} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.2f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'door_count': {
                            'value': row['door_count'],
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour'].isoformat(),
                            'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': baseline_mean_traffic,
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': z_score,
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour'].isoformat(),
                            'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_hourly),
                    'coverage_notes': [f'Baseline: {len(baseline_hourly)} hourly observations from 4 weeks prior',
                                      f'Dead sensor days excluded',
                                      f'Analysis period: {len(analysis_hourly)} hours'],
                    'assumptions': ['Normal distribution of hourly traffic',
                                   'Sensor reliability consistent across periods',
                                   'No structural changes in cafe operations'],
                    'confidence': min(0.95, 1.0 - (0.05 * (abs(z_score) - 2.5)))
                })

# 3. Daily Transaction Count Anomaly
print("Analyzing daily transaction counts...")
daily_transactions = pos_df.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_transactions.columns = ['date', 'transaction_count']
daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])

# Get baseline transaction counts
baseline_tx_counts = []
for period_start, period_end in baseline_periods:
    period_data = daily_transactions[(daily_transactions['date'] >= period_start) & 
                                     (daily_transactions['date'] < period_end)]
    baseline_tx_counts.extend(period_data['transaction_count'].values)

# Get analysis period transaction counts
analysis_tx_counts = daily_transactions[(daily_transactions['date'] >= analysis_start) & 
                                        (daily_transactions['date'] < analysis_end)]

if len(baseline_tx_counts) > 5 and len(analysis_tx_counts) > 0:
    baseline_mean_tx = np.mean(baseline_tx_counts)
    baseline_std_tx = np.std(baseline_tx_counts)
    
    if baseline_std_tx > 0:
        for idx, row in analysis_tx_counts.iterrows():
            z_score = (row['transaction_count'] - baseline_mean_tx) / baseline_std_tx
            if abs(z_score) > 2.0:
                findings.append({
                    'title': f'Daily Transaction Count Anomaly on {row["date"].date()}',
                    'claim': f'Transaction count of {row["transaction_count"]} on {row["date"].date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_tx:.2f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'transaction_count': {
                            'value': row['transaction_count'],
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': baseline_mean_tx,
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': z_score,
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_tx_counts),
                    'coverage_notes': [f'Baseline: {len(baseline_tx_counts)} daily observations from 4 weeks prior',
                                      f'Analysis period: {len(analysis_tx_counts)} days'],
                    'assumptions': ['Normal distribution of daily transaction counts',
                                   'No structural breaks in business operations',
                                   'Baseline periods are representative'],
                    'confidence': min(0.95, 1.0 - (0.05 * (abs(z_score) - 2.0)))
                })

# Sort findings by magnitude of z-score and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Found {len(findings_sorted)} anomalies.")
