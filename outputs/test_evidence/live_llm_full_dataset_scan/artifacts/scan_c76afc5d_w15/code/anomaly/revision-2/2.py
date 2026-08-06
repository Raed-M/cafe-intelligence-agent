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
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods - convert to naive timestamps for comparison
analysis_start = pd.Timestamp('2026-04-20T00:00:00', tz=None)
analysis_end = pd.Timestamp('2026-04-27T00:00:00', tz=None)

baseline_periods = [
    (pd.Timestamp('2026-04-13T00:00:00', tz=None), pd.Timestamp('2026-04-20T00:00:00', tz=None)),
    (pd.Timestamp('2026-04-06T00:00:00', tz=None), pd.Timestamp('2026-04-13T00:00:00', tz=None)),
    (pd.Timestamp('2026-03-30T00:00:00', tz=None), pd.Timestamp('2026-04-06T00:00:00', tz=None)),
    (pd.Timestamp('2026-03-23T00:00:00', tz=None), pd.Timestamp('2026-03-30T00:00:00', tz=None))
]

findings = []

# Ensure timestamps are timezone-naive for comparison
pos_df['timestamp_naive'] = pos_df['timestamp'].dt.tz_localize(None) if pos_df['timestamp'].dt.tz is not None else pos_df['timestamp']
pos_df['calendar_date_naive'] = pos_df['calendar_date'].dt.tz_localize(None) if pos_df['calendar_date'].dt.tz is not None else pos_df['calendar_date']

# 1. Daily Revenue Anomaly Detection
print("Analyzing daily revenue...")

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date_naive'] >= analysis_start) & 
                      (pos_df['calendar_date_naive'] < analysis_end)]
analysis_daily_revenue = analysis_pos.groupby('calendar_date_naive')['line_total_sar'].sum()

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date_naive'] >= period_start) & 
                          (pos_df['calendar_date_naive'] < period_end)]
    daily_rev = baseline_pos.groupby('calendar_date_naive')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 1:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Find anomalies in analysis period
    for date, revenue in analysis_daily_revenue.items():
        z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2:  # 2 standard deviations
            findings.append({
                'title': f'Daily Revenue Anomaly on {date.date()}',
                'claim': f'Daily revenue of {revenue:.2f} SAR on {date.date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
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
                    'baseline_std_daily_revenue': {
                        'value': round(baseline_std, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': date.isoformat(),
                        'period_end': (date + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: {len(baseline_daily_revenues)} daily observations from 4 trailing weeks',
                    f'Threshold: |z-score| > 2.0'
                ],
                'assumptions': [
                    'Daily revenue follows approximately normal distribution',
                    'Baseline periods are representative of normal operations',
                    'No structural breaks in business model during baseline'
                ],
                'confidence': 0.85
            })

# 2. Hourly Traffic Anomaly Detection
print("Analyzing hourly traffic...")

traffic_df['date_naive'] = traffic_df['date'].dt.tz_localize(None) if traffic_df['date'].dt.tz is not None else traffic_df['date']
traffic_df['hour_naive'] = traffic_df['hour'].dt.tz_localize(None) if traffic_df['hour'].dt.tz is not None else traffic_df['hour']

# Filter out dead sensor days
traffic_df_valid = traffic_df[traffic_df['is_dead_sensor_day'] == False]

# Get analysis period traffic
analysis_traffic = traffic_df_valid[(traffic_df_valid['date_naive'] >= analysis_start) & 
                                    (traffic_df_valid['date_naive'] < analysis_end)]

# Get baseline traffic
baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df_valid[(traffic_df_valid['date_naive'] >= period_start) & 
                                        (traffic_df_valid['date_naive'] < period_end)]
    baseline_traffic_list.append(baseline_traffic)

baseline_traffic_combined = pd.concat(baseline_traffic_list) if baseline_traffic_list else pd.DataFrame()

if len(baseline_traffic_combined) > 1:
    baseline_hourly_mean = baseline_traffic_combined['door_count'].mean()
    baseline_hourly_std = baseline_traffic_combined['door_count'].std()
    
    if baseline_hourly_std > 0:
        # Find anomalies
        for idx, row in analysis_traffic.iterrows():
            z_score = (row['door_count'] - baseline_hourly_mean) / baseline_hourly_std
            if abs(z_score) > 2.5:  # Higher threshold for hourly data
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date_naive"].date()} at {row["hour_naive"].hour}:00',
                    'claim': f'Door count of {int(row["door_count"])} at {row["hour_naive"].hour}:00 on {row["date_naive"].date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_hourly_mean:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_traffic': {
                            'value': int(row['door_count']),
                            'unit': 'door_count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour_naive'].isoformat(),
                            'period_end': (row['hour_naive'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean_hourly_traffic': {
                            'value': round(baseline_hourly_mean, 1),
                            'unit': 'door_count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'baseline_std_hourly_traffic': {
                            'value': round(baseline_hourly_std, 1),
                            'unit': 'door_count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour_naive'].isoformat(),
                            'period_end': (row['hour_naive'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic_combined),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: {len(baseline_traffic_combined)} hourly observations from 4 trailing weeks',
                        f'Dead sensor days excluded',
                        f'Threshold: |z-score| > 2.5'
                    ],
                    'assumptions': [
                        'Hourly traffic follows approximately normal distribution',
                        'Baseline periods are representative of normal traffic patterns',
                        'Sensor readings are accurate and consistent'
                    ],
                    'confidence': 0.80
                })

# 3. Daily Transaction Count Anomaly Detection
print("Analyzing daily transaction counts...")

# Calculate daily transaction counts
analysis_transactions = analysis_pos.groupby('calendar_date_naive')['transaction_id'].nunique()

baseline_transaction_counts = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date_naive'] >= period_start) & 
                          (pos_df['calendar_date_naive'] < period_end)]
    daily_trans = baseline_pos.groupby('calendar_date_naive')['transaction_id'].nunique()
    baseline_transaction_counts.extend(daily_trans.values)

if len(baseline_transaction_counts) > 1:
    baseline_trans_mean = np.mean(baseline_transaction_counts)
    baseline_trans_std = np.std(baseline_transaction_counts)
    
    for date, trans_count in analysis_transactions.items():
        z_score = (trans_count - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        if abs(z_score) > 2:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {date.date()}',
                'claim': f'Transaction count of {int(trans_count)} on {date.date()} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f}',
                'finding_type': 'transaction_volume_anomaly',
                'metrics': {
                    'observed_daily_transactions': {
                        'value': int(trans_count),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': date.isoformat(),
                        'period_end': (date + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_daily_transactions': {
                        'value': round(baseline_trans_mean, 1),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'baseline_std_daily_transactions': {
                        'value': round(baseline_trans_std, 1),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': date.isoformat(),
                        'period_end': (date + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_transaction_counts),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: {len(baseline_transaction_counts)} daily observations from 4 trailing weeks',
                    f'Threshold: |z-score| > 2.0'
                ],
                'assumptions': [
                    'Daily transaction counts follow approximately normal distribution',
                    'Baseline periods are representative of normal transaction volumes',
                    'No structural changes in customer behavior'
                ],
                'confidence': 0.85
            })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)[:3]

# Prepare output
result = {
    'status': 'success' if findings_sorted else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Analysis complete. Found {len(findings_sorted)} anomalies.")
