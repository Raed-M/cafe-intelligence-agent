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
analysis_start = pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None)
baseline_periods = [
    (pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-29T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None))
]

findings = []

# 1. Daily Revenue Anomaly Detection
print("Analyzing daily revenue...")
pos_analysis = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
pos_baseline = pos_df[(pos_df['calendar_date'] >= baseline_periods[0][0]) & (pos_df['calendar_date'] < baseline_periods[0][1])].copy()

# Calculate daily revenue (excluding refunds)
daily_revenue_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('calendar_date')['line_total_sar'].sum()
daily_revenue_baseline = pos_baseline[~pos_baseline['is_refund']].groupby('calendar_date')['line_total_sar'].sum()

# Combine all baseline periods
all_baseline_revenue = []
for period_start, period_end in baseline_periods:
    period_data = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)]
    daily_rev = period_data[~period_data['is_refund']].groupby('calendar_date')['line_total_sar'].sum()
    all_baseline_revenue.extend(daily_rev.values)

if len(all_baseline_revenue) > 0 and np.std(all_baseline_revenue) > 0:
    baseline_mean = np.mean(all_baseline_revenue)
    baseline_std = np.std(all_baseline_revenue)
    
    for date, revenue in daily_revenue_analysis.items():
        z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2:  # 2 standard deviations
            findings.append({
                'title': f'Daily Revenue Anomaly on {date.date()}',
                'claim': f'Revenue of {revenue:.2f} SAR on {date.date()} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
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
                'sample_size': len(all_baseline_revenue),
                'coverage_notes': [f'Analysis period: {analysis_start.date()} to {analysis_end.date()}', 
                                  f'Baseline: {len(all_baseline_revenue)} daily observations from 4 weeks'],
                'assumptions': ['Normal distribution of daily revenue', 'Z-score threshold of 2.0 standard deviations'],
                'confidence': 0.85
            })

# 2. Hourly Traffic Anomaly Detection
print("Analyzing hourly traffic...")
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
traffic_baseline = traffic_df[(traffic_df['date'] >= baseline_periods[0][0]) & (traffic_df['date'] < baseline_periods[0][1])].copy()

# Filter out dead sensor days
traffic_baseline = traffic_baseline[~traffic_baseline['is_dead_sensor_day']]
traffic_analysis = traffic_analysis[~traffic_analysis['is_dead_sensor_day']]

if len(traffic_baseline) > 0 and traffic_baseline['door_count'].std() > 0:
    baseline_traffic_mean = traffic_baseline['door_count'].mean()
    baseline_traffic_std = traffic_baseline['door_count'].std()
    
    for idx, row in traffic_analysis.iterrows():
        z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std if baseline_traffic_std > 0 else 0
        if abs(z_score) > 2:
            findings.append({
                'title': f'Hourly Traffic Anomaly on {row["date"].date()} at {row["hour"].hour}:00',
                'claim': f'Door count of {int(row["door_count"])} at {row["hour"]} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.0f}',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'hourly_door_count': {
                        'value': int(row['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['hour'].isoformat(),
                        'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_traffic_mean, 0),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[0][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['hour'].isoformat(),
                        'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(traffic_baseline),
                'coverage_notes': [f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                                  f'Baseline: {len(traffic_baseline)} hourly observations from previous week',
                                  'Dead sensor days excluded'],
                'assumptions': ['Normal distribution of hourly traffic', 'Z-score threshold of 2.0 standard deviations'],
                'confidence': 0.80
            })

# 3. Daily Transaction Count Anomaly Detection
print("Analyzing daily transaction counts...")
daily_transactions_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('calendar_date')['transaction_id'].nunique()
daily_transactions_baseline = pos_baseline[~pos_baseline['is_refund']].groupby('calendar_date')['transaction_id'].nunique()

all_baseline_transactions = []
for period_start, period_end in baseline_periods:
    period_data = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)]
    daily_trans = period_data[~period_data['is_refund']].groupby('calendar_date')['transaction_id'].nunique()
    all_baseline_transactions.extend(daily_trans.values)

if len(all_baseline_transactions) > 0 and np.std(all_baseline_transactions) > 0:
    baseline_trans_mean = np.mean(all_baseline_transactions)
    baseline_trans_std = np.std(all_baseline_transactions)
    
    for date, trans_count in daily_transactions_analysis.items():
        z_score = (trans_count - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        if abs(z_score) > 2:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {date.date()}',
                'claim': f'Transaction count of {int(trans_count)} on {date.date()} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.0f}',
                'finding_type': 'transaction_volume_anomaly',
                'metrics': {
                    'daily_transaction_count': {
                        'value': int(trans_count),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': date.isoformat(),
                        'period_end': (date + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_trans_mean, 0),
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
                'sample_size': len(all_baseline_transactions),
                'coverage_notes': [f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                                  f'Baseline: {len(all_baseline_transactions)} daily observations from 4 weeks'],
                'assumptions': ['Normal distribution of daily transaction counts', 'Z-score threshold of 2.0 standard deviations'],
                'confidence': 0.85
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
    json.dump(output, f, indent=2)

print(f"Analysis complete. Found {len(findings_sorted)} anomalies.")
