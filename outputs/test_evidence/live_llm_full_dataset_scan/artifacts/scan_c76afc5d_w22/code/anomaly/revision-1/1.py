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

# Parse analysis periods
analysis_start = datetime.fromisoformat("2026-06-08T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-06-15T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-06-01T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-06-08T00:00:00+03:00")

baseline_periods = [
    (datetime.fromisoformat("2026-06-01T00:00:00+03:00"), datetime.fromisoformat("2026-06-08T00:00:00+03:00")),
    (datetime.fromisoformat("2026-05-25T00:00:00+03:00"), datetime.fromisoformat("2026-06-01T00:00:00+03:00")),
    (datetime.fromisoformat("2026-05-18T00:00:00+03:00"), datetime.fromisoformat("2026-05-25T00:00:00+03:00")),
    (datetime.fromisoformat("2026-05-11T00:00:00+03:00"), datetime.fromisoformat("2026-05-18T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamps to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baseline periods
pos_df['date'] = pos_df['timestamp'].dt.date

# Analysis period daily revenue
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
analysis_daily_revenue = analysis_pos.groupby('date')['line_total_sar'].sum()

# Baseline daily revenue (all baseline periods combined)
baseline_pos_list = []
for period_start, period_end in baseline_periods:
    baseline_pos_list.append(pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)])
baseline_pos = pd.concat(baseline_pos_list, ignore_index=True)
baseline_daily_revenue = baseline_pos.groupby('date')['line_total_sar'].sum()

if len(baseline_daily_revenue) > 0 and baseline_daily_revenue.std() > 0:
    baseline_mean = baseline_daily_revenue.mean()
    baseline_std = baseline_daily_revenue.std()
    
    # Find anomalies in analysis period
    for date, revenue in analysis_daily_revenue.items():
        z_score = (revenue - baseline_mean) / baseline_std
        if abs(z_score) > 2.0:  # 2-sigma threshold
            findings.append({
                'title': f'Daily Revenue Anomaly on {date}',
                'claim': f'Daily revenue on {date} was {revenue:.2f} SAR, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'observed_daily_revenue': {
                        'value': round(revenue, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(date),
                        'period_end': str(date)
                    },
                    'baseline_mean_daily_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-11',
                        'period_end': '2026-06-08'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': None,
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(date),
                        'period_end': str(date)
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenue),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}',
                    f'Baseline sample size: {len(baseline_daily_revenue)} days'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Baseline calculated from trailing 4 weeks',
                    'Refunds included in net revenue per metric definition'
                ],
                'confidence': 0.95
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Analysis period hourly traffic
analysis_traffic = traffic_clean[(traffic_clean['date'] >= analysis_start.date()) & (traffic_clean['date'] < analysis_end.date())]
analysis_hourly_traffic = analysis_traffic.groupby('hour')['door_count'].mean()

# Baseline hourly traffic
baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    baseline_traffic_list.append(traffic_clean[(traffic_clean['date'] >= period_start.date()) & (traffic_clean['date'] < period_end.date())])
baseline_traffic = pd.concat(baseline_traffic_list, ignore_index=True)
baseline_hourly_traffic = baseline_traffic.groupby('hour')['door_count'].mean()

if len(baseline_hourly_traffic) > 0 and baseline_hourly_traffic.std() > 0:
    baseline_mean_traffic = baseline_hourly_traffic.mean()
    baseline_std_traffic = baseline_hourly_traffic.std()
    
    # Find anomalies in analysis period
    for hour, traffic in analysis_hourly_traffic.items():
        if pd.notna(hour) and pd.notna(traffic):
            z_score = (traffic - baseline_mean_traffic) / baseline_std_traffic
            if abs(z_score) > 2.0:
                findings.append({
                    'title': f'Hourly Traffic Anomaly at Hour {int(hour)}',
                    'claim': f'Average hourly door count at hour {int(hour)} during analysis period was {traffic:.1f}, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_door_count': {
                            'value': round(traffic, 1),
                            'unit': 'door_count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-08',
                            'period_end': '2026-06-15'
                        },
                        'baseline_mean_hourly_door_count': {
                            'value': round(baseline_mean_traffic, 1),
                            'unit': 'door_count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-05-11',
                            'period_end': '2026-06-08'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': None,
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-08',
                            'period_end': '2026-06-15'
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_hourly_traffic),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}',
                        f'Dead sensor days excluded',
                        f'Baseline sample size: {len(baseline_hourly_traffic)} hours'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.0 (95% confidence)',
                        'Baseline calculated from trailing 4 weeks',
                        'Dead sensor intervals excluded per data quality notes'
                    ],
                    'confidence': 0.95
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count
analysis_pos_txn = analysis_pos.groupby('date')['transaction_id'].nunique()
baseline_pos_txn = baseline_pos.groupby('date')['transaction_id'].nunique()

if len(baseline_pos_txn) > 0 and baseline_pos_txn.std() > 0:
    baseline_mean_txn = baseline_pos_txn.mean()
    baseline_std_txn = baseline_pos_txn.std()
    
    # Find anomalies in analysis period
    for date, txn_count in analysis_pos_txn.items():
        z_score = (txn_count - baseline_mean_txn) / baseline_std_txn
        if abs(z_score) > 2.0:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {date}',
                'claim': f'Daily transaction count on {date} was {txn_count}, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_txn:.1f}.',
                'finding_type': 'transaction_anomaly',
                'metrics': {
                    'observed_daily_transaction_count': {
                        'value': int(txn_count),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(date),
                        'period_end': str(date)
                    },
                    'baseline_mean_daily_transaction_count': {
                        'value': round(baseline_mean_txn, 1),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-11',
                        'period_end': '2026-06-08'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': None,
                        'numerator': None,
                        'denominator': None,
                        'period_start': str(date),
                        'period_end': str(date)
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_pos_txn),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}',
                    f'Baseline sample size: {len(baseline_pos_txn)} days'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Baseline calculated from trailing 4 weeks',
                    'Transaction count derived from unique transaction_id per day'
                ],
                'confidence': 0.95
            })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)