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

# Parse analysis periods
analysis_start = datetime.fromisoformat("2026-01-19T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-26T00:00:00+03:00")

baseline_periods = [
    (datetime.fromisoformat("2026-01-12T00:00:00+03:00"), datetime.fromisoformat("2026-01-19T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-05T00:00:00+03:00"), datetime.fromisoformat("2026-01-12T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-29T00:00:00+03:00"), datetime.fromisoformat("2026-01-05T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-22T00:00:00+03:00"), datetime.fromisoformat("2025-12-29T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamps to datetime with timezone
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('UTC+03:00')
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True).dt.tz_convert('UTC+03:00')
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True).dt.tz_convert('UTC+03:00')
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True).dt.tz_convert('UTC+03:00')

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baselines
pos_df['date'] = pos_df['timestamp'].dt.date

# Analysis period daily revenue
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
analysis_daily_revenue = analysis_pos.groupby('date')['line_total_sar'].sum()

# Baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    baseline_daily = baseline_pos.groupby('date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(baseline_daily.values)

if len(baseline_daily_revenues) > 3 and np.var(baseline_daily_revenues) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Find anomalous days in analysis period
    for date, revenue in analysis_daily_revenue.items():
        z_score = (revenue - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        if abs(z_score) > 2.0:  # 2-sigma threshold
            findings.append({
                'title': 'Unusual Daily Revenue',
                'claim': f'Daily revenue on {date} was {abs(z_score):.2f} standard deviations from baseline mean',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'daily_revenue': {
                        'value': float(revenue),
                        'unit': 'SAR',
                        'numerator': float(revenue),
                        'denominator': None,
                        'period_start': f'{date}T00:00:00+03:00',
                        'period_end': f'{date}T23:59:59+03:00'
                    },
                    'baseline_mean_revenue': {
                        'value': float(baseline_mean),
                        'unit': 'SAR',
                        'numerator': float(baseline_mean),
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    },
                    'z_score_revenue': {
                        'value': float(z_score),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks prior (2025-12-22 to 2026-01-19)',
                    f'Baseline sample size: {len(baseline_daily_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold: 2.0 (95% confidence)',
                    'Baseline calculated from 4 trailing weeks',
                    'Refunds included in net revenue per metric definition'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Analysis period hourly traffic
analysis_traffic = traffic_clean[(traffic_clean['date'] >= analysis_start.date()) & 
                                 (traffic_clean['date'] < analysis_end.date())]

# Baseline hourly traffic
baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_clean[(traffic_clean['date'] >= period_start.date()) & 
                                     (traffic_clean['date'] < period_end.date())]
    baseline_traffic_list.append(baseline_traffic)

baseline_traffic_combined = pd.concat(baseline_traffic_list, ignore_index=True)

if len(baseline_traffic_combined) > 10 and baseline_traffic_combined['door_count'].var() > 0:
    baseline_mean_traffic = baseline_traffic_combined['door_count'].mean()
    baseline_std_traffic = baseline_traffic_combined['door_count'].std()
    
    # Find anomalous hours
    for idx, row in analysis_traffic.iterrows():
        door_count = row['door_count']
        z_score = (door_count - baseline_mean_traffic) / baseline_std_traffic if baseline_std_traffic > 0 else 0
        
        if abs(z_score) > 2.5:  # Slightly higher threshold for hourly data
            findings.append({
                'title': 'Unusual Hourly Door Traffic',
                'claim': f'Door traffic on {row["date"]} hour {int(row["hour"])} was {abs(z_score):.2f} standard deviations from baseline',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'hourly_door_count': {
                        'value': float(door_count),
                        'unit': 'persons',
                        'numerator': float(door_count),
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    },
                    'baseline_mean_traffic': {
                        'value': float(baseline_mean_traffic),
                        'unit': 'persons',
                        'numerator': float(baseline_mean_traffic),
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    },
                    'z_score_traffic': {
                        'value': float(z_score),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic_combined),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks prior (2025-12-22 to 2026-01-19)',
                    f'Dead sensor days excluded',
                    f'Baseline sample size: {len(baseline_traffic_combined)} hourly observations'
                ],
                'assumptions': [
                    'Z-score threshold: 2.5 (hourly data)',
                    'Baseline calculated from 4 trailing weeks',
                    'Dead sensor intervals excluded per metadata'
                ],
                'confidence': 0.80
            })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count
analysis_pos_txn = analysis_pos.groupby('date')['transaction_id'].nunique()
baseline_txn_counts = []

for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    baseline_daily_txn = baseline_pos.groupby('date')['transaction_id'].nunique()
    baseline_txn_counts.extend(baseline_daily_txn.values)

if len(baseline_txn_counts) > 3 and np.var(baseline_txn_counts) > 0:
    baseline_mean_txn = np.mean(baseline_txn_counts)
    baseline_std_txn = np.std(baseline_txn_counts)
    
    for date, txn_count in analysis_pos_txn.items():
        z_score = (txn_count - baseline_mean_txn) / baseline_std_txn if baseline_std_txn > 0 else 0
        
        if abs(z_score) > 2.0:
            findings.append({
                'title': 'Unusual Daily Transaction Count',
                'claim': f'Transaction count on {date} was {abs(z_score):.2f} standard deviations from baseline',
                'finding_type': 'transaction_anomaly',
                'metrics': {
                    'daily_transaction_count': {
                        'value': int(txn_count),
                        'unit': 'transactions',
                        'numerator': int(txn_count),
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    },
                    'baseline_mean_transactions': {
                        'value': float(baseline_mean_txn),
                        'unit': 'transactions',
                        'numerator': float(baseline_mean_txn),
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    },
                    'z_score_transactions': {
                        'value': float(z_score),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00+03:00',
                        'period_end': '2026-01-19T00:00:00+03:00'
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_txn_counts),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks prior (2025-12-22 to 2026-01-19)',
                    f'Baseline sample size: {len(baseline_txn_counts)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold: 2.0 (95% confidence)',
                    'Transaction count derived from unique transaction_id per day',
                    'Baseline calculated from 4 trailing weeks'
                ],
                'confidence': 0.85
            })

# Sort by z-score magnitude and keep top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score_revenue', x['metrics'].get('z_score_traffic', x['metrics'].get('z_score_transactions', {'value': 0})))['value']), reverse=True)[:3]

# Build output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
