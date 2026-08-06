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
analysis_start = pd.Timestamp('2026-03-09T00:00:00+03:00')
analysis_end = pd.Timestamp('2026-03-16T00:00:00+03:00')
baseline_periods = [
    (pd.Timestamp('2026-03-02T00:00:00+03:00'), pd.Timestamp('2026-03-09T00:00:00+03:00')),
    (pd.Timestamp('2026-02-23T00:00:00+03:00'), pd.Timestamp('2026-03-02T00:00:00+03:00')),
    (pd.Timestamp('2026-02-16T00:00:00+03:00'), pd.Timestamp('2026-02-23T00:00:00+03:00')),
    (pd.Timestamp('2026-02-09T00:00:00+03:00'), pd.Timestamp('2026-02-16T00:00:00+03:00'))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Filter POS data for analysis and baseline periods
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis['date'] = pos_analysis['timestamp'].dt.date

# Calculate daily revenue (excluding refunds)
daily_revenue_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('date')['line_total_sar'].sum()

# Baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline['date'] = pos_baseline['timestamp'].dt.date
    daily_rev = pos_baseline[~pos_baseline['is_refund']].groupby('date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue_analysis) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalous days
        for date, revenue in daily_revenue_analysis.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Daily Revenue Anomaly on {date}',
                    'claim': f'Daily revenue of {revenue:.2f} SAR on {date} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': round(revenue, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(date),
                            'period_end': str(date)
                        },
                        'baseline_mean': {
                            'value': round(baseline_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(baseline_periods[0][0].date()),
                            'period_end': str(baseline_periods[-1][1].date())
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(date),
                            'period_end': str(date)
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_revenues),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: {len(baseline_daily_revenues)} daily observations from 4 weeks prior',
                        'Refunds excluded from revenue calculation'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.0 standard deviations',
                        'Baseline periods are representative of normal operations',
                        'No structural breaks in business model'
                    ],
                    'confidence': 0.85
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis and baseline periods
traffic_analysis = traffic_df[(traffic_df['date'] >= pd.Timestamp(analysis_start.date())) & (traffic_df['date'] < pd.Timestamp(analysis_end.date()))].copy()
traffic_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]

# Baseline traffic
baseline_hourly_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[(traffic_df['date'] >= pd.Timestamp(period_start.date())) & (traffic_df['date'] < pd.Timestamp(period_end.date()))].copy()
    traffic_baseline = traffic_baseline[traffic_baseline['is_dead_sensor_day'] == False]
    baseline_hourly_traffic.extend(traffic_baseline['door_count'].values)

if len(baseline_hourly_traffic) > 0 and len(traffic_analysis) > 0:
    baseline_traffic_mean = np.mean(baseline_hourly_traffic)
    baseline_traffic_std = np.std(baseline_hourly_traffic)
    
    if baseline_traffic_std > 0:
        # Find anomalous hours
        for idx, row in traffic_analysis.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std
            if abs(z_score) > 2.5:  # 2.5 standard deviations for traffic
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date"].date()} at {row["hour"]}',
                    'claim': f'Door count of {int(row["door_count"])} at {row["hour"]} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(row['hour']),
                            'period_end': str(row['hour'])
                        },
                        'baseline_mean': {
                            'value': round(baseline_traffic_mean, 1),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(baseline_periods[0][0].date()),
                            'period_end': str(baseline_periods[-1][1].date())
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(row['hour']),
                            'period_end': str(row['hour'])
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_hourly_traffic),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: {len(baseline_hourly_traffic)} hourly observations from 4 weeks prior',
                        'Dead sensor days excluded'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.5 standard deviations',
                        'Baseline periods represent normal traffic patterns',
                        'No structural changes in foot traffic patterns'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Filter POS data for transaction counts
pos_analysis_tx = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis_tx['date'] = pos_analysis_tx['timestamp'].dt.date

# Count unique transactions per day
daily_tx_analysis = pos_analysis_tx.groupby('date')['transaction_id'].nunique()

# Baseline transaction counts
baseline_daily_tx = []
for period_start, period_end in baseline_periods:
    pos_baseline_tx = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline_tx['date'] = pos_baseline_tx['timestamp'].dt.date
    daily_tx = pos_baseline_tx.groupby('date')['transaction_id'].nunique()
    baseline_daily_tx.extend(daily_tx.values)

if len(baseline_daily_tx) > 0 and len(daily_tx_analysis) > 0:
    baseline_tx_mean = np.mean(baseline_daily_tx)
    baseline_tx_std = np.std(baseline_daily_tx)
    
    if baseline_tx_std > 0:
        # Find anomalous days
        for date, tx_count in daily_tx_analysis.items():
            z_score = (tx_count - baseline_tx_mean) / baseline_tx_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Daily Transaction Count Anomaly on {date}',
                    'claim': f'Transaction count of {int(tx_count)} on {date} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_tx_mean:.1f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'daily_transaction_count': {
                            'value': int(tx_count),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(date),
                            'period_end': str(date)
                        },
                        'baseline_mean': {
                            'value': round(baseline_tx_mean, 1),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(baseline_periods[0][0].date()),
                            'period_end': str(baseline_periods[-1][1].date())
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': str(date),
                            'period_end': str(date)
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_tx),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: {len(baseline_daily_tx)} daily observations from 4 weeks prior',
                        'Unique transaction_id count per day'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.0 standard deviations',
                        'Baseline periods are representative of normal transaction volume',
                        'No changes in transaction recording methodology'
                    ],
                    'confidence': 0.82
                })

# Sort findings by magnitude of z-score and limit to 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
