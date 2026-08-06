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
analysis_start = pd.Timestamp("2026-01-26T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-02-02T00:00:00+03:00")
previous_start = pd.Timestamp("2026-01-19T00:00:00+03:00")
previous_end = pd.Timestamp("2026-01-26T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-01-19T00:00:00+03:00"), pd.Timestamp("2026-01-26T00:00:00+03:00")),
    (pd.Timestamp("2026-01-12T00:00:00+03:00"), pd.Timestamp("2026-01-19T00:00:00+03:00")),
    (pd.Timestamp("2026-01-05T00:00:00+03:00"), pd.Timestamp("2026-01-12T00:00:00+03:00")),
    (pd.Timestamp("2025-12-29T00:00:00+03:00"), pd.Timestamp("2026-01-05T00:00:00+03:00")),
]

baseline_start = baseline_periods[-1][0]
baseline_end = baseline_periods[0][1]

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('Asia/Riyadh')
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True).dt.tz_convert('Asia/Riyadh')
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True).dt.tz_convert('Asia/Riyadh')

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis['date'] = pos_analysis['timestamp'].dt.date

daily_revenue_analysis = pos_analysis.groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
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
                date_str = str(row['date'])
                findings.append({
                    'title': f"Daily Revenue Anomaly on {date_str}",
                    'claim': f"Daily revenue of {row['revenue']:.2f} SAR on {date_str} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR",
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': round(row['revenue'], 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{date_str}T23:59:59+03:00"
                        },
                        'baseline_mean_daily_revenue': {
                            'value': round(baseline_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_start.isoformat(),
                            'period_end': baseline_end.isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{date_str}T23:59:59+03:00"
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_revenues),
                    'coverage_notes': [
                        f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                        f"Baseline: 4 weeks of historical data from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                        f"Baseline sample size: {len(baseline_daily_revenues)} daily observations"
                    ],
                    'assumptions': [
                        'Normal distribution of daily revenue',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'Refunds included in net revenue calculation',
                        'No product launches or known events during baseline'
                    ],
                    'confidence': 0.95
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & 
                               (traffic_df['date'] < analysis_end.date()) &
                               (traffic_df['is_dead_sensor_day'] == False)].copy()

# Filter baseline traffic
baseline_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[(traffic_df['date'] >= period_start.date()) & 
                                   (traffic_df['date'] < period_end.date()) &
                                   (traffic_df['is_dead_sensor_day'] == False)]
    baseline_traffic.append(traffic_baseline)

baseline_traffic_combined = pd.concat(baseline_traffic, ignore_index=True)

if len(traffic_analysis) > 0 and len(baseline_traffic_combined) > 0:
    # Group by hour for analysis
    traffic_analysis_hourly = traffic_analysis.groupby('hour')['door_count'].agg(['mean', 'std', 'count']).reset_index()
    baseline_traffic_hourly = baseline_traffic_combined.groupby('hour')['door_count'].agg(['mean', 'std', 'count']).reset_index()
    
    # Merge to compare
    comparison = traffic_analysis_hourly.merge(baseline_traffic_hourly, on='hour', suffixes=('_analysis', '_baseline'))
    
    # Calculate z-scores for each hour
    anomalous_hours = []
    for idx, row in comparison.iterrows():
        if row['std_baseline'] > 0 and row['count_baseline'] >= 3:
            z_score = (row['mean_analysis'] - row['mean_baseline']) / row['std_baseline']
            if abs(z_score) > 2.0:
                anomalous_hours.append({
                    'hour': int(row['hour']),
                    'analysis_mean': row['mean_analysis'],
                    'baseline_mean': row['mean_baseline'],
                    'z_score': z_score,
                    'baseline_count': int(row['count_baseline'])
                })
    
    # Sort by magnitude and take top anomaly
    if anomalous_hours:
        anomalous_hours.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_anomaly = anomalous_hours[0]
        
        findings.append({
            'title': f"Hourly Traffic Anomaly at Hour {top_anomaly['hour']}",
            'claim': f"Average hourly door count of {top_anomaly['analysis_mean']:.1f} at hour {top_anomaly['hour']} during analysis period is {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {top_anomaly['baseline_mean']:.1f}",
            'finding_type': 'traffic_anomaly',
            'metrics': {
                'hourly_door_count_analysis': {
                    'value': round(top_anomaly['analysis_mean'], 1),
                    'unit': 'door_count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': analysis_start.isoformat(),
                    'period_end': analysis_end.isoformat()
                },
                'hourly_door_count_baseline': {
                    'value': round(top_anomaly['baseline_mean'], 1),
                    'unit': 'door_count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': baseline_start.isoformat(),
                    'period_end': baseline_end.isoformat()
                },
                'z_score': {
                    'value': round(top_anomaly['z_score'], 2),
                    'unit': 'standard_deviations',
                    'numerator': None,
                    'denominator': None,
                    'period_start': analysis_start.isoformat(),
                    'period_end': analysis_end.isoformat()
                }
            },
            'source_names': ['traffic'],
            'sample_size': top_anomaly['baseline_count'],
            'coverage_notes': [
                f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                f"Baseline: 4 weeks of historical data from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                f"Excluded dead sensor days from analysis",
                f"Hour {top_anomaly['hour']} baseline sample size: {top_anomaly['baseline_count']} observations"
            ],
            'assumptions': [
                'Normal distribution of hourly door counts',
                'Z-score threshold of 2.0 (95% confidence)',
                'Dead sensor days excluded from baseline',
                'Hour is consistent across all dates'
            ],
            'confidence': 0.95
        })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count for analysis period
pos_analysis_txn = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis_txn['date'] = pos_analysis_txn['timestamp'].dt.date

daily_txn_analysis = pos_analysis_txn.groupby('date')['transaction_id'].nunique().reset_index()
daily_txn_analysis.columns = ['date', 'transaction_count']

# Calculate baseline daily transaction count
baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    pos_baseline_txn = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline_txn['date'] = pos_baseline_txn['timestamp'].dt.date
    daily_txn = pos_baseline_txn.groupby('date')['transaction_id'].nunique()
    baseline_daily_txns.extend(daily_txn.values)

if len(baseline_daily_txns) > 0 and len(daily_txn_analysis) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    if baseline_txn_std > 0:
        # Find anomalies in analysis period
        for idx, row in daily_txn_analysis.iterrows():
            z_score = (row['transaction_count'] - baseline_txn_mean) / baseline_txn_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                date_str = str(row['date'])
                findings.append({
                    'title': f"Daily Transaction Count Anomaly on {date_str}",
                    'claim': f"Daily transaction count of {row['transaction_count']} on {date_str} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_txn_mean:.1f}",
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'daily_transaction_count': {
                            'value': int(row['transaction_count']),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{date_str}T23:59:59+03:00"
                        },
                        'baseline_mean_daily_transaction_count': {
                            'value': round(baseline_txn_mean, 1),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_start.isoformat(),
                            'period_end': baseline_end.isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{date_str}T23:59:59+03:00"
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_txns),
                    'coverage_notes': [
                        f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                        f"Baseline: 4 weeks of historical data from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                        f"Baseline sample size: {len(baseline_daily_txns)} daily observations",
                        "Transaction count based on unique transaction_id per day"
                    ],
                    'assumptions': [
                        'Normal distribution of daily transaction counts',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'Transaction_id uniqueness indicates distinct baskets',
                        'No known operational changes during baseline'
                    ],
                    'confidence': 0.95
                })

# Sort findings by magnitude (z-score) and keep top 3
findings.sort(key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)
findings = findings[:3]

# Prepare output
result = {
    'status': 'success' if findings else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
