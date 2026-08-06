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

# Load data
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define periods
analysis_start = pd.Timestamp("2026-05-11T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-05-18T00:00:00+03:00")
previous_start = pd.Timestamp("2026-05-04T00:00:00+03:00")
previous_end = pd.Timestamp("2026-05-11T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-05-04T00:00:00+03:00"), pd.Timestamp("2026-05-11T00:00:00+03:00")),
    (pd.Timestamp("2026-04-27T00:00:00+03:00"), pd.Timestamp("2026-05-04T00:00:00+03:00")),
    (pd.Timestamp("2026-04-20T00:00:00+03:00"), pd.Timestamp("2026-04-27T00:00:00+03:00")),
    (pd.Timestamp("2026-04-13T00:00:00+03:00"), pd.Timestamp("2026-04-20T00:00:00+03:00")),
]

findings = []

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Ensure timezone awareness
if pos_df['timestamp'].dt.tz is None:
    pos_df['timestamp'] = pos_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Riyadh')
if traffic_df['hour'].dt.tz is None:
    traffic_df['hour'] = traffic_df['hour'].dt.tz_localize('UTC').dt.tz_convert('Asia/Riyadh')
if reviews_df['date'].dt.tz is None:
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Riyadh')

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue (excluding refunds)
pos_df['is_refund_bool'] = pos_df['is_refund'].astype(bool)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis['date'] = pos_analysis['timestamp'].dt.date

daily_revenue_analysis = pos_analysis[~pos_analysis['is_refund_bool']].groupby('date')['line_total_sar'].sum()

# Calculate baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline['date'] = pos_baseline['timestamp'].dt.date
    daily_rev = pos_baseline[~pos_baseline['is_refund_bool']].groupby('date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue_analysis) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalies in analysis period
        for date, revenue in daily_revenue_analysis.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2-sigma threshold
                findings.append({
                    'title': f'Unusual Daily Revenue on {date}',
                    'claim': f'Daily revenue of {revenue:.2f} SAR on {date} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'observed_daily_revenue': {
                            'value': round(revenue, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
                        },
                        'baseline_mean_daily_revenue': {
                            'value': round(baseline_mean, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score_revenue': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
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
                        'Normal distribution of daily revenue',
                        '2-sigma threshold for anomaly detection',
                        'Baseline periods are representative of normal operations'
                    ],
                    'confidence': 0.85
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic data for analysis period
traffic_analysis = traffic_df[(traffic_df['hour'] >= analysis_start) & (traffic_df['hour'] < analysis_end)].copy()
traffic_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]  # Exclude dead sensor days
traffic_analysis['hour_only'] = traffic_analysis['hour'].dt.hour

# Calculate baseline hourly traffic
baseline_hourly_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[(traffic_df['hour'] >= period_start) & (traffic_df['hour'] < period_end)].copy()
    traffic_baseline = traffic_baseline[traffic_baseline['is_dead_sensor_day'] == False]
    baseline_hourly_traffic.extend(traffic_baseline['door_count'].values)

if len(baseline_hourly_traffic) > 0 and len(traffic_analysis) > 0:
    baseline_mean_traffic = np.mean(baseline_hourly_traffic)
    baseline_std_traffic = np.std(baseline_hourly_traffic)
    
    if baseline_std_traffic > 0:
        # Find anomalies
        for idx, row in traffic_analysis.iterrows():
            z_score = (row['door_count'] - baseline_mean_traffic) / baseline_std_traffic
            if abs(z_score) > 2.5:  # 2.5-sigma threshold for traffic
                findings.append({
                    'title': f'Unusual Hourly Traffic on {row["hour"].date()} at {row["hour"].hour}:00',
                    'claim': f'Door count of {int(row["door_count"])} at {row["hour"]} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
                        },
                        'baseline_mean_hourly_door_count': {
                            'value': round(baseline_mean_traffic, 1),
                            'unit': 'count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score_traffic': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
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
                        'Normal distribution of hourly traffic',
                        '2.5-sigma threshold for anomaly detection',
                        'Baseline periods are representative of normal operations'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count
pos_analysis_txn = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_analysis_txn['date'] = pos_analysis_txn['timestamp'].dt.date
daily_txn_analysis = pos_analysis_txn.groupby('date')['transaction_id'].nunique()

# Calculate baseline daily transaction count
baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    pos_baseline_txn = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)].copy()
    pos_baseline_txn['date'] = pos_baseline_txn['timestamp'].dt.date
    daily_txn = pos_baseline_txn.groupby('date')['transaction_id'].nunique()
    baseline_daily_txns.extend(daily_txn.values)

if len(baseline_daily_txns) > 0 and len(daily_txn_analysis) > 0:
    baseline_mean_txn = np.mean(baseline_daily_txns)
    baseline_std_txn = np.std(baseline_daily_txns)
    
    if baseline_std_txn > 0:
        # Find anomalies
        for date, txn_count in daily_txn_analysis.items():
            z_score = (txn_count - baseline_mean_txn) / baseline_std_txn
            if abs(z_score) > 2.0:  # 2-sigma threshold
                findings.append({
                    'title': f'Unusual Daily Transaction Count on {date}',
                    'claim': f'Transaction count of {int(txn_count)} on {date} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_txn:.1f}',
                    'finding_type': 'transaction_anomaly',
                    'metrics': {
                        'observed_daily_transaction_count': {
                            'value': int(txn_count),
                            'unit': 'count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
                        },
                        'baseline_mean_daily_transaction_count': {
                            'value': round(baseline_mean_txn, 1),
                            'unit': 'count',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score_transactions': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_txns),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Baseline: {len(baseline_daily_txns)} daily observations from 4 weeks prior',
                        'Unique transaction_id count per day'
                    ],
                    'assumptions': [
                        'Normal distribution of daily transaction counts',
                        '2-sigma threshold for anomaly detection',
                        'Baseline periods are representative of normal operations'
                    ],
                    'confidence': 0.82
                })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score_revenue', {}).get('value', 0) or 
                                                       x['metrics'].get('z_score_traffic', {}).get('value', 0) or 
                                                       x['metrics'].get('z_score_transactions', {}).get('value', 0)), reverse=True)
findings_sorted = findings_sorted[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
