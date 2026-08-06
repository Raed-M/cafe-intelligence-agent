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
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.Timestamp('2026-04-27T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')
analysis_end = pd.Timestamp('2026-05-04T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')

baseline_periods = [
    (pd.Timestamp('2026-04-20T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-04-27T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-04-13T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-04-20T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-04-06T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-04-13T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
    (pd.Timestamp('2026-03-30T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh'),
     pd.Timestamp('2026-04-06T00:00:00+03:00', tz='UTC').tz_convert('Asia/Riyadh')),
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period (excluding refunds per net definition)
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

pos_analysis['calendar_date'] = pd.to_datetime(pos_analysis['calendar_date'])
daily_revenue_analysis = pos_analysis.groupby('calendar_date')['line_total_sar'].sum()

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp_local'] >= period_start) & 
        (pos_df['timestamp_local'] < period_end) &
        (pos_df['is_refund'] == False)
    ].copy()
    pos_baseline['calendar_date'] = pd.to_datetime(pos_baseline['calendar_date'])
    daily_rev = pos_baseline.groupby('calendar_date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue_analysis) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find the day with highest z-score
        z_scores = (daily_revenue_analysis.values - baseline_mean) / baseline_std
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z_score = z_scores[max_z_idx]
        anomaly_date = daily_revenue_analysis.index[max_z_idx]
        anomaly_revenue = daily_revenue_analysis.iloc[max_z_idx]
        
        if abs(max_z_score) >= 2.0:  # 2-sigma threshold
            findings.append({
                'title': 'Unusual Daily Revenue Spike',
                'claim': f'Daily revenue on {anomaly_date.date()} reached {anomaly_revenue:.2f} SAR, {abs(max_z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'daily_revenue_anomaly_date': {
                        'value': round(anomaly_revenue, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date.strftime('%Y-%m-%dT00:00:00+03:00'),
                        'period_end': (anomaly_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                    },
                    'baseline_mean_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-03-30T00:00:00+03:00',
                        'period_end': '2026-04-27T00:00:00+03:00'
                    },
                    'z_score_revenue': {
                        'value': round(max_z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date.strftime('%Y-%m-%dT00:00:00+03:00'),
                        'period_end': (anomaly_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    'Analysis period: 2026-04-27 to 2026-05-04',
                    'Baseline: 4 weeks prior (2026-03-30 to 2026-04-27)',
                    'Refunds excluded from revenue calculation',
                    f'Baseline sample size: {len(baseline_daily_revenues)} daily observations'
                ],
                'assumptions': [
                    'Normal distribution of daily revenue',
                    '2-sigma threshold for anomaly detection',
                    'Baseline periods are representative of expected behavior',
                    'No structural breaks in business operations'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start.date()) & 
    (traffic_df['date'] < analysis_end.date()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

traffic_baseline = traffic_df[
    (traffic_df['date'] >= baseline_periods[0][0].date()) & 
    (traffic_df['date'] < baseline_periods[0][1].date()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

if len(traffic_baseline) > 0 and len(traffic_analysis) > 0:
    baseline_hourly_traffic = traffic_baseline['door_count'].values
    baseline_traffic_mean = np.mean(baseline_hourly_traffic)
    baseline_traffic_std = np.std(baseline_hourly_traffic)
    
    if baseline_traffic_std > 0:
        z_scores_traffic = (traffic_analysis['door_count'].values - baseline_traffic_mean) / baseline_traffic_std
        max_z_idx_traffic = np.argmax(np.abs(z_scores_traffic))
        max_z_score_traffic = z_scores_traffic[max_z_idx_traffic]
        anomaly_hour_traffic = traffic_analysis.iloc[max_z_idx_traffic]
        
        if abs(max_z_score_traffic) >= 2.0:
            hour_start = pd.Timestamp(anomaly_hour_traffic['hour'])
            hour_end = hour_start + timedelta(hours=1)
            findings.append({
                'title': 'Unusual Hourly Traffic Spike',
                'claim': f'Door count on {anomaly_hour_traffic["date"].date()} at hour {anomaly_hour_traffic["hour"]} reached {anomaly_hour_traffic["door_count"]:.0f} visitors, {abs(max_z_score_traffic):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.2f}.',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'hourly_door_count_anomaly': {
                        'value': int(anomaly_hour_traffic['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': hour_start.strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                        'period_end': hour_end.strftime('%Y-%m-%dT%H:%M:%S+03:00')
                    },
                    'baseline_mean_traffic': {
                        'value': round(baseline_traffic_mean, 2),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-04-20T00:00:00+03:00',
                        'period_end': '2026-04-27T00:00:00+03:00'
                    },
                    'z_score_traffic': {
                        'value': round(max_z_score_traffic, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': hour_start.strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                        'period_end': hour_end.strftime('%Y-%m-%dT%H:%M:%S+03:00')
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_hourly_traffic),
                'coverage_notes': [
                    'Analysis period: 2026-04-27 to 2026-05-04',
                    'Baseline: week of 2026-04-20 to 2026-04-27',
                    'Dead sensor days excluded',
                    f'Baseline sample size: {len(baseline_hourly_traffic)} hourly observations'
                ],
                'assumptions': [
                    'Normal distribution of hourly traffic',
                    '2-sigma threshold for anomaly detection',
                    'Sensor data is reliable and consistent',
                    'No structural changes in store layout or hours'
                ],
                'confidence': 0.80
            })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

pos_analysis_txn = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

pos_analysis_txn['calendar_date'] = pd.to_datetime(pos_analysis_txn['calendar_date'])
daily_txn_analysis = pos_analysis_txn.groupby('calendar_date')['transaction_id'].nunique()

baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    pos_baseline_txn = pos_df[
        (pos_df['timestamp_local'] >= period_start) & 
        (pos_df['timestamp_local'] < period_end)
    ].copy()
    pos_baseline_txn['calendar_date'] = pd.to_datetime(pos_baseline_txn['calendar_date'])
    daily_txn = pos_baseline_txn.groupby('calendar_date')['transaction_id'].nunique()
    baseline_daily_txns.extend(daily_txn.values)

if len(baseline_daily_txns) > 0 and len(daily_txn_analysis) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    if baseline_txn_std > 0:
        z_scores_txn = (daily_txn_analysis.values - baseline_txn_mean) / baseline_txn_std
        max_z_idx_txn = np.argmax(np.abs(z_scores_txn))
        max_z_score_txn = z_scores_txn[max_z_idx_txn]
        anomaly_date_txn = daily_txn_analysis.index[max_z_idx_txn]
        anomaly_txn_count = daily_txn_analysis.iloc[max_z_idx_txn]
        
        if abs(max_z_score_txn) >= 2.0:
            findings.append({
                'title': 'Unusual Daily Transaction Count',
                'claim': f'Daily transaction count on {anomaly_date_txn.date()} reached {anomaly_txn_count:.0f} baskets, {abs(max_z_score_txn):.2f} standard deviations from baseline mean of {baseline_txn_mean:.2f}.',
                'finding_type': 'transaction_anomaly',
                'metrics': {
                    'daily_transaction_count_anomaly': {
                        'value': int(anomaly_txn_count),
                        'unit': 'baskets',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date_txn.strftime('%Y-%m-%dT00:00:00+03:00'),
                        'period_end': (anomaly_date_txn + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                    },
                    'baseline_mean_transactions': {
                        'value': round(baseline_txn_mean, 2),
                        'unit': 'baskets',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-03-30T00:00:00+03:00',
                        'period_end': '2026-04-27T00:00:00+03:00'
                    },
                    'z_score_transactions': {
                        'value': round(max_z_score_txn, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date_txn.strftime('%Y-%m-%dT00:00:00+03:00'),
                        'period_end': (anomaly_date_txn + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_txns),
                'coverage_notes': [
                    'Analysis period: 2026-04-27 to 2026-05-04',
                    'Baseline: 4 weeks prior (2026-03-30 to 2026-04-27)',
                    'Transaction count includes all transactions (refunds and sales)',
                    f'Baseline sample size: {len(baseline_daily_txns)} daily observations'
                ],
                'assumptions': [
                    'Normal distribution of daily transaction counts',
                    '2-sigma threshold for anomaly detection',
                    'Baseline periods are representative of expected behavior',
                    'No changes in POS system or transaction recording'
                ],
                'confidence': 0.82
            })

# Sort findings by z-score magnitude
findings.sort(key=lambda x: abs(float(x['metrics'][list(x['metrics'].keys())[-1]]['value'])), reverse=True)

# Keep only top 3
findings = findings[:3]

# Prepare output
result = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
