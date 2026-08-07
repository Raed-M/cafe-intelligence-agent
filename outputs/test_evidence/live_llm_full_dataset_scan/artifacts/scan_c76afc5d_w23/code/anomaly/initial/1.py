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
analysis_start = pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                      (pos_df['calendar_date'] < analysis_end)].copy()

daily_revenue_analysis = analysis_pos.groupby('calendar_date').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
daily_revenue_analysis.columns = ['date', 'revenue', 'transactions']

# Calculate baseline daily revenue
baseline_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_baseline = baseline_pos.groupby('calendar_date')['line_total_sar'].sum()
    baseline_revenues.extend(daily_baseline.values)

if len(baseline_revenues) > 0 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Check for anomalies in analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        if abs(z_score) > 2.0:  # 2 standard deviations
            findings.append({
                'title': f'Daily Revenue Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily revenue of {row["revenue"]:.2f} SAR on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'daily_revenue': {
                        'value': round(row['revenue'], 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-18T00:00:00',
                        'period_end': '2026-06-15T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks of historical data (2026-05-18 to 2026-06-15)',
                    f'Baseline sample size: {len(baseline_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks prior to analysis period',
                    'Revenue includes all line items, refunds netted per line_total_sar'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Calculate hourly traffic for analysis period
    hourly_traffic_analysis = analysis_traffic.groupby('hour')['door_count'].agg(['sum', 'count']).reset_index()
    hourly_traffic_analysis.columns = ['hour', 'total_count', 'day_count']
    hourly_traffic_analysis['avg_hourly'] = hourly_traffic_analysis['total_count'] / hourly_traffic_analysis['day_count']
    
    # Calculate baseline hourly traffic
    baseline_traffic_data = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        if len(baseline_traffic) > 0:
            baseline_traffic_data.append(baseline_traffic)
    
    if len(baseline_traffic_data) > 0:
        all_baseline_traffic = pd.concat(baseline_traffic_data, ignore_index=True)
        baseline_hourly = all_baseline_traffic.groupby('hour')['door_count'].agg(['sum', 'count']).reset_index()
        baseline_hourly.columns = ['hour', 'total_count', 'day_count']
        baseline_hourly['avg_hourly'] = baseline_hourly['total_count'] / baseline_hourly['day_count']
        
        # Find anomalies
        for idx, row in hourly_traffic_analysis.iterrows():
            hour_str = row['hour'].strftime('%H:00')
            baseline_row = baseline_hourly[baseline_hourly['hour'] == row['hour']]
            
            if len(baseline_row) > 0:
                baseline_avg = baseline_row.iloc[0]['avg_hourly']
                baseline_std = all_baseline_traffic[all_baseline_traffic['hour'] == row['hour']]['door_count'].std()
                
                if baseline_std > 0:
                    z_score = (row['avg_hourly'] - baseline_avg) / baseline_std
                    
                    if abs(z_score) > 2.5:  # Higher threshold for hourly data
                        findings.append({
                            'title': f'Hourly Traffic Anomaly at {hour_str}',
                            'claim': f'Average hourly traffic of {row["avg_hourly"]:.0f} door counts at {hour_str} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_avg:.0f}',
                            'finding_type': 'traffic_anomaly',
                            'metrics': {
                                'hourly_traffic': {
                                    'value': round(row['avg_hourly'], 0),
                                    'unit': 'door_counts',
                                    'numerator': int(row['total_count']),
                                    'denominator': int(row['day_count']),
                                    'period_start': analysis_start.isoformat(),
                                    'period_end': analysis_end.isoformat()
                                },
                                'baseline_hourly_mean': {
                                    'value': round(baseline_avg, 0),
                                    'unit': 'door_counts',
                                    'numerator': None,
                                    'denominator': None,
                                    'period_start': '2026-05-18T00:00:00',
                                    'period_end': '2026-06-15T00:00:00'
                                },
                                'z_score': {
                                    'value': round(z_score, 2),
                                    'unit': 'std_dev',
                                    'numerator': None,
                                    'denominator': None,
                                    'period_start': analysis_start.isoformat(),
                                    'period_end': analysis_end.isoformat()
                                }
                            },
                            'source_names': ['traffic'],
                            'sample_size': int(row['day_count']),
                            'coverage_notes': [
                                f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                                f'Baseline: 4 weeks of historical data (2026-05-18 to 2026-06-15)',
                                f'Dead sensor days excluded from analysis',
                                f'Hour {hour_str} observed on {int(row["day_count"])} days in analysis period'
                            ],
                            'assumptions': [
                                'Z-score threshold of 2.5 standard deviations used for hourly data',
                                'Dead sensor days excluded per is_dead_sensor_day flag',
                                'Baseline calculated from same hours across 4 weeks'
                            ],
                            'confidence': 0.80
                        })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

daily_transactions = analysis_pos.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_transactions.columns = ['date', 'transaction_count']

baseline_transactions = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_baseline = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_transactions.extend(daily_baseline.values)

if len(baseline_transactions) > 0 and np.std(baseline_transactions) > 0:
    baseline_mean_tx = np.mean(baseline_transactions)
    baseline_std_tx = np.std(baseline_transactions)
    
    for idx, row in daily_transactions.iterrows():
        z_score_tx = (row['transaction_count'] - baseline_mean_tx) / baseline_std_tx if baseline_std_tx > 0 else 0
        
        if abs(z_score_tx) > 2.0:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score_tx):.2f} standard deviations from baseline mean of {baseline_mean_tx:.0f}',
                'finding_type': 'transaction_volume_anomaly',
                'metrics': {
                    'daily_transactions': {
                        'value': int(row['transaction_count']),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_transactions': {
                        'value': round(baseline_mean_tx, 0),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-18T00:00:00',
                        'period_end': '2026-06-15T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score_tx, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_transactions),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks of historical data (2026-05-18 to 2026-06-15)',
                    f'Baseline sample size: {len(baseline_transactions)} daily observations',
                    'Transaction count based on unique transaction_id per day'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks prior to analysis period',
                    'Transaction count uses unique transaction_id values'
                ],
                'confidence': 0.82
            })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
