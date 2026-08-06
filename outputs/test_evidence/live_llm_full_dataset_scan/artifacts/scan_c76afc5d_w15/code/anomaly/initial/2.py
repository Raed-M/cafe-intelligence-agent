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

# Parse dates
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.to_datetime('2026-04-20T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-04-27T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-04-13T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-20T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-04-06T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-13T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-30T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-06T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-23T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-30T00:00:00+03:00').tz_localize(None))
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
    daily_rev = baseline_pos.groupby('calendar_date')['line_total_sar'].sum()
    baseline_revenues.extend(daily_rev.values)

if len(baseline_revenues) > 0 and np.var(baseline_revenues) > 0:
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
                    'observed_daily_revenue': {
                        'value': round(row['revenue'], 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_daily_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-03-23T00:00:00',
                        'period_end': '2026-04-20T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'standard_deviations',
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
                    f'Baseline: 4 weeks of historical data (2026-03-23 to 2026-04-20)',
                    f'Daily transactions on anomaly date: {int(row["transactions"])}'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks of historical data',
                    'Refunds included in net revenue calculation per metric definition'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Calculate hourly traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    analysis_traffic['hour_of_day'] = analysis_traffic['hour'].dt.hour
    hourly_traffic_analysis = analysis_traffic.groupby('hour_of_day')['door_count'].agg(['sum', 'count', 'mean']).reset_index()
    
    # Calculate baseline hourly traffic
    baseline_traffic_list = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
        hourly_traffic = baseline_traffic.groupby('hour_of_day')['door_count'].mean()
        baseline_traffic_list.append(hourly_traffic)
    
    if len(baseline_traffic_list) > 0:
        # Combine baseline data
        baseline_combined = pd.concat(baseline_traffic_list, axis=1).mean(axis=1)
        baseline_std_hourly = pd.concat(baseline_traffic_list, axis=1).std(axis=1)
        
        # Check for anomalies
        for idx, row in hourly_traffic_analysis.iterrows():
            hour = int(row['hour_of_day'])
            if hour in baseline_combined.index and baseline_std_hourly[hour] > 0:
                z_score = (row['mean'] - baseline_combined[hour]) / baseline_std_hourly[hour]
                
                if abs(z_score) > 2.0 and row['count'] >= 3:  # At least 3 observations
                    findings.append({
                        'title': f'Hourly Traffic Anomaly at {hour:02d}:00',
                        'claim': f'Average hourly traffic at {hour:02d}:00 was {row["mean"]:.1f} visitors/hour, deviating {abs(z_score):.2f} standard deviations from baseline mean of {baseline_combined[hour]:.1f}',
                        'finding_type': 'traffic_anomaly',
                        'metrics': {
                            'observed_hourly_traffic': {
                                'value': round(row['mean'], 1),
                                'unit': 'visitors/hour',
                                'numerator': int(row['sum']),
                                'denominator': int(row['count']),
                                'period_start': analysis_start.isoformat(),
                                'period_end': analysis_end.isoformat()
                            },
                            'baseline_mean_hourly_traffic': {
                                'value': round(baseline_combined[hour], 1),
                                'unit': 'visitors/hour',
                                'numerator': None,
                                'denominator': None,
                                'period_start': '2026-03-23T00:00:00',
                                'period_end': '2026-04-20T00:00:00'
                            },
                            'z_score': {
                                'value': round(z_score, 2),
                                'unit': 'standard_deviations',
                                'numerator': None,
                                'denominator': None,
                                'period_start': analysis_start.isoformat(),
                                'period_end': analysis_end.isoformat()
                            }
                        },
                        'source_names': ['traffic'],
                        'sample_size': int(row['count']),
                        'coverage_notes': [
                            f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                            f'Baseline: 4 weeks of historical data (2026-03-23 to 2026-04-20)',
                            f'Dead sensor days excluded from analysis',
                            f'Hour {hour:02d}:00 observations in analysis period: {int(row["count"])}'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                            'Baseline calculated from 4 complete weeks of historical data',
                            'Dead sensor days excluded per data quality notes'
                        ],
                        'confidence': 0.80
                    })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

daily_transactions_analysis = analysis_pos.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_transactions_analysis.columns = ['date', 'transaction_count']

# Calculate baseline daily transactions
baseline_transactions = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_trans = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_transactions.extend(daily_trans.values)

if len(baseline_transactions) > 0 and np.var(baseline_transactions) > 0:
    baseline_mean_trans = np.mean(baseline_transactions)
    baseline_std_trans = np.std(baseline_transactions)
    
    # Check for anomalies
    for idx, row in daily_transactions_analysis.iterrows():
        z_score = (row['transaction_count'] - baseline_mean_trans) / baseline_std_trans if baseline_std_trans > 0 else 0
        
        if abs(z_score) > 2.0:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_trans:.0f} transactions',
                'finding_type': 'transaction_volume_anomaly',
                'metrics': {
                    'observed_daily_transactions': {
                        'value': int(row['transaction_count']),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_daily_transactions': {
                        'value': round(baseline_mean_trans, 0),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-03-23T00:00:00',
                        'period_end': '2026-04-20T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': 'standard_deviations',
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
                    f'Baseline: 4 weeks of historical data (2026-03-23 to 2026-04-20)',
                    f'Transaction count based on unique transaction_id values'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks of historical data',
                    'Transaction count includes refunds as separate transactions'
                ],
                'confidence': 0.85
            })

# Sort findings by magnitude (z-score) and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
result = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
