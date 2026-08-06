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
analysis_start = pd.to_datetime("2026-05-18T00:00:00+03:00").tz_localize(None)
analysis_end = pd.to_datetime("2026-05-25T00:00:00+03:00").tz_localize(None)

baseline_periods = [
    (pd.to_datetime("2026-05-11T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-05-18T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-05-04T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-05-11T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-04-27T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-05-04T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-04-20T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-04-27T00:00:00+03:00").tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                      (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_revenue = analysis_pos.groupby('calendar_date')['line_total_sar'].sum()

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    baseline_daily = baseline_pos.groupby('calendar_date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(baseline_daily.values)

if len(baseline_daily_revenues) > 0 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Calculate z-scores for analysis period
        analysis_daily_revenue_values = analysis_daily_revenue.values
        z_scores = (analysis_daily_revenue_values - baseline_mean) / baseline_std
        
        # Find max anomaly
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z_score = z_scores[max_z_idx]
        max_z_date = analysis_daily_revenue.index[max_z_idx]
        max_z_value = analysis_daily_revenue_values[max_z_idx]
        
        if abs(max_z_score) > 2.0:  # 2 standard deviations
            findings.append({
                'title': 'Unusual Daily Revenue',
                'claim': f'Daily revenue on {max_z_date.date()} was {max_z_value:.2f} SAR, {abs(max_z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'daily_revenue': {
                        'value': round(max_z_value, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': max_z_date.isoformat(),
                        'period_end': (max_z_date + timedelta(days=1)).isoformat()
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
                        'value': round(max_z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': max_z_date.isoformat(),
                        'period_end': (max_z_date + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks of historical data',
                    f'Baseline sample size: {len(baseline_daily_revenues)} daily observations'
                ],
                'assumptions': [
                    'Normal distribution of daily revenue',
                    'No structural breaks in baseline period',
                    'Threshold: |z-score| > 2.0'
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
    analysis_hourly_traffic = analysis_traffic.groupby('hour_of_day')['door_count'].sum()
    
    # Calculate baseline hourly traffic
    baseline_traffic_list = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
        baseline_hourly = baseline_traffic.groupby('hour_of_day')['door_count'].sum()
        baseline_traffic_list.append(baseline_hourly)
    
    if len(baseline_traffic_list) > 0:
        # Combine baseline data
        baseline_combined = pd.concat(baseline_traffic_list, axis=1).mean(axis=1)
        
        # Calculate z-scores for analysis period
        if baseline_combined.std() > 0:
            z_scores_traffic = (analysis_hourly_traffic - baseline_combined.mean()) / baseline_combined.std()
            
            # Find max anomaly
            max_traffic_z_idx = np.argmax(np.abs(z_scores_traffic))
            max_traffic_z_score = z_scores_traffic.iloc[max_traffic_z_idx]
            max_traffic_hour = analysis_hourly_traffic.index[max_traffic_z_idx]
            max_traffic_value = analysis_hourly_traffic.iloc[max_traffic_z_idx]
            baseline_traffic_value = baseline_combined.iloc[max_traffic_hour] if max_traffic_hour in baseline_combined.index else 0
            
            if abs(max_traffic_z_score) > 2.0:
                findings.append({
                    'title': 'Unusual Hourly Traffic Pattern',
                    'claim': f'Hour {max_traffic_hour}:00 had {max_traffic_value:.0f} door counts, {abs(max_traffic_z_score):.2f} standard deviations from baseline mean of {baseline_traffic_value:.0f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(max_traffic_value),
                            'unit': 'door_counts',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f'{analysis_start.date()}T{max_traffic_hour:02d}:00:00',
                            'period_end': f'{analysis_start.date()}T{max_traffic_hour+1:02d}:00:00'
                        },
                        'baseline_mean': {
                            'value': round(baseline_traffic_value, 2),
                            'unit': 'door_counts',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(max_traffic_z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f'{analysis_start.date()}T{max_traffic_hour:02d}:00:00',
                            'period_end': f'{analysis_start.date()}T{max_traffic_hour+1:02d}:00:00'
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic_list) * 24,
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        f'Excluded dead sensor days',
                        f'Baseline: 4 weeks of hourly data'
                    ],
                    'assumptions': [
                        'Normal distribution of hourly traffic',
                        'Threshold: |z-score| > 2.0',
                        'Dead sensor days excluded'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count for analysis period
analysis_pos_txn = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                          (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_txn = analysis_pos_txn.groupby('calendar_date')['transaction_id'].nunique()

# Calculate daily transaction count for baseline periods
baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    baseline_pos_txn = pos_df[(pos_df['calendar_date'] >= period_start) & 
                              (pos_df['calendar_date'] < period_end)].copy()
    baseline_daily_txn = baseline_pos_txn.groupby('calendar_date')['transaction_id'].nunique()
    baseline_daily_txns.extend(baseline_daily_txn.values)

if len(baseline_daily_txns) > 0 and len(analysis_daily_txn) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    if baseline_txn_std > 0:
        # Calculate z-scores for analysis period
        analysis_daily_txn_values = analysis_daily_txn.values
        z_scores_txn = (analysis_daily_txn_values - baseline_txn_mean) / baseline_txn_std
        
        # Find max anomaly
        max_txn_z_idx = np.argmax(np.abs(z_scores_txn))
        max_txn_z_score = z_scores_txn[max_txn_z_idx]
        max_txn_date = analysis_daily_txn.index[max_txn_z_idx]
        max_txn_value = analysis_daily_txn_values[max_txn_z_idx]
        
        if abs(max_txn_z_score) > 2.0:
            findings.append({
                'title': 'Unusual Daily Transaction Count',
                'claim': f'Daily transaction count on {max_txn_date.date()} was {max_txn_value:.0f}, {abs(max_txn_z_score):.2f} standard deviations from baseline mean of {baseline_txn_mean:.0f}',
                'finding_type': 'transaction_anomaly',
                'metrics': {
                    'daily_transaction_count': {
                        'value': int(max_txn_value),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': max_txn_date.isoformat(),
                        'period_end': (max_txn_date + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_txn_mean, 2),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(max_txn_z_score, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': max_txn_date.isoformat(),
                        'period_end': (max_txn_date + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily_txns),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                    f'Baseline: 4 weeks of historical data',
                    f'Baseline sample size: {len(baseline_daily_txns)} daily observations'
                ],
                'assumptions': [
                    'Normal distribution of daily transaction counts',
                    'No structural breaks in baseline period',
                    'Threshold: |z-score| > 2.0'
                ],
                'confidence': 0.82
            })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: x['confidence'], reverse=True)

# Keep only top 3
findings = findings[:3]

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
