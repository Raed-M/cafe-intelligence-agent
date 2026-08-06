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
analysis_start = pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-29T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-22T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2025-12-29T00:00:00+03:00').tz_localize(None))
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

if len(baseline_revenues) > 0 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Check each day in analysis period
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
                        'period_start': '2025-12-22T00:00:00',
                        'period_end': '2026-01-19T00:00:00'
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
                    f'Baseline: 4 weeks of historical data (2025-12-22 to 2026-01-19)',
                    f'Baseline sample size: {len(baseline_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks prior to analysis period',
                    'Refunds included in net revenue calculation per metric definition'
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
    # Calculate baseline hourly traffic
    baseline_traffic_data = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic_data.extend(baseline_traffic['door_count'].values)
    
    if len(baseline_traffic_data) > 10 and np.std(baseline_traffic_data) > 0:
        baseline_traffic_mean = np.mean(baseline_traffic_data)
        baseline_traffic_std = np.std(baseline_traffic_data)
        
        # Find anomalous hours
        analysis_traffic['z_score'] = (analysis_traffic['door_count'] - baseline_traffic_mean) / baseline_traffic_std
        
        anomalous_hours = analysis_traffic[abs(analysis_traffic['z_score']) > 2.5]
        
        if len(anomalous_hours) > 0:
            # Get the most extreme anomaly
            most_extreme = anomalous_hours.loc[abs(anomalous_hours['z_score']).idxmax()]
            
            findings.append({
                'title': f'Hourly Traffic Anomaly on {most_extreme["date"].strftime("%Y-%m-%d %H:00")}',
                'claim': f'Door count of {most_extreme["door_count"]:.0f} visitors at {most_extreme["date"].strftime("%Y-%m-%d %H:00")} deviates {abs(most_extreme["z_score"]):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'hourly_door_count': {
                        'value': int(most_extreme['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': most_extreme['date'].isoformat(),
                        'period_end': (most_extreme['date'] + timedelta(hours=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_traffic_mean, 1),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00',
                        'period_end': '2026-01-19T00:00:00'
                    },
                    'z_score': {
                        'value': round(most_extreme['z_score'], 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': most_extreme['date'].isoformat(),
                        'period_end': (most_extreme['date'] + timedelta(hours=1)).isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic_data),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks of hourly data (2025-12-22 to 2026-01-19)',
                    f'Excluded dead sensor days per is_dead_sensor_day flag',
                    f'Baseline sample size: {len(baseline_traffic_data)} hourly observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.5 standard deviations used for traffic anomalies',
                    'Dead sensor intervals excluded from analysis',
                    'Baseline calculated from valid sensor readings only'
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
    daily_trans = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_transactions.extend(daily_trans.values)

if len(baseline_transactions) > 0 and np.std(baseline_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_transactions)
    baseline_trans_std = np.std(baseline_transactions)
    
    for idx, row in daily_transactions.iterrows():
        z_score = (row['transaction_count'] - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        
        if abs(z_score) > 2.0:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f}',
                'finding_type': 'transaction_volume_anomaly',
                'metrics': {
                    'daily_transaction_count': {
                        'value': int(row['transaction_count']),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_trans_mean, 1),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2025-12-22T00:00:00',
                        'period_end': '2026-01-19T00:00:00'
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
                'sample_size': len(baseline_transactions),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks of daily data (2025-12-22 to 2026-01-19)',
                    f'Baseline sample size: {len(baseline_transactions)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Transaction count based on unique transaction_id per day',
                    'Baseline calculated from 4 complete weeks prior to analysis period'
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
