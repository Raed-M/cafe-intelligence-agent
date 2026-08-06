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

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-11T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-04T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-11T00:00:00+03:00').tz_localize(None))
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
        
        # Flag if z-score > 2 (2 standard deviations)
        if abs(z_score) > 2:
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
                        'period_start': '2026-05-04',
                        'period_end': '2026-06-01'
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
                'coverage_notes': [f'Baseline calculated from {len(baseline_revenues)} daily observations across 4 weeks prior to analysis period'],
                'assumptions': ['Normal distribution of daily revenue', 'Z-score threshold of 2.0 standard deviations'],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period and exclude dead sensor days
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Extract hour of day
    analysis_traffic['hour_of_day'] = analysis_traffic['hour'].dt.hour
    
    # Calculate baseline hourly traffic
    baseline_traffic = traffic_df[(traffic_df['date'] >= baseline_periods[0][0]) & 
                                  (traffic_df['date'] < baseline_periods[0][1]) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    
    if len(baseline_traffic) > 0:
        baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
        
        # Group by hour of day for baseline
        hourly_baseline = baseline_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
        
        # Check analysis period hourly traffic
        analysis_hourly = analysis_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
        
        for idx, row in analysis_hourly.iterrows():
            hour = row['hour_of_day']
            baseline_row = hourly_baseline[hourly_baseline['hour_of_day'] == hour]
            
            if len(baseline_row) > 0:
                baseline_mean = baseline_row.iloc[0]['mean']
                baseline_std = baseline_row.iloc[0]['std']
                
                if baseline_std > 0:
                    z_score = (row['mean'] - baseline_mean) / baseline_std
                    
                    if abs(z_score) > 2:
                        findings.append({
                            'title': f'Hourly Traffic Anomaly at {int(hour):02d}:00',
                            'claim': f'Average hourly traffic at {int(hour):02d}:00 is {row["mean"]:.0f} visitors, deviating {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.0f}',
                            'finding_type': 'traffic_anomaly',
                            'metrics': {
                                'hourly_traffic_mean': {
                                    'value': round(row['mean'], 1),
                                    'unit': 'visitors',
                                    'numerator': None,
                                    'denominator': None,
                                    'period_start': analysis_start.isoformat(),
                                    'period_end': analysis_end.isoformat()
                                },
                                'baseline_hourly_mean': {
                                    'value': round(baseline_mean, 1),
                                    'unit': 'visitors',
                                    'numerator': None,
                                    'denominator': None,
                                    'period_start': baseline_periods[0][0].isoformat(),
                                    'period_end': baseline_periods[0][1].isoformat()
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
                            'sample_size': int(baseline_row.iloc[0]['count']),
                            'coverage_notes': [f'Baseline calculated from {int(baseline_row.iloc[0]["count"])} observations at hour {int(hour):02d}:00', 'Excluded dead sensor days'],
                            'assumptions': ['Normal distribution of hourly traffic', 'Z-score threshold of 2.0 standard deviations'],
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

if len(baseline_transactions) > 0 and np.std(baseline_transactions) > 0:
    baseline_mean_trans = np.mean(baseline_transactions)
    baseline_std_trans = np.std(baseline_transactions)
    
    # Check each day in analysis period
    for idx, row in daily_transactions_analysis.iterrows():
        z_score_trans = (row['transaction_count'] - baseline_mean_trans) / baseline_std_trans if baseline_std_trans > 0 else 0
        
        # Flag if z-score > 2
        if abs(z_score_trans) > 2:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score_trans):.2f} standard deviations from baseline mean of {baseline_mean_trans:.0f}',
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
                    'baseline_mean_transactions': {
                        'value': round(baseline_mean_trans, 0),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-04',
                        'period_end': '2026-06-01'
                    },
                    'z_score': {
                        'value': round(z_score_trans, 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_transactions),
                'coverage_notes': [f'Baseline calculated from {len(baseline_transactions)} daily observations across 4 weeks prior to analysis period'],
                'assumptions': ['Normal distribution of daily transaction counts', 'Z-score threshold of 2.0 standard deviations'],
                'confidence': 0.85
            })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: x['confidence'], reverse=True)

# Keep only top 3 findings
findings = findings[:3]

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
