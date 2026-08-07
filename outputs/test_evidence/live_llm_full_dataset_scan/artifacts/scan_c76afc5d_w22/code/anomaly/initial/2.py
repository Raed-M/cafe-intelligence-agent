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
analysis_start = pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-11T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None))
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
                        'period_start': '2026-05-11',
                        'period_end': '2026-06-08'
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
                    f'Baseline calculated from {len(baseline_revenues)} daily observations across 4 weeks (2026-05-11 to 2026-06-08)',
                    f'Analysis period: 2026-06-08 to 2026-06-15',
                    f'Transactions on anomaly date: {int(row["transactions"])}'
                ],
                'assumptions': [
                    'Normal distribution of daily revenues',
                    'Z-score threshold of 2.0 standard deviations',
                    'No structural breaks in business operations',
                    'Refunds included in net revenue calculation'
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
    baseline_traffic_values = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)]
        baseline_traffic_values.extend(baseline_traffic['door_count'].values)
    
    if len(baseline_traffic_values) > 10 and np.std(baseline_traffic_values) > 0:
        baseline_traffic_mean = np.mean(baseline_traffic_values)
        baseline_traffic_std = np.std(baseline_traffic_values)
        
        # Check for anomalies
        for idx, row in analysis_traffic.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std
            
            if abs(z_score) > 2.5:  # Higher threshold for hourly data
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date"].strftime("%Y-%m-%d %H:00")}',
                    'claim': f'Hourly door count of {int(row["door_count"])} on {row["date"].strftime("%Y-%m-%d %H:00")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.0f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean_hourly_door_count': {
                            'value': round(baseline_traffic_mean, 0),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-05-11',
                            'period_end': '2026-06-08'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic_values),
                    'coverage_notes': [
                        f'Baseline calculated from {len(baseline_traffic_values)} hourly observations',
                        'Excluded dead sensor days',
                        f'Analysis period: 2026-06-08 to 2026-06-15'
                    ],
                    'assumptions': [
                        'Normal distribution of hourly traffic',
                        'Z-score threshold of 2.5 standard deviations for hourly data',
                        'Sensor reliability consistent across periods'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

daily_transactions = analysis_pos.groupby('calendar_date').agg({
    'transaction_id': 'nunique'
}).reset_index()
daily_transactions.columns = ['date', 'transaction_count']

# Calculate baseline transaction counts
baseline_transactions = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_trans = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_transactions.extend(daily_trans.values)

if len(baseline_transactions) > 0 and np.std(baseline_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_transactions)
    baseline_trans_std = np.std(baseline_transactions)
    
    # Check for anomalies
    for idx, row in daily_transactions.iterrows():
        z_score = (row['transaction_count'] - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        
        if abs(z_score) > 2.0:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily transaction count of {int(row["transaction_count"])} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.0f}',
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
                        'value': round(baseline_trans_mean, 0),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-05-11',
                        'period_end': '2026-06-08'
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
                    f'Baseline calculated from {len(baseline_transactions)} daily observations across 4 weeks',
                    f'Analysis period: 2026-06-08 to 2026-06-15'
                ],
                'assumptions': [
                    'Normal distribution of daily transaction counts',
                    'Z-score threshold of 2.0 standard deviations',
                    'Transaction_id uniqueness preserved across periods'
                ],
                'confidence': 0.85
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
