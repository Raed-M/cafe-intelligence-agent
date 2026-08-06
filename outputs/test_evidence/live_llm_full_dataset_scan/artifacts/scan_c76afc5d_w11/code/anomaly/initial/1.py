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
analysis_start = pd.to_datetime('2026-03-23T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-03-30T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-03-16T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-23T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-09T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-16T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-09T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                      (pos_df['calendar_date'] < analysis_end)].copy()

# Exclude refunds from revenue calculation
analysis_pos['net_revenue'] = analysis_pos['line_total_sar'] * (~analysis_pos['is_refund']).astype(int)

daily_revenue_analysis = analysis_pos.groupby('calendar_date')['net_revenue'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate baseline daily revenue
baseline_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    baseline_pos['net_revenue'] = baseline_pos['line_total_sar'] * (~baseline_pos['is_refund']).astype(int)
    daily_baseline = baseline_pos.groupby('calendar_date')['net_revenue'].sum()
    baseline_revenues.extend(daily_baseline.values)

baseline_revenues = np.array(baseline_revenues)

if len(baseline_revenues) > 0 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Find anomalies using z-score
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = abs((row['revenue'] - baseline_mean) / baseline_std)
        if z_score > 2.0:  # 2 standard deviations
            findings.append({
                'title': f'Daily Revenue Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily revenue of {row["revenue"]:.2f} SAR on {row["date"].strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
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
                        'period_start': '2026-02-23T00:00:00',
                        'period_end': '2026-03-23T00:00:00'
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
                    'Analysis period: 2026-03-23 to 2026-03-30',
                    'Baseline: 4 weeks of historical data (2026-02-23 to 2026-03-23)',
                    'Refunds excluded from revenue calculation'
                ],
                'assumptions': [
                    'Normal distribution of daily revenues',
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Baseline periods are representative of normal operations'
                ],
                'confidence': 0.95
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

# Calculate hourly traffic for analysis period
analysis_traffic['hour_of_day'] = analysis_traffic['hour'].dt.hour

# Get baseline hourly traffic
baseline_traffic_data = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                  (traffic_df['date'] < period_end) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
    baseline_traffic_data.append(baseline_traffic)

if len(baseline_traffic_data) > 0:
    combined_baseline = pd.concat(baseline_traffic_data, ignore_index=True)
    
    # Calculate baseline hourly means
    baseline_hourly = combined_baseline.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
    baseline_hourly = baseline_hourly[baseline_hourly['count'] >= 3]  # At least 3 observations
    
    # Check for anomalies in analysis period
    analysis_hourly = analysis_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
    
    for idx, row in analysis_hourly.iterrows():
        hour = row['hour_of_day']
        baseline_row = baseline_hourly[baseline_hourly['hour_of_day'] == hour]
        
        if len(baseline_row) > 0 and baseline_row.iloc[0]['std'] > 0:
            baseline_mean = baseline_row.iloc[0]['mean']
            baseline_std = baseline_row.iloc[0]['std']
            z_score = abs((row['mean'] - baseline_mean) / baseline_std)
            
            if z_score > 2.0 and row['count'] >= 2:
                findings.append({
                    'title': f'Hourly Traffic Anomaly at {int(hour):02d}:00',
                    'claim': f'Average hourly traffic of {row["mean"]:.0f} visitors at {int(hour):02d}:00 deviates {z_score:.2f} standard deviations from baseline mean of {baseline_mean:.0f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_traffic': {
                            'value': round(row['mean'], 0),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f'2026-03-23T{int(hour):02d}:00:00',
                            'period_end': f'2026-03-30T{int(hour):02d}:00:00'
                        },
                        'baseline_mean_hourly_traffic': {
                            'value': round(baseline_mean, 0),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-02-23T00:00:00',
                            'period_end': '2026-03-23T00:00:00'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f'2026-03-23T{int(hour):02d}:00:00',
                            'period_end': f'2026-03-30T{int(hour):02d}:00:00'
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': int(row['count']),
                    'coverage_notes': [
                        'Analysis period: 2026-03-23 to 2026-03-30',
                        'Baseline: 4 weeks of historical data',
                        'Dead sensor days excluded'
                    ],
                    'assumptions': [
                        'Normal distribution of hourly traffic',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'Baseline periods are representative'
                    ],
                    'confidence': 0.95
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count for analysis period
analysis_pos_txn = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                          (pos_df['calendar_date'] < analysis_end)].copy()

daily_txn_analysis = analysis_pos_txn.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_txn_analysis.columns = ['date', 'transaction_count']

# Calculate baseline daily transaction counts
baseline_txn_counts = []
for period_start, period_end in baseline_periods:
    baseline_pos_txn = pos_df[(pos_df['calendar_date'] >= period_start) & 
                              (pos_df['calendar_date'] < period_end)].copy()
    daily_baseline_txn = baseline_pos_txn.groupby('calendar_date')['transaction_id'].nunique()
    baseline_txn_counts.extend(daily_baseline_txn.values)

baseline_txn_counts = np.array(baseline_txn_counts)

if len(baseline_txn_counts) > 0 and np.std(baseline_txn_counts) > 0:
    baseline_txn_mean = np.mean(baseline_txn_counts)
    baseline_txn_std = np.std(baseline_txn_counts)
    
    # Find anomalies using z-score
    for idx, row in daily_txn_analysis.iterrows():
        z_score = abs((row['transaction_count'] - baseline_txn_mean) / baseline_txn_std)
        if z_score > 2.0:  # 2 standard deviations
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_txn_mean:.0f}',
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
                        'value': round(baseline_txn_mean, 0),
                        'unit': 'transactions',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-02-23T00:00:00',
                        'period_end': '2026-03-23T00:00:00'
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
                'sample_size': len(baseline_txn_counts),
                'coverage_notes': [
                    'Analysis period: 2026-03-23 to 2026-03-30',
                    'Baseline: 4 weeks of historical data (2026-02-23 to 2026-03-23)',
                    'Transaction count based on unique transaction_id'
                ],
                'assumptions': [
                    'Normal distribution of daily transaction counts',
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Baseline periods are representative of normal operations'
                ],
                'confidence': 0.95
            })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: float(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
result = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
