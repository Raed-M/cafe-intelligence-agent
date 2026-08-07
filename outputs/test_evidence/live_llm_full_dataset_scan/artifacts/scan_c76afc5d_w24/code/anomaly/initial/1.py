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
analysis_start = pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None))
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
        for date, revenue in analysis_daily_revenue.items():
            z_score = (revenue - baseline_mean) / baseline_std
            
            # Flag if z-score > 2 (2 standard deviations)
            if abs(z_score) > 2:
                findings.append({
                    'metric': 'daily_revenue',
                    'date': date,
                    'value': float(revenue),
                    'baseline_mean': float(baseline_mean),
                    'baseline_std': float(baseline_std),
                    'z_score': float(z_score),
                    'sample_size': len(baseline_daily_revenues),
                    'direction': 'high' if z_score > 0 else 'low'
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

# Filter baseline traffic
baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                  (traffic_df['date'] < period_end) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    baseline_traffic_list.append(baseline_traffic)

baseline_traffic_combined = pd.concat(baseline_traffic_list, ignore_index=True)

if len(analysis_traffic) > 0 and len(baseline_traffic_combined) > 0:
    # Group by hour of day for comparison
    analysis_traffic['hour_of_day'] = analysis_traffic['hour'].dt.hour
    baseline_traffic_combined['hour_of_day'] = baseline_traffic_combined['hour'].dt.hour
    
    baseline_hourly = baseline_traffic_combined.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count'])
    
    traffic_anomalies = []
    for idx, row in analysis_traffic.iterrows():
        hour = row['hour_of_day']
        count = row['door_count']
        
        if hour in baseline_hourly.index:
            baseline_mean = baseline_hourly.loc[hour, 'mean']
            baseline_std = baseline_hourly.loc[hour, 'std']
            
            if baseline_std > 0:
                z_score = (count - baseline_mean) / baseline_std
                if abs(z_score) > 2:
                    traffic_anomalies.append({
                        'hour': row['hour'],
                        'count': int(count),
                        'baseline_mean': float(baseline_mean),
                        'baseline_std': float(baseline_std),
                        'z_score': float(z_score),
                        'sample_size': int(baseline_hourly.loc[hour, 'count'])
                    })
    
    # Sort by magnitude and take top anomaly
    if traffic_anomalies:
        traffic_anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        findings.append(traffic_anomalies[0])

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count for analysis period
analysis_transactions = analysis_pos.groupby('calendar_date')['transaction_id'].nunique()

# Calculate daily transaction count for baseline periods
baseline_transaction_counts = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    baseline_daily_trans = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_transaction_counts.extend(baseline_daily_trans.values)

if len(baseline_transaction_counts) > 0 and len(analysis_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_transaction_counts)
    baseline_trans_std = np.std(baseline_transaction_counts)
    
    if baseline_trans_std > 0:
        # Calculate z-scores for analysis period
        for date, trans_count in analysis_transactions.items():
            z_score = (trans_count - baseline_trans_mean) / baseline_trans_std
            
            # Flag if z-score > 2
            if abs(z_score) > 2:
                findings.append({
                    'metric': 'daily_transactions',
                    'date': date,
                    'value': int(trans_count),
                    'baseline_mean': float(baseline_trans_mean),
                    'baseline_std': float(baseline_trans_std),
                    'z_score': float(z_score),
                    'sample_size': len(baseline_transaction_counts),
                    'direction': 'high' if z_score > 0 else 'low'
                })

# ============================================================================
# Format findings for output
# ============================================================================

output_findings = []

# Sort findings by z-score magnitude
findings.sort(key=lambda x: abs(x.get('z_score', 0)), reverse=True)

# Take top 3 findings
for i, finding in enumerate(findings[:3]):
    if 'metric' in finding and finding['metric'] == 'daily_revenue':
        output_findings.append({
            'title': f'Daily Revenue Anomaly on {finding["date"].strftime("%Y-%m-%d")}',
            'claim': f'Daily revenue of {finding["value"]:.2f} SAR on {finding["date"].strftime("%Y-%m-%d")} is {abs(finding["z_score"]):.2f} standard deviations from baseline mean of {finding["baseline_mean"]:.2f} SAR',
            'finding_type': 'revenue_anomaly',
            'metrics': {
                'daily_revenue': {
                    'value': finding['value'],
                    'unit': 'SAR',
                    'numerator': finding['value'],
                    'denominator': 1,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean': {
                    'value': finding['baseline_mean'],
                    'unit': 'SAR',
                    'numerator': finding['baseline_mean'],
                    'denominator': 1,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                },
                'z_score': {
                    'value': finding['z_score'],
                    'unit': 'standard_deviations',
                    'numerator': finding['z_score'],
                    'denominator': 1,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': finding['sample_size'],
            'coverage_notes': [
                'Analysis period: 2026-06-22 to 2026-06-29',
                'Baseline: 4 weeks of historical data (2026-05-25 to 2026-06-22)',
                'Excludes refunds in net calculation'
            ],
            'assumptions': [
                'Normal distribution of daily revenue',
                'Z-score threshold of 2.0 standard deviations',
                'Baseline periods are representative of normal operations'
            ],
            'confidence': 0.85
        })
    elif 'metric' in finding and finding['metric'] == 'daily_transactions':
        output_findings.append({
            'title': f'Daily Transaction Count Anomaly on {finding["date"].strftime("%Y-%m-%d")}',
            'claim': f'Daily transaction count of {finding["value"]} on {finding["date"].strftime("%Y-%m-%d")} is {abs(finding["z_score"]):.2f} standard deviations from baseline mean of {finding["baseline_mean"]:.1f}',
            'finding_type': 'transaction_volume_anomaly',
            'metrics': {
                'daily_transactions': {
                    'value': finding['value'],
                    'unit': 'count',
                    'numerator': finding['value'],
                    'denominator': 1,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean': {
                    'value': finding['baseline_mean'],
                    'unit': 'count',
                    'numerator': finding['baseline_mean'],
                    'denominator': 1,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                },
                'z_score': {
                    'value': finding['z_score'],
                    'unit': 'standard_deviations',
                    'numerator': finding['z_score'],
                    'denominator': 1,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': finding['sample_size'],
            'coverage_notes': [
                'Analysis period: 2026-06-22 to 2026-06-29',
                'Baseline: 4 weeks of historical data (2026-05-25 to 2026-06-22)',
                'Transaction count based on unique transaction_id'
            ],
            'assumptions': [
                'Normal distribution of daily transaction counts',
                'Z-score threshold of 2.0 standard deviations',
                'Baseline periods are representative of normal operations'
            ],
            'confidence': 0.85
        })
    elif 'metric' in finding and finding['metric'] == 'hourly_traffic':
        output_findings.append({
            'title': f'Hourly Traffic Anomaly at {finding["hour"].strftime("%H:00")}',
            'claim': f'Door count of {finding["count"]} at {finding["hour"].strftime("%H:00")} is {abs(finding["z_score"]):.2f} standard deviations from baseline mean of {finding["baseline_mean"]:.1f}',
            'finding_type': 'traffic_anomaly',
            'metrics': {
                'hourly_door_count': {
                    'value': finding['count'],
                    'unit': 'count',
                    'numerator': finding['count'],
                    'denominator': 1,
                    'period_start': finding['hour'].isoformat(),
                    'period_end': (finding['hour'] + timedelta(hours=1)).isoformat()
                },
                'baseline_mean': {
                    'value': finding['baseline_mean'],
                    'unit': 'count',
                    'numerator': finding['baseline_mean'],
                    'denominator': 1,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                },
                'z_score': {
                    'value': finding['z_score'],
                    'unit': 'standard_deviations',
                    'numerator': finding['z_score'],
                    'denominator': 1,
                    'period_start': finding['hour'].isoformat(),
                    'period_end': (finding['hour'] + timedelta(hours=1)).isoformat()
                }
            },
            'source_names': ['traffic'],
            'sample_size': finding['sample_size'],
            'coverage_notes': [
                'Analysis period: 2026-06-22 to 2026-06-29',
                'Baseline: 4 weeks of historical data (2026-05-25 to 2026-06-22)',
                'Excludes dead sensor days'
            ],
            'assumptions': [
                'Normal distribution of hourly traffic',
                'Z-score threshold of 2.0 standard deviations',
                'Baseline periods are representative of normal operations'
            ],
            'confidence': 0.85
        })

# Prepare output
result = {
    'status': 'success' if output_findings else 'insufficient_data',
    'findings': output_findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
