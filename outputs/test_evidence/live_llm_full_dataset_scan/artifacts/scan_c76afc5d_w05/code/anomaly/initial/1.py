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
analysis_start = pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None))
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
                        'period_start': '2026-01-12T00:00:00',
                        'period_end': '2026-02-09T00:00:00'
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
                    f'Baseline: 4 weeks of historical data (2026-01-12 to 2026-02-09)',
                    f'Baseline sample size: {len(baseline_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks of historical data',
                    'Revenue includes all transactions, refunds netted in line_total_sar'
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

# Calculate hourly traffic for analysis period
hourly_traffic_analysis = analysis_traffic.groupby('hour')['door_count'].agg(['sum', 'count']).reset_index()
hourly_traffic_analysis.columns = ['hour', 'total_count', 'days']
hourly_traffic_analysis['avg_hourly'] = hourly_traffic_analysis['total_count'] / hourly_traffic_analysis['days']

# Calculate baseline hourly traffic
baseline_traffic_data = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                  (traffic_df['date'] < period_end) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    if len(baseline_traffic) > 0:
        baseline_traffic_data.append(baseline_traffic)

if len(baseline_traffic_data) > 0:
    combined_baseline = pd.concat(baseline_traffic_data, ignore_index=True)
    baseline_hourly = combined_baseline.groupby('hour')['door_count'].agg(['sum', 'count']).reset_index()
    baseline_hourly.columns = ['hour', 'total_count', 'days']
    baseline_hourly['avg_hourly'] = baseline_hourly['total_count'] / baseline_hourly['days']
    
    # Find anomalies
    for idx, row in hourly_traffic_analysis.iterrows():
        hour_val = row['hour']
        baseline_row = baseline_hourly[baseline_hourly['hour'] == hour_val]
        
        if len(baseline_row) > 0:
            baseline_avg = baseline_row.iloc[0]['avg_hourly']
            baseline_std = combined_baseline[combined_baseline['hour'] == hour_val]['door_count'].std()
            
            if baseline_std > 0 and baseline_avg > 0:
                z_score = (row['avg_hourly'] - baseline_avg) / baseline_std
                
                if abs(z_score) > 2.5:  # Higher threshold for hourly data
                    findings.append({
                        'title': f'Hourly Traffic Anomaly at {hour_val.strftime("%H:00")}',
                        'claim': f'Average hourly traffic of {row["avg_hourly"]:.1f} door counts at {hour_val.strftime("%H:00")} deviates {abs(z_score):.2f} standard deviations from baseline average of {baseline_avg:.1f}',
                        'finding_type': 'traffic_anomaly',
                        'metrics': {
                            'observed_avg_hourly_traffic': {
                                'value': round(row['avg_hourly'], 1),
                                'unit': 'door_counts',
                                'numerator': round(row['total_count'], 0),
                                'denominator': int(row['days']),
                                'period_start': analysis_start.isoformat(),
                                'period_end': analysis_end.isoformat()
                            },
                            'baseline_avg_hourly_traffic': {
                                'value': round(baseline_avg, 1),
                                'unit': 'door_counts',
                                'numerator': round(baseline_row.iloc[0]['total_count'], 0),
                                'denominator': int(baseline_row.iloc[0]['days']),
                                'period_start': '2026-01-12T00:00:00',
                                'period_end': '2026-02-09T00:00:00'
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
                        'sample_size': int(row['days']),
                        'coverage_notes': [
                            f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                            f'Baseline: 4 weeks of historical data (2026-01-12 to 2026-02-09)',
                            'Dead sensor days excluded from analysis'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.5 standard deviations used for hourly data',
                            'Baseline calculated from 4 complete weeks of historical data',
                            'Only non-dead-sensor days included'
                        ],
                        'confidence': 0.80
                    })

# ============================================================================
# ANOMALY 3: Daily Item Volume Analysis
# ============================================================================

# Calculate daily item volume for analysis period
daily_volume_analysis = analysis_pos.groupby('calendar_date')['quantity'].sum().reset_index()
daily_volume_analysis.columns = ['date', 'total_quantity']

# Calculate baseline daily volume
baseline_volumes = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_vol = baseline_pos.groupby('calendar_date')['quantity'].sum()
    baseline_volumes.extend(daily_vol.values)

if len(baseline_volumes) > 0 and np.std(baseline_volumes) > 0:
    baseline_vol_mean = np.mean(baseline_volumes)
    baseline_vol_std = np.std(baseline_volumes)
    
    # Check for anomalies
    for idx, row in daily_volume_analysis.iterrows():
        z_score = (row['total_quantity'] - baseline_vol_mean) / baseline_vol_std if baseline_vol_std > 0 else 0
        
        if abs(z_score) > 2.0:
            findings.append({
                'title': f'Daily Item Volume Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily item volume of {row["total_quantity"]:.0f} units on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_vol_mean:.0f} units',
                'finding_type': 'volume_anomaly',
                'metrics': {
                    'observed_daily_volume': {
                        'value': round(row['total_quantity'], 0),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': (row['date'] + timedelta(days=1)).isoformat()
                    },
                    'baseline_mean_daily_volume': {
                        'value': round(baseline_vol_mean, 0),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-01-12T00:00:00',
                        'period_end': '2026-02-09T00:00:00'
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
                'sample_size': len(baseline_volumes),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks of historical data (2026-01-12 to 2026-02-09)',
                    f'Baseline sample size: {len(baseline_volumes)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks of historical data',
                    'Volume includes all items sold, refunds netted in quantity'
                ],
                'confidence': 0.85
            })

# Sort findings by z-score magnitude and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
result = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
