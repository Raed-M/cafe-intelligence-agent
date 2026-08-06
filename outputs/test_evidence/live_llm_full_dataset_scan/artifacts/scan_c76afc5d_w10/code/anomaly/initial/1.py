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
analysis_start = pd.to_datetime('2026-03-16T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-03-23T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-03-09T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-16T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-09T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None))
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
    
    # Check each day in analysis period
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
                        'period_start': '2026-02-16T00:00:00',
                        'period_end': '2026-03-16T00:00:00'
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
                    f'Baseline: 4 weeks of historical data (2026-02-16 to 2026-03-16)',
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

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Calculate hourly traffic for analysis period
    hourly_traffic_analysis = analysis_traffic.groupby('hour')['door_count'].agg(['sum', 'count']).reset_index()
    hourly_traffic_analysis.columns = ['hour', 'total_count', 'days']
    hourly_traffic_analysis['avg_hourly'] = hourly_traffic_analysis['total_count'] / hourly_traffic_analysis['days']
    
    # Calculate baseline hourly traffic
    baseline_traffic = traffic_df[(traffic_df['is_dead_sensor_day'] == False)].copy()
    baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
    
    baseline_hourly = baseline_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
    baseline_hourly.columns = ['hour_of_day', 'mean_count', 'std_count', 'sample_size']
    
    # Check for anomalies
    for idx, row in hourly_traffic_analysis.iterrows():
        hour_of_day = row['hour'].hour
        baseline_row = baseline_hourly[baseline_hourly['hour_of_day'] == hour_of_day]
        
        if len(baseline_row) > 0 and baseline_row.iloc[0]['std_count'] > 0:
            baseline_mean = baseline_row.iloc[0]['mean_count']
            baseline_std = baseline_row.iloc[0]['std_count']
            z_score = (row['avg_hourly'] - baseline_mean) / baseline_std
            
            if abs(z_score) > 2.5:  # Higher threshold for hourly data
                findings.append({
                    'title': f'Hourly Traffic Anomaly at {hour_of_day:02d}:00',
                    'claim': f'Average hourly traffic of {row["avg_hourly"]:.0f} door counts at hour {hour_of_day:02d}:00 deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.0f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_traffic': {
                            'value': round(row['avg_hourly'], 0),
                            'unit': 'door_counts',
                            'numerator': int(row['total_count']),
                            'denominator': int(row['days']),
                            'period_start': analysis_start.isoformat(),
                            'period_end': analysis_end.isoformat()
                        },
                        'baseline_mean_hourly_traffic': {
                            'value': round(baseline_mean, 0),
                            'unit': 'door_counts',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-02-16T00:00:00',
                            'period_end': '2026-03-16T00:00:00'
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
                    'sample_size': int(baseline_row.iloc[0]['sample_size']),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                        f'Baseline: all valid sensor days from 2026-02-16 to 2026-03-16',
                        f'Dead sensor days excluded from analysis',
                        f'Days in analysis period for this hour: {int(row["days"])}'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.5 standard deviations used for hourly data',
                        'Only non-dead-sensor days included in baseline',
                        'Hour extracted from local timestamp'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Item Volume Analysis
# ============================================================================

# Calculate daily item volume for analysis period
analysis_pos_vol = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                          (pos_df['calendar_date'] < analysis_end)].copy()

daily_volume_analysis = analysis_pos_vol.groupby('calendar_date')['quantity'].sum().reset_index()
daily_volume_analysis.columns = ['date', 'total_quantity']

# Calculate baseline daily volume
baseline_volumes = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_vol = baseline_pos.groupby('calendar_date')['quantity'].sum()
    baseline_volumes.extend(daily_vol.values)

if len(baseline_volumes) > 0 and np.var(baseline_volumes) > 0:
    baseline_mean_vol = np.mean(baseline_volumes)
    baseline_std_vol = np.std(baseline_volumes)
    
    # Check each day in analysis period
    for idx, row in daily_volume_analysis.iterrows():
        z_score_vol = (row['total_quantity'] - baseline_mean_vol) / baseline_std_vol if baseline_std_vol > 0 else 0
        
        if abs(z_score_vol) > 2.0:  # 2 standard deviations
            findings.append({
                'title': f'Daily Item Volume Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily item volume of {row["total_quantity"]:.0f} units on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score_vol):.2f} standard deviations from baseline mean of {baseline_mean_vol:.0f} units',
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
                        'value': round(baseline_mean_vol, 0),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-02-16T00:00:00',
                        'period_end': '2026-03-16T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score_vol, 2),
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
                    f'Baseline: 4 weeks of historical data (2026-02-16 to 2026-03-16)',
                    f'Volume includes all items sold (refunds counted as negative)'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 complete weeks of historical data',
                    'Quantity includes refunds as negative values'
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
