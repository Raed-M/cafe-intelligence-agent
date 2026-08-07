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
analysis_start = pd.to_datetime('2026-07-13T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-07-20T00:00:00+03:00').tz_localize(None)
baseline_periods = [
    (pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-07-13T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_revenue = analysis_pos.groupby('calendar_date')['line_total_sar'].sum().reset_index()
analysis_daily_revenue.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    daily_rev = baseline_pos.groupby('calendar_date')['line_total_sar'].sum().reset_index()
    baseline_daily_revenues.extend(daily_rev['line_total_sar'].values)

if len(baseline_daily_revenues) >= 10 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalous days
        for idx, row in analysis_daily_revenue.iterrows():
            z_score = abs((row['revenue'] - baseline_mean) / baseline_std)
            if z_score > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Unusual Daily Revenue on {row["date"].strftime("%Y-%m-%d")}',
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
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
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
                    'sample_size': len(baseline_daily_revenues),
                    'coverage_notes': [
                        f'Baseline calculated from {len(baseline_daily_revenues)} daily observations across 4 weeks',
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}',
                        'Refunds included in net revenue calculation'
                    ],
                    'assumptions': [
                        'Daily revenue follows approximately normal distribution',
                        'Baseline periods are representative of normal operations',
                        'Z-score threshold of 2.0 (95% confidence) used for anomaly detection'
                    ],
                    'confidence': 0.85
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic data
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
analysis_traffic = analysis_traffic[analysis_traffic['is_dead_sensor_day'] == False]

baseline_traffic_data = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & (traffic_df['date'] < period_end)].copy()
    baseline_traffic = baseline_traffic[baseline_traffic['is_dead_sensor_day'] == False]
    baseline_traffic_data.extend(baseline_traffic['door_count'].values)

if len(baseline_traffic_data) >= 20 and len(analysis_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic_data)
    baseline_traffic_std = np.std(baseline_traffic_data)
    
    if baseline_traffic_std > 0:
        # Find anomalous hours
        for idx, row in analysis_traffic.iterrows():
            z_score = abs((row['door_count'] - baseline_traffic_mean) / baseline_traffic_std)
            if z_score > 2.5:  # Higher threshold for hourly data
                findings.append({
                    'title': f'Unusual Hourly Traffic on {row["date"].strftime("%Y-%m-%d")} at {row["hour"].strftime("%H:00")}',
                    'claim': f'Door count of {int(row["door_count"])} at {row["hour"].strftime("%Y-%m-%d %H:00")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'observed_hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour'].isoformat(),
                            'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean_hourly_door_count': {
                            'value': round(baseline_traffic_mean, 1),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour'].isoformat(),
                            'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic_data),
                    'coverage_notes': [
                        f'Baseline calculated from {len(baseline_traffic_data)} hourly observations across 4 weeks',
                        'Excluded hours marked as dead_sensor_day',
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'
                    ],
                    'assumptions': [
                        'Hourly traffic follows approximately normal distribution',
                        'Baseline periods are representative of normal operations',
                        'Z-score threshold of 2.5 used for hourly anomaly detection'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction counts
analysis_pos_trans = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_trans = analysis_pos_trans.groupby('calendar_date')['transaction_id'].nunique().reset_index()
analysis_daily_trans.columns = ['date', 'transaction_count']

# Calculate baseline transaction counts
baseline_daily_trans = []
for period_start, period_end in baseline_periods:
    baseline_pos_trans = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    daily_trans = baseline_pos_trans.groupby('calendar_date')['transaction_id'].nunique().reset_index()
    baseline_daily_trans.extend(daily_trans['transaction_id'].values)

if len(baseline_daily_trans) >= 10 and len(analysis_daily_trans) > 0:
    baseline_trans_mean = np.mean(baseline_daily_trans)
    baseline_trans_std = np.std(baseline_daily_trans)
    
    if baseline_trans_std > 0:
        # Find anomalous days
        for idx, row in analysis_daily_trans.iterrows():
            z_score = abs((row['transaction_count'] - baseline_trans_mean) / baseline_trans_std)
            if z_score > 2.0:
                findings.append({
                    'title': f'Unusual Daily Transaction Count on {row["date"].strftime("%Y-%m-%d")}',
                    'claim': f'Daily transaction count of {int(row["transaction_count"])} on {row["date"].strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'observed_daily_transaction_count': {
                            'value': int(row['transaction_count']),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean_daily_transaction_count': {
                            'value': round(baseline_trans_mean, 1),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
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
                    'sample_size': len(baseline_daily_trans),
                    'coverage_notes': [
                        f'Baseline calculated from {len(baseline_daily_trans)} daily observations across 4 weeks',
                        'Transaction count based on unique transaction_id values',
                        f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'
                    ],
                    'assumptions': [
                        'Daily transaction counts follow approximately normal distribution',
                        'Baseline periods are representative of normal operations',
                        'Z-score threshold of 2.0 (95% confidence) used for anomaly detection'
                    ],
                    'confidence': 0.85
                })

# Sort findings by z-score magnitude and limit to top 3
findings_with_zscore = []
for finding in findings:
    z_score = finding['metrics'].get('z_score', {}).get('value', 0)
    findings_with_zscore.append((z_score, finding))

findings_with_zscore.sort(key=lambda x: x[0], reverse=True)
top_findings = [f[1] for f in findings_with_zscore[:3]]

# Prepare output
result = {
    'status': 'success' if len(top_findings) > 0 else 'insufficient_data',
    'findings': top_findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
