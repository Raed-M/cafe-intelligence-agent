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

# Parse periods
analysis_start = datetime.fromisoformat("2026-06-22T00:00:00+03:00").replace(tzinfo=None)
analysis_end = datetime.fromisoformat("2026-06-29T00:00:00+03:00").replace(tzinfo=None)

baseline_periods = [
    (datetime.fromisoformat("2026-06-15T00:00:00+03:00").replace(tzinfo=None), 
     datetime.fromisoformat("2026-06-22T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-06-08T00:00:00+03:00").replace(tzinfo=None), 
     datetime.fromisoformat("2026-06-15T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-06-01T00:00:00+03:00").replace(tzinfo=None), 
     datetime.fromisoformat("2026-06-08T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-05-25T00:00:00+03:00").replace(tzinfo=None), 
     datetime.fromisoformat("2026-06-01T00:00:00+03:00").replace(tzinfo=None))
]

baseline_start_date = datetime.fromisoformat("2026-05-25T00:00:00+03:00").replace(tzinfo=None)
baseline_end_date = datetime.fromisoformat("2026-06-22T00:00:00+03:00").replace(tzinfo=None)

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue
# ============================================================================

# Convert timestamp to datetime
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])
pos_df['date'] = pos_df['timestamp_dt'].dt.normalize()

# Calculate daily revenue (net of refunds)
daily_revenue = pos_df.groupby('date').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
daily_revenue.columns = ['date', 'revenue_sar', 'transaction_count']

# Split into analysis and baseline periods
analysis_start_dt = pd.Timestamp(analysis_start)
analysis_end_dt = pd.Timestamp(analysis_end)
baseline_start_dt = pd.Timestamp(baseline_start_date)
baseline_end_dt = pd.Timestamp(baseline_end_date)

analysis_revenue = daily_revenue[
    (daily_revenue['date'] >= analysis_start_dt) & 
    (daily_revenue['date'] < analysis_end_dt)
]

baseline_revenue_list = []
for period_start, period_end in baseline_periods:
    period_start_dt = pd.Timestamp(period_start)
    period_end_dt = pd.Timestamp(period_end)
    period_data = daily_revenue[
        (daily_revenue['date'] >= period_start_dt) & 
        (daily_revenue['date'] < period_end_dt)
    ]
    baseline_revenue_list.extend(period_data['revenue_sar'].values)

if len(baseline_revenue_list) >= 5 and len(analysis_revenue) > 0:
    baseline_mean = np.mean(baseline_revenue_list)
    baseline_std = np.std(baseline_revenue_list)
    
    if baseline_std > 0:
        # Find the most anomalous day in analysis period
        analysis_revenue = analysis_revenue.copy()
        analysis_revenue['z_score'] = (analysis_revenue['revenue_sar'] - baseline_mean) / baseline_std
        max_anomaly_idx = analysis_revenue['z_score'].abs().idxmax()
        max_anomaly = analysis_revenue.loc[max_anomaly_idx]
        
        if abs(max_anomaly['z_score']) > 2.0:
            anomaly_date = max_anomaly['date']
            anomaly_date_end = anomaly_date + timedelta(days=1)
            
            findings.append({
                'title': 'Unusual Daily Revenue',
                'claim': f"Daily revenue on {anomaly_date.date()} was {abs(max_anomaly['z_score']):.2f} standard deviations from baseline mean, suggesting anomalous transaction volume or value.",
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'observed_daily_revenue': {
                        'value': round(float(max_anomaly['revenue_sar']), 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date.isoformat(),
                        'period_end': anomaly_date_end.isoformat()
                    },
                    'baseline_mean_daily_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'baseline_std_daily_revenue': {
                        'value': round(baseline_std, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'z_score': {
                        'value': round(float(max_anomaly['z_score']), 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': anomaly_date.isoformat(),
                        'period_end': anomaly_date_end.isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_revenue_list),
                'coverage_notes': [
                    f"Baseline computed from {len(baseline_revenue_list)} daily observations across 4 weeks ({baseline_start_dt.date()} to {baseline_end_dt.date()})",
                    f"Analysis period: {analysis_start_dt.date()} to {analysis_end_dt.date()} ({len(analysis_revenue)} days)"
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Refunds included in net revenue calculation per line_total_sar',
                    'Daily revenue aggregated from all transactions regardless of payment method or channel'
                ],
                'confidence': 0.75
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Split into analysis and baseline
analysis_traffic = traffic_clean[
    (traffic_clean['date'] >= analysis_start_dt) & 
    (traffic_clean['date'] < analysis_end_dt)
]

baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    period_start_dt = pd.Timestamp(period_start)
    period_end_dt = pd.Timestamp(period_end)
    period_data = traffic_clean[
        (traffic_clean['date'] >= period_start_dt) & 
        (traffic_clean['date'] < period_end_dt)
    ]
    baseline_traffic_list.extend(period_data['door_count'].values)

if len(baseline_traffic_list) >= 10 and len(analysis_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic_list)
    baseline_traffic_std = np.std(baseline_traffic_list)
    
    if baseline_traffic_std > 0:
        analysis_traffic = analysis_traffic.copy()
        analysis_traffic['z_score'] = (analysis_traffic['door_count'] - baseline_traffic_mean) / baseline_traffic_std
        max_traffic_anomaly_idx = analysis_traffic['z_score'].abs().idxmax()
        max_traffic_anomaly = analysis_traffic.loc[max_traffic_anomaly_idx]
        
        if abs(max_traffic_anomaly['z_score']) > 2.0:
            hour_start = max_traffic_anomaly['hour']
            hour_end = hour_start + timedelta(hours=1)
            
            findings.append({
                'title': 'Unusual Hourly Door Traffic',
                'claim': f"Hourly door count on {max_traffic_anomaly['date'].date()} at hour {max_traffic_anomaly['hour'].hour}:00 was {abs(max_traffic_anomaly['z_score']):.2f} standard deviations from baseline, indicating unusual foot traffic.",
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_hourly_door_count': {
                        'value': int(max_traffic_anomaly['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': hour_start.isoformat(),
                        'period_end': hour_end.isoformat()
                    },
                    'baseline_mean_hourly_door_count': {
                        'value': round(baseline_traffic_mean, 2),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'baseline_std_hourly_door_count': {
                        'value': round(baseline_traffic_std, 2),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'z_score': {
                        'value': round(float(max_traffic_anomaly['z_score']), 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': hour_start.isoformat(),
                        'period_end': hour_end.isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic_list),
                'coverage_notes': [
                    f"Baseline computed from {len(baseline_traffic_list)} hourly observations across 4 weeks ({baseline_start_dt.date()} to {baseline_end_dt.date()})",
                    "Dead sensor days excluded per is_dead_sensor_day flag",
                    f"Analysis period: {analysis_start_dt.date()} to {analysis_end_dt.date()}"
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Only non-dead-sensor hours included in baseline and analysis'
                ],
                'confidence': 0.70
            })

# ============================================================================
# ANOMALY 3: Daily Item Volume (units sold)
# ============================================================================

inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Map weeks to dates for comparison
weekly_volume = inventory_df.groupby('week_starting').agg({
    'units_sold': 'sum'
}).reset_index()
weekly_volume.columns = ['week_starting', 'total_units_sold']

# Identify which weeks fall into analysis vs baseline
analysis_week = weekly_volume[
    (weekly_volume['week_starting'] >= analysis_start_dt) & 
    (weekly_volume['week_starting'] < analysis_end_dt)
]

baseline_weeks = []
for period_start, period_end in baseline_periods:
    period_start_dt = pd.Timestamp(period_start)
    period_end_dt = pd.Timestamp(period_end)
    period_data = weekly_volume[
        (weekly_volume['week_starting'] >= period_start_dt) & 
        (weekly_volume['week_starting'] < period_end_dt)
    ]
    baseline_weeks.extend(period_data['total_units_sold'].values)

if len(baseline_weeks) >= 3 and len(analysis_week) > 0:
    baseline_volume_mean = np.mean(baseline_weeks)
    baseline_volume_std = np.std(baseline_weeks)
    
    if baseline_volume_std > 0:
        analysis_week = analysis_week.copy()
        analysis_week['z_score'] = (analysis_week['total_units_sold'] - baseline_volume_mean) / baseline_volume_std
        max_volume_anomaly_idx = analysis_week['z_score'].abs().idxmax()
        max_volume_anomaly = analysis_week.loc[max_volume_anomaly_idx]
        
        if abs(max_volume_anomaly['z_score']) > 1.5:
            week_start = max_volume_anomaly['week_starting']
            week_end = week_start + timedelta(days=7)
            
            findings.append({
                'title': 'Unusual Weekly Item Volume',
                'claim': f"Weekly item volume for week starting {week_start.date()} was {abs(max_volume_anomaly['z_score']):.2f} standard deviations from baseline, indicating unusual sales velocity.",
                'finding_type': 'volume_anomaly',
                'metrics': {
                    'observed_weekly_units_sold': {
                        'value': int(max_volume_anomaly['total_units_sold']),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': week_start.isoformat(),
                        'period_end': week_end.isoformat()
                    },
                    'baseline_mean_weekly_units': {
                        'value': round(baseline_volume_mean, 2),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'baseline_std_weekly_units': {
                        'value': round(baseline_volume_std, 2),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_start_dt.isoformat(),
                        'period_end': baseline_end_dt.isoformat()
                    },
                    'z_score': {
                        'value': round(float(max_volume_anomaly['z_score']), 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': week_start.isoformat(),
                        'period_end': week_end.isoformat()
                    }
                },
                'source_names': ['inventory'],
                'sample_size': len(baseline_weeks),
                'coverage_notes': [
                    f"Baseline computed from {len(baseline_weeks)} weekly observations ({baseline_start_dt.date()} to {baseline_end_dt.date()})",
                    f"Analysis period: {analysis_start_dt.date()} to {analysis_end_dt.date()}",
                    "Units sold aggregated across all SKUs per week"
                ],
                'assumptions': [
                    'Z-score threshold of 1.5 standard deviations used (lower threshold due to smaller sample)',
                    'Weekly aggregation used due to inventory data granularity'
                ],
                'confidence': 0.65
            })

# Sort by z-score magnitude and keep top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)[:3]

# Build output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
