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

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Convert timestamp to datetime
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])
pos_df['date'] = pos_df['timestamp_dt'].dt.date

# Calculate daily revenue (excluding refunds)
pos_df['net_line_total'] = pos_df['line_total_sar']
daily_revenue = pos_df[~pos_df['is_refund']].groupby('date')['net_line_total'].sum().reset_index()
daily_revenue.columns = ['date', 'revenue']
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

# Split into analysis and baseline periods
analysis_revenue = daily_revenue[
    (daily_revenue['date'] >= analysis_start) & 
    (daily_revenue['date'] < analysis_end)
]

baseline_revenue_list = []
for period_start, period_end in baseline_periods:
    period_data = daily_revenue[
        (daily_revenue['date'] >= period_start) & 
        (daily_revenue['date'] < period_end)
    ]
    baseline_revenue_list.extend(period_data['revenue'].values)

if len(baseline_revenue_list) >= 5 and len(analysis_revenue) > 0:
    baseline_mean = np.mean(baseline_revenue_list)
    baseline_std = np.std(baseline_revenue_list)
    
    if baseline_std > 0:
        for idx, row in analysis_revenue.iterrows():
            z_score = (row['revenue'] - baseline_mean) / baseline_std
            
            if abs(z_score) > 2.0:  # 2-sigma threshold
                findings.append({
                    'title': 'Daily Revenue Anomaly',
                    'claim': f"Daily revenue on {row['date'].date()} was {abs(z_score):.2f} standard deviations from baseline mean",
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
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
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
                    'sample_size': len(baseline_revenue_list),
                    'coverage_notes': [
                        f"Baseline: {len(baseline_revenue_list)} daily observations across 4 weeks",
                        f"Analysis period: {len(analysis_revenue)} days",
                        "Refunds excluded from revenue calculation"
                    ],
                    'assumptions': [
                        'Normal distribution of daily revenue',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'No structural breaks in baseline period'
                    ],
                    'confidence': 0.95
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour_dt'] = pd.to_datetime(traffic_df['hour'])

# Filter out dead sensor days
traffic_clean = traffic_df[~traffic_df['is_dead_sensor_day']].copy()

# Split into analysis and baseline
analysis_traffic = traffic_clean[
    (traffic_clean['date'] >= analysis_start) & 
    (traffic_clean['date'] < analysis_end)
]

baseline_traffic_list = []
for period_start, period_end in baseline_periods:
    period_data = traffic_clean[
        (traffic_clean['date'] >= period_start) & 
        (traffic_clean['date'] < period_end)
    ]
    baseline_traffic_list.extend(period_data['door_count'].values)

if len(baseline_traffic_list) >= 10 and len(analysis_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic_list)
    baseline_traffic_std = np.std(baseline_traffic_list)
    
    if baseline_traffic_std > 0:
        for idx, row in analysis_traffic.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std
            
            if abs(z_score) > 2.0:
                findings.append({
                    'title': 'Hourly Traffic Anomaly',
                    'claim': f"Hourly door count on {row['date'].date()} hour {row['hour']} was {abs(z_score):.2f} standard deviations from baseline",
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour_dt'].isoformat(),
                            'period_end': (row['hour_dt'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_traffic_mean, 2),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour_dt'].isoformat(),
                            'period_end': (row['hour_dt'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic_list),
                    'coverage_notes': [
                        f"Baseline: {len(baseline_traffic_list)} hourly observations across 4 weeks",
                        f"Analysis period: {len(analysis_traffic)} hourly observations",
                        "Dead sensor days excluded"
                    ],
                    'assumptions': [
                        'Normal distribution of hourly traffic',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'Sensor reliability consistent across periods'
                    ],
                    'confidence': 0.95
                })

# ============================================================================
# ANOMALY 3: Daily Item Volume Analysis
# ============================================================================

daily_volume = pos_df[~pos_df['is_refund']].groupby('date')['quantity'].sum().reset_index()
daily_volume.columns = ['date', 'volume']
daily_volume['date'] = pd.to_datetime(daily_volume['date'])

analysis_volume = daily_volume[
    (daily_volume['date'] >= analysis_start) & 
    (daily_volume['date'] < analysis_end)
]

baseline_volume_list = []
for period_start, period_end in baseline_periods:
    period_data = daily_volume[
        (daily_volume['date'] >= period_start) & 
        (daily_volume['date'] < period_end)
    ]
    baseline_volume_list.extend(period_data['volume'].values)

if len(baseline_volume_list) >= 5 and len(analysis_volume) > 0:
    baseline_volume_mean = np.mean(baseline_volume_list)
    baseline_volume_std = np.std(baseline_volume_list)
    
    if baseline_volume_std > 0:
        for idx, row in analysis_volume.iterrows():
            z_score = (row['volume'] - baseline_volume_mean) / baseline_volume_std
            
            if abs(z_score) > 2.0:
                findings.append({
                    'title': 'Daily Item Volume Anomaly',
                    'claim': f"Daily item volume on {row['date'].date()} was {abs(z_score):.2f} standard deviations from baseline mean",
                    'finding_type': 'volume_anomaly',
                    'metrics': {
                        'daily_volume': {
                            'value': int(row['volume']),
                            'unit': 'items',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_volume_mean, 2),
                            'unit': 'items',
                            'numerator': None,
                            'denominator': None,
                            'period_start': baseline_periods[0][0].isoformat(),
                            'period_end': baseline_periods[-1][1].isoformat()
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
                    'sample_size': len(baseline_volume_list),
                    'coverage_notes': [
                        f"Baseline: {len(baseline_volume_list)} daily observations across 4 weeks",
                        f"Analysis period: {len(analysis_volume)} days",
                        "Refunds excluded from volume calculation"
                    ],
                    'assumptions': [
                        'Normal distribution of daily item volume',
                        'Z-score threshold of 2.0 (95% confidence)',
                        'No structural breaks in baseline period'
                    ],
                    'confidence': 0.95
                })

# Sort by z-score magnitude and keep top 3
findings_sorted = sorted(
    findings,
    key=lambda x: abs(x['metrics']['z_score']['value']),
    reverse=True
)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
