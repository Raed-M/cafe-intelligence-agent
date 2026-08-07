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

# Define periods
analysis_start = datetime.fromisoformat("2026-06-08T00:00:00+03:00").replace(tzinfo=None)
analysis_end = datetime.fromisoformat("2026-06-15T00:00:00+03:00").replace(tzinfo=None)

baseline_periods = [
    (datetime.fromisoformat("2026-06-01T00:00:00+03:00").replace(tzinfo=None),
     datetime.fromisoformat("2026-06-08T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-05-25T00:00:00+03:00").replace(tzinfo=None),
     datetime.fromisoformat("2026-06-01T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-05-18T00:00:00+03:00").replace(tzinfo=None),
     datetime.fromisoformat("2026-05-25T00:00:00+03:00").replace(tzinfo=None)),
    (datetime.fromisoformat("2026-05-11T00:00:00+03:00").replace(tzinfo=None),
     datetime.fromisoformat("2026-05-18T00:00:00+03:00").replace(tzinfo=None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue
# ============================================================================

# Convert timestamp to datetime
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])
pos_df['date'] = pos_df['timestamp_dt'].dt.date

# Calculate daily revenue (net of refunds)
daily_revenue = pos_df.groupby('date').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
daily_revenue.columns = ['date', 'revenue_sar', 'transaction_count']
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

# Split into analysis and baseline periods
analysis_daily = daily_revenue[
    (daily_revenue['date'] >= analysis_start.date()) & 
    (daily_revenue['date'] < analysis_end.date())
]

baseline_daily = daily_revenue[
    (daily_revenue['date'] >= baseline_periods[0][0].date()) & 
    (daily_revenue['date'] < baseline_periods[0][1].date())
]

if len(baseline_daily) > 2 and baseline_daily['revenue_sar'].std() > 0:
    baseline_mean = baseline_daily['revenue_sar'].mean()
    baseline_std = baseline_daily['revenue_sar'].std()
    
    # Find anomalies in analysis period
    for idx, row in analysis_daily.iterrows():
        z_score = (row['revenue_sar'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2.0:  # 2-sigma threshold
            findings.append({
                'title': 'Unusual Daily Revenue',
                'claim': f"Daily revenue on {row['date'].date()} was {row['revenue_sar']:.2f} SAR, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
                'finding_type': 'revenue_anomaly',
                'metrics': {
                    'observed_daily_revenue': {
                        'value': round(row['revenue_sar'], 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': row['date'].isoformat()
                    },
                    'baseline_mean_daily_revenue': {
                        'value': round(baseline_mean, 2),
                        'unit': 'SAR',
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
                        'period_start': row['date'].isoformat(),
                        'period_end': row['date'].isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_daily),
                'coverage_notes': [
                    f"Baseline period: {baseline_periods[0][0].date()} to {baseline_periods[0][1].date()}",
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline sample size: {len(baseline_daily)} days"
                ],
                'assumptions': [
                    'Daily revenue follows normal distribution',
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Refunds included in net revenue calculation'
                ],
                'confidence': 0.95
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic (Door Count)
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Split into analysis and baseline
analysis_traffic = traffic_clean[
    (traffic_clean['date'] >= analysis_start.date()) & 
    (traffic_clean['date'] < analysis_end.date())
]

baseline_traffic = traffic_clean[
    (traffic_clean['date'] >= baseline_periods[0][0].date()) & 
    (traffic_clean['date'] < baseline_periods[0][1].date())
]

if len(baseline_traffic) > 10 and baseline_traffic['door_count'].std() > 0:
    baseline_mean_traffic = baseline_traffic['door_count'].mean()
    baseline_std_traffic = baseline_traffic['door_count'].std()
    
    # Find anomalies
    for idx, row in analysis_traffic.iterrows():
        z_score = (row['door_count'] - baseline_mean_traffic) / baseline_std_traffic if baseline_std_traffic > 0 else 0
        if abs(z_score) > 2.0:
            findings.append({
                'title': 'Unusual Hourly Traffic',
                'claim': f"Door count on {row['date'].date()} hour {int(row['hour'])} was {int(row['door_count'])} visitors, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.",
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_hourly_door_count': {
                        'value': int(row['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': f"{row['date'].date()}T{int(row['hour']):02d}:00:00",
                        'period_end': f"{row['date'].date()}T{int(row['hour']):02d}:59:59"
                    },
                    'baseline_mean_hourly_door_count': {
                        'value': round(baseline_mean_traffic, 1),
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
                        'period_start': f"{row['date'].date()}T{int(row['hour']):02d}:00:00",
                        'period_end': f"{row['date'].date()}T{int(row['hour']):02d}:59:59"
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic),
                'coverage_notes': [
                    f"Baseline period: {baseline_periods[0][0].date()} to {baseline_periods[0][1].date()}",
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Dead sensor days excluded",
                    f"Baseline sample size: {len(baseline_traffic)} hourly observations"
                ],
                'assumptions': [
                    'Hourly door count follows normal distribution',
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Dead sensor intervals excluded'
                ],
                'confidence': 0.95
            })

# ============================================================================
# ANOMALY 3: Daily Item Volume (Units Sold)
# ============================================================================

# Calculate daily item volume
pos_df['quantity_numeric'] = pd.to_numeric(pos_df['quantity'], errors='coerce')
daily_volume = pos_df.groupby('date').agg({
    'quantity_numeric': 'sum'
}).reset_index()
daily_volume.columns = ['date', 'units_sold']
daily_volume['date'] = pd.to_datetime(daily_volume['date'])

# Split into analysis and baseline
analysis_volume = daily_volume[
    (daily_volume['date'] >= analysis_start.date()) & 
    (daily_volume['date'] < analysis_end.date())
]

baseline_volume = daily_volume[
    (daily_volume['date'] >= baseline_periods[0][0].date()) & 
    (daily_volume['date'] < baseline_periods[0][1].date())
]

if len(baseline_volume) > 2 and baseline_volume['units_sold'].std() > 0:
    baseline_mean_volume = baseline_volume['units_sold'].mean()
    baseline_std_volume = baseline_volume['units_sold'].std()
    
    # Find anomalies
    for idx, row in analysis_volume.iterrows():
        z_score = (row['units_sold'] - baseline_mean_volume) / baseline_std_volume if baseline_std_volume > 0 else 0
        if abs(z_score) > 2.0:
            findings.append({
                'title': 'Unusual Daily Item Volume',
                'claim': f"Daily item volume on {row['date'].date()} was {row['units_sold']:.0f} units, {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean_volume:.1f} units.",
                'finding_type': 'volume_anomaly',
                'metrics': {
                    'observed_daily_units_sold': {
                        'value': round(row['units_sold'], 1),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': row['date'].isoformat(),
                        'period_end': row['date'].isoformat()
                    },
                    'baseline_mean_daily_units_sold': {
                        'value': round(baseline_mean_volume, 1),
                        'unit': 'units',
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
                        'period_start': row['date'].isoformat(),
                        'period_end': row['date'].isoformat()
                    }
                },
                'source_names': ['pos'],
                'sample_size': len(baseline_volume),
                'coverage_notes': [
                    f"Baseline period: {baseline_periods[0][0].date()} to {baseline_periods[0][1].date()}",
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline sample size: {len(baseline_volume)} days"
                ],
                'assumptions': [
                    'Daily item volume follows normal distribution',
                    'Z-score threshold of 2.0 (95% confidence)',
                    'Refunds included in net volume'
                ],
                'confidence': 0.95
            })

# Sort by z-score magnitude and keep top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)[:3]

# Build output
result = {
    'status': 'success' if findings_sorted else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)