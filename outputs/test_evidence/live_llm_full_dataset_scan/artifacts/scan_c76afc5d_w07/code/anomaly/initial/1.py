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
analysis_start = pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-03-02T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None))
]

findings = []

# ANOMALY 1: Daily Revenue Analysis
print("Analyzing daily revenue...")
pos_analysis = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                       (pos_df['calendar_date'] < analysis_end)].copy()

# Calculate daily revenue (excluding refunds)
daily_revenue = pos_analysis[~pos_analysis['is_refund']].groupby('calendar_date')['line_total_sar'].sum()

# Calculate baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_data = pos_df[(pos_df['calendar_date'] >= period_start) & 
                           (pos_df['calendar_date'] < period_end) &
                           (~pos_df['is_refund'])]
    baseline_daily = baseline_data.groupby('calendar_date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(baseline_daily.values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Calculate z-scores for analysis period
        for date, revenue in daily_revenue.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Unusual Daily Revenue on {date.strftime("%Y-%m-%d")}',
                    'claim': f'Daily revenue of {revenue:.2f} SAR on {date.strftime("%Y-%m-%d")} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': round(revenue, 2),
                            'unit': 'SAR',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
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
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_revenues),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                        f'Baseline: {len(baseline_daily_revenues)} daily observations from 4 weeks prior',
                        'Refunds excluded from revenue calculation'
                    ],
                    'assumptions': [
                        'Normal distribution of daily revenue',
                        'Z-score threshold of 2.0 standard deviations',
                        'Baseline periods are representative of normal operations'
                    ],
                    'confidence': 0.85
                })

# ANOMALY 2: Hourly Traffic Analysis
print("Analyzing hourly traffic...")
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

# Calculate baseline hourly traffic
baseline_hourly_traffic = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                  (traffic_df['date'] < period_end) &
                                  (traffic_df['is_dead_sensor_day'] == False)]
    baseline_hourly_traffic.extend(baseline_traffic['door_count'].values)

if len(baseline_hourly_traffic) > 0 and len(traffic_analysis) > 0:
    baseline_traffic_mean = np.mean(baseline_hourly_traffic)
    baseline_traffic_std = np.std(baseline_hourly_traffic)
    
    if baseline_traffic_std > 0:
        # Find anomalous hours
        traffic_analysis['z_score'] = (traffic_analysis['door_count'] - baseline_traffic_mean) / baseline_traffic_std
        anomalous_traffic = traffic_analysis[abs(traffic_analysis['z_score']) > 2.0].sort_values('z_score', key=abs, ascending=False)
        
        if len(anomalous_traffic) > 0:
            top_anomaly = anomalous_traffic.iloc[0]
            findings.append({
                'title': f'Unusual Hourly Traffic on {top_anomaly["date"].strftime("%Y-%m-%d %H:00")}',
                'claim': f'Hourly door count of {int(top_anomaly["door_count"])} on {top_anomaly["date"].strftime("%Y-%m-%d %H:00")} is {abs(top_anomaly["z_score"]):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.0f}',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'hourly_door_count': {
                        'value': int(top_anomaly['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': top_anomaly['date'].isoformat(),
                        'period_end': (top_anomaly['date'] + timedelta(hours=1)).isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_traffic_mean, 0),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(top_anomaly['z_score'], 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': top_anomaly['date'].isoformat(),
                        'period_end': (top_anomaly['date'] + timedelta(hours=1)).isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_hourly_traffic),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: {len(baseline_hourly_traffic)} hourly observations from 4 weeks prior',
                    'Dead sensor days excluded'
                ],
                'assumptions': [
                    'Normal distribution of hourly traffic',
                    'Z-score threshold of 2.0 standard deviations',
                    'Baseline periods are representative of normal operations'
                ],
                'confidence': 0.80
            })

# ANOMALY 3: Daily Transaction Count Analysis
print("Analyzing daily transaction counts...")
daily_transactions = pos_analysis[~pos_analysis['is_refund']].groupby('calendar_date')['transaction_id'].nunique()

# Calculate baseline daily transactions
baseline_daily_transactions = []
for period_start, period_end in baseline_periods:
    baseline_data = pos_df[(pos_df['calendar_date'] >= period_start) & 
                           (pos_df['calendar_date'] < period_end) &
                           (~pos_df['is_refund'])]
    baseline_daily_trans = baseline_data.groupby('calendar_date')['transaction_id'].nunique()
    baseline_daily_transactions.extend(baseline_daily_trans.values)

if len(baseline_daily_transactions) > 0 and len(daily_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_daily_transactions)
    baseline_trans_std = np.std(baseline_daily_transactions)
    
    if baseline_trans_std > 0:
        # Calculate z-scores for analysis period
        for date, trans_count in daily_transactions.items():
            z_score = (trans_count - baseline_trans_mean) / baseline_trans_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Unusual Daily Transaction Count on {date.strftime("%Y-%m-%d")}',
                    'claim': f'Daily transaction count of {int(trans_count)} on {date.strftime("%Y-%m-%d")} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.0f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'daily_transaction_count': {
                            'value': int(trans_count),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_trans_mean, 0),
                            'unit': 'transactions',
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
                            'period_start': date.isoformat(),
                            'period_end': (date + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_daily_transactions),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                        f'Baseline: {len(baseline_daily_transactions)} daily observations from 4 weeks prior',
                        'Refunds excluded from transaction count'
                    ],
                    'assumptions': [
                        'Normal distribution of daily transaction counts',
                        'Z-score threshold of 2.0 standard deviations',
                        'Baseline periods are representative of normal operations'
                    ],
                    'confidence': 0.85
                })

# Sort findings by magnitude and limit to top 3
findings.sort(key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)
findings = findings[:3]

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Found {len(findings)} anomalies.")
