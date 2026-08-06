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

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define analysis periods
analysis_start = pd.to_datetime("2026-05-11T00:00:00+03:00").tz_localize(None)
analysis_end = pd.to_datetime("2026-05-18T00:00:00+03:00").tz_localize(None)

baseline_periods = [
    (pd.to_datetime("2026-05-04T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-05-11T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-04-27T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-05-04T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-04-20T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-04-27T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-04-13T00:00:00+03:00").tz_localize(None), 
     pd.to_datetime("2026-04-20T00:00:00+03:00").tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                      (pos_df['calendar_date'] < analysis_end)].copy()
analysis_pos['net_revenue'] = analysis_pos['line_total_sar']

daily_revenue_analysis = analysis_pos.groupby('calendar_date')['net_revenue'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate baseline daily revenues
baseline_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    baseline_pos['net_revenue'] = baseline_pos['line_total_sar']
    daily_baseline = baseline_pos.groupby('calendar_date')['net_revenue'].sum().reset_index()
    baseline_revenues.extend(daily_baseline['revenue'].values)

if len(baseline_revenues) > 0 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Calculate z-scores for analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # Flag if z-score > 2 (2 standard deviations)
        if abs(z_score) > 2:
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
                'sample_size': len(baseline_revenues),
                'coverage_notes': [
                    f'Baseline calculated from {len(baseline_revenues)} daily observations across 4 weeks',
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    'Refunds included in net revenue calculation'
                ],
                'assumptions': [
                    'Daily revenue follows normal distribution',
                    'Baseline periods are representative of normal operations',
                    'Z-score threshold of 2.0 indicates statistical significance'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic data for analysis and baseline periods
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

baseline_traffic = []
for period_start, period_end in baseline_periods:
    period_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                (traffic_df['date'] < period_end) &
                                (traffic_df['is_dead_sensor_day'] == False)].copy()
    baseline_traffic.extend(period_traffic['door_count'].values)

if len(baseline_traffic) > 10 and np.std(baseline_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic)
    baseline_traffic_std = np.std(baseline_traffic)
    
    # Calculate z-scores for analysis period
    for idx, row in analysis_traffic.iterrows():
        z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std if baseline_traffic_std > 0 else 0
        
        # Flag if z-score > 2.5 (more stringent for traffic)
        if abs(z_score) > 2.5:
            findings.append({
                'title': f'Hourly Traffic Anomaly on {row["date"].strftime("%Y-%m-%d")} at {int(row["hour"].hour):02d}:00',
                'claim': f'Door count of {int(row["door_count"])} at {row["date"].strftime("%Y-%m-%d")} {int(row["hour"].hour):02d}:00 deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
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
                'sample_size': len(baseline_traffic),
                'coverage_notes': [
                    f'Baseline calculated from {len(baseline_traffic)} hourly observations across 4 weeks',
                    'Excluded hours marked as dead_sensor_day',
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}'
                ],
                'assumptions': [
                    'Hourly traffic follows normal distribution',
                    'Baseline periods represent normal traffic patterns',
                    'Z-score threshold of 2.5 indicates statistical significance for traffic'
                ],
                'confidence': 0.80
            })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction counts
analysis_pos_trans = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                            (pos_df['calendar_date'] < analysis_end)].copy()

daily_transactions_analysis = analysis_pos_trans.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_transactions_analysis.columns = ['date', 'transaction_count']

# Calculate baseline daily transaction counts
baseline_transactions = []
for period_start, period_end in baseline_periods:
    baseline_pos_trans = pos_df[(pos_df['calendar_date'] >= period_start) & 
                                (pos_df['calendar_date'] < period_end)].copy()
    daily_baseline_trans = baseline_pos_trans.groupby('calendar_date')['transaction_id'].nunique().reset_index()
    baseline_transactions.extend(daily_baseline_trans['transaction_id'].values)

if len(baseline_transactions) > 0 and np.std(baseline_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_transactions)
    baseline_trans_std = np.std(baseline_transactions)
    
    # Calculate z-scores for analysis period
    for idx, row in daily_transactions_analysis.iterrows():
        z_score = (row['transaction_count'] - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        
        # Flag if z-score > 2
        if abs(z_score) > 2:
            findings.append({
                'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Transaction count of {int(row["transaction_count"])} on {row["date"].strftime("%Y-%m-%d")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f}',
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
                'sample_size': len(baseline_transactions),
                'coverage_notes': [
                    f'Baseline calculated from {len(baseline_transactions)} daily observations across 4 weeks',
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    'Transaction count based on unique transaction_id values'
                ],
                'assumptions': [
                    'Daily transaction counts follow normal distribution',
                    'Baseline periods are representative of normal operations',
                    'Z-score threshold of 2.0 indicates statistical significance'
                ],
                'confidence': 0.85
            })

# Sort findings by magnitude (z-score) and limit to top 3
findings_sorted = sorted(findings, key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings_sorted) > 0 else 'insufficient_data',
    'findings': findings_sorted
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
