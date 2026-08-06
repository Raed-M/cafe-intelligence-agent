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

# Parse dates
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define periods
analysis_start = pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None)
baseline_periods = [
    (pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-29T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-22T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2025-12-29T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2025-12-15T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2025-12-22T00:00:00+03:00').tz_localize(None))
]

findings = []

# 1. Daily Revenue Anomaly Detection
print("Analyzing daily revenue...")
pos_df['date'] = pos_df['calendar_date'].dt.date
daily_revenue = pos_df[~pos_df['is_refund']].groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

# Get baseline daily revenues
baseline_revenues = []
for period_start, period_end in baseline_periods:
    period_data = pos_df[(pos_df['calendar_date'] >= period_start) & 
                         (pos_df['calendar_date'] < period_end) & 
                         (~pos_df['is_refund'])]
    period_daily = period_data.groupby('date')['line_total_sar'].sum()
    baseline_revenues.extend(period_daily.values)

if len(baseline_revenues) > 5 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Analyze analysis period
    analysis_daily = daily_revenue[(daily_revenue['date'] >= analysis_start) & 
                                   (daily_revenue['date'] < analysis_end)]
    
    if len(analysis_daily) > 0:
        for idx, row in analysis_daily.iterrows():
            z_score = abs((row['line_total_sar'] - baseline_mean) / baseline_std)
            if z_score > 2.0:  # 2 standard deviations
                findings.append({
                    'title': f'Daily Revenue Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                    'claim': f'Daily revenue of {row["line_total_sar"]:.2f} SAR on {row["date"].strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
                    'finding_type': 'revenue_anomaly',
                    'metrics': {
                        'daily_revenue': {
                            'value': round(row['line_total_sar'], 2),
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
                            'unit': 'standard_deviations',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        }
                    },
                    'source_names': ['pos'],
                    'sample_size': len(baseline_revenues),
                    'coverage_notes': [f'Baseline from {len(baseline_periods)} weeks of historical data', 
                                      f'Analysis period: {analysis_start.date()} to {analysis_end.date()}'],
                    'assumptions': ['Normal distribution of daily revenues', 
                                   'No structural breaks in baseline period',
                                   'Refunds excluded from revenue calculation'],
                    'confidence': 0.85
                })

# 2. Hourly Traffic Anomaly Detection
print("Analyzing hourly traffic...")
traffic_df['date'] = traffic_df['hour'].dt.date
traffic_df['hour_of_day'] = traffic_df['hour'].dt.hour

# Get baseline hourly traffic
baseline_traffic = []
for period_start, period_end in baseline_periods:
    period_traffic = traffic_df[(traffic_df['date'] >= period_start.date()) & 
                                (traffic_df['date'] < period_end.date()) &
                                (traffic_df['is_dead_sensor_day'] == False)]
    baseline_traffic.extend(period_traffic['door_count'].values)

if len(baseline_traffic) > 10 and np.std(baseline_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic)
    baseline_traffic_std = np.std(baseline_traffic)
    
    # Analyze analysis period
    analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start.date()) & 
                                  (traffic_df['date'] < analysis_end.date()) &
                                  (traffic_df['is_dead_sensor_day'] == False)]
    
    if len(analysis_traffic) > 0:
        for idx, row in analysis_traffic.iterrows():
            z_score = abs((row['door_count'] - baseline_traffic_mean) / baseline_traffic_std)
            if z_score > 2.5:  # Higher threshold for traffic
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date"]} at {row["hour_of_day"]:02d}:00',
                    'claim': f'Door count of {row["door_count"]} on {row["date"]} at {row["hour_of_day"]:02d}:00 deviates {z_score:.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['hour'].isoformat(),
                            'period_end': (row['hour'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean': {
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
                    'coverage_notes': [f'Baseline from {len(baseline_periods)} weeks of historical data',
                                      'Dead sensor days excluded from analysis'],
                    'assumptions': ['Normal distribution of hourly traffic',
                                   'No structural breaks in baseline period',
                                   'Sensor reliability consistent across periods'],
                    'confidence': 0.80
                })

# 3. Daily Transaction Count Anomaly Detection
print("Analyzing daily transaction counts...")
daily_transactions = pos_df[~pos_df['is_refund']].groupby('date')['transaction_id'].nunique().reset_index()
daily_transactions.columns = ['date', 'transaction_count']
daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])

# Get baseline transaction counts
baseline_tx_counts = []
for period_start, period_end in baseline_periods:
    period_data = pos_df[(pos_df['calendar_date'] >= period_start) & 
                         (pos_df['calendar_date'] < period_end) & 
                         (~pos_df['is_refund'])]
    period_daily_tx = period_data.groupby('date')['transaction_id'].nunique()
    baseline_tx_counts.extend(period_daily_tx.values)

if len(baseline_tx_counts) > 5 and np.std(baseline_tx_counts) > 0:
    baseline_tx_mean = np.mean(baseline_tx_counts)
    baseline_tx_std = np.std(baseline_tx_counts)
    
    # Analyze analysis period
    analysis_tx = daily_transactions[(daily_transactions['date'] >= analysis_start) & 
                                     (daily_transactions['date'] < analysis_end)]
    
    if len(analysis_tx) > 0:
        for idx, row in analysis_tx.iterrows():
            z_score = abs((row['transaction_count'] - baseline_tx_mean) / baseline_tx_std)
            if z_score > 2.0:
                findings.append({
                    'title': f'Daily Transaction Count Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                    'claim': f'Transaction count of {row["transaction_count"]} on {row["date"].strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations from baseline mean of {baseline_tx_mean:.1f}',
                    'finding_type': 'transaction_volume_anomaly',
                    'metrics': {
                        'daily_transaction_count': {
                            'value': int(row['transaction_count']),
                            'unit': 'transactions',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(days=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_tx_mean, 1),
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
                    'sample_size': len(baseline_tx_counts),
                    'coverage_notes': [f'Baseline from {len(baseline_periods)} weeks of historical data',
                                      'Refunds excluded from transaction count'],
                    'assumptions': ['Normal distribution of daily transaction counts',
                                   'No structural breaks in baseline period',
                                   'Transaction_id uniqueness preserved'],
                    'confidence': 0.82
                })

# Sort findings by z-score magnitude (confidence proxy)
findings.sort(key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)

# Keep only top 3
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
