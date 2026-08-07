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
analysis_start = pd.to_datetime('2026-07-20T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-07-27T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-07-13T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-07-20T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-07-13T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                       (pos_df['calendar_date'] < analysis_end)].copy()

daily_revenue_analysis = pos_analysis.groupby('calendar_date').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
daily_revenue_analysis.columns = ['date', 'revenue', 'transactions']

# Calculate baseline daily revenue
baseline_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_rev = pos_baseline.groupby('calendar_date')['line_total_sar'].sum()
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
                        'period_start': '2026-06-22T00:00:00',
                        'period_end': '2026-07-20T00:00:00'
                    },
                    'z_score': {
                        'value': round(z_score, 2),
                        'unit': None,
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
                    f'Baseline: 4 weeks of historical data (2026-06-22 to 2026-07-20)',
                    f'Baseline sample size: {len(baseline_revenues)} daily observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 preceding weeks',
                    'Refunds included in net revenue calculation'
                ],
                'confidence': 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic data for analysis period
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(traffic_analysis) > 0:
    # Calculate baseline hourly traffic
    baseline_traffic = []
    for period_start, period_end in baseline_periods:
        traffic_baseline = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic.extend(traffic_baseline['door_count'].values)
    
    if len(baseline_traffic) > 10 and np.std(baseline_traffic) > 0:
        baseline_traffic_mean = np.mean(baseline_traffic)
        baseline_traffic_std = np.std(baseline_traffic)
        
        # Check for anomalies
        for idx, row in traffic_analysis.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std
            
            if abs(z_score) > 2.5:  # Higher threshold for traffic
                findings.append({
                    'title': f'Hourly Traffic Anomaly on {row["date"].strftime("%Y-%m-%d %H:00")}',
                    'claim': f'Door count of {row["door_count"]} at {row["date"].strftime("%Y-%m-%d %H:00")} deviates {abs(z_score):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                    'finding_type': 'traffic_anomaly',
                    'metrics': {
                        'hourly_door_count': {
                            'value': int(row['door_count']),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(hours=1)).isoformat()
                        },
                        'baseline_mean': {
                            'value': round(baseline_traffic_mean, 1),
                            'unit': 'visitors',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-22T00:00:00',
                            'period_end': '2026-07-20T00:00:00'
                        },
                        'z_score': {
                            'value': round(z_score, 2),
                            'unit': None,
                            'numerator': None,
                            'denominator': None,
                            'period_start': row['date'].isoformat(),
                            'period_end': (row['date'] + timedelta(hours=1)).isoformat()
                        }
                    },
                    'source_names': ['traffic'],
                    'sample_size': len(baseline_traffic),
                    'coverage_notes': [
                        f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                        f'Excluded dead sensor days',
                        f'Baseline: 4 weeks of hourly observations'
                    ],
                    'assumptions': [
                        'Z-score threshold of 2.5 standard deviations used',
                        'Dead sensor days excluded from analysis',
                        'Baseline calculated from 4 preceding weeks'
                    ],
                    'confidence': 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Waste Analysis
# ============================================================================

# Calculate weekly waste for analysis period
inventory_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start].copy()

if len(inventory_analysis) > 0:
    # Calculate baseline waste
    baseline_waste = []
    for period_start, period_end in baseline_periods:
        inv_baseline = inventory_df[(inventory_df['week_starting'] >= period_start) & 
                                    (inventory_df['week_starting'] < period_end)].copy()
        # Only include rows with known waste values
        inv_baseline = inv_baseline[inv_baseline['units_wasted'].notna()]
        baseline_waste.extend(inv_baseline['units_wasted'].values)
    
    if len(baseline_waste) > 5 and np.std(baseline_waste) > 0:
        baseline_waste_mean = np.mean(baseline_waste)
        baseline_waste_std = np.std(baseline_waste)
        
        # Check for anomalies in analysis period
        for idx, row in inventory_analysis.iterrows():
            if pd.notna(row['units_wasted']):
                z_score = (row['units_wasted'] - baseline_waste_mean) / baseline_waste_std
                
                if z_score > 2.0:  # Only flag high waste
                    findings.append({
                        'title': f'High Waste Anomaly for {row["item"]} (Week of {row["week_starting"].strftime("%Y-%m-%d")})',
                        'claim': f'Waste of {row["units_wasted"]} units for {row["item"]} deviates {z_score:.2f} standard deviations above baseline mean of {baseline_waste_mean:.1f} units',
                        'finding_type': 'waste_anomaly',
                        'metrics': {
                            'units_wasted': {
                                'value': int(row['units_wasted']),
                                'unit': 'units',
                                'numerator': None,
                                'denominator': None,
                                'period_start': row['week_starting'].isoformat(),
                                'period_end': (row['week_starting'] + timedelta(days=7)).isoformat()
                            },
                            'baseline_mean': {
                                'value': round(baseline_waste_mean, 1),
                                'unit': 'units',
                                'numerator': None,
                                'denominator': None,
                                'period_start': '2026-06-22T00:00:00',
                                'period_end': '2026-07-20T00:00:00'
                            },
                            'z_score': {
                                'value': round(z_score, 2),
                                'unit': None,
                                'numerator': None,
                                'denominator': None,
                                'period_start': row['week_starting'].isoformat(),
                                'period_end': (row['week_starting'] + timedelta(days=7)).isoformat()
                            }
                        },
                        'source_names': ['inventory'],
                        'sample_size': len(baseline_waste),
                        'coverage_notes': [
                            f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                            f'Only items with known waste values included',
                            f'Baseline: {len(baseline_waste)} weekly waste observations'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.0 standard deviations used',
                            'Only flagging high waste (positive z-score)',
                            'Baseline calculated from 4 preceding weeks'
                        ],
                        'confidence': 0.75
                    })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: x['confidence'], reverse=True)

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
