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
analysis_start = pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-07-06T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-29T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-22T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-15T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-06-08T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# FINDING 1: Daily Revenue Anomaly Detection
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
    
    # Find anomalies using z-score
    anomalies = []
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2:  # 2 standard deviations
            anomalies.append({
                'date': row['date'],
                'revenue': row['revenue'],
                'z_score': z_score,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std
            })
    
    if anomalies:
        # Sort by magnitude
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_anomaly = anomalies[0]
        
        findings.append({
            'title': 'Daily Revenue Anomaly',
            'claim': f"Daily revenue on {top_anomaly['date'].strftime('%Y-%m-%d')} was {top_anomaly['revenue']:.2f} SAR, "
                    f"which is {abs(top_anomaly['z_score']):.2f} standard deviations from the baseline mean of {top_anomaly['baseline_mean']:.2f} SAR.",
            'finding_type': 'revenue_anomaly',
            'metrics': {
                'observed_daily_revenue': {
                    'value': round(top_anomaly['revenue'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': top_anomaly['date'].isoformat(),
                    'period_end': (top_anomaly['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_revenue': {
                    'value': round(top_anomaly['baseline_mean'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': baseline_periods[0][0].isoformat(),
                    'period_end': baseline_periods[-1][1].isoformat()
                },
                'z_score': {
                    'value': round(top_anomaly['z_score'], 2),
                    'unit': 'standard_deviations',
                    'numerator': None,
                    'denominator': None,
                    'period_start': top_anomaly['date'].isoformat(),
                    'period_end': (top_anomaly['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': len(daily_revenue_analysis),
            'coverage_notes': [
                f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                f'Baseline: {len(baseline_revenues)} daily observations from 4 weeks',
                'Excludes refunds: False (line_total_sar includes refunds as negative values)'
            ],
            'assumptions': [
                'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                'Baseline calculated from 4 preceding weeks',
                'Daily revenue aggregated from POS line items'
            ],
            'confidence': 0.85
        })

# ============================================================================
# FINDING 2: Hourly Traffic Anomaly Detection
# ============================================================================

# Filter traffic data for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Calculate baseline hourly traffic
    baseline_traffic_data = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic_data.extend(baseline_traffic['door_count'].values)
    
    if len(baseline_traffic_data) > 0 and np.std(baseline_traffic_data) > 0:
        baseline_traffic_mean = np.mean(baseline_traffic_data)
        baseline_traffic_std = np.std(baseline_traffic_data)
        
        # Find anomalies
        traffic_anomalies = []
        for idx, row in analysis_traffic.iterrows():
            z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std if baseline_traffic_std > 0 else 0
            if abs(z_score) > 2:
                traffic_anomalies.append({
                    'hour': row['hour'],
                    'door_count': row['door_count'],
                    'z_score': z_score,
                    'baseline_mean': baseline_traffic_mean,
                    'baseline_std': baseline_traffic_std
                })
        
        if traffic_anomalies:
            traffic_anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
            top_traffic_anomaly = traffic_anomalies[0]
            
            findings.append({
                'title': 'Hourly Traffic Anomaly',
                'claim': f"Door count at {top_traffic_anomaly['hour'].strftime('%Y-%m-%d %H:00')} was {top_traffic_anomaly['door_count']:.0f}, "
                        f"which is {abs(top_traffic_anomaly['z_score']):.2f} standard deviations from the baseline mean of {top_traffic_anomaly['baseline_mean']:.2f}.",
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_hourly_door_count': {
                        'value': int(top_traffic_anomaly['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': top_traffic_anomaly['hour'].isoformat(),
                        'period_end': (top_traffic_anomaly['hour'] + timedelta(hours=1)).isoformat()
                    },
                    'baseline_mean_hourly_door_count': {
                        'value': round(top_traffic_anomaly['baseline_mean'], 2),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(top_traffic_anomaly['z_score'], 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': top_traffic_anomaly['hour'].isoformat(),
                        'period_end': (top_traffic_anomaly['hour'] + timedelta(hours=1)).isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(analysis_traffic),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: {len(baseline_traffic_data)} hourly observations from 4 weeks',
                    'Excluded dead sensor days as marked in is_dead_sensor_day'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 preceding weeks',
                    'Dead sensor intervals excluded from analysis'
                ],
                'confidence': 0.80
            })

# ============================================================================
# FINDING 3: Weekly Waste Cost Anomaly Detection
# ============================================================================

# Filter inventory for analysis and baseline weeks
analysis_week_start = pd.to_datetime('2026-06-29').date()
analysis_inventory = inventory_df[inventory_df['week_starting'].dt.date == analysis_week_start].copy()

if len(analysis_inventory) > 0:
    analysis_waste_cost = analysis_inventory['known_waste_cost_sar'].sum()
    
    # Calculate baseline waste costs
    baseline_waste_costs = []
    for period_start, period_end in baseline_periods:
        baseline_week = period_start.date()
        baseline_inv = inventory_df[inventory_df['week_starting'].dt.date == baseline_week].copy()
        if len(baseline_inv) > 0:
            baseline_waste_costs.append(baseline_inv['known_waste_cost_sar'].sum())
    
    if len(baseline_waste_costs) > 0 and np.std(baseline_waste_costs) > 0:
        baseline_waste_mean = np.mean(baseline_waste_costs)
        baseline_waste_std = np.std(baseline_waste_costs)
        
        z_score_waste = (analysis_waste_cost - baseline_waste_mean) / baseline_waste_std if baseline_waste_std > 0 else 0
        
        if abs(z_score_waste) > 1.5:  # Lower threshold for waste due to smaller sample
            findings.append({
                'title': 'Weekly Waste Cost Anomaly',
                'claim': f"Weekly waste cost for week starting {analysis_week_start} was {analysis_waste_cost:.2f} SAR, "
                        f"which is {abs(z_score_waste):.2f} standard deviations from the baseline mean of {baseline_waste_mean:.2f} SAR.",
                'finding_type': 'waste_anomaly',
                'metrics': {
                    'observed_weekly_waste_cost': {
                        'value': round(analysis_waste_cost, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': analysis_week_start.isoformat(),
                        'period_end': (analysis_week_start + timedelta(days=7)).isoformat()
                    },
                    'baseline_mean_weekly_waste_cost': {
                        'value': round(baseline_waste_mean, 2),
                        'unit': 'SAR',
                        'numerator': None,
                        'denominator': None,
                        'period_start': baseline_periods[0][0].isoformat(),
                        'period_end': baseline_periods[-1][1].isoformat()
                    },
                    'z_score': {
                        'value': round(z_score_waste, 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': analysis_week_start.isoformat(),
                        'period_end': (analysis_week_start + timedelta(days=7)).isoformat()
                    }
                },
                'source_names': ['inventory'],
                'sample_size': len(baseline_waste_costs),
                'coverage_notes': [
                    f'Analysis week: {analysis_week_start}',
                    f'Baseline: {len(baseline_waste_costs)} weekly observations',
                    'Uses known_waste_cost_sar field; unknown waste values excluded'
                ],
                'assumptions': [
                    'Z-score threshold of 1.5 standard deviations used (lower due to small sample)',
                    'Baseline calculated from 4 preceding weeks',
                    'Weekly waste cost aggregated from inventory records'
                ],
                'confidence': 0.70
            })

# Sort findings by confidence and limit to 3
findings.sort(key=lambda x: x['confidence'], reverse=True)
findings = findings[:3]

# Prepare output
output = {
    'status': 'success' if findings else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
