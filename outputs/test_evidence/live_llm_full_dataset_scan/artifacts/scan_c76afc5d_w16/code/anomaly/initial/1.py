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
analysis_start = pd.to_datetime('2026-04-27T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-05-04T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-04-20T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-27T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-04-13T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-20T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-04-06T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-13T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-03-30T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-04-06T00:00:00+03:00').tz_localize(None))
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

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_rev = baseline_pos.groupby('calendar_date').agg({
        'line_total_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    baseline_daily_revenues.extend(daily_rev['line_total_sar'].values)

if len(baseline_daily_revenues) > 0 and np.std(baseline_daily_revenues) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Check each day in analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        if abs(z_score) > 2.0:  # 2 standard deviations
            findings.append({
                'title': f'Daily Revenue Anomaly on {row["date"].strftime("%Y-%m-%d")}',
                'claim': f'Daily revenue of {row["revenue"]:.2f} SAR on {row["date"].strftime("%Y-%m-%d")} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR',
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
                'sample_size': len(baseline_daily_revenues),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: {len(baseline_daily_revenues)} daily observations from 4 weeks prior',
                    f'Transactions on anomaly day: {int(row["transactions"])}'
                ],
                'assumptions': [
                    'Z-score threshold of 2.0 standard deviations',
                    'Baseline calculated from 4 preceding weeks',
                    'Refunds included in net revenue calculation'
                ],
                'confidence': 0.75
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Calculate hourly traffic for baseline
    baseline_traffic = traffic_df[(traffic_df['date'] >= baseline_periods[0][0]) & 
                                  (traffic_df['date'] < baseline_periods[-1][1]) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    
    if len(baseline_traffic) > 0:
        # Group by hour of day
        baseline_traffic['hour_of_day'] = baseline_traffic['hour'].dt.hour
        analysis_traffic['hour_of_day'] = analysis_traffic['hour'].dt.hour
        
        baseline_hourly = baseline_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()
        
        # Check analysis period hourly traffic
        analysis_hourly = analysis_traffic.groupby('hour_of_day')['door_count'].agg(['mean', 'count']).reset_index()
        
        for idx, row in analysis_hourly.iterrows():
            baseline_row = baseline_hourly[baseline_hourly['hour_of_day'] == row['hour_of_day']]
            
            if len(baseline_row) > 0 and baseline_row.iloc[0]['std'] > 0:
                baseline_mean = baseline_row.iloc[0]['mean']
                baseline_std = baseline_row.iloc[0]['std']
                z_score = (row['mean'] - baseline_mean) / baseline_std
                
                if abs(z_score) > 2.0 and row['count'] >= 3:  # At least 3 observations
                    findings.append({
                        'title': f'Hourly Traffic Anomaly at Hour {int(row["hour_of_day"])}:00',
                        'claim': f'Average hourly traffic of {row["mean"]:.0f} visitors at hour {int(row["hour_of_day"])} is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_mean:.0f}',
                        'finding_type': 'traffic_anomaly',
                        'metrics': {
                            'hourly_traffic': {
                                'value': round(row['mean'], 0),
                                'unit': 'visitors',
                                'numerator': None,
                                'denominator': None,
                                'period_start': analysis_start.isoformat(),
                                'period_end': analysis_end.isoformat()
                            },
                            'baseline_mean': {
                                'value': round(baseline_mean, 0),
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
                                'period_start': analysis_start.isoformat(),
                                'period_end': analysis_end.isoformat()
                            }
                        },
                        'source_names': ['traffic'],
                        'sample_size': int(row['count']),
                        'coverage_notes': [
                            f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                            f'Baseline: {int(baseline_row.iloc[0]["count"])} hourly observations from 4 weeks prior',
                            f'Dead sensor days excluded'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.0 standard deviations',
                            'Baseline calculated from same hour across 4 preceding weeks',
                            'Minimum 3 observations required for analysis'
                        ],
                        'confidence': 0.70
                    })

# ============================================================================
# ANOMALY 3: Daily Waste Analysis
# ============================================================================

# Get waste data for analysis period
analysis_inventory = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

baseline_inventory = inventory_df[
    (inventory_df['week_starting'] >= baseline_periods[0][0]) & 
    (inventory_df['week_starting'] < baseline_periods[-1][1])
].copy()

if len(analysis_inventory) > 0 and len(baseline_inventory) > 0:
    # Calculate waste metrics
    analysis_waste = analysis_inventory['units_wasted'].sum()
    baseline_waste_values = baseline_inventory.groupby('week_starting')['units_wasted'].sum().values
    
    if len(baseline_waste_values) > 0 and np.std(baseline_waste_values) > 0:
        baseline_waste_mean = np.mean(baseline_waste_values)
        baseline_waste_std = np.std(baseline_waste_values)
        
        z_score = (analysis_waste - baseline_waste_mean) / baseline_waste_std
        
        if abs(z_score) > 1.5:  # 1.5 standard deviations for waste
            findings.append({
                'title': 'Weekly Waste Volume Anomaly',
                'claim': f'Total waste of {analysis_waste:.0f} units in analysis week is {abs(z_score):.2f} standard deviations from baseline mean of {baseline_waste_mean:.0f} units',
                'finding_type': 'waste_anomaly',
                'metrics': {
                    'weekly_waste_units': {
                        'value': round(analysis_waste, 0),
                        'unit': 'units',
                        'numerator': None,
                        'denominator': None,
                        'period_start': analysis_start.isoformat(),
                        'period_end': analysis_end.isoformat()
                    },
                    'baseline_mean': {
                        'value': round(baseline_waste_mean, 0),
                        'unit': 'units',
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
                        'period_start': analysis_start.isoformat(),
                        'period_end': analysis_end.isoformat()
                    }
                },
                'source_names': ['inventory'],
                'sample_size': len(baseline_waste_values),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: {len(baseline_waste_values)} weekly observations from 4 weeks prior',
                    f'Unknown waste values excluded per schema'
                ],
                'assumptions': [
                    'Z-score threshold of 1.5 standard deviations',
                    'Baseline calculated from 4 preceding weeks',
                    'Weekly aggregation used'
                ],
                'confidence': 0.65
            })

# Sort findings by z-score magnitude
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
    json.dump(output, f, indent=2)
