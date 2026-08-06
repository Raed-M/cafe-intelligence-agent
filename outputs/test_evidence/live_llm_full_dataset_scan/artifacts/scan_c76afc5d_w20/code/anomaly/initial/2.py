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
analysis_start = pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-06-01T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-25T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-11T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-18T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-05-04T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-11T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-04-27T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-05-04T00:00:00+03:00').tz_localize(None))
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
    
    # Check each day in analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        if abs(z_score) > 2.0:  # 2 standard deviations
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
                        'period_start': '2026-04-27T00:00:00',
                        'period_end': '2026-05-25T00:00:00'
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
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks prior (2026-04-27 to 2026-05-25)',
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

# Filter traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

if len(analysis_traffic) > 0:
    # Calculate baseline hourly traffic
    baseline_traffic_list = []
    for period_start, period_end in baseline_periods:
        baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                      (traffic_df['date'] < period_end) &
                                      (traffic_df['is_dead_sensor_day'] == False)].copy()
        baseline_traffic_list.extend(baseline_traffic['door_count'].values)
    
    if len(baseline_traffic_list) > 10 and np.std(baseline_traffic_list) > 0:
        baseline_traffic_mean = np.mean(baseline_traffic_list)
        baseline_traffic_std = np.std(baseline_traffic_list)
        
        # Find anomalous hours
        analysis_traffic['z_score'] = (analysis_traffic['door_count'] - baseline_traffic_mean) / baseline_traffic_std
        
        anomalous_hours = analysis_traffic[abs(analysis_traffic['z_score']) > 2.5].copy()
        
        if len(anomalous_hours) > 0:
            # Get the most extreme anomaly
            most_extreme = anomalous_hours.loc[abs(anomalous_hours['z_score']).idxmax()]
            
            findings.append({
                'title': f'Hourly Traffic Anomaly on {most_extreme["date"].strftime("%Y-%m-%d %H:00")}',
                'claim': f'Door count of {int(most_extreme["door_count"])} at {most_extreme["date"].strftime("%Y-%m-%d %H:00")} deviates {abs(most_extreme["z_score"]):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}',
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_hourly_door_count': {
                        'value': int(most_extreme['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': most_extreme['date'].isoformat(),
                        'period_end': (most_extreme['date'] + timedelta(hours=1)).isoformat()
                    },
                    'baseline_mean_hourly_door_count': {
                        'value': round(baseline_traffic_mean, 1),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-04-27T00:00:00',
                        'period_end': '2026-05-25T00:00:00'
                    },
                    'z_score': {
                        'value': round(most_extreme['z_score'], 2),
                        'unit': 'standard_deviations',
                        'numerator': None,
                        'denominator': None,
                        'period_start': most_extreme['date'].isoformat(),
                        'period_end': (most_extreme['date'] + timedelta(hours=1)).isoformat()
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic_list),
                'coverage_notes': [
                    f'Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}',
                    f'Baseline: 4 weeks prior (2026-04-27 to 2026-05-25)',
                    f'Excluded dead sensor days',
                    f'Baseline sample size: {len(baseline_traffic_list)} hourly observations'
                ],
                'assumptions': [
                    'Z-score threshold of 2.5 standard deviations used to identify anomalies',
                    'Baseline calculated from 4 preceding weeks',
                    'Dead sensor intervals excluded from analysis'
                ],
                'confidence': 0.80
            })

# ============================================================================
# ANOMALY 3: Daily Waste Analysis
# ============================================================================

# Get inventory data for analysis week
analysis_week_start = pd.to_datetime('2026-05-25T00:00:00').tz_localize(None)
analysis_inventory = inventory_df[inventory_df['week_starting'] == analysis_week_start].copy()

if len(analysis_inventory) > 0:
    # Get baseline waste data
    baseline_waste_list = []
    baseline_waste_by_item = {}
    
    for period_start, period_end in baseline_periods:
        baseline_inv = inventory_df[inventory_df['week_starting'] == period_start].copy()
        baseline_waste_list.extend(baseline_inv['units_wasted'].values)
        
        for idx, row in baseline_inv.iterrows():
            sku = row['sku']
            if sku not in baseline_waste_by_item:
                baseline_waste_by_item[sku] = []
            baseline_waste_by_item[sku].append(row['units_wasted'])
    
    if len(baseline_waste_list) > 5 and np.std(baseline_waste_list) > 0:
        baseline_waste_mean = np.mean(baseline_waste_list)
        baseline_waste_std = np.std(baseline_waste_list)
        
        # Check for anomalous waste items
        for idx, row in analysis_inventory.iterrows():
            sku = row['sku']
            waste = row['units_wasted']
            
            if pd.notna(waste) and waste > 0:
                # Use item-specific baseline if available
                if sku in baseline_waste_by_item and len(baseline_waste_by_item[sku]) > 2:
                    item_baseline_mean = np.mean(baseline_waste_by_item[sku])
                    item_baseline_std = np.std(baseline_waste_by_item[sku])
                    if item_baseline_std > 0:
                        z_score = (waste - item_baseline_mean) / item_baseline_std
                    else:
                        z_score = 0
                else:
                    z_score = (waste - baseline_waste_mean) / baseline_waste_std if baseline_waste_std > 0 else 0
                
                if z_score > 2.0:  # Only flag high waste
                    findings.append({
                        'title': f'High Waste Anomaly for {row["item"]} (Week of {analysis_week_start.strftime("%Y-%m-%d")})',
                        'claim': f'Waste of {int(waste)} units for {row["item"]} (SKU: {sku}) in week of {analysis_week_start.strftime("%Y-%m-%d")} deviates {z_score:.2f} standard deviations above baseline',
                        'finding_type': 'waste_anomaly',
                        'metrics': {
                            'observed_weekly_waste_units': {
                                'value': int(waste),
                                'unit': 'units',
                                'numerator': None,
                                'denominator': None,
                                'period_start': analysis_week_start.isoformat(),
                                'period_end': (analysis_week_start + timedelta(days=7)).isoformat()
                            },
                            'baseline_mean_waste_units': {
                                'value': round(baseline_waste_mean, 1),
                                'unit': 'units',
                                'numerator': None,
                                'denominator': None,
                                'period_start': '2026-04-27T00:00:00',
                                'period_end': '2026-05-25T00:00:00'
                            },
                            'z_score': {
                                'value': round(z_score, 2),
                                'unit': 'standard_deviations',
                                'numerator': None,
                                'denominator': None,
                                'period_start': analysis_week_start.isoformat(),
                                'period_end': (analysis_week_start + timedelta(days=7)).isoformat()
                            }
                        },
                        'source_names': ['inventory'],
                        'sample_size': len(baseline_waste_list),
                        'coverage_notes': [
                            f'Analysis period: Week of {analysis_week_start.strftime("%Y-%m-%d")}',
                            f'Baseline: 4 preceding weeks',
                            f'Baseline sample size: {len(baseline_waste_list)} weekly observations',
                            'Unknown waste values excluded from analysis'
                        ],
                        'assumptions': [
                            'Z-score threshold of 2.0 standard deviations used to identify anomalies',
                            'Baseline calculated from 4 preceding weeks',
                            'Item-specific baseline used when available'
                        ],
                        'confidence': 0.75
                    })

# Sort findings by confidence and limit to 3
findings = sorted(findings, key=lambda x: x['confidence'], reverse=True)[:3]

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
