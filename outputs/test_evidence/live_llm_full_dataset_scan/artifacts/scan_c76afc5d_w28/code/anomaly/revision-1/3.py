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

# Parse analysis periods
analysis_start = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-27T00:00:00+03:00")

baseline_periods = [
    (datetime.fromisoformat("2026-07-13T00:00:00+03:00"), datetime.fromisoformat("2026-07-20T00:00:00+03:00")),
    (datetime.fromisoformat("2026-07-06T00:00:00+03:00"), datetime.fromisoformat("2026-07-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-29T00:00:00+03:00"), datetime.fromisoformat("2026-07-06T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-22T00:00:00+03:00"), datetime.fromisoformat("2026-06-29T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period (excluding refunds in net)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

pos_analysis['date'] = pos_analysis['timestamp'].dt.date
daily_revenue_analysis = pos_analysis.groupby('date').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
daily_revenue_analysis.columns = ['date', 'revenue_sar', 'transaction_count']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp'] >= period_start) & 
        (pos_df['timestamp'] < period_end)
    ].copy()
    pos_baseline['date'] = pos_baseline['timestamp'].dt.date
    daily_rev = pos_baseline.groupby('date')['line_total_sar'].sum().values
    baseline_daily_revenues.extend(daily_rev)

baseline_daily_revenues = np.array(baseline_daily_revenues)

if len(baseline_daily_revenues) > 0 and np.std(baseline_daily_revenues) > 0:
    baseline_mean_revenue = np.mean(baseline_daily_revenues)
    baseline_std_revenue = np.std(baseline_daily_revenues)
    
    # Find anomalies in analysis period
    anomalies_revenue = []
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue_sar'] - baseline_mean_revenue) / baseline_std_revenue
        if abs(z_score) > 2.0:  # 2-sigma threshold
            anomalies_revenue.append({
                'date': row['date'],
                'revenue': row['revenue_sar'],
                'z_score': z_score,
                'baseline_mean': baseline_mean_revenue,
                'baseline_std': baseline_std_revenue
            })
    
    if anomalies_revenue:
        # Sort by magnitude
        anomalies_revenue.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_anomaly = anomalies_revenue[0]
        
        findings.append({
            'title': 'Daily Revenue Anomaly',
            'claim': f"Daily revenue on {top_anomaly['date']} was {top_anomaly['revenue']:.2f} SAR, {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {top_anomaly['baseline_mean']:.2f} SAR.",
            'finding_type': 'revenue_anomaly',
            'metrics': {
                'daily_revenue_sar': {
                    'value': round(top_anomaly['revenue'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': str(top_anomaly['date']),
                    'period_end': str(top_anomaly['date'])
                },
                'baseline_mean_revenue_sar': {
                    'value': round(top_anomaly['baseline_mean'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-06-22',
                    'period_end': '2026-07-20'
                },
                'z_score_revenue': {
                    'value': round(top_anomaly['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': str(top_anomaly['date']),
                    'period_end': str(top_anomaly['date'])
                }
            },
            'source_names': ['pos'],
            'sample_size': len(baseline_daily_revenues),
            'coverage_notes': [
                f"Baseline computed from {len(baseline_daily_revenues)} daily observations across 4 weeks (2026-06-22 to 2026-07-20)",
                f"Analysis period: 2026-07-20 to 2026-07-27",
                "Refunds included in net revenue calculation"
            ],
            'assumptions': [
                'Daily revenue follows approximately normal distribution',
                'Z-score threshold of 2.0 (95% confidence)',
                'Baseline periods are representative of normal operations'
            ],
            'confidence': 0.85
        })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter traffic for analysis period - convert analysis_start.date() to datetime for comparison
analysis_start_date = pd.Timestamp(analysis_start.date())
analysis_end_date = pd.Timestamp(analysis_end.date())

traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start_date) & 
    (traffic_df['date'] < analysis_end_date) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

# Filter traffic for baseline periods
baseline_traffic = []
for period_start, period_end in baseline_periods:
    period_start_date = pd.Timestamp(period_start.date())
    period_end_date = pd.Timestamp(period_end.date())
    traffic_baseline = traffic_df[
        (traffic_df['date'] >= period_start_date) & 
        (traffic_df['date'] < period_end_date) &
        (traffic_df['is_dead_sensor_day'] == False)
    ].copy()
    baseline_traffic.append(traffic_baseline)

baseline_traffic_combined = pd.concat(baseline_traffic, ignore_index=True)

if len(baseline_traffic_combined) > 0 and baseline_traffic_combined['door_count'].std() > 0:
    baseline_mean_traffic = baseline_traffic_combined['door_count'].mean()
    baseline_std_traffic = baseline_traffic_combined['door_count'].std()
    
    # Find anomalies in analysis period
    anomalies_traffic = []
    for idx, row in traffic_analysis.iterrows():
        z_score = (row['door_count'] - baseline_mean_traffic) / baseline_std_traffic
        if abs(z_score) > 2.0:
            anomalies_traffic.append({
                'date': row['date'],
                'hour': row['hour'],
                'door_count': row['door_count'],
                'z_score': z_score,
                'baseline_mean': baseline_mean_traffic,
                'baseline_std': baseline_std_traffic
            })
    
    if anomalies_traffic:
        anomalies_traffic.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_traffic_anomaly = anomalies_traffic[0]
        
        # Format date as string for period fields
        anomaly_date_str = str(top_traffic_anomaly['date'].date())
        
        findings.append({
            'title': 'Hourly Traffic Anomaly',
            'claim': f"Hourly door count on {anomaly_date_str} at hour {top_traffic_anomaly['hour']} was {top_traffic_anomaly['door_count']} visitors, {abs(top_traffic_anomaly['z_score']):.2f} standard deviations from baseline mean of {top_traffic_anomaly['baseline_mean']:.1f}.",
            'finding_type': 'traffic_anomaly',
            'metrics': {
                'hourly_door_count': {
                    'value': int(top_traffic_anomaly['door_count']),
                    'unit': 'visitors',
                    'numerator': None,
                    'denominator': None,
                    'period_start': anomaly_date_str,
                    'period_end': anomaly_date_str
                },
                'baseline_mean_door_count': {
                    'value': round(top_traffic_anomaly['baseline_mean'], 1),
                    'unit': 'visitors',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-06-22',
                    'period_end': '2026-07-20'
                },
                'z_score_traffic': {
                    'value': round(top_traffic_anomaly['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': anomaly_date_str,
                    'period_end': anomaly_date_str
                }
            },
            'source_names': ['traffic'],
            'sample_size': len(baseline_traffic_combined),
            'coverage_notes': [
                f"Baseline computed from {len(baseline_traffic_combined)} hourly observations across 4 weeks (2026-06-22 to 2026-07-20)",
                "Excluded dead sensor days",
                f"Analysis period: 2026-07-20 to 2026-07-27"
            ],
            'assumptions': [
                'Hourly door counts follow approximately normal distribution',
                'Z-score threshold of 2.0 (95% confidence)',
                'Baseline periods represent normal traffic patterns'
            ],
            'confidence': 0.80
        })

# ============================================================================
# ANOMALY 3: Daily Waste Cost Analysis
# ============================================================================

# Calculate daily waste cost from inventory
inventory_analysis = inventory_df[
    inventory_df['week_starting'] >= analysis_start
].copy()

inventory_baseline = inventory_df[
    (inventory_df['week_starting'] >= baseline_periods[0][0]) &
    (inventory_df['week_starting'] < baseline_periods[0][1])
].copy()

if len(inventory_baseline) > 0:
    baseline_waste_cost = inventory_baseline['known_waste_cost_sar'].sum()
    baseline_waste_count = len(inventory_baseline[inventory_baseline['known_waste_cost_sar'] > 0])
    
    if baseline_waste_count > 0:
        baseline_mean_waste = inventory_baseline['known_waste_cost_sar'].mean()
        baseline_std_waste = inventory_baseline['known_waste_cost_sar'].std()
        
        if baseline_std_waste > 0:
            analysis_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
            analysis_waste_count = len(inventory_analysis[inventory_analysis['known_waste_cost_sar'] > 0])
            
            if analysis_waste_count > 0:
                analysis_mean_waste = inventory_analysis['known_waste_cost_sar'].mean()
                z_score_waste = (analysis_mean_waste - baseline_mean_waste) / baseline_std_waste
                
                if abs(z_score_waste) > 1.5:  # Lower threshold for waste due to smaller sample
                    findings.append({
                        'title': 'Waste Cost Anomaly',
                        'claim': f"Mean waste cost per item in analysis period was {analysis_mean_waste:.2f} SAR, {abs(z_score_waste):.2f} standard deviations from baseline mean of {baseline_mean_waste:.2f} SAR.",
                        'finding_type': 'waste_anomaly',
                        'metrics': {
                            'mean_waste_cost_sar': {
                                'value': round(analysis_mean_waste, 2),
                                'unit': 'SAR',
                                'numerator': round(analysis_waste_cost, 2),
                                'denominator': analysis_waste_count,
                                'period_start': '2026-07-20',
                                'period_end': '2026-07-27'
                            },
                            'baseline_mean_waste_cost_sar': {
                                'value': round(baseline_mean_waste, 2),
                                'unit': 'SAR',
                                'numerator': round(baseline_waste_cost, 2),
                                'denominator': baseline_waste_count,
                                'period_start': '2026-07-13',
                                'period_end': '2026-07-20'
                            },
                            'z_score_waste': {
                                'value': round(z_score_waste, 2),
                                'unit': 'std_dev',
                                'numerator': None,
                                'denominator': None,
                                'period_start': '2026-07-20',
                                'period_end': '2026-07-27'
                            }
                        },
                        'source_names': ['inventory'],
                        'sample_size': analysis_waste_count,
                        'coverage_notes': [
                            f"Analysis period waste items: {analysis_waste_count}",
                            f"Baseline period waste items: {baseline_waste_count}",
                            "Only items with known waste cost included"
                        ],
                        'assumptions': [
                            'Waste cost per item follows approximately normal distribution',
                            'Z-score threshold of 1.5 (lower due to small sample)',
                            'Baseline week is representative'
                        ],
                        'confidence': 0.70
                    })

# ============================================================================
# Output Result
# ============================================================================

result = {
    'status': 'success' if findings else 'insufficient_data',
    'findings': findings[:3]  # Return at most 3 findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
