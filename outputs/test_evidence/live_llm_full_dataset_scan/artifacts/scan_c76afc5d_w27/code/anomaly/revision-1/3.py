import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load input metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse periods
analysis_start = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-13T00:00:00+03:00")

trailing_periods = [
    (datetime.fromisoformat("2026-07-06T00:00:00+03:00"), datetime.fromisoformat("2026-07-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-29T00:00:00+03:00"), datetime.fromisoformat("2026-07-06T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-22T00:00:00+03:00"), datetime.fromisoformat("2026-06-29T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-15T00:00:00+03:00"), datetime.fromisoformat("2026-06-22T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamps to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue
# ============================================================================

# Calculate daily revenue (net of refunds)
pos_df['is_refund_flag'] = pos_df['is_refund'].astype(bool)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

pos_analysis['calendar_date'] = pd.to_datetime(pos_analysis['calendar_date'])
daily_revenue_analysis = pos_analysis.groupby('calendar_date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Baseline: trailing periods
baseline_revenue_data = []
for period_start, period_end in trailing_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp'] >= period_start) & 
        (pos_df['timestamp'] < period_end)
    ].copy()
    pos_baseline['calendar_date'] = pd.to_datetime(pos_baseline['calendar_date'])
    daily_rev = pos_baseline.groupby('calendar_date')['line_total_sar'].sum().reset_index()
    baseline_revenue_data.append(daily_rev['line_total_sar'].values)

baseline_revenue_flat = np.concatenate(baseline_revenue_data)
baseline_revenue_mean = np.mean(baseline_revenue_flat)
baseline_revenue_std = np.std(baseline_revenue_flat)

if baseline_revenue_std > 0 and len(baseline_revenue_flat) >= 10:
    daily_revenue_analysis['z_score'] = (daily_revenue_analysis['revenue'] - baseline_revenue_mean) / baseline_revenue_std
    anomaly_revenue = daily_revenue_analysis[daily_revenue_analysis['z_score'].abs() > 2.0]
    
    if len(anomaly_revenue) > 0:
        top_anomaly_rev = anomaly_revenue.loc[anomaly_revenue['z_score'].abs().idxmax()]
        findings.append({
            'title': 'Unusual Daily Revenue',
            'claim': f"Daily revenue on {top_anomaly_rev['date'].strftime('%Y-%m-%d')} was {top_anomaly_rev['revenue']:.2f} SAR, {abs(top_anomaly_rev['z_score']):.2f} standard deviations from baseline mean of {baseline_revenue_mean:.2f} SAR.",
            'finding_type': 'revenue_anomaly',
            'metrics': {
                'observed_daily_revenue_sar': {
                    'value': round(top_anomaly_rev['revenue'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': top_anomaly_rev['date'].strftime('%Y-%m-%dT00:00:00+03:00'),
                    'period_end': (pd.to_datetime(top_anomaly_rev['date']) + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                },
                'baseline_mean_daily_revenue_sar': {
                    'value': round(baseline_revenue_mean, 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-06-15T00:00:00+03:00',
                    'period_end': '2026-07-13T00:00:00+03:00'
                },
                'z_score': {
                    'value': round(top_anomaly_rev['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': top_anomaly_rev['date'].strftime('%Y-%m-%dT00:00:00+03:00'),
                    'period_end': (pd.to_datetime(top_anomaly_rev['date']) + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00+03:00')
                }
            },
            'source_names': ['pos'],
            'sample_size': len(baseline_revenue_flat),
            'coverage_notes': [
                'Analysis period: 2026-07-13 to 2026-07-20',
                'Baseline: 4 trailing weeks (2026-06-15 to 2026-07-13)',
                'Threshold: |z-score| > 2.0',
                'Refunds included in net revenue calculation'
            ],
            'assumptions': [
                'Daily revenue follows approximately normal distribution',
                'Baseline periods are representative of typical operations',
                'No known system outages or data quality issues in analysis period'
            ],
            'confidence': 0.75
        })

# ============================================================================
# ANOMALY 2: Hourly Traffic (Door Count)
# ============================================================================

traffic_analysis = traffic_df[
    (traffic_df['date'] >= pd.to_datetime(analysis_start.date())) & 
    (traffic_df['date'] < pd.to_datetime(analysis_end.date())) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

if len(traffic_analysis) > 0:
    # Baseline: trailing periods
    baseline_traffic_data = []
    for period_start, period_end in trailing_periods:
        traffic_baseline = traffic_df[
            (traffic_df['date'] >= pd.to_datetime(period_start.date())) & 
            (traffic_df['date'] < pd.to_datetime(period_end.date())) &
            (traffic_df['is_dead_sensor_day'] == False)
        ].copy()
        baseline_traffic_data.append(traffic_baseline['door_count'].values)
    
    baseline_traffic_flat = np.concatenate(baseline_traffic_data)
    baseline_traffic_mean = np.mean(baseline_traffic_flat)
    baseline_traffic_std = np.std(baseline_traffic_flat)
    
    if baseline_traffic_std > 0 and len(baseline_traffic_flat) >= 20:
        traffic_analysis['z_score'] = (traffic_analysis['door_count'] - baseline_traffic_mean) / baseline_traffic_std
        anomaly_traffic = traffic_analysis[traffic_analysis['z_score'].abs() > 2.5]
        
        if len(anomaly_traffic) > 0:
            top_anomaly_traffic = anomaly_traffic.loc[anomaly_traffic['z_score'].abs().idxmax()]
            hour_str = int(top_anomaly_traffic['hour'])
            date_str = pd.to_datetime(top_anomaly_traffic['date']).strftime('%Y-%m-%d')
            findings.append({
                'title': 'Unusual Hourly Door Traffic',
                'claim': f"Hourly door count on {date_str} at hour {hour_str} was {int(top_anomaly_traffic['door_count'])} visitors, {abs(top_anomaly_traffic['z_score']):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}.",
                'finding_type': 'traffic_anomaly',
                'metrics': {
                    'observed_hourly_door_count': {
                        'value': int(top_anomaly_traffic['door_count']),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': f"{date_str}T{hour_str:02d}:00:00+03:00",
                        'period_end': f"{date_str}T{hour_str+1:02d}:00:00+03:00"
                    },
                    'baseline_mean_hourly_door_count': {
                        'value': round(baseline_traffic_mean, 1),
                        'unit': 'visitors',
                        'numerator': None,
                        'denominator': None,
                        'period_start': '2026-06-15T00:00:00+03:00',
                        'period_end': '2026-07-13T00:00:00+03:00'
                    },
                    'z_score': {
                        'value': round(top_anomaly_traffic['z_score'], 2),
                        'unit': 'std_dev',
                        'numerator': None,
                        'denominator': None,
                        'period_start': f"{date_str}T{hour_str:02d}:00:00+03:00",
                        'period_end': f"{date_str}T{hour_str+1:02d}:00:00+03:00"
                    }
                },
                'source_names': ['traffic'],
                'sample_size': len(baseline_traffic_flat),
                'coverage_notes': [
                    'Analysis period: 2026-07-13 to 2026-07-20',
                    'Baseline: 4 trailing weeks (2026-06-15 to 2026-07-13)',
                    'Excluded dead sensor days',
                    'Threshold: |z-score| > 2.5'
                ],
                'assumptions': [
                    'Hourly door counts follow approximately normal distribution',
                    'Baseline periods represent typical traffic patterns',
                    'Sensor is functioning correctly during analysis period'
                ],
                'confidence': 0.70
            })

# ============================================================================
# ANOMALY 3: Average Daily Rating
# ============================================================================

reviews_analysis = reviews_df[
    (reviews_df['date'] >= pd.to_datetime(analysis_start.date())) & 
    (reviews_df['date'] < pd.to_datetime(analysis_end.date()))
].copy()

if len(reviews_analysis) > 0:
    daily_rating_analysis = reviews_analysis.groupby('date')['rating'].agg(['mean', 'count']).reset_index()
    daily_rating_analysis.columns = ['date', 'avg_rating', 'count']
    
    # Baseline: trailing periods
    baseline_rating_data = []
    for period_start, period_end in trailing_periods:
        reviews_baseline = reviews_df[
            (reviews_df['date'] >= pd.to_datetime(period_start.date())) & 
            (reviews_df['date'] < pd.to_datetime(period_end.date()))
        ].copy()
        if len(reviews_baseline) > 0:
            baseline_rating_data.append(reviews_baseline['rating'].values)
    
    if len(baseline_rating_data) > 0:
        baseline_rating_flat = np.concatenate(baseline_rating_data)
        baseline_rating_mean = np.mean(baseline_rating_flat)
        baseline_rating_std = np.std(baseline_rating_flat)
        
        if baseline_rating_std > 0 and len(baseline_rating_flat) >= 15:
            daily_rating_analysis['z_score'] = (daily_rating_analysis['avg_rating'] - baseline_rating_mean) / baseline_rating_std
            anomaly_rating = daily_rating_analysis[daily_rating_analysis['z_score'].abs() > 1.8]
            
            if len(anomaly_rating) > 0:
                top_anomaly_rating = anomaly_rating.loc[anomaly_rating['z_score'].abs().idxmax()]
                date_str = pd.to_datetime(top_anomaly_rating['date']).strftime('%Y-%m-%d')
                findings.append({
                    'title': 'Unusual Daily Average Rating',
                    'claim': f"Average rating on {date_str} was {top_anomaly_rating['avg_rating']:.2f} ({int(top_anomaly_rating['count'])} reviews), {abs(top_anomaly_rating['z_score']):.2f} standard deviations from baseline mean of {baseline_rating_mean:.2f}.",
                    'finding_type': 'rating_anomaly',
                    'metrics': {
                        'observed_daily_avg_rating': {
                            'value': round(top_anomaly_rating['avg_rating'], 2),
                            'unit': 'stars',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{(pd.to_datetime(date_str) + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00+03:00"
                        },
                        'baseline_mean_rating': {
                            'value': round(baseline_rating_mean, 2),
                            'unit': 'stars',
                            'numerator': None,
                            'denominator': None,
                            'period_start': '2026-06-15T00:00:00+03:00',
                            'period_end': '2026-07-13T00:00:00+03:00'
                        },
                        'z_score': {
                            'value': round(top_anomaly_rating['z_score'], 2),
                            'unit': 'std_dev',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{(pd.to_datetime(date_str) + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00+03:00"
                        },
                        'review_count': {
                            'value': int(top_anomaly_rating['count']),
                            'unit': 'reviews',
                            'numerator': None,
                            'denominator': None,
                            'period_start': f"{date_str}T00:00:00+03:00",
                            'period_end': f"{(pd.to_datetime(date_str) + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00+03:00"
                        }
                    },
                    'source_names': ['reviews'],
                    'sample_size': len(baseline_rating_flat),
                    'coverage_notes': [
                        'Analysis period: 2026-07-13 to 2026-07-20',
                        'Baseline: 4 trailing weeks (2026-06-15 to 2026-07-13)',
                        'Threshold: |z-score| > 1.8',
                        'Daily average computed from all reviews regardless of source'
                    ],
                    'assumptions': [
                        'Daily average ratings follow approximately normal distribution',
                        'Baseline periods represent typical customer satisfaction',
                        'Review volume is sufficient for meaningful daily averages'
                    ],
                    'confidence': 0.65
                })

# Sort by confidence and take top 3
findings = sorted(findings, key=lambda x: x['confidence'], reverse=True)[:3]

result = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
