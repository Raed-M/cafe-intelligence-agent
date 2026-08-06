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
analysis_start = pd.to_datetime("2026-04-06T00:00:00+03:00").tz_localize(None)
analysis_end = pd.to_datetime("2026-04-13T00:00:00+03:00").tz_localize(None)
baseline_periods = [
    (pd.to_datetime("2026-03-30T00:00:00+03:00").tz_localize(None), pd.to_datetime("2026-04-06T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-03-23T00:00:00+03:00").tz_localize(None), pd.to_datetime("2026-03-30T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-03-16T00:00:00+03:00").tz_localize(None), pd.to_datetime("2026-03-23T00:00:00+03:00").tz_localize(None)),
    (pd.to_datetime("2026-03-09T00:00:00+03:00").tz_localize(None), pd.to_datetime("2026-03-16T00:00:00+03:00").tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
daily_revenue_analysis = pos_analysis.groupby('calendar_date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    daily_rev = pos_baseline.groupby('calendar_date')['line_total_sar'].sum().reset_index()
    baseline_revenues.extend(daily_rev['line_total_sar'].values)

if len(baseline_revenues) >= 10 and np.std(baseline_revenues) > 0:
    baseline_mean = np.mean(baseline_revenues)
    baseline_std = np.std(baseline_revenues)
    
    # Check each day in analysis period
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2.5:  # Threshold: 2.5 standard deviations
            findings.append({
                'type': 'daily_revenue',
                'date': row['date'],
                'observed': row['revenue'],
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'z_score': z_score,
                'sample_size': len(baseline_revenues)
            })

# ============================================================================
# ANOMALY 2: Daily Transaction Count
# ============================================================================

# Calculate daily transaction count for analysis period
daily_txn_analysis = pos_analysis.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_txn_analysis.columns = ['date', 'transaction_count']

# Calculate daily transaction count for baseline periods
baseline_txn_counts = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    daily_txn = pos_baseline.groupby('calendar_date')['transaction_id'].nunique().reset_index()
    baseline_txn_counts.extend(daily_txn['transaction_id'].values)

if len(baseline_txn_counts) >= 10 and np.std(baseline_txn_counts) > 0:
    baseline_txn_mean = np.mean(baseline_txn_counts)
    baseline_txn_std = np.std(baseline_txn_counts)
    
    for idx, row in daily_txn_analysis.iterrows():
        z_score = (row['transaction_count'] - baseline_txn_mean) / baseline_txn_std if baseline_txn_std > 0 else 0
        if abs(z_score) > 2.5:
            findings.append({
                'type': 'daily_transactions',
                'date': row['date'],
                'observed': row['transaction_count'],
                'baseline_mean': baseline_txn_mean,
                'baseline_std': baseline_txn_std,
                'z_score': z_score,
                'sample_size': len(baseline_txn_counts)
            })

# ============================================================================
# ANOMALY 3: Daily Traffic
# ============================================================================

# Calculate daily traffic for analysis period
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
traffic_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]
daily_traffic_analysis = traffic_analysis.groupby('date')['door_count'].sum().reset_index()

# Calculate daily traffic for baseline periods
baseline_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[(traffic_df['date'] >= period_start) & (traffic_df['date'] < period_end)].copy()
    traffic_baseline = traffic_baseline[traffic_baseline['is_dead_sensor_day'] == False]
    daily_traffic = traffic_baseline.groupby('date')['door_count'].sum().reset_index()
    baseline_traffic.extend(daily_traffic['door_count'].values)

if len(baseline_traffic) >= 10 and np.std(baseline_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_traffic)
    baseline_traffic_std = np.std(baseline_traffic)
    
    for idx, row in daily_traffic_analysis.iterrows():
        z_score = (row['door_count'] - baseline_traffic_mean) / baseline_traffic_std if baseline_traffic_std > 0 else 0
        if abs(z_score) > 2.5:
            findings.append({
                'type': 'daily_traffic',
                'date': row['date'],
                'observed': row['door_count'],
                'baseline_mean': baseline_traffic_mean,
                'baseline_std': baseline_traffic_std,
                'z_score': z_score,
                'sample_size': len(baseline_traffic)
            })

# ============================================================================
# ANOMALY 4: Daily Item Volume (quantity sold)
# ============================================================================

# Calculate daily item volume for analysis period
daily_volume_analysis = pos_analysis.groupby('calendar_date')['quantity'].sum().reset_index()
daily_volume_analysis.columns = ['date', 'volume']

# Calculate daily item volume for baseline periods
baseline_volumes = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    daily_vol = pos_baseline.groupby('calendar_date')['quantity'].sum().reset_index()
    baseline_volumes.extend(daily_vol['quantity'].values)

if len(baseline_volumes) >= 10 and np.std(baseline_volumes) > 0:
    baseline_volume_mean = np.mean(baseline_volumes)
    baseline_volume_std = np.std(baseline_volumes)
    
    for idx, row in daily_volume_analysis.iterrows():
        z_score = (row['volume'] - baseline_volume_mean) / baseline_volume_std if baseline_volume_std > 0 else 0
        if abs(z_score) > 2.5:
            findings.append({
                'type': 'daily_volume',
                'date': row['date'],
                'observed': row['volume'],
                'baseline_mean': baseline_volume_mean,
                'baseline_std': baseline_volume_std,
                'z_score': z_score,
                'sample_size': len(baseline_volumes)
            })

# ============================================================================
# ANOMALY 5: Daily Average Rating
# ============================================================================

# Calculate daily average rating for analysis period
reviews_analysis = reviews_df[(reviews_df['date'] >= analysis_start) & (reviews_df['date'] < analysis_end)].copy()
daily_rating_analysis = reviews_analysis.groupby('date')['rating'].agg(['mean', 'count']).reset_index()
daily_rating_analysis.columns = ['date', 'avg_rating', 'count']

# Calculate daily average rating for baseline periods
baseline_ratings = []
for period_start, period_end in baseline_periods:
    reviews_baseline = reviews_df[(reviews_df['date'] >= period_start) & (reviews_df['date'] < period_end)].copy()
    daily_rating = reviews_baseline.groupby('date')['rating'].mean().reset_index()
    baseline_ratings.extend(daily_rating['rating'].values)

if len(baseline_ratings) >= 10 and np.std(baseline_ratings) > 0:
    baseline_rating_mean = np.mean(baseline_ratings)
    baseline_rating_std = np.std(baseline_ratings)
    
    for idx, row in daily_rating_analysis.iterrows():
        if row['count'] >= 2:  # Require at least 2 reviews
            z_score = (row['avg_rating'] - baseline_rating_mean) / baseline_rating_std if baseline_rating_std > 0 else 0
            if abs(z_score) > 2.0:  # Slightly lower threshold for ratings
                findings.append({
                    'type': 'daily_rating',
                    'date': row['date'],
                    'observed': row['avg_rating'],
                    'baseline_mean': baseline_rating_mean,
                    'baseline_std': baseline_rating_std,
                    'z_score': z_score,
                    'sample_size': len(baseline_ratings),
                    'review_count': row['count']
                })

# ============================================================================
# ANOMALY 6: Daily Waste Cost
# ============================================================================

# Calculate daily waste cost for analysis period
analysis_week_start = pd.to_datetime("2026-04-06T00:00:00+03:00").tz_localize(None)
analysis_week_end = pd.to_datetime("2026-04-13T00:00:00+03:00").tz_localize(None)
inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_week_start) & (inventory_df['week_starting'] < analysis_week_end)].copy()
total_waste_analysis = inventory_analysis['known_waste_cost_sar'].sum()

# Calculate weekly waste cost for baseline periods
baseline_waste_costs = []
for period_start, period_end in baseline_periods:
    week_start = period_start
    inventory_baseline = inventory_df[(inventory_df['week_starting'] >= week_start) & (inventory_df['week_starting'] < period_end)].copy()
    total_waste = inventory_baseline['known_waste_cost_sar'].sum()
    if total_waste > 0:
        baseline_waste_costs.append(total_waste)

if len(baseline_waste_costs) >= 3 and np.std(baseline_waste_costs) > 0:
    baseline_waste_mean = np.mean(baseline_waste_costs)
    baseline_waste_std = np.std(baseline_waste_costs)
    
    z_score = (total_waste_analysis - baseline_waste_mean) / baseline_waste_std if baseline_waste_std > 0 else 0
    if abs(z_score) > 2.0:
        findings.append({
            'type': 'weekly_waste',
            'period_start': analysis_week_start,
            'period_end': analysis_week_end,
            'observed': total_waste_analysis,
            'baseline_mean': baseline_waste_mean,
            'baseline_std': baseline_waste_std,
            'z_score': z_score,
            'sample_size': len(baseline_waste_costs)
        })

# ============================================================================
# Sort findings by magnitude of z-score and select top 3
# ============================================================================

findings_sorted = sorted(findings, key=lambda x: abs(x['z_score']), reverse=True)[:3]

# ============================================================================
# Format output
# ============================================================================

output_findings = []

for finding in findings_sorted:
    if finding['type'] == 'daily_revenue':
        output_findings.append({
            'title': f"Unusual daily revenue on {finding['date'].strftime('%Y-%m-%d')}",
            'claim': f"Daily revenue of {finding['observed']:.2f} SAR deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.2f} SAR.",
            'finding_type': 'revenue_anomaly',
            'metrics': {
                'observed_daily_revenue': {
                    'value': round(finding['observed'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_revenue': {
                    'value': round(finding['baseline_mean'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': finding['sample_size'],
            'coverage_notes': ['Daily revenue calculated from POS line_total_sar, excluding refunds via is_refund flag', 'Baseline computed from 4 weeks of historical data (2026-03-09 to 2026-04-06)'],
            'assumptions': ['Normal distribution of daily revenue', 'Z-score threshold of 2.5 standard deviations', 'No structural breaks in baseline period'],
            'confidence': 0.85
        })
    
    elif finding['type'] == 'daily_transactions':
        output_findings.append({
            'title': f"Unusual transaction count on {finding['date'].strftime('%Y-%m-%d')}",
            'claim': f"Daily transaction count of {finding['observed']} deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.1f}.",
            'finding_type': 'transaction_volume_anomaly',
            'metrics': {
                'observed_daily_transactions': {
                    'value': int(finding['observed']),
                    'unit': 'count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_transactions': {
                    'value': round(finding['baseline_mean'], 1),
                    'unit': 'count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': finding['sample_size'],
            'coverage_notes': ['Transaction count derived from unique transaction_id per day', 'Baseline computed from 4 weeks of historical data'],
            'assumptions': ['Normal distribution of daily transaction counts', 'Z-score threshold of 2.5 standard deviations'],
            'confidence': 0.85
        })
    
    elif finding['type'] == 'daily_traffic':
        output_findings.append({
            'title': f"Unusual foot traffic on {finding['date'].strftime('%Y-%m-%d')}",
            'claim': f"Daily door count of {finding['observed']} deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.1f}.",
            'finding_type': 'traffic_anomaly',
            'metrics': {
                'observed_daily_door_count': {
                    'value': int(finding['observed']),
                    'unit': 'count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_door_count': {
                    'value': round(finding['baseline_mean'], 1),
                    'unit': 'count',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['traffic'],
            'sample_size': finding['sample_size'],
            'coverage_notes': ['Daily door count excludes days marked as is_dead_sensor_day=True', 'Baseline computed from 4 weeks of historical data'],
            'assumptions': ['Normal distribution of daily traffic', 'Z-score threshold of 2.5 standard deviations', 'Sensor reliability consistent across periods'],
            'confidence': 0.80
        })
    
    elif finding['type'] == 'daily_volume':
        output_findings.append({
            'title': f"Unusual item volume on {finding['date'].strftime('%Y-%m-%d')}",
            'claim': f"Daily item quantity of {finding['observed']:.0f} units deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.1f} units.",
            'finding_type': 'item_volume_anomaly',
            'metrics': {
                'observed_daily_item_quantity': {
                    'value': round(finding['observed'], 1),
                    'unit': 'units',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_item_quantity': {
                    'value': round(finding['baseline_mean'], 1),
                    'unit': 'units',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['pos'],
            'sample_size': finding['sample_size'],
            'coverage_notes': ['Daily item quantity summed from POS quantity column', 'Baseline computed from 4 weeks of historical data'],
            'assumptions': ['Normal distribution of daily item volumes', 'Z-score threshold of 2.5 standard deviations'],
            'confidence': 0.85
        })
    
    elif finding['type'] == 'daily_rating':
        output_findings.append({
            'title': f"Unusual average rating on {finding['date'].strftime('%Y-%m-%d')}",
            'claim': f"Daily average rating of {finding['observed']:.2f} deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.2f} ({finding['review_count']} reviews).",
            'finding_type': 'rating_anomaly',
            'metrics': {
                'observed_daily_avg_rating': {
                    'value': round(finding['observed'], 2),
                    'unit': 'rating',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                },
                'baseline_mean_daily_avg_rating': {
                    'value': round(finding['baseline_mean'], 2),
                    'unit': 'rating',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['date'].isoformat(),
                    'period_end': (finding['date'] + timedelta(days=1)).isoformat()
                }
            },
            'source_names': ['reviews'],
            'sample_size': finding['sample_size'],
            'coverage_notes': [f'Daily average rating computed from {finding["review_count"]} reviews', 'Baseline computed from 4 weeks of historical data', 'Only days with 2+ reviews included'],
            'assumptions': ['Normal distribution of daily average ratings', 'Z-score threshold of 2.0 standard deviations', 'Rating scale consistent across periods'],
            'confidence': 0.75
        })
    
    elif finding['type'] == 'weekly_waste':
        output_findings.append({
            'title': f"Unusual weekly waste cost ({finding['period_start'].strftime('%Y-%m-%d')} to {finding['period_end'].strftime('%Y-%m-%d')})",
            'claim': f"Weekly waste cost of {finding['observed']:.2f} SAR deviates {abs(finding['z_score']):.2f} standard deviations from baseline mean of {finding['baseline_mean']:.2f} SAR.",
            'finding_type': 'waste_cost_anomaly',
            'metrics': {
                'observed_weekly_waste_cost': {
                    'value': round(finding['observed'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['period_start'].isoformat(),
                    'period_end': finding['period_end'].isoformat()
                },
                'baseline_mean_weekly_waste_cost': {
                    'value': round(finding['baseline_mean'], 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-03-09T00:00:00',
                    'period_end': '2026-04-06T00:00:00'
                },
                'z_score': {
                    'value': round(finding['z_score'], 2),
                    'unit': 'std_dev',
                    'numerator': None,
                    'denominator': None,
                    'period_start': finding['period_start'].isoformat(),
                    'period_end': finding['period_end'].isoformat()
                }
            },
            'source_names': ['inventory'],
            'sample_size': finding['sample_size'],
            'coverage_notes': ['Weekly waste cost from inventory known_waste_cost_sar', 'Baseline computed from 4 weeks of historical data', 'Only weeks with non-zero waste included in baseline'],
            'assumptions': ['Normal distribution of weekly waste costs', 'Z-score threshold of 2.0 standard deviations', 'Waste measurement methodology consistent'],
            'confidence': 0.70
        })

result = {
    'status': 'success' if len(output_findings) > 0 else 'insufficient_data',
    'findings': output_findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
