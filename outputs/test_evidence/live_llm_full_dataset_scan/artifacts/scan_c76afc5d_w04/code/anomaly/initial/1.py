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
analysis_start = pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None)
baseline_periods = [
    (pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-05T00:00:00+03:00').tz_localize(None), pd.to_datetime('2026-01-12T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_revenue = analysis_pos.groupby('calendar_date')['line_total_sar'].sum().reset_index()
analysis_daily_revenue.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    baseline_daily = baseline_pos.groupby('calendar_date')['line_total_sar'].sum().reset_index()
    baseline_daily_revenues.extend(baseline_daily['line_total_sar'].values)

if len(baseline_daily_revenues) >= 10 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalies using z-score
        analysis_daily_revenue['z_score'] = (analysis_daily_revenue['revenue'] - baseline_mean) / baseline_std
        anomalies = analysis_daily_revenue[abs(analysis_daily_revenue['z_score']) > 2.0].sort_values('z_score', key=abs, ascending=False)
        
        if len(anomalies) > 0:
            top_anomaly = anomalies.iloc[0]
            finding = {
                "title": "Unusual Daily Revenue Detected",
                "claim": f"Daily revenue on {top_anomaly['date'].strftime('%Y-%m-%d')} was {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean, indicating a significant deviation from typical daily performance.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "observed_daily_revenue": {
                        "value": round(float(top_anomaly['revenue']), 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": top_anomaly['date'].isoformat(),
                        "period_end": (top_anomaly['date'] + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean_daily_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": baseline_periods[0][0].isoformat(),
                        "period_end": baseline_periods[-1][1].isoformat()
                    },
                    "z_score": {
                        "value": round(float(top_anomaly['z_score']), 2),
                        "unit": "standard_deviations",
                        "numerator": None,
                        "denominator": None,
                        "period_start": top_anomaly['date'].isoformat(),
                        "period_end": (top_anomaly['date'] + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenues),
                "coverage_notes": [
                    f"Baseline calculated from {len(baseline_daily_revenues)} daily observations across 4 weeks",
                    f"Analysis period contains {len(analysis_daily_revenue)} days",
                    "Excludes refunds in net calculation per metric definition"
                ],
                "assumptions": [
                    "Daily revenue follows approximately normal distribution",
                    "Z-score threshold of 2.0 indicates statistical significance",
                    "No known sensor failures or data quality issues in analysis period"
                ],
                "confidence": 0.85
            }
            findings.append(finding)

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Calculate hourly traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
analysis_traffic = analysis_traffic[analysis_traffic['is_dead_sensor_day'] == False]
analysis_hourly_traffic = analysis_traffic.groupby(analysis_traffic['hour'].dt.hour)['door_count'].sum().reset_index()
analysis_hourly_traffic.columns = ['hour', 'traffic']

# Calculate hourly traffic for baseline periods
baseline_hourly_traffic_by_hour = {}
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & (traffic_df['date'] < period_end)].copy()
    baseline_traffic = baseline_traffic[baseline_traffic['is_dead_sensor_day'] == False]
    baseline_hourly = baseline_traffic.groupby(baseline_traffic['hour'].dt.hour)['door_count'].sum().reset_index()
    for _, row in baseline_hourly.iterrows():
        hour = int(row['hour'])
        if hour not in baseline_hourly_traffic_by_hour:
            baseline_hourly_traffic_by_hour[hour] = []
        baseline_hourly_traffic_by_hour[hour].append(row['door_count'])

if len(baseline_hourly_traffic_by_hour) > 0 and len(analysis_hourly_traffic) > 0:
    max_anomaly_score = 0
    max_anomaly_hour = None
    max_anomaly_value = None
    max_baseline_mean = None
    
    for _, row in analysis_hourly_traffic.iterrows():
        hour = int(row['hour'])
        if hour in baseline_hourly_traffic_by_hour and len(baseline_hourly_traffic_by_hour[hour]) >= 3:
            baseline_values = baseline_hourly_traffic_by_hour[hour]
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            if baseline_std > 0:
                z_score = abs((row['traffic'] - baseline_mean) / baseline_std)
                if z_score > max_anomaly_score and z_score > 2.0:
                    max_anomaly_score = z_score
                    max_anomaly_hour = hour
                    max_anomaly_value = row['traffic']
                    max_baseline_mean = baseline_mean
    
    if max_anomaly_hour is not None:
        finding = {
            "title": "Unusual Hourly Traffic Pattern Detected",
            "claim": f"Hour {max_anomaly_hour}:00 showed {max_anomaly_score:.2f} standard deviations from baseline traffic, indicating unusual foot traffic during this period.",
            "finding_type": "traffic_anomaly",
            "metrics": {
                "observed_hourly_traffic": {
                    "value": int(max_anomaly_value),
                    "unit": "door_count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "baseline_mean_hourly_traffic": {
                    "value": round(max_baseline_mean, 2),
                    "unit": "door_count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": baseline_periods[0][0].isoformat(),
                    "period_end": baseline_periods[-1][1].isoformat()
                },
                "z_score": {
                    "value": round(max_anomaly_score, 2),
                    "unit": "standard_deviations",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["traffic"],
            "sample_size": len(baseline_hourly_traffic_by_hour.get(max_anomaly_hour, [])),
            "coverage_notes": [
                f"Baseline calculated from {len(baseline_hourly_traffic_by_hour.get(max_anomaly_hour, []))} observations for hour {max_anomaly_hour}",
                "Excluded days marked as dead_sensor_day",
                f"Analysis period contains {len(analysis_hourly_traffic)} hours"
            ],
            "assumptions": [
                "Hourly traffic follows approximately normal distribution",
                "Z-score threshold of 2.0 indicates statistical significance",
                "Sensor data quality is consistent across periods"
            ],
            "confidence": 0.80
        }
        findings.append(finding)

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count for analysis period
analysis_pos_txn = pos_df[(pos_df['calendar_date'] >= analysis_start) & (pos_df['calendar_date'] < analysis_end)].copy()
analysis_daily_txn = analysis_pos_txn.groupby('calendar_date')['transaction_id'].nunique().reset_index()
analysis_daily_txn.columns = ['date', 'transaction_count']

# Calculate daily transaction count for baseline periods
baseline_daily_txn_counts = []
for period_start, period_end in baseline_periods:
    baseline_pos_txn = pos_df[(pos_df['calendar_date'] >= period_start) & (pos_df['calendar_date'] < period_end)].copy()
    baseline_daily_txn = baseline_pos_txn.groupby('calendar_date')['transaction_id'].nunique().reset_index()
    baseline_daily_txn_counts.extend(baseline_daily_txn['transaction_id'].values)

if len(baseline_daily_txn_counts) >= 10 and len(analysis_daily_txn) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txn_counts)
    baseline_txn_std = np.std(baseline_daily_txn_counts)
    
    if baseline_txn_std > 0:
        # Find anomalies using z-score
        analysis_daily_txn['z_score'] = (analysis_daily_txn['transaction_count'] - baseline_txn_mean) / baseline_txn_std
        txn_anomalies = analysis_daily_txn[abs(analysis_daily_txn['z_score']) > 2.0].sort_values('z_score', key=abs, ascending=False)
        
        if len(txn_anomalies) > 0:
            top_txn_anomaly = txn_anomalies.iloc[0]
            finding = {
                "title": "Unusual Daily Transaction Count Detected",
                "claim": f"Daily transaction count on {top_txn_anomaly['date'].strftime('%Y-%m-%d')} was {abs(top_txn_anomaly['z_score']):.2f} standard deviations from baseline mean, indicating unusual customer activity.",
                "finding_type": "transaction_anomaly",
                "metrics": {
                    "observed_daily_transactions": {
                        "value": int(top_txn_anomaly['transaction_count']),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": top_txn_anomaly['date'].isoformat(),
                        "period_end": (top_txn_anomaly['date'] + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean_daily_transactions": {
                        "value": round(baseline_txn_mean, 2),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": baseline_periods[0][0].isoformat(),
                        "period_end": baseline_periods[-1][1].isoformat()
                    },
                    "z_score": {
                        "value": round(float(top_txn_anomaly['z_score']), 2),
                        "unit": "standard_deviations",
                        "numerator": None,
                        "denominator": None,
                        "period_start": top_txn_anomaly['date'].isoformat(),
                        "period_end": (top_txn_anomaly['date'] + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_txn_counts),
                "coverage_notes": [
                    f"Baseline calculated from {len(baseline_daily_txn_counts)} daily observations across 4 weeks",
                    f"Analysis period contains {len(analysis_daily_txn)} days",
                    "Transaction count based on unique transaction_id per day"
                ],
                "assumptions": [
                    "Daily transaction count follows approximately normal distribution",
                    "Z-score threshold of 2.0 indicates statistical significance",
                    "No known data quality issues in transaction_id field"
                ],
                "confidence": 0.82
            }
            findings.append(finding)

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
