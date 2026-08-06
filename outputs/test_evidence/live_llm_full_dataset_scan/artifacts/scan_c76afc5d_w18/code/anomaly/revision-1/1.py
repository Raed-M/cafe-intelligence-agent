import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow.parquet as pq

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define periods
analysis_period = {
    "start": datetime.fromisoformat("2026-05-11T00:00:00+03:00"),
    "end": datetime.fromisoformat("2026-05-18T00:00:00+03:00")
}

previous_period = {
    "start": datetime.fromisoformat("2026-05-04T00:00:00+03:00"),
    "end": datetime.fromisoformat("2026-05-11T00:00:00+03:00")
}

trailing_baseline_periods = [
    {
        "start": datetime.fromisoformat("2026-05-04T00:00:00+03:00"),
        "end": datetime.fromisoformat("2026-05-11T00:00:00+03:00")
    },
    {
        "start": datetime.fromisoformat("2026-04-27T00:00:00+03:00"),
        "end": datetime.fromisoformat("2026-05-04T00:00:00+03:00")
    },
    {
        "start": datetime.fromisoformat("2026-04-20T00:00:00+03:00"),
        "end": datetime.fromisoformat("2026-04-27T00:00:00+03:00")
    },
    {
        "start": datetime.fromisoformat("2026-04-13T00:00:00+03:00"),
        "end": datetime.fromisoformat("2026-04-20T00:00:00+03:00")
    }
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
# ANOMALY 1: Daily Revenue Anomaly Detection
# ============================================================================

# Calculate daily revenue for analysis period and baseline periods
def get_daily_revenue(df, period_start, period_end):
    """Calculate daily net revenue (excluding refunds from totals)"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    # Net revenue: sum of line_total_sar (which includes refunds as negative)
    daily_revenue = period_data.groupby('date')['line_total_sar'].sum()
    return daily_revenue

# Get analysis period daily revenue
analysis_daily_revenue = get_daily_revenue(pos_df, analysis_period['start'], analysis_period['end'])

# Get baseline daily revenues
baseline_daily_revenues = []
for period in trailing_baseline_periods:
    daily_rev = get_daily_revenue(pos_df, period['start'], period['end'])
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalies in analysis period
        anomalies = []
        for date, revenue in analysis_daily_revenue.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2-sigma threshold
                anomalies.append({
                    'date': date,
                    'revenue': revenue,
                    'z_score': z_score,
                    'baseline_mean': baseline_mean
                })
        
        # Sort by magnitude
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        if anomalies:
            top_anomaly = anomalies[0]
            findings.append({
                "title": "Daily Revenue Anomaly",
                "claim": f"Daily revenue on {top_anomaly['date']} was {top_anomaly['revenue']:.2f} SAR, {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "observed_daily_revenue": {
                        "value": round(top_anomaly['revenue'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_period['start'].isoformat(),
                        "period_end": analysis_period['end'].isoformat()
                    },
                    "baseline_mean_daily_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": trailing_baseline_periods[0]['start'].isoformat(),
                        "period_end": trailing_baseline_periods[-1]['end'].isoformat()
                    },
                    "z_score_daily_revenue": {
                        "value": round(top_anomaly['z_score'], 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_period['start'].isoformat(),
                        "period_end": analysis_period['end'].isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenues),
                "coverage_notes": [
                    f"Analysis period: {analysis_period['start'].date()} to {analysis_period['end'].date()}",
                    f"Baseline: {len(baseline_daily_revenues)} daily observations from 4 trailing weeks",
                    f"Threshold: 2.0 sigma"
                ],
                "assumptions": [
                    "Daily revenue calculated as sum of line_total_sar (net of refunds)",
                    "Baseline computed from 4 trailing weeks",
                    "Normal distribution assumed for z-score calculation"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Anomaly Detection
# ============================================================================

# Calculate hourly traffic for analysis period and baseline
def get_hourly_traffic(df, period_start, period_end):
    """Calculate hourly door counts"""
    mask = (df['date'] >= period_start.date()) & (df['date'] < period_end.date())
    mask = mask & (df['is_dead_sensor_day'] == False)
    period_data = df[mask].copy()
    
    # Group by hour
    hourly_traffic = period_data.groupby('hour')['door_count'].mean()
    return hourly_traffic

# Get analysis period hourly traffic
analysis_hourly_traffic = get_hourly_traffic(traffic_df, analysis_period['start'], analysis_period['end'])

# Get baseline hourly traffic
baseline_hourly_traffic_list = []
for period in trailing_baseline_periods:
    hourly_traffic = get_hourly_traffic(traffic_df, period['start'], period['end'])
    baseline_hourly_traffic_list.append(hourly_traffic)

if len(baseline_hourly_traffic_list) > 0 and len(analysis_hourly_traffic) > 0:
    # Combine baseline hourly data
    baseline_hourly_combined = pd.concat(baseline_hourly_traffic_list)
    baseline_hourly_mean = baseline_hourly_combined.groupby(baseline_hourly_combined.index).mean()
    baseline_hourly_std = baseline_hourly_combined.groupby(baseline_hourly_combined.index).std()
    
    # Find anomalies
    anomalies = []
    for hour in analysis_hourly_traffic.index:
        if hour in baseline_hourly_mean.index and baseline_hourly_std[hour] > 0:
            observed = analysis_hourly_traffic[hour]
            expected = baseline_hourly_mean[hour]
            std = baseline_hourly_std[hour]
            z_score = (observed - expected) / std
            
            if abs(z_score) > 2.0:
                anomalies.append({
                    'hour': hour,
                    'observed': observed,
                    'expected': expected,
                    'z_score': z_score
                })
    
    anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    if anomalies:
        top_anomaly = anomalies[0]
        findings.append({
            "title": "Hourly Traffic Anomaly",
            "claim": f"Average hourly door count at hour {int(top_anomaly['hour'])} was {top_anomaly['observed']:.1f}, {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {top_anomaly['expected']:.1f}.",
            "finding_type": "traffic_anomaly",
            "metrics": {
                "observed_hourly_door_count": {
                    "value": round(top_anomaly['observed'], 1),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'].isoformat(),
                    "period_end": analysis_period['end'].isoformat()
                },
                "baseline_mean_hourly_door_count": {
                    "value": round(top_anomaly['expected'], 1),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": trailing_baseline_periods[0]['start'].isoformat(),
                    "period_end": trailing_baseline_periods[-1]['end'].isoformat()
                },
                "z_score_hourly_traffic": {
                    "value": round(top_anomaly['z_score'], 2),
                    "unit": "std_dev",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'].isoformat(),
                    "period_end": analysis_period['end'].isoformat()
                }
            },
            "source_names": ["traffic"],
            "sample_size": len(baseline_hourly_combined),
            "coverage_notes": [
                f"Analysis period: {analysis_period['start'].date()} to {analysis_period['end'].date()}",
                f"Baseline: hourly averages from 4 trailing weeks",
                f"Dead sensor days excluded",
                f"Threshold: 2.0 sigma"
            ],
            "assumptions": [
                "Hourly traffic calculated as mean door_count per hour across valid days",
                "Dead sensor days (is_dead_sensor_day=True) excluded from analysis",
                "Normal distribution assumed for z-score calculation"
            ],
            "confidence": 0.80
        })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Anomaly
# ============================================================================

def get_daily_transaction_count(df, period_start, period_end):
    """Calculate daily unique transaction count"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    # Count unique transaction_ids per day
    daily_txn = period_data.groupby('date')['transaction_id'].nunique()
    return daily_txn

# Get analysis period daily transactions
analysis_daily_txn = get_daily_transaction_count(pos_df, analysis_period['start'], analysis_period['end'])

# Get baseline daily transactions
baseline_daily_txn = []
for period in trailing_baseline_periods:
    daily_txn = get_daily_transaction_count(pos_df, period['start'], period['end'])
    baseline_daily_txn.extend(daily_txn.values)

if len(baseline_daily_txn) > 0 and len(analysis_daily_txn) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txn)
    baseline_txn_std = np.std(baseline_daily_txn)
    
    if baseline_txn_std > 0:
        # Find anomalies
        anomalies = []
        for date, txn_count in analysis_daily_txn.items():
            z_score = (txn_count - baseline_txn_mean) / baseline_txn_std
            if abs(z_score) > 2.0:
                anomalies.append({
                    'date': date,
                    'txn_count': txn_count,
                    'z_score': z_score,
                    'baseline_mean': baseline_txn_mean
                })
        
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        if anomalies:
            top_anomaly = anomalies[0]
            findings.append({
                "title": "Daily Transaction Count Anomaly",
                "claim": f"Daily transaction count on {top_anomaly['date']} was {int(top_anomaly['txn_count'])}, {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_txn_mean:.1f}.",
                "finding_type": "transaction_volume_anomaly",
                "metrics": {
                    "observed_daily_transaction_count": {
                        "value": int(top_anomaly['txn_count']),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_period['start'].isoformat(),
                        "period_end": analysis_period['end'].isoformat()
                    },
                    "baseline_mean_daily_transaction_count": {
                        "value": round(baseline_txn_mean, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": trailing_baseline_periods[0]['start'].isoformat(),
                        "period_end": trailing_baseline_periods[-1]['end'].isoformat()
                    },
                    "z_score_daily_transactions": {
                        "value": round(top_anomaly['z_score'], 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_period['start'].isoformat(),
                        "period_end": analysis_period['end'].isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_txn),
                "coverage_notes": [
                    f"Analysis period: {analysis_period['start'].date()} to {analysis_period['end'].date()}",
                    f"Baseline: {len(baseline_daily_txn)} daily observations from 4 trailing weeks",
                    f"Threshold: 2.0 sigma"
                ],
                "assumptions": [
                    "Daily transaction count calculated as unique transaction_id count per day",
                    "Baseline computed from 4 trailing weeks",
                    "Normal distribution assumed for z-score calculation"
                ],
                "confidence": 0.82
            })

# ============================================================================
# Output Result
# ============================================================================

result = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)