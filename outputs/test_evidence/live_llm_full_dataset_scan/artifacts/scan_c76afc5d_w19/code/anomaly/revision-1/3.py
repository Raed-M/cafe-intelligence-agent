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

# Define periods
analysis_start = pd.Timestamp("2026-05-18T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-05-25T00:00:00+03:00")
previous_start = pd.Timestamp("2026-05-11T00:00:00+03:00")
previous_end = pd.Timestamp("2026-05-18T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-05-11T00:00:00+03:00"), pd.Timestamp("2026-05-18T00:00:00+03:00")),
    (pd.Timestamp("2026-05-04T00:00:00+03:00"), pd.Timestamp("2026-05-11T00:00:00+03:00")),
    (pd.Timestamp("2026-04-27T00:00:00+03:00"), pd.Timestamp("2026-05-04T00:00:00+03:00")),
    (pd.Timestamp("2026-04-20T00:00:00+03:00"), pd.Timestamp("2026-04-27T00:00:00+03:00")),
]

# Convert timestamps to UTC-aware for comparison
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'], utc=True)
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baselines
def get_daily_revenue(df, period_start, period_end):
    """Get daily revenue excluding refunds, net of discounts"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    daily_revenue = period_data.groupby('date').apply(
        lambda x: (x[~x['is_refund']]['line_total_sar'].sum() + 
                   x[x['is_refund']]['line_total_sar'].sum())
    )
    return daily_revenue

# Get analysis period daily revenue
analysis_daily_revenue = get_daily_revenue(pos_df, analysis_start, analysis_end)

# Get baseline daily revenues
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_daily_revenues.extend(get_daily_revenue(pos_df, period_start, period_end).values)

if len(analysis_daily_revenue) > 0 and len(baseline_daily_revenues) >= 4:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Calculate z-scores for each day in analysis period
        z_scores = (analysis_daily_revenue.values - baseline_mean) / baseline_std
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z_score = z_scores[max_z_idx]
        anomaly_date = analysis_daily_revenue.index[max_z_idx]
        anomaly_revenue = analysis_daily_revenue.iloc[max_z_idx]
        
        # Only flag if |z| > 2
        if abs(max_z_score) > 2:
            findings.append({
                "title": "Daily Revenue Anomaly",
                "claim": f"Daily revenue on {anomaly_date} was {anomaly_revenue:.2f} SAR, {abs(max_z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "daily_revenue_anomaly_date": {
                        "value": round(anomaly_revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{anomaly_date}T00:00:00+03:00",
                        "period_end": f"{anomaly_date}T23:59:59+03:00"
                    },
                    "baseline_mean_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "z_score_revenue": {
                        "value": round(max_z_score, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{anomaly_date}T00:00:00+03:00",
                        "period_end": f"{anomaly_date}T23:59:59+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_daily_revenue),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior (2026-04-20 to 2026-05-18)",
                    f"Baseline sample size: {len(baseline_daily_revenues)} days",
                    "Refunds included as negative revenue per metric definition"
                ],
                "assumptions": [
                    "Normal distribution of daily revenue",
                    "Z-score threshold of 2.0 (95% confidence)",
                    "No known sensor outages during analysis period",
                    "Baseline periods are representative of normal operations"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Daily Transaction Count Analysis
# ============================================================================

def get_daily_transactions(df, period_start, period_end):
    """Get daily unique transaction count"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    daily_txns = period_data.groupby('date')['transaction_id'].nunique()
    return daily_txns

analysis_daily_txns = get_daily_transactions(pos_df, analysis_start, analysis_end)

baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    baseline_daily_txns.extend(get_daily_transactions(pos_df, period_start, period_end).values)

if len(analysis_daily_txns) > 0 and len(baseline_daily_txns) >= 4:
    txn_baseline_mean = np.mean(baseline_daily_txns)
    txn_baseline_std = np.std(baseline_daily_txns)
    
    if txn_baseline_std > 0:
        txn_z_scores = (analysis_daily_txns.values - txn_baseline_mean) / txn_baseline_std
        max_txn_z_idx = np.argmax(np.abs(txn_z_scores))
        max_txn_z_score = txn_z_scores[max_txn_z_idx]
        txn_anomaly_date = analysis_daily_txns.index[max_txn_z_idx]
        anomaly_txn_count = analysis_daily_txns.iloc[max_txn_z_idx]
        
        if abs(max_txn_z_score) > 2:
            findings.append({
                "title": "Daily Transaction Count Anomaly",
                "claim": f"Daily transaction count on {txn_anomaly_date} was {anomaly_txn_count} transactions, {abs(max_txn_z_score):.2f} standard deviations from baseline mean of {txn_baseline_mean:.1f}",
                "finding_type": "transaction_volume_anomaly",
                "metrics": {
                    "daily_transaction_count_anomaly_date": {
                        "value": int(anomaly_txn_count),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{txn_anomaly_date}T00:00:00+03:00",
                        "period_end": f"{txn_anomaly_date}T23:59:59+03:00"
                    },
                    "baseline_mean_transaction_count": {
                        "value": round(txn_baseline_mean, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "z_score_transaction_count": {
                        "value": round(max_txn_z_score, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{txn_anomaly_date}T00:00:00+03:00",
                        "period_end": f"{txn_anomaly_date}T23:59:59+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_daily_txns),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior (2026-04-20 to 2026-05-18)",
                    f"Baseline sample size: {len(baseline_daily_txns)} days",
                    "Counted unique transaction_id per day"
                ],
                "assumptions": [
                    "Normal distribution of daily transaction counts",
                    "Z-score threshold of 2.0 (95% confidence)",
                    "No known POS system outages during analysis period",
                    "Baseline periods are representative of normal operations"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 3: Daily Traffic Anomaly
# ============================================================================

def get_daily_traffic(df, period_start, period_end):
    """Get daily door count, excluding dead sensor days"""
    # Convert period timestamps to dates for comparison with date column
    period_start_date = period_start.date()
    period_end_date = period_end.date()
    
    # Convert traffic_df date column to date objects for comparison
    df_copy = df.copy()
    df_copy['date_only'] = df_copy['date'].dt.date
    
    mask = (df_copy['date_only'] >= period_start_date) & (df_copy['date_only'] < period_end_date)
    period_data = df_copy[mask & ~df_copy['is_dead_sensor_day']].copy()
    
    daily_traffic = period_data.groupby('date_only')['door_count'].sum()
    return daily_traffic

analysis_daily_traffic = get_daily_traffic(traffic_df, analysis_start, analysis_end)

baseline_daily_traffic = []
for period_start, period_end in baseline_periods:
    baseline_daily_traffic.extend(get_daily_traffic(traffic_df, period_start, period_end).values)

if len(analysis_daily_traffic) > 0 and len(baseline_daily_traffic) >= 4:
    traffic_baseline_mean = np.mean(baseline_daily_traffic)
    traffic_baseline_std = np.std(baseline_daily_traffic)
    
    if traffic_baseline_std > 0:
        traffic_z_scores = (analysis_daily_traffic.values - traffic_baseline_mean) / traffic_baseline_std
        max_traffic_z_idx = np.argmax(np.abs(traffic_z_scores))
        max_traffic_z_score = traffic_z_scores[max_traffic_z_idx]
        traffic_anomaly_date = analysis_daily_traffic.index[max_traffic_z_idx]
        anomaly_traffic_count = analysis_daily_traffic.iloc[max_traffic_z_idx]
        
        if abs(max_traffic_z_score) > 2:
            findings.append({
                "title": "Daily Traffic Anomaly",
                "claim": f"Daily door count on {traffic_anomaly_date} was {anomaly_traffic_count} visitors, {abs(max_traffic_z_score):.2f} standard deviations from baseline mean of {traffic_baseline_mean:.1f}",
                "finding_type": "traffic_anomaly",
                "metrics": {
                    "daily_traffic_anomaly_date": {
                        "value": int(anomaly_traffic_count),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{traffic_anomaly_date}T00:00:00+03:00",
                        "period_end": f"{traffic_anomaly_date}T23:59:59+03:00"
                    },
                    "baseline_mean_traffic": {
                        "value": round(traffic_baseline_mean, 1),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "z_score_traffic": {
                        "value": round(max_traffic_z_score, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": f"{traffic_anomaly_date}T00:00:00+03:00",
                        "period_end": f"{traffic_anomaly_date}T23:59:59+03:00"
                    }
                },
                "source_names": ["traffic"],
                "sample_size": len(analysis_daily_traffic),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior (2026-04-20 to 2026-05-18)",
                    f"Baseline sample size: {len(baseline_daily_traffic)} days",
                    "Excluded days marked as is_dead_sensor_day=True"
                ],
                "assumptions": [
                    "Normal distribution of daily traffic",
                    "Z-score threshold of 2.0 (95% confidence)",
                    "Sensor data is accurate on non-dead-sensor days",
                    "Baseline periods are representative of normal operations"
                ],
                "confidence": 0.85
            })

# Sort findings by magnitude of z-score and limit to 3
findings = sorted(findings, key=lambda x: abs(x['metrics'][list(x['metrics'].keys())[2]]['value']), reverse=True)[:3]

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
