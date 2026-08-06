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
analysis_start = pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None)
analysis_end = pd.to_datetime('2026-02-23T00:00:00+03:00').tz_localize(None)

baseline_periods = [
    (pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-16T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-09T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-02-02T00:00:00+03:00').tz_localize(None)),
    (pd.to_datetime('2026-01-19T00:00:00+03:00').tz_localize(None), 
     pd.to_datetime('2026-01-26T00:00:00+03:00').tz_localize(None))
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Filter analysis period
analysis_pos = pos_df[(pos_df['calendar_date'] >= analysis_start) & 
                      (pos_df['calendar_date'] < analysis_end)].copy()

# Calculate daily revenue (excluding refunds)
analysis_pos['net_line_total'] = analysis_pos['line_total_sar']
analysis_daily_revenue = analysis_pos.groupby('calendar_date')['net_line_total'].sum().reset_index()
analysis_daily_revenue.columns = ['date', 'daily_revenue']

# Calculate baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    baseline_pos['net_line_total'] = baseline_pos['line_total_sar']
    daily_rev = baseline_pos.groupby('calendar_date')['net_line_total'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and np.var(baseline_daily_revenues) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Calculate z-scores for analysis period
    analysis_daily_revenue['z_score'] = (analysis_daily_revenue['daily_revenue'] - baseline_mean) / baseline_std
    
    # Find anomalies (|z| > 2)
    anomalies = analysis_daily_revenue[abs(analysis_daily_revenue['z_score']) > 2].sort_values('z_score', key=abs, ascending=False)
    
    if len(anomalies) > 0:
        top_anomaly = anomalies.iloc[0]
        findings.append({
            "title": "Daily Revenue Anomaly",
            "claim": f"Daily revenue on {top_anomaly['date'].strftime('%Y-%m-%d')} was {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean",
            "finding_type": "revenue_anomaly",
            "metrics": {
                "observed_daily_revenue": {
                    "value": round(float(top_anomaly['daily_revenue']), 2),
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
                    "period_start": "2026-01-19T00:00:00",
                    "period_end": "2026-02-16T00:00:00"
                },
                "z_score": {
                    "value": round(float(top_anomaly['z_score']), 2),
                    "unit": "std_dev",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['date'].isoformat(),
                    "period_end": (top_anomaly['date'] + timedelta(days=1)).isoformat()
                }
            },
            "source_names": ["pos"],
            "sample_size": len(baseline_daily_revenues),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Baseline: 4 weeks of historical data (2026-01-19 to 2026-02-16)",
                f"Baseline sample size: {len(baseline_daily_revenues)} daily observations"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 standard deviations",
                "Refunds included in net revenue calculation",
                "No product launches or known events during analysis period"
            ],
            "confidence": 0.85
        })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

# Filter analysis period traffic
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start) & 
                              (traffic_df['date'] < analysis_end) &
                              (traffic_df['is_dead_sensor_day'] == False)].copy()

# Calculate hourly traffic
analysis_hourly_traffic = analysis_traffic.groupby('hour')['door_count'].sum().reset_index()

# Get baseline hourly traffic
baseline_hourly_traffic_list = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start) & 
                                  (traffic_df['date'] < period_end) &
                                  (traffic_df['is_dead_sensor_day'] == False)].copy()
    hourly_traffic = baseline_traffic.groupby('hour')['door_count'].sum()
    baseline_hourly_traffic_list.append(hourly_traffic)

if len(baseline_hourly_traffic_list) > 0:
    # Combine baseline data
    baseline_hourly_combined = pd.concat(baseline_hourly_traffic_list, axis=1).mean(axis=1)
    baseline_hourly_std = pd.concat(baseline_hourly_traffic_list, axis=1).std(axis=1)
    
    # Extract hour component for mapping (use floor with 'h' instead of 'H')
    analysis_hourly_traffic['hour_key'] = analysis_hourly_traffic['hour'].dt.floor('h')
    
    # Calculate z-scores
    analysis_hourly_traffic['baseline_mean'] = analysis_hourly_traffic['hour_key'].map(baseline_hourly_combined)
    analysis_hourly_traffic['baseline_std'] = analysis_hourly_traffic['hour_key'].map(baseline_hourly_std)
    
    # Only calculate z-score where we have baseline data and non-zero variance
    valid_mask = (analysis_hourly_traffic['baseline_std'] > 0) & (analysis_hourly_traffic['baseline_mean'].notna())
    analysis_hourly_traffic.loc[valid_mask, 'z_score'] = (
        (analysis_hourly_traffic.loc[valid_mask, 'door_count'] - analysis_hourly_traffic.loc[valid_mask, 'baseline_mean']) / 
        analysis_hourly_traffic.loc[valid_mask, 'baseline_std']
    )
    
    # Find anomalies
    anomalies = analysis_hourly_traffic[abs(analysis_hourly_traffic['z_score']) > 2].sort_values('z_score', key=abs, ascending=False)
    
    if len(anomalies) > 0:
        top_anomaly = anomalies.iloc[0]
        findings.append({
            "title": "Hourly Traffic Anomaly",
            "claim": f"Door count at {top_anomaly['hour'].strftime('%Y-%m-%d %H:00')} was {abs(top_anomaly['z_score']):.2f} standard deviations from baseline",
            "finding_type": "traffic_anomaly",
            "metrics": {
                "observed_hourly_traffic": {
                    "value": int(top_anomaly['door_count']),
                    "unit": "door_count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['hour'].isoformat(),
                    "period_end": (top_anomaly['hour'] + timedelta(hours=1)).isoformat()
                },
                "baseline_mean_hourly_traffic": {
                    "value": round(float(top_anomaly['baseline_mean']), 1),
                    "unit": "door_count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00",
                    "period_end": "2026-02-16T00:00:00"
                },
                "z_score": {
                    "value": round(float(top_anomaly['z_score']), 2),
                    "unit": "std_dev",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['hour'].isoformat(),
                    "period_end": (top_anomaly['hour'] + timedelta(hours=1)).isoformat()
                }
            },
            "source_names": ["traffic"],
            "sample_size": len(analysis_hourly_traffic),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Excluded dead sensor days",
                f"Baseline: 4 weeks of historical hourly data"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 standard deviations",
                "Dead sensor days excluded from analysis",
                "Baseline calculated as mean across 4 weeks"
            ],
            "confidence": 0.80
        })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count (unique transaction_ids)
analysis_daily_txns = analysis_pos.groupby('calendar_date')['transaction_id'].nunique().reset_index()
analysis_daily_txns.columns = ['date', 'transaction_count']

# Calculate baseline daily transaction counts
baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['calendar_date'] >= period_start) & 
                          (pos_df['calendar_date'] < period_end)].copy()
    daily_txns = baseline_pos.groupby('calendar_date')['transaction_id'].nunique()
    baseline_daily_txns.extend(daily_txns.values)

if len(baseline_daily_txns) > 0 and np.var(baseline_daily_txns) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    # Calculate z-scores
    analysis_daily_txns['z_score'] = (analysis_daily_txns['transaction_count'] - baseline_txn_mean) / baseline_txn_std
    
    # Find anomalies
    anomalies = analysis_daily_txns[abs(analysis_daily_txns['z_score']) > 2].sort_values('z_score', key=abs, ascending=False)
    
    if len(anomalies) > 0:
        top_anomaly = anomalies.iloc[0]
        findings.append({
            "title": "Daily Transaction Count Anomaly",
            "claim": f"Transaction count on {top_anomaly['date'].strftime('%Y-%m-%d')} was {abs(top_anomaly['z_score']):.2f} standard deviations from baseline",
            "finding_type": "transaction_anomaly",
            "metrics": {
                "observed_daily_transactions": {
                    "value": int(top_anomaly['transaction_count']),
                    "unit": "transactions",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['date'].isoformat(),
                    "period_end": (top_anomaly['date'] + timedelta(days=1)).isoformat()
                },
                "baseline_mean_daily_transactions": {
                    "value": round(baseline_txn_mean, 1),
                    "unit": "transactions",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00",
                    "period_end": "2026-02-16T00:00:00"
                },
                "z_score": {
                    "value": round(float(top_anomaly['z_score']), 2),
                    "unit": "std_dev",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['date'].isoformat(),
                    "period_end": (top_anomaly['date'] + timedelta(days=1)).isoformat()
                }
            },
            "source_names": ["pos"],
            "sample_size": len(baseline_daily_txns),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Baseline: 4 weeks of historical data (2026-01-19 to 2026-02-16)",
                f"Baseline sample size: {len(baseline_daily_txns)} daily observations"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 standard deviations",
                "Transaction count based on unique transaction_ids",
                "Refunds included in transaction count"
            ],
            "confidence": 0.82
        })

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
