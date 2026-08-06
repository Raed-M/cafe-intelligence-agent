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

# Define analysis periods
analysis_period = {
    "start": "2026-01-26T00:00:00+03:00",
    "end": "2026-02-02T00:00:00+03:00"
}
previous_period = {
    "start": "2026-01-19T00:00:00+03:00",
    "end": "2026-01-26T00:00:00+03:00"
}
trailing_baseline_periods = [
    {
        "start": "2026-01-19T00:00:00+03:00",
        "end": "2026-01-26T00:00:00+03:00"
    },
    {
        "start": "2026-01-12T00:00:00+03:00",
        "end": "2026-01-19T00:00:00+03:00"
    },
    {
        "start": "2026-01-05T00:00:00+03:00",
        "end": "2026-01-12T00:00:00+03:00"
    },
    {
        "start": "2025-12-29T00:00:00+03:00",
        "end": "2026-01-05T00:00:00+03:00"
    }
]

# Parse dates
def parse_iso_date(date_str):
    return pd.to_datetime(date_str)

analysis_start = parse_iso_date(analysis_period["start"])
analysis_end = parse_iso_date(analysis_period["end"])
previous_start = parse_iso_date(previous_period["start"])
previous_end = parse_iso_date(previous_period["end"])

baseline_periods = []
for period in trailing_baseline_periods:
    baseline_periods.append({
        "start": parse_iso_date(period["start"]),
        "end": parse_iso_date(period["end"])
    })

# Read data
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

pos_analysis['date'] = pos_analysis['timestamp_local'].dt.date
daily_revenue_analysis = pos_analysis.groupby('date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']
daily_revenue_analysis['date'] = pd.to_datetime(daily_revenue_analysis['date'])

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp_local'] >= period['start']) & 
        (pos_df['timestamp_local'] < period['end'])
    ].copy()
    pos_baseline['date'] = pos_baseline['timestamp_local'].dt.date
    daily_rev = pos_baseline.groupby('date')['line_total_sar'].sum().reset_index()
    daily_rev.columns = ['date', 'revenue']
    daily_rev['date'] = pd.to_datetime(daily_rev['date'])
    baseline_daily_revenues.extend(daily_rev['revenue'].values)

if len(baseline_daily_revenues) > 0 and len(daily_revenue_analysis) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find anomalies in analysis period
        for idx, row in daily_revenue_analysis.iterrows():
            z_score = (row['revenue'] - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    "title": "Daily Revenue Anomaly",
                    "claim": f"Daily revenue on {row['date'].strftime('%Y-%m-%d')} was {abs(z_score):.2f} standard deviations from baseline mean",
                    "finding_type": "revenue_anomaly",
                    "metrics": {
                        "daily_revenue": {
                            "value": round(row['revenue'], 2),
                            "unit": "SAR",
                            "numerator": None,
                            "denominator": None,
                            "period_start": row['date'].isoformat(),
                            "period_end": (row['date'] + timedelta(days=1)).isoformat()
                        },
                        "baseline_mean_daily_revenue": {
                            "value": round(baseline_mean, 2),
                            "unit": "SAR",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        },
                        "z_score": {
                            "value": round(z_score, 2),
                            "unit": "standard_deviations",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        }
                    },
                    "source_names": ["pos"],
                    "sample_size": len(baseline_daily_revenues),
                    "coverage_notes": [
                        f"Baseline calculated from {len(baseline_daily_revenues)} daily observations across 4 trailing weeks",
                        f"Analysis period: {analysis_period['start']} to {analysis_period['end']}"
                    ],
                    "assumptions": [
                        "Daily revenue follows approximately normal distribution",
                        "Baseline periods are representative of normal operations",
                        "Z-score threshold of 2.0 (95% confidence) used to identify anomalies"
                    ],
                    "confidence": 0.85
                })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_analysis = traffic_df[
    (traffic_df['date'] >= pd.Timestamp(analysis_start.date())) & 
    (traffic_df['date'] < pd.Timestamp(analysis_end.date())) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

traffic_baseline = traffic_df[
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

# Group by hour of day for baseline
traffic_baseline['hour_of_day'] = traffic_baseline['hour'].astype(int) % 24
baseline_hourly_traffic = traffic_baseline.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()

# Analyze analysis period
traffic_analysis['hour_of_day'] = traffic_analysis['hour'].astype(int) % 24
analysis_hourly_traffic = traffic_analysis.groupby('hour_of_day')['door_count'].agg(['mean', 'std', 'count']).reset_index()

# Find anomalies
for idx, row in analysis_hourly_traffic.iterrows():
    hour = int(row['hour_of_day'])
    baseline_row = baseline_hourly_traffic[baseline_hourly_traffic['hour_of_day'] == hour]
    
    if len(baseline_row) > 0 and baseline_row.iloc[0]['std'] > 0:
        baseline_mean = baseline_row.iloc[0]['mean']
        baseline_std = baseline_row.iloc[0]['std']
        baseline_count = baseline_row.iloc[0]['count']
        
        if baseline_count >= 5:  # Need at least 5 observations
            z_score = (row['mean'] - baseline_mean) / baseline_std
            if abs(z_score) > 2.0:
                findings.append({
                    "title": f"Hourly Traffic Anomaly (Hour {hour:02d}:00)",
                    "claim": f"Average hourly door count at {hour:02d}:00 was {abs(z_score):.2f} standard deviations from baseline",
                    "finding_type": "traffic_anomaly",
                    "metrics": {
                        "hourly_door_count_analysis": {
                            "value": round(row['mean'], 1),
                            "unit": "doors/hour",
                            "numerator": None,
                            "denominator": None,
                            "period_start": analysis_period["start"],
                            "period_end": analysis_period["end"]
                        },
                        "hourly_door_count_baseline": {
                            "value": round(baseline_mean, 1),
                            "unit": "doors/hour",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        },
                        "z_score": {
                            "value": round(z_score, 2),
                            "unit": "standard_deviations",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        }
                    },
                    "source_names": ["traffic"],
                    "sample_size": int(baseline_count),
                    "coverage_notes": [
                        f"Baseline calculated from {int(baseline_count)} observations for hour {hour:02d}:00",
                        f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
                        "Dead sensor days excluded from analysis"
                    ],
                    "assumptions": [
                        "Hourly traffic follows approximately normal distribution",
                        "Baseline periods are representative of normal operations",
                        "Z-score threshold of 2.0 (95% confidence) used to identify anomalies"
                    ],
                    "confidence": 0.80
                })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

pos_analysis_txn = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

pos_analysis_txn['date'] = pos_analysis_txn['timestamp_local'].dt.date
daily_txn_analysis = pos_analysis_txn.groupby('date')['transaction_id'].nunique().reset_index()
daily_txn_analysis.columns = ['date', 'transaction_count']
daily_txn_analysis['date'] = pd.to_datetime(daily_txn_analysis['date'])

# Calculate baseline transaction counts
baseline_daily_txns = []
for period in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp_local'] >= period['start']) & 
        (pos_df['timestamp_local'] < period['end'])
    ].copy()
    pos_baseline['date'] = pos_baseline['timestamp_local'].dt.date
    daily_txn = pos_baseline.groupby('date')['transaction_id'].nunique().reset_index()
    daily_txn.columns = ['date', 'transaction_count']
    daily_txn['date'] = pd.to_datetime(daily_txn['date'])
    baseline_daily_txns.extend(daily_txn['transaction_count'].values)

if len(baseline_daily_txns) > 0 and len(daily_txn_analysis) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    if baseline_txn_std > 0:
        # Find anomalies in analysis period
        for idx, row in daily_txn_analysis.iterrows():
            z_score = (row['transaction_count'] - baseline_txn_mean) / baseline_txn_std
            if abs(z_score) > 2.0:  # 2 standard deviations
                findings.append({
                    "title": "Daily Transaction Count Anomaly",
                    "claim": f"Daily transaction count on {row['date'].strftime('%Y-%m-%d')} was {abs(z_score):.2f} standard deviations from baseline mean",
                    "finding_type": "transaction_anomaly",
                    "metrics": {
                        "daily_transaction_count": {
                            "value": int(row['transaction_count']),
                            "unit": "transactions",
                            "numerator": None,
                            "denominator": None,
                            "period_start": row['date'].isoformat(),
                            "period_end": (row['date'] + timedelta(days=1)).isoformat()
                        },
                        "baseline_mean_daily_transactions": {
                            "value": round(baseline_txn_mean, 1),
                            "unit": "transactions",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        },
                        "z_score": {
                            "value": round(z_score, 2),
                            "unit": "standard_deviations",
                            "numerator": None,
                            "denominator": None,
                            "period_start": trailing_baseline_periods[0]["start"],
                            "period_end": trailing_baseline_periods[-1]["end"]
                        }
                    },
                    "source_names": ["pos"],
                    "sample_size": len(baseline_daily_txns),
                    "coverage_notes": [
                        f"Baseline calculated from {len(baseline_daily_txns)} daily observations across 4 trailing weeks",
                        f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
                        "Transaction count based on unique transaction_id values"
                    ],
                    "assumptions": [
                        "Daily transaction counts follow approximately normal distribution",
                        "Baseline periods are representative of normal operations",
                        "Z-score threshold of 2.0 (95% confidence) used to identify anomalies"
                    ],
                    "confidence": 0.85
                })

# Sort findings by magnitude of z-score and keep top 3
findings_with_zscore = []
for finding in findings:
    if 'z_score' in finding['metrics']:
        z_val = abs(finding['metrics']['z_score']['value'])
        findings_with_zscore.append((z_val, finding))

findings_with_zscore.sort(key=lambda x: x[0], reverse=True)
top_findings = [f[1] for f in findings_with_zscore[:3]]

# Prepare output
output = {
    "status": "success" if len(top_findings) > 0 else "insufficient_data",
    "findings": top_findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
