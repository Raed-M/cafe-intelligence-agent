import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define timezone
tz = pytz.timezone('Asia/Riyadh')

# Parse analysis periods
analysis_start = datetime.fromisoformat('2026-05-25T00:00:00+03:00')
analysis_end = datetime.fromisoformat('2026-06-01T00:00:00+03:00')

baseline_periods = [
    (datetime.fromisoformat('2026-05-18T00:00:00+03:00'), datetime.fromisoformat('2026-05-25T00:00:00+03:00')),
    (datetime.fromisoformat('2026-05-11T00:00:00+03:00'), datetime.fromisoformat('2026-05-18T00:00:00+03:00')),
    (datetime.fromisoformat('2026-05-04T00:00:00+03:00'), datetime.fromisoformat('2026-05-11T00:00:00+03:00')),
    (datetime.fromisoformat('2026-04-27T00:00:00+03:00'), datetime.fromisoformat('2026-05-04T00:00:00+03:00')),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Initialize findings list
findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period (excluding refunds in net)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

pos_analysis['calendar_date'] = pd.to_datetime(pos_analysis['calendar_date'])
daily_revenue_analysis = pos_analysis.groupby('calendar_date')['line_total_sar'].sum().reset_index()
daily_revenue_analysis.columns = ['date', 'revenue']

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp'] >= period_start) & 
        (pos_df['timestamp'] < period_end)
    ].copy()
    pos_baseline['calendar_date'] = pd.to_datetime(pos_baseline['calendar_date'])
    daily_rev = pos_baseline.groupby('calendar_date')['line_total_sar'].sum().reset_index()
    baseline_daily_revenues.extend(daily_rev['line_total_sar'].values)

if len(baseline_daily_revenues) >= 5 and np.std(baseline_daily_revenues) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    # Find anomalies in analysis period
    anomalies = []
    for idx, row in daily_revenue_analysis.iterrows():
        z_score = (row['revenue'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2.0:  # 2-sigma threshold
            anomalies.append({
                'date': row['date'],
                'revenue': row['revenue'],
                'z_score': z_score,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std
            })
    
    # Sort by magnitude
    anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    if anomalies:
        top_anomaly = anomalies[0]
        finding = {
            "title": "Unusual Daily Revenue Detected",
            "claim": f"Daily revenue on {top_anomaly['date'].strftime('%Y-%m-%d')} was {top_anomaly['revenue']:.2f} SAR, {abs(top_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
            "finding_type": "revenue_anomaly",
            "metrics": {
                "observed_daily_revenue": {
                    "value": round(top_anomaly['revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                },
                "baseline_mean_daily_revenue": {
                    "value": round(baseline_mean, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-27",
                    "period_end": "2026-05-25"
                },
                "z_score": {
                    "value": round(top_anomaly['z_score'], 2),
                    "unit": "standard_deviations",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                }
            },
            "source_names": ["pos"],
            "sample_size": len(baseline_daily_revenues),
            "coverage_notes": [
                f"Baseline computed from {len(baseline_daily_revenues)} daily observations across 4 weeks (2026-04-27 to 2026-05-25)",
                f"Analysis period: 2026-05-25 to 2026-06-01 ({len(daily_revenue_analysis)} days)"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 (95% confidence under normal distribution)",
                "Refunds included in net revenue calculation",
                "No exclusion of known invalid periods in analysis window"
            ],
            "confidence": 0.85
        }
        findings.append(finding)

# ============================================================================
# ANOMALY 2: Daily Transaction Count Analysis
# ============================================================================

pos_analysis['transaction_id'] = pos_analysis['transaction_id'].astype(str)
daily_transactions_analysis = pos_analysis.groupby('calendar_date')['transaction_id'].nunique().reset_index()
daily_transactions_analysis.columns = ['date', 'transaction_count']

baseline_daily_transactions = []
for period_start, period_end in baseline_periods:
    pos_baseline = pos_df[
        (pos_df['timestamp'] >= period_start) & 
        (pos_df['timestamp'] < period_end)
    ].copy()
    pos_baseline['calendar_date'] = pd.to_datetime(pos_baseline['calendar_date'])
    pos_baseline['transaction_id'] = pos_baseline['transaction_id'].astype(str)
    daily_trans = pos_baseline.groupby('calendar_date')['transaction_id'].nunique().reset_index()
    baseline_daily_transactions.extend(daily_trans['transaction_id'].values)

if len(baseline_daily_transactions) >= 5 and np.std(baseline_daily_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_daily_transactions)
    baseline_trans_std = np.std(baseline_daily_transactions)
    
    trans_anomalies = []
    for idx, row in daily_transactions_analysis.iterrows():
        z_score = (row['transaction_count'] - baseline_trans_mean) / baseline_trans_std if baseline_trans_std > 0 else 0
        if abs(z_score) > 2.0:
            trans_anomalies.append({
                'date': row['date'],
                'transaction_count': row['transaction_count'],
                'z_score': z_score,
                'baseline_mean': baseline_trans_mean,
                'baseline_std': baseline_trans_std
            })
    
    trans_anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    if trans_anomalies and len(findings) < 3:
        top_trans_anomaly = trans_anomalies[0]
        finding = {
            "title": "Unusual Daily Transaction Count Detected",
            "claim": f"Daily transaction count on {top_trans_anomaly['date'].strftime('%Y-%m-%d')} was {top_trans_anomaly['transaction_count']} transactions, {abs(top_trans_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f} transactions.",
            "finding_type": "transaction_volume_anomaly",
            "metrics": {
                "observed_daily_transactions": {
                    "value": int(top_trans_anomaly['transaction_count']),
                    "unit": "transactions",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_trans_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_trans_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                },
                "baseline_mean_daily_transactions": {
                    "value": round(baseline_trans_mean, 1),
                    "unit": "transactions",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-27",
                    "period_end": "2026-05-25"
                },
                "z_score": {
                    "value": round(top_trans_anomaly['z_score'], 2),
                    "unit": "standard_deviations",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_trans_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_trans_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                }
            },
            "source_names": ["pos"],
            "sample_size": len(baseline_daily_transactions),
            "coverage_notes": [
                f"Baseline computed from {len(baseline_daily_transactions)} daily observations across 4 weeks (2026-04-27 to 2026-05-25)",
                f"Analysis period: 2026-05-25 to 2026-06-01 ({len(daily_transactions_analysis)} days)"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 (95% confidence under normal distribution)",
                "Transaction count derived from unique transaction_id per day",
                "Refunds counted as transactions"
            ],
            "confidence": 0.82
        }
        findings.append(finding)

# ============================================================================
# ANOMALY 3: Daily Traffic Analysis
# ============================================================================

traffic_analysis = traffic_df[
    (traffic_df['date'] >= analysis_start.date()) & 
    (traffic_df['date'] < analysis_end.date()) &
    (traffic_df['is_dead_sensor_day'] == False)
].copy()

daily_traffic_analysis = traffic_analysis.groupby('date')['door_count'].sum().reset_index()
daily_traffic_analysis.columns = ['date', 'traffic_count']

baseline_daily_traffic = []
for period_start, period_end in baseline_periods:
    traffic_baseline = traffic_df[
        (traffic_df['date'] >= period_start.date()) & 
        (traffic_df['date'] < period_end.date()) &
        (traffic_df['is_dead_sensor_day'] == False)
    ].copy()
    daily_traffic = traffic_baseline.groupby('date')['door_count'].sum().reset_index()
    baseline_daily_traffic.extend(daily_traffic['door_count'].values)

if len(baseline_daily_traffic) >= 5 and np.std(baseline_daily_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_daily_traffic)
    baseline_traffic_std = np.std(baseline_daily_traffic)
    
    traffic_anomalies = []
    for idx, row in daily_traffic_analysis.iterrows():
        z_score = (row['traffic_count'] - baseline_traffic_mean) / baseline_traffic_std if baseline_traffic_std > 0 else 0
        if abs(z_score) > 2.0:
            traffic_anomalies.append({
                'date': row['date'],
                'traffic_count': row['traffic_count'],
                'z_score': z_score,
                'baseline_mean': baseline_traffic_mean,
                'baseline_std': baseline_traffic_std
            })
    
    traffic_anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    if traffic_anomalies and len(findings) < 3:
        top_traffic_anomaly = traffic_anomalies[0]
        finding = {
            "title": "Unusual Daily Traffic Count Detected",
            "claim": f"Daily foot traffic on {top_traffic_anomaly['date'].strftime('%Y-%m-%d')} was {top_traffic_anomaly['traffic_count']} visitors, {abs(top_traffic_anomaly['z_score']):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f} visitors.",
            "finding_type": "traffic_anomaly",
            "metrics": {
                "observed_daily_traffic": {
                    "value": int(top_traffic_anomaly['traffic_count']),
                    "unit": "visitors",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_traffic_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_traffic_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                },
                "baseline_mean_daily_traffic": {
                    "value": round(baseline_traffic_mean, 1),
                    "unit": "visitors",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-27",
                    "period_end": "2026-05-25"
                },
                "z_score": {
                    "value": round(top_traffic_anomaly['z_score'], 2),
                    "unit": "standard_deviations",
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_traffic_anomaly['date'].strftime('%Y-%m-%d'),
                    "period_end": (top_traffic_anomaly['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
                }
            },
            "source_names": ["traffic"],
            "sample_size": len(baseline_daily_traffic),
            "coverage_notes": [
                f"Baseline computed from {len(baseline_daily_traffic)} daily observations across 4 weeks (2026-04-27 to 2026-05-25)",
                f"Analysis period: 2026-05-25 to 2026-06-01 ({len(daily_traffic_analysis)} days)",
                "Excluded days marked as is_dead_sensor_day=True"
            ],
            "assumptions": [
                "Z-score threshold of 2.0 (95% confidence under normal distribution)",
                "Traffic count aggregated from hourly door_count readings",
                "Dead sensor days excluded from both baseline and analysis"
            ],
            "confidence": 0.80
        }
        findings.append(finding)

# Prepare output
result = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)