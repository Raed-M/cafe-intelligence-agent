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
analysis_start = datetime.fromisoformat("2026-04-13T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-06T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-13T00:00:00+03:00")

baseline_periods = [
    (datetime.fromisoformat("2026-04-06T00:00:00+03:00"), datetime.fromisoformat("2026-04-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-30T00:00:00+03:00"), datetime.fromisoformat("2026-04-06T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-23T00:00:00+03:00"), datetime.fromisoformat("2026-03-30T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-16T00:00:00+03:00"), datetime.fromisoformat("2026-03-23T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamps to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baselines
pos_df['date'] = pos_df['timestamp'].dt.date

# Analysis period daily revenue
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
analysis_daily_revenue = analysis_pos.groupby('date')['line_total_sar'].sum()

# Baseline daily revenue
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    baseline_daily = baseline_pos.groupby('date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(baseline_daily.values)

if len(baseline_daily_revenues) > 0 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find most anomalous day in analysis period
        max_z_score = -np.inf
        max_z_day = None
        max_z_value = None
        
        for day, revenue in analysis_daily_revenue.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > abs(max_z_score):
                max_z_score = z_score
                max_z_day = day
                max_z_value = revenue
        
        if max_z_day is not None and abs(max_z_score) > 2.0:
            findings.append({
                "title": "Anomalous Daily Revenue",
                "claim": f"Daily revenue on {max_z_day} was {max_z_value:.2f} SAR, {abs(max_z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "daily_revenue": {
                        "value": round(max_z_value, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day.isoformat(),
                        "period_end": max_z_day.isoformat()
                    },
                    "baseline_mean": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16",
                        "period_end": "2026-04-13"
                    },
                    "z_score": {
                        "value": round(max_z_score, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day.isoformat(),
                        "period_end": max_z_day.isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenues),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                    f"Baseline sample size: {len(baseline_daily_revenues)} daily observations"
                ],
                "assumptions": [
                    "Z-score threshold: |z| > 2.0 (p < 0.05)",
                    "Baseline computed from trailing 4 weeks",
                    "Refunds included in net revenue per metric definition"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Hourly Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
traffic_df['hour'] = pd.to_datetime(traffic_df['hour']).dt.hour

# Filter out dead sensor days
traffic_clean = traffic_df[traffic_df['is_dead_sensor_day'] == False].copy()

# Analysis period hourly traffic
analysis_traffic = traffic_clean[(traffic_clean['date'] >= analysis_start.date()) & (traffic_clean['date'] < analysis_end.date())]
analysis_hourly_traffic = analysis_traffic['door_count'].values

# Baseline hourly traffic
baseline_hourly_traffic = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_clean[(traffic_clean['date'] >= period_start.date()) & (traffic_clean['date'] < period_end.date())]
    baseline_hourly_traffic.extend(baseline_traffic['door_count'].values)

if len(baseline_hourly_traffic) > 10 and len(analysis_hourly_traffic) > 0:
    baseline_mean_traffic = np.mean(baseline_hourly_traffic)
    baseline_std_traffic = np.std(baseline_hourly_traffic)
    
    if baseline_std_traffic > 0:
        # Find most anomalous hour in analysis period
        max_z_traffic = -np.inf
        max_z_traffic_idx = None
        max_z_traffic_value = None
        
        for idx, traffic_val in enumerate(analysis_hourly_traffic):
            z_score = (traffic_val - baseline_mean_traffic) / baseline_std_traffic
            if abs(z_score) > abs(max_z_traffic):
                max_z_traffic = z_score
                max_z_traffic_idx = idx
                max_z_traffic_value = traffic_val
        
        if max_z_traffic_idx is not None and abs(max_z_traffic) > 2.0:
            findings.append({
                "title": "Anomalous Hourly Door Traffic",
                "claim": f"Hourly door count reached {max_z_traffic_value:.0f} visitors, {abs(max_z_traffic):.2f} standard deviations from baseline mean of {baseline_mean_traffic:.1f}.",
                "finding_type": "traffic_anomaly",
                "metrics": {
                    "hourly_door_count": {
                        "value": int(max_z_traffic_value),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.date().isoformat(),
                        "period_end": analysis_end.date().isoformat()
                    },
                    "baseline_mean_traffic": {
                        "value": round(baseline_mean_traffic, 1),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16",
                        "period_end": "2026-04-13"
                    },
                    "z_score_traffic": {
                        "value": round(max_z_traffic, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.date().isoformat(),
                        "period_end": analysis_end.date().isoformat()
                    }
                },
                "source_names": ["traffic"],
                "sample_size": len(baseline_hourly_traffic),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                    f"Dead sensor days excluded",
                    f"Baseline sample size: {len(baseline_hourly_traffic)} hourly observations"
                ],
                "assumptions": [
                    "Z-score threshold: |z| > 2.0 (p < 0.05)",
                    "Baseline computed from trailing 4 weeks, excluding dead sensor intervals",
                    "Hourly granularity"
                ],
                "confidence": 0.80
            })

# ============================================================================
# ANOMALY 3: Daily Transaction Count Analysis
# ============================================================================

# Calculate daily transaction count
analysis_pos_txn = analysis_pos.groupby('date')['transaction_id'].nunique()

# Baseline daily transaction count
baseline_daily_txn = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    baseline_txn = baseline_pos.groupby('date')['transaction_id'].nunique()
    baseline_daily_txn.extend(baseline_txn.values)

if len(baseline_daily_txn) > 0 and len(analysis_pos_txn) > 0:
    baseline_mean_txn = np.mean(baseline_daily_txn)
    baseline_std_txn = np.std(baseline_daily_txn)
    
    if baseline_std_txn > 0:
        # Find most anomalous day
        max_z_txn = -np.inf
        max_z_txn_day = None
        max_z_txn_value = None
        
        for day, txn_count in analysis_pos_txn.items():
            z_score = (txn_count - baseline_mean_txn) / baseline_std_txn
            if abs(z_score) > abs(max_z_txn):
                max_z_txn = z_score
                max_z_txn_day = day
                max_z_txn_value = txn_count
        
        if max_z_txn_day is not None and abs(max_z_txn) > 2.5:
            findings.append({
                "title": "Anomalous Daily Transaction Count",
                "claim": f"Daily transaction count on {max_z_txn_day} was {max_z_txn_value:.0f}, {abs(max_z_txn):.2f} standard deviations from baseline mean of {baseline_mean_txn:.1f}.",
                "finding_type": "transaction_anomaly",
                "metrics": {
                    "daily_transaction_count": {
                        "value": int(max_z_txn_value),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_txn_day.isoformat(),
                        "period_end": max_z_txn_day.isoformat()
                    },
                    "baseline_mean_txn": {
                        "value": round(baseline_mean_txn, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16",
                        "period_end": "2026-04-13"
                    },
                    "z_score_txn": {
                        "value": round(max_z_txn, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_txn_day.isoformat(),
                        "period_end": max_z_txn_day.isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_txn),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks from {baseline_periods[3][0].date()} to {baseline_periods[0][1].date()}",
                    f"Baseline sample size: {len(baseline_daily_txn)} daily observations"
                ],
                "assumptions": [
                    "Z-score threshold: |z| > 2.5 (p < 0.01)",
                    "Baseline computed from trailing 4 weeks",
                    "Transaction counted by unique transaction_id per day"
                ],
                "confidence": 0.82
            })

# Sort findings by confidence and magnitude
findings.sort(key=lambda x: x['confidence'], reverse=True)
findings = findings[:3]

# Build output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
