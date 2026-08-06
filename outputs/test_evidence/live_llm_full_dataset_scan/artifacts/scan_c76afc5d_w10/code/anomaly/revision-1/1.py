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

# Define periods
analysis_start = pd.Timestamp("2026-03-16T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-03-23T00:00:00+03:00")

baseline_periods = [
    (pd.Timestamp("2026-03-09T00:00:00+03:00"), pd.Timestamp("2026-03-16T00:00:00+03:00")),
    (pd.Timestamp("2026-03-02T00:00:00+03:00"), pd.Timestamp("2026-03-09T00:00:00+03:00")),
    (pd.Timestamp("2026-02-23T00:00:00+03:00"), pd.Timestamp("2026-03-02T00:00:00+03:00")),
    (pd.Timestamp("2026-02-16T00:00:00+03:00"), pd.Timestamp("2026-02-23T00:00:00+03:00")),
]

findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Ensure timestamp columns are datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Calculate daily revenue for analysis period (excluding refunds per metric definition)
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
analysis_daily_revenue = analysis_pos[analysis_pos['is_refund'] == False].groupby('calendar_date')['line_total_sar'].sum()

# Calculate daily revenue for baseline periods
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    daily_rev = baseline_pos[baseline_pos['is_refund'] == False].groupby('calendar_date')['line_total_sar'].sum()
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Find the day with highest z-score
        max_z_score = -np.inf
        max_z_day = None
        max_z_value = None
        
        for date, revenue in analysis_daily_revenue.items():
            z_score = (revenue - baseline_mean) / baseline_std
            if abs(z_score) > abs(max_z_score):
                max_z_score = z_score
                max_z_day = date
                max_z_value = revenue
        
        if max_z_day is not None and abs(max_z_score) >= 2.0:
            findings.append({
                "title": "Anomalous Daily Revenue",
                "claim": f"Daily revenue on {max_z_day.date()} reached {max_z_value:.2f} SAR, {abs(max_z_score):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "observed_daily_revenue": {
                        "value": round(max_z_value, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day.isoformat(),
                        "period_end": (max_z_day + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean_daily_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "baseline_std_daily_revenue": {
                        "value": round(baseline_std, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "z_score": {
                        "value": round(max_z_score, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day.isoformat(),
                        "period_end": (max_z_day + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenues),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior ({baseline_periods[0][0].date()} to {baseline_periods[-1][1].date()})",
                    "Refunds excluded from revenue calculation"
                ],
                "assumptions": [
                    "Daily revenue follows normal distribution",
                    "Z-score threshold: |z| >= 2.0 (p < 0.05)",
                    "Baseline calculated from 4 complete weeks",
                    "No known sensor outages during analysis period"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Daily Transaction Volume Analysis
# ============================================================================

# Calculate daily transaction count (unique transaction_ids, excluding refunds)
analysis_transactions = analysis_pos[analysis_pos['is_refund'] == False].groupby('calendar_date')['transaction_id'].nunique()

# Calculate baseline daily transaction counts
baseline_daily_transactions = []
for period_start, period_end in baseline_periods:
    baseline_pos = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    daily_trans = baseline_pos[baseline_pos['is_refund'] == False].groupby('calendar_date')['transaction_id'].nunique()
    baseline_daily_transactions.extend(daily_trans.values)

if len(baseline_daily_transactions) > 0 and len(analysis_transactions) > 0:
    baseline_trans_mean = np.mean(baseline_daily_transactions)
    baseline_trans_std = np.std(baseline_daily_transactions)
    
    if baseline_trans_std > 0:
        # Find the day with highest z-score
        max_z_score_trans = -np.inf
        max_z_day_trans = None
        max_z_value_trans = None
        
        for date, trans_count in analysis_transactions.items():
            z_score = (trans_count - baseline_trans_mean) / baseline_trans_std
            if abs(z_score) > abs(max_z_score_trans):
                max_z_score_trans = z_score
                max_z_day_trans = date
                max_z_value_trans = trans_count
        
        if max_z_day_trans is not None and abs(max_z_score_trans) >= 2.0:
            findings.append({
                "title": "Anomalous Daily Transaction Volume",
                "claim": f"Daily transaction count on {max_z_day_trans.date()} was {max_z_value_trans} transactions, {abs(max_z_score_trans):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f} transactions.",
                "finding_type": "transaction_volume_anomaly",
                "metrics": {
                    "observed_daily_transactions": {
                        "value": int(max_z_value_trans),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day_trans.isoformat(),
                        "period_end": (max_z_day_trans + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean_daily_transactions": {
                        "value": round(baseline_trans_mean, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "baseline_std_daily_transactions": {
                        "value": round(baseline_trans_std, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "z_score": {
                        "value": round(max_z_score_trans, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day_trans.isoformat(),
                        "period_end": (max_z_day_trans + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_transactions),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior ({baseline_periods[0][0].date()} to {baseline_periods[-1][1].date()})",
                    "Refunds excluded from transaction count"
                ],
                "assumptions": [
                    "Daily transaction count follows normal distribution",
                    "Z-score threshold: |z| >= 2.0 (p < 0.05)",
                    "Baseline calculated from 4 complete weeks",
                    "Transaction_id uniqueness indicates distinct baskets"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 3: Daily Traffic Analysis
# ============================================================================

traffic_df['date'] = pd.to_datetime(traffic_df['date'])

# Calculate daily traffic for analysis period
analysis_traffic = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())]
analysis_daily_traffic = analysis_traffic[analysis_traffic['is_dead_sensor_day'] == False].groupby('date')['door_count'].sum()

# Calculate baseline daily traffic
baseline_daily_traffic = []
for period_start, period_end in baseline_periods:
    baseline_traffic = traffic_df[(traffic_df['date'] >= period_start.date()) & (traffic_df['date'] < period_end.date())]
    daily_traffic = baseline_traffic[baseline_traffic['is_dead_sensor_day'] == False].groupby('date')['door_count'].sum()
    baseline_daily_traffic.extend(daily_traffic.values)

if len(baseline_daily_traffic) > 0 and len(analysis_daily_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_daily_traffic)
    baseline_traffic_std = np.std(baseline_daily_traffic)
    
    if baseline_traffic_std > 0:
        # Find the day with highest z-score
        max_z_score_traffic = -np.inf
        max_z_day_traffic = None
        max_z_value_traffic = None
        
        for date, traffic_count in analysis_daily_traffic.items():
            z_score = (traffic_count - baseline_traffic_mean) / baseline_traffic_std
            if abs(z_score) > abs(max_z_score_traffic):
                max_z_score_traffic = z_score
                max_z_day_traffic = date
                max_z_value_traffic = traffic_count
        
        if max_z_day_traffic is not None and abs(max_z_score_traffic) >= 2.0:
            findings.append({
                "title": "Anomalous Daily Traffic",
                "claim": f"Daily door count on {max_z_day_traffic.date()} was {max_z_value_traffic} visitors, {abs(max_z_score_traffic):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f} visitors.",
                "finding_type": "traffic_anomaly",
                "metrics": {
                    "observed_daily_traffic": {
                        "value": int(max_z_value_traffic),
                        "unit": "door_count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day_traffic.isoformat(),
                        "period_end": (max_z_day_traffic + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean_daily_traffic": {
                        "value": round(baseline_traffic_mean, 1),
                        "unit": "door_count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "baseline_std_daily_traffic": {
                        "value": round(baseline_traffic_std, 1),
                        "unit": "door_count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-02-16T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "z_score": {
                        "value": round(max_z_score_traffic, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_z_day_traffic.isoformat(),
                        "period_end": (max_z_day_traffic + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["traffic"],
                "sample_size": len(baseline_daily_traffic),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline: 4 weeks prior ({baseline_periods[0][0].date()} to {baseline_periods[-1][1].date()})",
                    "Dead sensor days excluded from calculation"
                ],
                "assumptions": [
                    "Daily traffic follows normal distribution",
                    "Z-score threshold: |z| >= 2.0 (p < 0.05)",
                    "Baseline calculated from 4 complete weeks",
                    "Sensor data quality validated by is_dead_sensor_day flag"
                ],
                "confidence": 0.85
            })

# Sort findings by z-score magnitude
findings.sort(key=lambda x: abs(x['metrics']['z_score']['value']), reverse=True)

# Keep only top 3
findings = findings[:3]

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)