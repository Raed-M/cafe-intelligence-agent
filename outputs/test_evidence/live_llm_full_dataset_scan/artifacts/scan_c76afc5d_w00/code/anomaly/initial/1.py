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

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define analysis periods
analysis_start = datetime(2026, 1, 5, 0, 0, 0)
analysis_end = datetime(2026, 1, 12, 0, 0, 0)
baseline_start = datetime(2025, 12, 8, 0, 0, 0)
baseline_end = datetime(2026, 1, 5, 0, 0, 0)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
reviews_analysis = reviews_df[(reviews_df['date'] >= analysis_start) & (reviews_df['date'] < analysis_end)].copy()

# Filter data for baseline periods
pos_baseline = pos_df[(pos_df['timestamp'] >= baseline_start) & (pos_df['timestamp'] < baseline_end)].copy()
traffic_baseline = traffic_df[(traffic_df['date'] >= baseline_start) & (traffic_df['date'] < baseline_end)].copy()
reviews_baseline = reviews_df[(reviews_df['date'] >= baseline_start) & (reviews_df['date'] < baseline_end)].copy()

findings = []

# ANOMALY 1: Daily Revenue Analysis
# Calculate daily revenue (excluding refunds)
pos_analysis['is_refund'] = pos_analysis['is_refund'].fillna(False)
pos_baseline['is_refund'] = pos_baseline['is_refund'].fillna(False)

# Daily revenue for analysis period
daily_revenue_analysis = pos_analysis[~pos_analysis['is_refund']].groupby('calendar_date')['line_total_sar'].sum()
daily_revenue_baseline = pos_baseline[~pos_baseline['is_refund']].groupby('calendar_date')['line_total_sar'].sum()

if len(daily_revenue_analysis) > 0 and len(daily_revenue_baseline) > 2:
    baseline_mean = daily_revenue_baseline.mean()
    baseline_std = daily_revenue_baseline.std()
    
    if baseline_std > 0:
        # Calculate z-scores for analysis period
        z_scores = (daily_revenue_analysis - baseline_mean) / baseline_std
        
        # Find anomalies (|z| > 2)
        anomalies = z_scores[abs(z_scores) > 2]
        
        if len(anomalies) > 0:
            # Get the most extreme anomaly
            max_anomaly_idx = anomalies.abs().idxmax()
            max_anomaly_value = daily_revenue_analysis[max_anomaly_idx]
            max_anomaly_zscore = z_scores[max_anomaly_idx]
            
            findings.append({
                "title": "Unusual Daily Revenue Spike",
                "claim": f"Daily revenue on {max_anomaly_idx.date()} was {max_anomaly_value:.2f} SAR, {abs(max_anomaly_zscore):.2f} standard deviations from baseline mean of {baseline_mean:.2f} SAR",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "daily_revenue": {
                        "value": round(max_anomaly_value, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_idx.isoformat(),
                        "period_end": (max_anomaly_idx + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": baseline_start.isoformat(),
                        "period_end": baseline_end.isoformat()
                    },
                    "z_score": {
                        "value": round(max_anomaly_zscore, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_idx.isoformat(),
                        "period_end": (max_anomaly_idx + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(daily_revenue_baseline),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline period: {baseline_start.date()} to {baseline_end.date()}",
                    f"Baseline observations: {len(daily_revenue_baseline)} days",
                    "Refunds excluded from revenue calculation"
                ],
                "assumptions": [
                    "Normal distribution of daily revenue",
                    "Z-score threshold of 2.0 standard deviations",
                    "Baseline period is representative of normal operations"
                ],
                "confidence": 0.85
            })

# ANOMALY 2: Daily Transaction Count Analysis
daily_transactions_analysis = pos_analysis.groupby('calendar_date')['transaction_id'].nunique()
daily_transactions_baseline = pos_baseline.groupby('calendar_date')['transaction_id'].nunique()

if len(daily_transactions_analysis) > 0 and len(daily_transactions_baseline) > 2:
    baseline_trans_mean = daily_transactions_baseline.mean()
    baseline_trans_std = daily_transactions_baseline.std()
    
    if baseline_trans_std > 0:
        z_scores_trans = (daily_transactions_analysis - baseline_trans_mean) / baseline_trans_std
        anomalies_trans = z_scores_trans[abs(z_scores_trans) > 2]
        
        if len(anomalies_trans) > 0:
            max_anomaly_trans_idx = anomalies_trans.abs().idxmax()
            max_anomaly_trans_value = daily_transactions_analysis[max_anomaly_trans_idx]
            max_anomaly_trans_zscore = z_scores_trans[max_anomaly_trans_idx]
            
            findings.append({
                "title": "Unusual Daily Transaction Volume",
                "claim": f"Daily transaction count on {max_anomaly_trans_idx.date()} was {int(max_anomaly_trans_value)} transactions, {abs(max_anomaly_trans_zscore):.2f} standard deviations from baseline mean of {baseline_trans_mean:.1f}",
                "finding_type": "transaction_volume_anomaly",
                "metrics": {
                    "daily_transactions": {
                        "value": int(max_anomaly_trans_value),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_trans_idx.isoformat(),
                        "period_end": (max_anomaly_trans_idx + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean": {
                        "value": round(baseline_trans_mean, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": baseline_start.isoformat(),
                        "period_end": baseline_end.isoformat()
                    },
                    "z_score": {
                        "value": round(max_anomaly_trans_zscore, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_trans_idx.isoformat(),
                        "period_end": (max_anomaly_trans_idx + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(daily_transactions_baseline),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline period: {baseline_start.date()} to {baseline_end.date()}",
                    f"Baseline observations: {len(daily_transactions_baseline)} days",
                    "Unique transaction_id used for basket counting"
                ],
                "assumptions": [
                    "Normal distribution of daily transaction counts",
                    "Z-score threshold of 2.0 standard deviations",
                    "Baseline period is representative of normal operations"
                ],
                "confidence": 0.85
            })

# ANOMALY 3: Daily Traffic Analysis
traffic_analysis_clean = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False].copy()
traffic_baseline_clean = traffic_baseline[traffic_baseline['is_dead_sensor_day'] == False].copy()

daily_traffic_analysis = traffic_analysis_clean.groupby('date')['door_count'].sum()
daily_traffic_baseline = traffic_baseline_clean.groupby('date')['door_count'].sum()

if len(daily_traffic_analysis) > 0 and len(daily_traffic_baseline) > 2:
    baseline_traffic_mean = daily_traffic_baseline.mean()
    baseline_traffic_std = daily_traffic_baseline.std()
    
    if baseline_traffic_std > 0:
        z_scores_traffic = (daily_traffic_analysis - baseline_traffic_mean) / baseline_traffic_std
        anomalies_traffic = z_scores_traffic[abs(z_scores_traffic) > 2]
        
        if len(anomalies_traffic) > 0:
            max_anomaly_traffic_idx = anomalies_traffic.abs().idxmax()
            max_anomaly_traffic_value = daily_traffic_analysis[max_anomaly_traffic_idx]
            max_anomaly_traffic_zscore = z_scores_traffic[max_anomaly_traffic_idx]
            
            findings.append({
                "title": "Unusual Daily Door Traffic",
                "claim": f"Daily door count on {max_anomaly_traffic_idx.date()} was {int(max_anomaly_traffic_value)} visitors, {abs(max_anomaly_traffic_zscore):.2f} standard deviations from baseline mean of {baseline_traffic_mean:.1f}",
                "finding_type": "traffic_anomaly",
                "metrics": {
                    "daily_door_count": {
                        "value": int(max_anomaly_traffic_value),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_traffic_idx.isoformat(),
                        "period_end": (max_anomaly_traffic_idx + timedelta(days=1)).isoformat()
                    },
                    "baseline_mean": {
                        "value": round(baseline_traffic_mean, 1),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": baseline_start.isoformat(),
                        "period_end": baseline_end.isoformat()
                    },
                    "z_score": {
                        "value": round(max_anomaly_traffic_zscore, 2),
                        "unit": "std_dev",
                        "numerator": None,
                        "denominator": None,
                        "period_start": max_anomaly_traffic_idx.isoformat(),
                        "period_end": (max_anomaly_traffic_idx + timedelta(days=1)).isoformat()
                    }
                },
                "source_names": ["traffic"],
                "sample_size": len(daily_traffic_baseline),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                    f"Baseline period: {baseline_start.date()} to {baseline_end.date()}",
                    f"Baseline observations: {len(daily_traffic_baseline)} days",
                    "Dead sensor days excluded from analysis"
                ],
                "assumptions": [
                    "Normal distribution of daily door counts",
                    "Z-score threshold of 2.0 standard deviations",
                    "Baseline period is representative of normal operations",
                    "Sensor data is accurate and consistent"
                ],
                "confidence": 0.80
            })

# Sort findings by z-score magnitude (confidence)
findings.sort(key=lambda x: abs(x['metrics'].get('z_score', {}).get('value', 0)), reverse=True)

# Keep only top 3
findings = findings[:3]

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} anomalies detected.")
