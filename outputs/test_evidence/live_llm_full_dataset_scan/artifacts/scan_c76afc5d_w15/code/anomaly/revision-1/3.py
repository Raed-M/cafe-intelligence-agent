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
analysis_start = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-27T00:00:00+03:00")

baseline_periods = [
    (datetime.fromisoformat("2026-04-13T00:00:00+03:00"), datetime.fromisoformat("2026-04-20T00:00:00+03:00")),
    (datetime.fromisoformat("2026-04-06T00:00:00+03:00"), datetime.fromisoformat("2026-04-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-30T00:00:00+03:00"), datetime.fromisoformat("2026-04-06T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-23T00:00:00+03:00"), datetime.fromisoformat("2026-03-30T00:00:00+03:00")),
]

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
inventory_df = pd.read_parquet(inputs['inventory'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert timestamps to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Initialize findings list
findings = []

# ============================================================================
# ANOMALY 1: Daily Revenue Analysis
# ============================================================================

# Calculate daily revenue for analysis period and baseline periods
def calculate_daily_revenue(df, period_start, period_end):
    """Calculate daily revenue excluding refunds"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    # Exclude refunds from revenue calculation
    daily_revenue = period_data[~period_data['is_refund']].groupby('date')['line_total_sar'].sum()
    return daily_revenue

# Analysis period daily revenue
analysis_daily_revenue = calculate_daily_revenue(pos_df, analysis_start, analysis_end)

# Baseline daily revenues
baseline_daily_revenues = []
for period_start, period_end in baseline_periods:
    daily_rev = calculate_daily_revenue(pos_df, period_start, period_end)
    baseline_daily_revenues.extend(daily_rev.values)

if len(baseline_daily_revenues) > 0 and len(analysis_daily_revenue) > 0:
    baseline_mean = np.mean(baseline_daily_revenues)
    baseline_std = np.std(baseline_daily_revenues)
    
    if baseline_std > 0:
        # Calculate z-scores for analysis period
        z_scores = (analysis_daily_revenue.values - baseline_mean) / baseline_std
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z_score = z_scores[max_z_idx]
        max_z_date = analysis_daily_revenue.index[max_z_idx]
        max_z_value = analysis_daily_revenue.values[max_z_idx]
        
        # Flag if |z| > 2
        if abs(max_z_score) > 2.0:
            findings.append({
                "title": "Unusual Daily Revenue Spike",
                "claim": f"Daily revenue on {max_z_date} reached {max_z_value:.2f} SAR, {abs(max_z_score):.2f} standard deviations above the 4-week baseline mean of {baseline_mean:.2f} SAR.",
                "finding_type": "revenue_anomaly",
                "metrics": {
                    "observed_daily_revenue": {
                        "value": round(max_z_value, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "baseline_mean_daily_revenue": {
                        "value": round(baseline_mean, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "baseline_std_daily_revenue": {
                        "value": round(baseline_std, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "z_score": {
                        "value": round(max_z_score, 2),
                        "unit": "std_deviations",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_revenues),
                "coverage_notes": [
                    "Analysis period: 2026-04-20 to 2026-04-27 (7 days)",
                    "Baseline: 4 weeks (2026-03-23 to 2026-04-20), 28 daily observations",
                    "Refunds excluded from revenue calculation"
                ],
                "assumptions": [
                    "Daily revenue follows approximately normal distribution",
                    "Baseline periods are representative of typical operations",
                    "Z-score threshold of 2.0 indicates statistical significance (p < 0.05)"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 2: Daily Transaction Count Analysis
# ============================================================================

def calculate_daily_transactions(df, period_start, period_end):
    """Calculate daily transaction count (unique transaction_ids)"""
    mask = (df['timestamp'] >= period_start) & (df['timestamp'] < period_end)
    period_data = df[mask].copy()
    period_data['date'] = period_data['timestamp'].dt.date
    
    # Count unique transactions per day
    daily_transactions = period_data.groupby('date')['transaction_id'].nunique()
    return daily_transactions

# Analysis period daily transactions
analysis_daily_txns = calculate_daily_transactions(pos_df, analysis_start, analysis_end)

# Baseline daily transactions
baseline_daily_txns = []
for period_start, period_end in baseline_periods:
    daily_txns = calculate_daily_transactions(pos_df, period_start, period_end)
    baseline_daily_txns.extend(daily_txns.values)

if len(baseline_daily_txns) > 0 and len(analysis_daily_txns) > 0:
    baseline_txn_mean = np.mean(baseline_daily_txns)
    baseline_txn_std = np.std(baseline_daily_txns)
    
    if baseline_txn_std > 0:
        # Calculate z-scores for analysis period
        txn_z_scores = (analysis_daily_txns.values - baseline_txn_mean) / baseline_txn_std
        max_txn_z_idx = np.argmax(np.abs(txn_z_scores))
        max_txn_z_score = txn_z_scores[max_txn_z_idx]
        max_txn_z_date = analysis_daily_txns.index[max_txn_z_idx]
        max_txn_z_value = analysis_daily_txns.values[max_txn_z_idx]
        
        # Flag if |z| > 2
        if abs(max_txn_z_score) > 2.0:
            findings.append({
                "title": "Unusual Daily Transaction Volume",
                "claim": f"Daily transaction count on {max_txn_z_date} reached {int(max_txn_z_value)} transactions, {abs(max_txn_z_score):.2f} standard deviations from the 4-week baseline mean of {baseline_txn_mean:.1f} transactions.",
                "finding_type": "transaction_volume_anomaly",
                "metrics": {
                    "observed_daily_transactions": {
                        "value": int(max_txn_z_value),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "baseline_mean_daily_transactions": {
                        "value": round(baseline_txn_mean, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "baseline_std_daily_transactions": {
                        "value": round(baseline_txn_std, 1),
                        "unit": "transactions",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "z_score_transactions": {
                        "value": round(max_txn_z_score, 2),
                        "unit": "std_deviations",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(baseline_daily_txns),
                "coverage_notes": [
                    "Analysis period: 2026-04-20 to 2026-04-27 (7 days)",
                    "Baseline: 4 weeks (2026-03-23 to 2026-04-20), 28 daily observations",
                    "Transaction count based on unique transaction_id values"
                ],
                "assumptions": [
                    "Daily transaction counts follow approximately normal distribution",
                    "Baseline periods are representative of typical operations",
                    "Z-score threshold of 2.0 indicates statistical significance (p < 0.05)"
                ],
                "confidence": 0.85
            })

# ============================================================================
# ANOMALY 3: Daily Traffic Analysis
# ============================================================================

def calculate_daily_traffic(df, period_start, period_end):
    """Calculate daily door count from traffic data"""
    # Convert period dates to date objects for comparison
    period_start_date = period_start.date()
    period_end_date = period_end.date()
    
    # Convert df['date'] to date if it's datetime
    df_dates = df['date'].dt.date if hasattr(df['date'], 'dt') else df['date']
    
    mask = (df_dates >= period_start_date) & (df_dates < period_end_date)
    period_data = df[mask].copy()
    
    # Exclude dead sensor days
    period_data = period_data[~period_data['is_dead_sensor_day']]
    
    # Sum hourly counts by date
    period_data['date_only'] = period_data['date'].dt.date if hasattr(period_data['date'], 'dt') else period_data['date']
    daily_traffic = period_data.groupby('date_only')['door_count'].sum()
    return daily_traffic

# Analysis period daily traffic
analysis_daily_traffic = calculate_daily_traffic(traffic_df, analysis_start, analysis_end)

# Baseline daily traffic
baseline_daily_traffic = []
for period_start, period_end in baseline_periods:
    daily_traffic = calculate_daily_traffic(traffic_df, period_start, period_end)
    baseline_daily_traffic.extend(daily_traffic.values)

if len(baseline_daily_traffic) > 0 and len(analysis_daily_traffic) > 0:
    baseline_traffic_mean = np.mean(baseline_daily_traffic)
    baseline_traffic_std = np.std(baseline_daily_traffic)
    
    if baseline_traffic_std > 0:
        # Calculate z-scores for analysis period
        traffic_z_scores = (analysis_daily_traffic.values - baseline_traffic_mean) / baseline_traffic_std
        max_traffic_z_idx = np.argmax(np.abs(traffic_z_scores))
        max_traffic_z_score = traffic_z_scores[max_traffic_z_idx]
        max_traffic_z_date = analysis_daily_traffic.index[max_traffic_z_idx]
        max_traffic_z_value = analysis_daily_traffic.values[max_traffic_z_idx]
        
        # Flag if |z| > 2
        if abs(max_traffic_z_score) > 2.0:
            findings.append({
                "title": "Unusual Daily Door Traffic",
                "claim": f"Daily door count on {max_traffic_z_date} reached {int(max_traffic_z_value)} visitors, {abs(max_traffic_z_score):.2f} standard deviations from the 4-week baseline mean of {baseline_traffic_mean:.1f} visitors.",
                "finding_type": "traffic_anomaly",
                "metrics": {
                    "observed_daily_door_count": {
                        "value": int(max_traffic_z_value),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "baseline_mean_daily_door_count": {
                        "value": round(baseline_traffic_mean, 1),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "baseline_std_daily_door_count": {
                        "value": round(baseline_traffic_std, 1),
                        "unit": "visitors",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-04-20T00:00:00+03:00"
                    },
                    "z_score_traffic": {
                        "value": round(max_traffic_z_score, 2),
                        "unit": "std_deviations",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    }
                },
                "source_names": ["traffic"],
                "sample_size": len(baseline_daily_traffic),
                "coverage_notes": [
                    "Analysis period: 2026-04-20 to 2026-04-27 (7 days)",
                    "Baseline: 4 weeks (2026-03-23 to 2026-04-20), 28 daily observations",
                    "Dead sensor days excluded from calculation",
                    "Daily traffic is sum of hourly door_count values"
                ],
                "assumptions": [
                    "Daily traffic counts follow approximately normal distribution",
                    "Baseline periods are representative of typical operations",
                    "Z-score threshold of 2.0 indicates statistical significance (p < 0.05)",
                    "Sensor data is reliable on non-dead-sensor days"
                ],
                "confidence": 0.85
            })

# Sort findings by absolute z-score magnitude (descending)
def get_z_score(finding):
    metrics = finding['metrics']
    for key in metrics:
        if 'z_score' in key:
            return abs(metrics[key].get('value', 0) or 0)
    return 0

findings.sort(key=get_z_score, reverse=True)

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
