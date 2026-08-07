import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-27T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-20T00:00:00+03:00")

trailing_periods = [
    ("2026-07-13T00:00:00+03:00", "2026-07-20T00:00:00+03:00"),
    ("2026-07-06T00:00:00+03:00", "2026-07-13T00:00:00+03:00"),
    ("2026-06-29T00:00:00+03:00", "2026-07-06T00:00:00+03:00"),
    ("2026-06-22T00:00:00+03:00", "2026-06-29T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
def filter_by_period(df, start_str, end_str):
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

analysis_data = filter_by_period(pos_df, "2026-07-20T00:00:00+03:00", "2026-07-27T00:00:00+03:00")
previous_data = filter_by_period(pos_df, "2026-07-13T00:00:00+03:00", "2026-07-20T00:00:00+03:00")

# Calculate baseline (average of trailing periods)
trailing_data_list = []
for start_str, end_str in trailing_periods:
    trailing_data_list.append(filter_by_period(pos_df, start_str, end_str))
trailing_data = pd.concat(trailing_data_list, ignore_index=True)

# Helper function to calculate metrics
def calculate_metrics(df, period_start, period_end):
    # Remove refunds for transaction count
    non_refund = df[df['is_refund'] == False]
    
    # Count unique transactions
    transaction_count = non_refund['transaction_id'].nunique()
    
    # Total revenue (net, including refunds)
    total_revenue = df['line_total_sar'].sum()
    
    # Average order value (net revenue / transaction count)
    aov = total_revenue / transaction_count if transaction_count > 0 else 0
    
    return {
        'transaction_count': transaction_count,
        'total_revenue': total_revenue,
        'aov': aov,
        'row_count': len(df)
    }

# Calculate metrics for each period
analysis_metrics = calculate_metrics(analysis_data, analysis_start, analysis_end)
previous_metrics = calculate_metrics(previous_data, previous_start, previous_end)

# Calculate baseline metrics
baseline_metrics = calculate_metrics(trailing_data, previous_start, previous_end)
baseline_avg_transaction_count = baseline_metrics['transaction_count']
baseline_avg_revenue = baseline_metrics['total_revenue']
baseline_avg_aov = baseline_metrics['aov']

# Findings list
findings = []

# Finding 1: Revenue change analysis
revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_pct_change = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] != 0 else 0

if abs(revenue_pct_change) >= 5:  # Significant change threshold
    findings.append({
        "title": "Weekly Revenue Change",
        "claim": f"Net revenue in analysis week (2026-07-20 to 2026-07-27) was {analysis_metrics['total_revenue']:.2f} SAR, representing a {revenue_pct_change:.1f}% change from previous week ({previous_metrics['total_revenue']:.2f} SAR).",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_week_revenue": {
                "value": round(analysis_metrics['total_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "previous_week_revenue": {
                "value": round(previous_metrics['total_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_metrics['total_revenue'], 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['row_count'],
        "coverage_notes": [
            "Analysis period: 2026-07-20 to 2026-07-27",
            "Previous period: 2026-07-13 to 2026-07-20",
            "Refunds included in net revenue calculation"
        ],
        "assumptions": [
            "line_total_sar represents net realized revenue",
            "All transactions in period are valid",
            "No data quality issues affecting revenue calculation"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction count change
transaction_change = analysis_metrics['transaction_count'] - previous_metrics['transaction_count']
transaction_pct_change = (transaction_change / previous_metrics['transaction_count'] * 100) if previous_metrics['transaction_count'] > 0 else 0

if abs(transaction_pct_change) >= 5:
    findings.append({
        "title": "Transaction Volume Change",
        "claim": f"Valid transaction count in analysis week was {analysis_metrics['transaction_count']}, a {transaction_pct_change:.1f}% change from {previous_metrics['transaction_count']} in the previous week.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_week_transactions": {
                "value": analysis_metrics['transaction_count'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "previous_week_transactions": {
                "value": previous_metrics['transaction_count'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(transaction_pct_change, 1),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_metrics['transaction_count'],
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['row_count'],
        "coverage_notes": [
            "Transactions counted from unique transaction_id where is_refund=False",
            "Analysis period: 2026-07-20 to 2026-07-27",
            "Previous period: 2026-07-13 to 2026-07-20"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "is_refund flag correctly identifies refund transactions",
            "All rows represent valid POS transactions"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

if abs(aov_pct_change) >= 3:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis week was {analysis_metrics['aov']:.2f} SAR, a {aov_pct_change:.1f}% change from {previous_metrics['aov']:.2f} SAR in the previous week.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_week_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['total_revenue'], 2),
                "denominator": analysis_metrics['transaction_count'],
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "previous_week_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['total_revenue'], 2),
                "denominator": previous_metrics['transaction_count'],
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_metrics['aov'], 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['row_count'],
        "coverage_notes": [
            "AOV calculated as net revenue / unique transaction count",
            "Refunds included in net revenue",
            "Analysis period: 2026-07-20 to 2026-07-27",
            "Previous period: 2026-07-13 to 2026-07-20"
        ],
        "assumptions": [
            "line_total_sar represents net realized revenue per line item",
            "transaction_id uniquely identifies a basket",
            "is_refund flag correctly identifies refund transactions"
        ],
        "confidence": 0.95
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)