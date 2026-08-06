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

# Parse period boundaries
analysis_start = datetime.fromisoformat("2026-01-19T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-26T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-01-12T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-01-19T00:00:00+03:00")

trailing_baselines = [
    (datetime.fromisoformat("2026-01-12T00:00:00+03:00"), datetime.fromisoformat("2026-01-19T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-05T00:00:00+03:00"), datetime.fromisoformat("2026-01-12T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-29T00:00:00+03:00"), datetime.fromisoformat("2026-01-05T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-22T00:00:00+03:00"), datetime.fromisoformat("2025-12-29T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
def filter_by_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
previous_data = filter_by_period(pos_df, previous_start, previous_end)
trailing_data = [filter_by_period(pos_df, start, end) for start, end in trailing_baselines]

# Helper function to calculate metrics
def calculate_period_metrics(df, menu_df):
    # Count valid transactions (unique transaction_id, excluding refunds for basket count)
    valid_txns = df[~df['is_refund']]['transaction_id'].nunique()
    
    # Net revenue (includes refunds as negative)
    net_revenue = df['line_total_sar'].sum()
    
    # Average order value
    aov = net_revenue / valid_txns if valid_txns > 0 else 0
    
    # Product mix by category
    category_revenue = df.groupby('category')['line_total_sar'].sum().to_dict()
    
    # Channel mix
    channel_revenue = df.groupby('channel')['line_total_sar'].sum().to_dict()
    
    # Refund count and amount
    refund_count = df[df['is_refund']]['transaction_id'].nunique()
    refund_amount = df[df['is_refund']]['line_total_sar'].sum()
    
    return {
        'valid_transactions': valid_txns,
        'net_revenue': net_revenue,
        'aov': aov,
        'category_revenue': category_revenue,
        'channel_revenue': channel_revenue,
        'refund_count': refund_count,
        'refund_amount': refund_amount,
        'total_rows': len(df)
    }

# Calculate metrics for all periods
analysis_metrics = calculate_period_metrics(analysis_data, menu_df)
previous_metrics = calculate_period_metrics(previous_data, menu_df)
trailing_metrics = [calculate_period_metrics(df, menu_df) for df in trailing_data]

# Calculate average of trailing baseline
avg_trailing_txns = np.mean([m['valid_transactions'] for m in trailing_metrics])
avg_trailing_revenue = np.mean([m['net_revenue'] for m in trailing_metrics])
avg_trailing_aov = np.mean([m['aov'] for m in trailing_metrics])

findings = []

# Finding 1: Transaction count change vs previous week
if previous_metrics['valid_transactions'] > 0:
    txn_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
    txn_pct_change = (txn_change / previous_metrics['valid_transactions']) * 100
    
    if abs(txn_pct_change) >= 5:  # Meaningful threshold
        findings.append({
            "title": "Transaction Count Change vs Previous Week",
            "claim": f"Valid transaction count in analysis period (2026-01-19 to 2026-01-26) was {analysis_metrics['valid_transactions']}, compared to {previous_metrics['valid_transactions']} in previous week (2026-01-12 to 2026-01-19), representing a {txn_pct_change:.1f}% change.",
            "finding_type": "transaction_volume",
            "metrics": {
                "analysis_period_transactions": {
                    "value": analysis_metrics['valid_transactions'],
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "previous_period_transactions": {
                    "value": previous_metrics['valid_transactions'],
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-12T00:00:00+03:00",
                    "period_end": "2026-01-19T00:00:00+03:00"
                },
                "absolute_change": {
                    "value": txn_change,
                    "unit": "baskets",
                    "numerator": txn_change,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "percent_change": {
                    "value": round(txn_pct_change, 2),
                    "unit": "%",
                    "numerator": txn_change,
                    "denominator": previous_metrics['valid_transactions'],
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": analysis_metrics['total_rows'],
            "coverage_notes": [
                "Analysis period: 2026-01-19 to 2026-01-26",
                "Previous period: 2026-01-12 to 2026-01-19",
                "Baskets counted using unique transaction_id excluding refunds"
            ],
            "assumptions": [
                "Valid transaction_id values represent unique customer baskets",
                "Refunds excluded from basket count per methodology"
            ],
            "confidence": 0.95
        })

# Finding 2: Net Revenue change vs previous week
if previous_metrics['net_revenue'] != 0:
    revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
    revenue_pct_change = (revenue_change / previous_metrics['net_revenue']) * 100
    
    if abs(revenue_pct_change) >= 5:  # Meaningful threshold
        findings.append({
            "title": "Net Revenue Change vs Previous Week",
            "claim": f"Net revenue in analysis period (2026-01-19 to 2026-01-26) was {analysis_metrics['net_revenue']:.2f} SAR, compared to {previous_metrics['net_revenue']:.2f} SAR in previous week (2026-01-12 to 2026-01-19), representing a {revenue_pct_change:.1f}% change. Refunds included in net calculation.",
            "finding_type": "revenue",
            "metrics": {
                "analysis_period_revenue": {
                    "value": round(analysis_metrics['net_revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "previous_period_revenue": {
                    "value": round(previous_metrics['net_revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-12T00:00:00+03:00",
                    "period_end": "2026-01-19T00:00:00+03:00"
                },
                "absolute_change": {
                    "value": round(revenue_change, 2),
                    "unit": "SAR",
                    "numerator": revenue_change,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "percent_change": {
                    "value": round(revenue_pct_change, 2),
                    "unit": "%",
                    "numerator": revenue_change,
                    "denominator": previous_metrics['net_revenue'],
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "refund_impact": {
                    "value": round(previous_metrics['refund_amount'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-12T00:00:00+03:00",
                    "period_end": "2026-01-19T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": analysis_metrics['total_rows'],
            "coverage_notes": [
                "Analysis period: 2026-01-19 to 2026-01-26",
                "Previous period: 2026-01-12 to 2026-01-19",
                "Net revenue includes refunds as negative values per line_total_sar"
            ],
            "assumptions": [
                "line_total_sar represents realized net revenue including discounts and refunds",
                "All transactions in period are valid and complete"
            ],
            "confidence": 0.95
        })

# Finding 3: Average Order Value change
if previous_metrics['aov'] > 0:
    aov_change = analysis_metrics['aov'] - previous_metrics['aov']
    aov_pct_change = (aov_change / previous_metrics['aov']) * 100
    
    if abs(aov_pct_change) >= 3:  # Lower threshold for AOV
        findings.append({
            "title": "Average Order Value Change vs Previous Week",
            "claim": f"Average order value in analysis period (2026-01-19 to 2026-01-26) was {analysis_metrics['aov']:.2f} SAR, compared to {previous_metrics['aov']:.2f} SAR in previous week (2026-01-12 to 2026-01-19), representing a {aov_pct_change:.1f}% change.",
            "finding_type": "average_order_value",
            "metrics": {
                "analysis_period_aov": {
                    "value": round(analysis_metrics['aov'], 2),
                    "unit": "SAR",
                    "numerator": analysis_metrics['net_revenue'],
                    "denominator": analysis_metrics['valid_transactions'],
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "previous_period_aov": {
                    "value": round(previous_metrics['aov'], 2),
                    "unit": "SAR",
                    "numerator": previous_metrics['net_revenue'],
                    "denominator": previous_metrics['valid_transactions'],
                    "period_start": "2026-01-12T00:00:00+03:00",
                    "period_end": "2026-01-19T00:00:00+03:00"
                },
                "absolute_change": {
                    "value": round(aov_change, 2),
                    "unit": "SAR",
                    "numerator": aov_change,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "percent_change": {
                    "value": round(aov_pct_change, 2),
                    "unit": "%",
                    "numerator": aov_change,
                    "denominator": previous_metrics['aov'],
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": analysis_metrics['total_rows'],
            "coverage_notes": [
                "Analysis period: 2026-01-19 to 2026-01-26",
                "Previous period: 2026-01-12 to 2026-01-19",
                "AOV calculated as net revenue divided by valid transaction count"
            ],
            "assumptions": [
                "Valid transactions represent complete customer baskets",
                "Net revenue includes all discounts and refunds"
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
    json.dump(output, f, indent=2, default=str)