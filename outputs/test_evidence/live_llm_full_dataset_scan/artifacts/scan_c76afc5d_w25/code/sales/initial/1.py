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
analysis_start = datetime.fromisoformat("2026-06-29T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-06-22T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-06-29T00:00:00+03:00")

trailing_baselines = [
    (datetime.fromisoformat("2026-06-22T00:00:00+03:00"), datetime.fromisoformat("2026-06-29T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-15T00:00:00+03:00"), datetime.fromisoformat("2026-06-22T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-08T00:00:00+03:00"), datetime.fromisoformat("2026-06-15T00:00:00+03:00")),
    (datetime.fromisoformat("2026-06-01T00:00:00+03:00"), datetime.fromisoformat("2026-06-08T00:00:00+03:00")),
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
def calculate_metrics(data, period_start, period_end):
    # Count valid transactions (unique transaction_id, excluding refunds for transaction count)
    valid_txns = data[~data['is_refund']]['transaction_id'].nunique()
    
    # Net revenue (includes refunds as negative)
    net_revenue = data['line_total_sar'].sum()
    
    # Average order value
    aov = net_revenue / valid_txns if valid_txns > 0 else 0
    
    # Total line items
    total_items = len(data)
    
    # Refund count
    refund_count = data[data['is_refund']]['transaction_id'].nunique()
    
    return {
        'valid_transactions': valid_txns,
        'net_revenue': net_revenue,
        'aov': aov,
        'total_items': total_items,
        'refund_transactions': refund_count,
        'period_start': period_start.isoformat(),
        'period_end': period_end.isoformat()
    }

# Calculate metrics for all periods
analysis_metrics = calculate_metrics(analysis_data, analysis_start, analysis_end)
previous_metrics = calculate_metrics(previous_data, previous_start, previous_end)
trailing_metrics = [calculate_metrics(data, start, end) for data, (start, end) in zip(trailing_data, trailing_baselines)]

# Calculate average of trailing baseline
avg_trailing_txns = np.mean([m['valid_transactions'] for m in trailing_metrics])
avg_trailing_revenue = np.mean([m['net_revenue'] for m in trailing_metrics])
avg_trailing_aov = np.mean([m['aov'] for m in trailing_metrics])

findings = []

# Finding 1: Transaction count change (analysis vs previous week)
txn_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
txn_pct_change = (txn_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] > 0 else 0

if abs(txn_pct_change) >= 5:  # Threshold for meaningful change
    findings.append({
        "title": "Transaction Volume Change Week-over-Week",
        "claim": f"Valid transaction count changed from {previous_metrics['valid_transactions']} (previous week) to {analysis_metrics['valid_transactions']} (analysis week), a {txn_pct_change:.1f}% change.",
        "finding_type": "transaction_count",
        "metrics": {
            "analysis_week_transactions": {
                "value": analysis_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            },
            "previous_week_transactions": {
                "value": previous_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": previous_metrics['period_start'],
                "period_end": previous_metrics['period_end']
            },
            "transaction_change_pct": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_metrics['valid_transactions'],
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_items'],
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Previous period: 2026-06-22 to 2026-06-29",
            "Transactions counted as unique transaction_id excluding refunds"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "is_refund flag correctly identifies refund transactions",
            "All transactions in both periods are valid and comparable"
        ],
        "confidence": 0.95
    })

# Finding 2: Revenue change (analysis vs previous week)
revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
revenue_pct_change = (revenue_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] > 0 else 0

if abs(revenue_pct_change) >= 5:  # Threshold for meaningful change
    findings.append({
        "title": "Net Revenue Change Week-over-Week",
        "claim": f"Net revenue changed from {previous_metrics['net_revenue']:.2f} SAR (previous week) to {analysis_metrics['net_revenue']:.2f} SAR (analysis week), a {revenue_pct_change:.1f}% change.",
        "finding_type": "revenue",
        "metrics": {
            "analysis_week_revenue": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            },
            "previous_week_revenue": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_metrics['period_start'],
                "period_end": previous_metrics['period_end']
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_metrics['net_revenue'], 2),
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_items'],
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Previous period: 2026-06-22 to 2026-06-29",
            "Net revenue includes refunds as negative values per line_total_sar"
        ],
        "assumptions": [
            "line_total_sar accurately reflects net transaction value",
            "Refunds are properly captured in line_total_sar",
            "All transactions are in SAR currency"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

if abs(aov_pct_change) >= 3:  # Lower threshold for AOV
    findings.append({
        "title": "Average Order Value Change Week-over-Week",
        "claim": f"Average order value changed from {previous_metrics['aov']:.2f} SAR (previous week) to {analysis_metrics['aov']:.2f} SAR (analysis week), a {aov_pct_change:.1f}% change.",
        "finding_type": "average_order_value",
        "metrics": {
            "analysis_week_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['net_revenue'], 2),
                "denominator": analysis_metrics['valid_transactions'],
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            },
            "previous_week_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['net_revenue'], 2),
                "denominator": previous_metrics['valid_transactions'],
                "period_start": previous_metrics['period_start'],
                "period_end": previous_metrics['period_end']
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_metrics['aov'], 2),
                "period_start": analysis_metrics['period_start'],
                "period_end": analysis_metrics['period_end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_items'],
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Previous period: 2026-06-22 to 2026-06-29",
            "AOV calculated as net revenue divided by valid transaction count"
        ],
        "assumptions": [
            "Valid transactions are those with non-refund line items",
            "line_total_sar includes all discounts and refunds",
            "Transaction count is stable and comparable across periods"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)