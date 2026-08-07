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
analysis_start = datetime.fromisoformat("2026-06-22T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-06-29T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-06-15T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-06-22T00:00:00+03:00")

trailing_periods = [
    ("2026-06-15T00:00:00+03:00", "2026-06-22T00:00:00+03:00"),
    ("2026-06-08T00:00:00+03:00", "2026-06-15T00:00:00+03:00"),
    ("2026-06-01T00:00:00+03:00", "2026-06-08T00:00:00+03:00"),
    ("2026-05-25T00:00:00+03:00", "2026-06-01T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period (exclude refunds from transaction counts, but include in revenue)
def filter_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_df = filter_period(pos_df, analysis_start, analysis_end)
previous_df = filter_period(pos_df, previous_start, previous_end)

# Build trailing baseline average
trailing_dfs = []
for period_start_str, period_end_str in trailing_periods:
    p_start = datetime.fromisoformat(period_start_str)
    p_end = datetime.fromisoformat(period_end_str)
    trailing_dfs.append(filter_period(pos_df, p_start, p_end))

trailing_combined = pd.concat(trailing_dfs, ignore_index=True)

# Calculate metrics
def calc_metrics(df, period_name):
    # Valid transactions (non-refund rows)
    valid_txns = df[df['is_refund'] == False]['transaction_id'].nunique()
    
    # Net revenue (includes refunds as negative)
    net_revenue = df['line_total_sar'].sum()
    
    # Average order value (net revenue / valid transactions)
    aov = net_revenue / valid_txns if valid_txns > 0 else 0
    
    # Total line items
    total_items = len(df)
    
    # Refund count
    refund_count = df[df['is_refund'] == True].shape[0]
    
    return {
        'valid_transactions': valid_txns,
        'net_revenue': net_revenue,
        'aov': aov,
        'total_items': total_items,
        'refund_count': refund_count
    }

analysis_metrics = calc_metrics(analysis_df, "analysis")
previous_metrics = calc_metrics(previous_df, "previous")
trailing_metrics = calc_metrics(trailing_combined, "trailing")

# Calculate trailing average
trailing_avg_txns = trailing_metrics['valid_transactions'] / len(trailing_periods)
trailing_avg_revenue = trailing_metrics['net_revenue'] / len(trailing_periods)
trailing_avg_aov = trailing_metrics['aov'] / len(trailing_periods) if trailing_metrics['aov'] > 0 else 0

# Category mix analysis
def get_category_mix(df):
    category_revenue = df.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    category_items = df.groupby('category').size().sort_values(ascending=False)
    return category_revenue, category_items

analysis_cat_rev, analysis_cat_items = get_category_mix(analysis_df)
previous_cat_rev, previous_cat_items = get_category_mix(previous_df)

# Channel mix analysis
def get_channel_mix(df):
    channel_revenue = df.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    channel_txns = df[df['is_refund'] == False].groupby('channel')['transaction_id'].nunique().sort_values(ascending=False)
    return channel_revenue, channel_txns

analysis_ch_rev, analysis_ch_txns = get_channel_mix(analysis_df)
previous_ch_rev, previous_ch_txns = get_channel_mix(previous_df)

# Product performance (top SKUs)
def get_product_performance(df):
    product_revenue = df.groupby(['sku', 'item_name_en']).agg({
        'line_total_sar': 'sum',
        'quantity': 'sum'
    }).sort_values('line_total_sar', ascending=False)
    return product_revenue

analysis_prod = get_product_performance(analysis_df)
previous_prod = get_product_performance(previous_df)

# Prepare findings
findings = []

# Finding 1: Revenue change week-over-week
rev_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
rev_pct_change = (rev_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] != 0 else 0

if abs(rev_pct_change) > 2:  # Only report if >2% change
    findings.append({
        "title": "Net Revenue Change Week-over-Week",
        "claim": f"Net revenue in analysis week (Jun 22-29) was SAR {analysis_metrics['net_revenue']:.2f}, compared to SAR {previous_metrics['net_revenue']:.2f} in previous week (Jun 15-22), representing a {rev_pct_change:.1f}% change.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_net_revenue": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "previous_period_net_revenue": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(rev_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(rev_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_items'],
        "coverage_notes": [
            f"Analysis period includes {analysis_metrics['total_items']} line items from {analysis_metrics['valid_transactions']} valid transactions",
            f"Previous period includes {previous_metrics['total_items']} line items from {previous_metrics['valid_transactions']} valid transactions",
            f"Refunds included in net revenue: {analysis_metrics['refund_count']} refund lines in analysis period"
        ],
        "assumptions": [
            "Valid transactions counted as unique transaction_id where is_refund=False",
            "Net revenue includes refund line items as negative values",
            "Timestamp converted to datetime for period filtering"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction count change
txn_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
txn_pct_change = (txn_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] > 0 else 0

if abs(txn_pct_change) > 2:
    findings.append({
        "title": "Valid Transaction Count Change",
        "claim": f"Valid transaction count in analysis week (Jun 22-29) was {analysis_metrics['valid_transactions']}, compared to {previous_metrics['valid_transactions']} in previous week (Jun 15-22), representing a {txn_pct_change:.1f}% change.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_period_transactions": {
                "value": analysis_metrics['valid_transactions'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_metrics['valid_transactions'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "transaction_change": {
                "value": txn_change,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(txn_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['valid_transactions'],
        "coverage_notes": [
            f"Valid transactions counted as unique transaction_id where is_refund=False",
            f"Analysis period: {analysis_metrics['valid_transactions']} transactions",
            f"Previous period: {previous_metrics['valid_transactions']} transactions"
        ],
        "assumptions": [
            "Transaction_id uniqueness indicates distinct baskets",
            "Refund rows excluded from transaction count"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

if abs(aov_pct_change) > 2:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis week (Jun 22-29) was SAR {analysis_metrics['aov']:.2f}, compared to SAR {previous_metrics['aov']:.2f} in previous week (Jun 15-22), representing a {aov_pct_change:.1f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['net_revenue'], 2),
                "denominator": analysis_metrics['valid_transactions'],
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['net_revenue'], 2),
                "denominator": previous_metrics['valid_transactions'],
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['valid_transactions'],
        "coverage_notes": [
            f"AOV calculated as net revenue / valid transactions",
            f"Analysis period: SAR {round(analysis_metrics['net_revenue'], 2)} / {analysis_metrics['valid_transactions']} transactions",
            f"Previous period: SAR {round(previous_metrics['net_revenue'], 2)} / {previous_metrics['valid_transactions']} transactions"
        ],
        "assumptions": [
            "AOV = net revenue / valid transaction count",
            "Net revenue includes refunds as negative values",
            "Valid transactions exclude refund-only rows"
        ],
        "confidence": 0.95
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)