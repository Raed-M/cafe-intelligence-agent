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
analysis_start = datetime.fromisoformat("2026-03-02T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-02-23T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-03-02T00:00:00+03:00")

trailing_periods = [
    ("2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00"),
    ("2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00"),
    ("2026-02-09T00:00:00+03:00", "2026-02-16T00:00:00+03:00"),
    ("2026-02-02T00:00:00+03:00", "2026-02-09T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period (exclude refunds from transaction counts, but include in revenue)
def filter_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_df = filter_period(pos_df, analysis_start, analysis_end)
previous_df = filter_period(pos_df, previous_start, previous_end)

# Calculate trailing baseline average
trailing_dfs = []
for period_start_str, period_end_str in trailing_periods:
    p_start = datetime.fromisoformat(period_start_str)
    p_end = datetime.fromisoformat(period_end_str)
    trailing_dfs.append(filter_period(pos_df, p_start, p_end))

trailing_combined = pd.concat(trailing_dfs, ignore_index=True)

# Helper function to calculate metrics
def calculate_metrics(df, period_name):
    # Valid transactions (non-refund rows)
    valid_txns = df[df['is_refund'] == False]['transaction_id'].nunique()
    
    # Net revenue (includes refunds as negative)
    net_revenue = df['line_total_sar'].sum()
    
    # Average order value
    aov = net_revenue / valid_txns if valid_txns > 0 else 0
    
    # Total line items
    total_items = len(df)
    
    return {
        'valid_transactions': valid_txns,
        'net_revenue': net_revenue,
        'aov': aov,
        'total_line_items': total_items
    }

analysis_metrics = calculate_metrics(analysis_df, "analysis")
previous_metrics = calculate_metrics(previous_df, "previous")
trailing_metrics = calculate_metrics(trailing_combined, "trailing")

# Calculate changes
revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
revenue_pct_change = (revenue_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] != 0 else 0

txn_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
txn_pct_change = (txn_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] > 0 else 0

aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

# Category mix analysis
def get_category_mix(df):
    category_revenue = df.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    category_txns = df[df['is_refund'] == False].groupby('category')['transaction_id'].nunique()
    return category_revenue, category_txns

analysis_cat_rev, analysis_cat_txns = get_category_mix(analysis_df)
previous_cat_rev, previous_cat_txns = get_category_mix(previous_df)

# Product performance (top SKUs)
def get_product_performance(df):
    product_data = df.groupby(['sku', 'item_name_en']).agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': lambda x: x[df.loc[x.index, 'is_refund'] == False].nunique()
    }).rename(columns={'transaction_id': 'valid_txns'})
    return product_data.sort_values('line_total_sar', ascending=False)

analysis_products = get_product_performance(analysis_df)
previous_products = get_product_performance(previous_df)

# Channel mix
def get_channel_mix(df):
    channel_revenue = df.groupby('channel')['line_total_sar'].sum()
    channel_txns = df[df['is_refund'] == False].groupby('channel')['transaction_id'].nunique()
    return channel_revenue, channel_txns

analysis_ch_rev, analysis_ch_txns = get_channel_mix(analysis_df)
previous_ch_rev, previous_ch_txns = get_channel_mix(previous_df)

# Identify findings
findings = []

# Finding 1: Revenue change
if abs(revenue_pct_change) > 0.1:  # More than 0.1% change
    findings.append({
        "title": "Net Revenue Change Week-over-Week",
        "claim": f"Net revenue in analysis period (2026-03-02 to 2026-03-09) was SAR {analysis_metrics['net_revenue']:.2f}, compared to SAR {previous_metrics['net_revenue']:.2f} in previous period (2026-02-23 to 2026-03-02), representing a {revenue_pct_change:.2f}% change of SAR {revenue_change:.2f}.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_net_revenue": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "previous_period_net_revenue": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_df),
        "coverage_notes": [
            f"Analysis period contains {len(analysis_df)} line items from {analysis_metrics['valid_transactions']} valid transactions",
            f"Previous period contains {len(previous_df)} line items from {previous_metrics['valid_transactions']} valid transactions",
            "Refunds included in net revenue calculations as negative values"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue after discounts",
            "is_refund flag correctly identifies refund transactions",
            "transaction_id uniquely identifies a basket"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction count change
if abs(txn_pct_change) > 0.1:
    findings.append({
        "title": "Valid Transaction Count Change",
        "claim": f"Valid transaction count (non-refund baskets) in analysis period was {analysis_metrics['valid_transactions']}, compared to {previous_metrics['valid_transactions']} in previous period, representing a {txn_pct_change:.2f}% change of {txn_change} transactions.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_period_valid_txns": {
                "value": analysis_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "previous_period_valid_txns": {
                "value": previous_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "txn_change": {
                "value": txn_change,
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "txn_change_pct": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_df),
        "coverage_notes": [
            f"Analysis period: {analysis_metrics['valid_transactions']} unique transaction_ids with is_refund=False",
            f"Previous period: {previous_metrics['valid_transactions']} unique transaction_ids with is_refund=False"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "is_refund flag correctly identifies refund transactions"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value change
if abs(aov_pct_change) > 0.1:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period was SAR {analysis_metrics['aov']:.2f}, compared to SAR {previous_metrics['aov']:.2f} in previous period, representing a {aov_pct_change:.2f}% change of SAR {aov_change:.2f}.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['net_revenue'], 2),
                "denominator": analysis_metrics['valid_transactions'],
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['net_revenue'], 2),
                "denominator": previous_metrics['valid_transactions'],
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['valid_transactions'],
        "coverage_notes": [
            f"AOV calculated as net revenue / valid transactions",
            f"Analysis period: {analysis_metrics['valid_transactions']} valid transactions",
            f"Previous period: {previous_metrics['valid_transactions']} valid transactions"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue after discounts",
            "is_refund flag correctly identifies refund transactions",
            "transaction_id uniquely identifies a basket"
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