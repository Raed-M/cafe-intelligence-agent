import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse period boundaries
analysis_start = datetime.fromisoformat("2026-01-26T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-02T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-01-19T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-01-26T00:00:00+03:00")

trailing_baselines = [
    (datetime.fromisoformat("2026-01-19T00:00:00+03:00"), datetime.fromisoformat("2026-01-26T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-12T00:00:00+03:00"), datetime.fromisoformat("2026-01-19T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-05T00:00:00+03:00"), datetime.fromisoformat("2026-01-12T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-29T00:00:00+03:00"), datetime.fromisoformat("2026-01-05T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Combine all trailing baseline periods
trailing_data = pd.DataFrame()
for start, end in trailing_baselines:
    period_data = pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] < end)]
    trailing_data = pd.concat([trailing_data, period_data], ignore_index=True)

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
analysis_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0

# Calculate metrics for previous period
previous_valid_txns = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
previous_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()
previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate metrics for trailing baseline (average)
trailing_valid_txns = trailing_data[trailing_data['is_refund'] == False]['transaction_id'].nunique()
trailing_revenue = trailing_data[trailing_data['is_refund'] == False]['line_total_sar'].sum()
trailing_aov = trailing_revenue / trailing_valid_txns if trailing_valid_txns > 0 else 0

# Average per period (4 weeks)
trailing_avg_txns = trailing_valid_txns / 4
trailing_avg_revenue = trailing_revenue / 4
trailing_avg_aov = trailing_aov

# Calculate changes
txn_change = analysis_valid_txns - previous_valid_txns
txn_pct_change = (txn_change / previous_valid_txns * 100) if previous_valid_txns > 0 else 0

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

# Category analysis for analysis period
analysis_category_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data[previous_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Channel analysis
analysis_channel_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum()
previous_channel_revenue = previous_data[previous_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum()

# Product performance - top products in analysis period
analysis_product_revenue = analysis_data[analysis_data['is_refund'] == False].groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_revenue.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']
analysis_product_revenue = analysis_product_revenue.sort_values('revenue', ascending=False)

previous_product_revenue = previous_data[previous_data['is_refund'] == False].groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_revenue.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']
previous_product_revenue = previous_product_revenue.sort_values('revenue', ascending=False)

# Merge to find top product changes
top_products_analysis = analysis_product_revenue.head(5)
top_products_previous = previous_product_revenue.head(5)

# Find products in both periods for comparison
product_comparison = analysis_product_revenue.merge(
    previous_product_revenue[['sku', 'revenue', 'quantity', 'transactions']],
    on='sku',
    suffixes=('_analysis', '_previous'),
    how='inner'
)

if len(product_comparison) > 0:
    product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
    product_comparison['revenue_pct_change'] = (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100)
    product_comparison = product_comparison.sort_values('revenue_change', ascending=False)

# Check for refunds impact
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

findings = []

# Finding 1: Transaction count change
if previous_valid_txns > 0:
    finding1 = {
        "title": "Transaction Count Change Week-over-Week",
        "claim": f"Valid transaction count in analysis period (Jan 26 - Feb 2) was {analysis_valid_txns}, compared to {previous_valid_txns} in previous period (Jan 19-26), representing a {txn_pct_change:.1f}% change.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_period_transactions": {
                "value": analysis_valid_txns,
                "unit": "transactions",
                "numerator": analysis_valid_txns,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_valid_txns,
                "unit": "transactions",
                "numerator": previous_valid_txns,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "transaction_change": {
                "value": txn_change,
                "unit": "transactions",
                "numerator": txn_change,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "transaction_pct_change": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_valid_txns,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            "Excludes refund transactions (is_refund=True)",
            "Uses unique transaction_id for basket counting",
            "Analysis period: 2026-01-26 to 2026-02-02",
            "Previous period: 2026-01-19 to 2026-01-26"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "is_refund flag accurately marks refund transactions",
            "timestamp field is reliable for period filtering"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Revenue change
if previous_revenue > 0:
    finding2 = {
        "title": "Net Revenue Change Week-over-Week",
        "claim": f"Net revenue (excluding refunds) in analysis period was SAR {analysis_revenue:.2f}, compared to SAR {previous_revenue:.2f} in previous period, representing a {revenue_pct_change:.1f}% change.",
        "finding_type": "revenue",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": analysis_revenue,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": previous_revenue,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": revenue_change,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": revenue_change,
                "denominator": previous_revenue,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            "Uses line_total_sar for net revenue calculation",
            "Excludes refund transactions (is_refund=True)",
            f"Refunds in analysis period: SAR {abs(analysis_refunds):.2f}",
            f"Refunds in previous period: SAR {abs(previous_refunds):.2f}"
        ],
        "assumptions": [
            "line_total_sar accurately reflects net transaction value",
            "is_refund flag correctly identifies refund transactions",
            "All transactions have valid line_total_sar values"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Average Order Value change
if previous_aov > 0:
    finding3 = {
        "title": "Average Order Value Change Week-over-Week",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f}, compared to SAR {previous_aov:.2f} in previous period, representing a {aov_pct_change:.1f}% change.",
        "finding_type": "average_order_value",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": analysis_revenue,
                "denominator": analysis_valid_txns,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": previous_revenue,
                "denominator": previous_valid_txns,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": aov_change,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": aov_change,
                "denominator": previous_aov,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            "Calculated as net revenue divided by valid transaction count",
            "Excludes refund transactions",
            "Based on line_total_sar and unique transaction_id"
        ],
        "assumptions": [
            "Each transaction_id represents one basket/order",
            "line_total_sar is accurate for all transactions",
            "is_refund flag correctly identifies refunds"
        ],
        "confidence": 0.95
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)