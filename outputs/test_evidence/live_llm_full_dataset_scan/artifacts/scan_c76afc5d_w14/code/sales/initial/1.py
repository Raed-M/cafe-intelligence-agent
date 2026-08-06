import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-04-13T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-06T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-13T00:00:00+03:00")

trailing_baselines = [
    (datetime.fromisoformat("2026-04-06T00:00:00+03:00"), datetime.fromisoformat("2026-04-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-30T00:00:00+03:00"), datetime.fromisoformat("2026-04-06T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-23T00:00:00+03:00"), datetime.fromisoformat("2026-03-30T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-16T00:00:00+03:00"), datetime.fromisoformat("2026-03-23T00:00:00+03:00")),
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
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

# Calculate metrics for previous period
previous_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

# Calculate metrics for trailing baseline (average)
trailing_transactions = trailing_data['transaction_id'].nunique()
trailing_revenue = trailing_data['line_total_sar'].sum()
trailing_aov = trailing_revenue / trailing_transactions if trailing_transactions > 0 else 0

# Calculate average per period for trailing baseline
num_trailing_periods = len(trailing_baselines)
avg_trailing_transactions = trailing_transactions / num_trailing_periods
avg_trailing_revenue = trailing_revenue / num_trailing_periods
avg_trailing_aov = trailing_aov

# Calculate changes
revenue_change_vs_previous = analysis_revenue - previous_revenue
revenue_pct_change_vs_previous = (revenue_change_vs_previous / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change_vs_previous = analysis_transactions - previous_transactions
transaction_pct_change_vs_previous = (transaction_change_vs_previous / previous_transactions * 100) if previous_transactions > 0 else 0

aov_change_vs_previous = analysis_aov - previous_aov
aov_pct_change_vs_previous = (aov_change_vs_previous / previous_aov * 100) if previous_aov != 0 else 0

# Revenue change vs trailing baseline
revenue_change_vs_trailing = analysis_revenue - avg_trailing_revenue
revenue_pct_change_vs_trailing = (revenue_change_vs_trailing / avg_trailing_revenue * 100) if avg_trailing_revenue != 0 else 0

# Transaction change vs trailing baseline
transaction_change_vs_trailing = analysis_transactions - avg_trailing_transactions
transaction_pct_change_vs_trailing = (transaction_change_vs_trailing / avg_trailing_transactions * 100) if avg_trailing_transactions > 0 else 0

# Analyze by category
analysis_by_category = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'transactions', 'quantity']

previous_by_category = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'transactions', 'quantity']

# Merge category data
category_comparison = analysis_by_category.merge(
    previous_by_category,
    on='category',
    suffixes=('_analysis', '_previous')
)

category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100).fillna(0)

# Analyze by channel
analysis_by_channel = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'transactions']

previous_by_channel = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'transactions']

# Merge channel data
channel_comparison = analysis_by_channel.merge(
    previous_by_channel,
    on='channel',
    suffixes=('_analysis', '_previous')
)

channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
channel_comparison['revenue_pct_change'] = (channel_comparison['revenue_change'] / channel_comparison['revenue_previous'] * 100).fillna(0)

# Analyze top products
analysis_by_sku = analysis_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_sku.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']
analysis_by_sku = analysis_by_sku.sort_values('revenue', ascending=False)

previous_by_sku = previous_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_sku.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']

# Merge SKU data
sku_comparison = analysis_by_sku.merge(
    previous_by_sku,
    on=['sku', 'item_name'],
    suffixes=('_analysis', '_previous'),
    how='left'
)

sku_comparison['revenue_previous'] = sku_comparison['revenue_previous'].fillna(0)
sku_comparison['quantity_previous'] = sku_comparison['quantity_previous'].fillna(0)
sku_comparison['revenue_change'] = sku_comparison['revenue_analysis'] - sku_comparison['revenue_previous']
sku_comparison['revenue_pct_change'] = (sku_comparison['revenue_change'] / sku_comparison['revenue_previous'] * 100).fillna(0)

# Check for refunds
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Prepare findings
findings = []

# Finding 1: Revenue change vs previous week
if abs(revenue_pct_change_vs_previous) > 0.1:  # More than 0.1% change
    findings.append({
        "title": "Revenue Performance vs Previous Week",
        "claim": f"Total revenue in analysis period (2026-04-13 to 2026-04-20) was SAR {analysis_revenue:.2f}, representing a {revenue_pct_change_vs_previous:.1f}% change from previous week's SAR {previous_revenue:.2f}.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "revenue_change": {
                "value": round(revenue_change_vs_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change_vs_previous, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Analysis period transactions: {analysis_transactions}",
            f"Previous period transactions: {previous_transactions}",
            f"Analysis period refunds: SAR {analysis_refunds:.2f}",
            f"Previous period refunds: SAR {previous_refunds:.2f}"
        ],
        "assumptions": [
            "Revenue includes refunds as negative values",
            "Transaction counted by unique transaction_id",
            "All line_total_sar values used as-is from cleaned POS data"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction count change
if abs(transaction_pct_change_vs_previous) > 0.1:
    findings.append({
        "title": "Transaction Volume Change",
        "claim": f"Transaction count in analysis period was {analysis_transactions}, representing a {transaction_pct_change_vs_previous:.1f}% change from previous week's {previous_transactions} transactions.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "transaction_change": {
                "value": transaction_change_vs_previous,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "transaction_pct_change": {
                "value": round(transaction_pct_change_vs_previous, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Transactions counted by unique transaction_id",
            f"Analysis period: {len(analysis_data)} line items from {analysis_transactions} transactions",
            f"Previous period: {len(previous_data)} line items from {previous_transactions} transactions"
        ],
        "assumptions": [
            "Each unique transaction_id represents one basket/transaction",
            "Refunds included in transaction count"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value change
if abs(aov_pct_change_vs_previous) > 0.1:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f}, representing a {aov_pct_change_vs_previous:.1f}% change from previous week's SAR {previous_aov:.2f}.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change_vs_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change_vs_previous, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"AOV calculated as total revenue / unique transaction count",
            f"Analysis period: SAR {analysis_revenue:.2f} / {analysis_transactions} transactions",
            f"Previous period: SAR {previous_revenue:.2f} / {previous_transactions} transactions"
        ],
        "assumptions": [
            "AOV = Total Revenue / Transaction Count",
            "Revenue includes refunds as negative values",
            "Transaction counted by unique transaction_id"
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
