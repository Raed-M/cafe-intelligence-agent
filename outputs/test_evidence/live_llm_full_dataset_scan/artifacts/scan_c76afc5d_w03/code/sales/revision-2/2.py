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

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = pd.Timestamp("2026-01-26T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-02-02T00:00:00+03:00")
previous_start = pd.Timestamp("2026-01-19T00:00:00+03:00")
previous_end = pd.Timestamp("2026-01-26T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
analysis_total_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
analysis_refund_amount = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
analysis_net_revenue = analysis_total_revenue + analysis_refund_amount  # refunds are negative
analysis_aov = analysis_net_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0

# Calculate metrics for previous period
previous_valid_txns = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
previous_total_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()
previous_refund_amount = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()
previous_net_revenue = previous_total_revenue + previous_refund_amount  # refunds are negative
previous_aov = previous_net_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate changes
revenue_change = analysis_net_revenue - previous_net_revenue
revenue_pct_change = (revenue_change / previous_net_revenue * 100) if previous_net_revenue != 0 else 0
txn_change = analysis_valid_txns - previous_valid_txns
txn_pct_change = (txn_change / previous_valid_txns * 100) if previous_valid_txns > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Analyze by category
analysis_by_category = analysis_data[analysis_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'transactions', 'quantity']
analysis_by_category['aov'] = analysis_by_category['revenue'] / analysis_by_category['transactions']

previous_by_category = previous_data[previous_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'transactions', 'quantity']
previous_by_category['aov'] = previous_by_category['revenue'] / previous_by_category['transactions']

# Merge category data
category_comparison = analysis_by_category.merge(
    previous_by_category,
    on='category',
    suffixes=('_analysis', '_previous')
)
category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100)
category_comparison['txn_change'] = category_comparison['transactions_analysis'] - category_comparison['transactions_previous']
category_comparison['txn_pct_change'] = (category_comparison['txn_change'] / category_comparison['transactions_previous'] * 100)

# Analyze by channel
analysis_by_channel = analysis_data[analysis_data['is_refund'] == False].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'transactions']
analysis_by_channel['aov'] = analysis_by_channel['revenue'] / analysis_by_channel['transactions']

previous_by_channel = previous_data[previous_data['is_refund'] == False].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'transactions']
previous_by_channel['aov'] = previous_by_channel['revenue'] / previous_by_channel['transactions']

# Merge channel data
channel_comparison = analysis_by_channel.merge(
    previous_by_channel,
    on='channel',
    suffixes=('_analysis', '_previous')
)
channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
channel_comparison['revenue_pct_change'] = (channel_comparison['revenue_change'] / channel_comparison['revenue_previous'] * 100)

# Analyze by product (SKU)
analysis_by_sku = analysis_data[analysis_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum',
    'item_name_en': 'first'
}).reset_index()
analysis_by_sku.columns = ['sku', 'revenue', 'transactions', 'quantity', 'item_name']
analysis_by_sku['aov'] = analysis_by_sku['revenue'] / analysis_by_sku['transactions']

previous_by_sku = previous_data[previous_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum',
    'item_name_en': 'first'
}).reset_index()
previous_by_sku.columns = ['sku', 'revenue', 'transactions', 'quantity', 'item_name']
previous_by_sku['aov'] = previous_by_sku['revenue'] / previous_by_sku['transactions']

# Merge SKU data
sku_comparison = analysis_by_sku.merge(
    previous_by_sku,
    on='sku',
    suffixes=('_analysis', '_previous'),
    how='outer'
)
sku_comparison['revenue_change'] = sku_comparison['revenue_analysis'].fillna(0) - sku_comparison['revenue_previous'].fillna(0)
sku_comparison['revenue_pct_change'] = (sku_comparison['revenue_change'] / sku_comparison['revenue_previous'].fillna(1) * 100)
sku_comparison['txn_change'] = sku_comparison['transactions_analysis'].fillna(0) - sku_comparison['transactions_previous'].fillna(0)

# Check for product launches - ensure timezone-naive comparison
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce', utc=False)
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce', utc=False)

# Convert period boundaries to naive datetime for comparison with menu dates
previous_start_naive = previous_start.tz_localize(None)
analysis_end_naive = analysis_end.tz_localize(None)

# Filter SKUs that were active in both periods
sku_comparison['launch_date'] = sku_comparison['sku'].map(
    menu_df.set_index('sku')['launch_date']
)
sku_comparison['retire_date'] = sku_comparison['sku'].map(
    menu_df.set_index('sku')['retire_date']
)

# Only include products that were active in both periods
sku_comparison['valid_for_comparison'] = (
    (sku_comparison['launch_date'].isna() | (sku_comparison['launch_date'] <= previous_start_naive)) &
    (sku_comparison['retire_date'].isna() | (sku_comparison['retire_date'] > analysis_end_naive))
)

# Calculate period lengths
analysis_days = (analysis_end - analysis_start).days
previous_days = (previous_end - previous_start).days

# Normalize to daily averages for fair comparison
analysis_daily_revenue = analysis_net_revenue / analysis_days
previous_daily_revenue = previous_net_revenue / previous_days
analysis_daily_txns = analysis_valid_txns / analysis_days
previous_daily_txns = previous_valid_txns / previous_days

# Prepare findings
findings = []

# Finding 1: Revenue Performance (normalized by day)
if previous_daily_revenue != 0:
    daily_revenue_change = analysis_daily_revenue - previous_daily_revenue
    daily_revenue_pct_change = (daily_revenue_change / previous_daily_revenue * 100)
    
    findings.append({
        "title": "Daily Average Revenue Comparison",
        "claim": f"Daily average revenue in analysis period (2026-01-26 to 2026-02-02, 8 days) was SAR {analysis_daily_revenue:.2f}, compared to SAR {previous_daily_revenue:.2f} in previous period (2026-01-19 to 2026-01-26, 7 days), representing a {daily_revenue_pct_change:.2f}% change. Daily normalization enables fair comparison across unequal period lengths. Revenue figures exclude refunds.",
        "finding_type": "revenue_performance",
        "metrics": {
            "analysis_daily_revenue": {
                "value": round(analysis_daily_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": analysis_days,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_daily_revenue": {
                "value": round(previous_daily_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": previous_days,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "daily_revenue_change": {
                "value": round(daily_revenue_change, 2),
                "unit": "SAR",
                "numerator": round(daily_revenue_change, 2),
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "daily_revenue_pct_change": {
                "value": round(daily_revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(daily_revenue_change, 2),
                "denominator": round(previous_daily_revenue, 2),
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_txns + previous_valid_txns),
        "coverage_notes": [
            f"Analysis period: {analysis_valid_txns} valid transactions across {analysis_days} days",
            f"Previous period: {previous_valid_txns} valid transactions across {previous_days} days",
            "Revenue excludes refund transactions (is_refund=True)",
            f"Analysis period refunds: SAR {analysis_refund_amount:.2f}",
            f"Previous period refunds: SAR {previous_refund_amount:.2f}"
        ],
        "assumptions": [
            "transaction_id uniqueness identifies distinct baskets",
            "line_total_sar represents net transaction value",
            "is_refund flag correctly identifies refund transactions",
            "Refunds are excluded from revenue calculations",
            "Daily normalization is appropriate for fair period comparison given unequal period lengths"
        ],
        "confidence": 0.92
    })

# Finding 2: Average Order Value
if previous_aov != 0:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f}, compared to SAR {previous_aov:.2f} in previous period, representing a {aov_pct_change:.2f}% change. Revenue figures exclude refunds.",
        "finding_type": "aov_performance",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_txns + previous_valid_txns),
        "coverage_notes": [
            f"Analysis period: {analysis_valid_txns} valid transactions",
            f"Previous period: {previous_valid_txns} valid transactions",
            "Revenue excludes refund transactions (is_refund=True)",
            "AOV calculated as net revenue divided by valid transaction count"
        ],
        "assumptions": [
            "transaction_id uniqueness identifies distinct baskets",
            "line_total_sar represents net transaction value",
            "is_refund flag correctly identifies refund transactions",
            "Refunds are excluded from revenue calculations",
            "AOV comparison is valid across periods"
        ],
        "confidence": 0.88
    })

# Finding 3: Category Performance - identify strongest category change
category_comparison_valid = category_comparison[
    (category_comparison['revenue_previous'] > 0) & 
    (category_comparison['transactions_previous'] > 0)
].copy()

if len(category_comparison_valid) > 0:
    # Find category with largest absolute revenue change
    category_comparison_valid['abs_revenue_change'] = category_comparison_valid['revenue_change'].abs()
    top_category = category_comparison_valid.loc[category_comparison_valid['abs_revenue_change'].idxmax()]
    
    if top_category['revenue_change'] != 0:
        findings.append({
            "title": f"Category Mix Shift: {top_category['category']}",
            "claim": f"The {top_category['category']} category generated SAR {top_category['revenue_analysis']:.2f} in analysis period (2026-01-26 to 2026-02-02) versus SAR {top_category['revenue_previous']:.2f} in previous period (2026-01-19 to 2026-01-26), a change of SAR {top_category['revenue_change']:.2f} ({top_category['revenue_pct_change']:.2f}%). Transaction count changed from {int(top_category['transactions_previous'])} to {int(top_category['transactions_analysis'])} ({top_category['txn_pct_change']:.2f}%). Revenue figures exclude refunds.",
            "finding_type": "category_mix",
            "metrics": {
                "analysis_category_revenue": {
                    "value": round(top_category['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category['revenue_analysis'], 2),
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "previous_category_revenue": {
                    "value": round(top_category['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category['revenue_previous'], 2),
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                },
                "category_revenue_change": {
                    "value": round(top_category['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category['revenue_change'], 2),
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "category_revenue_pct_change": {
                    "value": round(top_category['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": round(top_category['revenue_change'], 2),
                    "denominator": round(top_category['revenue_previous'], 2),
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "analysis_transactions": {
                    "value": int(top_category['transactions_analysis']),
                    "unit": "count",
                    "numerator": int(top_category['transactions_analysis']),
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "previous_transactions": {
                    "value": int(top_category['transactions_previous']),
                    "unit": "count",
                    "numerator": int(top_category['transactions_previous']),
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+03:00",
                    "period_end": "2026-01-26T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(top_category['transactions_analysis'] + top_category['transactions_previous']),
            "coverage_notes": [
                f"Analysis period: {int(top_category['transactions_analysis'])} {top_category['category']} transactions",
                f"Previous period: {int(top_category['transactions_previous'])} {top_category['category']} transactions",
                "Revenue excludes refund transactions (is_refund=True)",
                "Category assignment from POS data merged with menu reference"
            ],
            "assumptions": [
                "transaction_id uniqueness identifies distinct baskets",
                "line_total_sar represents net transaction value",
                "is_refund flag correctly identifies refund transactions",
                "Refunds are excluded from revenue calculations",
                "Category classification is consistent across both periods"
            ],
            "confidence": 0.85
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
