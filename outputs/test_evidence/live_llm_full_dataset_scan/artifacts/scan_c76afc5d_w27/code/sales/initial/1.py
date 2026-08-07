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
analysis_start = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-13T00:00:00+03:00")

trailing_periods = [
    ("2026-07-06T00:00:00+03:00", "2026-07-13T00:00:00+03:00"),
    ("2026-06-29T00:00:00+03:00", "2026-07-06T00:00:00+03:00"),
    ("2026-06-22T00:00:00+03:00", "2026-06-29T00:00:00+03:00"),
    ("2026-06-15T00:00:00+03:00", "2026-06-22T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter by period
def filter_by_period(df, start_iso, end_iso):
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

analysis_df = filter_by_period(pos_df, "2026-07-13T00:00:00+03:00", "2026-07-20T00:00:00+03:00")
previous_df = filter_by_period(pos_df, "2026-07-06T00:00:00+03:00", "2026-07-13T00:00:00+03:00")

# Trailing baseline: average of 4 weeks
trailing_dfs = []
for start_iso, end_iso in trailing_periods:
    trailing_dfs.append(filter_by_period(pos_df, start_iso, end_iso))
trailing_df = pd.concat(trailing_dfs, ignore_index=True)

# Exclude refunds from net calculations but track them
analysis_sales = analysis_df[analysis_df['is_refund'] == False].copy()
previous_sales = previous_df[previous_df['is_refund'] == False].copy()
trailing_sales = trailing_df[trailing_df['is_refund'] == False].copy()

# Calculate metrics for analysis period
analysis_revenue = analysis_sales['line_total_sar'].sum()
analysis_baskets = analysis_sales['transaction_id'].nunique()
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0

# Calculate metrics for previous period
previous_revenue = previous_sales['line_total_sar'].sum()
previous_baskets = previous_sales['transaction_id'].nunique()
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0

# Calculate metrics for trailing baseline
trailing_revenue = trailing_sales['line_total_sar'].sum()
trailing_baskets = trailing_sales['transaction_id'].nunique()
trailing_aov = trailing_revenue / trailing_baskets if trailing_baskets > 0 else 0

# Revenue change analysis
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0

# Basket count change
basket_change = analysis_baskets - previous_baskets
basket_pct_change = (basket_change / previous_baskets * 100) if previous_baskets > 0 else 0

# AOV change
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

# Product mix analysis
analysis_product_mix = analysis_sales.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'baskets']
analysis_product_mix['pct_revenue'] = (analysis_product_mix['revenue'] / analysis_revenue * 100)

previous_product_mix = previous_sales.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'baskets']
previous_product_mix['pct_revenue'] = (previous_product_mix['revenue'] / previous_revenue * 100)

# Merge with menu for launch/retire dates
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')

analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Category mix analysis
analysis_category_mix = analysis_sales.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_category_mix.columns = ['category', 'revenue', 'baskets']
analysis_category_mix['pct_revenue'] = (analysis_category_mix['revenue'] / analysis_revenue * 100)

previous_category_mix = previous_sales.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_category_mix.columns = ['category', 'revenue', 'baskets']
previous_category_mix['pct_revenue'] = (previous_category_mix['revenue'] / previous_revenue * 100)

# Channel mix analysis
analysis_channel_mix = analysis_sales.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel_mix.columns = ['channel', 'revenue', 'baskets']
analysis_channel_mix['pct_revenue'] = (analysis_channel_mix['revenue'] / analysis_revenue * 100)

previous_channel_mix = previous_sales.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel_mix.columns = ['channel', 'revenue', 'baskets']
previous_channel_mix['pct_revenue'] = (previous_channel_mix['revenue'] / previous_revenue * 100)

# Identify top products by revenue change
product_comparison = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'pct_revenue']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
)
product_comparison['revenue_analysis'] = product_comparison['revenue_analysis'].fillna(0)
product_comparison['revenue_previous'] = product_comparison['revenue_previous'].fillna(0)
product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['revenue_pct_change'] = (
    (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100)
    .where(product_comparison['revenue_previous'] > 0, 0)
)

# Filter for products with meaningful activity
significant_products = product_comparison[
    (product_comparison['revenue_analysis'] > 0) | (product_comparison['revenue_previous'] > 0)
].sort_values('revenue_change', ascending=False)

# Refund analysis
analysis_refunds = analysis_df[analysis_df['is_refund'] == True]
previous_refunds = previous_df[previous_df['is_refund'] == True]
analysis_refund_value = analysis_refunds['line_total_sar'].sum()
previous_refund_value = previous_refunds['line_total_sar'].sum()

findings = []

# Finding 1: Revenue change week-over-week
if previous_revenue > 0:
    findings.append({
        "title": "Weekly Revenue Change",
        "claim": f"Net revenue for week of 2026-07-13 to 2026-07-20 was SAR {analysis_revenue:.2f}, representing a {revenue_pct_change:.1f}% change from previous week (SAR {previous_revenue:.2f}). Refunds totaled SAR {abs(analysis_refund_value):.2f} in analysis period vs SAR {abs(previous_refund_value):.2f} previously.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
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
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_baskets),
        "coverage_notes": [
            f"Analysis period: {len(analysis_sales)} non-refund line items from {int(analysis_baskets)} unique transactions",
            f"Previous period: {len(previous_sales)} non-refund line items from {int(previous_baskets)} unique transactions",
            f"Refunds excluded from net revenue; analysis period refunds: SAR {abs(analysis_refund_value):.2f}"
        ],
        "assumptions": [
            "is_refund flag correctly identifies refund transactions",
            "line_total_sar represents net realized revenue per line item",
            "transaction_id uniquely identifies a basket"
        ],
        "confidence": 0.95
    })

# Finding 2: Average Order Value change
if previous_aov > 0:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value increased from SAR {previous_aov:.2f} (previous week) to SAR {analysis_aov:.2f} (analysis week), a change of SAR {aov_change:.2f} ({aov_pct_change:.1f}%). This reflects {int(analysis_baskets)} baskets in analysis period vs {int(previous_baskets)} in previous period.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": int(analysis_baskets),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": int(previous_baskets),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_baskets),
        "coverage_notes": [
            f"AOV calculated from {len(analysis_sales)} non-refund line items across {int(analysis_baskets)} unique transactions",
            "Refunds excluded from calculation"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "line_total_sar represents net revenue per line item",
            "All transactions in period are valid sales"
        ],
        "confidence": 0.92
    })

# Finding 3: Top category performance change
category_comparison = analysis_category_mix.merge(
    previous_category_mix[['category', 'revenue', 'pct_revenue']],
    on='category',
    how='outer',
    suffixes=('_analysis', '_previous')
)
category_comparison['revenue_analysis'] = category_comparison['revenue_analysis'].fillna(0)
category_comparison['revenue_previous'] = category_comparison['revenue_previous'].fillna(0)
category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (
    (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100)
    .where(category_comparison['revenue_previous'] > 0, 0)
)

top_category_change = category_comparison.loc[category_comparison['revenue_change'].idxmax()]

if top_category_change['revenue_previous'] > 0:
    findings.append({
        "title": "Top Category Revenue Growth",
        "claim": f"Category '{top_category_change['category']}' generated SAR {top_category_change['revenue_analysis']:.2f} in analysis week vs SAR {top_category_change['revenue_previous']:.2f} previously, a {top_category_change['revenue_pct_change']:.1f}% increase. This category represented {top_category_change['pct_revenue_analysis']:.1f}% of weekly revenue.",
        "finding_type": "category_mix_change",
        "metrics": {
            "category_analysis_revenue": {
                "value": round(top_category_change['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "category_previous_revenue": {
                "value": round(top_category_change['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "category_revenue_change_sar": {
                "value": round(top_category_change['revenue_change'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "category_revenue_change_pct": {
                "value": round(top_category_change['revenue_pct_change'], 1),
                "unit": "%",
                "numerator": round(top_category_change['revenue_change'], 2),
                "denominator": round(top_category_change['revenue_previous'], 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "category_pct_of_weekly_revenue": {
                "value": round(top_category_change['pct_revenue_analysis'], 1),
                "unit": "%",
                "numerator": round(top_category_change['revenue_analysis'], 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_sales[analysis_sales['category'] == top_category_change['category']]),
        "coverage_notes": [
            f"Category data sourced from POS line items with category field populated",
            f"Analysis period: {len(analysis_sales[analysis_sales['category'] == top_category_change['category']])} line items in this category",
            "Refunds excluded from revenue calculations"
        ],
        "assumptions": [
            "category field in POS correctly classifies products",
            "line_total_sar represents net revenue",
            "No material data quality issues in category assignment"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)