import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
analysis_end = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_start = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_end = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Calculate metrics for analysis period
analysis_valid_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_refund_count = analysis_data[analysis_data['is_refund'] == True].shape[0]
analysis_refund_value = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()

# Calculate metrics for previous period
previous_valid_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_refund_count = previous_data[previous_data['is_refund'] == True].shape[0]
previous_refund_value = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Calculate AOV
analysis_aov = analysis_revenue / analysis_valid_transactions if analysis_valid_transactions > 0 else 0
previous_aov = previous_revenue / previous_valid_transactions if previous_valid_transactions > 0 else 0

# Calculate trailing baseline (average of 4 weeks)
trailing_baseline_revenue = 0
trailing_baseline_transactions = 0
trailing_baseline_count = 0

for period in [
    (datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))))
]:
    period_data = pos_df[(pos_df['timestamp'] >= period[0]) & (pos_df['timestamp'] < period[1])]
    trailing_baseline_revenue += period_data['line_total_sar'].sum()
    trailing_baseline_transactions += period_data['transaction_id'].nunique()
    trailing_baseline_count += 1

trailing_baseline_revenue_avg = trailing_baseline_revenue / trailing_baseline_count if trailing_baseline_count > 0 else 0
trailing_baseline_transactions_avg = trailing_baseline_transactions / trailing_baseline_count if trailing_baseline_count > 0 else 0
trailing_baseline_aov = trailing_baseline_revenue_avg / trailing_baseline_transactions_avg if trailing_baseline_transactions_avg > 0 else 0

# Analyze product mix changes
analysis_product_mix = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

previous_product_mix = previous_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

# Merge with menu to get product names and launch dates
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Find top products by revenue change
product_comparison = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'transactions']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
)
product_comparison['revenue_analysis'] = product_comparison['revenue_analysis'].fillna(0)
product_comparison['revenue_previous'] = product_comparison['revenue_previous'].fillna(0)
product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['revenue_pct_change'] = (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100).replace([np.inf, -np.inf], 0)

# Analyze channel mix
analysis_channel_mix = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel_mix.columns = ['channel', 'revenue', 'transactions']

previous_channel_mix = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel_mix.columns = ['channel', 'revenue', 'transactions']

# Analyze category mix
analysis_category_mix = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_category_mix.columns = ['category', 'revenue', 'transactions']

previous_category_mix = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_category_mix.columns = ['category', 'revenue', 'transactions']

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
transaction_change = analysis_valid_transactions - previous_valid_transactions
transaction_pct_change = (transaction_change / previous_valid_transactions * 100) if previous_valid_transactions > 0 else 0

if abs(revenue_pct_change) > 5 or abs(transaction_pct_change) > 5:
    findings.append({
        "title": "Revenue and Transaction Volume Change",
        "claim": f"Week of 2026-06-15 to 2026-06-22 showed revenue of {analysis_revenue:.2f} SAR ({revenue_pct_change:+.1f}% vs previous week) with {analysis_valid_transactions} valid transactions ({transaction_pct_change:+.1f}% vs previous week). Average order value changed from {previous_aov:.2f} SAR to {analysis_aov:.2f} SAR.",
        "finding_type": "revenue_and_transaction_change",
        "metrics": {
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "previous_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+03:00",
                "period_end": "2026-06-15T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "analysis_transactions": {
                "value": analysis_valid_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "previous_transactions": {
                "value": previous_valid_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+03:00",
                "period_end": "2026-06-15T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(transaction_pct_change, 2),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_valid_transactions,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_valid_transactions,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_valid_transactions,
                "period_start": "2026-06-08T00:00:00+03:00",
                "period_end": "2026-06-15T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_transactions,
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22 (7 days)",
            f"Previous period: 2026-06-08 to 2026-06-15 (7 days)",
            f"Refunds included in net revenue: {analysis_refund_count} refunds totaling {analysis_refund_value:.2f} SAR in analysis period"
        ],
        "assumptions": [
            "transaction_id uniqueness indicates valid basket",
            "line_total_sar represents net revenue after discounts",
            "Refunds are included in net calculations as per metric definitions"
        ],
        "confidence": 0.95
    })

# Finding 2: Category Mix Changes
category_comparison = analysis_category_mix.merge(
    previous_category_mix,
    on='category',
    how='outer',
    suffixes=('_analysis', '_previous')
)
category_comparison['revenue_analysis'] = category_comparison['revenue_analysis'].fillna(0)
category_comparison['revenue_previous'] = category_comparison['revenue_previous'].fillna(0)
category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100).replace([np.inf, -np.inf], 0)

# Find category with largest change
largest_category_change = category_comparison.loc[category_comparison['revenue_change'].abs().idxmax()]

if abs(largest_category_change['revenue_pct_change']) > 10:
    findings.append({
        "title": "Category Mix Shift",
        "claim": f"Category '{largest_category_change['category']}' revenue changed from {largest_category_change['revenue_previous']:.2f} SAR to {largest_category_change['revenue_analysis']:.2f} SAR ({largest_category_change['revenue_pct_change']:+.1f}%) between analysis and previous week.",
        "finding_type": "category_mix_change",
        "metrics": {
            "category_name": {
                "value": str(largest_category_change['category']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "analysis_category_revenue": {
                "value": round(largest_category_change['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "previous_category_revenue": {
                "value": round(largest_category_change['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+03:00",
                "period_end": "2026-06-15T00:00:00+03:00"
            },
            "category_revenue_change_pct": {
                "value": round(largest_category_change['revenue_pct_change'], 2),
                "unit": "%",
                "numerator": round(largest_category_change['revenue_change'], 2),
                "denominator": round(largest_category_change['revenue_previous'], 2),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(largest_category_change['transactions_analysis']) if pd.notna(largest_category_change['transactions_analysis']) else None,
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Previous period: 2026-06-08 to 2026-06-15",
            f"Category transactions in analysis period: {int(largest_category_change['transactions_analysis']) if pd.notna(largest_category_change['transactions_analysis']) else 'N/A'}"
        ],
        "assumptions": [
            "Category assignment from POS data is accurate",
            "Revenue comparison uses line_total_sar net of discounts"
        ],
        "confidence": 0.90
    })

# Finding 3: Top Product Performance
top_products = product_comparison.nlargest(3, 'revenue_change')
if len(top_products) > 0 and top_products.iloc[0]['revenue_change'] > 100:
    top_product = top_products.iloc[0]
    findings.append({
        "title": "Top Product Revenue Growth",
        "claim": f"Product '{top_product['item_en']}' (SKU: {top_product['sku']}) generated {top_product['revenue_analysis']:.2f} SAR in analysis week vs {top_product['revenue_previous']:.2f} SAR in previous week, representing {top_product['revenue_pct_change']:+.1f}% change.",
        "finding_type": "product_performance_change",
        "metrics": {
            "product_sku": {
                "value": str(top_product['sku']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "product_name": {
                "value": str(top_product['item_en']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "analysis_product_revenue": {
                "value": round(top_product['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "previous_product_revenue": {
                "value": round(top_product['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+03:00",
                "period_end": "2026-06-15T00:00:00+03:00"
            },
            "product_revenue_change_pct": {
                "value": round(top_product['revenue_pct_change'], 2),
                "unit": "%",
                "numerator": round(top_product['revenue_change'], 2),
                "denominator": round(top_product['revenue_previous'], 2),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_product['transactions_analysis']) if pd.notna(top_product['transactions_analysis']) else None,
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Previous period: 2026-06-08 to 2026-06-15",
            f"Product transactions in analysis period: {int(top_product['transactions_analysis']) if pd.notna(top_product['transactions_analysis']) else 'N/A'}",
            f"Launch date: {top_product['launch_date'] if pd.notna(top_product['launch_date']) else 'Not specified'}"
        ],
        "assumptions": [
            "SKU to product name mapping from menu is accurate",
            "Revenue includes all line items for the product",
            "Product is eligible for comparison (no launch/retire date conflicts)"
        ],
        "confidence": 0.88
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
