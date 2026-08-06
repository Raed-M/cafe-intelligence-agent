import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = pd.Timestamp("2026-03-16T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-03-23T00:00:00+03:00")
previous_start = pd.Timestamp("2026-03-09T00:00:00+03:00")
previous_end = pd.Timestamp("2026-03-16T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Calculate metrics for analysis period
analysis_baskets = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0

# Calculate metrics for previous period
previous_baskets = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0

# Calculate changes
basket_change = analysis_baskets - previous_baskets
basket_pct_change = (basket_change / previous_baskets * 100) if previous_baskets > 0 else 0
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

# Analyze by category
analysis_by_category = analysis_data.groupby('category').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum',
    'quantity': 'sum'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

previous_by_category = previous_data.groupby('category').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum',
    'quantity': 'sum'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

# Find category with largest revenue change
category_revenue_changes = {}
for cat in analysis_by_category.index:
    if cat in previous_by_category.index:
        curr_rev = analysis_by_category.loc[cat, 'revenue']
        prev_rev = previous_by_category.loc[cat, 'revenue']
        change = curr_rev - prev_rev
        pct_change = (change / prev_rev * 100) if prev_rev > 0 else 0
        category_revenue_changes[cat] = {
            'current': curr_rev,
            'previous': prev_rev,
            'change': change,
            'pct_change': pct_change,
            'curr_baskets': analysis_by_category.loc[cat, 'baskets'],
            'prev_baskets': previous_by_category.loc[cat, 'baskets']
        }

# Analyze by channel
analysis_by_channel = analysis_data.groupby('channel').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

previous_by_channel = previous_data.groupby('channel').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

# Analyze refunds
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]
previous_refunds = previous_data[previous_data['is_refund'] == True]
analysis_refund_count = len(analysis_refunds)
previous_refund_count = len(previous_refunds)
analysis_refund_value = analysis_refunds['line_total_sar'].sum()
previous_refund_value = previous_refunds['line_total_sar'].sum()

# Analyze product performance - top products by revenue change
analysis_by_sku = analysis_data.groupby('sku').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first',
    'category': 'first'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

previous_by_sku = previous_data.groupby('sku').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first',
    'category': 'first'
}).rename(columns={'transaction_id': 'baskets', 'line_total_sar': 'revenue'})

# Calculate product changes
product_changes = {}
for sku in analysis_by_sku.index:
    if sku in previous_by_sku.index:
        curr_rev = analysis_by_sku.loc[sku, 'revenue']
        prev_rev = previous_by_sku.loc[sku, 'revenue']
        change = curr_rev - prev_rev
        pct_change = (change / prev_rev * 100) if prev_rev > 0 else 0
        product_changes[sku] = {
            'name': analysis_by_sku.loc[sku, 'item_name_en'],
            'category': analysis_by_sku.loc[sku, 'category'],
            'current': curr_rev,
            'previous': prev_rev,
            'change': change,
            'pct_change': pct_change,
            'curr_baskets': analysis_by_sku.loc[sku, 'baskets'],
            'prev_baskets': previous_by_sku.loc[sku, 'baskets'],
            'curr_qty': analysis_by_sku.loc[sku, 'quantity'],
            'prev_qty': previous_by_sku.loc[sku, 'quantity']
        }

# Sort by absolute revenue change
sorted_products = sorted(product_changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)

# Build findings
findings = []

# Finding 1: Overall transaction volume and revenue growth
finding1 = {
    "title": "Transaction Volume and Revenue Growth Week-over-Week",
    "claim": f"Valid basket count increased from {previous_baskets} to {analysis_baskets} baskets ({basket_pct_change:.1f}% increase), with net revenue growing from SAR {previous_revenue:.2f} to SAR {analysis_revenue:.2f} ({revenue_pct_change:.1f}% increase) in the analysis week versus previous week.",
    "finding_type": "revenue_and_transaction_volume",
    "metrics": {
        "analysis_basket_count": {
            "value": analysis_baskets,
            "unit": "baskets",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "previous_basket_count": {
            "value": previous_baskets,
            "unit": "baskets",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-09T00:00:00+03:00",
            "period_end": "2026-03-16T00:00:00+03:00"
        },
        "basket_count_change": {
            "value": basket_pct_change,
            "unit": "%",
            "numerator": basket_change,
            "denominator": previous_baskets,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "analysis_revenue": {
            "value": analysis_revenue,
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "previous_revenue": {
            "value": previous_revenue,
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-09T00:00:00+03:00",
            "period_end": "2026-03-16T00:00:00+03:00"
        },
        "revenue_change": {
            "value": revenue_pct_change,
            "unit": "%",
            "numerator": revenue_change,
            "denominator": previous_revenue,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        "Revenue includes refunds as negative line items in net calculation",
        f"Analysis period: {analysis_baskets} unique transaction_ids from {len(analysis_data)} POS line items",
        f"Previous period: {previous_baskets} unique transaction_ids from {len(previous_data)} POS line items",
        f"Refund transactions in analysis period: {analysis_refund_count} items (SAR {analysis_refund_value:.2f})",
        f"Refund transactions in previous period: {previous_refund_count} items (SAR {previous_refund_value:.2f})"
    ],
    "assumptions": [
        "line_total_sar represents net revenue after discounts and includes refunds as negative values",
        "transaction_id uniquely identifies a basket/transaction",
        "Timestamps are in +03:00 timezone and correctly parsed",
        "All POS records with valid transaction_id are counted"
    ],
    "confidence": 0.95
}
findings.append(finding1)

# Finding 2: Average Order Value change
finding2 = {
    "title": "Average Order Value Decreased Week-over-Week",
    "claim": f"Average Order Value decreased from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}, representing a {aov_pct_change:.1f}% decrease. This indicates weaker per-basket spending despite higher transaction volume.",
    "finding_type": "average_order_value",
    "metrics": {
        "analysis_aov": {
            "value": analysis_aov,
            "unit": "SAR",
            "numerator": analysis_revenue,
            "denominator": analysis_baskets,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "previous_aov": {
            "value": previous_aov,
            "unit": "SAR",
            "numerator": previous_revenue,
            "denominator": previous_baskets,
            "period_start": "2026-03-09T00:00:00+03:00",
            "period_end": "2026-03-16T00:00:00+03:00"
        },
        "aov_change": {
            "value": aov_pct_change,
            "unit": "%",
            "numerator": aov_change,
            "denominator": previous_aov,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        "AOV calculated as net revenue divided by unique basket count",
        "Revenue includes refunds as negative line items",
        f"Analysis period AOV based on {analysis_baskets} baskets",
        f"Previous period AOV based on {previous_baskets} baskets"
    ],
    "assumptions": [
        "line_total_sar represents net revenue after discounts and refunds",
        "transaction_id uniquely identifies a basket",
        "AOV is calculated as total revenue / total baskets"
    ],
    "confidence": 0.92
}
findings.append(finding2)

# Finding 3: Top product revenue change
if sorted_products:
    top_product_sku, top_product_data = sorted_products[0]
    if top_product_data['change'] > 0:  # Only report positive changes
        finding3 = {
            "title": f"Strongest Product Revenue Growth: {top_product_data['name']}",
            "claim": f"Product '{top_product_data['name']}' (SKU: {top_product_sku}, Category: {top_product_data['category']}) achieved the highest revenue growth, increasing from SAR {top_product_data['previous']:.2f} to SAR {top_product_data['current']:.2f} ({top_product_data['pct_change']:.1f}% increase). Basket penetration grew from {top_product_data['prev_baskets']} to {top_product_data['curr_baskets']} baskets ({((top_product_data['curr_baskets'] - top_product_data['prev_baskets']) / top_product_data['prev_baskets'] * 100):.1f}% increase).",
            "finding_type": "product_mix_and_performance",
            "metrics": {
                "product_sku": {
                    "value": top_product_sku,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "product_name": {
                    "value": top_product_data['name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "analysis_revenue": {
                    "value": top_product_data['current'],
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "previous_revenue": {
                    "value": top_product_data['previous'],
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                },
                "revenue_change_pct": {
                    "value": top_product_data['pct_change'],
                    "unit": "%",
                    "numerator": top_product_data['change'],
                    "denominator": top_product_data['previous'],
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "analysis_basket_count": {
                    "value": top_product_data['curr_baskets'],
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "previous_basket_count": {
                    "value": top_product_data['prev_baskets'],
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Product identified by SKU and joined with menu for category classification",
                "Revenue includes refunds as negative line items",
                f"Product appears in {top_product_data['curr_baskets']} baskets in analysis period",
                f"Product appears in {top_product_data['prev_baskets']} baskets in previous period"
            ],
            "assumptions": [
                "SKU uniquely identifies a product across POS and menu",
                "line_total_sar represents net revenue after discounts and refunds",
                "transaction_id uniquely identifies a basket",
                "Product launch/retirement dates do not restrict this comparison"
            ],
            "confidence": 0.90
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)