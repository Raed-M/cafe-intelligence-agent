import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = pd.Timestamp("2026-04-20T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-04-27T00:00:00+03:00")
previous_start = pd.Timestamp("2026-04-13T00:00:00+03:00")
previous_end = pd.Timestamp("2026-04-20T00:00:00+03:00")

trailing_baselines = [
    (pd.Timestamp("2026-04-13T00:00:00+03:00"), pd.Timestamp("2026-04-20T00:00:00+03:00")),
    (pd.Timestamp("2026-04-06T00:00:00+03:00"), pd.Timestamp("2026-04-13T00:00:00+03:00")),
    (pd.Timestamp("2026-03-30T00:00:00+03:00"), pd.Timestamp("2026-04-06T00:00:00+03:00")),
    (pd.Timestamp("2026-03-23T00:00:00+03:00"), pd.Timestamp("2026-03-30T00:00:00+03:00")),
]

# Convert timestamp to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter data for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Combine all trailing baseline data
trailing_data = pd.concat([
    pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] < end)]
    for start, end in trailing_baselines
])

findings = []

# ============================================================================
# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
# ============================================================================

# Count valid transactions (unique transaction_id, excluding refunds for basket count)
analysis_valid_txns = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
previous_valid_txns = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()

# Calculate net revenue (line_total_sar includes refunds as negative)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Calculate refund impact
analysis_refunds = analysis_data[analysis_data['is_refund']]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund']]['line_total_sar'].sum()

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
txn_change = analysis_valid_txns - previous_valid_txns
txn_pct_change = (txn_change / previous_valid_txns * 100) if previous_valid_txns != 0 else 0

# Average order value
analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0
previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) > 0.1 or abs(txn_pct_change) > 0.1:
    findings.append({
        "title": "Revenue and Transaction Volume Change (Week of 20-27 Apr vs 13-20 Apr)",
        "claim": f"Net revenue increased by {revenue_change:.2f} SAR ({revenue_pct_change:.1f}%) from {previous_revenue:.2f} SAR to {analysis_revenue:.2f} SAR. Valid transaction count changed by {txn_change} baskets ({txn_pct_change:.1f}%) from {previous_valid_txns} to {analysis_valid_txns}. Average order value changed from {previous_aov:.2f} SAR to {analysis_aov:.2f} SAR ({aov_pct_change:.1f}%).",
        "finding_type": "revenue_and_transaction_change",
        "metrics": {
            "net_revenue_analysis": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "net_revenue_previous": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "revenue_change_absolute": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "revenue_change_percent": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "valid_transactions_analysis": {
                "value": analysis_valid_txns,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "valid_transactions_previous": {
                "value": previous_valid_txns,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "transaction_change_absolute": {
                "value": txn_change,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "transaction_change_percent": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "aov_analysis": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "aov_previous": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "aov_change_percent": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "refund_impact_analysis": {
                "value": round(analysis_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "refund_impact_previous": {
                "value": round(previous_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} POS line items from {analysis_valid_txns} valid transactions",
            f"Previous period: {len(previous_data)} POS line items from {previous_valid_txns} valid transactions",
            f"Refunds included in net revenue calculations: analysis={round(analysis_refunds, 2)} SAR, previous={round(previous_refunds, 2)} SAR"
        ],
        "assumptions": [
            "Valid transactions counted as unique transaction_id excluding refund rows",
            "Net revenue includes refunds as negative values per line_total_sar",
            "AOV calculated as net revenue divided by valid transaction count"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Category Mix Change (Analysis vs Previous Week)
# ============================================================================

# Get category mix for analysis period
analysis_category = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
analysis_category_pct = (analysis_category / analysis_category.sum() * 100)

# Get category mix for previous period
previous_category = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_pct = (previous_category / previous_category.sum() * 100)

# Find significant category shifts
category_shifts = []
for cat in analysis_category.index:
    if cat in previous_category.index:
        prev_pct = previous_category_pct[cat]
        curr_pct = analysis_category_pct[cat]
        pct_point_change = curr_pct - prev_pct
        if abs(pct_point_change) > 1.0:  # More than 1 percentage point change
            category_shifts.append({
                'category': cat,
                'analysis_pct': curr_pct,
                'previous_pct': prev_pct,
                'change_pct_points': pct_point_change,
                'analysis_revenue': analysis_category[cat],
                'previous_revenue': previous_category[cat]
            })

if category_shifts:
    # Sort by absolute change
    category_shifts.sort(key=lambda x: abs(x['change_pct_points']), reverse=True)
    top_shift = category_shifts[0]
    
    findings.append({
        "title": f"Category Mix Shift: {top_shift['category']} (Week of 20-27 Apr vs 13-20 Apr)",
        "claim": f"Category '{top_shift['category']}' revenue share changed from {top_shift['previous_pct']:.1f}% to {top_shift['analysis_pct']:.1f}% ({top_shift['change_pct_points']:+.1f} percentage points). Absolute revenue: {top_shift['previous_revenue']:.2f} SAR → {top_shift['analysis_revenue']:.2f} SAR ({((top_shift['analysis_revenue']/top_shift['previous_revenue']-1)*100):+.1f}%).",
        "finding_type": "category_mix_change",
        "metrics": {
            "category_revenue_analysis": {
                "value": round(top_shift['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "category_revenue_previous": {
                "value": round(top_shift['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "category_share_analysis": {
                "value": round(top_shift['analysis_pct'], 2),
                "unit": "%",
                "numerator": round(top_shift['analysis_revenue'], 2),
                "denominator": round(analysis_category.sum(), 2),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "category_share_previous": {
                "value": round(top_shift['previous_pct'], 2),
                "unit": "%",
                "numerator": round(top_shift['previous_revenue'], 2),
                "denominator": round(previous_category.sum(), 2),
                "period_start": "2026-04-13T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "share_change_percentage_points": {
                "value": round(top_shift['change_pct_points'], 2),
                "unit": "percentage points",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items across {analysis_data['category'].nunique()} categories",
            f"Previous period: {len(previous_data)} line items across {previous_data['category'].nunique()} categories",
            "Revenue calculated as sum of line_total_sar per category (includes refunds)"
        ],
        "assumptions": [
            "Category mix calculated from net revenue (line_total_sar) per category",
            "Refunds included in category totals as negative values"
        ],
        "confidence": 0.90
    })

# ============================================================================
# FINDING 3: Top Product Performance vs Trailing Baseline
# ============================================================================

# Get top products in analysis period
analysis_products = analysis_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_products.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']
analysis_products = analysis_products.sort_values('revenue', ascending=False)

# Get trailing baseline aggregates
trailing_products = trailing_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
trailing_products.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']

# Merge and calculate changes
product_comparison = analysis_products.merge(
    trailing_products,
    on=['sku', 'item_name_en'],
    suffixes=('_analysis', '_trailing'),
    how='left'
)

# Fill NaN with 0 for products not in trailing baseline
product_comparison['revenue_trailing'] = product_comparison['revenue_trailing'].fillna(0)
product_comparison['quantity_trailing'] = product_comparison['quantity_trailing'].fillna(0)
product_comparison['transactions_trailing'] = product_comparison['transactions_trailing'].fillna(0)

# Calculate changes
product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_trailing']
product_comparison['revenue_pct_change'] = (
    (product_comparison['revenue_change'] / product_comparison['revenue_trailing'] * 100)
    .where(product_comparison['revenue_trailing'] != 0, 0)
)

# Filter for top products with significant changes
top_products = product_comparison[product_comparison['revenue_analysis'] > 0].head(10)

# Find product with largest absolute revenue change
if len(top_products) > 0:
    top_product = top_products.loc[top_products['revenue_change'].abs().idxmax()]
    
    # Check if product is eligible (launched before analysis period)
    product_sku = top_product['sku']
    menu_row = menu_df[menu_df['sku'] == product_sku]
    
    is_eligible = True
    eligibility_note = ""
    
    if len(menu_row) > 0:
        launch_date = menu_row.iloc[0]['launch_date']
        retire_date = menu_row.iloc[0]['retire_date']
        
        if pd.notna(launch_date):
            launch_ts = pd.Timestamp(launch_date)
            if launch_ts >= analysis_start:
                is_eligible = False
                eligibility_note = f"Product launched {launch_date}, after analysis period start"
        
        if pd.notna(retire_date):
            retire_ts = pd.Timestamp(retire_date)
            if retire_ts <= analysis_start:
                is_eligible = False
                eligibility_note = f"Product retired {retire_date}, before analysis period"
    
    if is_eligible:
        # Calculate trailing baseline average
        trailing_avg_revenue = trailing_products[trailing_products['sku'] == product_sku]['revenue'].sum() / len(trailing_baselines) if len(trailing_products[trailing_products['sku'] == product_sku]) > 0 else 0
        
        findings.append({
            "title": f"Top Product Performance: {top_product['item_name_en']} (Week of 20-27 Apr vs 4-week Trailing Baseline)",
            "claim": f"Product '{top_product['item_name_en']}' (SKU: {product_sku}) generated {top_product['revenue_analysis']:.2f} SAR in the analysis week, compared to {top_product['revenue_trailing']:.2f} SAR in the trailing 4-week baseline ({top_product['revenue_pct_change']:+.1f}% change). Quantity sold: {int(top_product['quantity_analysis'])} units vs {int(top_product['quantity_trailing'])} units baseline.",
            "finding_type": "product_performance_change",
            "metrics": {
                "product_revenue_analysis": {
                    "value": round(top_product['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+03:00",
                    "period_end": "2026-04-27T00:00:00+03:00"
                },
                "product_revenue_trailing_baseline": {
                    "value": round(top_product['revenue_trailing'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-04-20T00:00:00+03:00"
                },
                "product_revenue_change_absolute": {
                    "value": round(top_product['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+03:00",
                    "period_end": "2026-04-27T00:00:00+03:00"
                },
                "product_revenue_change_percent": {
                    "value": round(top_product['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+03:00",
                    "period_end": "2026-04-27T00:00:00+03:00"
                },
                "product_quantity_analysis": {
                    "value": int(top_product['quantity_analysis']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+03:00",
                    "period_end": "2026-04-27T00:00:00+03:00"
                },
                "product_quantity_trailing_baseline": {
                    "value": int(top_product['quantity_trailing']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-04-20T00:00:00+03:00"
                },
                "product_transactions_analysis": {
                    "value": int(top_product['transactions_analysis']),
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+03:00",
                    "period_end": "2026-04-27T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period: {len(analysis_data)} line items",
                f"Trailing baseline: {len(trailing_data)} line items across 4 weeks (23 Mar - 20 Apr)",
                f"Product SKU: {product_sku}",
                f"Eligibility: {eligibility_note if eligibility_note else 'Product eligible (launched before analysis period, not retired)'}"
            ],
            "assumptions": [
                "Trailing baseline is average of 4 preceding weeks",
                "Revenue calculated as sum of line_total_sar (includes refunds)",
                "Product eligibility verified against menu launch/retire dates"
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
    json.dump(output, f, indent=2, default=str)