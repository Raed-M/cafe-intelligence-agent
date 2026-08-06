import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = pd.Timestamp("2026-03-23T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-03-30T00:00:00+03:00")
previous_start = pd.Timestamp("2026-03-16T00:00:00+03:00")
previous_end = pd.Timestamp("2026-03-23T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Revenue and Transaction Performance Week-over-Week
# ============================================================================

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
analysis_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0

# Calculate metrics for previous period
previous_valid_txns = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
previous_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()
previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
txn_change = analysis_valid_txns - previous_valid_txns
txn_change_pct = (txn_change / previous_valid_txns * 100) if previous_valid_txns != 0 else 0
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_change_pct) > 2 or abs(txn_change_pct) > 2:
    findings.append({
        "title": "Weekly Revenue and Transaction Performance",
        "claim": f"Net revenue for week of 2026-03-23 to 2026-03-30 was SAR {analysis_revenue:.2f} across {analysis_valid_txns} valid transactions (AOV: SAR {analysis_aov:.2f}), compared to SAR {previous_revenue:.2f} across {previous_valid_txns} transactions (AOV: SAR {previous_aov:.2f}) in the prior week. Revenue changed by SAR {revenue_change:.2f} ({revenue_change_pct:.2f}%), transactions by {txn_change} ({txn_change_pct:.2f}%), and AOV by SAR {aov_change:.2f} ({aov_change_pct:.2f}%).",
        "finding_type": "revenue_and_transaction_performance",
        "metrics": {
            "analysis_revenue_sar": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_valid_transactions": {
                "value": analysis_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_aov_sar": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "previous_revenue_sar": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "previous_valid_transactions": {
                "value": previous_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "previous_aov_sar": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "transaction_change_count": {
                "value": txn_change,
                "unit": "count",
                "numerator": analysis_valid_txns,
                "denominator": previous_valid_txns,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(txn_change_pct, 2),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_valid_txns,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(analysis_aov, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items, {analysis_valid_txns} valid transactions (refunds excluded)",
            f"Previous period: {len(previous_data)} line items, {previous_valid_txns} valid transactions (refunds excluded)",
            "All transactions with is_refund=False included in calculations",
            "line_total_sar used for revenue calculations"
        ],
        "assumptions": [
            "Valid transactions are those with is_refund=False",
            "Revenue includes all discounts applied at line level",
            "AOV calculated as total revenue divided by unique transaction_id count",
            "Periods are mutually exclusive and cover full calendar weeks"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Category Mix Shift Analysis
# ============================================================================

# Analysis period category breakdown
analysis_by_category = analysis_data[analysis_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'baskets']
analysis_total_revenue = analysis_by_category['revenue'].sum()
analysis_by_category['pct_of_total'] = (analysis_by_category['revenue'] / analysis_total_revenue * 100)

# Previous period category breakdown
previous_by_category = previous_data[previous_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'baskets']
previous_total_revenue = previous_by_category['revenue'].sum()
previous_by_category['pct_of_total'] = (previous_by_category['revenue'] / previous_total_revenue * 100)

# Find categories with significant mix changes
category_changes = []
for cat in analysis_by_category['category'].unique():
    analysis_cat = analysis_by_category[analysis_by_category['category'] == cat]
    previous_cat = previous_by_category[previous_by_category['category'] == cat]
    
    if len(analysis_cat) > 0 and len(previous_cat) > 0:
        analysis_pct = analysis_cat['pct_of_total'].values[0]
        previous_pct = previous_cat['pct_of_total'].values[0]
        pct_point_change = analysis_pct - previous_pct
        
        analysis_rev = analysis_cat['revenue'].values[0]
        previous_rev = previous_cat['revenue'].values[0]
        rev_change = analysis_rev - previous_rev
        rev_change_pct = (rev_change / previous_rev * 100) if previous_rev != 0 else 0
        
        if abs(pct_point_change) > 0.5:
            category_changes.append({
                'category': cat,
                'analysis_pct': analysis_pct,
                'previous_pct': previous_pct,
                'pct_point_change': pct_point_change,
                'analysis_rev': analysis_rev,
                'previous_rev': previous_rev,
                'rev_change': rev_change,
                'rev_change_pct': rev_change_pct,
                'analysis_baskets': analysis_cat['baskets'].values[0],
                'previous_baskets': previous_cat['baskets'].values[0]
            })

# Report largest category mix shift
if category_changes:
    largest_shift = max(category_changes, key=lambda x: abs(x['pct_point_change']))
    
    findings.append({
        "title": f"Category Mix Shift: {largest_shift['category']}",
        "claim": f"The {largest_shift['category']} category shifted from {largest_shift['previous_pct']:.2f}% of total revenue in the prior week (SAR {largest_shift['previous_rev']:.2f}) to {largest_shift['analysis_pct']:.2f}% in the analysis week (SAR {largest_shift['analysis_rev']:.2f}), a change of {largest_shift['pct_point_change']:.2f} percentage points and SAR {largest_shift['rev_change']:.2f} ({largest_shift['rev_change_pct']:.2f}%). Basket penetration changed from {largest_shift['previous_baskets']} to {largest_shift['analysis_baskets']} baskets.",
        "finding_type": "category_mix_shift",
        "metrics": {
            "analysis_category_revenue_sar": {
                "value": round(largest_shift['analysis_rev'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_category_pct_of_total": {
                "value": round(largest_shift['analysis_pct'], 2),
                "unit": "%",
                "numerator": round(largest_shift['analysis_rev'], 2),
                "denominator": round(analysis_total_revenue, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_category_baskets": {
                "value": largest_shift['analysis_baskets'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "previous_category_revenue_sar": {
                "value": round(largest_shift['previous_rev'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "previous_category_pct_of_total": {
                "value": round(largest_shift['previous_pct'], 2),
                "unit": "%",
                "numerator": round(largest_shift['previous_rev'], 2),
                "denominator": round(previous_total_revenue, 2),
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "previous_category_baskets": {
                "value": largest_shift['previous_baskets'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "category_revenue_change_sar": {
                "value": round(largest_shift['rev_change'], 2),
                "unit": "SAR",
                "numerator": round(largest_shift['analysis_rev'], 2),
                "denominator": round(largest_shift['previous_rev'], 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "category_revenue_change_pct": {
                "value": round(largest_shift['rev_change_pct'], 2),
                "unit": "%",
                "numerator": round(largest_shift['rev_change'], 2),
                "denominator": round(largest_shift['previous_rev'], 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "category_mix_change_pct_points": {
                "value": round(largest_shift['pct_point_change'], 2),
                "unit": "percentage points",
                "numerator": round(largest_shift['analysis_pct'], 2),
                "denominator": round(largest_shift['previous_pct'], 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items across {analysis_valid_txns} transactions",
            f"Previous period: {len(previous_data)} line items across {previous_valid_txns} transactions",
            "Refunds excluded from all calculations",
            f"Category {largest_shift['category']} tracked across both periods"
        ],
        "assumptions": [
            "Category field populated for all non-refund line items",
            "Revenue percentages calculated from total net revenue (line_total_sar)",
            "Basket penetration = unique transaction_id per category",
            "Mix shift threshold: >0.5 percentage points"
        ],
        "confidence": 0.92
    })

# ============================================================================
# FINDING 3: Top SKU Performance Week-over-Week
# ============================================================================

# Analysis period SKU performance
analysis_by_sku = analysis_data[analysis_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique',
    'item_name_en': 'first'
}).reset_index()
analysis_by_sku.columns = ['sku', 'revenue', 'quantity', 'baskets', 'item_name']
analysis_by_sku = analysis_by_sku.sort_values('revenue', ascending=False)

# Previous period SKU performance
previous_by_sku = previous_data[previous_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique',
    'item_name_en': 'first'
}).reset_index()
previous_by_sku.columns = ['sku', 'revenue', 'quantity', 'baskets', 'item_name']
previous_by_sku = previous_by_sku.sort_values('revenue', ascending=False)

# Get top SKU from analysis period
if len(analysis_by_sku) > 0:
    top_sku = analysis_by_sku.iloc[0]
    top_sku_code = top_sku['sku']
    
    # Find same SKU in previous period
    prev_top = previous_by_sku[previous_by_sku['sku'] == top_sku_code]
    
    if len(prev_top) > 0:
        prev_top = prev_top.iloc[0]
        
        rev_change = top_sku['revenue'] - prev_top['revenue']
        rev_change_pct = (rev_change / prev_top['revenue'] * 100) if prev_top['revenue'] != 0 else 0
        qty_change = top_sku['quantity'] - prev_top['quantity']
        qty_change_pct = (qty_change / prev_top['quantity'] * 100) if prev_top['quantity'] != 0 else 0
        basket_change = top_sku['baskets'] - prev_top['baskets']
        
        findings.append({
            "title": f"Top SKU Performance: {top_sku['item_name']}",
            "claim": f"The top-revenue SKU {top_sku['item_name']} (SKU: {top_sku_code}) generated SAR {top_sku['revenue']:.2f} across {int(top_sku['quantity'])} units and {int(top_sku['baskets'])} baskets in the analysis week (2026-03-23 to 2026-03-30), compared to SAR {prev_top['revenue']:.2f} ({int(prev_top['quantity'])} units, {int(prev_top['baskets'])} baskets) in the prior week. Revenue changed by SAR {rev_change:.2f} ({rev_change_pct:.2f}%), quantity by {int(qty_change)} units ({qty_change_pct:.2f}%), and basket penetration by {int(basket_change)} baskets.",
            "finding_type": "top_sku_performance",
            "metrics": {
                "analysis_sku_revenue_sar": {
                    "value": round(top_sku['revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "analysis_sku_quantity": {
                    "value": int(top_sku['quantity']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "analysis_sku_baskets": {
                    "value": int(top_sku['baskets']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "previous_sku_revenue_sar": {
                    "value": round(prev_top['revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "previous_sku_quantity": {
                    "value": int(prev_top['quantity']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "previous_sku_baskets": {
                    "value": int(prev_top['baskets']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+03:00",
                    "period_end": "2026-03-23T00:00:00+03:00"
                },
                "sku_revenue_change_sar": {
                    "value": round(rev_change, 2),
                    "unit": "SAR",
                    "numerator": round(top_sku['revenue'], 2),
                    "denominator": round(prev_top['revenue'], 2),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "sku_revenue_change_pct": {
                    "value": round(rev_change_pct, 2),
                    "unit": "%",
                    "numerator": round(rev_change, 2),
                    "denominator": round(prev_top['revenue'], 2),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "sku_quantity_change": {
                    "value": int(qty_change),
                    "unit": "units",
                    "numerator": int(top_sku['quantity']),
                    "denominator": int(prev_top['quantity']),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "sku_quantity_change_pct": {
                    "value": round(qty_change_pct, 2),
                    "unit": "%",
                    "numerator": int(qty_change),
                    "denominator": int(prev_top['quantity']),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "sku_basket_change": {
                    "value": int(basket_change),
                    "unit": "count",
                    "numerator": int(top_sku['baskets']),
                    "denominator": int(prev_top['baskets']),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period: {len(analysis_data)} line items, {analysis_valid_txns} transactions",
                f"Previous period: {len(previous_data)} line items, {previous_valid_txns} transactions",
                f"Top SKU identified by highest revenue in analysis period",
                "Refunds excluded from all calculations",
                f"SKU {top_sku_code} present in both periods"
            ],
            "assumptions": [
                "Top SKU = highest revenue-generating SKU in analysis period",
                "Quantity = sum of quantity field across all line items for SKU",
                "Basket penetration = unique transaction_id count for SKU",
                "Revenue includes all discounts applied at line level",
                "Comparison is week-over-week (7-day periods)"
            ],
            "confidence": 0.93
        })

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)