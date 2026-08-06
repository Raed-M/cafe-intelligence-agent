import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-06-01T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-06-08T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-05-25T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-06-01T00:00:00+03:00")

trailing_periods = [
    ("2026-05-25T00:00:00+03:00", "2026-06-01T00:00:00+03:00"),
    ("2026-05-18T00:00:00+03:00", "2026-05-25T00:00:00+03:00"),
    ("2026-05-11T00:00:00+03:00", "2026-05-18T00:00:00+03:00"),
    ("2026-05-04T00:00:00+03:00", "2026-05-11T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter by period
def filter_by_period(df, start_iso, end_iso):
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Filter analysis period
analysis_df = filter_by_period(pos_df, "2026-06-01T00:00:00+03:00", "2026-06-08T00:00:00+03:00")

# Filter previous period
previous_df = filter_by_period(pos_df, "2026-05-25T00:00:00+03:00", "2026-06-01T00:00:00+03:00")

# Filter trailing periods
trailing_dfs = []
for start_iso, end_iso in trailing_periods:
    trailing_dfs.append(filter_by_period(pos_df, start_iso, end_iso))

# Combine all trailing periods for baseline
trailing_baseline_df = pd.concat(trailing_dfs, ignore_index=True)

findings = []

# ============================================================================
# FINDING 1: Revenue Change (Analysis Period vs Previous Period)
# ============================================================================

# Calculate net revenue for analysis period (excluding refunds from totals)
analysis_revenue = analysis_df[analysis_df['is_refund'] == False]['line_total_sar'].sum()
analysis_refunds = analysis_df[analysis_df['is_refund'] == True]['line_total_sar'].sum()
analysis_net_revenue = analysis_revenue + analysis_refunds  # refunds are negative

# Calculate net revenue for previous period
previous_revenue = previous_df[previous_df['is_refund'] == False]['line_total_sar'].sum()
previous_refunds = previous_df[previous_df['is_refund'] == True]['line_total_sar'].sum()
previous_net_revenue = previous_revenue + previous_refunds

# Calculate transaction counts (unique transaction_id)
analysis_transactions = analysis_df['transaction_id'].nunique()
previous_transactions = previous_df['transaction_id'].nunique()

# Calculate AOV
analysis_aov = analysis_net_revenue / analysis_transactions if analysis_transactions > 0 else 0
previous_aov = previous_net_revenue / previous_transactions if previous_transactions > 0 else 0

# Revenue change
revenue_change = analysis_net_revenue - previous_net_revenue
revenue_pct_change = (revenue_change / previous_net_revenue * 100) if previous_net_revenue != 0 else 0

if abs(revenue_pct_change) > 0.1:  # Only report if meaningful
    findings.append({
        "title": "Net Revenue Change: Analysis Week vs Previous Week",
        "claim": f"Net revenue in analysis period (2026-06-01 to 2026-06-08) was SAR {analysis_net_revenue:.2f}, compared to SAR {previous_net_revenue:.2f} in previous period (2026-05-25 to 2026-06-01), representing a {revenue_pct_change:.1f}% change.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_net_revenue": {
                "value": round(analysis_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_net_revenue": {
                "value": round(previous_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_df),
        "coverage_notes": [
            f"Analysis period rows: {len(analysis_df)}",
            f"Previous period rows: {len(previous_df)}",
            f"Analysis transactions: {analysis_transactions}",
            f"Previous transactions: {previous_transactions}",
            f"Refunds included in net calculation: Analysis SAR {analysis_refunds:.2f}, Previous SAR {previous_refunds:.2f}"
        ],
        "assumptions": [
            "Refunds are negative line_total_sar values and included in net revenue",
            "Valid transactions identified by unique transaction_id",
            "All rows in cleaned POS are valid for analysis"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Average Order Value Change
# ============================================================================

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(aov_pct_change) > 0.5:  # Only report if meaningful
    findings.append({
        "title": "Average Order Value Change: Analysis Week vs Previous Week",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f} per transaction, compared to SAR {previous_aov:.2f} in previous period, representing a {aov_pct_change:.1f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": previous_transactions,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            f"Analysis transactions: {analysis_transactions}",
            f"Previous transactions: {previous_transactions}",
            f"AOV calculated as net revenue / unique transaction count"
        ],
        "assumptions": [
            "AOV = net revenue / unique transaction_id count",
            "Refunds included in net revenue calculation"
        ],
        "confidence": 0.92
    })

# ============================================================================
# FINDING 3: Category Mix Change (Top Categories)
# ============================================================================

# Merge POS with menu to get category info
analysis_with_menu = analysis_df.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                       on='sku', how='left', suffixes=('', '_menu'))
previous_with_menu = previous_df.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                       on='sku', how='left', suffixes=('', '_menu'))

# Use category from menu if available, otherwise from POS
analysis_with_menu['category_final'] = analysis_with_menu['category_menu'].fillna(analysis_with_menu['category'])
previous_with_menu['category_final'] = previous_with_menu['category_menu'].fillna(previous_with_menu['category'])

# Filter out refunds for category analysis
analysis_sales = analysis_with_menu[analysis_with_menu['is_refund'] == False]
previous_sales = previous_with_menu[previous_with_menu['is_refund'] == False]

# Calculate category revenue
analysis_category_revenue = analysis_sales.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_sales.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)

# Calculate category mix percentages
analysis_total = analysis_category_revenue.sum()
previous_total = previous_category_revenue.sum()

analysis_category_pct = (analysis_category_revenue / analysis_total * 100) if analysis_total > 0 else 0
previous_category_pct = (previous_category_revenue / previous_total * 100) if previous_total > 0 else 0

# Find top category with significant change
top_categories = set(analysis_category_revenue.index.tolist()[:3] + previous_category_revenue.index.tolist()[:3])

category_changes = {}
for cat in top_categories:
    curr_pct = analysis_category_pct.get(cat, 0)
    prev_pct = previous_category_pct.get(cat, 0)
    pct_point_change = curr_pct - prev_pct
    category_changes[cat] = {
        'current_pct': curr_pct,
        'previous_pct': prev_pct,
        'pct_point_change': pct_point_change,
        'current_revenue': analysis_category_revenue.get(cat, 0),
        'previous_revenue': previous_category_revenue.get(cat, 0)
    }

# Find category with largest absolute percentage point change
max_change_cat = max(category_changes.items(), 
                     key=lambda x: abs(x[1]['pct_point_change']))

if abs(max_change_cat[1]['pct_point_change']) > 0.5:  # Only report if > 0.5 percentage points
    cat_name = max_change_cat[0]
    cat_data = max_change_cat[1]
    
    findings.append({
        "title": f"Category Mix Shift: {cat_name}",
        "claim": f"Category '{cat_name}' represented {cat_data['current_pct']:.1f}% of sales revenue in analysis period (SAR {cat_data['current_revenue']:.2f}), compared to {cat_data['previous_pct']:.1f}% in previous period (SAR {cat_data['previous_revenue']:.2f}), a change of {cat_data['pct_point_change']:.1f} percentage points.",
        "finding_type": "category_mix_change",
        "metrics": {
            "category_name": {
                "value": cat_name,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "analysis_category_pct": {
                "value": round(cat_data['current_pct'], 2),
                "unit": "%",
                "numerator": round(cat_data['current_revenue'], 2),
                "denominator": round(analysis_total, 2),
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_category_pct": {
                "value": round(cat_data['previous_pct'], 2),
                "unit": "%",
                "numerator": round(cat_data['previous_revenue'], 2),
                "denominator": round(previous_total, 2),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "pct_point_change": {
                "value": round(cat_data['pct_point_change'], 2),
                "unit": "percentage points",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_sales),
        "coverage_notes": [
            f"Analysis period sales rows (non-refund): {len(analysis_sales)}",
            f"Previous period sales rows (non-refund): {len(previous_sales)}",
            f"Category data joined from menu SKU reference",
            f"Analysis total revenue: SAR {analysis_total:.2f}",
            f"Previous total revenue: SAR {previous_total:.2f}"
        ],
        "assumptions": [
            "Category sourced from menu SKU reference where available",
            "Refunds excluded from category mix calculation",
            "Revenue-weighted category analysis"
        ],
        "confidence": 0.90
    })

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)