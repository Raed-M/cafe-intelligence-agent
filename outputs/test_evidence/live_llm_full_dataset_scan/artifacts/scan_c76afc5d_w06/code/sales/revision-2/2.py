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

# Define periods
analysis_start = pd.Timestamp("2026-02-16T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-02-23T00:00:00+03:00")
previous_start = pd.Timestamp("2026-02-09T00:00:00+03:00")
previous_end = pd.Timestamp("2026-02-16T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Baseline periods for trailing average
baseline_periods = [
    ("2026-02-09T00:00:00+03:00", "2026-02-16T00:00:00+03:00"),
    ("2026-02-02T00:00:00+03:00", "2026-02-09T00:00:00+03:00"),
    ("2026-01-26T00:00:00+03:00", "2026-02-02T00:00:00+03:00"),
    ("2026-01-19T00:00:00+03:00", "2026-01-26T00:00:00+03:00"),
]

baseline_data_list = []
for start_str, end_str in baseline_periods:
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    baseline_data_list.append(pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] < end)])

baseline_data = pd.concat(baseline_data_list, ignore_index=True)

findings = []

# FINDING 1: Revenue and AOV comparison (analysis vs previous week)
# Exclude refunds from revenue calculations
analysis_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
previous_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()

analysis_baskets = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
previous_baskets = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()

analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0

# Check if change is material
if abs(aov_pct_change) >= 2.0:
    findings.append({
        "title": "Average Order Value Change Week-over-Week",
        "claim": f"Average order value changed from {previous_aov:.2f} SAR (week of {previous_start.date()}) to {analysis_aov:.2f} SAR (week of {analysis_start.date()}), a change of {aov_change:.2f} SAR ({aov_pct_change:.2f}%).",
        "finding_type": "AOV_CHANGE",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_baskets,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_baskets,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_baskets,
        "coverage_notes": [
            "Refunds excluded from net revenue calculations",
            f"Analysis period: {analysis_baskets} valid transactions",
            f"Previous period: {previous_baskets} valid transactions"
        ],
        "assumptions": [
            "line_total_sar is net of discounts and excludes refunds",
            "transaction_id uniqueness identifies distinct baskets",
            "is_refund flag accurately identifies refund transactions"
        ],
        "confidence": 0.92
    })

# FINDING 2: Category revenue comparison (analysis vs previous)
analysis_category_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('category')['line_total_sar'].sum()
previous_category_revenue = previous_data[previous_data['is_refund'] == False].groupby('category')['line_total_sar'].sum()

# Find category with largest absolute change
category_changes = {}
for cat in analysis_category_revenue.index.union(previous_category_revenue.index):
    curr = analysis_category_revenue.get(cat, 0)
    prev = previous_category_revenue.get(cat, 0)
    change = curr - prev
    pct_change = (change / prev * 100) if prev > 0 else 0
    category_changes[cat] = {
        'current': curr,
        'previous': prev,
        'change': change,
        'pct_change': pct_change
    }

# Find category with largest absolute change
largest_change_cat = max(category_changes.items(), key=lambda x: abs(x[1]['change']))
cat_name = largest_change_cat[0]
cat_metrics = largest_change_cat[1]

if abs(cat_metrics['pct_change']) >= 5.0:
    findings.append({
        "title": f"Category Revenue Change: {cat_name}",
        "claim": f"Revenue from {cat_name} category changed from {cat_metrics['previous']:.2f} SAR (week of {previous_start.date()}) to {cat_metrics['current']:.2f} SAR (week of {analysis_start.date()}), a change of {cat_metrics['change']:.2f} SAR ({cat_metrics['pct_change']:.2f}%).",
        "finding_type": "CATEGORY_REVENUE_CHANGE",
        "metrics": {
            "analysis_category_revenue": {
                "value": round(cat_metrics['current'], 2),
                "unit": "SAR",
                "numerator": round(cat_metrics['current'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_category_revenue": {
                "value": round(cat_metrics['previous'], 2),
                "unit": "SAR",
                "numerator": round(cat_metrics['previous'], 2),
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "category_revenue_change": {
                "value": round(cat_metrics['change'], 2),
                "unit": "SAR",
                "numerator": round(cat_metrics['change'], 2),
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "category_revenue_pct_change": {
                "value": round(cat_metrics['pct_change'], 2),
                "unit": "%",
                "numerator": round(cat_metrics['change'], 2),
                "denominator": round(cat_metrics['previous'], 2),
                "period_start": previous_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data[analysis_data['is_refund'] == False]),
        "coverage_notes": [
            "Refunds excluded from net revenue calculations",
            f"Category: {cat_name}",
            f"Analysis period line items: {len(analysis_data[analysis_data['is_refund'] == False])}"
        ],
        "assumptions": [
            "line_total_sar is net of discounts and excludes refunds",
            "category field accurately classifies products",
            "is_refund flag accurately identifies refund transactions"
        ],
        "confidence": 0.88
    })

# FINDING 3: Product-level revenue change (top SKU by absolute change)
# Merge POS with menu to get product names and launch dates
analysis_with_menu = analysis_data.merge(menu_df, on='sku', how='left')
previous_with_menu = previous_data.merge(menu_df, on='sku', how='left')

# Convert launch_date and retire_date to timezone-naive datetime for comparison
# Strip timezone info from analysis_start and analysis_end for comparison
analysis_start_naive = analysis_start.tz_localize(None)
analysis_end_naive = analysis_end.tz_localize(None)
previous_start_naive = previous_start.tz_localize(None)
previous_end_naive = previous_end.tz_localize(None)

# Convert launch_date and retire_date columns to datetime without timezone
analysis_with_menu['launch_date'] = pd.to_datetime(analysis_with_menu['launch_date'], errors='coerce')
analysis_with_menu['retire_date'] = pd.to_datetime(analysis_with_menu['retire_date'], errors='coerce')
previous_with_menu['launch_date'] = pd.to_datetime(previous_with_menu['launch_date'], errors='coerce')
previous_with_menu['retire_date'] = pd.to_datetime(previous_with_menu['retire_date'], errors='coerce')

# Filter for products within launch/retire window
analysis_with_menu = analysis_with_menu[
    ((analysis_with_menu['launch_date'].isna()) | (analysis_with_menu['launch_date'] <= analysis_end_naive)) &
    ((analysis_with_menu['retire_date'].isna()) | (analysis_with_menu['retire_date'] > analysis_start_naive))
]
previous_with_menu = previous_with_menu[
    ((previous_with_menu['launch_date'].isna()) | (previous_with_menu['launch_date'] <= previous_end_naive)) &
    ((previous_with_menu['retire_date'].isna()) | (previous_with_menu['retire_date'] > previous_start_naive))
]

# Calculate product revenue (excluding refunds)
analysis_product_revenue = analysis_with_menu[analysis_with_menu['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'item_name_en': 'first'
}).reset_index()
analysis_product_revenue.columns = ['sku', 'revenue', 'product_name']

previous_product_revenue = previous_with_menu[previous_with_menu['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'item_name_en': 'first'
}).reset_index()
previous_product_revenue.columns = ['sku', 'revenue', 'product_name']

# Merge to find changes
product_comparison = analysis_product_revenue.merge(
    previous_product_revenue,
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
)
product_comparison['revenue_analysis'] = product_comparison['revenue_analysis'].fillna(0)
product_comparison['revenue_previous'] = product_comparison['revenue_previous'].fillna(0)
product_comparison['change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['pct_change'] = (product_comparison['change'] / product_comparison['revenue_previous'] * 100).replace([np.inf, -np.inf], 0)
product_comparison['pct_change'] = product_comparison['pct_change'].fillna(0)

# Find product with largest absolute change
if len(product_comparison) > 0:
    largest_product_change = product_comparison.loc[product_comparison['change'].abs().idxmax()]

    if abs(largest_product_change['pct_change']) >= 10.0 or abs(largest_product_change['change']) >= 500:
        product_name = largest_product_change['product_name_analysis'] if pd.notna(largest_product_change['product_name_analysis']) else largest_product_change['product_name_previous']
        findings.append({
            "title": f"Product Revenue Change: {product_name}",
            "claim": f"Revenue from {product_name} (SKU: {largest_product_change['sku']}) changed from {largest_product_change['revenue_previous']:.2f} SAR (week of {previous_start.date()}) to {largest_product_change['revenue_analysis']:.2f} SAR (week of {analysis_start.date()}), a change of {largest_product_change['change']:.2f} SAR ({largest_product_change['pct_change']:.2f}%).",
            "finding_type": "PRODUCT_REVENUE_CHANGE",
            "metrics": {
                "product_current_revenue": {
                    "value": round(largest_product_change['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": round(largest_product_change['revenue_analysis'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "product_previous_revenue": {
                    "value": round(largest_product_change['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": round(largest_product_change['revenue_previous'], 2),
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "product_revenue_change": {
                    "value": round(largest_product_change['change'], 2),
                    "unit": "SAR",
                    "numerator": round(largest_product_change['change'], 2),
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "product_revenue_pct_change": {
                    "value": round(largest_product_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(largest_product_change['change'], 2),
                    "denominator": round(largest_product_change['revenue_previous'], 2) if largest_product_change['revenue_previous'] > 0 else None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_with_menu[analysis_with_menu['is_refund'] == False]),
            "coverage_notes": [
                "Refunds excluded from net revenue calculations",
                f"Product: {product_name} (SKU: {largest_product_change['sku']})",
                "Product launch/retire dates respected in eligibility"
            ],
            "assumptions": [
                "line_total_sar is net of discounts and excludes refunds",
                "SKU-to-product-name mapping from menu is authoritative",
                "is_refund flag accurately identifies refund transactions",
                "launch_date and retire_date fields define product eligibility windows"
            ],
            "confidence": 0.85
        })

# Build output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
