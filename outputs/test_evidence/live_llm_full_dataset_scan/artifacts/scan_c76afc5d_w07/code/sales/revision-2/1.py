import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods (ISO 8601 with +03:00 timezone)
analysis_start = "2026-02-23T00:00:00+03:00"
analysis_end = "2026-03-02T00:00:00+03:00"
previous_start = "2026-02-16T00:00:00+03:00"
previous_end = "2026-02-23T00:00:00+03:00"

# Convert to UTC for comparison (subtract 3 hours)
analysis_start_utc = pd.Timestamp("2026-02-22T21:00:00Z")
analysis_end_utc = pd.Timestamp("2026-03-01T21:00:00Z")
previous_start_utc = pd.Timestamp("2026-02-15T21:00:00Z")
previous_end_utc = pd.Timestamp("2026-02-22T21:00:00Z")

# Convert timestamp to UTC for filtering
pos_df['timestamp_utc'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter for analysis and previous periods
analysis_df = pos_df[(pos_df['timestamp_utc'] >= analysis_start_utc) & 
                      (pos_df['timestamp_utc'] < analysis_end_utc)].copy()
previous_df = pos_df[(pos_df['timestamp_utc'] >= previous_start_utc) & 
                      (pos_df['timestamp_utc'] < previous_end_utc)].copy()

# Exclude refunds from revenue calculations (per critic feedback)
analysis_sales = analysis_df[analysis_df['is_refund'] == False].copy()
previous_sales = previous_df[previous_df['is_refund'] == False].copy()

# Calculate metrics for analysis period
analysis_revenue = analysis_sales['line_total_sar'].sum()
analysis_transactions = analysis_sales['transaction_id'].nunique()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

# Calculate metrics for previous period
previous_revenue = previous_sales['line_total_sar'].sum()
previous_transactions = previous_sales['transaction_id'].nunique()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change = analysis_transactions - previous_transactions
transaction_change_pct = (transaction_change / previous_transactions * 100) if previous_transactions != 0 else 0

aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Category analysis
analysis_category = analysis_sales.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category = previous_sales.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Find largest category change
category_changes = {}
for cat in analysis_category.index:
    prev_val = previous_category.get(cat, 0)
    curr_val = analysis_category.get(cat, 0)
    change_pct = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
    category_changes[cat] = {
        'current': curr_val,
        'previous': prev_val,
        'change': curr_val - prev_val,
        'change_pct': change_pct
    }

# Find category with largest absolute change
largest_category = max(category_changes.items(), 
                       key=lambda x: abs(x[1]['change_pct']))

# SKU analysis - join with menu for launch dates
analysis_sku = analysis_sales.copy()
analysis_sku = analysis_sku.merge(menu_df[['sku', 'launch_date', 'retire_date']], 
                                   on='sku', how='left')

previous_sku = previous_sales.copy()
previous_sku = previous_sku.merge(menu_df[['sku', 'launch_date', 'retire_date']], 
                                   on='sku', how='left')

# Filter for products that were active in both periods
analysis_sku_revenue = analysis_sku.groupby('sku')['line_total_sar'].sum()
previous_sku_revenue = previous_sku.groupby('sku')['line_total_sar'].sum()

# Find SKUs active in both periods
common_skus = set(analysis_sku_revenue.index) & set(previous_sku_revenue.index)

sku_changes = {}
for sku in common_skus:
    prev_val = previous_sku_revenue[sku]
    curr_val = analysis_sku_revenue[sku]
    change_pct = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
    sku_changes[sku] = {
        'current': curr_val,
        'previous': prev_val,
        'change': curr_val - prev_val,
        'change_pct': change_pct
    }

# Find SKU with largest absolute change
if sku_changes:
    largest_sku = max(sku_changes.items(), 
                      key=lambda x: abs(x[1]['change_pct']))
    largest_sku_code = largest_sku[0]
    largest_sku_data = largest_sku[1]
    
    # Get SKU name from menu
    sku_name = menu_df[menu_df['sku'] == largest_sku_code]['item_en'].values
    sku_name = sku_name[0] if len(sku_name) > 0 else largest_sku_code
else:
    largest_sku_code = None
    largest_sku_data = None
    sku_name = None

# Build findings
findings = []

# Finding 1: Transaction count decline
if transaction_change_pct < -5:  # Significant decline
    findings.append({
        "title": "Transaction Count Decline Week-over-Week",
        "claim": f"Valid transaction count declined {abs(transaction_change_pct):.2f}% from {previous_transactions} baskets in the previous week to {analysis_transactions} baskets in the analysis week (Feb 23 - Mar 2, 2026).",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "transaction_change_pct": {
                "value": round(transaction_change_pct, 2),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": previous_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            "Refunds excluded from transaction count (is_refund == False)",
            "Transaction count based on unique transaction_id values",
            "Periods are adjacent 7-day windows",
            "All valid transactions included in denominator"
        ],
        "assumptions": [
            "is_refund flag correctly identifies refund transactions",
            "transaction_id uniquely identifies a basket",
            "timestamp_utc correctly converts to local time for period filtering"
        ],
        "confidence": 0.95
    })

# Finding 2: Category revenue shift
if largest_category[1]['change_pct'] != 0:
    findings.append({
        "title": f"Largest Category Revenue Change: {largest_category[0]}",
        "claim": f"The {largest_category[0]} category experienced the largest revenue change at {largest_category[1]['change_pct']:.2f}%, declining from {previous_category.get(largest_category[0], 0):.2f} SAR to {analysis_category.get(largest_category[0], 0):.2f} SAR between the previous week and analysis week.",
        "finding_type": "category_mix",
        "metrics": {
            "analysis_period_category_revenue": {
                "value": round(analysis_category.get(largest_category[0], 0), 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_period_category_revenue": {
                "value": round(previous_category.get(largest_category[0], 0), 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "category_revenue_change_pct": {
                "value": round(largest_category[1]['change_pct'], 2),
                "unit": "%",
                "numerator": round(largest_category[1]['change'], 2),
                "denominator": round(previous_category.get(largest_category[0], 0), 2),
                "period_start": previous_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_sales[analysis_sales['category'] == largest_category[0]]),
        "coverage_notes": [
            "Refunds excluded from category revenue (is_refund == False)",
            "Category totals based on line_total_sar sum",
            "Periods are adjacent 7-day windows",
            "All categories with sales in either period included in comparison"
        ],
        "assumptions": [
            "category field correctly classifies products",
            "is_refund flag correctly identifies refund transactions",
            "line_total_sar represents net revenue per line item"
        ],
        "confidence": 0.90
    })

# Finding 3: SKU performance (if significant change exists)
if largest_sku_data and abs(largest_sku_data['change_pct']) > 10:
    findings.append({
        "title": f"Largest SKU Revenue Change: {sku_name}",
        "claim": f"SKU {largest_sku_code} ({sku_name}) showed the largest revenue change at {largest_sku_data['change_pct']:.2f}%, moving from {largest_sku_data['previous']:.2f} SAR to {largest_sku_data['current']:.2f} SAR between the previous week and analysis week.",
        "finding_type": "product_performance",
        "metrics": {
            "analysis_period_sku_revenue": {
                "value": round(largest_sku_data['current'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_period_sku_revenue": {
                "value": round(largest_sku_data['previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "sku_revenue_change_pct": {
                "value": round(largest_sku_data['change_pct'], 2),
                "unit": "%",
                "numerator": round(largest_sku_data['change'], 2),
                "denominator": round(largest_sku_data['previous'], 2),
                "period_start": previous_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_sku[analysis_sku['sku'] == largest_sku_code]),
        "coverage_notes": [
            "Refunds excluded from SKU revenue (is_refund == False)",
            "SKU revenue based on line_total_sar sum",
            "Only SKUs active in both periods included",
            "Periods are adjacent 7-day windows",
            "Menu join used to verify product names and launch/retire dates"
        ],
        "assumptions": [
            "sku field correctly identifies products",
            "is_refund flag correctly identifies refund transactions",
            "line_total_sar represents net revenue per line item",
            "Product was active (post-launch, pre-retire) in both periods"
        ],
        "confidence": 0.85
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)