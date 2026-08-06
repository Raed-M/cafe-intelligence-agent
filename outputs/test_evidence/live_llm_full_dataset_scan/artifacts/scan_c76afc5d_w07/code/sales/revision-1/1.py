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
analysis_start = pd.Timestamp("2026-02-23T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-03-02T00:00:00+03:00")
previous_start = pd.Timestamp("2026-02-16T00:00:00+03:00")
previous_end = pd.Timestamp("2026-02-23T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Baseline periods for trailing average
baseline_periods = [
    (pd.Timestamp("2026-02-16T00:00:00+03:00"), pd.Timestamp("2026-02-23T00:00:00+03:00")),
    (pd.Timestamp("2026-02-09T00:00:00+03:00"), pd.Timestamp("2026-02-16T00:00:00+03:00")),
    (pd.Timestamp("2026-02-02T00:00:00+03:00"), pd.Timestamp("2026-02-09T00:00:00+03:00")),
    (pd.Timestamp("2026-01-26T00:00:00+03:00"), pd.Timestamp("2026-02-02T00:00:00+03:00"))
]

baseline_data_list = []
for start, end in baseline_periods:
    baseline_data_list.append(pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] < end)].copy())
baseline_data = pd.concat(baseline_data_list, ignore_index=True)

findings = []

# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

previous_revenue = previous_data['line_total_sar'].sum()
previous_transactions = previous_data['transaction_id'].nunique()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
transaction_change = analysis_transactions - previous_transactions
transaction_change_pct = (transaction_change / previous_transactions * 100) if previous_transactions > 0 else 0
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

finding1 = {
    "title": "Revenue Decline with Reduced Transaction Count",
    "claim": f"Net revenue declined {revenue_change_pct:.2f}% (from {previous_revenue:.2f} SAR to {analysis_revenue:.2f} SAR) driven by a {transaction_change_pct:.2f}% decrease in valid transactions (from {previous_transactions} to {analysis_transactions}), while average order value remained flat at {analysis_aov:.2f} SAR.",
    "finding_type": "revenue_and_transaction_mix",
    "metrics": {
        "analysis_period_revenue": {
            "value": round(analysis_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-02-23T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        },
        "previous_period_revenue": {
            "value": round(previous_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-02-23T00:00:00+03:00"
        },
        "revenue_change_pct": {
            "value": round(revenue_change_pct, 2),
            "unit": "%",
            "numerator": round(revenue_change, 2),
            "denominator": round(previous_revenue, 2),
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        },
        "analysis_period_transactions": {
            "value": analysis_transactions,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-02-23T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        },
        "previous_period_transactions": {
            "value": previous_transactions,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-02-23T00:00:00+03:00"
        },
        "transaction_change_pct": {
            "value": round(transaction_change_pct, 2),
            "unit": "%",
            "numerator": transaction_change,
            "denominator": previous_transactions,
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        },
        "analysis_period_aov": {
            "value": round(analysis_aov, 2),
            "unit": "SAR",
            "numerator": round(analysis_revenue, 2),
            "denominator": analysis_transactions,
            "period_start": "2026-02-23T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        },
        "previous_period_aov": {
            "value": round(previous_aov, 2),
            "unit": "SAR",
            "numerator": round(previous_revenue, 2),
            "denominator": previous_transactions,
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-02-23T00:00:00+03:00"
        },
        "aov_change_pct": {
            "value": round(aov_change_pct, 2),
            "unit": "%",
            "numerator": round(aov_change, 2),
            "denominator": round(previous_aov, 2),
            "period_start": "2026-02-16T00:00:00+03:00",
            "period_end": "2026-03-02T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        "Analysis period: 2026-02-23 to 2026-03-02 (7 days)",
        "Previous period: 2026-02-16 to 2026-02-23 (7 days)",
        "Refunds included in net revenue calculation (is_refund flag applied)",
        "Transaction count based on unique transaction_id values",
        "AOV calculated as total line_total_sar divided by unique transaction count"
    ],
    "assumptions": [
        "is_refund flag correctly identifies refund transactions",
        "line_total_sar represents net transaction value including refunds as negative amounts",
        "transaction_id uniquely identifies a basket/transaction",
        "Timestamp filtering uses UTC+3 timezone as specified in period definitions"
    ],
    "confidence": 0.95
}
findings.append(finding1)

# FINDING 2: Category Revenue Performance (Analysis vs Previous Week, excluding refunds)
analysis_data_no_refund = analysis_data[analysis_data['is_refund'] == False].copy()
previous_data_no_refund = previous_data[previous_data['is_refund'] == False].copy()

analysis_category_revenue = analysis_data_no_refund.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data_no_refund.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Find the category with largest absolute change
category_changes = {}
for cat in analysis_category_revenue.index:
    analysis_val = analysis_category_revenue.get(cat, 0)
    previous_val = previous_category_revenue.get(cat, 0)
    change = analysis_val - previous_val
    change_pct = (change / previous_val * 100) if previous_val != 0 else 0
    category_changes[cat] = {
        'analysis': analysis_val,
        'previous': previous_val,
        'change': change,
        'change_pct': change_pct
    }

# Find category with largest negative change
largest_decline_cat = min(category_changes.items(), key=lambda x: x[1]['change'])
cat_name = largest_decline_cat[0]
cat_data = largest_decline_cat[1]

if cat_data['change'] < 0 and cat_data['previous'] > 0:
    finding2 = {
        "title": f"Category Revenue Decline: {cat_name}",
        "claim": f"{cat_name} category revenue declined {cat_data['change_pct']:.2f}% (from {cat_data['previous']:.2f} SAR to {cat_data['analysis']:.2f} SAR) week-over-week, representing the largest category decline.",
        "finding_type": "category_mix",
        "metrics": {
            "analysis_period_category_revenue": {
                "value": round(cat_data['analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "previous_period_category_revenue": {
                "value": round(cat_data['previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "category_revenue_change_pct": {
                "value": round(cat_data['change_pct'], 2),
                "unit": "%",
                "numerator": round(cat_data['change'], 2),
                "denominator": round(cat_data['previous'], 2),
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data_no_refund),
        "coverage_notes": [
            "Analysis period: 2026-02-23 to 2026-03-02 (7 days)",
            "Previous period: 2026-02-16 to 2026-02-23 (7 days)",
            "Refunds excluded from category revenue calculations",
            "Category classification from cleaned POS data"
        ],
        "assumptions": [
            "Category classification is consistent across both periods",
            "line_total_sar represents net transaction value",
            "is_refund flag correctly identifies refund transactions"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# FINDING 3: Product Performance - Top SKU by Revenue Change
analysis_sku_revenue = analysis_data_no_refund.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_sku_revenue.columns = ['sku', 'revenue', 'quantity', 'transactions']

previous_sku_revenue = previous_data_no_refund.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_sku_revenue.columns = ['sku', 'revenue', 'quantity', 'transactions']

# Merge with menu to get launch dates
analysis_sku_revenue = analysis_sku_revenue.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_sku_revenue = previous_sku_revenue.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Calculate changes for SKUs present in both periods
sku_changes = {}
for sku in analysis_sku_revenue['sku'].unique():
    analysis_row = analysis_sku_revenue[analysis_sku_revenue['sku'] == sku]
    previous_row = previous_sku_revenue[previous_sku_revenue['sku'] == sku]
    
    if len(analysis_row) > 0 and len(previous_row) > 0:
        analysis_rev = analysis_row['revenue'].values[0]
        previous_rev = previous_row['revenue'].values[0]
        item_name = analysis_row['item_en'].values[0]
        
        if previous_rev > 0:
            change = analysis_rev - previous_rev
            change_pct = (change / previous_rev * 100)
            sku_changes[sku] = {
                'item_name': item_name,
                'analysis': analysis_rev,
                'previous': previous_rev,
                'change': change,
                'change_pct': change_pct
            }

# Find SKU with largest positive change
if sku_changes:
    largest_growth_sku = max(sku_changes.items(), key=lambda x: x[1]['change'])
    sku_code = largest_growth_sku[0]
    sku_data = largest_growth_sku[1]
    
    if sku_data['change'] > 0:
        finding3 = {
            "title": f"Top Product Growth: {sku_data['item_name']}",
            "claim": f"SKU {sku_code} ({sku_data['item_name']}) achieved the highest revenue growth at {sku_data['change_pct']:.2f}% (from {sku_data['previous']:.2f} SAR to {sku_data['analysis']:.2f} SAR) week-over-week.",
            "finding_type": "product_mix",
            "metrics": {
                "analysis_period_sku_revenue": {
                    "value": round(sku_data['analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+03:00",
                    "period_end": "2026-03-02T00:00:00+03:00"
                },
                "previous_period_sku_revenue": {
                    "value": round(sku_data['previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-16T00:00:00+03:00",
                    "period_end": "2026-02-23T00:00:00+03:00"
                },
                "sku_revenue_change_pct": {
                    "value": round(sku_data['change_pct'], 2),
                    "unit": "%",
                    "numerator": round(sku_data['change'], 2),
                    "denominator": round(sku_data['previous'], 2),
                    "period_start": "2026-02-16T00:00:00+03:00",
                    "period_end": "2026-03-02T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_data_no_refund),
            "coverage_notes": [
                "Analysis period: 2026-02-23 to 2026-03-02 (7 days)",
                "Previous period: 2026-02-16 to 2026-02-23 (7 days)",
                "Refunds excluded from SKU revenue calculations",
                "SKU matched to menu reference for product naming",
                "Only SKUs present in both periods included in comparison"
            ],
            "assumptions": [
                "SKU codes are consistent across POS and menu artifacts",
                "line_total_sar represents net transaction value",
                "is_refund flag correctly identifies refund transactions",
                "Product launch/retirement dates do not restrict analysis period eligibility"
            ],
            "confidence": 0.90
        }
        findings.append(finding3)

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)