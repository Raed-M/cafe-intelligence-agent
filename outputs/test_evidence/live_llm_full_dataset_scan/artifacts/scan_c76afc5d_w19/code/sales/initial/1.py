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
analysis_start = datetime.fromisoformat("2026-05-18T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-05-25T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-05-11T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-05-18T00:00:00+03:00")

trailing_periods = [
    ("2026-05-11T00:00:00+03:00", "2026-05-18T00:00:00+03:00"),
    ("2026-05-04T00:00:00+03:00", "2026-05-11T00:00:00+03:00"),
    ("2026-04-27T00:00:00+03:00", "2026-05-04T00:00:00+03:00"),
    ("2026-04-20T00:00:00+03:00", "2026-04-27T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter by period
def filter_period(df, start_iso, end_iso):
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Filter analysis period
analysis_df = filter_period(pos_df, "2026-05-18T00:00:00+03:00", "2026-05-25T00:00:00+03:00")
previous_df = filter_period(pos_df, "2026-05-11T00:00:00+03:00", "2026-05-18T00:00:00+03:00")

# Filter trailing baseline periods
trailing_dfs = []
for start_iso, end_iso in trailing_periods:
    trailing_dfs.append(filter_period(pos_df, start_iso, end_iso))

# Combine trailing baseline
trailing_df = pd.concat(trailing_dfs, ignore_index=True)

# Calculate metrics for analysis period
analysis_valid_txns = analysis_df[~analysis_df['is_refund']]['transaction_id'].nunique()
analysis_revenue = analysis_df['line_total_sar'].sum()
analysis_refund_count = analysis_df[analysis_df['is_refund']]['transaction_id'].nunique()
analysis_refund_revenue = analysis_df[analysis_df['is_refund']]['line_total_sar'].sum()

# Calculate metrics for previous period
previous_valid_txns = previous_df[~previous_df['is_refund']]['transaction_id'].nunique()
previous_revenue = previous_df['line_total_sar'].sum()
previous_refund_count = previous_df[previous_df['is_refund']]['transaction_id'].nunique()
previous_refund_revenue = previous_df[previous_df['is_refund']]['line_total_sar'].sum()

# Calculate metrics for trailing baseline
trailing_valid_txns = trailing_df[~trailing_df['is_refund']]['transaction_id'].nunique()
trailing_revenue = trailing_df['line_total_sar'].sum()

# Calculate AOV
analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0
previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0
trailing_aov = trailing_revenue / trailing_valid_txns if trailing_valid_txns > 0 else 0

# Calculate transaction change
txn_change = analysis_valid_txns - previous_valid_txns
txn_change_pct = (txn_change / previous_valid_txns * 100) if previous_valid_txns > 0 else 0

# Calculate revenue change
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0

# Calculate AOV change
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

# Category mix analysis
analysis_category_revenue = analysis_df.groupby('category')['line_total_sar'].sum()
previous_category_revenue = previous_df.groupby('category')['line_total_sar'].sum()

# Product performance - join with menu for launch dates
analysis_product = analysis_df.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product = previous_df.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')

# Filter products by eligibility (launched before analysis period, not retired)
analysis_product['launch_date'] = pd.to_datetime(analysis_product['launch_date'], errors='coerce')
analysis_product['retire_date'] = pd.to_datetime(analysis_product['retire_date'], errors='coerce')

analysis_eligible = analysis_product[
    (analysis_product['launch_date'].isna() | (analysis_product['launch_date'] < analysis_start)) &
    (analysis_product['retire_date'].isna() | (analysis_product['retire_date'] >= analysis_start))
]

previous_product['launch_date'] = pd.to_datetime(previous_product['launch_date'], errors='coerce')
previous_product['retire_date'] = pd.to_datetime(previous_product['retire_date'], errors='coerce')

previous_eligible = previous_product[
    (previous_product['launch_date'].isna() | (previous_product['launch_date'] < previous_start)) &
    (previous_product['retire_date'].isna() | (previous_product['retire_date'] >= previous_start))
]

# Top products by revenue
analysis_top_products = analysis_eligible.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first'
}).sort_values('line_total_sar', ascending=False).head(5)

previous_top_products = previous_eligible.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first'
}).sort_values('line_total_sar', ascending=False).head(5)

# Channel mix
analysis_channel = analysis_df.groupby('channel')['line_total_sar'].sum()
previous_channel = previous_df.groupby('channel')['line_total_sar'].sum()

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Performance
if analysis_valid_txns > 0 and previous_valid_txns > 0:
    findings.append({
        "title": "Weekly Revenue and Transaction Performance",
        "claim": f"Analysis week (2026-05-18 to 2026-05-25) generated SAR {analysis_revenue:.2f} in net revenue across {analysis_valid_txns} valid transactions, representing a {revenue_change_pct:.1f}% change from previous week (SAR {previous_revenue:.2f}, {previous_valid_txns} transactions). Refunds totaled SAR {abs(analysis_refund_revenue):.2f} ({analysis_refund_count} transactions) in analysis week vs SAR {abs(previous_refund_revenue):.2f} ({previous_refund_count} transactions) previously.",
        "finding_type": "revenue_and_transaction_performance",
        "metrics": {
            "analysis_week_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "analysis_week_transactions": {
                "value": analysis_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "previous_week_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "previous_week_transactions": {
                "value": previous_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "transaction_change": {
                "value": txn_change,
                "unit": "count",
                "numerator": txn_change,
                "denominator": previous_valid_txns,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(txn_change_pct, 1),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_valid_txns,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "analysis_week_refund_revenue": {
                "value": round(analysis_refund_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "previous_week_refund_revenue": {
                "value": round(previous_refund_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            "Analysis period: 2026-05-18 to 2026-05-25 (7 days)",
            "Previous period: 2026-05-11 to 2026-05-18 (7 days)",
            "Refunds included in net revenue calculations",
            "Valid transactions counted by unique transaction_id excluding refund-only transactions"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue",
            "is_refund flag correctly identifies refund transactions",
            "transaction_id uniquely identifies a basket"
        ],
        "confidence": 0.95
    })

# Finding 2: Average Order Value
if analysis_valid_txns > 0 and previous_valid_txns > 0:
    findings.append({
        "title": "Average Order Value Trend",
        "claim": f"Average order value in analysis week was SAR {analysis_aov:.2f} per transaction, compared to SAR {previous_aov:.2f} in previous week, representing a {aov_change_pct:.1f}% change. Trailing 4-week baseline AOV was SAR {trailing_aov:.2f}.",
        "finding_type": "average_order_value",
        "metrics": {
            "analysis_week_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "previous_week_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "trailing_4week_aov": {
                "value": round(trailing_aov, 2),
                "unit": "SAR",
                "numerator": round(trailing_revenue, 2),
                "denominator": trailing_valid_txns,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            "AOV calculated as net revenue divided by valid transaction count",
            "Refunds included in net revenue",
            "Trailing baseline spans 4 weeks: 2026-04-20 to 2026-05-18"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue per line item",
            "transaction_id uniquely identifies a basket",
            "is_refund flag correctly identifies refund transactions"
        ],
        "confidence": 0.92
    })

# Finding 3: Top Product Performance with Menu Join
if len(analysis_top_products) > 0 and len(previous_top_products) > 0:
    # Get top product from analysis week
    top_sku = analysis_top_products.index[0]
    top_name = analysis_top_products.iloc[0]['item_name_en']
    top_analysis_revenue = analysis_top_products.iloc[0]['line_total_sar']
    top_analysis_qty = analysis_top_products.iloc[0]['quantity']
    
    # Get same product from previous week if available
    if top_sku in previous_top_products.index:
        top_previous_revenue = previous_top_products.loc[top_sku, 'line_total_sar']
        top_previous_qty = previous_top_products.loc[top_sku, 'quantity']
        top_revenue_change = top_analysis_revenue - top_previous_revenue
        top_revenue_change_pct = (top_revenue_change / top_previous_revenue * 100) if top_previous_revenue > 0 else 0
        
        findings.append({
            "title": "Top Product Performance",
            "claim": f"Top revenue product {top_name} (SKU: {top_sku}) generated SAR {top_analysis_revenue:.2f} across {int(top_analysis_qty)} units in analysis week, compared to SAR {top_previous_revenue:.2f} ({int(top_previous_qty)} units) in previous week, a {top_revenue_change_pct:.1f}% change. Product eligibility verified through menu launch/retire dates.",
            "finding_type": "product_performance",
            "metrics": {
                "analysis_week_product_revenue": {
                    "value": round(top_analysis_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "analysis_week_product_quantity": {
                    "value": int(top_analysis_qty),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "previous_week_product_revenue": {
                    "value": round(top_previous_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-11T00:00:00+03:00",
                    "period_end": "2026-05-18T00:00:00+03:00"
                },
                "previous_week_product_quantity": {
                    "value": int(top_previous_qty),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-11T00:00:00+03:00",
                    "period_end": "2026-05-18T00:00:00+03:00"
                },
                "product_revenue_change_sar": {
                    "value": round(top_revenue_change, 2),
                    "unit": "SAR",
                    "numerator": round(top_revenue_change, 2),
                    "denominator": round(top_previous_revenue, 2),
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "product_revenue_change_pct": {
                    "value": round(top_revenue_change_pct, 1),
                    "unit": "%",
                    "numerator": round(top_revenue_change, 2),
                    "denominator": round(top_previous_revenue, 2),
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "product_sku": {
                    "value": top_sku,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "product_name": {
                    "value": top_name,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(top_analysis_qty),
            "coverage_notes": [
                "Product eligibility verified: launched before analysis period start and not retired",
                "Top product identified by revenue in analysis week",
                "Comparison includes only products eligible in both periods"
            ],
            "assumptions": [
                "Menu launch_date and retire_date fields correctly indicate product availability",
                "line_total_sar represents realized net revenue",
                "SKU correctly links POS and menu records"
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