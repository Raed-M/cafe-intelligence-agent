import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-01-05T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-12T00:00:00+03:00")
previous_start = datetime.fromisoformat("2025-12-29T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-01-05T00:00:00+03:00")

trailing_periods = [
    (datetime.fromisoformat("2025-12-29T00:00:00+03:00"), datetime.fromisoformat("2026-01-05T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-22T00:00:00+03:00"), datetime.fromisoformat("2025-12-29T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-15T00:00:00+03:00"), datetime.fromisoformat("2025-12-22T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-08T00:00:00+03:00"), datetime.fromisoformat("2025-12-15T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Calculate metrics for analysis period
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

# Calculate metrics for previous period
previous_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

# Calculate trailing baseline average
trailing_revenue_total = 0
trailing_transactions_total = 0
for period_start, period_end in trailing_periods:
    period_data = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    trailing_revenue_total += period_data['line_total_sar'].sum()
    trailing_transactions_total += period_data['transaction_id'].nunique()

trailing_avg_revenue = trailing_revenue_total / len(trailing_periods) if len(trailing_periods) > 0 else 0
trailing_avg_transactions = trailing_transactions_total / len(trailing_periods) if len(trailing_periods) > 0 else 0
trailing_avg_aov = trailing_revenue_total / trailing_transactions_total if trailing_transactions_total > 0 else 0

# Revenue change analysis
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

# Transaction count change
transaction_change = analysis_transactions - previous_transactions
transaction_change_pct = (transaction_change / previous_transactions * 100) if previous_transactions != 0 else 0

# AOV change
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Category mix analysis
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Channel mix analysis
analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

# Product performance - top products in analysis period
analysis_product_revenue = analysis_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_revenue.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']
analysis_product_revenue = analysis_product_revenue.sort_values('revenue', ascending=False)

# Previous period product performance
previous_product_revenue = previous_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_revenue.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']
previous_product_revenue = previous_product_revenue.sort_values('revenue', ascending=False)

# Check for product launches/retirements
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')

# Find products launched during or just before analysis period
recently_launched = menu_df[
    (menu_df['launch_date'] >= (analysis_start - timedelta(days=7))) & 
    (menu_df['launch_date'] <= analysis_end)
]

findings = []

# Finding 1: Revenue change week-over-week
if previous_revenue != 0:
    finding1 = {
        "title": "Weekly Revenue Performance",
        "claim": f"Total net revenue for week of {analysis_start.date()} was SAR {analysis_revenue:.2f}, representing a {revenue_change_pct:.1f}% change from the previous week (SAR {previous_revenue:.2f}). This includes refunds totaling SAR {analysis_refunds:.2f}.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "refunds_included": {
                "value": round(analysis_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Previous period: {previous_start.isoformat()} to {previous_end.isoformat()}",
            f"POS records in analysis period: {len(analysis_data)}",
            f"Unique transactions in analysis period: {analysis_transactions}"
        ],
        "assumptions": [
            "line_total_sar represents net revenue after discounts",
            "is_refund flag correctly identifies refund transactions",
            "All transactions within the specified periods are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Transaction count and AOV change
if previous_transactions > 0:
    finding2 = {
        "title": "Transaction Volume and Average Order Value",
        "claim": f"Transaction count for the analysis week was {analysis_transactions}, a {transaction_change_pct:.1f}% change from {previous_transactions} in the previous week. Average order value changed from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}, a {aov_change_pct:.1f}% change.",
        "finding_type": "transaction_and_aov_change",
        "metrics": {
            "analysis_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "transaction_change_pct": {
                "value": round(transaction_change_pct, 1),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Transaction count based on unique transaction_id values",
            f"AOV calculated as total revenue divided by transaction count",
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Previous period: {previous_start.isoformat()} to {previous_end.isoformat()}"
        ],
        "assumptions": [
            "Each unique transaction_id represents one basket/transaction",
            "line_total_sar includes all discounts and refunds",
            "Transaction timestamps are accurate and in the specified timezone"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Top category performance change
if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
    top_category_analysis = analysis_category_revenue.index[0]
    top_category_previous = previous_category_revenue.index[0]
    
    if top_category_analysis in previous_category_revenue.index:
        category_revenue_analysis = analysis_category_revenue[top_category_analysis]
        category_revenue_previous = previous_category_revenue[top_category_analysis]
        category_change = category_revenue_analysis - category_revenue_previous
        category_change_pct = (category_change / category_revenue_previous * 100) if category_revenue_previous != 0 else 0
        
        finding3 = {
            "title": "Top Category Revenue Performance",
            "claim": f"The top revenue category '{top_category_analysis}' generated SAR {category_revenue_analysis:.2f} in the analysis week, compared to SAR {category_revenue_previous:.2f} in the previous week, representing a {category_change_pct:.1f}% change.",
            "finding_type": "category_performance",
            "metrics": {
                "top_category_name": {
                    "value": top_category_analysis,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "analysis_category_revenue": {
                    "value": round(category_revenue_analysis, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "previous_category_revenue": {
                    "value": round(category_revenue_previous, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "category_revenue_change_pct": {
                    "value": round(category_change_pct, 1),
                    "unit": "%",
                    "numerator": round(category_change, 2),
                    "denominator": round(category_revenue_previous, 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Category revenue aggregated from POS line items",
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Previous period: {previous_start.isoformat()} to {previous_end.isoformat()}",
                f"Number of categories in analysis period: {len(analysis_category_revenue)}"
            ],
            "assumptions": [
                "Category field in POS data is accurate and complete",
                "line_total_sar represents net revenue for each line item",
                "All transactions are properly categorized"
            ],
            "confidence": 0.90
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
