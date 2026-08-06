import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load environment configuration
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_period = {
    "start": "2026-02-16T00:00:00+03:00",
    "end": "2026-02-23T00:00:00+03:00"
}
previous_period = {
    "start": "2026-02-09T00:00:00+03:00",
    "end": "2026-02-16T00:00:00+03:00"
}
trailing_baseline_periods = [
    {
        "start": "2026-02-09T00:00:00+03:00",
        "end": "2026-02-16T00:00:00+03:00"
    },
    {
        "start": "2026-02-02T00:00:00+03:00",
        "end": "2026-02-09T00:00:00+03:00"
    },
    {
        "start": "2026-01-26T00:00:00+03:00",
        "end": "2026-02-02T00:00:00+03:00"
    },
    {
        "start": "2026-01-19T00:00:00+03:00",
        "end": "2026-01-26T00:00:00+03:00"
    }
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter data by period
def filter_by_period(df, period_start, period_end):
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Analysis 1: Revenue and Transaction Count Comparison (Analysis Period vs Previous Period)
analysis_data = filter_by_period(pos_df, analysis_period['start'], analysis_period['end'])
previous_data = filter_by_period(pos_df, previous_period['start'], previous_period['end'])

# Count valid transactions (unique transaction_id)
analysis_transactions = analysis_data['transaction_id'].nunique()
previous_transactions = previous_data['transaction_id'].nunique()

# Calculate revenue (sum of line_total_sar)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Calculate AOV
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change = analysis_transactions - previous_transactions
transaction_pct_change = (transaction_change / previous_transactions * 100) if previous_transactions > 0 else 0

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Analysis 2: Category Mix Analysis
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum()
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum()

# Find categories with significant changes
category_changes = {}
for category in analysis_category_revenue.index:
    if category in previous_category_revenue.index:
        curr_rev = analysis_category_revenue[category]
        prev_rev = previous_category_revenue[category]
        change = curr_rev - prev_rev
        pct_change = (change / prev_rev * 100) if prev_rev != 0 else 0
        category_changes[category] = {
            'current': curr_rev,
            'previous': prev_rev,
            'change': change,
            'pct_change': pct_change
        }

# Find the category with largest absolute change
largest_category_change = max(category_changes.items(), key=lambda x: abs(x[1]['change']))

# Analysis 3: Channel Mix Analysis
analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum()
previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum()

# Calculate channel mix percentages
analysis_channel_mix = (analysis_channel_revenue / analysis_revenue * 100) if analysis_revenue != 0 else 0
previous_channel_mix = (previous_channel_revenue / previous_revenue * 100) if previous_revenue != 0 else 0

# Analysis 4: Product Performance (Top products by revenue change)
analysis_product_revenue = analysis_data.groupby('sku')['line_total_sar'].sum()
previous_product_revenue = previous_data.groupby('sku')['line_total_sar'].sum()

product_changes = {}
for sku in analysis_product_revenue.index:
    if sku in previous_product_revenue.index:
        curr_rev = analysis_product_revenue[sku]
        prev_rev = previous_product_revenue[sku]
        change = curr_rev - prev_rev
        pct_change = (change / prev_rev * 100) if prev_rev != 0 else 0
        product_changes[sku] = {
            'current': curr_rev,
            'previous': prev_rev,
            'change': change,
            'pct_change': pct_change
        }

# Find products with significant changes
top_product_changes = sorted(product_changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)[:3]

# Check for refunds
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Performance
if analysis_revenue != 0 and previous_revenue != 0:
    findings.append({
        "title": "Revenue and Transaction Performance",
        "claim": f"Analysis period (Feb 16-23) generated SAR {analysis_revenue:.2f} in revenue across {analysis_transactions} transactions, compared to SAR {previous_revenue:.2f} across {previous_transactions} transactions in the previous week. Revenue changed by SAR {revenue_change:.2f} ({revenue_pct_change:.2f}%), while transaction count changed by {transaction_change} ({transaction_pct_change:.2f}%). Average order value increased from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}, a change of SAR {aov_change:.2f} ({aov_pct_change:.2f}%).",
        "finding_type": "revenue_and_transaction_analysis",
        "metrics": {
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "analysis_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "previous_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_change": {
                "value": transaction_change,
                "unit": "count",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_pct_change": {
                "value": round(transaction_pct_change, 2),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Refunds included in net revenue calculations",
            f"Analysis period refunds: SAR {analysis_refunds:.2f}",
            f"Previous period refunds: SAR {previous_refunds:.2f}",
            "All valid transactions counted using unique transaction_id"
        ],
        "assumptions": [
            "line_total_sar represents net revenue (including refunds)",
            "AOV = total revenue / unique transaction count",
            "Periods are non-overlapping and consecutive weeks"
        ],
        "confidence": 0.95
    })

# Finding 2: Category Mix Analysis
if largest_category_change:
    category_name = largest_category_change[0]
    category_data = largest_category_change[1]
    
    findings.append({
        "title": f"Category Revenue Change: {category_name}",
        "claim": f"The {category_name} category experienced a revenue change from SAR {category_data['previous']:.2f} in the previous week to SAR {category_data['current']:.2f} in the analysis period, representing a change of SAR {category_data['change']:.2f} ({category_data['pct_change']:.2f}%). This is the largest absolute category revenue change between the two periods.",
        "finding_type": "category_mix_analysis",
        "metrics": {
            "category_current_revenue": {
                "value": round(category_data['current'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "category_previous_revenue": {
                "value": round(category_data['previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "category_revenue_change": {
                "value": round(category_data['change'], 2),
                "unit": "SAR",
                "numerator": round(category_data['change'], 2),
                "denominator": round(category_data['previous'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "category_revenue_pct_change": {
                "value": round(category_data['pct_change'], 2),
                "unit": "%",
                "numerator": round(category_data['change'], 2),
                "denominator": round(category_data['previous'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Refunds included in net revenue calculations",
            "Category analysis based on POS line items"
        ],
        "assumptions": [
            "line_total_sar represents net revenue (including refunds)",
            "Category classification from POS data is accurate"
        ],
        "confidence": 0.90
    })

# Finding 3: Top Product Performance
if top_product_changes:
    top_product = top_product_changes[0]
    sku = top_product[0]
    product_data = top_product[1]
    
    # Get product name from menu
    product_name = sku
    menu_match = menu_df[menu_df['sku'] == sku]
    if not menu_match.empty:
        product_name = menu_match.iloc[0]['item_en']
    
    findings.append({
        "title": f"Top Product Revenue Change: {product_name}",
        "claim": f"SKU {sku} ({product_name}) showed the largest absolute revenue change, moving from SAR {product_data['previous']:.2f} in the previous week to SAR {product_data['current']:.2f} in the analysis period, a change of SAR {product_data['change']:.2f} ({product_data['pct_change']:.2f}%).",
        "finding_type": "product_performance_analysis",
        "metrics": {
            "product_current_revenue": {
                "value": round(product_data['current'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "product_previous_revenue": {
                "value": round(product_data['previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "product_revenue_change": {
                "value": round(product_data['change'], 2),
                "unit": "SAR",
                "numerator": round(product_data['change'], 2),
                "denominator": round(product_data['previous'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "product_revenue_pct_change": {
                "value": round(product_data['pct_change'], 2),
                "unit": "%",
                "numerator": round(product_data['change'], 2),
                "denominator": round(product_data['previous'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Refunds included in net revenue calculations",
            "Product name resolved from menu SKU reference"
        ],
        "assumptions": [
            "line_total_sar represents net revenue (including refunds)",
            "SKU to product name mapping from menu is accurate"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
