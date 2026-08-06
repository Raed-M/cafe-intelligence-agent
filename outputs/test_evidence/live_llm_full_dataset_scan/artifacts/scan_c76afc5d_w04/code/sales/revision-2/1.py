import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

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
    "start": "2026-02-02T00:00:00+03:00",
    "end": "2026-02-09T00:00:00+03:00"
}
previous_period = {
    "start": "2026-01-26T00:00:00+03:00",
    "end": "2026-02-02T00:00:00+03:00"
}
trailing_baseline_periods = [
    {
        "start": "2026-01-26T00:00:00+03:00",
        "end": "2026-02-02T00:00:00+03:00"
    },
    {
        "start": "2026-01-19T00:00:00+03:00",
        "end": "2026-01-26T00:00:00+03:00"
    },
    {
        "start": "2026-01-12T00:00:00+03:00",
        "end": "2026-01-19T00:00:00+03:00"
    },
    {
        "start": "2026-01-05T00:00:00+03:00",
        "end": "2026-01-12T00:00:00+03:00"
    }
]

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

def filter_by_period(df, period_start, period_end):
    """Filter dataframe by period"""
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Filter data for each period
analysis_data = filter_by_period(pos_df, analysis_period['start'], analysis_period['end'])
previous_data = filter_by_period(pos_df, previous_period['start'], previous_period['end'])

# Calculate metrics for analysis period
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

# Calculate metrics for previous period
previous_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

# Calculate trailing baseline (average of 4 weeks)
trailing_data_list = []
for period in trailing_baseline_periods:
    period_data = filter_by_period(pos_df, period['start'], period['end'])
    trailing_data_list.append(period_data)

trailing_data = pd.concat(trailing_data_list, ignore_index=True)
trailing_transactions = trailing_data['transaction_id'].nunique()
trailing_revenue = trailing_data['line_total_sar'].sum()
trailing_aov = trailing_revenue / trailing_transactions if trailing_transactions > 0 else 0

# Calculate changes
revenue_change_vs_previous = analysis_revenue - previous_revenue
revenue_pct_change_vs_previous = (revenue_change_vs_previous / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change_vs_previous = analysis_transactions - previous_transactions
transaction_pct_change_vs_previous = (transaction_change_vs_previous / previous_transactions * 100) if previous_transactions > 0 else 0

aov_change_vs_previous = analysis_aov - previous_aov
aov_pct_change_vs_previous = (aov_change_vs_previous / previous_aov * 100) if previous_aov != 0 else 0

revenue_change_vs_trailing = analysis_revenue - trailing_revenue
revenue_pct_change_vs_trailing = (revenue_change_vs_trailing / trailing_revenue * 100) if trailing_revenue != 0 else 0

# Analyze by category
analysis_by_category = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'transactions']

previous_by_category = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'transactions']

# Merge and calculate changes by category
category_comparison = analysis_by_category.merge(
    previous_by_category, 
    on='category', 
    suffixes=('_analysis', '_previous')
)

category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100).round(2)
category_comparison['transaction_change'] = category_comparison['transactions_analysis'] - category_comparison['transactions_previous']
category_comparison['transaction_pct_change'] = (category_comparison['transaction_change'] / category_comparison['transactions_previous'] * 100).round(2)

# Analyze by channel
analysis_by_channel = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'transactions']

previous_by_channel = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'transactions']

# Merge and calculate changes by channel
channel_comparison = analysis_by_channel.merge(
    previous_by_channel,
    on='channel',
    suffixes=('_analysis', '_previous')
)

channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
channel_comparison['revenue_pct_change'] = (channel_comparison['revenue_change'] / channel_comparison['revenue_previous'] * 100).round(2)
channel_comparison['transaction_change'] = channel_comparison['transactions_analysis'] - channel_comparison['transactions_previous']
channel_comparison['transaction_pct_change'] = (channel_comparison['transaction_change'] / channel_comparison['transactions_previous'] * 100).round(2)

# Analyze product performance
# Get product-level data for analysis period
analysis_products = analysis_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_products.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']

# Get product-level data for previous period
previous_products = previous_data.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_products.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']

# Merge product data
product_comparison = analysis_products.merge(
    previous_products,
    on=['sku', 'item_name'],
    suffixes=('_analysis', '_previous'),
    how='outer'
)

# Fill NaN with 0 for products that didn't exist in one period
product_comparison = product_comparison.fillna(0)

# Calculate changes
product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['revenue_pct_change'] = np.where(
    product_comparison['revenue_previous'] != 0,
    (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100).round(2),
    np.nan
)

# Check for refunds in analysis period
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]
analysis_refund_count = len(analysis_refunds)
analysis_refund_value = analysis_refunds['line_total_sar'].sum()

previous_refunds = previous_data[previous_data['is_refund'] == True]
previous_refund_count = len(previous_refunds)
previous_refund_value = previous_refunds['line_total_sar'].sum()

# Identify top performing categories by revenue change
top_category_changes = category_comparison.nlargest(3, 'revenue_change')
bottom_category_changes = category_comparison.nsmallest(3, 'revenue_change')

# Identify top performing products by revenue change
top_product_changes = product_comparison[product_comparison['revenue_analysis'] > 0].nlargest(5, 'revenue_change')

# Build findings
findings = []

# Finding 1: Overall revenue and transaction performance
if abs(revenue_pct_change_vs_previous) > 2 or abs(transaction_pct_change_vs_previous) > 2:
    findings.append({
        "title": "Revenue and Transaction Performance vs Previous Week",
        "claim": f"Analysis week (Feb 2-9, 2026) generated {analysis_revenue:.2f} SAR in net revenue across {analysis_transactions} transactions, representing a {revenue_pct_change_vs_previous:.1f}% change in revenue and {transaction_pct_change_vs_previous:.1f}% change in transaction count compared to the previous week (Jan 26-Feb 2, 2026).",
        "finding_type": "revenue_and_transaction_performance",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "revenue_change_sar": {
                "value": round(revenue_change_vs_previous, 2),
                "unit": "SAR",
                "numerator": round(revenue_change_vs_previous, 2),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change_vs_previous, 2),
                "unit": "%",
                "numerator": round(revenue_change_vs_previous, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_change": {
                "value": transaction_change_vs_previous,
                "unit": "count",
                "numerator": transaction_change_vs_previous,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_pct_change": {
                "value": round(transaction_pct_change_vs_previous, 2),
                "unit": "%",
                "numerator": transaction_change_vs_previous,
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
                "value": round(aov_change_vs_previous, 2),
                "unit": "SAR",
                "numerator": round(aov_change_vs_previous, 2),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Revenue includes refunds as negative line items (net calculation)",
            "Transaction count based on unique transaction_id values",
            "AOV calculated as total revenue divided by transaction count",
            f"Analysis period contains {analysis_refund_count} refund line items totaling {analysis_refund_value:.2f} SAR",
            f"Previous period contains {previous_refund_count} refund line items totaling {previous_refund_value:.2f} SAR"
        ],
        "assumptions": [
            "line_total_sar represents net realized revenue including refunds",
            "Each unique transaction_id represents one customer basket",
            "Refunds are included as negative values in line_total_sar",
            "Periods are non-overlapping and aligned to calendar weeks"
        ],
        "confidence": 0.95
    })

# Finding 2: Category-level performance
if len(category_comparison) > 0:
    # Find the category with largest absolute revenue change
    largest_category_change = category_comparison.loc[category_comparison['revenue_change'].abs().idxmax()]
    
    if abs(largest_category_change['revenue_pct_change']) > 5:
        findings.append({
            "title": f"Category Performance: {largest_category_change['category']} Revenue Change",
            "claim": f"The {largest_category_change['category']} category generated {largest_category_change['revenue_analysis']:.2f} SAR in the analysis week (Feb 2-9, 2026) compared to {largest_category_change['revenue_previous']:.2f} SAR in the previous week (Jan 26-Feb 2, 2026), representing a {largest_category_change['revenue_pct_change']:.1f}% change. Transaction count changed from {int(largest_category_change['transactions_previous'])} to {int(largest_category_change['transactions_analysis'])} transactions ({largest_category_change['transaction_pct_change']:.1f}% change).",
            "finding_type": "category_performance",
            "metrics": {
                "category_name": {
                    "value": largest_category_change['category'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "analysis_revenue": {
                    "value": round(largest_category_change['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "previous_revenue": {
                    "value": round(largest_category_change['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": previous_period['end']
                },
                "revenue_change": {
                    "value": round(largest_category_change['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(largest_category_change['revenue_change'], 2),
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "revenue_pct_change": {
                    "value": round(largest_category_change['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": round(largest_category_change['revenue_change'], 2),
                    "denominator": round(largest_category_change['revenue_previous'], 2),
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "analysis_transactions": {
                    "value": int(largest_category_change['transactions_analysis']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "previous_transactions": {
                    "value": int(largest_category_change['transactions_previous']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": previous_period['end']
                },
                "transaction_pct_change": {
                    "value": round(largest_category_change['transaction_pct_change'], 2),
                    "unit": "%",
                    "numerator": int(largest_category_change['transaction_change']),
                    "denominator": int(largest_category_change['transactions_previous']),
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Category-level analysis based on POS transaction data",
                "Revenue includes refunds as negative line items",
                "Transaction count represents unique transaction_id per category",
                "Categories derived from cleaned POS data"
            ],
            "assumptions": [
                "Category assignments are accurate in cleaned POS data",
                "Revenue figures are net of refunds",
                "Transactions are properly attributed to categories"
            ],
            "confidence": 0.90
        })

# Finding 3: Channel performance
if len(channel_comparison) > 0:
    # Find the channel with largest absolute revenue change
    largest_channel_change = channel_comparison.loc[channel_comparison['revenue_change'].abs().idxmax()]
    
    if abs(largest_channel_change['revenue_pct_change']) > 3:
        findings.append({
            "title": f"Channel Performance: {largest_channel_change['channel']} Revenue Change",
            "claim": f"The {largest_channel_change['channel']} channel generated {largest_channel_change['revenue_analysis']:.2f} SAR in the analysis week (Feb 2-9, 2026) compared to {largest_channel_change['revenue_previous']:.2f} SAR in the previous week (Jan 26-Feb 2, 2026), representing a {largest_channel_change['revenue_pct_change']:.1f}% change. Transaction count changed from {int(largest_channel_change['transactions_previous'])} to {int(largest_channel_change['transactions_analysis'])} transactions ({largest_channel_change['transaction_pct_change']:.1f}% change).",
            "finding_type": "channel_performance",
            "metrics": {
                "channel_name": {
                    "value": largest_channel_change['channel'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "analysis_revenue": {
                    "value": round(largest_channel_change['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "previous_revenue": {
                    "value": round(largest_channel_change['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": previous_period['end']
                },
                "revenue_change": {
                    "value": round(largest_channel_change['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(largest_channel_change['revenue_change'], 2),
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "revenue_pct_change": {
                    "value": round(largest_channel_change['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": round(largest_channel_change['revenue_change'], 2),
                    "denominator": round(largest_channel_change['revenue_previous'], 2),
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "analysis_transactions": {
                    "value": int(largest_channel_change['transactions_analysis']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "previous_transactions": {
                    "value": int(largest_channel_change['transactions_previous']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": previous_period['end']
                },
                "transaction_pct_change": {
                    "value": round(largest_channel_change['transaction_pct_change'], 2),
                    "unit": "%",
                    "numerator": int(largest_channel_change['transaction_change']),
                    "denominator": int(largest_channel_change['transactions_previous']),
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Channel-level analysis based on POS transaction data",
                "Revenue includes refunds as negative line items",
                "Transaction count represents unique transaction_id per channel",
                "Channels derived from cleaned POS data"
            ],
            "assumptions": [
                "Channel assignments are accurate in cleaned POS data",
                "Revenue figures are net of refunds",
                "Transactions are properly attributed to channels"
            ],
            "confidence": 0.90
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
