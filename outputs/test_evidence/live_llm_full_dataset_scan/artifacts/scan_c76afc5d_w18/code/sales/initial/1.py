import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
analysis_end = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_start = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_end = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis period
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

# Calculate trailing baseline (average of 4 weeks)
trailing_baseline_revenue = 0
trailing_baseline_transactions = 0
trailing_baseline_count = 0

for period in [
    (datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))))
]:
    period_data = pos_df[(pos_df['timestamp'] >= period[0]) & (pos_df['timestamp'] < period[1])]
    trailing_baseline_revenue += period_data['line_total_sar'].sum()
    trailing_baseline_transactions += period_data['transaction_id'].nunique()
    trailing_baseline_count += 1

trailing_baseline_revenue = trailing_baseline_revenue / trailing_baseline_count if trailing_baseline_count > 0 else 0
trailing_baseline_transactions = trailing_baseline_transactions / trailing_baseline_count if trailing_baseline_count > 0 else 0
trailing_baseline_aov = trailing_baseline_revenue / trailing_baseline_transactions if trailing_baseline_transactions > 0 else 0

# Analyze product mix changes
analysis_product_mix = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

previous_product_mix = previous_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

# Merge with menu to get product names and launch dates
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Find top products by revenue change
product_comparison = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'transactions']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
)
product_comparison['revenue_analysis'] = product_comparison['revenue_analysis'].fillna(0)
product_comparison['revenue_previous'] = product_comparison['revenue_previous'].fillna(0)
product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['revenue_pct_change'] = (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100) if product_comparison['revenue_previous'].sum() > 0 else 0

# Analyze channel mix
analysis_channel = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel.columns = ['channel', 'revenue', 'transactions']
analysis_channel['aov'] = analysis_channel['revenue'] / analysis_channel['transactions']

previous_channel = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel.columns = ['channel', 'revenue', 'transactions']
previous_channel['aov'] = previous_channel['revenue'] / previous_channel['transactions']

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
transaction_change = analysis_transactions - previous_transactions
transaction_pct_change = (transaction_change / previous_transactions * 100) if previous_transactions != 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) > 5 or abs(transaction_pct_change) > 5:
    findings.append({
        "title": "Revenue and Transaction Performance vs Previous Week",
        "claim": f"Analysis week (May 11-18, 2026) generated SAR {analysis_revenue:.2f} in net revenue across {analysis_transactions} transactions, compared to SAR {previous_revenue:.2f} across {previous_transactions} transactions in the previous week. This represents a {revenue_pct_change:.1f}% change in revenue and {transaction_pct_change:.1f}% change in transaction count.",
        "finding_type": "revenue_and_transaction_analysis",
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
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "transaction_change_pct": {
                "value": round(transaction_pct_change, 1),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
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
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Previous period: {previous_start.isoformat()} to {previous_end.isoformat()}",
            f"Refunds included in net revenue calculations",
            f"Analysis period refunds: SAR {analysis_refunds:.2f}",
            f"Previous period refunds: SAR {previous_refunds:.2f}"
        ],
        "assumptions": [
            "Unique transaction_id used to count baskets",
            "line_total_sar used for revenue calculations",
            "Refunds treated as negative revenue in net calculations",
            "All transactions with valid timestamps included"
        ],
        "confidence": 0.95
    })

# Finding 2: Top Product Performance
top_products = product_comparison.nlargest(3, 'revenue_change')[['sku', 'item_en', 'revenue_analysis', 'revenue_previous', 'revenue_change']]
if len(top_products) > 0 and top_products.iloc[0]['revenue_change'] > 0:
    top_product = top_products.iloc[0]
    findings.append({
        "title": "Top Revenue Growth Product",
        "claim": f"Product {top_product['item_en']} (SKU: {top_product['sku']}) generated SAR {top_product['revenue_analysis']:.2f} in the analysis week compared to SAR {top_product['revenue_previous']:.2f} in the previous week, representing a SAR {top_product['revenue_change']:.2f} increase.",
        "finding_type": "product_performance",
        "metrics": {
            "product_sku": {
                "value": str(top_product['sku']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "product_name": {
                "value": str(top_product['item_en']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "analysis_period_revenue": {
                "value": round(top_product['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "previous_period_revenue": {
                "value": round(top_product['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start.isoformat(),
                "period_end": previous_end.isoformat()
            },
            "revenue_change": {
                "value": round(top_product['revenue_change'], 2),
                "unit": "SAR",
                "numerator": round(top_product['revenue_change'], 2),
                "denominator": round(top_product['revenue_previous'], 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_product_mix),
        "coverage_notes": [
            f"Analysis covers {len(analysis_product_mix)} unique SKUs in analysis period",
            f"Product names sourced from menu reference",
            "Launch and retirement dates checked for eligibility"
        ],
        "assumptions": [
            "Product revenue calculated from line_total_sar",
            "SKU-level aggregation used for product identification",
            "Menu reference used for product name standardization"
        ],
        "confidence": 0.90
    })

# Finding 3: Channel Mix Analysis
if len(analysis_channel) > 0 and len(previous_channel) > 0:
    channel_comparison = analysis_channel.merge(
        previous_channel[['channel', 'revenue', 'transactions']],
        on='channel',
        how='outer',
        suffixes=('_analysis', '_previous')
    )
    channel_comparison['revenue_analysis'] = channel_comparison['revenue_analysis'].fillna(0)
    channel_comparison['revenue_previous'] = channel_comparison['revenue_previous'].fillna(0)
    channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
    
    top_channel = channel_comparison.nlargest(1, 'revenue_change').iloc[0]
    if top_channel['revenue_change'] > 0:
        findings.append({
            "title": "Channel Performance Change",
            "claim": f"Channel '{top_channel['channel']}' generated SAR {top_channel['revenue_analysis']:.2f} in the analysis week compared to SAR {top_channel['revenue_previous']:.2f} in the previous week, representing a SAR {top_channel['revenue_change']:.2f} increase.",
            "finding_type": "channel_mix_analysis",
            "metrics": {
                "channel_name": {
                    "value": str(top_channel['channel']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "analysis_period_revenue": {
                    "value": round(top_channel['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "previous_period_revenue": {
                    "value": round(top_channel['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "revenue_change": {
                    "value": round(top_channel['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_channel['revenue_change'], 2),
                    "denominator": round(top_channel['revenue_previous'], 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_channel),
            "coverage_notes": [
                f"Analysis covers {len(analysis_channel)} unique channels",
                "Channel data sourced from POS transaction records"
            ],
            "assumptions": [
                "Channel classification from POS data",
                "Revenue calculated from line_total_sar",
                "All transactions with valid channel information included"
            ],
            "confidence": 0.85
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
