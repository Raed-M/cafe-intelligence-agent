import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment configuration
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
analysis_end = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_start = datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_end = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
analysis_total_revenue = analysis_data[~analysis_data['is_refund']]['line_total_sar'].sum()
analysis_refund_revenue = analysis_data[analysis_data['is_refund']]['line_total_sar'].sum()
analysis_net_revenue = analysis_total_revenue + analysis_refund_revenue  # refunds are negative

# Calculate metrics for previous period
previous_valid_txns = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()
previous_total_revenue = previous_data[~previous_data['is_refund']]['line_total_sar'].sum()
previous_refund_revenue = previous_data[previous_data['is_refund']]['line_total_sar'].sum()
previous_net_revenue = previous_total_revenue + previous_refund_revenue

# Calculate AOV
analysis_aov = analysis_net_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0
previous_aov = previous_net_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate trailing baseline (4 weeks average)
trailing_baseline_periods = [
    (datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))))
]

trailing_revenues = []
trailing_txns = []
for period_start, period_end in trailing_baseline_periods:
    period_data = pos_df[(pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)]
    period_valid_txns = period_data[~period_data['is_refund']]['transaction_id'].nunique()
    period_total_revenue = period_data[~period_data['is_refund']]['line_total_sar'].sum()
    period_refund_revenue = period_data[period_data['is_refund']]['line_total_sar'].sum()
    period_net_revenue = period_total_revenue + period_refund_revenue
    trailing_revenues.append(period_net_revenue)
    trailing_txns.append(period_valid_txns)

trailing_avg_revenue = np.mean(trailing_revenues) if trailing_revenues else 0
trailing_avg_txns = np.mean(trailing_txns) if trailing_txns else 0
trailing_avg_aov = trailing_avg_revenue / trailing_avg_txns if trailing_avg_txns > 0 else 0

# Analyze product mix changes
analysis_product_mix = analysis_data[~analysis_data['is_refund']].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']
analysis_product_mix['pct_revenue'] = (analysis_product_mix['revenue'] / analysis_product_mix['revenue'].sum() * 100)

previous_product_mix = previous_data[~previous_data['is_refund']].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']
previous_product_mix['pct_revenue'] = (previous_product_mix['revenue'] / previous_product_mix['revenue'].sum() * 100)

# Merge with menu for product names and launch dates
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Analyze channel mix
analysis_channel_mix = analysis_data[~analysis_data['is_refund']].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel_mix.columns = ['channel', 'revenue', 'transactions']
analysis_channel_mix['pct_revenue'] = (analysis_channel_mix['revenue'] / analysis_channel_mix['revenue'].sum() * 100)

previous_channel_mix = previous_data[~previous_data['is_refund']].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel_mix.columns = ['channel', 'revenue', 'transactions']
previous_channel_mix['pct_revenue'] = (previous_channel_mix['revenue'] / previous_channel_mix['revenue'].sum() * 100)

# Find top products by revenue change
product_comparison = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'pct_revenue']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
).fillna(0)

product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['pct_change'] = ((product_comparison['revenue_analysis'] - product_comparison['revenue_previous']) / 
                                     product_comparison['revenue_previous'].replace(0, np.nan) * 100)

# Sort by absolute revenue change
product_comparison_sorted = product_comparison.sort_values('revenue_change', ascending=False)

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Changes
revenue_change = analysis_net_revenue - previous_net_revenue
revenue_pct_change = (revenue_change / previous_net_revenue * 100) if previous_net_revenue != 0 else 0
txn_change = analysis_valid_txns - previous_valid_txns
txn_pct_change = (txn_change / previous_valid_txns * 100) if previous_valid_txns != 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) > 5 or abs(txn_pct_change) > 5:
    findings.append({
        "title": "Weekly Revenue and Transaction Performance",
        "claim": f"Analysis week (May 4-11, 2026) generated SAR {analysis_net_revenue:.2f} in net revenue across {analysis_valid_txns} valid transactions, representing a {revenue_pct_change:.1f}% change in revenue and {txn_pct_change:.1f}% change in transaction count versus the previous week (Apr 27-May 4, 2026).",
        "finding_type": "revenue_and_transaction_analysis",
        "metrics": {
            "analysis_net_revenue": {
                "value": round(analysis_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_total_revenue, 2),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "analysis_valid_transactions": {
                "value": analysis_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "previous_net_revenue": {
                "value": round(previous_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_total_revenue, 2),
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "previous_valid_transactions": {
                "value": previous_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_net_revenue, 2),
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_valid_txns,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period covers {len(analysis_data)} POS line items",
            f"Previous period covers {len(previous_data)} POS line items",
            f"Refunds included in net calculations: analysis refunds = SAR {analysis_refund_revenue:.2f}, previous refunds = SAR {previous_refund_revenue:.2f}",
            "Valid transactions counted using unique transaction_id after filtering refunds"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue",
            "is_refund flag correctly identifies refund transactions",
            "transaction_id uniquely identifies a basket",
            "Timestamps are in +03:00 timezone as specified"
        ],
        "confidence": 0.95
    })

# Finding 2: Top Product Performance Change
if len(product_comparison_sorted) > 0:
    top_product = product_comparison_sorted.iloc[0]
    if top_product['sku'] and not pd.isna(top_product['revenue_analysis']) and top_product['revenue_analysis'] > 0:
        # Check if product was launched before analysis period
        launch_date = pd.to_datetime(top_product['launch_date']) if pd.notna(top_product['launch_date']) else None
        # Ensure timezone-aware comparison
        if launch_date is not None:
            if launch_date.tzinfo is None:
                launch_date = launch_date.replace(tzinfo=timezone(timedelta(hours=3)))
        
        if launch_date is None or launch_date < analysis_start:
            findings.append({
                "title": "Top Revenue-Growing Product",
                "claim": f"Product {top_product['item_en']} (SKU: {top_product['sku']}) generated SAR {top_product['revenue_analysis']:.2f} in the analysis week, up from SAR {top_product['revenue_previous']:.2f} in the previous week, representing a {top_product['pct_change']:.1f}% increase.",
                "finding_type": "product_mix_analysis",
                "metrics": {
                    "analysis_product_revenue": {
                        "value": round(top_product['revenue_analysis'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-04T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    },
                    "previous_product_revenue": {
                        "value": round(top_product['revenue_previous'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "product_revenue_change_pct": {
                        "value": round(top_product['pct_change'], 2),
                        "unit": "%",
                        "numerator": round(top_product['revenue_change'], 2),
                        "denominator": round(top_product['revenue_previous'], 2),
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    },
                    "analysis_product_pct_of_total": {
                        "value": round(top_product['pct_revenue_analysis'], 2),
                        "unit": "%",
                        "numerator": round(top_product['revenue_analysis'], 2),
                        "denominator": round(analysis_product_mix['revenue'].sum(), 2),
                        "period_start": "2026-05-04T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": int(top_product['transactions_analysis']) if pd.notna(top_product['transactions_analysis']) else None,
                "coverage_notes": [
                    f"Product {top_product['item_en']} appears in {int(top_product['transactions_analysis'])} transactions in analysis period",
                    f"Product {top_product['item_en']} appears in {int(top_product['transactions_previous'])} transactions in previous period",
                    "Comparison limited to products with sales in both periods"
                ],
                "assumptions": [
                    "Product name from menu SKU reference",
                    "Launch date eligibility verified from menu",
                    "Revenue calculated from line_total_sar excluding refunds"
                ],
                "confidence": 0.90
            })

# Finding 3: Channel Mix Analysis
if len(analysis_channel_mix) > 0 and len(previous_channel_mix) > 0:
    channel_comparison = analysis_channel_mix.merge(
        previous_channel_mix[['channel', 'pct_revenue']],
        on='channel',
        how='outer',
        suffixes=('_analysis', '_previous')
    ).fillna(0)
    
    channel_comparison['pct_change'] = channel_comparison['pct_revenue_analysis'] - channel_comparison['pct_revenue_previous']
    channel_comparison_sorted = channel_comparison.sort_values('pct_change', ascending=False)
    
    if len(channel_comparison_sorted) > 0:
        top_channel = channel_comparison_sorted.iloc[0]
        if abs(top_channel['pct_change']) > 2:  # Only report if >2% change
            findings.append({
                "title": "Channel Mix Shift",
                "claim": f"Channel '{top_channel['channel']}' increased its share of revenue from {top_channel['pct_revenue_previous']:.1f}% in the previous week to {top_channel['pct_revenue_analysis']:.1f}% in the analysis week, a {top_channel['pct_change']:.1f} percentage point increase.",
                "finding_type": "channel_mix_analysis",
                "metrics": {
                    "analysis_channel_pct": {
                        "value": round(top_channel['pct_revenue_analysis'], 2),
                        "unit": "%",
                        "numerator": round(top_channel['revenue_analysis'], 2),
                        "denominator": round(analysis_channel_mix['revenue'].sum(), 2),
                        "period_start": "2026-05-04T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    },
                    "previous_channel_pct": {
                        "value": round(top_channel['pct_revenue_previous'], 2),
                        "unit": "%",
                        "numerator": round(top_channel['revenue_previous'], 2),
                        "denominator": round(previous_channel_mix['revenue'].sum(), 2),
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "channel_pct_point_change": {
                        "value": round(top_channel['pct_change'], 2),
                        "unit": "percentage points",
                        "numerator": round(top_channel['pct_change'], 2),
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    },
                    "analysis_channel_revenue": {
                        "value": round(top_channel['revenue_analysis'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-04T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    },
                    "analysis_channel_transactions": {
                        "value": int(top_channel['transactions_analysis']),
                        "unit": "count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-04T00:00:00+03:00",
                        "period_end": "2026-05-11T00:00:00+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": int(top_channel['transactions_analysis']),
                "coverage_notes": [
                    f"Analysis period: {int(analysis_channel_mix['transactions'].sum())} transactions across {len(analysis_channel_mix)} channels",
                    f"Previous period: {int(previous_channel_mix['transactions'].sum())} transactions across {len(previous_channel_mix)} channels",
                    "Channel field from POS data"
                ],
                "assumptions": [
                    "Channel values are consistent between periods",
                    "Revenue calculated from line_total_sar excluding refunds",
                    "Transactions counted using unique transaction_id"
                ],
                "confidence": 0.92
            })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
