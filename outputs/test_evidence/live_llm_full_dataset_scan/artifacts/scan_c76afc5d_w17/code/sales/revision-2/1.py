import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load POS and Menu data
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_period = {
    "start": "2026-05-04T00:00:00+03:00",
    "end": "2026-05-11T00:00:00+03:00"
}
previous_period = {
    "start": "2026-04-27T00:00:00+03:00",
    "end": "2026-05-04T00:00:00+03:00"
}
trailing_baseline_periods = [
    {
        "start": "2026-04-27T00:00:00+03:00",
        "end": "2026-05-04T00:00:00+03:00"
    },
    {
        "start": "2026-04-20T00:00:00+03:00",
        "end": "2026-04-27T00:00:00+03:00"
    },
    {
        "start": "2026-04-13T00:00:00+03:00",
        "end": "2026-04-20T00:00:00+03:00"
    },
    {
        "start": "2026-04-06T00:00:00+03:00",
        "end": "2026-04-13T00:00:00+03:00"
    }
]

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter data by period
def filter_by_period(df, period_start, period_end):
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Filter data for each period
analysis_data = filter_by_period(pos_df, analysis_period['start'], analysis_period['end'])
previous_data = filter_by_period(pos_df, previous_period['start'], previous_period['end'])

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0

# Calculate metrics for previous period
previous_valid_txns = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate trailing baseline (average of 4 weeks)
trailing_metrics = []
for period in trailing_baseline_periods:
    period_data = filter_by_period(pos_df, period['start'], period['end'])
    period_txns = period_data['transaction_id'].nunique()
    period_revenue = period_data['line_total_sar'].sum()
    period_aov = period_revenue / period_txns if period_txns > 0 else 0
    trailing_metrics.append({
        'txns': period_txns,
        'revenue': period_revenue,
        'aov': period_aov
    })

trailing_avg_txns = np.mean([m['txns'] for m in trailing_metrics])
trailing_avg_revenue = np.mean([m['revenue'] for m in trailing_metrics])
trailing_avg_aov = np.mean([m['aov'] for m in trailing_metrics])

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

txn_change = analysis_valid_txns - previous_valid_txns
txn_change_pct = (txn_change / previous_valid_txns * 100) if previous_valid_txns != 0 else 0

# Analyze refunds
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Analyze by category
analysis_by_category = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'txns']

previous_by_category = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'txns']

# Merge category data
category_comparison = analysis_by_category.merge(
    previous_by_category,
    on='category',
    suffixes=('_analysis', '_previous')
)
category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_change_pct'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100)

# Analyze by channel
analysis_by_channel = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'txns']

previous_by_channel = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'txns']

# Merge channel data
channel_comparison = analysis_by_channel.merge(
    previous_by_channel,
    on='channel',
    suffixes=('_analysis', '_previous')
)
channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
channel_comparison['revenue_change_pct'] = (channel_comparison['revenue_change'] / channel_comparison['revenue_previous'] * 100)

# Analyze product performance
analysis_by_sku = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique',
    'item_name_en': 'first',
    'category': 'first'
}).reset_index()
analysis_by_sku.columns = ['sku', 'revenue', 'quantity', 'txns', 'item_name', 'category']

previous_by_sku = previous_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique',
    'item_name_en': 'first',
    'category': 'first'
}).reset_index()
previous_by_sku.columns = ['sku', 'revenue', 'quantity', 'txns', 'item_name', 'category']

# Merge SKU data
sku_comparison = analysis_by_sku.merge(
    previous_by_sku,
    on='sku',
    suffixes=('_analysis', '_previous'),
    how='outer'
)
sku_comparison['revenue_change'] = sku_comparison['revenue_analysis'].fillna(0) - sku_comparison['revenue_previous'].fillna(0)
sku_comparison['revenue_change_pct'] = (sku_comparison['revenue_change'] / sku_comparison['revenue_previous'].fillna(1) * 100)

# Check for products with significant changes
significant_changes = sku_comparison[
    (sku_comparison['revenue_change'].abs() > 100) & 
    (sku_comparison['revenue_previous'].notna())
].sort_values('revenue_change', ascending=False)

# Prepare findings
findings = []

# Finding 1: Revenue change
if abs(revenue_change_pct) > 0.5:  # Only report if meaningful change
    findings.append({
        "title": "Weekly Revenue Change",
        "claim": f"Net revenue decreased by {abs(revenue_change):.2f} SAR, representing a {revenue_change_pct:.2f}% change from the previous week.",
        "finding_type": "revenue_change",
        "metrics": {
            "net_revenue_analysis": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "net_revenue_previous": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "revenue_change_absolute": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": analysis_period['end']
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": previous_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
            f"Previous period: {previous_period['start']} to {previous_period['end']}",
            f"Analysis transactions: {analysis_valid_txns}",
            f"Previous transactions: {previous_valid_txns}",
            f"Refunds included as negative line items in net revenue"
        ],
        "assumptions": [
            "line_total_sar represents net revenue per line item including refunds as negative values",
            "transaction_id uniqueness defines basket count",
            "Periods are non-overlapping and adjacent"
        ],
        "confidence": 0.95
    })

# Finding 2: AOV change
if abs(aov_change_pct) > 0.5:  # Only report if meaningful change
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value increased by {aov_change:.2f} SAR, representing a {aov_change_pct:.2f}% change from the previous week.",
        "finding_type": "aov_change",
        "metrics": {
            "aov_analysis": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "aov_previous": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "aov_change_absolute": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": analysis_period['end']
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": previous_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
            f"Previous period: {previous_period['start']} to {previous_period['end']}",
            f"Analysis transactions: {analysis_valid_txns}",
            f"Previous transactions: {previous_valid_txns}"
        ],
        "assumptions": [
            "AOV calculated as total net revenue divided by unique transaction_id count",
            "Refunds included as negative values in revenue calculation",
            "Each transaction_id represents one basket"
        ],
        "confidence": 0.95
    })

# Finding 3: Category mix change
if len(category_comparison) > 0:
    top_category_change = category_comparison.loc[category_comparison['revenue_change'].abs().idxmax()]
    if abs(top_category_change['revenue_change_pct']) > 5:  # Only report significant category changes
        findings.append({
            "title": "Category Mix Shift",
            "claim": f"The {top_category_change['category']} category revenue changed by {top_category_change['revenue_change']:.2f} SAR, representing a {top_category_change['revenue_change_pct']:.2f}% change from the previous week.",
            "finding_type": "category_mix_change",
            "metrics": {
                "category_revenue_analysis": {
                    "value": round(top_category_change['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "category_revenue_previous": {
                    "value": round(top_category_change['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": previous_period['end']
                },
                "category_revenue_change": {
                    "value": round(top_category_change['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category_change['revenue_change'], 2),
                    "denominator": None,
                    "period_start": previous_period['start'],
                    "period_end": analysis_period['end']
                },
                "category_revenue_change_pct": {
                    "value": round(top_category_change['revenue_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_category_change['revenue_change'], 2),
                    "denominator": round(top_category_change['revenue_previous'], 2),
                    "period_start": previous_period['start'],
                    "period_end": analysis_period['end']
                }
            },
            "source_names": ["pos"],
            "sample_size": int(top_category_change['txns_analysis']),
            "coverage_notes": [
                f"Category: {top_category_change['category']}",
                f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
                f"Previous period: {previous_period['start']} to {previous_period['end']}",
                f"Analysis transactions in category: {int(top_category_change['txns_analysis'])}",
                f"Previous transactions in category: {int(top_category_change['txns_previous'])}"
            ],
            "assumptions": [
                "Category assignment from POS data",
                "Revenue includes refunds as negative values",
                "Transaction count based on unique transaction_id per category"
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
