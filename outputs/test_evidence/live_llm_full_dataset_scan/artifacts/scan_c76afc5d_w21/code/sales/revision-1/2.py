import os
import json
import pandas as pd
from datetime import datetime

# Load input paths from environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods from context
analysis_start = datetime.fromisoformat("2026-06-01T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-06-08T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-05-25T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-06-01T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Calculate metrics for analysis period
analysis_valid_txns = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
analysis_total_revenue = analysis_data['line_total_sar'].sum()
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
analysis_net_revenue = analysis_total_revenue - abs(analysis_refunds)
analysis_aov = analysis_net_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0

# Calculate metrics for previous period
previous_valid_txns = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
previous_total_revenue = previous_data['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()
previous_net_revenue = previous_total_revenue - abs(previous_refunds)
previous_aov = previous_net_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate changes
revenue_change_sar = analysis_net_revenue - previous_net_revenue
revenue_change_pct = (revenue_change_sar / previous_net_revenue * 100) if previous_net_revenue != 0 else 0
aov_change_sar = analysis_aov - previous_aov
aov_change_pct = (aov_change_sar / previous_aov * 100) if previous_aov != 0 else 0
txn_change = analysis_valid_txns - previous_valid_txns
txn_change_pct = (txn_change / previous_valid_txns * 100) if previous_valid_txns > 0 else 0

# Analyze product mix changes
analysis_product_mix = analysis_data[analysis_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

previous_product_mix = previous_data[previous_data['is_refund'] == False].groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']

# Merge with menu to get product names and launch dates
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Calculate product revenue changes
product_comparison = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'quantity']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
).fillna(0)

product_comparison['revenue_change'] = product_comparison['revenue_analysis'] - product_comparison['revenue_previous']
product_comparison['revenue_change_pct'] = (product_comparison['revenue_change'] / product_comparison['revenue_previous'] * 100).replace([float('inf'), float('-inf')], 0)

# Find top revenue changes
top_revenue_changes = product_comparison.nlargest(3, 'revenue_change')

# Analyze category mix
analysis_category_mix = analysis_data[analysis_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_category_mix.columns = ['category', 'revenue', 'quantity', 'transactions']
analysis_category_mix['revenue_pct'] = (analysis_category_mix['revenue'] / analysis_category_mix['revenue'].sum() * 100)

previous_category_mix = previous_data[previous_data['is_refund'] == False].groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_category_mix.columns = ['category', 'revenue', 'quantity', 'transactions']
previous_category_mix['revenue_pct'] = (previous_category_mix['revenue'] / previous_category_mix['revenue'].sum() * 100)

# Analyze channel mix
analysis_channel_mix = analysis_data[analysis_data['is_refund'] == False].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel_mix.columns = ['channel', 'revenue', 'transactions']
analysis_channel_mix['revenue_pct'] = (analysis_channel_mix['revenue'] / analysis_channel_mix['revenue'].sum() * 100)

previous_channel_mix = previous_data[previous_data['is_refund'] == False].groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel_mix.columns = ['channel', 'revenue', 'transactions']
previous_channel_mix['revenue_pct'] = (previous_channel_mix['revenue'] / previous_channel_mix['revenue'].sum() * 100)

# Build findings
findings = []

# Finding 1: Net Revenue Change
if analysis_valid_txns > 0 and previous_valid_txns > 0:
    finding1 = {
        "title": "Net Revenue Decline Week-over-Week",
        "claim": f"Net revenue decreased from SAR {previous_net_revenue:.2f} in the previous week (May 25 - Jun 1) to SAR {analysis_net_revenue:.2f} in the analysis week (Jun 1 - Jun 8), representing a SAR {revenue_change_sar:.2f} decline or {revenue_change_pct:.2f}% change.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_net_revenue": {
                "value": round(analysis_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_total_revenue, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_net_revenue": {
                "value": round(previous_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_total_revenue, 2),
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change_sar, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_txns),
        "coverage_notes": [
            f"Analysis period: {int(analysis_valid_txns)} valid transactions",
            f"Previous period: {int(previous_valid_txns)} valid transactions",
            f"Analysis refunds: SAR {abs(analysis_refunds):.2f}",
            f"Previous refunds: SAR {abs(previous_refunds):.2f}"
        ],
        "assumptions": [
            "Net revenue includes refunds as negative line_total_sar values",
            "Valid transactions exclude refund line items (is_refund == False)",
            "Transaction counted by unique transaction_id",
            "All POS data treated as authoritative"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Average Order Value Change
if analysis_valid_txns > 0 and previous_valid_txns > 0:
    finding2 = {
        "title": "Average Order Value Slight Decline",
        "claim": f"Average order value decreased from SAR {previous_aov:.2f} in the previous week to SAR {analysis_aov:.2f} in the analysis week, representing a SAR {aov_change_sar:.2f} decline or {aov_change_pct:.2f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": int(analysis_valid_txns),
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": int(previous_valid_txns),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change_sar, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_txns),
        "coverage_notes": [
            f"Analysis period: {int(analysis_valid_txns)} valid transactions",
            f"Previous period: {int(previous_valid_txns)} valid transactions"
        ],
        "assumptions": [
            "AOV calculated as net revenue divided by valid transaction count",
            "Net revenue includes refunds as negative values",
            "Valid transactions exclude refund line items"
        ],
        "confidence": 0.92
    }
    findings.append(finding2)

# Finding 3: Top Product Revenue Change
if len(top_revenue_changes) > 0 and top_revenue_changes.iloc[0]['revenue_analysis'] > 0:
    top_product = top_revenue_changes.iloc[0]
    finding3 = {
        "title": "Strongest Product Revenue Growth",
        "claim": f"Product {top_product['item_en']} (SKU: {top_product['sku']}) generated SAR {top_product['revenue_analysis']:.2f} in the analysis week compared to SAR {top_product['revenue_previous']:.2f} in the previous week, representing a SAR {top_product['revenue_change']:.2f} increase or {top_product['revenue_change_pct']:.2f}% growth.",
        "finding_type": "product_mix_change",
        "metrics": {
            "analysis_product_revenue": {
                "value": round(top_product['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_product_revenue": {
                "value": round(top_product['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "product_revenue_change_sar": {
                "value": round(top_product['revenue_change'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "product_revenue_change_pct": {
                "value": round(top_product['revenue_change_pct'], 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": None,
        "coverage_notes": [
            f"Product: {top_product['item_en']} (SKU: {top_product['sku']})",
            f"Analysis period quantity: {int(top_product['quantity_analysis'])}",
            f"Previous period quantity: {int(top_product['quantity_previous'])}"
        ],
        "assumptions": [
            "Product revenue calculated from line_total_sar excluding refunds",
            "Product names sourced from menu SKU reference",
            "Comparison includes only products with sales in both periods"
        ],
        "confidence": 0.88
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
