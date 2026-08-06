import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_period = {
    "start": "2026-05-04",
    "end": "2026-05-11"
}
previous_period = {
    "start": "2026-04-27",
    "end": "2026-05-04"
}
trailing_baseline = [
    {"start": "2026-04-27", "end": "2026-05-04"},
    {"start": "2026-04-20", "end": "2026-04-27"},
    {"start": "2026-04-13", "end": "2026-04-20"},
    {"start": "2026-04-06", "end": "2026-04-13"}
]

# Convert calendar_date to datetime for filtering
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter data by periods
def filter_by_period(df, start_str, end_str):
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    return df[(df['calendar_date'] >= start) & (df['calendar_date'] < end)]

analysis_data = filter_by_period(pos_df, analysis_period['start'], analysis_period['end'])
previous_data = filter_by_period(pos_df, previous_period['start'], previous_period['end'])

# Calculate metrics for analysis period
def calculate_metrics(df, period_start, period_end):
    # Remove refunds for transaction count
    non_refund_df = df[df['is_refund'] == False]
    
    # Count unique transactions
    transaction_count = non_refund_df['transaction_id'].nunique()
    
    # Calculate total revenue (including refunds as negative)
    total_revenue = df['line_total_sar'].sum()
    
    # Calculate AOV (average order value) - using non-refund transactions
    if transaction_count > 0:
        aov = total_revenue / transaction_count
    else:
        aov = 0
    
    return {
        'transaction_count': transaction_count,
        'total_revenue': total_revenue,
        'aov': aov,
        'period_start': period_start,
        'period_end': period_end
    }

# Calculate for analysis period
analysis_metrics = calculate_metrics(analysis_data, analysis_period['start'], analysis_period['end'])

# Calculate for previous period
previous_metrics = calculate_metrics(previous_data, previous_period['start'], previous_period['end'])

# Calculate for trailing baseline (average)
trailing_metrics_list = []
for period in trailing_baseline:
    period_data = filter_by_period(pos_df, period['start'], period['end'])
    metrics = calculate_metrics(period_data, period['start'], period['end'])
    trailing_metrics_list.append(metrics)

# Average trailing baseline
avg_trailing_transaction_count = np.mean([m['transaction_count'] for m in trailing_metrics_list])
avg_trailing_revenue = np.mean([m['total_revenue'] for m in trailing_metrics_list])
avg_trailing_aov = np.mean([m['aov'] for m in trailing_metrics_list])

# Calculate changes
transaction_count_change = analysis_metrics['transaction_count'] - previous_metrics['transaction_count']
transaction_count_change_pct = (transaction_count_change / previous_metrics['transaction_count'] * 100) if previous_metrics['transaction_count'] > 0 else 0

revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_change_pct = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] > 0 else 0

aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_change_pct = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

# Analyze product mix changes
def get_product_mix(df):
    product_sales = df.groupby('sku').agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    product_sales.columns = ['sku', 'revenue', 'quantity', 'transactions']
    return product_sales

analysis_product_mix = get_product_mix(analysis_data)
previous_product_mix = get_product_mix(previous_data)

# Merge with menu for product names
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'category', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'category', 'launch_date', 'retire_date']], on='sku', how='left')

# Find top products by revenue change
merged_products = analysis_product_mix.merge(
    previous_product_mix[['sku', 'revenue', 'transactions']],
    on='sku',
    how='outer',
    suffixes=('_analysis', '_previous')
).fillna(0)

merged_products['revenue_change'] = merged_products['revenue_analysis'] - merged_products['revenue_previous']
merged_products['revenue_change_pct'] = (merged_products['revenue_change'] / merged_products['revenue_previous'] * 100) if merged_products['revenue_previous'].sum() > 0 else 0

# Analyze channel mix
def get_channel_mix(df):
    channel_sales = df.groupby('channel').agg({
        'line_total_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    channel_sales.columns = ['channel', 'revenue', 'transactions']
    return channel_sales

analysis_channel_mix = get_channel_mix(analysis_data)
previous_channel_mix = get_channel_mix(previous_data)

# Prepare findings
findings = []

# Finding 1: Transaction count change
if analysis_metrics['transaction_count'] != previous_metrics['transaction_count']:
    findings.append({
        "title": "Transaction Count Change",
        "claim": f"Transaction count in analysis period ({analysis_period['start']} to {analysis_period['end']}) was {analysis_metrics['transaction_count']}, compared to {previous_metrics['transaction_count']} in previous period ({previous_period['start']} to {previous_period['end']}), representing a {transaction_count_change_pct:.1f}% change.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_transaction_count": {
                "value": int(analysis_metrics['transaction_count']),
                "unit": "transactions",
                "numerator": int(analysis_metrics['transaction_count']),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_transaction_count": {
                "value": int(previous_metrics['transaction_count']),
                "unit": "transactions",
                "numerator": int(previous_metrics['transaction_count']),
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "transaction_count_change": {
                "value": int(transaction_count_change),
                "unit": "transactions",
                "numerator": int(transaction_count_change),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_count_change_pct": {
                "value": round(transaction_count_change_pct, 2),
                "unit": "%",
                "numerator": int(transaction_count_change),
                "denominator": int(previous_metrics['transaction_count']),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_metrics['transaction_count']),
        "coverage_notes": [
            "Analysis period: 2026-05-04 to 2026-05-11",
            "Previous period: 2026-04-27 to 2026-05-04",
            "Transactions counted using unique transaction_id after removing refunds"
        ],
        "assumptions": [
            "Refunds excluded from transaction count",
            "Each unique transaction_id represents one basket"
        ],
        "confidence": 0.95
    })

# Finding 2: Revenue change
if analysis_metrics['total_revenue'] != previous_metrics['total_revenue']:
    findings.append({
        "title": "Net Revenue Change",
        "claim": f"Net revenue in analysis period ({analysis_period['start']} to {analysis_period['end']}) was {analysis_metrics['total_revenue']:.2f} SAR, compared to {previous_metrics['total_revenue']:.2f} SAR in previous period ({previous_period['start']} to {previous_period['end']}), representing a {revenue_change_pct:.1f}% change.",
        "finding_type": "revenue",
        "metrics": {
            "analysis_revenue": {
                "value": round(analysis_metrics['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['total_revenue'], 2),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_revenue": {
                "value": round(previous_metrics['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['total_revenue'], 2),
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_metrics['total_revenue'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_metrics['transaction_count']),
        "coverage_notes": [
            "Analysis period: 2026-05-04 to 2026-05-11",
            "Previous period: 2026-04-27 to 2026-05-04",
            "Revenue includes refunds as negative values"
        ],
        "assumptions": [
            "line_total_sar represents net revenue per line item",
            "Refunds are included in net calculations"
        ],
        "confidence": 0.95
    })

# Finding 3: AOV change
if analysis_metrics['aov'] != previous_metrics['aov']:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period ({analysis_period['start']} to {analysis_period['end']}) was {analysis_metrics['aov']:.2f} SAR, compared to {previous_metrics['aov']:.2f} SAR in previous period ({previous_period['start']} to {previous_period['end']}), representing a {aov_change_pct:.1f}% change.",
        "finding_type": "aov",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['total_revenue'], 2),
                "denominator": int(analysis_metrics['transaction_count']),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['total_revenue'], 2),
                "denominator": int(previous_metrics['transaction_count']),
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_metrics['aov'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_metrics['transaction_count']),
        "coverage_notes": [
            "Analysis period: 2026-05-04 to 2026-05-11",
            "Previous period: 2026-04-27 to 2026-05-04",
            "AOV calculated as total revenue divided by unique transaction count"
        ],
        "assumptions": [
            "AOV includes refunds as negative values",
            "Each unique transaction_id represents one basket"
        ],
        "confidence": 0.95
    })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Output written to {output_path}")
