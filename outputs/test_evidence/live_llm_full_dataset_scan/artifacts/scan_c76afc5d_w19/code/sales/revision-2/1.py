import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods from context
analysis_period = {
    "start": "2026-05-18T00:00:00+03:00",
    "end": "2026-05-25T00:00:00+03:00"
}
previous_period = {
    "start": "2026-05-11T00:00:00+03:00",
    "end": "2026-05-18T00:00:00+03:00"
}
trailing_baseline_periods = [
    {"start": "2026-05-11T00:00:00+03:00", "end": "2026-05-18T00:00:00+03:00"},
    {"start": "2026-05-04T00:00:00+03:00", "end": "2026-05-11T00:00:00+03:00"},
    {"start": "2026-04-27T00:00:00+03:00", "end": "2026-05-04T00:00:00+03:00"},
    {"start": "2026-04-20T00:00:00+03:00", "end": "2026-04-27T00:00:00+03:00"}
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter by period
def filter_by_period(df, period_start, period_end):
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Helper function to calculate metrics for a period
def calculate_period_metrics(df, period_start, period_end):
    period_df = filter_by_period(df, period_start, period_end)
    
    # Exclude refunds for transaction count
    valid_transactions = period_df[period_df['is_refund'] == False]['transaction_id'].nunique()
    
    # Net revenue includes refunds (they are negative)
    net_revenue = period_df['line_total_sar'].sum()
    
    # AOV: net revenue / valid transactions (excluding refund rows from denominator)
    aov = net_revenue / valid_transactions if valid_transactions > 0 else 0
    
    # Total line items (including refunds)
    total_line_items = len(period_df)
    
    return {
        'valid_transactions': valid_transactions,
        'net_revenue': net_revenue,
        'aov': aov,
        'total_line_items': total_line_items,
        'period_df': period_df
    }

# Calculate metrics for analysis period
analysis_metrics = calculate_period_metrics(pos_df, analysis_period['start'], analysis_period['end'])

# Calculate metrics for previous period
previous_metrics = calculate_period_metrics(pos_df, previous_period['start'], previous_period['end'])

# Calculate metrics for trailing baseline (average of 4 weeks)
trailing_metrics_list = []
for period in trailing_baseline_periods:
    metrics = calculate_period_metrics(pos_df, period['start'], period['end'])
    trailing_metrics_list.append(metrics)

trailing_avg_transactions = np.mean([m['valid_transactions'] for m in trailing_metrics_list])
trailing_avg_revenue = np.mean([m['net_revenue'] for m in trailing_metrics_list])
trailing_avg_aov = np.mean([m['aov'] for m in trailing_metrics_list])

# Finding 1: Revenue change analysis period vs previous period
revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
revenue_pct_change = (revenue_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] != 0 else 0

# Finding 2: Transaction count change
transaction_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
transaction_pct_change = (transaction_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] > 0 else 0

# Finding 3: AOV change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

# Finding 4: Category mix analysis
analysis_period_df = analysis_metrics['period_df']
previous_period_df = previous_metrics['period_df']

# Exclude refunds for category analysis
analysis_no_refunds = analysis_period_df[analysis_period_df['is_refund'] == False]
previous_no_refunds = previous_period_df[previous_period_df['is_refund'] == False]

analysis_category_revenue = analysis_no_refunds.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_no_refunds.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Finding 5: Channel mix analysis
analysis_channel_revenue = analysis_no_refunds.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_no_refunds.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

# Finding 6: Product performance - top products
analysis_product_revenue = analysis_no_refunds.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_revenue.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']
analysis_product_revenue = analysis_product_revenue.sort_values('revenue', ascending=False)

previous_product_revenue = previous_no_refunds.groupby(['sku', 'item_name_en']).agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_revenue.columns = ['sku', 'item_name_en', 'revenue', 'quantity', 'transactions']
previous_product_revenue = previous_product_revenue.sort_values('revenue', ascending=False)

# Check for product launch dates
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')

# Findings construction
findings = []

# Finding 1: Revenue Performance
if previous_metrics['net_revenue'] != 0:
    findings.append({
        "title": "Net Revenue Change: Analysis Period vs Previous Week",
        "claim": f"Net revenue in the analysis period (2026-05-18 to 2026-05-25) was SAR {analysis_metrics['net_revenue']:.2f}, compared to SAR {previous_metrics['net_revenue']:.2f} in the previous period (2026-05-11 to 2026-05-18), representing a change of SAR {revenue_change:.2f} ({revenue_pct_change:.2f}%).",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_net_revenue": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_period_net_revenue": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": 1,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_metrics['net_revenue'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_line_items'],
        "coverage_notes": [
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Previous period: 2026-05-11 to 2026-05-18",
            "Net revenue includes refunds (negative values)",
            "Refund rows are included in line_total_sar calculation"
        ],
        "assumptions": [
            "Valid transactions are counted using unique transaction_id values",
            "Refund rows (is_refund=True) are included in net revenue calculations",
            "line_total_sar represents realized net revenue per line item",
            "Timestamp filtering uses UTC+3 timezone as provided"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction Count Change
if previous_metrics['valid_transactions'] > 0:
    findings.append({
        "title": "Valid Transaction Count Change: Analysis Period vs Previous Week",
        "claim": f"Valid transaction count in the analysis period (2026-05-18 to 2026-05-25) was {analysis_metrics['valid_transactions']}, compared to {previous_metrics['valid_transactions']} in the previous period (2026-05-11 to 2026-05-18), representing a change of {transaction_change} transactions ({transaction_pct_change:.2f}%).",
        "finding_type": "transaction_count_change",
        "metrics": {
            "analysis_period_valid_transactions": {
                "value": analysis_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_period_valid_transactions": {
                "value": previous_metrics['valid_transactions'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "transaction_count_change": {
                "value": transaction_change,
                "unit": "transactions",
                "numerator": transaction_change,
                "denominator": 1,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "transaction_count_change_pct": {
                "value": round(transaction_pct_change, 2),
                "unit": "%",
                "numerator": transaction_change,
                "denominator": previous_metrics['valid_transactions'],
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_line_items'],
        "coverage_notes": [
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Previous period: 2026-05-11 to 2026-05-18",
            "Valid transactions exclude refund rows (is_refund=False)",
            "Counted using unique transaction_id values"
        ],
        "assumptions": [
            "Valid transactions are counted using unique transaction_id values where is_refund=False",
            "Refund rows are excluded from transaction count",
            "Each transaction_id represents one basket/order",
            "Timestamp filtering uses UTC+3 timezone as provided"
        ],
        "confidence": 0.95
    })

# Finding 3: Average Order Value Change
if previous_metrics['aov'] > 0:
    findings.append({
        "title": "Average Order Value Change: Analysis Period vs Previous Week",
        "claim": f"Average Order Value (AOV) in the analysis period (2026-05-18 to 2026-05-25) was SAR {analysis_metrics['aov']:.2f}, compared to SAR {previous_metrics['aov']:.2f} in the previous period (2026-05-11 to 2026-05-18), representing a change of SAR {aov_change:.2f} ({aov_pct_change:.2f}%). AOV is calculated as net revenue (including refunds) divided by valid transaction count (excluding refund rows).",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['net_revenue'], 2),
                "denominator": analysis_metrics['valid_transactions'],
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "previous_period_aov": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['net_revenue'], 2),
                "denominator": previous_metrics['valid_transactions'],
                "period_start": previous_period['start'],
                "period_end": previous_period['end']
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": 1,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_metrics['aov'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['total_line_items'],
        "coverage_notes": [
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Previous period: 2026-05-11 to 2026-05-18",
            "Net revenue includes refunds (negative values)",
            "Valid transactions exclude refund rows from denominator",
            "AOV calculation: net revenue / valid transaction count"
        ],
        "assumptions": [
            "Valid transactions are counted using unique transaction_id values where is_refund=False",
            "Net revenue includes refund rows (they are negative values in line_total_sar)",
            "AOV is intentionally calculated as net revenue (including refunds) divided by valid transaction count (excluding refund rows) to reflect the average value per customer transaction",
            "Timestamp filtering uses UTC+3 timezone as provided"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Results written to {output_path}")
