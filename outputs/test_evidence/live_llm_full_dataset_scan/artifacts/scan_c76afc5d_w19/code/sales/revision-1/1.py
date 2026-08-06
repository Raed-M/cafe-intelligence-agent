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

# Define periods (UTC+3)
analysis_period = {
    'start': pd.Timestamp('2026-05-18T00:00:00+03:00'),
    'end': pd.Timestamp('2026-05-25T00:00:00+03:00')
}
previous_period = {
    'start': pd.Timestamp('2026-05-11T00:00:00+03:00'),
    'end': pd.Timestamp('2026-05-18T00:00:00+03:00')
}
trailing_baselines = [
    {'start': pd.Timestamp('2026-05-11T00:00:00+03:00'), 'end': pd.Timestamp('2026-05-18T00:00:00+03:00')},
    {'start': pd.Timestamp('2026-05-04T00:00:00+03:00'), 'end': pd.Timestamp('2026-05-11T00:00:00+03:00')},
    {'start': pd.Timestamp('2026-04-27T00:00:00+03:00'), 'end': pd.Timestamp('2026-05-04T00:00:00+03:00')},
    {'start': pd.Timestamp('2026-04-20T00:00:00+03:00'), 'end': pd.Timestamp('2026-04-27T00:00:00+03:00')}
]

# Ensure timestamp is timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Helper function to filter by period
def filter_by_period(df, period):
    return df[(df['timestamp'] >= period['start']) & (df['timestamp'] < period['end'])]

# Helper function to calculate metrics for a period
def calculate_period_metrics(df, period_name):
    metrics = {}
    
    # Valid transactions (unique transaction_id, excluding refunds for basket count)
    valid_txns = df[~df['is_refund']]['transaction_id'].nunique()
    metrics['valid_transactions'] = valid_txns
    
    # Total revenue (net, including refunds)
    total_revenue = df['line_total_sar'].sum()
    metrics['total_revenue_sar'] = total_revenue
    
    # Average order value
    if valid_txns > 0:
        aov = total_revenue / valid_txns
        metrics['aov_sar'] = aov
    else:
        metrics['aov_sar'] = None
    
    # Refund impact
    refund_amount = df[df['is_refund']]['line_total_sar'].sum()
    metrics['refund_amount_sar'] = refund_amount
    
    # Channel mix
    channel_revenue = df.groupby('channel')['line_total_sar'].sum().to_dict()
    metrics['channel_mix'] = channel_revenue
    
    # Category mix
    category_revenue = df.groupby('category')['line_total_sar'].sum().to_dict()
    metrics['category_mix'] = category_revenue
    
    # Product performance (top 5 by revenue)
    product_revenue = df.groupby(['sku', 'item_name_en']).agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    product_revenue.columns = ['sku', 'item_name', 'revenue', 'quantity', 'transactions']
    product_revenue = product_revenue.sort_values('revenue', ascending=False)
    metrics['top_products'] = product_revenue.head(5).to_dict('records')
    
    return metrics

# Calculate metrics for each period
analysis_metrics = calculate_period_metrics(filter_by_period(pos_df, analysis_period), 'analysis')
previous_metrics = calculate_period_metrics(filter_by_period(pos_df, previous_period), 'previous')

# Calculate trailing baseline average
trailing_metrics_list = []
for baseline in trailing_baselines:
    trailing_metrics_list.append(calculate_period_metrics(filter_by_period(pos_df, baseline), 'trailing'))

# Compute trailing 4-week average
trailing_avg_revenue = np.mean([m['total_revenue_sar'] for m in trailing_metrics_list])
trailing_avg_txns = np.mean([m['valid_transactions'] for m in trailing_metrics_list])
trailing_avg_aov = np.mean([m['aov_sar'] for m in trailing_metrics_list if m['aov_sar'] is not None])

# Findings
findings = []

# Finding 1: Revenue change from previous week
if previous_metrics['total_revenue_sar'] > 0:
    revenue_change = analysis_metrics['total_revenue_sar'] - previous_metrics['total_revenue_sar']
    revenue_pct_change = (revenue_change / previous_metrics['total_revenue_sar']) * 100
    
    finding1 = {
        'title': 'Weekly Revenue Change',
        'claim': f"Total net revenue in analysis week (2026-05-18 to 2026-05-25) was {analysis_metrics['total_revenue_sar']:.2f} SAR, representing a {revenue_pct_change:.1f}% change from previous week ({previous_metrics['total_revenue_sar']:.2f} SAR).",
        'finding_type': 'revenue_trend',
        'metrics': {
            'analysis_week_revenue': {
                'value': round(analysis_metrics['total_revenue_sar'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'previous_week_revenue': {
                'value': round(previous_metrics['total_revenue_sar'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-11T00:00:00+03:00',
                'period_end': '2026-05-18T00:00:00+03:00'
            },
            'revenue_change_sar': {
                'value': round(revenue_change, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'revenue_pct_change': {
                'value': round(revenue_pct_change, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': len(filter_by_period(pos_df, analysis_period)),
        'coverage_notes': [
            'Analysis period: 2026-05-18 to 2026-05-25 (UTC+3)',
            'Previous period: 2026-05-11 to 2026-05-18 (UTC+3)',
            'Refunds included in net revenue calculation'
        ],
        'assumptions': [
            'line_total_sar represents realized net revenue including discounts and refunds',
            'Refund transactions marked with is_refund=True are included in totals'
        ],
        'confidence': 0.95
    }
    findings.append(finding1)

# Finding 2: Transaction count change
if previous_metrics['valid_transactions'] > 0:
    txn_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
    txn_pct_change = (txn_change / previous_metrics['valid_transactions']) * 100
    
    finding2 = {
        'title': 'Transaction Volume Change',
        'claim': f"Valid transaction count in analysis week was {analysis_metrics['valid_transactions']}, a {txn_pct_change:.1f}% change from {previous_metrics['valid_transactions']} in the previous week.",
        'finding_type': 'transaction_volume',
        'metrics': {
            'analysis_week_transactions': {
                'value': analysis_metrics['valid_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'previous_week_transactions': {
                'value': previous_metrics['valid_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-11T00:00:00+03:00',
                'period_end': '2026-05-18T00:00:00+03:00'
            },
            'transaction_change': {
                'value': txn_change,
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'transaction_pct_change': {
                'value': round(txn_pct_change, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': len(filter_by_period(pos_df, analysis_period)),
        'coverage_notes': [
            'Valid transactions counted as unique transaction_id excluding refund rows',
            'Analysis period: 2026-05-18 to 2026-05-25 (UTC+3)',
            'Previous period: 2026-05-11 to 2026-05-18 (UTC+3)'
        ],
        'assumptions': [
            'Each unique transaction_id represents one basket',
            'Refund transactions are excluded from basket count'
        ],
        'confidence': 0.95
    }
    findings.append(finding2)

# Finding 3: AOV change
if analysis_metrics['aov_sar'] is not None and previous_metrics['aov_sar'] is not None:
    aov_change = analysis_metrics['aov_sar'] - previous_metrics['aov_sar']
    aov_pct_change = (aov_change / previous_metrics['aov_sar']) * 100
    
    finding3 = {
        'title': 'Average Order Value Change',
        'claim': f"Average order value in analysis week was {analysis_metrics['aov_sar']:.2f} SAR, a {aov_pct_change:.1f}% change from {previous_metrics['aov_sar']:.2f} SAR in the previous week.",
        'finding_type': 'aov_trend',
        'metrics': {
            'analysis_week_aov': {
                'value': round(analysis_metrics['aov_sar'], 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue_sar'], 2),
                'denominator': analysis_metrics['valid_transactions'],
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'previous_week_aov': {
                'value': round(previous_metrics['aov_sar'], 2),
                'unit': 'SAR',
                'numerator': round(previous_metrics['total_revenue_sar'], 2),
                'denominator': previous_metrics['valid_transactions'],
                'period_start': '2026-05-11T00:00:00+03:00',
                'period_end': '2026-05-18T00:00:00+03:00'
            },
            'aov_change_sar': {
                'value': round(aov_change, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            },
            'aov_pct_change': {
                'value': round(aov_pct_change, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-05-18T00:00:00+03:00',
                'period_end': '2026-05-25T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['valid_transactions'],
        'coverage_notes': [
            'AOV calculated as total net revenue divided by valid transaction count',
            'Analysis period: 2026-05-18 to 2026-05-25 (UTC+3)',
            'Previous period: 2026-05-11 to 2026-05-18 (UTC+3)',
            'Refunds included in net revenue'
        ],
        'assumptions': [
            'AOV = total_revenue_sar / valid_transactions',
            'Valid transactions exclude refund rows'
        ],
        'confidence': 0.95
    }
    findings.append(finding3)

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)