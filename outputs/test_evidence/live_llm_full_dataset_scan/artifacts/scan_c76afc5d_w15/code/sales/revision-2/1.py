import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'])
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'])

# Define periods
analysis_period = {
    'start': pd.Timestamp('2026-04-20T00:00:00+03:00'),
    'end': pd.Timestamp('2026-04-27T00:00:00+03:00')
}

previous_period = {
    'start': pd.Timestamp('2026-04-13T00:00:00+03:00'),
    'end': pd.Timestamp('2026-04-20T00:00:00+03:00')
}

trailing_baseline_periods = [
    {
        'start': pd.Timestamp('2026-04-13T00:00:00+03:00'),
        'end': pd.Timestamp('2026-04-20T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2026-04-06T00:00:00+03:00'),
        'end': pd.Timestamp('2026-04-13T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2026-03-30T00:00:00+03:00'),
        'end': pd.Timestamp('2026-04-06T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2026-03-23T00:00:00+03:00'),
        'end': pd.Timestamp('2026-03-30T00:00:00+03:00')
    }
]

# Helper function to filter data by period
def filter_by_period(df, period):
    return df[(df['timestamp'] >= period['start']) & (df['timestamp'] < period['end'])]

# Helper function to check if product is eligible in period
def is_product_eligible(sku, period_start, period_end, menu_df):
    product = menu_df[menu_df['sku'] == sku]
    if product.empty:
        return False
    
    launch_date = product['launch_date'].iloc[0]
    retire_date = product['retire_date'].iloc[0]
    
    # Product is eligible if it launched before period end and (not retired or retired after period start)
    if pd.isna(launch_date):
        launch_eligible = True
    else:
        launch_eligible = launch_date < period_end
    
    if pd.isna(retire_date):
        retire_eligible = True
    else:
        retire_eligible = retire_date > period_start
    
    return launch_eligible and retire_eligible

# Calculate metrics for each period
def calculate_period_metrics(df, period, menu_df):
    period_data = filter_by_period(df, period)
    
    # Filter out refunds for transaction count
    non_refund_data = period_data[~period_data['is_refund']]
    
    # Count unique transactions
    transaction_count = non_refund_data['transaction_id'].nunique()
    
    # Calculate revenue (including refunds as negative)
    total_revenue = period_data['line_total_sar'].sum()
    
    # Calculate AOV
    if transaction_count > 0:
        aov = total_revenue / transaction_count
    else:
        aov = 0
    
    # Calculate by category
    category_revenue = {}
    for category in period_data['category'].unique():
        if pd.notna(category):
            cat_data = period_data[period_data['category'] == category]
            cat_revenue = cat_data['line_total_sar'].sum()
            cat_qty = cat_data['quantity'].sum()
            category_revenue[category] = {
                'revenue': cat_revenue,
                'quantity': cat_qty
            }
    
    # Calculate by channel
    channel_revenue = {}
    for channel in period_data['channel'].unique():
        if pd.notna(channel):
            ch_data = period_data[period_data['channel'] == channel]
            ch_revenue = ch_data['line_total_sar'].sum()
            ch_txn = ch_data[~ch_data['is_refund']]['transaction_id'].nunique()
            channel_revenue[channel] = {
                'revenue': ch_revenue,
                'transactions': ch_txn
            }
    
    return {
        'transaction_count': transaction_count,
        'total_revenue': total_revenue,
        'aov': aov,
        'category_revenue': category_revenue,
        'channel_revenue': channel_revenue,
        'period_data': period_data
    }

# Calculate metrics for all periods
analysis_metrics = calculate_period_metrics(pos_df, analysis_period, menu_df)
previous_metrics = calculate_period_metrics(pos_df, previous_period, menu_df)

trailing_metrics = []
for period in trailing_baseline_periods:
    trailing_metrics.append(calculate_period_metrics(pos_df, period, menu_df))

# Calculate average of trailing baseline
avg_trailing_revenue = np.mean([m['total_revenue'] for m in trailing_metrics])
avg_trailing_transactions = np.mean([m['transaction_count'] for m in trailing_metrics])
avg_trailing_aov = np.mean([m['aov'] for m in trailing_metrics])

# Prepare findings
findings = []

# Finding 1: Revenue change from previous week
revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_change_pct = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] != 0 else 0

# Check if this is material (>5%)
if abs(revenue_change_pct) > 5:
    findings.append({
        'title': 'Revenue Change vs Previous Week',
        'claim': f"Total revenue in analysis period (2026-04-20 to 2026-04-27) was {analysis_metrics['total_revenue']:.2f} SAR, compared to {previous_metrics['total_revenue']:.2f} SAR in previous period (2026-04-13 to 2026-04-20), representing a {revenue_change_pct:.1f}% change.",
        'finding_type': 'revenue_change',
        'metrics': {
            'analysis_period_revenue': {
                'value': round(analysis_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-20T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'previous_period_revenue': {
                'value': round(previous_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-20T00:00:00+03:00'
            },
            'revenue_change': {
                'value': round(revenue_change, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'revenue_change_pct': {
                'value': round(revenue_change_pct, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': len(analysis_metrics['period_data']),
        'coverage_notes': [
            'Analysis period: 2026-04-20 to 2026-04-27',
            'Previous period: 2026-04-13 to 2026-04-20',
            'Includes refunds as negative revenue'
        ],
        'assumptions': [
            'line_total_sar represents net revenue including refunds',
            'All transactions in period are valid'
        ],
        'confidence': 0.95
    })

# Finding 2: Transaction count change
txn_change = analysis_metrics['transaction_count'] - previous_metrics['transaction_count']
txn_change_pct = (txn_change / previous_metrics['transaction_count'] * 100) if previous_metrics['transaction_count'] != 0 else 0

if abs(txn_change_pct) > 5:
    findings.append({
        'title': 'Transaction Count Change vs Previous Week',
        'claim': f"Valid transaction count in analysis period (2026-04-20 to 2026-04-27) was {analysis_metrics['transaction_count']}, compared to {previous_metrics['transaction_count']} in previous period (2026-04-13 to 2026-04-20), representing a {txn_change_pct:.1f}% change.",
        'finding_type': 'transaction_count_change',
        'metrics': {
            'analysis_period_transactions': {
                'value': analysis_metrics['transaction_count'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-20T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'previous_period_transactions': {
                'value': previous_metrics['transaction_count'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-20T00:00:00+03:00'
            },
            'transaction_count_change': {
                'value': txn_change,
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'transaction_count_change_pct': {
                'value': round(txn_change_pct, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': len(analysis_metrics['period_data']),
        'coverage_notes': [
            'Analysis period: 2026-04-20 to 2026-04-27',
            'Previous period: 2026-04-13 to 2026-04-20',
            'Excludes refund transactions from count'
        ],
        'assumptions': [
            'transaction_id uniquely identifies a basket',
            'is_refund flag correctly identifies refund transactions'
        ],
        'confidence': 0.95
    })

# Finding 3: AOV change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_change_pct = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] != 0 else 0

if abs(aov_change_pct) > 5:
    findings.append({
        'title': 'Average Order Value Change vs Previous Week',
        'claim': f"Average order value in analysis period (2026-04-20 to 2026-04-27) was {analysis_metrics['aov']:.2f} SAR, compared to {previous_metrics['aov']:.2f} SAR in previous period (2026-04-13 to 2026-04-20), representing a {aov_change_pct:.1f}% change.",
        'finding_type': 'aov_change',
        'metrics': {
            'analysis_period_aov': {
                'value': round(analysis_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue'], 2),
                'denominator': analysis_metrics['transaction_count'],
                'period_start': '2026-04-20T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'previous_period_aov': {
                'value': round(previous_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(previous_metrics['total_revenue'], 2),
                'denominator': previous_metrics['transaction_count'],
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-20T00:00:00+03:00'
            },
            'aov_change': {
                'value': round(aov_change, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            },
            'aov_change_pct': {
                'value': round(aov_change_pct, 1),
                'unit': '%',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-04-13T00:00:00+03:00',
                'period_end': '2026-04-27T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': len(analysis_metrics['period_data']),
        'coverage_notes': [
            'Analysis period: 2026-04-20 to 2026-04-27',
            'Previous period: 2026-04-13 to 2026-04-20',
            'AOV calculated as total revenue / valid transaction count'
        ],
        'assumptions': [
            'line_total_sar represents net revenue including refunds',
            'transaction_id uniquely identifies a basket',
            'is_refund flag correctly identifies refund transactions'
        ],
        'confidence': 0.95
    })

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
