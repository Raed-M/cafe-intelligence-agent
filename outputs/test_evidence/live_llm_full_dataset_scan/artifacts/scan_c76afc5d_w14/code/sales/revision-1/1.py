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

# Define periods
analysis_period = {
    'start': datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc),
    'end': datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
}

previous_period = {
    'start': datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc),
    'end': datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
}

trailing_baseline_periods = [
    {
        'start': datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        'start': datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        'start': datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        'start': datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
    }
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for each period
def filter_by_period(df, period):
    return df[(df['timestamp'] >= period['start']) & (df['timestamp'] < period['end'])]

analysis_data = filter_by_period(pos_df, analysis_period)
previous_data = filter_by_period(pos_df, previous_period)
baseline_data = [filter_by_period(pos_df, p) for p in trailing_baseline_periods]

# Calculate metrics for analysis period
def calculate_metrics(df):
    # Count valid transactions (unique transaction_id)
    valid_transactions = df['transaction_id'].nunique()
    
    # Total revenue (net, including refunds as negative)
    total_revenue = df['line_total_sar'].sum()
    
    # Average order value
    if valid_transactions > 0:
        aov = total_revenue / valid_transactions
    else:
        aov = 0
    
    # Product mix (by category)
    category_revenue = df.groupby('category')['line_total_sar'].sum()
    
    # Channel mix
    channel_revenue = df.groupby('channel')['line_total_sar'].sum()
    
    # Refund analysis
    refund_rows = df[df['is_refund'] == True]
    refund_count = refund_rows['transaction_id'].nunique()
    refund_amount = refund_rows['line_total_sar'].sum()
    
    return {
        'valid_transactions': valid_transactions,
        'total_revenue': total_revenue,
        'aov': aov,
        'category_revenue': category_revenue,
        'channel_revenue': channel_revenue,
        'refund_count': refund_count,
        'refund_amount': refund_amount,
        'row_count': len(df)
    }

analysis_metrics = calculate_metrics(analysis_data)
previous_metrics = calculate_metrics(previous_data)

# Calculate baseline average
baseline_metrics_list = [calculate_metrics(df) for df in baseline_data]
baseline_avg_revenue = np.mean([m['total_revenue'] for m in baseline_metrics_list])
baseline_avg_aov = np.mean([m['aov'] for m in baseline_metrics_list])
baseline_avg_transactions = np.mean([m['valid_transactions'] for m in baseline_metrics_list])

# Calculate changes
revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_pct_change = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] != 0 else 0

aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] != 0 else 0

transaction_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
transaction_pct_change = (transaction_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] != 0 else 0

# Analyze product performance with menu data
pos_with_menu = analysis_data.merge(menu_df, on='sku', how='left')

# Check for products launched during analysis period
product_performance = []
for _, menu_row in menu_df.iterrows():
    sku = menu_row['sku']
    launch_date = pd.to_datetime(menu_row['launch_date']) if pd.notna(menu_row['launch_date']) else None
    retire_date = pd.to_datetime(menu_row['retire_date']) if pd.notna(menu_row['retire_date']) else None
    
    # Check if product is eligible for analysis period
    is_eligible = True
    if launch_date and launch_date >= analysis_period['end']:
        is_eligible = False
    if retire_date and retire_date <= analysis_period['start']:
        is_eligible = False
    
    if is_eligible:
        product_data = analysis_data[analysis_data['sku'] == sku]
        if len(product_data) > 0:
            product_revenue = product_data['line_total_sar'].sum()
            product_qty = product_data['quantity'].sum()
            product_transactions = product_data['transaction_id'].nunique()
            
            product_performance.append({
                'sku': sku,
                'item_en': menu_row['item_en'],
                'category': menu_row['category'],
                'revenue': product_revenue,
                'quantity': product_qty,
                'transactions': product_transactions,
                'launch_date': launch_date,
                'retire_date': retire_date
            })

# Sort by revenue
product_performance.sort(key=lambda x: x['revenue'], reverse=True)

# Prepare findings
findings = []

# Finding 1: Revenue change week-over-week
if previous_metrics['total_revenue'] != 0:
    finding1 = {
        'title': 'Weekly Revenue Change',
        'claim': f'Total net revenue in analysis week (Apr 13-20) was SAR {analysis_metrics["total_revenue"]:.2f}, compared to SAR {previous_metrics["total_revenue"]:.2f} in previous week (Apr 6-13), representing a {revenue_pct_change:.2f}% change.',
        'finding_type': 'revenue_change',
        'metrics': {
            'analysis_period_revenue': {
                'value': round(analysis_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'previous_period_revenue': {
                'value': round(previous_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_period['start'].isoformat(),
                'period_end': previous_period['end'].isoformat()
            },
            'revenue_change_sar': {
                'value': round(revenue_change, 2),
                'unit': 'SAR',
                'numerator': round(revenue_change, 2),
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'revenue_pct_change': {
                'value': round(revenue_pct_change, 2),
                'unit': '%',
                'numerator': round(revenue_change, 2),
                'denominator': round(previous_metrics['total_revenue'], 2),
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['row_count'],
        'coverage_notes': [
            f'Analysis period: {analysis_period["start"].isoformat()} to {analysis_period["end"].isoformat()}',
            f'Previous period: {previous_period["start"].isoformat()} to {previous_period["end"].isoformat()}',
            f'Refunds included as negative values in net revenue calculation',
            f'Refund transactions in analysis period: {analysis_metrics["refund_count"]}, amount: SAR {analysis_metrics["refund_amount"]:.2f}'
        ],
        'assumptions': [
            'Transactions counted by unique transaction_id',
            'Revenue calculated as sum of line_total_sar including refunds',
            'Timestamp converted to UTC for period filtering'
        ],
        'confidence': 0.95
    }
    findings.append(finding1)

# Finding 2: Average Order Value change
if previous_metrics['aov'] != 0:
    finding2 = {
        'title': 'Average Order Value Change',
        'claim': f'Average order value in analysis week was SAR {analysis_metrics["aov"]:.2f}, compared to SAR {previous_metrics["aov"]:.2f} in previous week, representing a {aov_pct_change:.2f}% change.',
        'finding_type': 'aov_change',
        'metrics': {
            'analysis_period_aov': {
                'value': round(analysis_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue'], 2),
                'denominator': analysis_metrics['valid_transactions'],
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'previous_period_aov': {
                'value': round(previous_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(previous_metrics['total_revenue'], 2),
                'denominator': previous_metrics['valid_transactions'],
                'period_start': previous_period['start'].isoformat(),
                'period_end': previous_period['end'].isoformat()
            },
            'aov_change_sar': {
                'value': round(aov_change, 2),
                'unit': 'SAR',
                'numerator': round(aov_change, 2),
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'aov_pct_change': {
                'value': round(aov_pct_change, 2),
                'unit': '%',
                'numerator': round(aov_change, 2),
                'denominator': round(previous_metrics['aov'], 2),
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['valid_transactions'],
        'coverage_notes': [
            f'Analysis period: {analysis_period["start"].isoformat()} to {analysis_period["end"].isoformat()}',
            f'Previous period: {previous_period["start"].isoformat()} to {previous_period["end"].isoformat()}',
            f'AOV calculated as total revenue divided by unique transaction count',
            f'Refunds included in net revenue'
        ],
        'assumptions': [
            'Transactions counted by unique transaction_id',
            'Revenue calculated as sum of line_total_sar including refunds',
            'AOV = Total Revenue / Valid Transactions'
        ],
        'confidence': 0.95
    }
    findings.append(finding2)

# Finding 3: Transaction volume change
if previous_metrics['valid_transactions'] > 0:
    finding3 = {
        'title': 'Transaction Volume Change',
        'claim': f'Valid transaction count in analysis week was {analysis_metrics["valid_transactions"]}, compared to {previous_metrics["valid_transactions"]} in previous week, representing a {transaction_pct_change:.2f}% change.',
        'finding_type': 'transaction_volume_change',
        'metrics': {
            'analysis_period_transactions': {
                'value': analysis_metrics['valid_transactions'],
                'unit': 'transactions',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'previous_period_transactions': {
                'value': previous_metrics['valid_transactions'],
                'unit': 'transactions',
                'numerator': None,
                'denominator': None,
                'period_start': previous_period['start'].isoformat(),
                'period_end': previous_period['end'].isoformat()
            },
            'transaction_change': {
                'value': transaction_change,
                'unit': 'transactions',
                'numerator': transaction_change,
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'transaction_pct_change': {
                'value': round(transaction_pct_change, 2),
                'unit': '%',
                'numerator': transaction_change,
                'denominator': previous_metrics['valid_transactions'],
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['row_count'],
        'coverage_notes': [
            f'Analysis period: {analysis_period["start"].isoformat()} to {analysis_period["end"].isoformat()}',
            f'Previous period: {previous_period["start"].isoformat()} to {previous_period["end"].isoformat()}',
            f'Transactions counted by unique transaction_id'
        ],
        'assumptions': [
            'Transactions counted by unique transaction_id',
            'Timestamp converted to UTC for period filtering'
        ],
        'confidence': 0.95
    }
    findings.append(finding3)

# Prepare output
output = {
    'status': 'success',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
