import os
import json
import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq

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
    'start': pd.Timestamp('2026-01-12T00:00:00+03:00'),
    'end': pd.Timestamp('2026-01-19T00:00:00+03:00')
}
previous_period = {
    'start': pd.Timestamp('2026-01-05T00:00:00+03:00'),
    'end': pd.Timestamp('2026-01-12T00:00:00+03:00')
}
trailing_baseline = [
    {
        'start': pd.Timestamp('2026-01-05T00:00:00+03:00'),
        'end': pd.Timestamp('2026-01-12T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2025-12-29T00:00:00+03:00'),
        'end': pd.Timestamp('2026-01-05T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2025-12-22T00:00:00+03:00'),
        'end': pd.Timestamp('2025-12-29T00:00:00+03:00')
    },
    {
        'start': pd.Timestamp('2025-12-15T00:00:00+03:00'),
        'end': pd.Timestamp('2025-12-22T00:00:00+03:00')
    }
]

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis period
analysis_data = pos_df[
    (pos_df['timestamp'] >= analysis_period['start']) & 
    (pos_df['timestamp'] < analysis_period['end'])
].copy()

# Filter data for previous period
previous_data = pos_df[
    (pos_df['timestamp'] >= previous_period['start']) & 
    (pos_df['timestamp'] < previous_period['end'])
].copy()

# Filter data for trailing baseline (all 4 weeks combined)
trailing_data = pos_df[
    (pos_df['timestamp'] >= trailing_baseline[0]['start']) & 
    (pos_df['timestamp'] < trailing_baseline[-1]['end'])
].copy()

# Helper function to calculate metrics excluding refunds
def calculate_metrics(data, period_name):
    # Exclude refunds for revenue calculations
    non_refund_data = data[data['is_refund'] == False].copy()
    
    # Count unique transactions
    unique_transactions = non_refund_data['transaction_id'].nunique()
    
    # Calculate total revenue (excluding refunds)
    total_revenue = non_refund_data['line_total_sar'].sum()
    
    # Calculate average order value
    aov = total_revenue / unique_transactions if unique_transactions > 0 else 0
    
    # Count line items
    line_items = len(non_refund_data)
    
    return {
        'unique_transactions': unique_transactions,
        'total_revenue': total_revenue,
        'aov': aov,
        'line_items': line_items,
        'refund_count': len(data[data['is_refund'] == True])
    }

# Calculate metrics for each period
analysis_metrics = calculate_metrics(analysis_data, 'analysis')
previous_metrics = calculate_metrics(previous_data, 'previous')

# Calculate trailing baseline average
trailing_metrics_list = []
for period in trailing_baseline:
    period_data = pos_df[
        (pos_df['timestamp'] >= period['start']) & 
        (pos_df['timestamp'] < period['end'])
    ].copy()
    trailing_metrics_list.append(calculate_metrics(period_data, 'trailing'))

# Average trailing metrics
avg_trailing_transactions = sum(m['unique_transactions'] for m in trailing_metrics_list) / len(trailing_metrics_list)
avg_trailing_revenue = sum(m['total_revenue'] for m in trailing_metrics_list) / len(trailing_metrics_list)
avg_trailing_aov = sum(m['aov'] for m in trailing_metrics_list) / len(trailing_metrics_list)

# Finding 1: Transaction Count Change
transaction_change = analysis_metrics['unique_transactions'] - previous_metrics['unique_transactions']
transaction_change_pct = (transaction_change / previous_metrics['unique_transactions'] * 100) if previous_metrics['unique_transactions'] > 0 else 0

# Finding 2: Revenue Change
revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_change_pct = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] > 0 else 0

# Finding 3: AOV Change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_change_pct = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

# Category mix analysis
analysis_category_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('category')['line_total_sar'].sum()
previous_category_revenue = previous_data[previous_data['is_refund'] == False].groupby('category')['line_total_sar'].sum()

# Channel mix analysis
analysis_channel_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum()
previous_channel_revenue = previous_data[previous_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum()

# Prepare findings
findings = []

# Finding 1: Transaction Count
if transaction_change != 0:
    findings.append({
        'title': 'Transaction Count Change Week-over-Week',
        'claim': f'Valid transaction count increased from {previous_metrics["unique_transactions"]} to {analysis_metrics["unique_transactions"]} ({transaction_change_pct:+.1f}%)',
        'finding_type': 'transaction_volume',
        'metrics': {
            'analysis_period_transactions': {
                'value': analysis_metrics['unique_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'previous_period_transactions': {
                'value': previous_metrics['unique_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': previous_period['start'].isoformat(),
                'period_end': previous_period['end'].isoformat()
            },
            'transaction_change': {
                'value': transaction_change,
                'unit': 'count',
                'numerator': analysis_metrics['unique_transactions'],
                'denominator': previous_metrics['unique_transactions'],
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'transaction_change_pct': {
                'value': transaction_change_pct,
                'unit': '%',
                'numerator': transaction_change,
                'denominator': previous_metrics['unique_transactions'],
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['line_items'],
        'coverage_notes': [
            f'Analysis period: {analysis_metrics["line_items"]} line items, {analysis_metrics["unique_transactions"]} unique transactions',
            f'Previous period: {previous_metrics["line_items"]} line items, {previous_metrics["unique_transactions"]} unique transactions',
            'Refunds excluded from transaction count'
        ],
        'assumptions': [
            'Transaction count uses unique transaction_id values',
            'Refunds (is_refund=True) excluded from transaction count',
            'Line items with is_refund=False represent valid sales'
        ],
        'confidence': 0.95
    })

# Finding 2: Revenue Change
if revenue_change != 0:
    findings.append({
        'title': 'Net Revenue Change Week-over-Week',
        'claim': f'Net revenue increased from SAR {previous_metrics["total_revenue"]:.2f} to SAR {analysis_metrics["total_revenue"]:.2f} ({revenue_change_pct:+.1f}%)',
        'finding_type': 'revenue',
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
            'revenue_change': {
                'value': round(revenue_change, 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue'], 2),
                'denominator': round(previous_metrics['total_revenue'], 2),
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'revenue_change_pct': {
                'value': revenue_change_pct,
                'unit': '%',
                'numerator': round(revenue_change, 2),
                'denominator': round(previous_metrics['total_revenue'], 2),
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['line_items'],
        'coverage_notes': [
            f'Analysis period: {analysis_metrics["line_items"]} line items, SAR {analysis_metrics["total_revenue"]:.2f} net revenue',
            f'Previous period: {previous_metrics["line_items"]} line items, SAR {previous_metrics["total_revenue"]:.2f} net revenue',
            'Refunds excluded from revenue calculations'
        ],
        'assumptions': [
            'Revenue uses line_total_sar column',
            'Refunds (is_refund=True) excluded from revenue totals',
            'Line items with is_refund=False represent valid sales'
        ],
        'confidence': 0.95
    })

# Finding 3: AOV Change
if aov_change != 0:
    findings.append({
        'title': 'Average Order Value Change Week-over-Week',
        'claim': f'Average order value increased from SAR {previous_metrics["aov"]:.2f} to SAR {analysis_metrics["aov"]:.2f} ({aov_change_pct:+.1f}%)',
        'finding_type': 'average_order_value',
        'metrics': {
            'analysis_period_aov': {
                'value': round(analysis_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue'], 2),
                'denominator': analysis_metrics['unique_transactions'],
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'previous_period_aov': {
                'value': round(previous_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(previous_metrics['total_revenue'], 2),
                'denominator': previous_metrics['unique_transactions'],
                'period_start': previous_period['start'].isoformat(),
                'period_end': previous_period['end'].isoformat()
            },
            'aov_change': {
                'value': round(aov_change, 2),
                'unit': 'SAR',
                'numerator': round(aov_change, 2),
                'denominator': 1,
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            },
            'aov_change_pct': {
                'value': aov_change_pct,
                'unit': '%',
                'numerator': round(aov_change, 2),
                'denominator': round(previous_metrics['aov'], 2),
                'period_start': analysis_period['start'].isoformat(),
                'period_end': analysis_period['end'].isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['line_items'],
        'coverage_notes': [
            f'Analysis period: {analysis_metrics["unique_transactions"]} transactions, SAR {analysis_metrics["aov"]:.2f} average',
            f'Previous period: {previous_metrics["unique_transactions"]} transactions, SAR {previous_metrics["aov"]:.2f} average',
            'Refunds excluded from AOV calculations'
        ],
        'assumptions': [
            'AOV calculated as total net revenue / unique transaction count',
            'Refunds (is_refund=True) excluded from revenue and transaction count',
            'Each transaction_id represents one customer order'
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
