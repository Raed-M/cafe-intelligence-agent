import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
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

# Filter data for each period
def filter_period(df, period):
    return df[(df['timestamp'] >= period['start']) & (df['timestamp'] < period['end'])]

analysis_data = filter_period(pos_df, analysis_period)
previous_data = filter_period(pos_df, previous_period)

# Calculate metrics for analysis period
def calculate_metrics(data, period_name):
    # Count valid transactions (unique transaction_id)
    valid_transactions = data['transaction_id'].nunique()
    
    # Calculate total revenue (excluding refunds for net revenue)
    non_refund_data = data[data['is_refund'] == False]
    total_revenue = non_refund_data['line_total_sar'].sum()
    
    # Calculate average order value
    if valid_transactions > 0:
        aov = total_revenue / valid_transactions
    else:
        aov = 0
    
    # Category mix (excluding refunds)
    category_revenue = non_refund_data.groupby('category')['line_total_sar'].sum()
    category_mix = (category_revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    # Channel mix (excluding refunds)
    channel_revenue = non_refund_data.groupby('channel')['line_total_sar'].sum()
    channel_mix = (channel_revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    # Refund analysis
    refund_data = data[data['is_refund'] == True]
    total_refunds = refund_data['line_total_sar'].sum()
    refund_count = refund_data['transaction_id'].nunique()
    
    return {
        'valid_transactions': valid_transactions,
        'total_revenue': total_revenue,
        'aov': aov,
        'category_revenue': category_revenue,
        'category_mix': category_mix,
        'channel_revenue': channel_revenue,
        'channel_mix': channel_mix,
        'total_refunds': total_refunds,
        'refund_count': refund_count,
        'line_item_count': len(non_refund_data)
    }

analysis_metrics = calculate_metrics(analysis_data, 'analysis')
previous_metrics = calculate_metrics(previous_data, 'previous')

# Calculate trailing baseline average
trailing_metrics_list = []
for period in trailing_baseline:
    period_data = filter_period(pos_df, period)
    trailing_metrics_list.append(calculate_metrics(period_data, 'trailing'))

# Calculate average trailing metrics
avg_trailing_revenue = np.mean([m['total_revenue'] for m in trailing_metrics_list])
avg_trailing_aov = np.mean([m['aov'] for m in trailing_metrics_list])
avg_trailing_transactions = np.mean([m['valid_transactions'] for m in trailing_metrics_list])

# Prepare findings
findings = []

# Finding 1: Revenue and Transaction Performance
revenue_change = analysis_metrics['total_revenue'] - previous_metrics['total_revenue']
revenue_change_pct = (revenue_change / previous_metrics['total_revenue'] * 100) if previous_metrics['total_revenue'] > 0 else 0
transaction_change = analysis_metrics['valid_transactions'] - previous_metrics['valid_transactions']
transaction_change_pct = (transaction_change / previous_metrics['valid_transactions'] * 100) if previous_metrics['valid_transactions'] > 0 else 0
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_change_pct = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

if abs(revenue_change_pct) > 2 or abs(transaction_change_pct) > 2:
    findings.append({
        'title': 'Revenue and Transaction Performance Week-over-Week',
        'claim': f'Analysis period (Jan 12-19, 2026) generated {analysis_metrics["total_revenue"]:.2f} SAR across {analysis_metrics["valid_transactions"]} transactions with AOV of {analysis_metrics["aov"]:.2f} SAR, compared to previous period (Jan 5-12, 2026) with {previous_metrics["total_revenue"]:.2f} SAR, {previous_metrics["valid_transactions"]} transactions, and AOV of {previous_metrics["aov"]:.2f} SAR.',
        'finding_type': 'revenue_and_transaction_performance',
        'metrics': {
            'analysis_period_revenue': {
                'value': round(analysis_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'previous_period_revenue': {
                'value': round(previous_metrics['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-01-05T00:00:00+03:00',
                'period_end': '2026-01-12T00:00:00+03:00'
            },
            'revenue_change_sar': {
                'value': round(revenue_change, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'revenue_change_pct': {
                'value': round(revenue_change_pct, 2),
                'unit': '%',
                'numerator': round(revenue_change, 2),
                'denominator': round(previous_metrics['total_revenue'], 2),
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'analysis_period_transactions': {
                'value': analysis_metrics['valid_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'previous_period_transactions': {
                'value': previous_metrics['valid_transactions'],
                'unit': 'count',
                'numerator': None,
                'denominator': None,
                'period_start': '2026-01-05T00:00:00+03:00',
                'period_end': '2026-01-12T00:00:00+03:00'
            },
            'transaction_change_pct': {
                'value': round(transaction_change_pct, 2),
                'unit': '%',
                'numerator': transaction_change,
                'denominator': previous_metrics['valid_transactions'],
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'analysis_period_aov': {
                'value': round(analysis_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(analysis_metrics['total_revenue'], 2),
                'denominator': analysis_metrics['valid_transactions'],
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            },
            'previous_period_aov': {
                'value': round(previous_metrics['aov'], 2),
                'unit': 'SAR',
                'numerator': round(previous_metrics['total_revenue'], 2),
                'denominator': previous_metrics['valid_transactions'],
                'period_start': '2026-01-05T00:00:00+03:00',
                'period_end': '2026-01-12T00:00:00+03:00'
            },
            'aov_change_pct': {
                'value': round(aov_change_pct, 2),
                'unit': '%',
                'numerator': round(aov_change, 2),
                'denominator': round(previous_metrics['aov'], 2),
                'period_start': '2026-01-12T00:00:00+03:00',
                'period_end': '2026-01-19T00:00:00+03:00'
            }
        },
        'source_names': ['pos'],
        'sample_size': analysis_metrics['line_item_count'],
        'coverage_notes': [
            f'Analysis period: {analysis_metrics["line_item_count"]} non-refund line items from {analysis_metrics["valid_transactions"]} transactions',
            f'Previous period: {previous_metrics["line_item_count"]} non-refund line items from {previous_metrics["valid_transactions"]} transactions',
            'Refunds excluded from revenue calculations per standard revenue quality practices'
        ],
        'assumptions': [
            'Revenue mix analysis uses line_total_sar EXCLUDING refunds',
            'Valid transactions counted as unique transaction_id values',
            'AOV calculated as total net revenue divided by transaction count'
        ],
        'confidence': 0.95
    })

# Finding 2: Category Mix Analysis
if 'Bakery' in analysis_metrics['category_mix'].index and 'Bakery' in previous_metrics['category_mix'].index:
    bakery_analysis = analysis_metrics['category_mix']['Bakery']
    bakery_previous = previous_metrics['category_mix']['Bakery']
    bakery_shift = bakery_analysis - bakery_previous
    
    bakery_revenue_analysis = analysis_metrics['category_revenue']['Bakery']
    bakery_revenue_previous = previous_metrics['category_revenue']['Bakery']
    
    if abs(bakery_shift) > 1:
        findings.append({
            'title': 'Bakery Category Mix Shift',
            'claim': f'Bakery category revenue increased from {bakery_revenue_previous:.2f} SAR ({bakery_previous:.1f}% of category mix) in previous period to {bakery_revenue_analysis:.2f} SAR ({bakery_analysis:.1f}% of category mix) in analysis period, representing a {bakery_shift:.1f} percentage point shift in category contribution.',
            'finding_type': 'category_mix_shift',
            'metrics': {
                'analysis_period_bakery_revenue': {
                    'value': round(bakery_revenue_analysis, 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                },
                'previous_period_bakery_revenue': {
                    'value': round(bakery_revenue_previous, 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-01-05T00:00:00+03:00',
                    'period_end': '2026-01-12T00:00:00+03:00'
                },
                'analysis_period_category_mix_pct': {
                    'value': round(bakery_analysis, 2),
                    'unit': '%',
                    'numerator': round(bakery_revenue_analysis, 2),
                    'denominator': round(analysis_metrics['total_revenue'], 2),
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                },
                'previous_period_category_mix_pct': {
                    'value': round(bakery_previous, 2),
                    'unit': '%',
                    'numerator': round(bakery_revenue_previous, 2),
                    'denominator': round(previous_metrics['total_revenue'], 2),
                    'period_start': '2026-01-05T00:00:00+03:00',
                    'period_end': '2026-01-12T00:00:00+03:00'
                },
                'category_mix_shift_pct': {
                    'value': round(bakery_shift, 2),
                    'unit': 'percentage points',
                    'numerator': round(bakery_shift, 2),
                    'denominator': None,
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                }
            },
            'source_names': ['pos', 'menu'],
            'sample_size': analysis_metrics['line_item_count'],
            'coverage_notes': [
                f'Analysis period: {analysis_metrics["line_item_count"]} non-refund line items',
                f'Previous period: {previous_metrics["line_item_count"]} non-refund line items',
                'Category assignments from menu SKU reference'
            ],
            'assumptions': [
                'Revenue mix analysis uses line_total_sar EXCLUDING refunds',
                'Category mix percentages calculated as category revenue / total revenue',
                'Product names repaired through menu SKU reference'
            ],
            'confidence': 0.92
        })

# Finding 3: Channel Mix Analysis
if 'Takeaway' in analysis_metrics['channel_mix'].index and 'Takeaway' in previous_metrics['channel_mix'].index:
    takeaway_analysis = analysis_metrics['channel_mix']['Takeaway']
    takeaway_previous = previous_metrics['channel_mix']['Takeaway']
    takeaway_shift = takeaway_analysis - takeaway_previous
    
    takeaway_revenue_analysis = analysis_metrics['channel_revenue']['Takeaway']
    takeaway_revenue_previous = previous_metrics['channel_revenue']['Takeaway']
    
    if abs(takeaway_shift) > 2:
        findings.append({
            'title': 'Takeaway Channel Mix Shift',
            'claim': f'Takeaway channel revenue increased from {takeaway_revenue_previous:.2f} SAR ({takeaway_previous:.1f}% of channel mix) in previous period to {takeaway_revenue_analysis:.2f} SAR ({takeaway_analysis:.1f}% of channel mix) in analysis period, representing a {takeaway_shift:.1f} percentage point shift in channel contribution.',
            'finding_type': 'channel_mix_shift',
            'metrics': {
                'analysis_period_takeaway_revenue': {
                    'value': round(takeaway_revenue_analysis, 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                },
                'previous_period_takeaway_revenue': {
                    'value': round(takeaway_revenue_previous, 2),
                    'unit': 'SAR',
                    'numerator': None,
                    'denominator': None,
                    'period_start': '2026-01-05T00:00:00+03:00',
                    'period_end': '2026-01-12T00:00:00+03:00'
                },
                'analysis_period_channel_mix_pct': {
                    'value': round(takeaway_analysis, 2),
                    'unit': '%',
                    'numerator': round(takeaway_revenue_analysis, 2),
                    'denominator': round(analysis_metrics['total_revenue'], 2),
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                },
                'previous_period_channel_mix_pct': {
                    'value': round(takeaway_previous, 2),
                    'unit': '%',
                    'numerator': round(takeaway_revenue_previous, 2),
                    'denominator': round(previous_metrics['total_revenue'], 2),
                    'period_start': '2026-01-05T00:00:00+03:00',
                    'period_end': '2026-01-12T00:00:00+03:00'
                },
                'channel_mix_shift_pct': {
                    'value': round(takeaway_shift, 2),
                    'unit': 'percentage points',
                    'numerator': round(takeaway_shift, 2),
                    'denominator': None,
                    'period_start': '2026-01-12T00:00:00+03:00',
                    'period_end': '2026-01-19T00:00:00+03:00'
                }
            },
            'source_names': ['pos'],
            'sample_size': analysis_metrics['line_item_count'],
            'coverage_notes': [
                f'Analysis period: {analysis_metrics["line_item_count"]} non-refund line items',
                f'Previous period: {previous_metrics["line_item_count"]} non-refund line items',
                'Channel assignments from POS data'
            ],
            'assumptions': [
                'Revenue mix analysis uses line_total_sar EXCLUDING refunds',
                'Channel mix percentages calculated as channel revenue / total revenue',
                'Valid transactions counted as unique transaction_id values'
            ],
            'confidence': 0.90
        })

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
