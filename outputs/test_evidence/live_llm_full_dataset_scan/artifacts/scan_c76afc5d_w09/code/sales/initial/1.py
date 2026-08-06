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

# Parse periods
analysis_start = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-16T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-03-02T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-03-09T00:00:00+03:00")

trailing_periods = [
    ("2026-03-02T00:00:00+03:00", "2026-03-09T00:00:00+03:00"),
    ("2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00"),
    ("2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00"),
    ("2026-02-09T00:00:00+03:00", "2026-02-16T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_mask = (pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)
analysis_data = pos_df[analysis_mask].copy()

# Filter for previous period
previous_mask = (pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)
previous_data = pos_df[previous_mask].copy()

# Filter for trailing baseline (average of 4 weeks)
trailing_data_list = []
for period_start_str, period_end_str in trailing_periods:
    period_start = datetime.fromisoformat(period_start_str)
    period_end = datetime.fromisoformat(period_end_str)
    period_mask = (pos_df['timestamp'] >= period_start) & (pos_df['timestamp'] < period_end)
    trailing_data_list.append(pos_df[period_mask].copy())

findings = []

# Finding 1: Revenue and Transaction Count Change
def calculate_metrics(data):
    # Count valid transactions (unique transaction_id)
    valid_txns = data[~data['is_refund']]['transaction_id'].nunique()
    # Net revenue (including refunds as negative)
    net_revenue = data['line_total_sar'].sum()
    # Average order value
    aov = net_revenue / valid_txns if valid_txns > 0 else 0
    return {
        'transactions': valid_txns,
        'net_revenue': net_revenue,
        'aov': aov,
        'row_count': len(data)
    }

analysis_metrics = calculate_metrics(analysis_data)
previous_metrics = calculate_metrics(previous_data)

# Calculate trailing baseline average
trailing_metrics_list = [calculate_metrics(d) for d in trailing_data_list]
trailing_avg_txns = np.mean([m['transactions'] for m in trailing_metrics_list])
trailing_avg_revenue = np.mean([m['net_revenue'] for m in trailing_metrics_list])
trailing_avg_aov = np.mean([m['aov'] for m in trailing_metrics_list])

# Revenue change
revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
revenue_pct_change = (revenue_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] != 0 else 0

txn_change = analysis_metrics['transactions'] - previous_metrics['transactions']
txn_pct_change = (txn_change / previous_metrics['transactions'] * 100) if previous_metrics['transactions'] > 0 else 0

aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] > 0 else 0

if abs(revenue_pct_change) > 2 or abs(txn_pct_change) > 2:
    findings.append({
        "title": "Revenue and Transaction Count Change Week-over-Week",
        "claim": f"Net revenue in analysis week (Mar 9-16) was SAR {analysis_metrics['net_revenue']:.2f} vs SAR {previous_metrics['net_revenue']:.2f} in previous week (Mar 2-9), a change of SAR {revenue_change:.2f} ({revenue_pct_change:.1f}%). Valid transaction count changed from {previous_metrics['transactions']} to {analysis_metrics['transactions']} ({txn_pct_change:.1f}%). Average order value changed from SAR {previous_metrics['aov']:.2f} to SAR {analysis_metrics['aov']:.2f} ({aov_pct_change:.1f}%).",
        "finding_type": "revenue_and_transaction_change",
        "metrics": {
            "net_revenue_analysis": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "net_revenue_previous": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "transactions_analysis": {
                "value": analysis_metrics['transactions'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "transactions_previous": {
                "value": previous_metrics['transactions'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "aov_analysis": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "aov_previous": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['row_count'],
        "coverage_notes": [
            f"Analysis period: {analysis_metrics['row_count']} POS line items",
            f"Previous period: {previous_metrics['row_count']} POS line items",
            "Refunds included in net revenue calculations",
            "Valid transactions counted using unique transaction_id excluding refund rows"
        ],
        "assumptions": [
            "is_refund flag accurately identifies refund transactions",
            "line_total_sar represents net realized revenue per line item",
            "transaction_id uniquely identifies a basket"
        ],
        "confidence": 0.95
    })

# Finding 2: Category Mix Analysis
def get_category_mix(data):
    category_revenue = data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    total_revenue = category_revenue.sum()
    category_pct = (category_revenue / total_revenue * 100).round(1)
    return category_revenue, category_pct

analysis_cat_rev, analysis_cat_pct = get_category_mix(analysis_data)
previous_cat_rev, previous_cat_pct = get_category_mix(previous_data)

# Find significant category shifts
category_shifts = []
for cat in analysis_cat_pct.index:
    if cat in previous_cat_pct.index:
        shift = analysis_cat_pct[cat] - previous_cat_pct[cat]
        if abs(shift) > 1.5:  # More than 1.5% shift
            category_shifts.append({
                'category': cat,
                'analysis_pct': analysis_cat_pct[cat],
                'previous_pct': previous_cat_pct[cat],
                'shift': shift,
                'analysis_revenue': analysis_cat_rev[cat],
                'previous_revenue': previous_cat_rev[cat]
            })

if category_shifts:
    # Sort by absolute shift
    category_shifts.sort(key=lambda x: abs(x['shift']), reverse=True)
    top_shift = category_shifts[0]
    
    findings.append({
        "title": "Category Mix Shift",
        "claim": f"Category '{top_shift['category']}' represented {top_shift['analysis_pct']:.1f}% of revenue in analysis week vs {top_shift['previous_pct']:.1f}% in previous week, a shift of {top_shift['shift']:.1f} percentage points. Revenue for this category was SAR {top_shift['analysis_revenue']:.2f} vs SAR {top_shift['previous_revenue']:.2f}.",
        "finding_type": "category_mix_change",
        "metrics": {
            "category_name": {
                "value": top_shift['category'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "category_pct_analysis": {
                "value": round(top_shift['analysis_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "category_pct_previous": {
                "value": round(top_shift['previous_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "category_revenue_analysis": {
                "value": round(top_shift['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "category_revenue_previous": {
                "value": round(top_shift['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} POS line items",
            f"Previous period: {len(previous_data)} POS line items",
            "Category data from POS cleaned artifact"
        ],
        "assumptions": [
            "Category field accurately reflects product classification",
            "line_total_sar represents net revenue per line item"
        ],
        "confidence": 0.90
    })

# Finding 3: Channel Mix Analysis
def get_channel_mix(data):
    channel_revenue = data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    channel_txns = data[~data['is_refund']].groupby('channel')['transaction_id'].nunique()
    total_revenue = channel_revenue.sum()
    channel_pct = (channel_revenue / total_revenue * 100).round(1)
    return channel_revenue, channel_pct, channel_txns

analysis_ch_rev, analysis_ch_pct, analysis_ch_txns = get_channel_mix(analysis_data)
previous_ch_rev, previous_ch_pct, previous_ch_txns = get_channel_mix(previous_data)

# Find significant channel shifts
channel_shifts = []
for ch in analysis_ch_pct.index:
    if ch in previous_ch_pct.index:
        shift = analysis_ch_pct[ch] - previous_ch_pct[ch]
        if abs(shift) > 2:  # More than 2% shift
            channel_shifts.append({
                'channel': ch,
                'analysis_pct': analysis_ch_pct[ch],
                'previous_pct': previous_ch_pct[ch],
                'shift': shift,
                'analysis_revenue': analysis_ch_rev[ch],
                'previous_revenue': previous_ch_rev[ch],
                'analysis_txns': analysis_ch_txns.get(ch, 0),
                'previous_txns': previous_ch_txns.get(ch, 0)
            })

if channel_shifts:
    # Sort by absolute shift
    channel_shifts.sort(key=lambda x: abs(x['shift']), reverse=True)
    top_shift = channel_shifts[0]
    
    findings.append({
        "title": "Channel Mix Shift",
        "claim": f"Channel '{top_shift['channel']}' represented {top_shift['analysis_pct']:.1f}% of revenue in analysis week vs {top_shift['previous_pct']:.1f}% in previous week, a shift of {top_shift['shift']:.1f} percentage points. Revenue was SAR {top_shift['analysis_revenue']:.2f} vs SAR {top_shift['previous_revenue']:.2f}, with {top_shift['analysis_txns']} transactions vs {top_shift['previous_txns']} previously.",
        "finding_type": "channel_mix_change",
        "metrics": {
            "channel_name": {
                "value": top_shift['channel'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "channel_pct_analysis": {
                "value": round(top_shift['analysis_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "channel_pct_previous": {
                "value": round(top_shift['previous_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "channel_revenue_analysis": {
                "value": round(top_shift['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "channel_revenue_previous": {
                "value": round(top_shift['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "channel_txns_analysis": {
                "value": top_shift['analysis_txns'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "channel_txns_previous": {
                "value": top_shift['previous_txns'],
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} POS line items",
            f"Previous period: {len(previous_data)} POS line items",
            "Channel data from POS cleaned artifact"
        ],
        "assumptions": [
            "Channel field accurately reflects sales channel",
            "line_total_sar represents net revenue per line item",
            "transaction_id uniquely identifies a basket"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
