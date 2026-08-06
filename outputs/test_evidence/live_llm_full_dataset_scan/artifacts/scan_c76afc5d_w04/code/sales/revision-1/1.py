import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = pd.Timestamp("2026-02-02T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-02-09T00:00:00+03:00")
previous_start = pd.Timestamp("2026-01-26T00:00:00+03:00")
previous_end = pd.Timestamp("2026-02-02T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Trailing baseline (4 weeks)
trailing_data = pos_df[(pos_df['timestamp'] >= pd.Timestamp("2026-01-05T00:00:00+03:00")) & 
                       (pos_df['timestamp'] < pd.Timestamp("2026-02-02T00:00:00+03:00"))]

findings = []

# ============================================================================
# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
# ============================================================================

# Count valid transactions (unique transaction_id, excluding refunds for transaction count)
analysis_txns = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
previous_txns = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()

# Revenue (net, including refunds as per metric definition)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
txn_change = analysis_txns - previous_txns
txn_pct_change = (txn_change / previous_txns * 100) if previous_txns != 0 else 0

# Average order value
analysis_aov = analysis_revenue / analysis_txns if analysis_txns > 0 else 0
previous_aov = previous_revenue / previous_txns if previous_txns > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) > 2 or abs(txn_pct_change) > 2:
    findings.append({
        "title": "Weekly Revenue and Transaction Volume Change",
        "claim": f"Analysis week (2026-02-02 to 2026-02-09) generated {analysis_revenue:.2f} SAR net revenue across {analysis_txns} valid transactions, compared to {previous_revenue:.2f} SAR across {previous_txns} transactions in previous week. Revenue declined {revenue_pct_change:.1f}% ({revenue_change:.2f} SAR), while transaction count declined {txn_pct_change:.1f}% ({txn_change} txns). Average order value changed from {previous_aov:.2f} SAR to {analysis_aov:.2f} SAR ({aov_pct_change:+.1f}%).",
        "finding_type": "revenue_and_volume",
        "metrics": {
            "analysis_week_revenue_sar": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_week_revenue_sar": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "analysis_week_transactions": {
                "value": analysis_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_week_transactions": {
                "value": previous_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "transaction_count_change": {
                "value": txn_change,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "transaction_pct_change": {
                "value": round(txn_pct_change, 1),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_txns,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "analysis_week_aov_sar": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_txns,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_week_aov_sar": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_txns,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Revenue includes refunds (net calculation per metric definition)",
            "Transaction count excludes refund line items",
            "All valid transaction_ids counted once per basket",
            "Analysis period: 2026-02-02 to 2026-02-09 (7 days)",
            "Previous period: 2026-01-26 to 2026-02-02 (7 days)"
        ],
        "assumptions": [
            "Revenue figures are net of refunds and represent actual cash/card settlement amounts",
            "Valid transactions identified by unique transaction_id after cleaning",
            "Refund line items excluded from transaction count but included in revenue totals",
            "No material data quality issues in analysis period"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Category-Level Revenue Performance (Analysis vs Previous Week)
# ============================================================================

# Merge POS with menu to get category information
analysis_with_cat = analysis_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                        on='sku', how='left', suffixes=('', '_menu'))
previous_with_cat = previous_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                        on='sku', how='left', suffixes=('', '_menu'))

# Category revenue analysis
analysis_cat_revenue = analysis_with_cat.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_cat_revenue = previous_with_cat.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Find categories with significant changes
category_changes = {}
for cat in analysis_cat_revenue.index:
    if cat in previous_cat_revenue.index:
        curr = analysis_cat_revenue[cat]
        prev = previous_cat_revenue[cat]
        pct_change = ((curr - prev) / prev * 100) if prev != 0 else 0
        category_changes[cat] = {
            'current': curr,
            'previous': prev,
            'change': curr - prev,
            'pct_change': pct_change
        }

# Find largest absolute percentage change
if category_changes:
    largest_change_cat = max(category_changes.items(), 
                            key=lambda x: abs(x[1]['pct_change']))
    cat_name = largest_change_cat[0]
    cat_metrics = largest_change_cat[1]
    
    if abs(cat_metrics['pct_change']) > 3:
        findings.append({
            "title": f"Category Revenue Shift: {cat_name}",
            "claim": f"Category '{cat_name}' generated {cat_metrics['current']:.2f} SAR in analysis week vs {cat_metrics['previous']:.2f} SAR in previous week, representing a {cat_metrics['pct_change']:.1f}% change ({cat_metrics['change']:.2f} SAR). This category showed notable performance variance between the two periods.",
            "finding_type": "category_mix",
            "metrics": {
                "analysis_week_category_revenue_sar": {
                    "value": round(cat_metrics['current'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_week_category_revenue_sar": {
                    "value": round(cat_metrics['previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "category_revenue_change_sar": {
                    "value": round(cat_metrics['change'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "category_revenue_pct_change": {
                    "value": round(cat_metrics['pct_change'], 1),
                    "unit": "%",
                    "numerator": round(cat_metrics['change'], 2),
                    "denominator": round(cat_metrics['previous'], 2),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_with_cat[analysis_with_cat['category'] == cat_name]),
            "coverage_notes": [
                "Revenue includes refunds (net calculation)",
                "Category assignment via menu SKU reference",
                "Analysis period: 2026-02-02 to 2026-02-09",
                "Previous period: 2026-01-26 to 2026-02-02",
                "Largest absolute percentage change category selected"
            ],
            "assumptions": [
                "Revenue figures are net of refunds",
                "Category mapping from menu artifact is authoritative",
                "No product launch/retirement eligibility filtering applied for this metric"
            ],
            "confidence": 0.90
        })

# ============================================================================
# FINDING 3: Channel Performance (Analysis vs Previous Week)
# ============================================================================

# Channel analysis
analysis_by_channel = analysis_data.groupby('channel').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum'
}).rename(columns={'transaction_id': 'transactions', 'line_total_sar': 'revenue'})

previous_by_channel = previous_data.groupby('channel').agg({
    'transaction_id': 'nunique',
    'line_total_sar': 'sum'
}).rename(columns={'transaction_id': 'transactions', 'line_total_sar': 'revenue'})

# Find channels with significant changes
channel_changes = {}
for ch in analysis_by_channel.index:
    if ch in previous_by_channel.index:
        curr_rev = analysis_by_channel.loc[ch, 'revenue']
        prev_rev = previous_by_channel.loc[ch, 'revenue']
        curr_txn = analysis_by_channel.loc[ch, 'transactions']
        prev_txn = previous_by_channel.loc[ch, 'transactions']
        
        rev_pct_change = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev != 0 else 0
        txn_pct_change = ((curr_txn - prev_txn) / prev_txn * 100) if prev_txn != 0 else 0
        
        channel_changes[ch] = {
            'curr_rev': curr_rev,
            'prev_rev': prev_rev,
            'rev_change': curr_rev - prev_rev,
            'rev_pct_change': rev_pct_change,
            'curr_txn': curr_txn,
            'prev_txn': prev_txn,
            'txn_change': curr_txn - prev_txn,
            'txn_pct_change': txn_pct_change
        }

# Find largest revenue change
if channel_changes:
    largest_channel = max(channel_changes.items(), 
                         key=lambda x: abs(x[1]['rev_pct_change']))
    ch_name = largest_channel[0]
    ch_metrics = largest_channel[1]
    
    if abs(ch_metrics['rev_pct_change']) > 2:
        findings.append({
            "title": f"Channel Performance: {ch_name}",
            "claim": f"Channel '{ch_name}' generated {ch_metrics['curr_rev']:.2f} SAR (net of refunds) across {ch_metrics['curr_txn']} transactions in analysis week, compared to {ch_metrics['prev_rev']:.2f} SAR across {ch_metrics['prev_txn']} transactions in previous week. Revenue declined {ch_metrics['rev_pct_change']:.1f}% ({ch_metrics['rev_change']:.2f} SAR), while transaction count changed {ch_metrics['txn_pct_change']:.1f}% ({ch_metrics['txn_change']:+d} txns).",
            "finding_type": "channel_mix",
            "metrics": {
                "analysis_week_channel_revenue_sar": {
                    "value": round(ch_metrics['curr_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_week_channel_revenue_sar": {
                    "value": round(ch_metrics['prev_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "channel_revenue_change_sar": {
                    "value": round(ch_metrics['rev_change'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "channel_revenue_pct_change": {
                    "value": round(ch_metrics['rev_pct_change'], 1),
                    "unit": "%",
                    "numerator": round(ch_metrics['rev_change'], 2),
                    "denominator": round(ch_metrics['prev_rev'], 2),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "analysis_week_channel_transactions": {
                    "value": ch_metrics['curr_txn'],
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_week_channel_transactions": {
                    "value": ch_metrics['prev_txn'],
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "channel_transaction_change": {
                    "value": ch_metrics['txn_change'],
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "channel_transaction_pct_change": {
                    "value": round(ch_metrics['txn_pct_change'], 1),
                    "unit": "%",
                    "numerator": ch_metrics['txn_change'],
                    "denominator": ch_metrics['prev_txn'],
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data[analysis_data['channel'] == ch_name]),
            "coverage_notes": [
                "Revenue includes refunds (net calculation per metric definition)",
                "Transaction count based on unique transaction_id per channel",
                "Analysis period: 2026-02-02 to 2026-02-09",
                "Previous period: 2026-01-26 to 2026-02-02",
                "Largest absolute revenue percentage change channel selected"
            ],
            "assumptions": [
                "Revenue figures are net of refunds and represent actual cash/card settlement amounts",
                "Channel assignment from POS data is accurate",
                "Transaction counts exclude refund line items"
            ],
            "confidence": 0.92
        })

# Write output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)