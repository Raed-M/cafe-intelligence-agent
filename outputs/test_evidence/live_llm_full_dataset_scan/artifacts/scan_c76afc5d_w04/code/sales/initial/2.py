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

# Parse periods
analysis_start = datetime.fromisoformat("2026-02-02T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-09T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-01-26T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-02-02T00:00:00+03:00")

trailing_periods = [
    (datetime.fromisoformat("2026-01-26T00:00:00+03:00"), datetime.fromisoformat("2026-02-02T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-19T00:00:00+03:00"), datetime.fromisoformat("2026-01-26T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-12T00:00:00+03:00"), datetime.fromisoformat("2026-01-19T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-05T00:00:00+03:00"), datetime.fromisoformat("2026-01-12T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Trailing baseline: average of all 4 weeks
trailing_data = pos_df[(pos_df['timestamp'] >= trailing_periods[0][0]) & (pos_df['timestamp'] < trailing_periods[-1][1])].copy()

findings = []

# ===== FINDING 1: Revenue Change Analysis =====
# Calculate net revenue (including refunds)
analysis_revenue = float(analysis_data['line_total_sar'].sum())
previous_revenue = float(previous_data['line_total_sar'].sum())
trailing_revenue = float(trailing_data['line_total_sar'].sum())
trailing_avg_revenue = trailing_revenue / 4

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

# Count valid transactions (unique transaction_id)
analysis_txns = int(analysis_data['transaction_id'].nunique())
previous_txns = int(previous_data['transaction_id'].nunique())
trailing_txns = int(trailing_data['transaction_id'].nunique())
trailing_avg_txns = trailing_txns / 4

txn_change = analysis_txns - previous_txns
txn_pct_change = (txn_change / previous_txns * 100) if previous_txns != 0 else 0

# Calculate AOV
analysis_aov = analysis_revenue / analysis_txns if analysis_txns > 0 else 0
previous_aov = previous_revenue / previous_txns if previous_txns > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) >= 5 or abs(aov_pct_change) >= 5:
    findings.append({
        "title": "Revenue and AOV Performance vs Previous Week",
        "claim": f"Analysis week (Feb 2-9, 2026) generated {analysis_revenue:,.0f} SAR across {analysis_txns} transactions with AOV of {analysis_aov:.2f} SAR, compared to previous week's {previous_revenue:,.0f} SAR ({previous_txns} txns, AOV {previous_aov:.2f} SAR). Net change: {revenue_change:+,.0f} SAR ({revenue_pct_change:+.1f}%), AOV change {aov_pct_change:+.1f}%.",
        "finding_type": "revenue_and_transaction_metrics",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_period_revenue": {
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
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "analysis_period_transactions": {
                "value": int(analysis_txns),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": int(previous_txns),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(txn_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Analysis period: 2026-02-02 to 2026-02-09",
            "Previous period: 2026-01-26 to 2026-02-02",
            "Revenue includes refunds (net calculation)",
            "Transactions counted as unique transaction_id values"
        ],
        "assumptions": [
            "line_total_sar represents net realized revenue including refunds",
            "transaction_id uniquely identifies a basket",
            "All rows in analysis period are valid transactions"
        ],
        "confidence": 0.95
    })

# ===== FINDING 2: Category Mix Analysis =====
# Join POS with menu to get category information and launch dates
analysis_with_menu = analysis_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
previous_with_menu = previous_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')

# Filter for products that were active during each period
analysis_with_menu['launch_date'] = pd.to_datetime(analysis_with_menu['launch_date'], errors='coerce')
analysis_with_menu['retire_date'] = pd.to_datetime(analysis_with_menu['retire_date'], errors='coerce')
previous_with_menu['launch_date'] = pd.to_datetime(previous_with_menu['launch_date'], errors='coerce')
previous_with_menu['retire_date'] = pd.to_datetime(previous_with_menu['retire_date'], errors='coerce')

# Category revenue analysis
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Find top category changes
category_changes = {}
for cat in analysis_category_revenue.index:
    analysis_rev = float(analysis_category_revenue.get(cat, 0))
    previous_rev = float(previous_category_revenue.get(cat, 0))
    if previous_rev != 0:
        pct_change = (analysis_rev - previous_rev) / previous_rev * 100
        category_changes[cat] = {
            'analysis': analysis_rev,
            'previous': previous_rev,
            'change': analysis_rev - previous_rev,
            'pct_change': pct_change
        }

# Find most significant category change
if category_changes:
    top_category = max(category_changes.items(), key=lambda x: abs(x[1]['pct_change']))
    cat_name = top_category[0]
    cat_metrics = top_category[1]
    
    if abs(cat_metrics['pct_change']) >= 5:
        findings.append({
            "title": f"Category Mix Shift: {cat_name}",
            "claim": f"Category '{cat_name}' generated {cat_metrics['analysis']:,.0f} SAR in analysis week vs {cat_metrics['previous']:,.0f} SAR in previous week, a change of {cat_metrics['pct_change']:+.1f}%. This represents the most significant category-level shift between periods.",
            "finding_type": "category_mix_analysis",
            "metrics": {
                "analysis_period_category_revenue": {
                    "value": round(cat_metrics['analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_period_category_revenue": {
                    "value": round(cat_metrics['previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "category_revenue_change_pct": {
                    "value": round(cat_metrics['pct_change'], 1),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "category_revenue_change_sar": {
                    "value": round(cat_metrics['change'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period category '{cat_name}' revenue: {len(analysis_data[analysis_data['category'] == cat_name])} line items",
                f"Previous period category '{cat_name}' revenue: {len(previous_data[previous_data['category'] == cat_name])} line items",
                "Revenue includes refunds (net calculation)"
            ],
            "assumptions": [
                "Category field in POS is accurate and consistent",
                "line_total_sar represents net realized revenue",
                "Category comparison is valid across both periods"
            ],
            "confidence": 0.90
        })

# ===== FINDING 3: Channel Mix Analysis =====
analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_txns = analysis_data.groupby('channel')['transaction_id'].nunique()
previous_channel_txns = previous_data.groupby('channel')['transaction_id'].nunique()

channel_changes = {}
for ch in analysis_channel_revenue.index:
    analysis_rev = float(analysis_channel_revenue.get(ch, 0))
    previous_rev = float(previous_channel_revenue.get(ch, 0))
    analysis_txn = int(analysis_channel_txns.get(ch, 0))
    previous_txn = int(previous_channel_txns.get(ch, 0))
    
    if previous_rev != 0:
        rev_pct_change = (analysis_rev - previous_rev) / previous_rev * 100
        txn_pct_change = (analysis_txn - previous_txn) / previous_txn * 100 if previous_txn > 0 else 0
        channel_changes[ch] = {
            'analysis_rev': analysis_rev,
            'previous_rev': previous_rev,
            'rev_change': analysis_rev - previous_rev,
            'rev_pct_change': rev_pct_change,
            'analysis_txn': analysis_txn,
            'previous_txn': previous_txn,
            'txn_change': analysis_txn - previous_txn,
            'txn_pct_change': txn_pct_change
        }

if channel_changes:
    top_channel = max(channel_changes.items(), key=lambda x: abs(x[1]['rev_pct_change']))
    ch_name = top_channel[0]
    ch_metrics = top_channel[1]
    
    if abs(ch_metrics['rev_pct_change']) >= 5:
        findings.append({
            "title": f"Channel Performance: {ch_name}",
            "claim": f"Channel '{ch_name}' generated {ch_metrics['analysis_rev']:,.0f} SAR across {ch_metrics['analysis_txn']} transactions in analysis week vs {ch_metrics['previous_rev']:,.0f} SAR ({ch_metrics['previous_txn']} txns) in previous week. Revenue change: {ch_metrics['rev_pct_change']:+.1f}%, transaction change: {ch_metrics['txn_pct_change']:+.1f}%.",
            "finding_type": "channel_mix_analysis",
            "metrics": {
                "analysis_period_channel_revenue": {
                    "value": round(ch_metrics['analysis_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_period_channel_revenue": {
                    "value": round(ch_metrics['previous_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "channel_revenue_change_pct": {
                    "value": round(ch_metrics['rev_pct_change'], 1),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "analysis_period_channel_transactions": {
                    "value": int(ch_metrics['analysis_txn']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "previous_period_channel_transactions": {
                    "value": int(ch_metrics['previous_txn']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "channel_transaction_change_pct": {
                    "value": round(ch_metrics['txn_pct_change'], 1),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period channel '{ch_name}': {len(analysis_data[analysis_data['channel'] == ch_name])} line items",
                f"Previous period channel '{ch_name}': {len(previous_data[previous_data['channel'] == ch_name])} line items",
                "Revenue includes refunds (net calculation)",
                "Transactions counted as unique transaction_id per channel"
            ],
            "assumptions": [
                "Channel field in POS is accurate and consistent",
                "line_total_sar represents net realized revenue",
                "transaction_id uniquely identifies a basket within a channel"
            ],
            "confidence": 0.90
        })

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
