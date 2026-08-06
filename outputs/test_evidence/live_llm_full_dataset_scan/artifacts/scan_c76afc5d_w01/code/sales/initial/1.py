import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-01-12T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-19T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-01-05T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-01-12T00:00:00+03:00")

trailing_periods = [
    (datetime.fromisoformat("2026-01-05T00:00:00+03:00"), datetime.fromisoformat("2026-01-12T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-29T00:00:00+03:00"), datetime.fromisoformat("2026-01-05T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-22T00:00:00+03:00"), datetime.fromisoformat("2025-12-29T00:00:00+03:00")),
    (datetime.fromisoformat("2025-12-15T00:00:00+03:00"), datetime.fromisoformat("2025-12-22T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Trailing baseline (average of 4 weeks)
trailing_data = pos_df[(pos_df['timestamp'] >= trailing_periods[3][0]) & (pos_df['timestamp'] < trailing_periods[0][1])].copy()

findings = []

# Finding 1: Revenue and Transaction Count Change
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

previous_revenue = previous_data['line_total_sar'].sum()
previous_transactions = previous_data['transaction_id'].nunique()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
transaction_change = analysis_transactions - previous_transactions
transaction_pct_change = (transaction_change / previous_transactions * 100) if previous_transactions > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) > 2 or abs(transaction_pct_change) > 2:
    findings.append({
        "title": "Revenue and Transaction Performance vs Previous Week",
        "claim": f"Week of 2026-01-12 generated SAR {analysis_revenue:.2f} in net revenue across {analysis_transactions} transactions (AOV: SAR {analysis_aov:.2f}), representing a {revenue_pct_change:.1f}% change in revenue and {transaction_pct_change:.1f}% change in transaction count versus the previous week.",
        "finding_type": "revenue_and_transaction_analysis",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "transaction_change_count": {
                "value": transaction_change,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(transaction_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} POS line items from {analysis_transactions} unique transactions",
            f"Previous period: {len(previous_data)} POS line items from {previous_transactions} unique transactions",
            "Refunds included in net revenue calculations per metric definition"
        ],
        "assumptions": [
            "transaction_id uniqueness identifies distinct baskets",
            "line_total_sar represents net revenue after discounts",
            "Timestamp filtering uses UTC+3 timezone as specified"
        ],
        "confidence": 0.95
    })

# Finding 2: Category Mix Analysis
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Calculate category mix percentages
analysis_total = analysis_category_revenue.sum()
previous_total = previous_category_revenue.sum()

analysis_category_pct = (analysis_category_revenue / analysis_total * 100).to_dict()
previous_category_pct = (previous_category_revenue / previous_total * 100).to_dict()

# Find significant category shifts
category_shifts = {}
for cat in set(list(analysis_category_pct.keys()) + list(previous_category_pct.keys())):
    analysis_pct = analysis_category_pct.get(cat, 0)
    previous_pct = previous_category_pct.get(cat, 0)
    shift = analysis_pct - previous_pct
    if abs(shift) > 1:  # More than 1% shift
        category_shifts[cat] = {
            'analysis_pct': analysis_pct,
            'previous_pct': previous_pct,
            'shift': shift,
            'analysis_revenue': analysis_category_revenue.get(cat, 0),
            'previous_revenue': previous_category_revenue.get(cat, 0)
        }

if category_shifts:
    top_shift_cat = max(category_shifts.items(), key=lambda x: abs(x[1]['shift']))
    cat_name = top_shift_cat[0]
    cat_data = top_shift_cat[1]
    
    findings.append({
        "title": "Category Mix Shift",
        "claim": f"Category '{cat_name}' shifted from {cat_data['previous_pct']:.1f}% to {cat_data['analysis_pct']:.1f}% of revenue mix (change: {cat_data['shift']:+.1f}%), with analysis period revenue of SAR {cat_data['analysis_revenue']:.2f} versus SAR {cat_data['previous_revenue']:.2f} in previous week.",
        "finding_type": "category_mix_analysis",
        "metrics": {
            "analysis_period_category_revenue": {
                "value": round(cat_data['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_category_revenue": {
                "value": round(cat_data['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "analysis_period_category_mix_pct": {
                "value": round(cat_data['analysis_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_category_mix_pct": {
                "value": round(cat_data['previous_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "category_mix_shift_pct": {
                "value": round(cat_data['shift'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items across {len(analysis_category_revenue)} categories",
            f"Previous period: {len(previous_data)} line items across {len(previous_category_revenue)} categories",
            "Mix percentages calculated from net revenue (line_total_sar)"
        ],
        "assumptions": [
            "Category field is populated and consistent across periods",
            "Revenue mix analysis uses line_total_sar including refunds"
        ],
        "confidence": 0.90
    })

# Finding 3: Channel Mix Analysis
analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_total = analysis_channel_revenue.sum()
previous_channel_total = previous_channel_revenue.sum()

analysis_channel_pct = (analysis_channel_revenue / analysis_channel_total * 100).to_dict()
previous_channel_pct = (previous_channel_revenue / previous_channel_total * 100).to_dict()

channel_shifts = {}
for ch in set(list(analysis_channel_pct.keys()) + list(previous_channel_pct.keys())):
    analysis_pct = analysis_channel_pct.get(ch, 0)
    previous_pct = previous_channel_pct.get(ch, 0)
    shift = analysis_pct - previous_pct
    if abs(shift) > 1:  # More than 1% shift
        channel_shifts[ch] = {
            'analysis_pct': analysis_pct,
            'previous_pct': previous_pct,
            'shift': shift,
            'analysis_revenue': analysis_channel_revenue.get(ch, 0),
            'previous_revenue': previous_channel_revenue.get(ch, 0)
        }

if channel_shifts:
    top_shift_ch = max(channel_shifts.items(), key=lambda x: abs(x[1]['shift']))
    ch_name = top_shift_ch[0]
    ch_data = top_shift_ch[1]
    
    findings.append({
        "title": "Channel Mix Shift",
        "claim": f"Channel '{ch_name}' shifted from {ch_data['previous_pct']:.1f}% to {ch_data['analysis_pct']:.1f}% of revenue mix (change: {ch_data['shift']:+.1f}%), with analysis period revenue of SAR {ch_data['analysis_revenue']:.2f} versus SAR {ch_data['previous_revenue']:.2f} in previous week.",
        "finding_type": "channel_mix_analysis",
        "metrics": {
            "analysis_period_channel_revenue": {
                "value": round(ch_data['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_channel_revenue": {
                "value": round(ch_data['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "analysis_period_channel_mix_pct": {
                "value": round(ch_data['analysis_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "previous_period_channel_mix_pct": {
                "value": round(ch_data['previous_pct'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "channel_mix_shift_pct": {
                "value": round(ch_data['shift'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items across {len(analysis_channel_revenue)} channels",
            f"Previous period: {len(previous_data)} line items across {len(previous_channel_revenue)} channels",
            "Mix percentages calculated from net revenue (line_total_sar)"
        ],
        "assumptions": [
            "Channel field is populated and consistent across periods",
            "Revenue mix analysis uses line_total_sar including refunds"
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
