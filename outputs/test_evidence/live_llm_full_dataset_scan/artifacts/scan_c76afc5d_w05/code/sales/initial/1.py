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
analysis_start = datetime.fromisoformat("2026-02-09T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-02-02T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-02-09T00:00:00+03:00")

trailing_baselines = [
    (datetime.fromisoformat("2026-02-02T00:00:00+03:00"), datetime.fromisoformat("2026-02-09T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-26T00:00:00+03:00"), datetime.fromisoformat("2026-02-02T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-19T00:00:00+03:00"), datetime.fromisoformat("2026-01-26T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-12T00:00:00+03:00"), datetime.fromisoformat("2026-01-19T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
def filter_by_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
previous_data = filter_by_period(pos_df, previous_start, previous_end)
trailing_data = [filter_by_period(pos_df, start, end) for start, end in trailing_baselines]

findings = []

# ============================================================================
# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
# ============================================================================

# Count valid transactions (unique transaction_id, excluding refunds for basket count)
analysis_baskets = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
previous_baskets = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()

# Calculate net revenue (line_total_sar includes refunds as negative)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Calculate refund impact
analysis_refunds = analysis_data[analysis_data['is_refund']]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund']]['line_total_sar'].sum()

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
basket_change = analysis_baskets - previous_baskets
basket_pct_change = (basket_change / previous_baskets * 100) if previous_baskets != 0 else 0

# AOV calculation
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if abs(revenue_pct_change) >= 5 or abs(basket_pct_change) >= 5:
    findings.append({
        "title": "Revenue and Transaction Volume Change (Week of 9-16 Feb vs 2-9 Feb)",
        "claim": f"Net revenue increased by SAR {revenue_change:.2f} ({revenue_pct_change:.1f}%) from SAR {previous_revenue:.2f} to SAR {analysis_revenue:.2f}. Valid transaction count changed by {basket_change} baskets ({basket_pct_change:.1f}%), from {previous_baskets} to {analysis_baskets}. Average order value changed by SAR {aov_change:.2f} ({aov_pct_change:.1f}%), from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}.",
        "finding_type": "revenue_and_transaction_mix",
        "metrics": {
            "net_revenue_analysis": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "net_revenue_previous": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "revenue_change_absolute": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "revenue_change_percent": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "valid_baskets_analysis": {
                "value": analysis_baskets,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "valid_baskets_previous": {
                "value": previous_baskets,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "basket_change_absolute": {
                "value": basket_change,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "basket_change_percent": {
                "value": round(basket_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "aov_analysis": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "aov_previous": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "aov_change_absolute": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "aov_change_percent": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "refunds_analysis": {
                "value": round(analysis_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "refunds_previous": {
                "value": round(previous_refunds, 2),
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
            f"Analysis period: {len(analysis_data)} POS line items across {analysis_baskets} valid baskets",
            f"Previous period: {len(previous_data)} POS line items across {previous_baskets} valid baskets",
            f"Refunds included in net revenue calculations: analysis={round(analysis_refunds, 2)} SAR, previous={round(previous_refunds, 2)} SAR"
        ],
        "assumptions": [
            "Valid transaction_id uniqueness defines basket count",
            "line_total_sar represents net realised revenue including refunds",
            "is_refund flag correctly identifies refund transactions",
            "Timestamps are accurate and in +03:00 timezone"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Category Mix Change (Analysis vs Previous Week)
# ============================================================================

# Join POS with menu to get category information
analysis_with_menu = analysis_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                         on='sku', how='left', suffixes=('', '_menu'))
previous_with_menu = previous_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                         on='sku', how='left', suffixes=('', '_menu'))

# Use category from menu if available, otherwise from POS
analysis_with_menu['category_final'] = analysis_with_menu['category_menu'].fillna(analysis_with_menu['category'])
previous_with_menu['category_final'] = previous_with_menu['category_menu'].fillna(previous_with_menu['category'])

# Calculate category revenue
analysis_category_revenue = analysis_with_menu.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_with_menu.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)

# Calculate category mix percentages
analysis_total_rev = analysis_with_menu['line_total_sar'].sum()
previous_total_rev = previous_with_menu['line_total_sar'].sum()

analysis_category_pct = (analysis_category_revenue / analysis_total_rev * 100) if analysis_total_rev != 0 else 0
previous_category_pct = (previous_category_revenue / previous_total_rev * 100) if previous_total_rev != 0 else 0

# Find largest mix shift
category_mix_changes = {}
for cat in set(list(analysis_category_pct.index) + list(previous_category_pct.index)):
    curr_pct = analysis_category_pct.get(cat, 0)
    prev_pct = previous_category_pct.get(cat, 0)
    pct_point_change = curr_pct - prev_pct
    category_mix_changes[cat] = {
        'current': curr_pct,
        'previous': prev_pct,
        'change': pct_point_change,
        'current_revenue': analysis_category_revenue.get(cat, 0),
        'previous_revenue': previous_category_revenue.get(cat, 0)
    }

# Sort by absolute change
sorted_changes = sorted(category_mix_changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)

if sorted_changes and abs(sorted_changes[0][1]['change']) >= 2:
    top_cat = sorted_changes[0][0]
    top_change = sorted_changes[0][1]
    
    findings.append({
        "title": f"Category Mix Shift: {top_cat}",
        "claim": f"Category '{top_cat}' revenue mix shifted by {top_change['change']:.1f} percentage points, from {top_change['previous']:.1f}% to {top_change['current']:.1f}% of total revenue. Absolute revenue changed from SAR {top_change['previous_revenue']:.2f} to SAR {top_change['current_revenue']:.2f}.",
        "finding_type": "product_category_mix",
        "metrics": {
            "category_revenue_analysis": {
                "value": round(top_change['current_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "category_revenue_previous": {
                "value": round(top_change['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "category_mix_percent_analysis": {
                "value": round(top_change['current'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "category_mix_percent_previous": {
                "value": round(top_change['previous'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "category_mix_change_percentage_points": {
                "value": round(top_change['change'], 1),
                "unit": "pp",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_with_menu),
        "coverage_notes": [
            f"Analysis period: {len(analysis_with_menu)} line items with category data",
            f"Previous period: {len(previous_with_menu)} line items with category data",
            f"Category data sourced from menu SKU reference where available"
        ],
        "assumptions": [
            "Menu SKU category reference is authoritative",
            "line_total_sar represents net revenue for category allocation",
            "All line items successfully matched to category"
        ],
        "confidence": 0.92
    })

# ============================================================================
# FINDING 3: Channel Mix Change (Analysis vs Previous Week)
# ============================================================================

analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_pct = (analysis_channel_revenue / analysis_total_rev * 100) if analysis_total_rev != 0 else 0
previous_channel_pct = (previous_channel_revenue / previous_total_rev * 100) if previous_total_rev != 0 else 0

# Find largest channel mix shift
channel_mix_changes = {}
for ch in set(list(analysis_channel_pct.index) + list(previous_channel_pct.index)):
    curr_pct = analysis_channel_pct.get(ch, 0)
    prev_pct = previous_channel_pct.get(ch, 0)
    pct_point_change = curr_pct - prev_pct
    channel_mix_changes[ch] = {
        'current': curr_pct,
        'previous': prev_pct,
        'change': pct_point_change,
        'current_revenue': analysis_channel_revenue.get(ch, 0),
        'previous_revenue': previous_channel_revenue.get(ch, 0)
    }

sorted_channel_changes = sorted(channel_mix_changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)

if sorted_channel_changes and abs(sorted_channel_changes[0][1]['change']) >= 2:
    top_ch = sorted_channel_changes[0][0]
    top_ch_change = sorted_channel_changes[0][1]
    
    findings.append({
        "title": f"Channel Mix Shift: {top_ch}",
        "claim": f"Channel '{top_ch}' revenue mix shifted by {top_ch_change['change']:.1f} percentage points, from {top_ch_change['previous']:.1f}% to {top_ch_change['current']:.1f}% of total revenue. Absolute revenue changed from SAR {top_ch_change['previous_revenue']:.2f} to SAR {top_ch_change['current_revenue']:.2f}.",
        "finding_type": "channel_mix",
        "metrics": {
            "channel_revenue_analysis": {
                "value": round(top_ch_change['current_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "channel_revenue_previous": {
                "value": round(top_ch_change['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "channel_mix_percent_analysis": {
                "value": round(top_ch_change['current'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "channel_mix_percent_previous": {
                "value": round(top_ch_change['previous'], 1),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "channel_mix_change_percentage_points": {
                "value": round(top_ch_change['change'], 1),
                "unit": "pp",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items across channels",
            f"Previous period: {len(previous_data)} line items across channels"
        ],
        "assumptions": [
            "channel field accurately represents sales channel",
            "line_total_sar represents net revenue for channel allocation"
        ],
        "confidence": 0.93
    })

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings[:3]
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)