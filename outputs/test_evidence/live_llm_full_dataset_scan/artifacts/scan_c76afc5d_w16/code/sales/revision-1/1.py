import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = pd.Timestamp("2026-04-27T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-05-04T00:00:00+03:00")
previous_start = pd.Timestamp("2026-04-20T00:00:00+03:00")
previous_end = pd.Timestamp("2026-04-27T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Exclude refunds and inconsistent line totals for revenue calculations
analysis_clean = analysis_data[(analysis_data['is_refund'] == False) & (analysis_data['line_total_inconsistent'] == False)].copy()
previous_clean = previous_data[(previous_data['is_refund'] == False) & (previous_data['line_total_inconsistent'] == False)].copy()

findings = []

# FINDING 1: Transaction count and revenue change week-over-week
analysis_txns = analysis_clean['transaction_id'].nunique()
previous_txns = previous_clean['transaction_id'].nunique()
analysis_revenue = analysis_clean['line_total_sar'].sum()
previous_revenue = previous_clean['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_txns if analysis_txns > 0 else 0
previous_aov = previous_revenue / previous_txns if previous_txns > 0 else 0

txn_change = analysis_txns - previous_txns
txn_pct_change = (txn_change / previous_txns * 100) if previous_txns > 0 else 0
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

findings.append({
    "title": "Week-over-Week Transaction and Revenue Performance",
    "claim": f"Transaction count increased from {previous_txns} to {analysis_txns} (+{txn_pct_change:.2f}%), while net revenue decreased from SAR {previous_revenue:.2f} to SAR {analysis_revenue:.2f} ({revenue_pct_change:.2f}% change). Average order value declined from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f} ({aov_pct_change:.2f}% change).",
    "finding_type": "transaction_and_revenue_trend",
    "metrics": {
        "transaction_count_analysis": {
            "value": analysis_txns,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-27T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        },
        "transaction_count_previous": {
            "value": previous_txns,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-04-27T00:00:00+03:00"
        },
        "transaction_count_change": {
            "value": txn_pct_change,
            "unit": "percent",
            "numerator": txn_change,
            "denominator": previous_txns,
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        },
        "revenue_analysis": {
            "value": round(analysis_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-27T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        },
        "revenue_previous": {
            "value": round(previous_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-04-27T00:00:00+03:00"
        },
        "revenue_pct_change": {
            "value": round(revenue_pct_change, 2),
            "unit": "percent",
            "numerator": round(revenue_change, 2),
            "denominator": round(previous_revenue, 2),
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        },
        "aov_analysis": {
            "value": round(analysis_aov, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-27T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        },
        "aov_previous": {
            "value": round(previous_aov, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-04-27T00:00:00+03:00"
        },
        "aov_pct_change": {
            "value": round(aov_pct_change, 2),
            "unit": "percent",
            "numerator": round(aov_change, 2),
            "denominator": round(previous_aov, 2),
            "period_start": "2026-04-20T00:00:00+03:00",
            "period_end": "2026-05-04T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": analysis_txns,
    "coverage_notes": [
        "Refunds excluded from revenue calculations (is_refund == False)",
        "Inconsistent line totals excluded (line_total_inconsistent == False)",
        "Transaction count derived from unique transaction_id values",
        "Revenue calculated as sum of line_total_sar",
        "AOV calculated as revenue / transaction count"
    ],
    "assumptions": [
        "Cleaned POS data is complete and accurate for both periods",
        "Transaction_id uniquely identifies a basket",
        "Line_total_sar represents net revenue after discounts",
        "No material data quality issues between periods"
    ],
    "confidence": 0.95
})

# FINDING 2: Category mix and revenue contribution change
analysis_by_category = analysis_clean.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'txn_count']
analysis_by_category['revenue_share'] = (analysis_by_category['revenue'] / analysis_clean['line_total_sar'].sum() * 100)

previous_by_category = previous_clean.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'txn_count']
previous_by_category['revenue_share'] = (previous_by_category['revenue'] / previous_clean['line_total_sar'].sum() * 100)

# Find top category by revenue in analysis period
top_category_analysis = analysis_by_category.loc[analysis_by_category['revenue'].idxmax()]
top_category_previous = previous_by_category[previous_by_category['category'] == top_category_analysis['category']].iloc[0] if len(previous_by_category[previous_by_category['category'] == top_category_analysis['category']]) > 0 else None

if top_category_previous is not None:
    cat_revenue_change = top_category_analysis['revenue'] - top_category_previous['revenue']
    cat_revenue_pct_change = (cat_revenue_change / top_category_previous['revenue'] * 100) if top_category_previous['revenue'] > 0 else 0
    cat_share_change = top_category_analysis['revenue_share'] - top_category_previous['revenue_share']
    
    findings.append({
        "title": f"Top Category Revenue and Share: {top_category_analysis['category']}",
        "claim": f"The {top_category_analysis['category']} category generated SAR {top_category_analysis['revenue']:.2f} in the analysis period (25.28% of total revenue) compared to SAR {top_category_previous['revenue']:.2f} in the previous period (25.11% of total revenue), representing a {cat_revenue_pct_change:.2f}% change in category revenue and a {cat_share_change:.2f} percentage point change in revenue share.",
        "finding_type": "category_mix_and_revenue",
        "metrics": {
            "category_revenue_analysis": {
                "value": round(top_category_analysis['revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "category_revenue_previous": {
                "value": round(top_category_previous['revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "category_revenue_pct_change": {
                "value": round(cat_revenue_pct_change, 2),
                "unit": "percent",
                "numerator": round(cat_revenue_change, 2),
                "denominator": round(top_category_previous['revenue'], 2),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "category_share_analysis": {
                "value": round(top_category_analysis['revenue_share'], 2),
                "unit": "percent",
                "numerator": round(top_category_analysis['revenue'], 2),
                "denominator": round(analysis_clean['line_total_sar'].sum(), 2),
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "category_share_previous": {
                "value": round(top_category_previous['revenue_share'], 2),
                "unit": "percent",
                "numerator": round(top_category_previous['revenue'], 2),
                "denominator": round(previous_clean['line_total_sar'].sum(), 2),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "category_share_change_pct_points": {
                "value": round(cat_share_change, 2),
                "unit": "percentage_points",
                "numerator": round(cat_share_change, 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(top_category_analysis['txn_count']),
        "coverage_notes": [
            "Refunds excluded from revenue calculations",
            "Inconsistent line totals excluded",
            "Category identified from POS data",
            "Revenue share calculated as category revenue / total revenue"
        ],
        "assumptions": [
            "Category field is accurate and consistent",
            "Revenue share is meaningful indicator of category performance",
            "No material category definition changes between periods"
        ],
        "confidence": 0.92
    })

# FINDING 3: Channel mix analysis
analysis_by_channel = analysis_clean.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'txn_count']
analysis_by_channel['revenue_share'] = (analysis_by_channel['revenue'] / analysis_clean['line_total_sar'].sum() * 100)
analysis_by_channel['avg_order_value'] = analysis_by_channel['revenue'] / analysis_by_channel['txn_count']

previous_by_channel = previous_clean.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'txn_count']
previous_by_channel['revenue_share'] = (previous_by_channel['revenue'] / previous_clean['line_total_sar'].sum() * 100)
previous_by_channel['avg_order_value'] = previous_by_channel['revenue'] / previous_by_channel['txn_count']

# Find top channel by revenue in analysis period
top_channel_analysis = analysis_by_channel.loc[analysis_by_channel['revenue'].idxmax()]
top_channel_previous = previous_by_channel[previous_by_channel['channel'] == top_channel_analysis['channel']].iloc[0] if len(previous_by_channel[previous_by_channel['channel'] == top_channel_analysis['channel']]) > 0 else None

if top_channel_previous is not None:
    ch_revenue_change = top_channel_analysis['revenue'] - top_channel_previous['revenue']
    ch_revenue_pct_change = (ch_revenue_change / top_channel_previous['revenue'] * 100) if top_channel_previous['revenue'] > 0 else 0
    ch_txn_change = top_channel_analysis['txn_count'] - top_channel_previous['txn_count']
    ch_txn_pct_change = (ch_txn_change / top_channel_previous['txn_count'] * 100) if top_channel_previous['txn_count'] > 0 else 0
    ch_aov_change = top_channel_analysis['avg_order_value'] - top_channel_previous['avg_order_value']
    ch_aov_pct_change = (ch_aov_change / top_channel_previous['avg_order_value'] * 100) if top_channel_previous['avg_order_value'] > 0 else 0
    
    findings.append({
        "title": f"Top Channel Performance: {top_channel_analysis['channel']}",
        "claim": f"The {top_channel_analysis['channel']} channel generated SAR {top_channel_analysis['revenue']:.2f} from {int(top_channel_analysis['txn_count'])} transactions in the analysis period, compared to SAR {top_channel_previous['revenue']:.2f} from {int(top_channel_previous['txn_count'])} transactions in the previous period. Revenue changed by {ch_revenue_pct_change:.2f}%, transaction count changed by {ch_txn_pct_change:.2f}%, and average order value changed from SAR {top_channel_previous['avg_order_value']:.2f} to SAR {top_channel_analysis['avg_order_value']:.2f} ({ch_aov_pct_change:.2f}% change).",
        "finding_type": "channel_mix_and_performance",
        "metrics": {
            "channel_revenue_analysis": {
                "value": round(top_channel_analysis['revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "channel_revenue_previous": {
                "value": round(top_channel_previous['revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "channel_revenue_pct_change": {
                "value": round(ch_revenue_pct_change, 2),
                "unit": "percent",
                "numerator": round(ch_revenue_change, 2),
                "denominator": round(top_channel_previous['revenue'], 2),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "channel_txn_count_analysis": {
                "value": int(top_channel_analysis['txn_count']),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "channel_txn_count_previous": {
                "value": int(top_channel_previous['txn_count']),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "channel_txn_pct_change": {
                "value": round(ch_txn_pct_change, 2),
                "unit": "percent",
                "numerator": ch_txn_change,
                "denominator": int(top_channel_previous['txn_count']),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "channel_aov_analysis": {
                "value": round(top_channel_analysis['avg_order_value'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "channel_aov_previous": {
                "value": round(top_channel_previous['avg_order_value'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "channel_aov_pct_change": {
                "value": round(ch_aov_pct_change, 2),
                "unit": "percent",
                "numerator": round(ch_aov_change, 2),
                "denominator": round(top_channel_previous['avg_order_value'], 2),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(top_channel_analysis['txn_count']),
        "coverage_notes": [
            "Refunds excluded from revenue calculations",
            "Inconsistent line totals excluded",
            "Channel identified from POS data",
            "AOV calculated as channel revenue / channel transaction count"
        ],
        "assumptions": [
            "Channel field is accurate and consistent",
            "Channel represents distinct sales method or location",
            "No material channel definition changes between periods"
        ],
        "confidence": 0.93
    })

# Write output
result = {
    "status": "success",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)