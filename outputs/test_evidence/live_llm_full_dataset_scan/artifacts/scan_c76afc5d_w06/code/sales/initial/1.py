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
analysis_start = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-23T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-02-09T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
trailing_baselines = [
    (datetime.fromisoformat("2026-02-09T00:00:00+03:00"), datetime.fromisoformat("2026-02-16T00:00:00+03:00")),
    (datetime.fromisoformat("2026-02-02T00:00:00+03:00"), datetime.fromisoformat("2026-02-09T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-26T00:00:00+03:00"), datetime.fromisoformat("2026-02-02T00:00:00+03:00")),
    (datetime.fromisoformat("2026-01-19T00:00:00+03:00"), datetime.fromisoformat("2026-01-26T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter by period
def filter_by_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
previous_data = filter_by_period(pos_df, previous_start, previous_end)
trailing_data = [filter_by_period(pos_df, start, end) for start, end in trailing_baselines]

findings = []

# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
analysis_txns = analysis_data['transaction_id'].nunique()
previous_txns = previous_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

txn_change = analysis_txns - previous_txns
txn_pct_change = (txn_change / previous_txns * 100) if previous_txns > 0 else 0
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0

if abs(txn_pct_change) > 0 or abs(revenue_pct_change) > 0:
    findings.append({
        "title": "Transaction Count and Revenue Change Week-over-Week",
        "claim": f"Analysis period (2026-02-16 to 2026-02-23) shows {txn_change:+d} transactions ({txn_pct_change:+.1f}%) and {revenue_change:+.2f} SAR revenue ({revenue_pct_change:+.1f}%) vs previous week (2026-02-09 to 2026-02-16).",
        "finding_type": "revenue_and_transaction_change",
        "metrics": {
            "analysis_transaction_count": {
                "value": analysis_txns,
                "unit": "transactions",
                "numerator": analysis_txns,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "previous_transaction_count": {
                "value": previous_txns,
                "unit": "transactions",
                "numerator": previous_txns,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "transaction_count_change": {
                "value": txn_change,
                "unit": "transactions",
                "numerator": txn_change,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "transaction_count_pct_change": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": txn_change,
                "denominator": previous_txns,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": analysis_revenue,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "previous_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": previous_revenue,
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": revenue_change,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": revenue_change,
                "denominator": previous_revenue,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_txns,
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items from {analysis_txns} unique transactions",
            f"Previous period: {len(previous_data)} line items from {previous_txns} unique transactions",
            "Refunds included in net revenue calculations"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "line_total_sar represents net revenue after discounts",
            "Timestamp filtering uses UTC+3 timezone as provided"
        ],
        "confidence": 0.95
    })

# FINDING 2: Average Order Value Change
analysis_aov = analysis_revenue / analysis_txns if analysis_txns > 0 else 0
previous_aov = previous_revenue / previous_txns if previous_txns > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov > 0 else 0

if abs(aov_pct_change) > 0:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period is {analysis_aov:.2f} SAR vs {previous_aov:.2f} SAR in previous week, a change of {aov_change:+.2f} SAR ({aov_pct_change:+.1f}%).",
        "finding_type": "average_order_value_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": analysis_revenue,
                "denominator": analysis_txns,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": previous_revenue,
                "denominator": previous_txns,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": aov_change,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": aov_change,
                "denominator": previous_aov,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_txns,
        "coverage_notes": [
            f"Analysis period AOV calculated from {analysis_txns} transactions",
            f"Previous period AOV calculated from {previous_txns} transactions",
            "Refunds included in net revenue"
        ],
        "assumptions": [
            "AOV = total revenue / unique transaction count",
            "line_total_sar includes all discounts and refunds"
        ],
        "confidence": 0.95
    })

# FINDING 3: Category Mix Analysis (Analysis vs Previous Week)
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Find top category changes
category_changes = {}
for cat in set(list(analysis_category_revenue.index) + list(previous_category_revenue.index)):
    analysis_cat_rev = analysis_category_revenue.get(cat, 0)
    previous_cat_rev = previous_category_revenue.get(cat, 0)
    change = analysis_cat_rev - previous_cat_rev
    pct_change = (change / previous_cat_rev * 100) if previous_cat_rev > 0 else (100 if analysis_cat_rev > 0 else 0)
    category_changes[cat] = {
        'analysis': analysis_cat_rev,
        'previous': previous_cat_rev,
        'change': change,
        'pct_change': pct_change
    }

# Sort by absolute change
sorted_categories = sorted(category_changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)

if sorted_categories:
    top_cat = sorted_categories[0]
    cat_name = top_cat[0]
    cat_data = top_cat[1]
    
    findings.append({
        "title": f"Category Mix Shift: {cat_name}",
        "claim": f"Category '{cat_name}' revenue changed from {cat_data['previous']:.2f} SAR to {cat_data['analysis']:.2f} SAR ({cat_data['pct_change']:+.1f}%) between previous week and analysis period.",
        "finding_type": "category_mix_change",
        "metrics": {
            "analysis_category_revenue": {
                "value": round(cat_data['analysis'], 2),
                "unit": "SAR",
                "numerator": cat_data['analysis'],
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "previous_category_revenue": {
                "value": round(cat_data['previous'], 2),
                "unit": "SAR",
                "numerator": cat_data['previous'],
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "category_revenue_change": {
                "value": round(cat_data['change'], 2),
                "unit": "SAR",
                "numerator": cat_data['change'],
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "category_revenue_pct_change": {
                "value": round(cat_data['pct_change'], 2),
                "unit": "%",
                "numerator": cat_data['change'],
                "denominator": cat_data['previous'] if cat_data['previous'] > 0 else None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data[analysis_data['category'] == cat_name]),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data[analysis_data['category'] == cat_name])} line items in category '{cat_name}'",
            f"Previous period: {len(previous_data[previous_data['category'] == cat_name])} line items in category '{cat_name}'",
            "Refunds included in net revenue"
        ],
        "assumptions": [
            "Category assignment from cleaned POS data",
            "line_total_sar represents net revenue"
        ],
        "confidence": 0.90
    })

# Write output
result = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)