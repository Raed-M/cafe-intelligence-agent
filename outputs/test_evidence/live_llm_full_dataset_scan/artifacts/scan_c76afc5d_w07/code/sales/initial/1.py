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
analysis_start = datetime.fromisoformat("2026-02-23T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-02T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-02-23T00:00:00+03:00")

trailing_periods = [
    ("2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00"),
    ("2026-02-09T00:00:00+03:00", "2026-02-16T00:00:00+03:00"),
    ("2026-02-02T00:00:00+03:00", "2026-02-09T00:00:00+03:00"),
    ("2026-01-26T00:00:00+03:00", "2026-02-02T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
def filter_by_period(df, start_str, end_str):
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

analysis_data = filter_by_period(pos_df, "2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00")
previous_data = filter_by_period(pos_df, "2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00")

# Calculate metrics for analysis period
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()
analysis_net_revenue = analysis_revenue + analysis_refunds  # refunds are negative

# Calculate metrics for previous period
previous_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()
previous_net_revenue = previous_revenue + previous_refunds

# Calculate trailing baseline (average of 4 weeks)
trailing_data_list = []
for start_str, end_str in trailing_periods:
    period_data = filter_by_period(pos_df, start_str, end_str)
    trailing_data_list.append(period_data)

trailing_data = pd.concat(trailing_data_list, ignore_index=True)
trailing_transactions = trailing_data['transaction_id'].nunique()
trailing_revenue = trailing_data['line_total_sar'].sum()
trailing_refunds = trailing_data[trailing_data['is_refund'] == True]['line_total_sar'].sum()
trailing_net_revenue = trailing_revenue + trailing_refunds
trailing_avg_transactions = trailing_transactions / 4
trailing_avg_revenue = trailing_net_revenue / 4

# Calculate AOV
analysis_aov = analysis_net_revenue / analysis_transactions if analysis_transactions > 0 else 0
previous_aov = previous_net_revenue / previous_transactions if previous_transactions > 0 else 0
trailing_avg_aov = trailing_avg_revenue / trailing_avg_transactions if trailing_avg_transactions > 0 else 0

# Category mix analysis
analysis_category_mix = analysis_data.groupby('category')['line_total_sar'].sum()
previous_category_mix = previous_data.groupby('category')['line_total_sar'].sum()

# Channel mix analysis
analysis_channel_mix = analysis_data.groupby('channel')['line_total_sar'].sum()
previous_channel_mix = previous_data.groupby('channel')['line_total_sar'].sum()

# Product performance - join with menu for launch dates
analysis_product = analysis_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product = previous_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')

# Filter products by launch/retire dates
def filter_eligible_products(df, period_start, period_end):
    eligible = []
    for idx, row in df.iterrows():
        launch = pd.to_datetime(row['launch_date']) if pd.notna(row['launch_date']) else None
        retire = pd.to_datetime(row['retire_date']) if pd.notna(row['retire_date']) else None
        
        # Check if product is eligible for this period
        if launch is None or launch <= period_start:
            if retire is None or retire > period_start:
                eligible.append(idx)
    return df.iloc[eligible]

analysis_product_eligible = filter_eligible_products(analysis_product, analysis_start, analysis_end)
previous_product_eligible = filter_eligible_products(previous_product, previous_start, previous_end)

# Top products by revenue
analysis_top_products = analysis_product_eligible.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first'
}).sort_values('line_total_sar', ascending=False).head(5)

previous_top_products = previous_product_eligible.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'item_name_en': 'first'
}).sort_values('line_total_sar', ascending=False).head(5)

# Prepare findings
findings = []

# Finding 1: Revenue change week-over-week
if analysis_net_revenue != 0 and previous_net_revenue != 0:
    revenue_change = analysis_net_revenue - previous_net_revenue
    revenue_pct_change = (revenue_change / previous_net_revenue) * 100
    
    finding1 = {
        "title": "Weekly Revenue Performance",
        "claim": f"Net revenue for week of 2026-02-23 to 2026-03-02 was {analysis_net_revenue:.2f} SAR, representing a {revenue_pct_change:.1f}% change from previous week ({previous_net_revenue:.2f} SAR)",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_net_revenue": {
                "value": round(analysis_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "previous_period_net_revenue": {
                "value": round(previous_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_net_revenue, 2),
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_transactions),
        "coverage_notes": [
            f"Analysis period transactions: {analysis_transactions}",
            f"Previous period transactions: {previous_transactions}",
            f"Refunds included in net revenue calculation"
        ],
        "assumptions": [
            "line_total_sar represents net revenue after discounts",
            "is_refund flag correctly identifies refund transactions",
            "All transactions within period boundaries are valid"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Transaction count and AOV change
if analysis_transactions > 0 and previous_transactions > 0:
    transaction_change = analysis_transactions - previous_transactions
    transaction_pct_change = (transaction_change / previous_transactions) * 100
    aov_change = analysis_aov - previous_aov
    aov_pct_change = (aov_change / previous_aov) * 100 if previous_aov != 0 else 0
    
    finding2 = {
        "title": "Transaction Volume and Average Order Value",
        "claim": f"Transaction count increased by {transaction_pct_change:.1f}% ({analysis_transactions} vs {previous_transactions}), while AOV changed by {aov_pct_change:.1f}% ({analysis_aov:.2f} SAR vs {previous_aov:.2f} SAR)",
        "finding_type": "transaction_and_aov_change",
        "metrics": {
            "analysis_period_transactions": {
                "value": int(analysis_transactions),
                "unit": "count",
                "numerator": int(analysis_transactions),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": int(previous_transactions),
                "unit": "count",
                "numerator": int(previous_transactions),
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": int(analysis_transactions),
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": int(previous_transactions),
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-02-23T00:00:00+03:00",
                "period_end": "2026-03-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_transactions),
        "coverage_notes": [
            f"Unique transaction_id used for basket counting",
            f"AOV calculated as net revenue divided by transaction count",
            f"Refunds included in net revenue"
        ],
        "assumptions": [
            "transaction_id uniquely identifies a basket",
            "line_total_sar is the final transaction amount",
            "All transactions are valid and complete"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Category mix shift
if len(analysis_category_mix) > 0 and len(previous_category_mix) > 0:
    # Find categories with significant changes
    all_categories = set(analysis_category_mix.index) | set(previous_category_mix.index)
    category_changes = []
    
    for cat in all_categories:
        analysis_val = analysis_category_mix.get(cat, 0)
        previous_val = previous_category_mix.get(cat, 0)
        
        if previous_val > 0:
            pct_change = ((analysis_val - previous_val) / previous_val) * 100
            category_changes.append({
                'category': cat,
                'analysis': analysis_val,
                'previous': previous_val,
                'pct_change': pct_change,
                'abs_change': analysis_val - previous_val
            })
    
    # Sort by absolute change
    category_changes.sort(key=lambda x: abs(x['abs_change']), reverse=True)
    
    if category_changes:
        top_change = category_changes[0]
        
        finding3 = {
            "title": "Category Mix Shift",
            "claim": f"Category '{top_change['category']}' showed the largest revenue change: {top_change['analysis']:.2f} SAR (analysis period) vs {top_change['previous']:.2f} SAR (previous period), a {top_change['pct_change']:.1f}% change",
            "finding_type": "category_mix_change",
            "metrics": {
                "analysis_period_category_revenue": {
                    "value": round(top_change['analysis'], 2),
                    "unit": "SAR",
                    "numerator": round(top_change['analysis'], 2),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+03:00",
                    "period_end": "2026-03-02T00:00:00+03:00"
                },
                "previous_period_category_revenue": {
                    "value": round(top_change['previous'], 2),
                    "unit": "SAR",
                    "numerator": round(top_change['previous'], 2),
                    "denominator": None,
                    "period_start": "2026-02-16T00:00:00+03:00",
                    "period_end": "2026-02-23T00:00:00+03:00"
                },
                "category_revenue_change_pct": {
                    "value": round(top_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(top_change['abs_change'], 2),
                    "denominator": round(top_change['previous'], 2),
                    "period_start": "2026-02-23T00:00:00+03:00",
                    "period_end": "2026-03-02T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": int(analysis_transactions),
            "coverage_notes": [
                f"Category revenue aggregated from line_total_sar",
                f"All {len(all_categories)} categories analyzed",
                f"Refunds included in category totals"
            ],
            "assumptions": [
                "category field correctly classifies all products",
                "line_total_sar represents final transaction amount",
                "Category assignments are consistent across periods"
            ],
            "confidence": 0.90
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
