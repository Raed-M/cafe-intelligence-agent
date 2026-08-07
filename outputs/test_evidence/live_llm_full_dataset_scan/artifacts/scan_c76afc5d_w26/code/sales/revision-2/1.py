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

# Parse periods
analysis_start = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-06-29T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-06T00:00:00+03:00")

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Baseline periods for trailing average
baseline_periods = [
    ("2026-06-29T00:00:00+03:00", "2026-07-06T00:00:00+03:00"),
    ("2026-06-22T00:00:00+03:00", "2026-06-29T00:00:00+03:00"),
    ("2026-06-15T00:00:00+03:00", "2026-06-22T00:00:00+03:00"),
    ("2026-06-08T00:00:00+03:00", "2026-06-15T00:00:00+03:00"),
]

baseline_data_list = []
for start_str, end_str in baseline_periods:
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    period_data = pos_df[(pos_df['timestamp'] >= start) & (pos_df['timestamp'] < end)].copy()
    baseline_data_list.append(period_data)

baseline_data = pd.concat(baseline_data_list, ignore_index=True)

findings = []

# FINDING 1: Revenue and Transaction Count Change
# Count unique transaction_ids (baskets) - include all transactions
analysis_baskets = analysis_data['transaction_id'].nunique()
previous_baskets = previous_data['transaction_id'].nunique()

# Net revenue (line_total_sar includes refunds as negative)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

basket_change = analysis_baskets - previous_baskets
basket_change_pct = (basket_change / previous_baskets * 100) if previous_baskets != 0 else 0

# AOV calculation
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

findings.append({
    "title": "Revenue and Transaction Volume Change Week-over-Week",
    "claim": f"Net revenue in analysis week (2026-07-06 to 2026-07-13) was {analysis_revenue:.2f} SAR across {analysis_baskets} transactions, compared to {previous_revenue:.2f} SAR across {previous_baskets} transactions in the previous week (2026-06-29 to 2026-07-06). This represents a {revenue_change:.2f} SAR ({revenue_change_pct:.2f}%) change in revenue and {basket_change} ({basket_change_pct:.2f}%) change in transaction count. Average order value changed from {previous_aov:.2f} SAR to {analysis_aov:.2f} SAR, a {aov_change:.2f} SAR ({aov_change_pct:.2f}%) decline.",
    "finding_type": "revenue_and_transaction_mix",
    "metrics": {
        "analysis_week_revenue": {
            "value": round(analysis_revenue, 2),
            "unit": "SAR",
            "numerator": round(analysis_revenue, 2),
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "previous_week_revenue": {
            "value": round(previous_revenue, 2),
            "unit": "SAR",
            "numerator": round(previous_revenue, 2),
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        },
        "revenue_change_sar": {
            "value": round(revenue_change, 2),
            "unit": "SAR",
            "numerator": round(revenue_change, 2),
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "revenue_change_pct": {
            "value": round(revenue_change_pct, 2),
            "unit": "%",
            "numerator": round(revenue_change, 2),
            "denominator": round(previous_revenue, 2),
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "analysis_week_baskets": {
            "value": analysis_baskets,
            "unit": "transactions",
            "numerator": analysis_baskets,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "previous_week_baskets": {
            "value": previous_baskets,
            "unit": "transactions",
            "numerator": previous_baskets,
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        },
        "basket_change": {
            "value": basket_change,
            "unit": "transactions",
            "numerator": basket_change,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "basket_change_pct": {
            "value": round(basket_change_pct, 2),
            "unit": "%",
            "numerator": basket_change,
            "denominator": previous_baskets,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "analysis_week_aov": {
            "value": round(analysis_aov, 2),
            "unit": "SAR",
            "numerator": round(analysis_revenue, 2),
            "denominator": analysis_baskets,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "previous_week_aov": {
            "value": round(previous_aov, 2),
            "unit": "SAR",
            "numerator": round(previous_revenue, 2),
            "denominator": previous_baskets,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        },
        "aov_change_sar": {
            "value": round(aov_change, 2),
            "unit": "SAR",
            "numerator": round(aov_change, 2),
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "aov_change_pct": {
            "value": round(aov_change_pct, 2),
            "unit": "%",
            "numerator": round(aov_change, 2),
            "denominator": round(previous_aov, 2),
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        f"Analysis period: 2026-07-06 to 2026-07-13 ({analysis_baskets} unique transactions, {len(analysis_data)} line items)",
        f"Previous period: 2026-06-29 to 2026-07-06 ({previous_baskets} unique transactions, {len(previous_data)} line items)",
        "Refunds included in net revenue calculations as negative line_total_sar values",
        "Transaction count based on unique transaction_id values"
    ],
    "assumptions": [
        "All transactions in POS data are valid and complete",
        "line_total_sar represents net revenue including refunds",
        "Unique transaction_id represents a basket/transaction",
        "Timestamp filtering uses UTC+3 timezone as provided"
    ],
    "confidence": 0.95
})

# FINDING 2: Category Mix Analysis
# Merge POS with menu to get category information
analysis_with_category = analysis_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                              on='sku', how='left', suffixes=('_pos', '_menu'))
previous_with_category = previous_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], 
                                              on='sku', how='left', suffixes=('_pos', '_menu'))

# Handle launch/retire dates
analysis_with_category['launch_date'] = pd.to_datetime(analysis_with_category['launch_date'], errors='coerce')
analysis_with_category['retire_date'] = pd.to_datetime(analysis_with_category['retire_date'], errors='coerce')
previous_with_category['launch_date'] = pd.to_datetime(previous_with_category['launch_date'], errors='coerce')
previous_with_category['retire_date'] = pd.to_datetime(previous_with_category['retire_date'], errors='coerce')

# Category revenue analysis
analysis_category_revenue = analysis_with_category.groupby('category_menu')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_with_category.groupby('category_menu')['line_total_sar'].sum().sort_values(ascending=False)

# Get top category
if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
    top_category = analysis_category_revenue.index[0]
    analysis_top_rev = analysis_category_revenue.iloc[0]
    previous_top_rev = previous_category_revenue.get(top_category, 0)
    
    top_cat_change = analysis_top_rev - previous_top_rev
    top_cat_change_pct = (top_cat_change / previous_top_rev * 100) if previous_top_rev != 0 else 0
    
    # Count items in top category
    analysis_top_items = len(analysis_with_category[analysis_with_category['category_menu'] == top_category])
    previous_top_items = len(previous_with_category[previous_with_category['category_menu'] == top_category])
    
    findings.append({
        "title": "Top Category Revenue Performance",
        "claim": f"The {top_category} category generated {analysis_top_rev:.2f} SAR in the analysis week (2026-07-06 to 2026-07-13) compared to {previous_top_rev:.2f} SAR in the previous week (2026-06-29 to 2026-07-06), representing a {top_cat_change:.2f} SAR ({top_cat_change_pct:.2f}%) change. This category accounted for {analysis_top_rev/analysis_revenue*100:.1f}% of total revenue in the analysis week.",
        "finding_type": "product_category_mix",
        "metrics": {
            "analysis_week_top_category_revenue": {
                "value": round(analysis_top_rev, 2),
                "unit": "SAR",
                "numerator": round(analysis_top_rev, 2),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "previous_week_top_category_revenue": {
                "value": round(previous_top_rev, 2),
                "unit": "SAR",
                "numerator": round(previous_top_rev, 2),
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "top_category_revenue_change": {
                "value": round(top_cat_change, 2),
                "unit": "SAR",
                "numerator": round(top_cat_change, 2),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "top_category_revenue_change_pct": {
                "value": round(top_cat_change_pct, 2),
                "unit": "%",
                "numerator": round(top_cat_change, 2),
                "denominator": round(previous_top_rev, 2),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "top_category_pct_of_total": {
                "value": round(analysis_top_rev/analysis_revenue*100, 1),
                "unit": "%",
                "numerator": round(analysis_top_rev, 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": analysis_top_items,
        "coverage_notes": [
            f"Top category: {top_category}",
            f"Analysis period line items in top category: {analysis_top_items}",
            f"Previous period line items in top category: {previous_top_items}",
            "Category information sourced from menu SKU reference",
            "Refunds included in net revenue"
        ],
        "assumptions": [
            "Menu category mapping is accurate and complete",
            "All SKUs in POS data have corresponding menu entries",
            "Category classification is stable across periods"
        ],
        "confidence": 0.92
    })

# FINDING 3: Channel Mix Analysis
analysis_channel = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_baskets = analysis_data.groupby('channel')['transaction_id'].nunique()
previous_channel_baskets = previous_data.groupby('channel')['transaction_id'].nunique()

if len(analysis_channel) > 0:
    top_channel = analysis_channel.index[0]
    analysis_top_channel_rev = analysis_channel.iloc[0]
    previous_top_channel_rev = previous_channel.get(top_channel, 0)
    
    analysis_top_channel_baskets = analysis_channel_baskets.get(top_channel, 0)
    previous_top_channel_baskets = previous_channel_baskets.get(top_channel, 0)
    
    top_channel_rev_change = analysis_top_channel_rev - previous_top_channel_rev
    top_channel_rev_change_pct = (top_channel_rev_change / previous_top_channel_rev * 100) if previous_top_channel_rev != 0 else 0
    
    top_channel_basket_change = analysis_top_channel_baskets - previous_top_channel_baskets
    top_channel_basket_change_pct = (top_channel_basket_change / previous_top_channel_baskets * 100) if previous_top_channel_baskets != 0 else 0
    
    findings.append({
        "title": "Primary Channel Revenue and Transaction Performance",
        "claim": f"The {top_channel} channel generated {analysis_top_channel_rev:.2f} SAR across {analysis_top_channel_baskets} transactions in the analysis week (2026-07-06 to 2026-07-13), compared to {previous_top_channel_rev:.2f} SAR across {previous_top_channel_baskets} transactions in the previous week (2026-06-29 to 2026-07-06). This represents a {top_channel_rev_change:.2f} SAR ({top_channel_rev_change_pct:.2f}%) revenue change and {top_channel_basket_change} ({top_channel_basket_change_pct:.2f}%) transaction change.",
        "finding_type": "channel_mix",
        "metrics": {
            "analysis_week_top_channel_revenue": {
                "value": round(analysis_top_channel_rev, 2),
                "unit": "SAR",
                "numerator": round(analysis_top_channel_rev, 2),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "previous_week_top_channel_revenue": {
                "value": round(previous_top_channel_rev, 2),
                "unit": "SAR",
                "numerator": round(previous_top_channel_rev, 2),
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "top_channel_revenue_change": {
                "value": round(top_channel_rev_change, 2),
                "unit": "SAR",
                "numerator": round(top_channel_rev_change, 2),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "top_channel_revenue_change_pct": {
                "value": round(top_channel_rev_change_pct, 2),
                "unit": "%",
                "numerator": round(top_channel_rev_change, 2),
                "denominator": round(previous_top_channel_rev, 2),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "analysis_week_top_channel_baskets": {
                "value": analysis_top_channel_baskets,
                "unit": "transactions",
                "numerator": analysis_top_channel_baskets,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "previous_week_top_channel_baskets": {
                "value": previous_top_channel_baskets,
                "unit": "transactions",
                "numerator": previous_top_channel_baskets,
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "top_channel_basket_change": {
                "value": top_channel_basket_change,
                "unit": "transactions",
                "numerator": top_channel_basket_change,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "top_channel_basket_change_pct": {
                "value": round(top_channel_basket_change_pct, 2),
                "unit": "%",
                "numerator": top_channel_basket_change,
                "denominator": previous_top_channel_baskets,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data[analysis_data['channel'] == top_channel]),
        "coverage_notes": [
            f"Primary channel: {top_channel}",
            f"Analysis period transactions: {analysis_top_channel_baskets}",
            f"Previous period transactions: {previous_top_channel_baskets}",
            "Channel data from POS transaction records",
            "Refunds included in net revenue"
        ],
        "assumptions": [
            "Channel classification is consistent across periods",
            "All transactions have valid channel assignment",
            "Channel represents point of sale or order method"
        ],
        "confidence": 0.93
    })

# Write output
output = {
    "status": "success",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)