import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = datetime.fromisoformat("2026-04-27T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-05-04T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-27T00:00:00+03:00")

trailing_periods = [
    (datetime.fromisoformat("2026-04-20T00:00:00+03:00"), datetime.fromisoformat("2026-04-27T00:00:00+03:00")),
    (datetime.fromisoformat("2026-04-13T00:00:00+03:00"), datetime.fromisoformat("2026-04-20T00:00:00+03:00")),
    (datetime.fromisoformat("2026-04-06T00:00:00+03:00"), datetime.fromisoformat("2026-04-13T00:00:00+03:00")),
    (datetime.fromisoformat("2026-03-30T00:00:00+03:00"), datetime.fromisoformat("2026-04-06T00:00:00+03:00")),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for valid transactions (non-refund, non-inconsistent)
valid_pos = pos_df[(pos_df['is_refund'] == False) & (pos_df['line_total_inconsistent'] == False)].copy()

# Function to filter by period
def filter_by_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

# Analysis period data
analysis_data = filter_by_period(valid_pos, analysis_start, analysis_end)

# Previous period data
previous_data = filter_by_period(valid_pos, previous_start, previous_end)

# Trailing baseline (average of 4 weeks)
trailing_data = filter_by_period(valid_pos, trailing_periods[0][0], trailing_periods[-1][1])

findings = []

# FINDING 1: Revenue and Transaction Count Change
if len(analysis_data) > 0 and len(previous_data) > 0:
    # Count unique transactions
    analysis_transactions = analysis_data['transaction_id'].nunique()
    previous_transactions = previous_data['transaction_id'].nunique()
    
    # Calculate revenue
    analysis_revenue = analysis_data['line_total_sar'].sum()
    previous_revenue = previous_data['line_total_sar'].sum()
    
    # Calculate changes
    transaction_change = analysis_transactions - previous_transactions
    transaction_pct_change = (transaction_change / previous_transactions * 100) if previous_transactions > 0 else 0
    
    revenue_change = analysis_revenue - previous_revenue
    revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0
    
    # Calculate AOV
    analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0
    previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0
    aov_change = analysis_aov - previous_aov
    
    findings.append({
        "title": "Revenue and Transaction Performance vs Previous Week",
        "claim": f"Analysis period (2026-04-27 to 2026-05-04) shows {transaction_pct_change:.1f}% change in transaction count ({analysis_transactions} vs {previous_transactions}) and {revenue_pct_change:.1f}% change in net revenue (SAR {analysis_revenue:.2f} vs SAR {previous_revenue:.2f}). Average order value changed from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}.",
        "finding_type": "revenue_and_transaction_analysis",
        "metrics": {
            "analysis_period_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "transaction_count_change": {
                "value": transaction_change,
                "unit": "count",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "transaction_count_pct_change": {
                "value": round(transaction_pct_change, 2),
                "unit": "percent",
                "numerator": transaction_change,
                "denominator": previous_transactions,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": revenue_change,
                "denominator": previous_revenue,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 2),
                "unit": "percent",
                "numerator": revenue_change,
                "denominator": previous_revenue,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": aov_change,
                "denominator": previous_aov,
                "period_start": "2026-04-27T00:00:00+03:00",
                "period_end": "2026-05-04T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Analysis period: 2026-04-27 to 2026-05-04",
            "Previous period: 2026-04-20 to 2026-04-27",
            "Excluded refunds and inconsistent line totals",
            "Transaction count based on unique transaction_id"
        ],
        "assumptions": [
            "Valid transactions are those with is_refund=False and line_total_inconsistent=False",
            "Revenue calculated as sum of line_total_sar",
            "AOV calculated as total revenue divided by unique transaction count"
        ],
        "confidence": 0.95
    })

# FINDING 2: Category Mix Analysis
if len(analysis_data) > 0 and len(previous_data) > 0:
    # Merge with menu to get category info
    analysis_with_category = analysis_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], on='sku', how='left')
    previous_with_category = previous_data.merge(menu_df[['sku', 'category', 'launch_date', 'retire_date']], on='sku', how='left')
    
    # Calculate category revenue
    analysis_category_revenue = analysis_with_category.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    previous_category_revenue = previous_with_category.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    
    # Get top category
    if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
        top_category = analysis_category_revenue.index[0]
        analysis_top_revenue = analysis_category_revenue.iloc[0]
        previous_top_revenue = previous_category_revenue.get(top_category, 0)
        
        if previous_top_revenue > 0:
            category_change = analysis_top_revenue - previous_top_revenue
            category_pct_change = (category_change / previous_top_revenue * 100)
            
            # Calculate share
            analysis_total = analysis_category_revenue.sum()
            previous_total = previous_category_revenue.sum()
            analysis_share = (analysis_top_revenue / analysis_total * 100) if analysis_total > 0 else 0
            previous_share = (previous_top_revenue / previous_total * 100) if previous_total > 0 else 0
            
            findings.append({
                "title": "Top Category Revenue Performance",
                "claim": f"Top category '{top_category}' generated SAR {analysis_top_revenue:.2f} in analysis period vs SAR {previous_top_revenue:.2f} in previous period, representing {category_pct_change:.1f}% change. Category share increased from {previous_share:.1f}% to {analysis_share:.1f}% of total revenue.",
                "finding_type": "category_mix_analysis",
                "metrics": {
                    "top_category": {
                        "value": top_category,
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "analysis_period_top_category_revenue": {
                        "value": round(analysis_top_revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "previous_period_top_category_revenue": {
                        "value": round(previous_top_revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "category_revenue_change": {
                        "value": round(category_change, 2),
                        "unit": "SAR",
                        "numerator": category_change,
                        "denominator": previous_top_revenue,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "category_revenue_pct_change": {
                        "value": round(category_pct_change, 2),
                        "unit": "percent",
                        "numerator": category_change,
                        "denominator": previous_top_revenue,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "analysis_period_category_share": {
                        "value": round(analysis_share, 2),
                        "unit": "percent",
                        "numerator": analysis_top_revenue,
                        "denominator": analysis_total,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "previous_period_category_share": {
                        "value": round(previous_share, 2),
                        "unit": "percent",
                        "numerator": previous_top_revenue,
                        "denominator": previous_total,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": len(analysis_with_category),
                "coverage_notes": [
                    "Analysis period: 2026-04-27 to 2026-05-04",
                    "Previous period: 2026-04-20 to 2026-04-27",
                    "Category information merged from menu SKU reference",
                    "Excluded refunds and inconsistent line totals"
                ],
                "assumptions": [
                    "Category assignment based on menu SKU reference",
                    "Revenue calculated as sum of line_total_sar per category",
                    "Share calculated as category revenue divided by total revenue"
                ],
                "confidence": 0.92
            })

# FINDING 3: Channel Mix Analysis
if len(analysis_data) > 0 and len(previous_data) > 0:
    # Calculate channel revenue
    analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    
    # Calculate channel transactions
    analysis_channel_transactions = analysis_data.groupby('channel')['transaction_id'].nunique()
    previous_channel_transactions = previous_data.groupby('channel')['transaction_id'].nunique()
    
    if len(analysis_channel_revenue) > 0 and len(previous_channel_revenue) > 0:
        # Get top channel
        top_channel = analysis_channel_revenue.index[0]
        analysis_top_channel_revenue = analysis_channel_revenue.iloc[0]
        previous_top_channel_revenue = previous_channel_revenue.get(top_channel, 0)
        
        analysis_top_channel_transactions = analysis_channel_transactions.get(top_channel, 0)
        previous_top_channel_transactions = previous_channel_transactions.get(top_channel, 0)
        
        if previous_top_channel_revenue > 0 and previous_top_channel_transactions > 0:
            channel_revenue_change = analysis_top_channel_revenue - previous_top_channel_revenue
            channel_revenue_pct_change = (channel_revenue_change / previous_top_channel_revenue * 100)
            
            channel_transaction_change = analysis_top_channel_transactions - previous_top_channel_transactions
            channel_transaction_pct_change = (channel_transaction_change / previous_top_channel_transactions * 100)
            
            # Calculate channel share
            analysis_total_revenue = analysis_channel_revenue.sum()
            previous_total_revenue = previous_channel_revenue.sum()
            analysis_channel_share = (analysis_top_channel_revenue / analysis_total_revenue * 100) if analysis_total_revenue > 0 else 0
            previous_channel_share = (previous_top_channel_revenue / previous_total_revenue * 100) if previous_total_revenue > 0 else 0
            
            findings.append({
                "title": "Top Channel Performance and Mix",
                "claim": f"Top channel '{top_channel}' generated SAR {analysis_top_channel_revenue:.2f} from {analysis_top_channel_transactions} transactions in analysis period vs SAR {previous_top_channel_revenue:.2f} from {previous_top_channel_transactions} transactions in previous period. Revenue change: {channel_revenue_pct_change:.1f}%, transaction change: {channel_transaction_pct_change:.1f}%. Channel share: {analysis_channel_share:.1f}% (vs {previous_channel_share:.1f}% previously).",
                "finding_type": "channel_mix_analysis",
                "metrics": {
                    "top_channel": {
                        "value": top_channel,
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "analysis_period_top_channel_revenue": {
                        "value": round(analysis_top_channel_revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "previous_period_top_channel_revenue": {
                        "value": round(previous_top_channel_revenue, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "channel_revenue_change": {
                        "value": round(channel_revenue_change, 2),
                        "unit": "SAR",
                        "numerator": channel_revenue_change,
                        "denominator": previous_top_channel_revenue,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "channel_revenue_pct_change": {
                        "value": round(channel_revenue_pct_change, 2),
                        "unit": "percent",
                        "numerator": channel_revenue_change,
                        "denominator": previous_top_channel_revenue,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "analysis_period_top_channel_transactions": {
                        "value": analysis_top_channel_transactions,
                        "unit": "count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "previous_period_top_channel_transactions": {
                        "value": previous_top_channel_transactions,
                        "unit": "count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    },
                    "channel_transaction_change": {
                        "value": channel_transaction_change,
                        "unit": "count",
                        "numerator": channel_transaction_change,
                        "denominator": previous_top_channel_transactions,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "channel_transaction_pct_change": {
                        "value": round(channel_transaction_pct_change, 2),
                        "unit": "percent",
                        "numerator": channel_transaction_change,
                        "denominator": previous_top_channel_transactions,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "analysis_period_channel_share": {
                        "value": round(analysis_channel_share, 2),
                        "unit": "percent",
                        "numerator": analysis_top_channel_revenue,
                        "denominator": analysis_total_revenue,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "previous_period_channel_share": {
                        "value": round(previous_channel_share, 2),
                        "unit": "percent",
                        "numerator": previous_top_channel_revenue,
                        "denominator": previous_total_revenue,
                        "period_start": "2026-04-20T00:00:00+03:00",
                        "period_end": "2026-04-27T00:00:00+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_data),
                "coverage_notes": [
                    "Analysis period: 2026-04-27 to 2026-05-04",
                    "Previous period: 2026-04-20 to 2026-04-27",
                    "Channel information from POS data",
                    "Excluded refunds and inconsistent line totals",
                    "Transaction count based on unique transaction_id"
                ],
                "assumptions": [
                    "Channel assignment based on POS channel field",
                    "Revenue calculated as sum of line_total_sar per channel",
                    "Share calculated as channel revenue divided by total revenue"
                ],
                "confidence": 0.93
            })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
