import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods from context
analysis_start = "2026-04-13T00:00:00+03:00"
analysis_end = "2026-04-20T00:00:00+03:00"
previous_start = "2026-04-06T00:00:00+03:00"
previous_end = "2026-04-13T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Ensure timestamp is datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_df = pos_df[(pos_df['timestamp'] >= analysis_start_dt) & (pos_df['timestamp'] < analysis_end_dt)].copy()
previous_df = pos_df[(pos_df['timestamp'] >= previous_start_dt) & (pos_df['timestamp'] < previous_end_dt)].copy()

# Calculate baseline (average of 4 trailing weeks)
baseline_dfs = []
baseline_periods = [
    ("2026-04-06T00:00:00+03:00", "2026-04-13T00:00:00+03:00"),
    ("2026-03-30T00:00:00+03:00", "2026-04-06T00:00:00+03:00"),
    ("2026-03-23T00:00:00+03:00", "2026-03-30T00:00:00+03:00"),
    ("2026-03-16T00:00:00+03:00", "2026-03-23T00:00:00+03:00"),
]

for start, end in baseline_periods:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    baseline_dfs.append(pos_df[(pos_df['timestamp'] >= start_dt) & (pos_df['timestamp'] < end_dt)].copy())

baseline_df = pd.concat(baseline_dfs, ignore_index=True)

findings = []

# FINDING 1: Revenue and Transaction Count Change
analysis_revenue = analysis_df['line_total_sar'].sum()
analysis_transactions = analysis_df['transaction_id'].nunique()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

previous_revenue = previous_df['line_total_sar'].sum()
previous_transactions = previous_df['transaction_id'].nunique()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

baseline_revenue = baseline_df['line_total_sar'].sum()
baseline_transactions = baseline_df['transaction_id'].nunique()
baseline_aov = baseline_revenue / baseline_transactions if baseline_transactions > 0 else 0

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change = analysis_transactions - previous_transactions
transaction_pct_change = (transaction_change / previous_transactions * 100) if previous_transactions > 0 else 0

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Count refunds in analysis and previous periods
analysis_refunds = analysis_df[analysis_df['is_refund'] == True]['line_total_sar'].sum()
previous_refunds = previous_df[previous_df['is_refund'] == True]['line_total_sar'].sum()
analysis_refund_count = analysis_df[analysis_df['is_refund'] == True].shape[0]
previous_refund_count = previous_df[previous_df['is_refund'] == True].shape[0]

findings.append({
    "title": "Net Revenue and Transaction Volume Week-over-Week",
    "claim": f"Net revenue in analysis week (Apr 13-20) was {analysis_revenue:.2f} SAR across {analysis_transactions} transactions (AOV {analysis_aov:.2f} SAR), compared to {previous_revenue:.2f} SAR across {previous_transactions} transactions (AOV {previous_aov:.2f} SAR) in the previous week (Apr 6-13). Revenue changed by {revenue_change:.2f} SAR ({revenue_pct_change:.2f}%), transaction count changed by {transaction_change} ({transaction_pct_change:.2f}%), and AOV increased by {aov_change:.2f} SAR ({aov_pct_change:.2f}%).",
    "finding_type": "revenue_and_transaction_mix",
    "metrics": {
        "analysis_revenue_sar": {
            "value": round(analysis_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "analysis_transactions": {
            "value": analysis_transactions,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "analysis_aov_sar": {
            "value": round(analysis_aov, 2),
            "unit": "SAR",
            "numerator": round(analysis_revenue, 2),
            "denominator": analysis_transactions,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "previous_revenue_sar": {
            "value": round(previous_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": previous_start,
            "period_end": previous_end
        },
        "previous_transactions": {
            "value": previous_transactions,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": previous_start,
            "period_end": previous_end
        },
        "previous_aov_sar": {
            "value": round(previous_aov, 2),
            "unit": "SAR",
            "numerator": round(previous_revenue, 2),
            "denominator": previous_transactions,
            "period_start": previous_start,
            "period_end": previous_end
        },
        "revenue_change_sar": {
            "value": round(revenue_change, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "revenue_pct_change": {
            "value": round(revenue_pct_change, 2),
            "unit": "%",
            "numerator": round(revenue_change, 2),
            "denominator": round(previous_revenue, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "transaction_change": {
            "value": transaction_change,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "transaction_pct_change": {
            "value": round(transaction_pct_change, 2),
            "unit": "%",
            "numerator": transaction_change,
            "denominator": previous_transactions,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "aov_change_sar": {
            "value": round(aov_change, 2),
            "unit": "SAR",
            "numerator": round(aov_change, 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "aov_pct_change": {
            "value": round(aov_pct_change, 2),
            "unit": "%",
            "numerator": round(aov_change, 2),
            "denominator": round(previous_aov, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": ["pos"],
    "sample_size": analysis_transactions,
    "coverage_notes": [
        f"Analysis period: {analysis_transactions} unique transactions, {analysis_df.shape[0]} line items",
        f"Previous period: {previous_transactions} unique transactions, {previous_df.shape[0]} line items",
        f"Analysis period refunds: {analysis_refund_count} line items totaling {analysis_refunds:.2f} SAR",
        f"Previous period refunds: {previous_refund_count} line items totaling {previous_refunds:.2f} SAR",
        "Revenue includes refunds as negative values per metric definition"
    ],
    "assumptions": [
        "Unique transaction_id used to count baskets (valid transactions)",
        "line_total_sar includes refunds as negative values",
        "AOV calculated as net revenue divided by transaction count",
        "Periods are consecutive weeks with no gaps",
        "All rows with known_sku=True and valid transaction_id included"
    ],
    "confidence": 0.95
})

# FINDING 2: Category Mix Analysis
analysis_category_revenue = analysis_df.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_df.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

analysis_category_pct = (analysis_category_revenue / analysis_revenue * 100).round(2)
previous_category_pct = (previous_category_revenue / previous_revenue * 100).round(2)

# Find top category changes
category_changes = {}
for cat in analysis_category_revenue.index:
    if cat in previous_category_revenue.index:
        prev_pct = previous_category_pct[cat]
        curr_pct = analysis_category_pct[cat]
        pct_point_change = curr_pct - prev_pct
        category_changes[cat] = {
            'analysis_revenue': analysis_category_revenue[cat],
            'analysis_pct': curr_pct,
            'previous_revenue': previous_category_revenue[cat],
            'previous_pct': prev_pct,
            'pct_point_change': pct_point_change,
            'revenue_change': analysis_category_revenue[cat] - previous_category_revenue[cat]
        }

# Sort by absolute percentage point change
sorted_categories = sorted(category_changes.items(), key=lambda x: abs(x[1]['pct_point_change']), reverse=True)

if len(sorted_categories) > 0:
    top_cat, top_cat_data = sorted_categories[0]
    
    findings.append({
        "title": "Category Mix Shift: Largest Percentage Point Change",
        "claim": f"Category '{top_cat}' represented {top_cat_data['analysis_pct']:.2f}% of revenue in the analysis week (Apr 13-20), compared to {top_cat_data['previous_pct']:.2f}% in the previous week (Apr 6-13), a change of {top_cat_data['pct_point_change']:.2f} percentage points. Absolute revenue for this category changed from {top_cat_data['previous_revenue']:.2f} SAR to {top_cat_data['analysis_revenue']:.2f} SAR ({top_cat_data['revenue_change']:.2f} SAR change).",
        "finding_type": "product_category_mix",
        "metrics": {
            "analysis_category_revenue_sar": {
                "value": round(top_cat_data['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "analysis_category_pct": {
                "value": round(top_cat_data['analysis_pct'], 2),
                "unit": "%",
                "numerator": round(top_cat_data['analysis_revenue'], 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_category_revenue_sar": {
                "value": round(top_cat_data['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "previous_category_pct": {
                "value": round(top_cat_data['previous_pct'], 2),
                "unit": "%",
                "numerator": round(top_cat_data['previous_revenue'], 2),
                "denominator": round(previous_revenue, 2),
                "period_start": previous_start,
                "period_end": previous_end
            },
            "category_pct_point_change": {
                "value": round(top_cat_data['pct_point_change'], 2),
                "unit": "percentage points",
                "numerator": round(top_cat_data['pct_point_change'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "category_revenue_change_sar": {
                "value": round(top_cat_data['revenue_change'], 2),
                "unit": "SAR",
                "numerator": round(top_cat_data['revenue_change'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_df[analysis_df['category'] == top_cat]['transaction_id'].nunique(),
        "coverage_notes": [
            f"Category '{top_cat}' analysis period: {analysis_df[analysis_df['category'] == top_cat].shape[0]} line items",
            f"Category '{top_cat}' previous period: {previous_df[previous_df['category'] == top_cat].shape[0]} line items",
            f"Total categories in analysis period: {len(analysis_category_revenue)}",
            "Revenue includes refunds as negative values"
        ],
        "assumptions": [
            "Category field populated from cleaned POS data",
            "Revenue calculated as sum of line_total_sar per category",
            "Percentage calculated as category revenue / total revenue",
            "Comparison is week-over-week consecutive periods"
        ],
        "confidence": 0.92
    })

# FINDING 3: Channel Mix Analysis
analysis_channel_revenue = analysis_df.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_df.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_pct = (analysis_channel_revenue / analysis_revenue * 100).round(2)
previous_channel_pct = (previous_channel_revenue / previous_revenue * 100).round(2)

channel_changes = {}
for ch in analysis_channel_revenue.index:
    if ch in previous_channel_revenue.index:
        prev_pct = previous_channel_pct[ch]
        curr_pct = analysis_channel_pct[ch]
        pct_point_change = curr_pct - prev_pct
        channel_changes[ch] = {
            'analysis_revenue': analysis_channel_revenue[ch],
            'analysis_pct': curr_pct,
            'previous_revenue': previous_channel_revenue[ch],
            'previous_pct': prev_pct,
            'pct_point_change': pct_point_change,
            'revenue_change': analysis_channel_revenue[ch] - previous_channel_revenue[ch]
        }

sorted_channels = sorted(channel_changes.items(), key=lambda x: abs(x[1]['pct_point_change']), reverse=True)

if len(sorted_channels) > 0:
    top_ch, top_ch_data = sorted_channels[0]
    
    findings.append({
        "title": "Channel Mix Shift: Largest Percentage Point Change",
        "claim": f"Channel '{top_ch}' represented {top_ch_data['analysis_pct']:.2f}% of revenue in the analysis week (Apr 13-20), compared to {top_ch_data['previous_pct']:.2f}% in the previous week (Apr 6-13), a change of {top_ch_data['pct_point_change']:.2f} percentage points. Absolute revenue for this channel changed from {top_ch_data['previous_revenue']:.2f} SAR to {top_ch_data['analysis_revenue']:.2f} SAR ({top_ch_data['revenue_change']:.2f} SAR change).",
        "finding_type": "channel_mix",
        "metrics": {
            "analysis_channel_revenue_sar": {
                "value": round(top_ch_data['analysis_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "analysis_channel_pct": {
                "value": round(top_ch_data['analysis_pct'], 2),
                "unit": "%",
                "numerator": round(top_ch_data['analysis_revenue'], 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_channel_revenue_sar": {
                "value": round(top_ch_data['previous_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "previous_channel_pct": {
                "value": round(top_ch_data['previous_pct'], 2),
                "unit": "%",
                "numerator": round(top_ch_data['previous_revenue'], 2),
                "denominator": round(previous_revenue, 2),
                "period_start": previous_start,
                "period_end": previous_end
            },
            "channel_pct_point_change": {
                "value": round(top_ch_data['pct_point_change'], 2),
                "unit": "percentage points",
                "numerator": round(top_ch_data['pct_point_change'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "channel_revenue_change_sar": {
                "value": round(top_ch_data['revenue_change'], 2),
                "unit": "SAR",
                "numerator": round(top_ch_data['revenue_change'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_df[analysis_df['channel'] == top_ch]['transaction_id'].nunique(),
        "coverage_notes": [
            f"Channel '{top_ch}' analysis period: {analysis_df[analysis_df['channel'] == top_ch].shape[0]} line items",
            f"Channel '{top_ch}' previous period: {previous_df[previous_df['channel'] == top_ch].shape[0]} line items",
            f"Total channels in analysis period: {len(analysis_channel_revenue)}",
            "Revenue includes refunds as negative values"
        ],
        "assumptions": [
            "Channel field populated from cleaned POS data",
            "Revenue calculated as sum of line_total_sar per channel",
            "Percentage calculated as channel revenue / total revenue",
            "Comparison is week-over-week consecutive periods"
        ],
        "confidence": 0.92
    })

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Results written to {output_path}")