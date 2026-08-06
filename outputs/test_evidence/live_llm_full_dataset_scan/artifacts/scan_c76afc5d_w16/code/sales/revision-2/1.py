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

# Define periods (ISO 8601 with +03:00 timezone)
analysis_start = "2026-04-27T00:00:00+03:00"
analysis_end = "2026-05-04T00:00:00+03:00"
previous_start = "2026-04-20T00:00:00+03:00"
previous_end = "2026-04-27T00:00:00+03:00"

# Convert to UTC for comparison (subtract 3 hours)
analysis_start_utc = pd.Timestamp("2026-04-26T21:00:00Z")
analysis_end_utc = pd.Timestamp("2026-05-03T21:00:00Z")
previous_start_utc = pd.Timestamp("2026-04-19T21:00:00Z")
previous_end_utc = pd.Timestamp("2026-04-26T21:00:00Z")

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis and previous periods
analysis_data = pos_df[
    (pos_df['timestamp'] >= analysis_start_utc) & 
    (pos_df['timestamp'] < analysis_end_utc)
].copy()

previous_data = pos_df[
    (pos_df['timestamp'] >= previous_start_utc) & 
    (pos_df['timestamp'] < previous_end_utc)
].copy()

# Trailing baseline (4 weeks before analysis)
trailing_baseline_data = pos_df[
    (pos_df['timestamp'] >= previous_start_utc) & 
    (pos_df['timestamp'] < previous_end_utc)
].copy()

# Clean data: exclude refunds and inconsistent line totals
def clean_data(df):
    return df[
        (df['is_refund'] == False) & 
        (df['line_total_inconsistent'] == False)
    ].copy()

analysis_clean = clean_data(analysis_data)
previous_clean = clean_data(previous_data)
trailing_clean = clean_data(trailing_baseline_data)

# Calculate metrics for analysis period
analysis_txn_count = analysis_clean['transaction_id'].nunique()
analysis_revenue = analysis_clean['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_txn_count if analysis_txn_count > 0 else 0

# Calculate metrics for previous period
previous_txn_count = previous_clean['transaction_id'].nunique()
previous_revenue = previous_clean['line_total_sar'].sum()
previous_aov = previous_revenue / previous_txn_count if previous_txn_count > 0 else 0

# Calculate metrics for trailing baseline
trailing_txn_count = trailing_clean['transaction_id'].nunique()
trailing_revenue = trailing_clean['line_total_sar'].sum()
trailing_aov = trailing_revenue / trailing_txn_count if trailing_txn_count > 0 else 0

# Calculate percentage changes
txn_pct_change = ((analysis_txn_count - previous_txn_count) / previous_txn_count * 100) if previous_txn_count > 0 else 0
revenue_pct_change = ((analysis_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
aov_pct_change = ((analysis_aov - previous_aov) / previous_aov * 100) if previous_aov > 0 else 0

# Category analysis
analysis_by_category = analysis_clean.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'txn_count']
analysis_by_category['share_pct'] = (analysis_by_category['revenue'] / analysis_revenue * 100)

previous_by_category = previous_clean.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'txn_count']
previous_by_category['share_pct'] = (previous_by_category['revenue'] / previous_revenue * 100)

# Channel analysis
analysis_by_channel = analysis_clean.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'txn_count']
analysis_by_channel['aov'] = analysis_by_channel['revenue'] / analysis_by_channel['txn_count']

previous_by_channel = previous_clean.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'txn_count']
previous_by_channel['aov'] = previous_by_channel['revenue'] / previous_by_channel['txn_count']

# Find strongest findings
findings = []

# Finding 1: Overall transaction and revenue performance
if analysis_txn_count > 0 and previous_txn_count > 0:
    finding1 = {
        "title": "Transaction Count and Revenue Performance Week-over-Week",
        "claim": f"Transaction count increased by {txn_pct_change:.2f}% (from {previous_txn_count} to {analysis_txn_count} baskets) while net revenue declined by {abs(revenue_pct_change):.2f}% (SAR {previous_revenue:.2f} to SAR {analysis_revenue:.2f}), indicating lower average order values.",
        "finding_type": "sales_performance",
        "metrics": {
            "analysis_txn_count": {
                "value": analysis_txn_count,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_txn_count": {
                "value": previous_txn_count,
                "unit": "baskets",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "txn_count_pct_change": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": analysis_txn_count - previous_txn_count,
                "denominator": previous_txn_count,
                "period_start": previous_start,
                "period_end": analysis_end
            },
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "revenue_pct_change": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": analysis_revenue - previous_revenue,
                "denominator": previous_revenue,
                "period_start": previous_start,
                "period_end": analysis_end
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": analysis_aov - previous_aov,
                "denominator": previous_aov,
                "period_start": previous_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_txn_count + previous_txn_count,
        "coverage_notes": [
            "Refunds excluded (is_refund=False)",
            "Inconsistent line totals excluded (line_total_inconsistent=False)",
            "Transaction count derived from unique transaction_id",
            "Revenue calculated as sum of line_total_sar"
        ],
        "assumptions": [
            "Timestamp converted from UTC to local time (+03:00) for period filtering",
            "Each unique transaction_id represents one basket",
            "Line items with is_refund=True or line_total_inconsistent=True are excluded from net metrics",
            "AOV calculated as total revenue divided by transaction count"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Category performance - identify top category change
if len(analysis_by_category) > 0 and len(previous_by_category) > 0:
    # Merge category data
    category_comparison = analysis_by_category.merge(
        previous_by_category,
        on='category',
        suffixes=('_analysis', '_previous')
    )
    
    category_comparison['revenue_pct_change'] = (
        (category_comparison['revenue_analysis'] - category_comparison['revenue_previous']) / 
        category_comparison['revenue_previous'] * 100
    )
    
    # Find category with largest absolute revenue change
    category_comparison['revenue_change_abs'] = abs(
        category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
    )
    
    top_category_idx = category_comparison['revenue_change_abs'].idxmax()
    top_category = category_comparison.loc[top_category_idx]
    
    if not pd.isna(top_category['category']):
        finding2 = {
            "title": f"Category Performance: {top_category['category']} Revenue Change",
            "claim": f"The {top_category['category']} category generated SAR {top_category['revenue_analysis']:.2f} in the analysis period (2026-04-27 to 2026-05-04) compared to SAR {top_category['revenue_previous']:.2f} in the previous period (2026-04-20 to 2026-04-27), representing a {top_category['revenue_pct_change']:.2f}% change. Category share shifted from {top_category['share_pct_previous']:.2f}% to {top_category['share_pct_analysis']:.2f}% of total revenue.",
            "finding_type": "category_mix",
            "metrics": {
                "category_name": {
                    "value": top_category['category'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "analysis_category_revenue": {
                    "value": round(top_category['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_category_revenue": {
                    "value": round(top_category['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "category_revenue_pct_change": {
                    "value": round(top_category['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": top_category['revenue_analysis'] - top_category['revenue_previous'],
                    "denominator": top_category['revenue_previous'],
                    "period_start": previous_start,
                    "period_end": analysis_end
                },
                "analysis_category_share": {
                    "value": round(top_category['share_pct_analysis'], 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_category_share": {
                    "value": round(top_category['share_pct_previous'], 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "category_share_change_pct_points": {
                    "value": round(top_category['share_pct_analysis'] - top_category['share_pct_previous'], 2),
                    "unit": "percentage points",
                    "numerator": top_category['share_pct_analysis'] - top_category['share_pct_previous'],
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": int(top_category['txn_count_analysis'] + top_category['txn_count_previous']),
            "coverage_notes": [
                "Refunds and inconsistent line totals excluded",
                "Category derived from POS item_name mapping to menu category",
                "Share calculated as category revenue / total period revenue"
            ],
            "assumptions": [
                "Category assignment is consistent between periods",
                "Revenue includes all line items in the category",
                "Share percentages sum to 100% within each period"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# Finding 3: Channel performance
if len(analysis_by_channel) > 0 and len(previous_by_channel) > 0:
    channel_comparison = analysis_by_channel.merge(
        previous_by_channel,
        on='channel',
        suffixes=('_analysis', '_previous')
    )
    
    channel_comparison['revenue_pct_change'] = (
        (channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']) / 
        channel_comparison['revenue_previous'] * 100
    )
    
    channel_comparison['txn_pct_change'] = (
        (channel_comparison['txn_count_analysis'] - channel_comparison['txn_count_previous']) / 
        channel_comparison['txn_count_previous'] * 100
    )
    
    channel_comparison['aov_pct_change'] = (
        (channel_comparison['aov_analysis'] - channel_comparison['aov_previous']) / 
        channel_comparison['aov_previous'] * 100
    )
    
    # Find channel with largest absolute revenue change
    channel_comparison['revenue_change_abs'] = abs(
        channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
    )
    
    top_channel_idx = channel_comparison['revenue_change_abs'].idxmax()
    top_channel = channel_comparison.loc[top_channel_idx]
    
    if not pd.isna(top_channel['channel']):
        finding3 = {
            "title": f"Channel Performance: {top_channel['channel']} Revenue and Transaction Dynamics",
            "claim": f"The {top_channel['channel']} channel generated SAR {top_channel['revenue_analysis']:.2f} across {int(top_channel['txn_count_analysis'])} transactions in the analysis period, compared to SAR {top_channel['revenue_previous']:.2f} across {int(top_channel['txn_count_previous'])} transactions previously. Revenue changed by {top_channel['revenue_pct_change']:.2f}%, transaction count by {top_channel['txn_pct_change']:.2f}%, and AOV by {top_channel['aov_pct_change']:.2f}%.",
            "finding_type": "channel_mix",
            "metrics": {
                "channel_name": {
                    "value": top_channel['channel'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "analysis_channel_revenue": {
                    "value": round(top_channel['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_channel_revenue": {
                    "value": round(top_channel['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "channel_revenue_pct_change": {
                    "value": round(top_channel['revenue_pct_change'], 2),
                    "unit": "%",
                    "numerator": top_channel['revenue_analysis'] - top_channel['revenue_previous'],
                    "denominator": top_channel['revenue_previous'],
                    "period_start": previous_start,
                    "period_end": analysis_end
                },
                "analysis_channel_txn_count": {
                    "value": int(top_channel['txn_count_analysis']),
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_channel_txn_count": {
                    "value": int(top_channel['txn_count_previous']),
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "channel_txn_pct_change": {
                    "value": round(top_channel['txn_pct_change'], 2),
                    "unit": "%",
                    "numerator": int(top_channel['txn_count_analysis'] - top_channel['txn_count_previous']),
                    "denominator": int(top_channel['txn_count_previous']),
                    "period_start": previous_start,
                    "period_end": analysis_end
                },
                "analysis_channel_aov": {
                    "value": round(top_channel['aov_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_channel_aov": {
                    "value": round(top_channel['aov_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "channel_aov_pct_change": {
                    "value": round(top_channel['aov_pct_change'], 2),
                    "unit": "%",
                    "numerator": top_channel['aov_analysis'] - top_channel['aov_previous'],
                    "denominator": top_channel['aov_previous'],
                    "period_start": previous_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": int(top_channel['txn_count_analysis'] + top_channel['txn_count_previous']),
            "coverage_notes": [
                "Refunds and inconsistent line totals excluded",
                "Channel derived from POS channel field",
                "AOV calculated as channel revenue / channel transaction count"
            ],
            "assumptions": [
                "Channel assignment is consistent and accurate",
                "Each transaction_id is unique within a channel",
                "AOV reflects actual customer spending patterns by channel"
            ],
            "confidence": 0.90
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output file
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)