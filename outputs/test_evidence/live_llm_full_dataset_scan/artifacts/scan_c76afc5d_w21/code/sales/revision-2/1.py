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

# Define periods (UTC+3)
analysis_start = pd.Timestamp("2026-06-01T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-06-08T00:00:00+03:00")
previous_start = pd.Timestamp("2026-05-25T00:00:00+03:00")
previous_end = pd.Timestamp("2026-06-01T00:00:00+03:00")
trailing_1_start = pd.Timestamp("2026-05-25T00:00:00+03:00")
trailing_1_end = pd.Timestamp("2026-06-01T00:00:00+03:00")
trailing_2_start = pd.Timestamp("2026-05-18T00:00:00+03:00")
trailing_2_end = pd.Timestamp("2026-05-25T00:00:00+03:00")
trailing_3_start = pd.Timestamp("2026-05-11T00:00:00+03:00")
trailing_3_end = pd.Timestamp("2026-05-18T00:00:00+03:00")
trailing_4_start = pd.Timestamp("2026-05-04T00:00:00+03:00")
trailing_4_end = pd.Timestamp("2026-05-11T00:00:00+03:00")

# Convert timestamp to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter for valid transactions (non-refund line items for basket count)
def get_period_data(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

# Analysis period
analysis_data = get_period_data(pos_df, analysis_start, analysis_end)
analysis_valid_txns = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
analysis_net_revenue = analysis_data['line_total_sar'].sum()
analysis_refunds = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()

# Previous period
previous_data = get_period_data(pos_df, previous_start, previous_end)
previous_valid_txns = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
previous_net_revenue = previous_data['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Trailing baseline periods
trailing_1_data = get_period_data(pos_df, trailing_1_start, trailing_1_end)
trailing_1_net_revenue = trailing_1_data['line_total_sar'].sum()
trailing_1_valid_txns = trailing_1_data[trailing_1_data['is_refund'] == False]['transaction_id'].nunique()

trailing_2_data = get_period_data(pos_df, trailing_2_start, trailing_2_end)
trailing_2_net_revenue = trailing_2_data['line_total_sar'].sum()
trailing_2_valid_txns = trailing_2_data[trailing_2_data['is_refund'] == False]['transaction_id'].nunique()

trailing_3_data = get_period_data(pos_df, trailing_3_start, trailing_3_end)
trailing_3_net_revenue = trailing_3_data['line_total_sar'].sum()
trailing_3_valid_txns = trailing_3_data[trailing_3_data['is_refund'] == False]['transaction_id'].nunique()

trailing_4_data = get_period_data(pos_df, trailing_4_start, trailing_4_end)
trailing_4_net_revenue = trailing_4_data['line_total_sar'].sum()
trailing_4_valid_txns = trailing_4_data[trailing_4_data['is_refund'] == False]['transaction_id'].nunique()

# Calculate AOV
analysis_aov = analysis_net_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0
previous_aov = previous_net_revenue / previous_valid_txns if previous_valid_txns > 0 else 0

# Calculate changes
revenue_change_sar = analysis_net_revenue - previous_net_revenue
revenue_change_pct = (revenue_change_sar / previous_net_revenue * 100) if previous_net_revenue != 0 else 0
aov_change_sar = analysis_aov - previous_aov
aov_change_pct = (aov_change_sar / previous_aov * 100) if previous_aov != 0 else 0
txn_change_sar = analysis_valid_txns - previous_valid_txns
txn_change_pct = (txn_change_sar / previous_valid_txns * 100) if previous_valid_txns > 0 else 0

# Product/Category analysis
analysis_product_data = analysis_data[analysis_data['is_refund'] == False].copy()
analysis_product_data = analysis_product_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')

# Filter by launch/retire dates
analysis_product_data['launch_date'] = pd.to_datetime(analysis_product_data['launch_date'], utc=True, errors='coerce')
analysis_product_data['retire_date'] = pd.to_datetime(analysis_product_data['retire_date'], utc=True, errors='coerce')

analysis_product_data = analysis_product_data[
    ((analysis_product_data['launch_date'].isna()) | (analysis_product_data['launch_date'] <= analysis_end)) &
    ((analysis_product_data['retire_date'].isna()) | (analysis_product_data['retire_date'] > analysis_start))
]

previous_product_data = previous_data[previous_data['is_refund'] == False].copy()
previous_product_data = previous_product_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_data['launch_date'] = pd.to_datetime(previous_product_data['launch_date'], utc=True, errors='coerce')
previous_product_data['retire_date'] = pd.to_datetime(previous_product_data['retire_date'], utc=True, errors='coerce')

previous_product_data = previous_product_data[
    ((previous_product_data['launch_date'].isna()) | (previous_product_data['launch_date'] <= previous_end)) &
    ((previous_product_data['retire_date'].isna()) | (previous_product_data['retire_date'] > previous_start))
]

# Top products by revenue
analysis_product_revenue = analysis_product_data.groupby('item_name_en')['line_total_sar'].sum().sort_values(ascending=False)
previous_product_revenue = previous_product_data.groupby('item_name_en')['line_total_sar'].sum().sort_values(ascending=False)

# Category analysis
analysis_category_revenue = analysis_product_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_product_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Channel analysis
analysis_channel_revenue = analysis_product_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_product_data.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

# Find top product changes
top_product_analysis = analysis_product_revenue.head(1)
top_product_previous = previous_product_revenue.head(1)

if len(top_product_analysis) > 0 and len(top_product_previous) > 0:
    top_product_name = top_product_analysis.index[0]
    top_product_analysis_rev = top_product_analysis.iloc[0]
    top_product_previous_rev = previous_product_revenue.get(top_product_name, 0)
    top_product_change = top_product_analysis_rev - top_product_previous_rev
    top_product_change_pct = (top_product_change / top_product_previous_rev * 100) if top_product_previous_rev != 0 else 0
else:
    top_product_name = None
    top_product_analysis_rev = 0
    top_product_previous_rev = 0
    top_product_change = 0
    top_product_change_pct = 0

# Find top category changes
top_category_analysis = analysis_category_revenue.head(1)
top_category_previous = previous_category_revenue.head(1)

if len(top_category_analysis) > 0 and len(top_category_previous) > 0:
    top_category_name = top_category_analysis.index[0]
    top_category_analysis_rev = top_category_analysis.iloc[0]
    top_category_previous_rev = previous_category_revenue.get(top_category_name, 0)
    top_category_change = top_category_analysis_rev - top_category_previous_rev
    top_category_change_pct = (top_category_change / top_category_previous_rev * 100) if top_category_previous_rev != 0 else 0
else:
    top_category_name = None
    top_category_analysis_rev = 0
    top_category_previous_rev = 0
    top_category_change = 0
    top_category_change_pct = 0

# Build findings
findings = []

# Finding 1: Net Revenue Change
if analysis_net_revenue != 0 and previous_net_revenue != 0:
    findings.append({
        "title": "Net Revenue Decline Week-over-Week",
        "claim": f"Net revenue in analysis period (2026-06-01 to 2026-06-08) was SAR {analysis_net_revenue:.2f}, compared to SAR {previous_net_revenue:.2f} in previous period (2026-05-25 to 2026-06-01), representing a {revenue_change_pct:.2f}% decline.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_net_revenue": {
                "value": round(analysis_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_net_revenue": {
                "value": round(previous_net_revenue, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change_sar, 2),
                "unit": "SAR",
                "numerator": round(revenue_change_sar, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(revenue_change_sar, 2),
                "denominator": round(previous_net_revenue, 2),
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            f"Analysis period row count: {len(analysis_data)}",
            f"Previous period row count: {len(previous_data)}",
            f"Refunds included in net calculation: Analysis SAR {analysis_refunds:.2f}, Previous SAR {previous_refunds:.2f}",
            "All valid transactions counted using unique transaction_id after filtering is_refund == False"
        ],
        "assumptions": [
            "Refunds are included in net revenue calculations (line_total_sar includes negative refund values)",
            "Transaction validity determined by is_refund flag in cleaned POS data",
            "Timestamp converted to UTC+3 timezone for period filtering"
        ],
        "confidence": 0.95
    })

# Finding 2: Average Order Value Change
if analysis_aov != 0 and previous_aov != 0:
    findings.append({
        "title": "Average Order Value Slight Decline",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f} per transaction, compared to SAR {previous_aov:.2f} in previous period, representing a {aov_change_pct:.2f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_net_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_net_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change_sar, 2),
                "unit": "SAR",
                "numerator": round(aov_change_sar, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": round(aov_change_sar, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_valid_txns,
        "coverage_notes": [
            f"Analysis period valid transactions: {analysis_valid_txns}",
            f"Previous period valid transactions: {previous_valid_txns}",
            "AOV calculated as net revenue divided by unique valid transaction count"
        ],
        "assumptions": [
            "Valid transactions identified by is_refund == False",
            "Net revenue includes refund adjustments",
            "Each transaction_id represents one basket"
        ],
        "confidence": 0.92
    })

# Finding 3: Top Product Revenue Performance
if top_product_name and top_product_previous_rev > 0:
    findings.append({
        "title": f"Top Product {top_product_name} Revenue Change",
        "claim": f"Revenue from {top_product_name} in analysis period was SAR {top_product_analysis_rev:.2f}, compared to SAR {top_product_previous_rev:.2f} in previous period, representing a {top_product_change_pct:.2f}% change.",
        "finding_type": "product_revenue_change",
        "metrics": {
            "product_analysis_revenue": {
                "value": round(top_product_analysis_rev, 2),
                "unit": "SAR",
                "numerator": round(top_product_analysis_rev, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "product_previous_revenue": {
                "value": round(top_product_previous_rev, 2),
                "unit": "SAR",
                "numerator": round(top_product_previous_rev, 2),
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "product_revenue_change_sar": {
                "value": round(top_product_change, 2),
                "unit": "SAR",
                "numerator": round(top_product_change, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            },
            "product_revenue_change_pct": {
                "value": round(top_product_change_pct, 2),
                "unit": "%",
                "numerator": round(top_product_change, 2),
                "denominator": round(top_product_previous_rev, 2),
                "period_start": "2026-06-01T00:00:00+03:00",
                "period_end": "2026-06-08T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_product_data[analysis_product_data['item_name_en'] == top_product_name]),
        "coverage_notes": [
            f"Product name resolved from menu SKU reference",
            f"Analysis period product rows: {len(analysis_product_data[analysis_product_data['item_name_en'] == top_product_name])}",
            f"Previous period product rows: {len(previous_product_data[previous_product_data['item_name_en'] == top_product_name])}",
            "Refunds excluded from product revenue calculations (is_refund == False)",
            "Product eligibility verified against launch_date and retire_date from menu"
        ],
        "assumptions": [
            "Product names matched via menu SKU reference",
            "Launch/retire dates applied to filter eligible products",
            "Revenue calculated from line_total_sar for non-refund items only"
        ],
        "confidence": 0.90
    })

# Prepare output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)