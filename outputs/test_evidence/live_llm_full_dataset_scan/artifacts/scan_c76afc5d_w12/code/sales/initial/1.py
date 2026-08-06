import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
analysis_end = datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_start = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_end = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Calculate metrics for analysis period
analysis_valid_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_refund_count = analysis_data[analysis_data['is_refund'] == True].shape[0]
analysis_refund_value = analysis_data[analysis_data['is_refund'] == True]['line_total_sar'].sum()

# Calculate metrics for previous period
previous_valid_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_refund_count = previous_data[previous_data['is_refund'] == True].shape[0]
previous_refund_value = previous_data[previous_data['is_refund'] == True]['line_total_sar'].sum()

# Calculate trailing baseline (average of 4 weeks)
trailing_baseline_revenue = 0
trailing_baseline_transactions = 0
trailing_baseline_count = 0

for period in [
    (datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))),
    (datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))), 
     datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone(timedelta(hours=3))))
]:
    period_data = pos_df[(pos_df['timestamp'] >= period[0]) & (pos_df['timestamp'] < period[1])]
    trailing_baseline_revenue += period_data['line_total_sar'].sum()
    trailing_baseline_transactions += period_data['transaction_id'].nunique()
    trailing_baseline_count += 1

trailing_baseline_revenue = trailing_baseline_revenue / trailing_baseline_count if trailing_baseline_count > 0 else 0
trailing_baseline_transactions = trailing_baseline_transactions / trailing_baseline_count if trailing_baseline_count > 0 else 0

# Calculate AOV
analysis_aov = analysis_revenue / analysis_valid_transactions if analysis_valid_transactions > 0 else 0
previous_aov = previous_revenue / previous_valid_transactions if previous_valid_transactions > 0 else 0
trailing_baseline_aov = trailing_baseline_revenue / trailing_baseline_transactions if trailing_baseline_transactions > 0 else 0

# Analyze product mix
analysis_product_mix = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']
analysis_product_mix['pct_revenue'] = (analysis_product_mix['revenue'] / analysis_product_mix['revenue'].sum() * 100).round(2)

previous_product_mix = previous_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_product_mix.columns = ['sku', 'revenue', 'quantity', 'transactions']
previous_product_mix['pct_revenue'] = (previous_product_mix['revenue'] / previous_product_mix['revenue'].sum() * 100).round(2)

# Merge with menu to get product names and launch dates
analysis_product_mix = analysis_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')
previous_product_mix = previous_product_mix.merge(menu_df[['sku', 'item_en', 'launch_date', 'retire_date']], on='sku', how='left')

# Analyze channel mix
analysis_channel_mix = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_channel_mix.columns = ['channel', 'revenue', 'transactions']
analysis_channel_mix['pct_revenue'] = (analysis_channel_mix['revenue'] / analysis_channel_mix['revenue'].sum() * 100).round(2)

previous_channel_mix = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_channel_mix.columns = ['channel', 'revenue', 'transactions']
previous_channel_mix['pct_revenue'] = (previous_channel_mix['revenue'] / previous_channel_mix['revenue'].sum() * 100).round(2)

# Analyze category mix
analysis_category_mix = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_category_mix.columns = ['category', 'revenue', 'quantity', 'transactions']
analysis_category_mix['pct_revenue'] = (analysis_category_mix['revenue'] / analysis_category_mix['revenue'].sum() * 100).round(2)

previous_category_mix = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_category_mix.columns = ['category', 'revenue', 'quantity', 'transactions']
previous_category_mix['pct_revenue'] = (previous_category_mix['revenue'] / previous_category_mix['revenue'].sum() * 100).round(2)

# Identify top products by revenue change
top_analysis = analysis_product_mix.nlargest(5, 'revenue')
top_previous = previous_product_mix.nlargest(5, 'revenue')

# Calculate revenue change
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

# Calculate transaction change
transaction_change = analysis_valid_transactions - previous_valid_transactions
transaction_change_pct = (transaction_change / previous_valid_transactions * 100) if previous_valid_transactions > 0 else 0

# Calculate AOV change
aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Prepare findings
findings = []

# Finding 1: Revenue change
if abs(revenue_change_pct) >= 5:  # Only report if change is >= 5%
    findings.append({
        "title": "Weekly Revenue Change",
        "claim": f"Revenue in analysis week (Mar 30 - Apr 6) was {analysis_revenue:.2f} SAR, compared to {previous_revenue:.2f} SAR in previous week (Mar 23-30), representing a {revenue_change_pct:.1f}% change.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "previous_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_data.shape[0]),
        "coverage_notes": [
            f"Analysis period: {analysis_data.shape[0]} line items from {analysis_valid_transactions} unique transactions",
            f"Previous period: {previous_data.shape[0]} line items from {previous_valid_transactions} unique transactions",
            f"Refunds included in net revenue: {analysis_refund_count} refund items ({analysis_refund_value:.2f} SAR) in analysis period"
        ],
        "assumptions": [
            "line_total_sar represents realized net revenue including refunds",
            "transaction_id uniqueness defines basket count",
            "All data quality flags respected per cleaned artifact"
        ],
        "confidence": 0.95
    })

# Finding 2: Transaction count change
if abs(transaction_change_pct) >= 5:
    findings.append({
        "title": "Transaction Volume Change",
        "claim": f"Valid transaction count in analysis week was {analysis_valid_transactions}, compared to {previous_valid_transactions} in previous week, representing a {transaction_change_pct:.1f}% change.",
        "finding_type": "transaction_volume",
        "metrics": {
            "analysis_transactions": {
                "value": int(analysis_valid_transactions),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "previous_transactions": {
                "value": int(previous_valid_transactions),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "transaction_change": {
                "value": int(transaction_change),
                "unit": "transactions",
                "numerator": int(transaction_change),
                "denominator": int(previous_valid_transactions),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "transaction_change_pct": {
                "value": round(transaction_change_pct, 1),
                "unit": "%",
                "numerator": int(transaction_change),
                "denominator": int(previous_valid_transactions),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_transactions),
        "coverage_notes": [
            f"Analysis period: {analysis_valid_transactions} unique transaction_ids",
            f"Previous period: {previous_valid_transactions} unique transaction_ids"
        ],
        "assumptions": [
            "transaction_id uniqueness defines basket count",
            "All transactions with valid transaction_id are counted"
        ],
        "confidence": 0.95
    })

# Finding 3: AOV change
if abs(aov_change_pct) >= 5:
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis week was {analysis_aov:.2f} SAR, compared to {previous_aov:.2f} SAR in previous week, representing a {aov_change_pct:.1f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": int(analysis_valid_transactions),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": int(previous_valid_transactions),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-03-30T00:00:00+03:00",
                "period_end": "2026-04-06T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_transactions),
        "coverage_notes": [
            f"Analysis period: {analysis_valid_transactions} transactions, {analysis_revenue:.2f} SAR revenue",
            f"Previous period: {previous_valid_transactions} transactions, {previous_revenue:.2f} SAR revenue"
        ],
        "assumptions": [
            "AOV calculated as total revenue / unique transaction count",
            "Refunds included in net revenue calculation"
        ],
        "confidence": 0.95
    })

# If no findings meet threshold, check for category mix changes
if len(findings) == 0:
    # Check for significant category shifts
    analysis_cat_top = analysis_category_mix.nlargest(1, 'revenue')
    previous_cat_top = previous_category_mix.nlargest(1, 'revenue')
    
    if not analysis_cat_top.empty and not previous_cat_top.empty:
        analysis_cat_pct = analysis_cat_top.iloc[0]['pct_revenue']
        previous_cat_pct = previous_cat_top.iloc[0]['pct_revenue']
        cat_change = analysis_cat_pct - previous_cat_pct
        
        if abs(cat_change) >= 3:
            findings.append({
                "title": "Category Mix Shift",
                "claim": f"Top category {analysis_cat_top.iloc[0]['category']} represented {analysis_cat_pct:.1f}% of revenue in analysis week vs {previous_cat_pct:.1f}% in previous week.",
                "finding_type": "category_mix",
                "metrics": {
                    "analysis_top_category_pct": {
                        "value": round(analysis_cat_pct, 1),
                        "unit": "%",
                        "numerator": round(analysis_cat_top.iloc[0]['revenue'], 2),
                        "denominator": round(analysis_category_mix['revenue'].sum(), 2),
                        "period_start": "2026-03-30T00:00:00+03:00",
                        "period_end": "2026-04-06T00:00:00+03:00"
                    },
                    "previous_top_category_pct": {
                        "value": round(previous_cat_pct, 1),
                        "unit": "%",
                        "numerator": round(previous_cat_top.iloc[0]['revenue'], 2),
                        "denominator": round(previous_category_mix['revenue'].sum(), 2),
                        "period_start": "2026-03-23T00:00:00+03:00",
                        "period_end": "2026-03-30T00:00:00+03:00"
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": int(analysis_data.shape[0]),
                "coverage_notes": [
                    f"Analysis period: {len(analysis_category_mix)} categories",
                    f"Previous period: {len(previous_category_mix)} categories"
                ],
                "assumptions": [
                    "Category assignment from menu SKU reference",
                    "Revenue calculated from line_total_sar"
                ],
                "confidence": 0.90
            })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
