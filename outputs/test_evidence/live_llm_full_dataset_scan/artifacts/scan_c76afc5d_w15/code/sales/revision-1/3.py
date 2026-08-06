import os
import json
import pandas as pd
from datetime import datetime, timezone

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load POS and Menu data
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_period = {
    "start": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc),
    "end": datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)
}

previous_period = {
    "start": datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc),
    "end": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
}

trailing_baseline_periods = [
    {
        "start": datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
    }
]

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Function to filter data by period
def filter_by_period(df, period_start, period_end):
    return df[(df['timestamp'] >= period_start) & (df['timestamp'] < period_end)]

# Analysis period data
analysis_data = filter_by_period(pos_df, analysis_period['start'], analysis_period['end'])

# Previous period data
previous_data = filter_by_period(pos_df, previous_period['start'], previous_period['end'])

# Trailing baseline data (combine all baseline periods)
trailing_baseline_data = pd.concat([
    filter_by_period(pos_df, period['start'], period['end'])
    for period in trailing_baseline_periods
])

# Calculate metrics for analysis period
analysis_valid_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_quantity = analysis_data['quantity'].sum()

# Calculate metrics for previous period
previous_valid_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_quantity = previous_data['quantity'].sum()

# Calculate metrics for trailing baseline
trailing_valid_transactions = trailing_baseline_data['transaction_id'].nunique()
trailing_revenue = trailing_baseline_data['line_total_sar'].sum()
trailing_quantity = trailing_baseline_data['quantity'].sum()

# Calculate AOV
analysis_aov = analysis_revenue / analysis_valid_transactions if analysis_valid_transactions > 0 else 0
previous_aov = previous_revenue / previous_valid_transactions if previous_valid_transactions > 0 else 0
trailing_aov = trailing_revenue / trailing_valid_transactions if trailing_valid_transactions > 0 else 0

# Calculate changes
revenue_change = analysis_revenue - previous_revenue
revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

transaction_change = analysis_valid_transactions - previous_valid_transactions
transaction_change_pct = (transaction_change / previous_valid_transactions * 100) if previous_valid_transactions > 0 else 0

aov_change = analysis_aov - previous_aov
aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Analyze by category
analysis_by_category = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'quantity', 'transactions']

previous_by_category = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'quantity', 'transactions']

# Merge category data
category_comparison = analysis_by_category.merge(
    previous_by_category,
    on='category',
    suffixes=('_analysis', '_previous')
)

category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_change_pct'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100).fillna(0)

# Analyze by channel
analysis_by_channel = analysis_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_channel.columns = ['channel', 'revenue', 'transactions']

previous_by_channel = previous_data.groupby('channel').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_channel.columns = ['channel', 'revenue', 'transactions']

# Merge channel data
channel_comparison = analysis_by_channel.merge(
    previous_by_channel,
    on='channel',
    suffixes=('_analysis', '_previous')
)

channel_comparison['revenue_change'] = channel_comparison['revenue_analysis'] - channel_comparison['revenue_previous']
channel_comparison['revenue_change_pct'] = (channel_comparison['revenue_change'] / channel_comparison['revenue_previous'] * 100).fillna(0)

# Analyze by product (SKU)
analysis_by_sku = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_sku.columns = ['sku', 'revenue', 'quantity', 'transactions']

previous_by_sku = previous_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_sku.columns = ['sku', 'revenue', 'quantity', 'transactions']

trailing_by_sku = trailing_baseline_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
trailing_by_sku.columns = ['sku', 'revenue', 'quantity', 'transactions']

# Merge SKU data
sku_comparison = analysis_by_sku.merge(
    previous_by_sku,
    on='sku',
    suffixes=('_analysis', '_previous'),
    how='outer'
).fillna(0)

sku_comparison = sku_comparison.merge(
    trailing_by_sku,
    on='sku',
    suffixes=('', '_trailing'),
    how='left'
).fillna(0)

sku_comparison['revenue_change'] = sku_comparison['revenue_analysis'] - sku_comparison['revenue_previous']
sku_comparison['revenue_change_pct'] = (sku_comparison['revenue_change'] / sku_comparison['revenue_previous'] * 100).fillna(0)

# Merge with menu to get product names and launch dates
sku_comparison = sku_comparison.merge(
    menu_df[['sku', 'item_en', 'launch_date', 'retire_date']],
    on='sku',
    how='left'
)

# Convert launch_date and retire_date to UTC-aware datetime
sku_comparison['launch_date'] = pd.to_datetime(sku_comparison['launch_date'], errors='coerce', utc=True)
sku_comparison['retire_date'] = pd.to_datetime(sku_comparison['retire_date'], errors='coerce', utc=True)

# Check if product was active during analysis period
sku_comparison['active_in_analysis'] = (
    (sku_comparison['launch_date'].isna() | (sku_comparison['launch_date'] <= analysis_period['end'])) &
    (sku_comparison['retire_date'].isna() | (sku_comparison['retire_date'] > analysis_period['start']))
)

# Check if product was active during previous period
sku_comparison['active_in_previous'] = (
    (sku_comparison['launch_date'].isna() | (sku_comparison['launch_date'] <= previous_period['end'])) &
    (sku_comparison['retire_date'].isna() | (sku_comparison['retire_date'] > previous_period['start']))
)

# Sort by revenue change
sku_comparison_sorted = sku_comparison.sort_values('revenue_change', ascending=False)

# Identify findings
findings = []

# Finding 1: Overall revenue change
if abs(revenue_change_pct) > 0.1:  # More than 0.1% change
    findings.append({
        "title": "Overall Revenue Change",
        "claim": f"Total revenue in analysis period (2026-04-20 to 2026-04-27) was SAR {analysis_revenue:.2f}, compared to SAR {previous_revenue:.2f} in previous period (2026-04-13 to 2026-04-20), representing a {revenue_change_pct:.1f}% change.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-20T00:00:00+00:00"
            },
            "revenue_change": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_transactions),
        "coverage_notes": [
            f"Analysis period: {len(analysis_data)} line items from {analysis_valid_transactions} unique transactions",
            f"Previous period: {len(previous_data)} line items from {previous_valid_transactions} unique transactions"
        ],
        "assumptions": [
            "line_total_sar includes refunds as negative values",
            "transaction_id uniqueness defines basket count",
            "Periods are non-overlapping and consecutive"
        ],
        "confidence": 0.95
    })

# Finding 2: AOV change
if abs(aov_change_pct) > 0.1:  # More than 0.1% change
    findings.append({
        "title": "Average Order Value Change",
        "claim": f"Average order value in analysis period was SAR {analysis_aov:.2f}, compared to SAR {previous_aov:.2f} in previous period, representing a {aov_change_pct:.1f}% change.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_period_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": int(analysis_valid_transactions),
                "period_start": "2026-04-20T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            },
            "previous_period_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": int(previous_valid_transactions),
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-20T00:00:00+00:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            },
            "aov_change_pct": {
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-04-13T00:00:00+00:00",
                "period_end": "2026-04-27T00:00:00+00:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(analysis_valid_transactions),
        "coverage_notes": [
            f"Analysis period: {analysis_valid_transactions} unique transactions",
            f"Previous period: {previous_valid_transactions} unique transactions"
        ],
        "assumptions": [
            "AOV calculated as total revenue divided by unique transaction_id count",
            "line_total_sar includes refunds as negative values"
        ],
        "confidence": 0.95
    })

# Finding 3: Category mix change - find most significant category change
if len(category_comparison) > 0:
    category_comparison_sorted = category_comparison.sort_values('revenue_change_pct', ascending=False)
    top_category = category_comparison_sorted.iloc[0]
    
    if abs(top_category['revenue_change_pct']) > 5:  # More than 5% change
        findings.append({
            "title": f"Category Mix Shift: {top_category['category']}",
            "claim": f"The {top_category['category']} category generated SAR {top_category['revenue_analysis']:.2f} in the analysis period, compared to SAR {top_category['revenue_previous']:.2f} in the previous period, representing a {top_category['revenue_change_pct']:.1f}% change.",
            "finding_type": "category_mix_change",
            "metrics": {
                "analysis_period_category_revenue": {
                    "value": round(top_category['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "previous_period_category_revenue": {
                    "value": round(top_category['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-20T00:00:00+00:00"
                },
                "category_revenue_change": {
                    "value": round(top_category['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category['revenue_change'], 2),
                    "denominator": round(top_category['revenue_previous'], 2),
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "category_revenue_change_pct": {
                    "value": round(top_category['revenue_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_category['revenue_change'], 2),
                    "denominator": round(top_category['revenue_previous'], 2),
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": int(top_category['transactions_analysis']),
            "coverage_notes": [
                f"Analysis period: {int(top_category['transactions_analysis'])} transactions in {top_category['category']}",
                f"Previous period: {int(top_category['transactions_previous'])} transactions in {top_category['category']}"
            ],
            "assumptions": [
                "Category assignment from POS data",
                "line_total_sar includes refunds as negative values"
            ],
            "confidence": 0.90
        })

# Finding 4: Top product change with trailing baseline
if len(sku_comparison_sorted) > 0:
    top_sku = sku_comparison_sorted.iloc[0]
    
    if abs(top_sku['revenue_change_pct']) > 10 and top_sku['revenue_analysis'] > 0:  # More than 10% change
        trailing_revenue_val = top_sku.get('revenue_trailing', 0)
        trailing_qty_val = top_sku.get('quantity_trailing', 0)
        
        findings.append({
            "title": f"Top Product Change: {top_sku['item_en']}",
            "claim": f"Product {top_sku['item_en']} (SKU: {top_sku['sku']}) generated SAR {top_sku['revenue_analysis']:.2f} in the analysis period, compared to SAR {top_sku['revenue_previous']:.2f} in the previous period, representing a {top_sku['revenue_change_pct']:.1f}% change. Trailing baseline (4 weeks prior) showed SAR {trailing_revenue_val:.2f}.",
            "finding_type": "product_revenue_change",
            "metrics": {
                "product_revenue_analysis": {
                    "value": round(top_sku['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "product_revenue_previous": {
                    "value": round(top_sku['revenue_previous'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-20T00:00:00+00:00"
                },
                "product_revenue_trailing_baseline": {
                    "value": round(trailing_revenue_val, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-04-13T00:00:00+00:00"
                },
                "product_revenue_change": {
                    "value": round(top_sku['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_sku['revenue_change'], 2),
                    "denominator": round(top_sku['revenue_previous'], 2),
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "product_revenue_change_pct": {
                    "value": round(top_sku['revenue_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_sku['revenue_change'], 2),
                    "denominator": round(top_sku['revenue_previous'], 2),
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "product_quantity_analysis": {
                    "value": round(top_sku['quantity_analysis'], 2),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-20T00:00:00+00:00",
                    "period_end": "2026-04-27T00:00:00+00:00"
                },
                "product_quantity_previous": {
                    "value": round(top_sku['quantity_previous'], 2),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00+00:00",
                    "period_end": "2026-04-20T00:00:00+00:00"
                },
                "product_quantity_trailing_baseline": {
                    "value": round(trailing_qty_val, 2),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-04-13T00:00:00+00:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(top_sku['transactions_analysis']),
            "coverage_notes": [
                f"Analysis period: {int(top_sku['transactions_analysis'])} transactions",
                f"Previous period: {int(top_sku['transactions_previous'])} transactions",
                f"Trailing baseline: {int(top_sku.get('transactions_trailing', 0))} transactions"
            ],
            "assumptions": [
                "SKU and product name from menu data",
                "line_total_sar includes refunds as negative values",
                "Trailing baseline covers 3 weeks prior to analysis period (2026-03-23 to 2026-04-13)"
            ],
            "confidence": 0.90
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
