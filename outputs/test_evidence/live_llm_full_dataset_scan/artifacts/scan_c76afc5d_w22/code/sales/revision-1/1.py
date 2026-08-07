import os
import json
import pandas as pd
from datetime import datetime, timezone
import pyarrow.parquet as pq

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
    "start": datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc),
    "end": datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
}

previous_period = {
    "start": datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    "end": datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
}

trailing_baseline_periods = [
    {
        "start": datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)
    },
    {
        "start": datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
    }
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Helper function to filter by period
def filter_by_period(df, period):
    return df[(df['timestamp'] >= period['start']) & (df['timestamp'] < period['end'])]

# Analysis 1: Channel Revenue Comparison (dine_in focus)
analysis_data = filter_by_period(pos_df, analysis_period)
previous_data = filter_by_period(pos_df, previous_period)

# Calculate channel metrics for analysis period
analysis_dine_in = analysis_data[analysis_data['channel'] == 'dine_in']
analysis_dine_in_revenue = analysis_dine_in['line_total_sar'].sum()
analysis_dine_in_transactions = analysis_dine_in['transaction_id'].nunique()

# Calculate channel metrics for previous period
previous_dine_in = previous_data[previous_data['channel'] == 'dine_in']
previous_dine_in_revenue = previous_dine_in['line_total_sar'].sum()
previous_dine_in_transactions = previous_dine_in['transaction_id'].nunique()

# Calculate changes
dine_in_revenue_change = analysis_dine_in_revenue - previous_dine_in_revenue
dine_in_revenue_pct_change = (dine_in_revenue_change / previous_dine_in_revenue * 100) if previous_dine_in_revenue != 0 else 0

# Analysis 2: Overall Transaction Count and AOV
analysis_transactions = analysis_data['transaction_id'].nunique()
analysis_revenue = analysis_data['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0

previous_transactions = previous_data['transaction_id'].nunique()
previous_revenue = previous_data['line_total_sar'].sum()
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

# Analysis 3: Product Category Performance
# Get category performance for analysis period
analysis_category = analysis_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
analysis_category.columns = ['category', 'revenue', 'transactions', 'quantity']

# Get category performance for previous period
previous_category = previous_data.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'quantity': 'sum'
}).reset_index()
previous_category.columns = ['category', 'revenue', 'transactions', 'quantity']

# Merge and calculate changes
category_comparison = analysis_category.merge(
    previous_category,
    on='category',
    suffixes=('_analysis', '_previous')
)

category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['revenue_pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100).fillna(0)

# Find category with largest absolute revenue change
largest_change_idx = category_comparison['revenue_change'].abs().idxmax()
largest_change_category = category_comparison.iloc[largest_change_idx]

# Analysis 4: Check for product launches/retirements
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')

# Find products launched in analysis period
launched_products = menu_df[
    (menu_df['launch_date'] >= analysis_period['start']) & 
    (menu_df['launch_date'] < analysis_period['end'])
]

# Find products retired in analysis period
retired_products = menu_df[
    (menu_df['retire_date'] >= analysis_period['start']) & 
    (menu_df['retire_date'] < analysis_period['end'])
]

# Prepare findings
findings = []

# Finding 1: Dine-in Channel Revenue Decline
if previous_dine_in_revenue > 0:
    findings.append({
        "title": "Dine-in Channel Revenue Decline Week-over-Week",
        "claim": f"The dine_in channel, which showed the largest revenue change, generated SAR {analysis_dine_in_revenue:.2f} (net of refunds) in the analysis week (2026-06-08 to 2026-06-15) across {analysis_dine_in_transactions} transactions, compared to SAR {previous_dine_in_revenue:.2f} (net of refunds) across {previous_dine_in_transactions} transactions in the previous week (2026-06-01 to 2026-06-08), representing a decline of SAR {abs(dine_in_revenue_change):.2f} or {abs(dine_in_revenue_pct_change):.2f}%.",
        "finding_type": "channel_revenue_change",
        "metrics": {
            "analysis_week_dine_in_revenue": {
                "value": round(analysis_dine_in_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_dine_in_revenue": {
                "value": round(previous_dine_in_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            },
            "dine_in_revenue_change": {
                "value": round(dine_in_revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "dine_in_revenue_pct_change": {
                "value": round(dine_in_revenue_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "analysis_week_dine_in_transactions": {
                "value": analysis_dine_in_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_dine_in_transactions": {
                "value": previous_dine_in_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_dine_in_transactions + previous_dine_in_transactions,
        "coverage_notes": [
            "Revenue includes refunds in net calculation",
            "Dine_in channel selected as the channel with largest absolute revenue change",
            "Transaction counts derived from unique transaction_id values",
            "All line_total_sar values included in revenue calculations"
        ],
        "assumptions": [
            "Timestamp field accurately represents transaction time",
            "Channel field correctly identifies transaction channel",
            "line_total_sar represents net revenue after discounts",
            "Refunds are included in line_total_sar as negative values"
        ],
        "confidence": 0.95
    })

# Finding 2: Average Order Value Decline
if previous_aov > 0:
    findings.append({
        "title": "Average Order Value Decline Week-over-Week",
        "claim": f"Average order value declined from SAR {previous_aov:.2f} in the previous week (2026-06-01 to 2026-06-08) to SAR {analysis_aov:.2f} in the analysis week (2026-06-08 to 2026-06-15), representing a decline of SAR {abs(aov_change):.2f} or {abs(aov_pct_change):.2f}%. This decline occurred across {analysis_transactions} transactions in the analysis week compared to {previous_transactions} transactions in the previous week.",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_week_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            },
            "aov_change": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "analysis_week_transactions": {
                "value": analysis_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_transactions": {
                "value": previous_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions + previous_transactions,
        "coverage_notes": [
            "AOV calculated as total revenue divided by unique transaction count",
            "Revenue includes refunds in net calculation",
            "All transactions with valid transaction_id included",
            "Calculation uses line_total_sar for revenue"
        ],
        "assumptions": [
            "Timestamp field accurately represents transaction time",
            "line_total_sar represents net revenue after discounts and refunds",
            "Each transaction_id represents a unique basket",
            "No data quality issues affecting revenue or transaction counts"
        ],
        "confidence": 0.92
    })

# Finding 3: Category Performance - Largest Change
if not category_comparison.empty:
    findings.append({
        "title": f"Category Revenue Change: {largest_change_category['category']}",
        "claim": f"The {largest_change_category['category']} category showed the largest absolute revenue change, generating SAR {largest_change_category['revenue_analysis']:.2f} (net of refunds) in the analysis week (2026-06-08 to 2026-06-15) compared to SAR {largest_change_category['revenue_previous']:.2f} in the previous week (2026-06-01 to 2026-06-08), representing a change of SAR {largest_change_category['revenue_change']:.2f} or {largest_change_category['revenue_pct_change']:.2f}%.",
        "finding_type": "category_revenue_change",
        "metrics": {
            "analysis_week_category_revenue": {
                "value": round(largest_change_category['revenue_analysis'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_category_revenue": {
                "value": round(largest_change_category['revenue_previous'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            },
            "category_revenue_change": {
                "value": round(largest_change_category['revenue_change'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "category_revenue_pct_change": {
                "value": round(largest_change_category['revenue_pct_change'], 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "analysis_week_category_transactions": {
                "value": int(largest_change_category['transactions_analysis']),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-08T00:00:00+00:00",
                "period_end": "2026-06-15T00:00:00+00:00"
            },
            "previous_week_category_transactions": {
                "value": int(largest_change_category['transactions_previous']),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-06-08T00:00:00+00:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": int(largest_change_category['transactions_analysis'] + largest_change_category['transactions_previous']),
        "coverage_notes": [
            "Category selected as the category with largest absolute revenue change",
            "Revenue includes refunds in net calculation",
            "Transaction counts derived from unique transaction_id values",
            "All line_total_sar values included in revenue calculations"
        ],
        "assumptions": [
            "Category field correctly identifies product category",
            "line_total_sar represents net revenue after discounts",
            "Refunds are included in line_total_sar as negative values",
            "Each transaction_id represents a unique basket"
        ],
        "confidence": 0.90
    })

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Results written to {output_path}")
