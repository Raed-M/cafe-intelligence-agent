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
analysis_start = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-06-29T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-06T00:00:00+03:00")

trailing_periods = [
    ("2026-06-29T00:00:00+03:00", "2026-07-06T00:00:00+03:00"),
    ("2026-06-22T00:00:00+03:00", "2026-06-29T00:00:00+03:00"),
    ("2026-06-15T00:00:00+03:00", "2026-06-22T00:00:00+03:00"),
    ("2026-06-08T00:00:00+03:00", "2026-06-15T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data by periods
def filter_by_period(df, start_iso, end_iso):
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

analysis_data = filter_by_period(pos_df, "2026-07-06T00:00:00+03:00", "2026-07-13T00:00:00+03:00")
previous_data = filter_by_period(pos_df, "2026-06-29T00:00:00+03:00", "2026-07-06T00:00:00+03:00")

# Prepare menu reference
menu_dict = {}
for _, row in menu_df.iterrows():
    menu_dict[row['sku']] = {
        'item_en': row['item_en'],
        'item_ar': row['item_ar'],
        'category': row['category'],
        'price_sar': row['price_sar'],
        'launch_date': row['launch_date'],
        'retire_date': row['retire_date']
    }

# Helper function to check if product is eligible in period
def is_eligible(sku, period_start, period_end):
    if sku not in menu_dict:
        return False
    launch = menu_dict[sku]['launch_date']
    retire = menu_dict[sku]['retire_date']
    
    if pd.notna(launch):
        launch_dt = pd.to_datetime(launch)
        if period_start < launch_dt:
            return False
    if pd.notna(retire):
        retire_dt = pd.to_datetime(retire)
        if period_end > retire_dt:
            return False
    return True

# Calculate metrics for analysis period
analysis_valid = analysis_data[analysis_data['is_refund'] == False].copy()
analysis_baskets = analysis_valid['transaction_id'].nunique()
analysis_revenue = analysis_valid['line_total_sar'].sum()
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0

# Calculate metrics for previous period
previous_valid = previous_data[previous_data['is_refund'] == False].copy()
previous_baskets = previous_valid['transaction_id'].nunique()
previous_revenue = previous_valid['line_total_sar'].sum()
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0

# Calculate trailing baseline (average of 4 weeks)
trailing_data_list = []
for start_iso, end_iso in trailing_periods:
    period_data = filter_by_period(pos_df, start_iso, end_iso)
    period_valid = period_data[period_data['is_refund'] == False].copy()
    trailing_data_list.append(period_valid)

trailing_combined = pd.concat(trailing_data_list, ignore_index=True)
trailing_baskets = trailing_combined['transaction_id'].nunique()
trailing_revenue = trailing_combined['line_total_sar'].sum()
trailing_aov = trailing_revenue / trailing_baskets if trailing_baskets > 0 else 0

# Finding 1: Revenue change analysis
findings = []

revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

finding1 = {
    "title": "Weekly Revenue Performance vs Previous Week",
    "claim": f"Analysis week (2026-07-06 to 2026-07-13) generated SAR {analysis_revenue:.2f} in net revenue, a {revenue_pct_change:.1f}% change from previous week's SAR {previous_revenue:.2f}.",
    "finding_type": "revenue_change",
    "metrics": {
        "analysis_week_revenue": {
            "value": round(analysis_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "previous_week_revenue": {
            "value": round(previous_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        },
        "revenue_change_sar": {
            "value": round(revenue_change, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "revenue_change_pct": {
            "value": round(revenue_pct_change, 1),
            "unit": "%",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_valid),
    "coverage_notes": [
        "Analysis period: 2026-07-06 to 2026-07-13",
        "Previous period: 2026-06-29 to 2026-07-06",
        "Refunds excluded from net revenue calculation",
        f"Analysis week transactions: {analysis_baskets}",
        f"Previous week transactions: {previous_baskets}"
    ],
    "assumptions": [
        "line_total_sar represents net realized revenue",
        "is_refund flag correctly identifies refund transactions",
        "transaction_id uniquely identifies baskets"
    ],
    "confidence": 0.95
}
findings.append(finding1)

# Finding 2: Average Order Value change
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

finding2 = {
    "title": "Average Order Value Trend",
    "claim": f"Average order value in analysis week was SAR {analysis_aov:.2f}, compared to SAR {previous_aov:.2f} in previous week, representing a {aov_pct_change:.1f}% change.",
    "finding_type": "aov_change",
    "metrics": {
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
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "aov_change_pct": {
            "value": round(aov_pct_change, 1),
            "unit": "%",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": analysis_baskets,
    "coverage_notes": [
        "AOV calculated as net revenue divided by unique transaction_id count",
        "Refunds excluded from calculation",
        f"Analysis week baskets: {analysis_baskets}",
        f"Previous week baskets: {previous_baskets}"
    ],
    "assumptions": [
        "Each transaction_id represents one basket/order",
        "line_total_sar is net of discounts and refunds",
        "is_refund flag correctly identifies refund transactions"
    ],
    "confidence": 0.95
}
findings.append(finding2)

# Finding 3: Category mix analysis
analysis_by_category = analysis_valid.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
analysis_by_category.columns = ['category', 'revenue', 'baskets']
analysis_by_category['pct_revenue'] = (analysis_by_category['revenue'] / analysis_by_category['revenue'].sum() * 100)

previous_by_category = previous_valid.groupby('category').agg({
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
previous_by_category.columns = ['category', 'revenue', 'baskets']
previous_by_category['pct_revenue'] = (previous_by_category['revenue'] / previous_by_category['revenue'].sum() * 100)

# Find category with largest revenue change
category_comparison = analysis_by_category.merge(
    previous_by_category,
    on='category',
    suffixes=('_analysis', '_previous')
)
category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
category_comparison['pct_change'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100)

top_category_change = category_comparison.loc[category_comparison['revenue_change'].abs().idxmax()]

finding3 = {
    "title": "Category Revenue Mix Shift",
    "claim": f"Category '{top_category_change['category']}' generated SAR {top_category_change['revenue_analysis']:.2f} in analysis week vs SAR {top_category_change['revenue_previous']:.2f} in previous week, a {top_category_change['pct_change']:.1f}% change.",
    "finding_type": "category_mix_change",
    "metrics": {
        "category_analysis_revenue": {
            "value": round(top_category_change['revenue_analysis'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "category_previous_revenue": {
            "value": round(top_category_change['revenue_previous'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        },
        "category_revenue_change_sar": {
            "value": round(top_category_change['revenue_change'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "category_revenue_change_pct": {
            "value": round(top_category_change['pct_change'], 1),
            "unit": "%",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "category_analysis_pct_mix": {
            "value": round(top_category_change['pct_revenue_analysis'], 1),
            "unit": "%",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-06T00:00:00+03:00",
            "period_end": "2026-07-13T00:00:00+03:00"
        },
        "category_previous_pct_mix": {
            "value": round(top_category_change['pct_revenue_previous'], 1),
            "unit": "%",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_valid),
    "coverage_notes": [
        f"Analysis week category revenue: {len(analysis_by_category)} categories",
        f"Previous week category revenue: {len(previous_by_category)} categories",
        "Refunds excluded from category analysis",
        f"Largest absolute change category: {top_category_change['category']}"
    ],
    "assumptions": [
        "category field in POS data is accurate",
        "line_total_sar represents net revenue per line item",
        "is_refund flag correctly identifies refund transactions"
    ],
    "confidence": 0.90
}
findings.append(finding3)

# Prepare output
output = {
    "status": "success",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)