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
analysis_start = pd.Timestamp("2026-01-26T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-02-02T00:00:00+03:00")
previous_start = pd.Timestamp("2026-01-19T00:00:00+03:00")
previous_end = pd.Timestamp("2026-01-26T00:00:00+03:00")

trailing_periods = [
    (pd.Timestamp("2026-01-19T00:00:00+03:00"), pd.Timestamp("2026-01-26T00:00:00+03:00")),
    (pd.Timestamp("2026-01-12T00:00:00+03:00"), pd.Timestamp("2026-01-19T00:00:00+03:00")),
    (pd.Timestamp("2026-01-05T00:00:00+03:00"), pd.Timestamp("2026-01-12T00:00:00+03:00")),
    (pd.Timestamp("2025-12-29T00:00:00+03:00"), pd.Timestamp("2026-01-05T00:00:00+03:00")),
]

# Convert timestamp to timezone-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert('Asia/Riyadh')

# Filter data for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Trailing baseline: combine all trailing periods
trailing_data = pos_df[(pos_df['timestamp'] >= trailing_periods[0][0]) & (pos_df['timestamp'] < trailing_periods[-1][1])].copy()

findings = []

# ============================================================================
# FINDING 1: Revenue Change (Analysis vs Previous Week)
# ============================================================================

# Calculate net revenue (line_total_sar includes refunds as negative)
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()
revenue_change = analysis_revenue - previous_revenue
revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0

# Count valid transactions (unique transaction_id, excluding refunds for basket count)
analysis_baskets = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
previous_baskets = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()

# Calculate refund impact
analysis_refunds = analysis_data[analysis_data['is_refund']]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund']]['line_total_sar'].sum()

if analysis_revenue != 0 and previous_revenue != 0:
    finding_1 = {
        "title": "Net Revenue Change: Analysis Week vs Previous Week",
        "claim": f"Net revenue in the analysis period (2026-01-26 to 2026-02-02) was {analysis_revenue:.2f} SAR, compared to {previous_revenue:.2f} SAR in the previous week (2026-01-19 to 2026-01-26), representing a change of {revenue_change:.2f} SAR ({revenue_pct_change:.2f}%). Valid transaction count increased from {previous_baskets} to {analysis_baskets} baskets.",
        "finding_type": "revenue_change",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "revenue_change_absolute": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "analysis_baskets": {
                "value": analysis_baskets,
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_baskets": {
                "value": previous_baskets,
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "analysis_refunds_total": {
                "value": round(analysis_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_refunds_total": {
                "value": round(previous_refunds, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Analysis period: 2026-01-26 to 2026-02-02 (8 days)",
            "Previous period: 2026-01-19 to 2026-01-26 (7 days)",
            "Revenue includes refunds as negative values per metric definition",
            f"Analysis refunds: {analysis_refunds:.2f} SAR; Previous refunds: {previous_refunds:.2f} SAR"
        ],
        "assumptions": [
            "line_total_sar is accurate and includes refunds as negative values",
            "transaction_id uniqueness identifies valid baskets",
            "is_refund flag correctly identifies refund transactions",
            "Timestamp conversion to Asia/Riyadh timezone is correct"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Average Order Value (AOV) Change
# ============================================================================

# Calculate AOV for analysis and previous periods (excluding refunds from basket count)
analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0
aov_change = analysis_aov - previous_aov
aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0

if analysis_baskets > 0 and previous_baskets > 0:
    finding_2 = {
        "title": "Average Order Value (AOV) Change: Analysis Week vs Previous Week",
        "claim": f"Average order value in the analysis period was {analysis_aov:.2f} SAR per basket, compared to {previous_aov:.2f} SAR in the previous week, representing a change of {aov_change:.2f} SAR ({aov_pct_change:.2f}%).",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR/basket",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_baskets,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR/basket",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_baskets,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "aov_change_absolute": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "aov_pct_change": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_baskets,
        "coverage_notes": [
            "AOV calculated as net revenue / valid transaction count",
            "Refunds included in net revenue per metric definition",
            "Basket count excludes refund-only transactions"
        ],
        "assumptions": [
            "line_total_sar is accurate",
            "transaction_id uniqueness identifies valid baskets",
            "is_refund flag correctly identifies refund transactions"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Category Mix Change (Analysis vs Previous Week)
# ============================================================================

# Join POS with menu to get category information
analysis_with_menu = analysis_data.merge(menu_df[['sku', 'category']], on='sku', how='left', suffixes=('', '_menu'))
previous_with_menu = previous_data.merge(menu_df[['sku', 'category']], on='sku', how='left', suffixes=('', '_menu'))

# Use category from menu if available, otherwise from POS
analysis_with_menu['category_final'] = analysis_with_menu['category_menu'].fillna(analysis_with_menu['category'])
previous_with_menu['category_final'] = previous_with_menu['category_menu'].fillna(previous_with_menu['category'])

# Calculate category revenue mix
analysis_category_revenue = analysis_with_menu.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_with_menu.groupby('category_final')['line_total_sar'].sum().sort_values(ascending=False)

# Get top category
if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
    top_category_analysis = analysis_category_revenue.index[0]
    top_category_previous = previous_category_revenue.index[0]
    
    top_cat_revenue_analysis = analysis_category_revenue.iloc[0]
    top_cat_revenue_previous = previous_category_revenue.get(top_category_analysis, 0)
    
    top_cat_pct_analysis = (top_cat_revenue_analysis / analysis_revenue * 100) if analysis_revenue != 0 else 0
    top_cat_pct_previous = (top_cat_revenue_previous / previous_revenue * 100) if previous_revenue != 0 else 0
    
    top_cat_pct_change = top_cat_pct_analysis - top_cat_pct_previous
    
    finding_3 = {
        "title": "Top Category Revenue Mix: Analysis Week vs Previous Week",
        "claim": f"The top revenue category in the analysis period was '{top_category_analysis}' with {top_cat_revenue_analysis:.2f} SAR ({top_cat_pct_analysis:.2f}% of total revenue). In the previous week, '{top_category_previous}' generated {top_cat_revenue_previous:.2f} SAR ({top_cat_pct_previous:.2f}% of total revenue). The top category's share changed by {top_cat_pct_change:.2f} percentage points.",
        "finding_type": "category_mix_change",
        "metrics": {
            "analysis_top_category": {
                "value": top_category_analysis,
                "unit": "category",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "analysis_top_category_revenue": {
                "value": round(top_cat_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "analysis_top_category_pct": {
                "value": round(top_cat_pct_analysis, 2),
                "unit": "%",
                "numerator": round(top_cat_revenue_analysis, 2),
                "denominator": round(analysis_revenue, 2),
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "previous_top_category": {
                "value": top_category_previous,
                "unit": "category",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "previous_top_category_revenue": {
                "value": round(top_cat_revenue_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "previous_top_category_pct": {
                "value": round(top_cat_pct_previous, 2),
                "unit": "%",
                "numerator": round(top_cat_revenue_previous, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-01-19T00:00:00+03:00",
                "period_end": "2026-01-26T00:00:00+03:00"
            },
            "top_category_pct_change": {
                "value": round(top_cat_pct_change, 2),
                "unit": "percentage points",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_with_menu),
        "coverage_notes": [
            "Category information joined from menu SKU reference",
            "Revenue includes refunds as negative values",
            "Top category identified by total revenue contribution"
        ],
        "assumptions": [
            "Menu SKU reference is authoritative for category assignment",
            "line_total_sar is accurate",
            "Category field in menu is complete and accurate"
        ],
        "confidence": 0.90
    }
    findings.append(finding_3)

# ============================================================================
# Output Result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)