import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Parse periods
analysis_start = pd.Timestamp("2026-07-06T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-07-13T00:00:00+03:00")
previous_start = pd.Timestamp("2026-06-29T00:00:00+03:00")
previous_end = pd.Timestamp("2026-07-06T00:00:00+03:00")

trailing_baselines = [
    (pd.Timestamp("2026-06-29T00:00:00+03:00"), pd.Timestamp("2026-07-06T00:00:00+03:00")),
    (pd.Timestamp("2026-06-22T00:00:00+03:00"), pd.Timestamp("2026-06-29T00:00:00+03:00")),
    (pd.Timestamp("2026-06-15T00:00:00+03:00"), pd.Timestamp("2026-06-22T00:00:00+03:00")),
    (pd.Timestamp("2026-06-08T00:00:00+03:00"), pd.Timestamp("2026-06-15T00:00:00+03:00")),
]

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()

# Filter for previous period
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter for trailing baseline (average of 4 weeks)
trailing_data = pos_df[(pos_df['timestamp'] >= trailing_baselines[0][0]) & (pos_df['timestamp'] < trailing_baselines[-1][1])].copy()

findings = []

# ===== FINDING 1: Revenue Change Analysis =====
# Calculate net revenue (including refunds) for analysis and previous periods
analysis_revenue = analysis_data['line_total_sar'].sum()
previous_revenue = previous_data['line_total_sar'].sum()

# Count valid transactions (unique transaction_id, excluding refunds for transaction count)
analysis_transactions = analysis_data[~analysis_data['is_refund']]['transaction_id'].nunique()
previous_transactions = previous_data[~previous_data['is_refund']]['transaction_id'].nunique()

if analysis_transactions > 0 and previous_transactions > 0:
    revenue_change = analysis_revenue - previous_revenue
    revenue_change_pct = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
    
    finding1 = {
        "title": "Net Revenue Change: Analysis Week vs Previous Week",
        "claim": f"Net revenue in analysis week (2026-07-06 to 2026-07-13) was SAR {analysis_revenue:.2f}, compared to SAR {previous_revenue:.2f} in previous week (2026-06-29 to 2026-07-06), representing a change of SAR {revenue_change:.2f} ({revenue_change_pct:.2f}%)",
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
                "value": round(revenue_change_pct, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Analysis period: 2026-07-06 to 2026-07-13",
            "Previous period: 2026-06-29 to 2026-07-06",
            "Net revenue includes refunds as per metric definition",
            f"Analysis week transactions: {analysis_transactions}",
            f"Previous week transactions: {previous_transactions}"
        ],
        "assumptions": [
            "Unique transaction_id identifies distinct baskets",
            "line_total_sar represents net realized revenue including refunds",
            "Refunds are included in net calculations per metric definition",
            "Timestamp field is reliable for period filtering"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# ===== FINDING 2: Average Order Value (AOV) Change =====
analysis_aov = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0
previous_aov = previous_revenue / previous_transactions if previous_transactions > 0 else 0

if analysis_transactions > 0 and previous_transactions > 0:
    aov_change = analysis_aov - previous_aov
    aov_change_pct = (aov_change / previous_aov * 100) if previous_aov != 0 else 0
    
    finding2 = {
        "title": "Average Order Value Change: Analysis Week vs Previous Week",
        "claim": f"Average order value in analysis week was SAR {analysis_aov:.2f}, compared to SAR {previous_aov:.2f} in previous week, representing a {aov_change_pct:.2f}% change",
        "finding_type": "aov_change",
        "metrics": {
            "analysis_week_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_transactions,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "previous_week_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_transactions,
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
                "value": round(aov_change_pct, 2),
                "unit": "%",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_transactions,
        "coverage_notes": [
            "Analysis period: 2026-07-06 to 2026-07-13",
            "Previous period: 2026-06-29 to 2026-07-06",
            "AOV calculated as net revenue / unique transaction count",
            "Refunds included in net revenue per metric definition"
        ],
        "assumptions": [
            "Unique transaction_id identifies distinct baskets",
            "line_total_sar represents net realized revenue",
            "Refunds are included in net calculations",
            "Non-refund transactions used for basket count"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# ===== FINDING 3: Product Mix Analysis - Top Category Performance =====
# Merge POS with menu to get category and launch/retire dates
analysis_with_menu = analysis_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
previous_with_menu = previous_data.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')

# Convert launch/retire dates to datetime
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')

# Calculate category revenue for analysis period
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

# Get top category
if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
    top_category = analysis_category_revenue.index[0]
    analysis_top_revenue = analysis_category_revenue.iloc[0]
    previous_top_revenue = previous_category_revenue.get(top_category, 0)
    
    if previous_top_revenue > 0:
        category_change = analysis_top_revenue - previous_top_revenue
        category_change_pct = (category_change / previous_top_revenue * 100)
        
        # Count transactions by category
        analysis_category_txns = analysis_data[analysis_data['category'] == top_category]['transaction_id'].nunique()
        previous_category_txns = previous_data[previous_data['category'] == top_category]['transaction_id'].nunique()
        
        finding3 = {
            "title": f"Top Category Revenue Change: {top_category}",
            "claim": f"Revenue from {top_category} category in analysis week was SAR {analysis_top_revenue:.2f}, compared to SAR {previous_top_revenue:.2f} in previous week, representing a {category_change_pct:.2f}% change",
            "finding_type": "category_mix_change",
            "metrics": {
                "analysis_week_category_revenue": {
                    "value": round(analysis_top_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-07-06T00:00:00+03:00",
                    "period_end": "2026-07-13T00:00:00+03:00"
                },
                "previous_week_category_revenue": {
                    "value": round(previous_top_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-29T00:00:00+03:00",
                    "period_end": "2026-07-06T00:00:00+03:00"
                },
                "category_revenue_change_sar": {
                    "value": round(category_change, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-07-06T00:00:00+03:00",
                    "period_end": "2026-07-13T00:00:00+03:00"
                },
                "category_revenue_change_pct": {
                    "value": round(category_change_pct, 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-07-06T00:00:00+03:00",
                    "period_end": "2026-07-13T00:00:00+03:00"
                },
                "analysis_week_category_transactions": {
                    "value": analysis_category_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-07-06T00:00:00+03:00",
                    "period_end": "2026-07-13T00:00:00+03:00"
                },
                "previous_week_category_transactions": {
                    "value": previous_category_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-29T00:00:00+03:00",
                    "period_end": "2026-07-06T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_data[analysis_data['category'] == top_category]),
            "coverage_notes": [
                f"Top category: {top_category}",
                "Analysis period: 2026-07-06 to 2026-07-13",
                "Previous period: 2026-06-29 to 2026-07-06",
                "Revenue includes refunds as per metric definition",
                f"Analysis week transactions in category: {analysis_category_txns}",
                f"Previous week transactions in category: {previous_category_txns}"
            ],
            "assumptions": [
                "Category field from POS is accurate",
                "line_total_sar represents net realized revenue",
                "Refunds included in net calculations",
                "Unique transaction_id identifies distinct baskets"
            ],
            "confidence": 0.92
        }
        findings.append(finding3)

# Write output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)