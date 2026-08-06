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
analysis_start = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-16T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-03-02T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-03-09T00:00:00+03:00")

trailing_baselines = [
    ("2026-03-02T00:00:00+03:00", "2026-03-09T00:00:00+03:00"),
    ("2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00"),
    ("2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00"),
    ("2026-02-09T00:00:00+03:00", "2026-02-16T00:00:00+03:00"),
]

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter by period
def filter_by_period(df, start, end):
    return df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

analysis_df = filter_by_period(pos_df, analysis_start, analysis_end)
previous_df = filter_by_period(pos_df, previous_start, previous_end)

# Compute metrics for analysis period
def compute_metrics(df, period_name):
    # Valid transactions (unique transaction_id, excluding refunds for net revenue)
    valid_txns = df[~df['is_refund']].copy()
    basket_count = valid_txns['transaction_id'].nunique()
    
    # Net revenue (line_total_sar includes refunds as negative)
    net_revenue = df['line_total_sar'].sum()
    
    # Average order value
    aov = net_revenue / basket_count if basket_count > 0 else 0
    
    # Category mix
    category_revenue = df.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    total_revenue = df['line_total_sar'].sum()
    category_pct = (category_revenue / total_revenue * 100) if total_revenue != 0 else 0
    
    # Channel mix
    channel_revenue = df.groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    channel_pct = (channel_revenue / total_revenue * 100) if total_revenue != 0 else 0
    
    # Product performance (top 5 by revenue)
    product_revenue = df.groupby(['sku', 'item_name_en']).agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': 'nunique'
    }).sort_values('line_total_sar', ascending=False)
    
    return {
        'basket_count': basket_count,
        'net_revenue': net_revenue,
        'aov': aov,
        'category_revenue': category_revenue,
        'category_pct': category_pct,
        'channel_revenue': channel_revenue,
        'channel_pct': channel_pct,
        'product_revenue': product_revenue,
        'total_revenue': total_revenue
    }

analysis_metrics = compute_metrics(analysis_df, 'analysis')
previous_metrics = compute_metrics(previous_df, 'previous')

# Compute trailing baseline average
trailing_metrics_list = []
for start_str, end_str in trailing_baselines:
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    trailing_df = filter_by_period(pos_df, start, end)
    trailing_metrics_list.append(compute_metrics(trailing_df, 'trailing'))

# Calculate average baseline
avg_baseline_revenue = np.mean([m['net_revenue'] for m in trailing_metrics_list])
avg_baseline_aov = np.mean([m['aov'] for m in trailing_metrics_list])
avg_baseline_baskets = np.mean([m['basket_count'] for m in trailing_metrics_list])

# Findings
findings = []

# Finding 1: Revenue change analysis period vs previous week
revenue_change = analysis_metrics['net_revenue'] - previous_metrics['net_revenue']
revenue_pct_change = (revenue_change / previous_metrics['net_revenue'] * 100) if previous_metrics['net_revenue'] != 0 else 0

if abs(revenue_pct_change) > 2:  # Threshold for meaningful change
    findings.append({
        "title": "Net Revenue Change: Analysis Week vs Previous Week",
        "claim": f"Net revenue in analysis week (Mar 9-16) was SAR {analysis_metrics['net_revenue']:.2f}, compared to SAR {previous_metrics['net_revenue']:.2f} in previous week (Mar 2-9), representing a change of SAR {revenue_change:.2f} ({revenue_pct_change:.1f}%).",
        "finding_type": "revenue_change",
        "metrics": {
            "net_revenue_analysis": {
                "value": round(analysis_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "net_revenue_previous": {
                "value": round(previous_metrics['net_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "revenue_change_sar": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "revenue_change_pct": {
                "value": round(revenue_pct_change, 1),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_metrics['net_revenue'], 2),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_df),
        "coverage_notes": [
            "Analysis period: 2026-03-09 to 2026-03-16",
            "Previous period: 2026-03-02 to 2026-03-09",
            "Refunds included in net revenue calculation",
            f"Analysis period transaction count: {analysis_metrics['basket_count']}",
            f"Previous period transaction count: {previous_metrics['basket_count']}"
        ],
        "assumptions": [
            "line_total_sar represents net realized revenue including refunds",
            "transaction_id uniqueness defines basket count",
            "Timestamp filtering uses UTC+3 timezone as provided"
        ],
        "confidence": 0.95
    })

# Finding 2: Average Order Value change
aov_change = analysis_metrics['aov'] - previous_metrics['aov']
aov_pct_change = (aov_change / previous_metrics['aov'] * 100) if previous_metrics['aov'] != 0 else 0

if abs(aov_pct_change) > 2:
    findings.append({
        "title": "Average Order Value Change: Analysis Week vs Previous Week",
        "claim": f"Average order value in analysis week was SAR {analysis_metrics['aov']:.2f}, compared to SAR {previous_metrics['aov']:.2f} in previous week, a change of SAR {aov_change:.2f} ({aov_pct_change:.1f}%).",
        "finding_type": "aov_change",
        "metrics": {
            "aov_analysis": {
                "value": round(analysis_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(analysis_metrics['net_revenue'], 2),
                "denominator": analysis_metrics['basket_count'],
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "aov_previous": {
                "value": round(previous_metrics['aov'], 2),
                "unit": "SAR",
                "numerator": round(previous_metrics['net_revenue'], 2),
                "denominator": previous_metrics['basket_count'],
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "aov_change_sar": {
                "value": round(aov_change, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "aov_change_pct": {
                "value": round(aov_pct_change, 1),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_metrics['aov'], 2),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": analysis_metrics['basket_count'],
        "coverage_notes": [
            "AOV calculated as net revenue / unique transaction count",
            f"Analysis period baskets: {analysis_metrics['basket_count']}",
            f"Previous period baskets: {previous_metrics['basket_count']}",
            "Refunds included in net revenue"
        ],
        "assumptions": [
            "Each unique transaction_id represents one basket/order",
            "line_total_sar includes refunds as negative values",
            "AOV = total net revenue / basket count"
        ],
        "confidence": 0.95
    })

# Finding 3: Category mix shift - identify largest category and its change
if len(analysis_metrics['category_pct']) > 0 and len(previous_metrics['category_pct']) > 0:
    top_category_analysis = analysis_metrics['category_pct'].index[0]
    top_category_pct_analysis = analysis_metrics['category_pct'].iloc[0]
    top_category_revenue_analysis = analysis_metrics['category_revenue'].iloc[0]
    
    if top_category_analysis in previous_metrics['category_pct'].index:
        top_category_pct_previous = previous_metrics['category_pct'][top_category_analysis]
        top_category_revenue_previous = previous_metrics['category_revenue'][top_category_analysis]
        
        pct_point_change = top_category_pct_analysis - top_category_pct_previous
        
        if abs(pct_point_change) > 1.5:
            findings.append({
                "title": f"Category Mix Shift: {top_category_analysis}",
                "claim": f"The {top_category_analysis} category represented {top_category_pct_analysis:.1f}% of revenue in analysis week (SAR {top_category_revenue_analysis:.2f}), compared to {top_category_pct_previous:.1f}% in previous week (SAR {top_category_revenue_previous:.2f}), a shift of {pct_point_change:.1f} percentage points.",
                "finding_type": "category_mix_shift",
                "metrics": {
                    "category_pct_analysis": {
                        "value": round(top_category_pct_analysis, 1),
                        "unit": "%",
                        "numerator": round(top_category_revenue_analysis, 2),
                        "denominator": round(analysis_metrics['total_revenue'], 2),
                        "period_start": "2026-03-09T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "category_pct_previous": {
                        "value": round(top_category_pct_previous, 1),
                        "unit": "%",
                        "numerator": round(top_category_revenue_previous, 2),
                        "denominator": round(previous_metrics['total_revenue'], 2),
                        "period_start": "2026-03-02T00:00:00+03:00",
                        "period_end": "2026-03-09T00:00:00+03:00"
                    },
                    "category_revenue_analysis": {
                        "value": round(top_category_revenue_analysis, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-09T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    },
                    "category_revenue_previous": {
                        "value": round(top_category_revenue_previous, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-02T00:00:00+03:00",
                        "period_end": "2026-03-09T00:00:00+03:00"
                    },
                    "pct_point_change": {
                        "value": round(pct_point_change, 1),
                        "unit": "percentage points",
                        "numerator": round(pct_point_change, 1),
                        "denominator": None,
                        "period_start": "2026-03-09T00:00:00+03:00",
                        "period_end": "2026-03-16T00:00:00+03:00"
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_df),
                "coverage_notes": [
                    f"Analysis period total revenue: SAR {analysis_metrics['total_revenue']:.2f}",
                    f"Previous period total revenue: SAR {previous_metrics['total_revenue']:.2f}",
                    "Category percentages calculated from net revenue (refunds included)",
                    f"Top category in analysis period: {top_category_analysis}"
                ],
                "assumptions": [
                    "Category mix percentages use total net revenue as denominator",
                    "Refunds are included in both numerator and denominator",
                    "Category assignment from POS data is authoritative"
                ],
                "confidence": 0.92
            })

# Write output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)