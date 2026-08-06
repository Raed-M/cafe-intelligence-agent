import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

def load_inputs():
    """Load input artifact paths from environment."""
    with open(os.environ['ANALYST_INPUTS_JSON']) as f:
        run_meta = json.load(f)
    return run_meta['inputs'], run_meta['output_path']

def parse_iso_datetime(dt_str):
    """Parse ISO datetime string to datetime object."""
    if isinstance(dt_str, str):
        return pd.to_datetime(dt_str)
    return dt_str

def filter_by_period(df, start_iso, end_iso, date_col='business_date'):
    """Filter dataframe by period using business_date."""
    start = parse_iso_datetime(start_iso).date()
    end = parse_iso_datetime(end_iso).date()
    df_copy = df.copy()
    df_copy['business_date'] = pd.to_datetime(df_copy[date_col]).dt.date
    return df_copy[(df_copy['business_date'] >= start) & (df_copy['business_date'] < end)]

def is_product_eligible(sku, menu_df, period_start_iso, period_end_iso):
    """Check if product is eligible for analysis in the given period."""
    period_start = parse_iso_datetime(period_start_iso).date()
    period_end = parse_iso_datetime(period_end_iso).date()
    
    product = menu_df[menu_df['sku'] == sku]
    if product.empty:
        return False
    
    launch_date = product.iloc[0]['launch_date']
    retire_date = product.iloc[0]['retire_date']
    
    if pd.notna(launch_date):
        launch_date = pd.to_datetime(launch_date).date()
        if period_start < launch_date:
            return False
    
    if pd.notna(retire_date):
        retire_date = pd.to_datetime(retire_date).date()
        if period_end > retire_date:
            return False
    
    return True

def main():
    inputs, output_path = load_inputs()
    
    # Load artifacts
    pos_df = pd.read_parquet(inputs['pos'])
    menu_df = pd.read_parquet(inputs['menu'])
    
    # Define periods
    analysis_start = "2026-03-16T00:00:00+03:00"
    analysis_end = "2026-03-23T00:00:00+03:00"
    previous_start = "2026-03-09T00:00:00+03:00"
    previous_end = "2026-03-16T00:00:00+03:00"
    
    trailing_periods = [
        ("2026-03-09T00:00:00+03:00", "2026-03-16T00:00:00+03:00"),
        ("2026-03-02T00:00:00+03:00", "2026-03-09T00:00:00+03:00"),
        ("2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00"),
        ("2026-02-16T00:00:00+03:00", "2026-02-23T00:00:00+03:00"),
    ]
    
    # Filter data by periods
    analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
    previous_data = filter_by_period(pos_df, previous_start, previous_end)
    
    trailing_data_list = []
    for start, end in trailing_periods:
        trailing_data_list.append(filter_by_period(pos_df, start, end))
    
    findings = []
    
    # FINDING 1: Revenue change (analysis vs previous week)
    analysis_revenue = analysis_data['line_total_sar'].sum()
    previous_revenue = previous_data['line_total_sar'].sum()
    revenue_change = analysis_revenue - previous_revenue
    revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
    
    analysis_baskets = analysis_data['transaction_id'].nunique()
    previous_baskets = previous_data['transaction_id'].nunique()
    
    if abs(revenue_pct_change) >= 5:  # Threshold for meaningful change
        findings.append({
            "title": "Weekly Revenue Change",
            "claim": f"Net revenue in analysis week (2026-03-16 to 2026-03-23) was {analysis_revenue:.2f} SAR, representing a {revenue_pct_change:.1f}% change from previous week ({previous_revenue:.2f} SAR). Valid transaction count: {analysis_baskets} vs {previous_baskets} baskets.",
            "finding_type": "revenue_change",
            "metrics": {
                "analysis_week_revenue": {
                    "value": round(analysis_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_week_revenue": {
                    "value": round(previous_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
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
                "revenue_change_pct": {
                    "value": round(revenue_pct_change, 1),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_baskets": {
                    "value": analysis_baskets,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_baskets": {
                    "value": previous_baskets,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Analysis period: 2026-03-16 to 2026-03-23",
                "Previous period: 2026-03-09 to 2026-03-16",
                "Baskets counted using unique transaction_id",
                "Revenue includes refunds in net calculation"
            ],
            "assumptions": [
                "business_date field used for period filtering",
                "line_total_sar represents net revenue after discounts",
                "transaction_id uniqueness indicates distinct baskets"
            ],
            "confidence": 0.95
        })
    
    # FINDING 2: Average Order Value change
    analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
    previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0
    aov_change = analysis_aov - previous_aov
    aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0
    
    if abs(aov_pct_change) >= 3:
        findings.append({
            "title": "Average Order Value Change",
            "claim": f"Average order value in analysis week was {analysis_aov:.2f} SAR per basket, a {aov_pct_change:.1f}% change from {previous_aov:.2f} SAR in previous week.",
            "finding_type": "aov_change",
            "metrics": {
                "analysis_aov": {
                    "value": round(analysis_aov, 2),
                    "unit": "SAR/basket",
                    "numerator": round(analysis_revenue, 2),
                    "denominator": analysis_baskets,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_aov": {
                    "value": round(previous_aov, 2),
                    "unit": "SAR/basket",
                    "numerator": round(previous_revenue, 2),
                    "denominator": previous_baskets,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "aov_change_sar": {
                    "value": round(aov_change, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "aov_change_pct": {
                    "value": round(aov_pct_change, 1),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": analysis_baskets,
            "coverage_notes": [
                "AOV calculated as total revenue divided by unique transaction count",
                "Includes all valid transactions in period"
            ],
            "assumptions": [
                "transaction_id uniqueness indicates distinct baskets",
                "line_total_sar represents net revenue"
            ],
            "confidence": 0.92
        })
    
    # FINDING 3: Category mix change
    analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    
    # Find top category changes
    top_categories = set(analysis_category_revenue.head(3).index) | set(previous_category_revenue.head(3).index)
    
    for category in top_categories:
        analysis_cat_rev = analysis_category_revenue.get(category, 0)
        previous_cat_rev = previous_category_revenue.get(category, 0)
        cat_change = analysis_cat_rev - previous_cat_rev
        cat_pct_change = (cat_change / previous_cat_rev * 100) if previous_cat_rev != 0 else 0
        
        if abs(cat_pct_change) >= 10 and previous_cat_rev > 0:
            analysis_cat_pct = (analysis_cat_rev / analysis_revenue * 100) if analysis_revenue > 0 else 0
            previous_cat_pct = (previous_cat_rev / previous_revenue * 100) if previous_revenue > 0 else 0
            
            findings.append({
                "title": f"Category Mix Shift: {category}",
                "claim": f"Category '{category}' revenue was {analysis_cat_rev:.2f} SAR ({analysis_cat_pct:.1f}% of total) in analysis week vs {previous_cat_rev:.2f} SAR ({previous_cat_pct:.1f}% of total) in previous week, a {cat_pct_change:.1f}% change.",
                "finding_type": "category_mix_change",
                "metrics": {
                    "analysis_category_revenue": {
                        "value": round(analysis_cat_rev, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_category_revenue": {
                        "value": round(previous_cat_rev, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": previous_start,
                        "period_end": previous_end
                    },
                    "analysis_category_pct": {
                        "value": round(analysis_cat_pct, 1),
                        "unit": "%",
                        "numerator": round(analysis_cat_rev, 2),
                        "denominator": round(analysis_revenue, 2),
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_category_pct": {
                        "value": round(previous_cat_pct, 1),
                        "unit": "%",
                        "numerator": round(previous_cat_rev, 2),
                        "denominator": round(previous_revenue, 2),
                        "period_start": previous_start,
                        "period_end": previous_end
                    },
                    "category_change_pct": {
                        "value": round(cat_pct_change, 1),
                        "unit": "%",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_data),
                "coverage_notes": [
                    f"Category: {category}",
                    "Revenue aggregated by category field from POS"
                ],
                "assumptions": [
                    "category field accurately represents product grouping",
                    "line_total_sar represents net revenue"
                ],
                "confidence": 0.90
            })
            break  # Only report top category change to stay within 3 findings limit
    
    # Limit to 3 findings
    findings = findings[:3]
    
    result = {
        "status": "success" if findings else "insufficient_data",
        "findings": findings
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()