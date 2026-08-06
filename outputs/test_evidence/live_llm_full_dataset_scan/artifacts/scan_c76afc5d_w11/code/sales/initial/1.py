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

def is_product_active(launch_date, retire_date, period_start, period_end):
    """Check if product is active during period."""
    period_start_date = parse_iso_datetime(period_start).date()
    period_end_date = parse_iso_datetime(period_end).date()
    
    if pd.notna(launch_date):
        launch = pd.to_datetime(launch_date).date()
        if launch >= period_end_date:
            return False
    
    if pd.notna(retire_date):
        retire = pd.to_datetime(retire_date).date()
        if retire <= period_start_date:
            return False
    
    return True

def main():
    inputs, output_path = load_inputs()
    
    # Load artifacts
    pos_df = pd.read_parquet(inputs['pos'])
    menu_df = pd.read_parquet(inputs['menu'])
    
    # Define periods
    analysis_start = "2026-03-23T00:00:00+03:00"
    analysis_end = "2026-03-30T00:00:00+03:00"
    previous_start = "2026-03-16T00:00:00+03:00"
    previous_end = "2026-03-23T00:00:00+03:00"
    
    # Filter data by periods
    analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
    previous_data = filter_by_period(pos_df, previous_start, previous_end)
    
    # Baseline: average of 4 trailing weeks
    baseline_data = pd.DataFrame()
    baseline_periods = [
        ("2026-03-16T00:00:00+03:00", "2026-03-23T00:00:00+03:00"),
        ("2026-03-09T00:00:00+03:00", "2026-03-16T00:00:00+03:00"),
        ("2026-03-02T00:00:00+03:00", "2026-03-09T00:00:00+03:00"),
        ("2026-02-23T00:00:00+03:00", "2026-03-02T00:00:00+03:00"),
    ]
    for start, end in baseline_periods:
        baseline_data = pd.concat([baseline_data, filter_by_period(pos_df, start, end)])
    
    findings = []
    
    # FINDING 1: Revenue and Transaction Count Change
    analysis_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
    previous_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()
    
    analysis_baskets = analysis_data['transaction_id'].nunique()
    previous_baskets = previous_data['transaction_id'].nunique()
    
    analysis_aov = analysis_revenue / analysis_baskets if analysis_baskets > 0 else 0
    previous_aov = previous_revenue / previous_baskets if previous_baskets > 0 else 0
    
    revenue_change = analysis_revenue - previous_revenue
    revenue_pct = (revenue_change / previous_revenue * 100) if previous_revenue > 0 else 0
    basket_change = analysis_baskets - previous_baskets
    basket_pct = (basket_change / previous_baskets * 100) if previous_baskets > 0 else 0
    aov_change = analysis_aov - previous_aov
    aov_pct = (aov_change / previous_aov * 100) if previous_aov > 0 else 0
    
    if abs(revenue_pct) > 2 or abs(basket_pct) > 2:
        findings.append({
            "title": "Weekly Revenue and Transaction Volume Change",
            "claim": f"Week of {analysis_start[:10]} generated SAR {analysis_revenue:.2f} in net revenue across {analysis_baskets} transactions (AOV: SAR {analysis_aov:.2f}), compared to SAR {previous_revenue:.2f} across {previous_baskets} transactions (AOV: SAR {previous_aov:.2f}) in the prior week.",
            "finding_type": "revenue_and_volume",
            "metrics": {
                "net_revenue_sar": {
                    "value": round(analysis_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "valid_transaction_count": {
                    "value": analysis_baskets,
                    "unit": "baskets",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "average_order_value_sar": {
                    "value": round(analysis_aov, 2),
                    "unit": "SAR",
                    "numerator": round(analysis_revenue, 2),
                    "denominator": analysis_baskets,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "revenue_change_sar": {
                    "value": round(revenue_change, 2),
                    "unit": "SAR",
                    "numerator": round(analysis_revenue, 2),
                    "denominator": round(previous_revenue, 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "revenue_change_pct": {
                    "value": round(revenue_pct, 2),
                    "unit": "%",
                    "numerator": round(revenue_change, 2),
                    "denominator": round(previous_revenue, 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "transaction_count_change": {
                    "value": basket_change,
                    "unit": "baskets",
                    "numerator": analysis_baskets,
                    "denominator": previous_baskets,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "transaction_count_change_pct": {
                    "value": round(basket_pct, 2),
                    "unit": "%",
                    "numerator": basket_change,
                    "denominator": previous_baskets,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "aov_change_sar": {
                    "value": round(aov_change, 2),
                    "unit": "SAR",
                    "numerator": round(analysis_aov, 2),
                    "denominator": round(previous_aov, 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "aov_change_pct": {
                    "value": round(aov_pct, 2),
                    "unit": "%",
                    "numerator": round(aov_change, 2),
                    "denominator": round(previous_aov, 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Analysis period: 2026-03-23 to 2026-03-30",
                "Previous period: 2026-03-16 to 2026-03-23",
                "Refunds excluded from revenue calculations",
                "Transaction count based on unique transaction_id"
            ],
            "assumptions": [
                "business_date field used for period filtering",
                "is_refund flag correctly identifies refund transactions",
                "line_total_sar represents net realized revenue"
            ],
            "confidence": 0.95
        })
    
    # FINDING 2: Category Mix Analysis
    analysis_by_category = analysis_data[analysis_data['is_refund'] == False].groupby('category').agg({
        'line_total_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    analysis_by_category.columns = ['category', 'revenue', 'baskets']
    analysis_by_category['aov'] = analysis_by_category['revenue'] / analysis_by_category['baskets']
    analysis_by_category['revenue_pct'] = (analysis_by_category['revenue'] / analysis_by_category['revenue'].sum() * 100)
    
    previous_by_category = previous_data[previous_data['is_refund'] == False].groupby('category').agg({
        'line_total_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    previous_by_category.columns = ['category', 'revenue', 'baskets']
    previous_by_category['aov'] = previous_by_category['revenue'] / previous_by_category['baskets']
    previous_by_category['revenue_pct'] = (previous_by_category['revenue'] / previous_by_category['revenue'].sum() * 100)
    
    # Merge and calculate changes
    category_comparison = analysis_by_category.merge(
        previous_by_category,
        on='category',
        suffixes=('_analysis', '_previous')
    )
    category_comparison['revenue_change'] = category_comparison['revenue_analysis'] - category_comparison['revenue_previous']
    category_comparison['revenue_change_pct'] = (category_comparison['revenue_change'] / category_comparison['revenue_previous'] * 100)
    category_comparison['mix_change_pct'] = category_comparison['revenue_pct_analysis'] - category_comparison['revenue_pct_previous']
    
    # Find most significant category change
    category_comparison['abs_change'] = category_comparison['revenue_change_pct'].abs()
    top_category = category_comparison.loc[category_comparison['abs_change'].idxmax()]
    
    if abs(top_category['revenue_change_pct']) > 5:
        findings.append({
            "title": "Category Mix Shift",
            "claim": f"Category '{top_category['category']}' generated SAR {top_category['revenue_analysis']:.2f} ({top_category['revenue_pct_analysis']:.1f}% of category revenue) in the analysis week, versus SAR {top_category['revenue_previous']:.2f} ({top_category['revenue_pct_previous']:.1f}% of category revenue) in the prior week, representing a {top_category['revenue_change_pct']:.1f}% change in category revenue and {top_category['mix_change_pct']:.1f} percentage point shift in overall mix.",
            "finding_type": "category_mix",
            "metrics": {
                "category_name": {
                    "value": str(top_category['category']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "category_revenue_sar": {
                    "value": round(top_category['revenue_analysis'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "category_revenue_pct_of_total": {
                    "value": round(top_category['revenue_pct_analysis'], 2),
                    "unit": "%",
                    "numerator": round(top_category['revenue_analysis'], 2),
                    "denominator": round(analysis_by_category['revenue'].sum(), 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "category_revenue_change_sar": {
                    "value": round(top_category['revenue_change'], 2),
                    "unit": "SAR",
                    "numerator": round(top_category['revenue_analysis'], 2),
                    "denominator": round(top_category['revenue_previous'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "category_revenue_change_pct": {
                    "value": round(top_category['revenue_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_category['revenue_change'], 2),
                    "denominator": round(top_category['revenue_previous'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "category_mix_change_pct": {
                    "value": round(top_category['mix_change_pct'], 2),
                    "unit": "percentage points",
                    "numerator": round(top_category['revenue_pct_analysis'], 2),
                    "denominator": round(top_category['revenue_pct_previous'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Analysis period: 2026-03-23 to 2026-03-30",
                "Previous period: 2026-03-16 to 2026-03-23",
                "Refunds excluded from revenue calculations",
                "Category field from cleaned POS data"
            ],
            "assumptions": [
                "Category classification is accurate and consistent",
                "line_total_sar represents net realized revenue per line item"
            ],
            "confidence": 0.90
        })
    
    # FINDING 3: Product Performance - Top SKU Change
    analysis_by_sku = analysis_data[analysis_data['is_refund'] == False].groupby('sku').agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': 'nunique',
        'item_name_en': 'first'
    }).reset_index()
    analysis_by_sku.columns = ['sku', 'revenue', 'quantity', 'baskets', 'item_name']
    analysis_by_sku['aov'] = analysis_by_sku['revenue'] / analysis_by_sku['baskets']
    
    previous_by_sku = previous_data[previous_data['is_refund'] == False].groupby('sku').agg({
        'line_total_sar': 'sum',
        'quantity': 'sum',
        'transaction_id': 'nunique',
        'item_name_en': 'first'
    }).reset_index()
    previous_by_sku.columns = ['sku', 'revenue', 'quantity', 'baskets', 'item_name']
    previous_by_sku['aov'] = previous_by_sku['revenue'] / previous_by_sku['baskets']
    
    # Merge with menu to check launch dates
    analysis_by_sku = analysis_by_sku.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
    previous_by_sku = previous_by_sku.merge(menu_df[['sku', 'launch_date', 'retire_date']], on='sku', how='left')
    
    # Filter for products active in both periods
    analysis_by_sku['active'] = analysis_by_sku.apply(
        lambda row: is_product_active(row['launch_date'], row['retire_date'], analysis_start, analysis_end),
        axis=1
    )
    previous_by_sku['active'] = previous_by_sku.apply(
        lambda row: is_product_active(row['launch_date'], row['retire_date'], previous_start, previous_end),
        axis=1
    )
    
    analysis_by_sku = analysis_by_sku[analysis_by_sku['active']]
    previous_by_sku = previous_by_sku[previous_by_sku['active']]
    
    # Merge and calculate changes
    sku_comparison = analysis_by_sku.merge(
        previous_by_sku,
        on='sku',
        suffixes=('_analysis', '_previous'),
        how='inner'
    )
    
    if len(sku_comparison) > 0:
        sku_comparison['revenue_change'] = sku_comparison['revenue_analysis'] - sku_comparison['revenue_previous']
        sku_comparison['revenue_change_pct'] = (sku_comparison['revenue_change'] / sku_comparison['revenue_previous'] * 100)
        sku_comparison['abs_change'] = sku_comparison['revenue_change_pct'].abs()
        
        # Find top performer by absolute change
        top_sku = sku_comparison.loc[sku_comparison['abs_change'].idxmax()]
        
        if abs(top_sku['revenue_change_pct']) > 10:
            findings.append({
                "title": "Top SKU Performance Change",
                "claim": f"SKU {top_sku['sku']} ({top_sku['item_name_analysis']}) generated SAR {top_sku['revenue_analysis']:.2f} across {int(top_sku['quantity_analysis'])} units in {int(top_sku['baskets_analysis'])} baskets during the analysis week, versus SAR {top_sku['revenue_previous']:.2f} across {int(top_sku['quantity_previous'])} units in {int(top_sku['baskets_previous'])} baskets in the prior week, representing a {top_sku['revenue_change_pct']:.1f}% change in SKU revenue.",
                "finding_type": "product_performance",
                "metrics": {
                    "sku": {
                        "value": str(top_sku['sku']),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "item_name": {
                        "value": str(top_sku['item_name_analysis']),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "sku_revenue_sar": {
                        "value": round(top_sku['revenue_analysis'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "sku_quantity_sold": {
                        "value": int(top_sku['quantity_analysis']),
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "sku_baskets": {
                        "value": int(top_sku['baskets_analysis']),
                        "unit": "baskets",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "sku_revenue_change_sar": {
                        "value": round(top_sku['revenue_change'], 2),
                        "unit": "SAR",
                        "numerator": round(top_sku['revenue_analysis'], 2),
                        "denominator": round(top_sku['revenue_previous'], 2),
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "sku_revenue_change_pct": {
                        "value": round(top_sku['revenue_change_pct'], 2),
                        "unit": "%",
                        "numerator": round(top_sku['revenue_change'], 2),
                        "denominator": round(top_sku['revenue_previous'], 2),
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": len(sku_comparison),
                "coverage_notes": [
                    "Analysis period: 2026-03-23 to 2026-03-30",
                    "Previous period: 2026-03-16 to 2026-03-23",
                    "Refunds excluded from revenue calculations",
                    "Only products active in both periods included",
                    f"Comparison includes {len(sku_comparison)} SKUs active in both periods"
                ],
                "assumptions": [
                    "SKU and item_name fields correctly identify products",
                    "launch_date and retire_date fields accurately reflect product availability",
                    "line_total_sar represents net realized revenue per line item"
                ],
                "confidence": 0.88
            })
    
    # Prepare output
    result = {
        "status": "success" if len(findings) > 0 else "insufficient_data",
        "findings": findings
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    main()