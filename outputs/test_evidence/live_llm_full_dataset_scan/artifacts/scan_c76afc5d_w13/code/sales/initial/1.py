import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

def load_inputs():
    """Load input paths from environment variable."""
    with open(os.environ['ANALYST_INPUTS_JSON']) as f:
        run_meta = json.load(f)
    return run_meta['inputs'], run_meta['output_path']

def parse_iso_datetime(dt_str):
    """Parse ISO datetime string to datetime object."""
    if isinstance(dt_str, str):
        return pd.to_datetime(dt_str)
    return dt_str

def filter_by_period(df, start_str, end_str, date_col='business_date'):
    """Filter dataframe by date period."""
    start = parse_iso_datetime(start_str).date()
    end = parse_iso_datetime(end_str).date()
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.date
    return df_copy[(df_copy[date_col] >= start) & (df_copy[date_col] < end)]

def is_product_active(launch_date, retire_date, period_start, period_end):
    """Check if product is active during the period."""
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
    
    # Load data
    pos_df = pd.read_parquet(inputs['pos'])
    menu_df = pd.read_parquet(inputs['menu'])
    
    # Define periods
    analysis_start = "2026-04-06T00:00:00+03:00"
    analysis_end = "2026-04-13T00:00:00+03:00"
    previous_start = "2026-03-30T00:00:00+03:00"
    previous_end = "2026-04-06T00:00:00+03:00"
    
    # Filter data by periods
    analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
    previous_data = filter_by_period(pos_df, previous_start, previous_end)
    
    # Ensure business_date is datetime for filtering
    pos_df['business_date'] = pd.to_datetime(pos_df['business_date']).dt.date
    analysis_data['business_date'] = pd.to_datetime(analysis_data['business_date']).dt.date
    previous_data['business_date'] = pd.to_datetime(previous_data['business_date']).dt.date
    
    findings = []
    
    # Finding 1: Revenue and Transaction Count Comparison
    analysis_revenue = analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()
    previous_revenue = previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()
    
    analysis_transactions = analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique()
    previous_transactions = previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique()
    
    if analysis_revenue > 0 and previous_revenue > 0:
        revenue_change = analysis_revenue - previous_revenue
        revenue_pct_change = (revenue_change / previous_revenue) * 100
        
        transaction_change = analysis_transactions - previous_transactions
        transaction_pct_change = (transaction_change / previous_transactions) * 100 if previous_transactions > 0 else 0
        
        aov_analysis = analysis_revenue / analysis_transactions if analysis_transactions > 0 else 0
        aov_previous = previous_revenue / previous_transactions if previous_transactions > 0 else 0
        aov_change = aov_analysis - aov_previous
        aov_pct_change = (aov_change / aov_previous) * 100 if aov_previous > 0 else 0
        
        findings.append({
            "title": "Revenue and Transaction Performance Week-over-Week",
            "claim": f"Analysis week (Apr 6-13) generated SAR {analysis_revenue:,.2f} in net revenue across {analysis_transactions} transactions, compared to SAR {previous_revenue:,.2f} across {previous_transactions} transactions in the previous week. Revenue changed by SAR {revenue_change:,.2f} ({revenue_pct_change:+.1f}%), while transaction count changed by {transaction_change:+d} ({transaction_pct_change:+.1f}%). Average order value increased from SAR {aov_previous:.2f} to SAR {aov_analysis:.2f}, a change of SAR {aov_change:+.2f} ({aov_pct_change:+.1f}%).",
            "finding_type": "revenue_and_transaction_analysis",
            "metrics": {
                "analysis_period_revenue": {
                    "value": round(analysis_revenue, 2),
                    "unit": "SAR",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_period_revenue": {
                    "value": round(previous_revenue, 2),
                    "unit": "SAR",
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "revenue_change_absolute": {
                    "value": round(revenue_change, 2),
                    "unit": "SAR",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "revenue_change_percent": {
                    "value": round(revenue_pct_change, 2),
                    "unit": "%",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_period_transactions": {
                    "value": analysis_transactions,
                    "unit": "count",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_period_transactions": {
                    "value": previous_transactions,
                    "unit": "count",
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "transaction_change_absolute": {
                    "value": transaction_change,
                    "unit": "count",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "transaction_change_percent": {
                    "value": round(transaction_pct_change, 2),
                    "unit": "%",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_aov": {
                    "value": round(aov_analysis, 2),
                    "unit": "SAR",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_aov": {
                    "value": round(aov_previous, 2),
                    "unit": "SAR",
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "aov_change": {
                    "value": round(aov_change, 2),
                    "unit": "SAR",
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period: {len(analysis_data)} POS line items from {analysis_transactions} unique transactions",
                f"Previous period: {len(previous_data)} POS line items from {previous_transactions} unique transactions",
                "Refunds excluded from revenue and transaction counts",
                "Line totals used as reported in cleaned POS data"
            ],
            "assumptions": [
                "business_date field accurately represents transaction date",
                "is_refund flag correctly identifies refund transactions",
                "line_total_sar represents net revenue after discounts",
                "transaction_id uniquely identifies a basket/transaction"
            ],
            "confidence": 0.95
        })
    
    # Finding 2: Category Mix Analysis
    analysis_category_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    previous_category_revenue = previous_data[previous_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
    
    if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
        # Get top category in analysis period
        top_category = analysis_category_revenue.index[0]
        top_category_analysis = analysis_category_revenue.iloc[0]
        top_category_previous = previous_category_revenue.get(top_category, 0)
        
        if top_category_previous > 0:
            category_change = top_category_analysis - top_category_previous
            category_pct_change = (category_change / top_category_previous) * 100
            
            analysis_category_pct = (top_category_analysis / analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()) * 100
            previous_category_pct = (top_category_previous / previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()) * 100
            
            findings.append({
                "title": f"Top Category Performance: {top_category}",
                "claim": f"The {top_category} category generated SAR {top_category_analysis:,.2f} in the analysis week, up from SAR {top_category_previous:,.2f} in the previous week, representing a change of SAR {category_change:+,.2f} ({category_pct_change:+.1f}%). This category's share of total revenue increased from {previous_category_pct:.1f}% to {analysis_category_pct:.1f}%.",
                "finding_type": "category_mix_analysis",
                "metrics": {
                    "analysis_category_revenue": {
                        "value": round(top_category_analysis, 2),
                        "unit": "SAR",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_category_revenue": {
                        "value": round(top_category_previous, 2),
                        "unit": "SAR",
                        "period_start": previous_start,
                        "period_end": previous_end
                    },
                    "category_revenue_change": {
                        "value": round(category_change, 2),
                        "unit": "SAR",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "category_revenue_change_percent": {
                        "value": round(category_pct_change, 2),
                        "unit": "%",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "analysis_category_share": {
                        "value": round(analysis_category_pct, 2),
                        "unit": "%",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_category_share": {
                        "value": round(previous_category_pct, 2),
                        "unit": "%",
                        "period_start": previous_start,
                        "period_end": previous_end
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_data),
                "coverage_notes": [
                    f"Analysis period category breakdown based on {len(analysis_data)} line items",
                    f"Previous period category breakdown based on {len(previous_data)} line items",
                    "Refunds excluded from category revenue calculations",
                    f"Top category identified as: {top_category}"
                ],
                "assumptions": [
                    "category field accurately represents product category",
                    "is_refund flag correctly identifies refund transactions",
                    "line_total_sar represents net revenue after discounts"
                ],
                "confidence": 0.92
            })
    
    # Finding 3: Channel Mix Analysis
    analysis_channel_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    previous_channel_revenue = previous_data[previous_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
    
    if len(analysis_channel_revenue) > 0 and len(previous_channel_revenue) > 0:
        # Get top channel in analysis period
        top_channel = analysis_channel_revenue.index[0]
        top_channel_analysis = analysis_channel_revenue.iloc[0]
        top_channel_previous = previous_channel_revenue.get(top_channel, 0)
        
        if top_channel_previous > 0:
            channel_change = top_channel_analysis - top_channel_previous
            channel_pct_change = (channel_change / top_channel_previous) * 100
            
            analysis_channel_pct = (top_channel_analysis / analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum()) * 100
            previous_channel_pct = (top_channel_previous / previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum()) * 100
            
            findings.append({
                "title": f"Top Channel Performance: {top_channel}",
                "claim": f"The {top_channel} channel generated SAR {top_channel_analysis:,.2f} in the analysis week, compared to SAR {top_channel_previous:,.2f} in the previous week, representing a change of SAR {channel_change:+,.2f} ({channel_pct_change:+.1f}%). This channel's share of total revenue changed from {previous_channel_pct:.1f}% to {analysis_channel_pct:.1f}%.",
                "finding_type": "channel_mix_analysis",
                "metrics": {
                    "analysis_channel_revenue": {
                        "value": round(top_channel_analysis, 2),
                        "unit": "SAR",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_channel_revenue": {
                        "value": round(top_channel_previous, 2),
                        "unit": "SAR",
                        "period_start": previous_start,
                        "period_end": previous_end
                    },
                    "channel_revenue_change": {
                        "value": round(channel_change, 2),
                        "unit": "SAR",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "channel_revenue_change_percent": {
                        "value": round(channel_pct_change, 2),
                        "unit": "%",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "analysis_channel_share": {
                        "value": round(analysis_channel_pct, 2),
                        "unit": "%",
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "previous_channel_share": {
                        "value": round(previous_channel_pct, 2),
                        "unit": "%",
                        "period_start": previous_start,
                        "period_end": previous_end
                    }
                },
                "source_names": ["pos"],
                "sample_size": len(analysis_data),
                "coverage_notes": [
                    f"Analysis period channel breakdown based on {len(analysis_data)} line items",
                    f"Previous period channel breakdown based on {len(previous_data)} line items",
                    "Refunds excluded from channel revenue calculations",
                    f"Top channel identified as: {top_channel}"
                ],
                "assumptions": [
                    "channel field accurately represents sales channel",
                    "is_refund flag correctly identifies refund transactions",
                    "line_total_sar represents net revenue after discounts"
                ],
                "confidence": 0.92
            })
    
    # Prepare output
    output = {
        "status": "success" if len(findings) > 0 else "insufficient_data",
        "findings": findings[:3]  # Return at most 3 findings
    }
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
