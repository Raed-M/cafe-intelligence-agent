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

def is_product_eligible(sku, menu_df, period_start, period_end):
    """Check if product is eligible for analysis in the given period."""
    period_start_date = parse_iso_datetime(period_start).date()
    period_end_date = parse_iso_datetime(period_end).date()
    
    product = menu_df[menu_df['sku'] == sku]
    if product.empty:
        return False
    
    launch_date = product.iloc[0]['launch_date']
    retire_date = product.iloc[0]['retire_date']
    
    # Check launch date
    if pd.notna(launch_date):
        launch = parse_iso_datetime(launch_date).date()
        if period_start_date < launch:
            return False
    
    # Check retire date
    if pd.notna(retire_date):
        retire = parse_iso_datetime(retire_date).date()
        if period_end_date > retire:
            return False
    
    return True

def main():
    inputs, output_path = load_inputs()
    
    # Load artifacts
    pos_df = pd.read_parquet(inputs['pos'])
    menu_df = pd.read_parquet(inputs['menu'])
    
    # Define periods
    analysis_start = "2026-06-08T00:00:00+03:00"
    analysis_end = "2026-06-15T00:00:00+03:00"
    previous_start = "2026-06-01T00:00:00+03:00"
    previous_end = "2026-06-08T00:00:00+03:00"
    
    trailing_baselines = [
        ("2026-06-01T00:00:00+03:00", "2026-06-08T00:00:00+03:00"),
        ("2026-05-25T00:00:00+03:00", "2026-06-01T00:00:00+03:00"),
        ("2026-05-18T00:00:00+03:00", "2026-05-25T00:00:00+03:00"),
        ("2026-05-11T00:00:00+03:00", "2026-05-18T00:00:00+03:00"),
    ]
    
    # Filter data for each period
    analysis_data = filter_by_period(pos_df, analysis_start, analysis_end)
    previous_data = filter_by_period(pos_df, previous_start, previous_end)
    
    trailing_data_list = []
    for start, end in trailing_baselines:
        trailing_data_list.append(filter_by_period(pos_df, start, end))
    
    findings = []
    
    # FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
    if len(analysis_data) > 0 and len(previous_data) > 0:
        # Count valid transactions (unique transaction_id)
        analysis_txns = analysis_data['transaction_id'].nunique()
        previous_txns = previous_data['transaction_id'].nunique()
        
        # Calculate net revenue (including refunds)
        analysis_revenue = analysis_data['line_total_sar'].sum()
        previous_revenue = previous_data['line_total_sar'].sum()
        
        # Calculate AOV
        analysis_aov = analysis_revenue / analysis_txns if analysis_txns > 0 else 0
        previous_aov = previous_revenue / previous_txns if previous_txns > 0 else 0
        
        revenue_change = analysis_revenue - previous_revenue
        revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
        
        txn_change = analysis_txns - previous_txns
        txn_pct_change = (txn_change / previous_txns * 100) if previous_txns > 0 else 0
        
        aov_change = analysis_aov - previous_aov
        aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0
        
        findings.append({
            "title": "Weekly Revenue and Transaction Performance",
            "claim": f"Week of 2026-06-08 generated SAR {analysis_revenue:.2f} in net revenue across {analysis_txns} transactions (AOV: SAR {analysis_aov:.2f}), compared to SAR {previous_revenue:.2f} across {previous_txns} transactions (AOV: SAR {previous_aov:.2f}) in the previous week.",
            "finding_type": "revenue_and_transaction_analysis",
            "metrics": {
                "analysis_period_revenue": {
                    "value": round(analysis_revenue, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_period_revenue": {
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
                    "value": round(revenue_pct_change, 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_transactions": {
                    "value": analysis_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_transactions": {
                    "value": previous_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "transaction_change": {
                    "value": txn_change,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "transaction_change_pct": {
                    "value": round(txn_pct_change, 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_aov": {
                    "value": round(analysis_aov, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_aov": {
                    "value": round(previous_aov, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "aov_change": {
                    "value": round(aov_change, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "aov_change_pct": {
                    "value": round(aov_pct_change, 2),
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
                f"Analysis period: {len(analysis_data)} POS line items across {analysis_txns} unique transactions",
                f"Previous period: {len(previous_data)} POS line items across {previous_txns} unique transactions",
                "Net revenue includes refunds as per metric definition",
                "AOV calculated as total revenue divided by unique transaction count"
            ],
            "assumptions": [
                "business_date field used for period filtering",
                "transaction_id uniqueness defines basket count",
                "line_total_sar represents net realised revenue including refunds",
                "All rows in cleaned POS are valid for analysis"
            ],
            "confidence": 0.95
        })
    
    # FINDING 2: Category Mix Analysis (Analysis vs Trailing Baseline Average)
    if len(analysis_data) > 0 and len(trailing_data_list) > 0:
        # Calculate category revenue for analysis period
        analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
        analysis_category_txns = analysis_data.groupby('category')['transaction_id'].nunique()
        
        # Calculate average for trailing baseline
        trailing_combined = pd.concat(trailing_data_list, ignore_index=True)
        trailing_category_revenue = trailing_combined.groupby('category')['line_total_sar'].sum()
        trailing_category_txns = trailing_combined.groupby('category')['transaction_id'].nunique()
        
        # Calculate averages
        trailing_avg_revenue = trailing_category_revenue / len(trailing_baselines)
        trailing_avg_txns = trailing_category_txns / len(trailing_baselines)
        
        # Find top category with significant change
        top_category = analysis_category_revenue.index[0]
        analysis_top_rev = analysis_category_revenue.iloc[0]
        trailing_top_rev = trailing_avg_revenue.get(top_category, 0)
        
        top_change = analysis_top_rev - trailing_top_rev
        top_pct_change = (top_change / trailing_top_rev * 100) if trailing_top_rev != 0 else 0
        
        analysis_top_txns = analysis_category_txns.get(top_category, 0)
        trailing_top_txns = trailing_avg_txns.get(top_category, 0)
        
        findings.append({
            "title": "Top Category Revenue Performance vs Trailing Baseline",
            "claim": f"The {top_category} category generated SAR {analysis_top_rev:.2f} in the analysis week (2026-06-08 to 2026-06-15) across {analysis_top_txns} transactions, compared to a trailing 4-week average of SAR {trailing_top_rev:.2f} across {trailing_top_txns:.0f} transactions.",
            "finding_type": "category_mix_analysis",
            "metrics": {
                "analysis_top_category_revenue": {
                    "value": round(analysis_top_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "trailing_baseline_avg_revenue": {
                    "value": round(trailing_top_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-11T00:00:00+03:00",
                    "period_end": "2026-06-08T00:00:00+03:00"
                },
                "revenue_change_vs_baseline": {
                    "value": round(top_change, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "revenue_change_pct_vs_baseline": {
                    "value": round(top_pct_change, 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_top_category_transactions": {
                    "value": analysis_top_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "trailing_baseline_avg_transactions": {
                    "value": round(trailing_top_txns, 2),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-11T00:00:00+03:00",
                    "period_end": "2026-06-08T00:00:00+03:00"
                },
                "top_category_name": {
                    "value": top_category,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period: {len(analysis_data)} line items",
                f"Trailing baseline: {len(trailing_combined)} line items across 4 weeks",
                "Category field from cleaned POS data",
                "Revenue includes refunds in net calculation"
            ],
            "assumptions": [
                "Category field is authoritative from cleaned POS",
                "Trailing baseline average calculated from 4 equal-weight weeks",
                "transaction_id uniqueness for transaction counting",
                "All rows valid for category analysis"
            ],
            "confidence": 0.90
        })
    
    # FINDING 3: Channel Mix Analysis (Analysis vs Previous Week)
    if len(analysis_data) > 0 and len(previous_data) > 0:
        # Calculate channel revenue and transaction distribution
        analysis_channel_revenue = analysis_data.groupby('channel')['line_total_sar'].sum()
        analysis_channel_txns = analysis_data.groupby('channel')['transaction_id'].nunique()
        
        previous_channel_revenue = previous_data.groupby('channel')['line_total_sar'].sum()
        previous_channel_txns = previous_data.groupby('channel')['transaction_id'].nunique()
        
        # Find channel with largest absolute change
        all_channels = set(analysis_channel_revenue.index) | set(previous_channel_revenue.index)
        channel_changes = {}
        
        for channel in all_channels:
            analysis_rev = analysis_channel_revenue.get(channel, 0)
            previous_rev = previous_channel_revenue.get(channel, 0)
            change = analysis_rev - previous_rev
            channel_changes[channel] = {
                'analysis_rev': analysis_rev,
                'previous_rev': previous_rev,
                'change': change,
                'analysis_txns': analysis_channel_txns.get(channel, 0),
                'previous_txns': previous_channel_txns.get(channel, 0)
            }
        
        # Get channel with largest absolute change
        max_channel = max(channel_changes.items(), key=lambda x: abs(x[1]['change']))
        channel_name = max_channel[0]
        channel_data = max_channel[1]
        
        change_pct = (channel_data['change'] / channel_data['previous_rev'] * 100) if channel_data['previous_rev'] != 0 else 0
        
        findings.append({
            "title": "Channel Revenue Performance Change",
            "claim": f"The {channel_name} channel generated SAR {channel_data['analysis_rev']:.2f} in the analysis week (2026-06-08 to 2026-06-15) across {channel_data['analysis_txns']} transactions, compared to SAR {channel_data['previous_rev']:.2f} across {channel_data['previous_txns']} transactions in the previous week.",
            "finding_type": "channel_mix_analysis",
            "metrics": {
                "analysis_channel_revenue": {
                    "value": round(channel_data['analysis_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_channel_revenue": {
                    "value": round(channel_data['previous_rev'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "channel_revenue_change": {
                    "value": round(channel_data['change'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "channel_revenue_change_pct": {
                    "value": round(change_pct, 2),
                    "unit": "%",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "analysis_channel_transactions": {
                    "value": channel_data['analysis_txns'],
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "previous_channel_transactions": {
                    "value": channel_data['previous_txns'],
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start,
                    "period_end": previous_end
                },
                "channel_name": {
                    "value": channel_name,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                f"Analysis period: {len(analysis_data)} line items",
                f"Previous period: {len(previous_data)} line items",
                "Channel field from cleaned POS data",
                "Revenue includes refunds in net calculation"
            ],
            "assumptions": [
                "Channel field is authoritative from cleaned POS",
                "transaction_id uniqueness for transaction counting",
                "All rows valid for channel analysis",
                "Selected channel with largest absolute revenue change for reporting"
            ],
            "confidence": 0.92
        })
    
    # Prepare output
    output = {
        "status": "success" if len(findings) > 0 else "insufficient_data",
        "findings": findings
    }
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()