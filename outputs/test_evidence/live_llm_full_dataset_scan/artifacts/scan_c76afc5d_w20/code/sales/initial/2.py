import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
analysis_end = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_start = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))
previous_end = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone(timedelta(hours=3)))

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis period
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Trailing baseline (4 weeks before analysis period)
trailing_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

findings = []

# Finding 1: Revenue and Transaction Count Analysis
analysis_valid_txns = int(analysis_data[analysis_data['is_refund'] == False]['transaction_id'].nunique())
analysis_revenue = float(analysis_data[analysis_data['is_refund'] == False]['line_total_sar'].sum())

previous_valid_txns = int(previous_data[previous_data['is_refund'] == False]['transaction_id'].nunique())
previous_revenue = float(previous_data[previous_data['is_refund'] == False]['line_total_sar'].sum())

if analysis_valid_txns > 0 and previous_valid_txns > 0:
    revenue_change = analysis_revenue - previous_revenue
    revenue_pct_change = (revenue_change / previous_revenue * 100) if previous_revenue != 0 else 0
    txn_change = analysis_valid_txns - previous_valid_txns
    txn_pct_change = (txn_change / previous_valid_txns * 100) if previous_valid_txns != 0 else 0
    
    analysis_aov = analysis_revenue / analysis_valid_txns if analysis_valid_txns > 0 else 0
    previous_aov = previous_revenue / previous_valid_txns if previous_valid_txns > 0 else 0
    aov_change = analysis_aov - previous_aov
    aov_pct_change = (aov_change / previous_aov * 100) if previous_aov != 0 else 0
    
    findings.append({
        "title": "Revenue and Transaction Performance Week-over-Week",
        "claim": f"Analysis period (2026-05-25 to 2026-06-01) generated SAR {analysis_revenue:.2f} in net revenue across {analysis_valid_txns} valid transactions, compared to SAR {previous_revenue:.2f} across {previous_valid_txns} transactions in the previous week. Revenue increased by SAR {revenue_change:.2f} ({revenue_pct_change:.1f}%), while transaction count changed by {txn_change} ({txn_pct_change:.1f}%). Average order value increased from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f} ({aov_pct_change:.1f}% change).",
        "finding_type": "revenue_and_transaction_analysis",
        "metrics": {
            "analysis_period_revenue": {
                "value": round(analysis_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "analysis_period_transactions": {
                "value": analysis_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "previous_period_revenue": {
                "value": round(previous_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "previous_period_transactions": {
                "value": previous_valid_txns,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "revenue_change_absolute": {
                "value": round(revenue_change, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "revenue_change_percent": {
                "value": round(revenue_pct_change, 2),
                "unit": "%",
                "numerator": round(revenue_change, 2),
                "denominator": round(previous_revenue, 2),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "transaction_change_absolute": {
                "value": int(txn_change),
                "unit": "count",
                "numerator": analysis_valid_txns,
                "denominator": previous_valid_txns,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "transaction_change_percent": {
                "value": round(txn_pct_change, 2),
                "unit": "%",
                "numerator": int(txn_change),
                "denominator": previous_valid_txns,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "analysis_aov": {
                "value": round(analysis_aov, 2),
                "unit": "SAR",
                "numerator": round(analysis_revenue, 2),
                "denominator": analysis_valid_txns,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "previous_aov": {
                "value": round(previous_aov, 2),
                "unit": "SAR",
                "numerator": round(previous_revenue, 2),
                "denominator": previous_valid_txns,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "aov_change_percent": {
                "value": round(aov_pct_change, 2),
                "unit": "%",
                "numerator": round(aov_change, 2),
                "denominator": round(previous_aov, 2),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            }
        },
        "source_names": ["pos"],
        "sample_size": len(analysis_data),
        "coverage_notes": [
            "Analysis period: 2026-05-25 to 2026-06-01",
            "Previous period: 2026-05-18 to 2026-05-25",
            "Refunds excluded from revenue and transaction counts",
            "Valid transactions identified by unique transaction_id",
            "Revenue calculated using line_total_sar"
        ],
        "assumptions": [
            "Refunds are correctly marked in is_refund column",
            "transaction_id uniquely identifies a basket",
            "line_total_sar represents net revenue after discounts",
            "Timestamp conversion to UTC+3 is accurate"
        ],
        "confidence": 0.95
    })

# Finding 2: Category Mix Analysis
analysis_category_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)
previous_category_revenue = previous_data[previous_data['is_refund'] == False].groupby('category')['line_total_sar'].sum().sort_values(ascending=False)

if len(analysis_category_revenue) > 0 and len(previous_category_revenue) > 0:
    # Find top category changes
    top_category_analysis = analysis_category_revenue.index[0] if len(analysis_category_revenue) > 0 else None
    top_category_previous = previous_category_revenue.index[0] if len(previous_category_revenue) > 0 else None
    
    if top_category_analysis and top_category_analysis in previous_category_revenue.index:
        analysis_top_rev = float(analysis_category_revenue[top_category_analysis])
        previous_top_rev = float(previous_category_revenue[top_category_analysis])
        top_cat_change = analysis_top_rev - previous_top_rev
        top_cat_pct_change = (top_cat_change / previous_top_rev * 100) if previous_top_rev != 0 else 0
        
        analysis_total_rev = float(analysis_category_revenue.sum())
        previous_total_rev = float(previous_category_revenue.sum())
        
        analysis_top_pct = (analysis_top_rev / analysis_total_rev * 100) if analysis_total_rev != 0 else 0
        previous_top_pct = (previous_top_rev / previous_total_rev * 100) if previous_total_rev != 0 else 0
        
        findings.append({
            "title": "Top Category Revenue Performance",
            "claim": f"The {top_category_analysis} category generated SAR {analysis_top_rev:.2f} in the analysis period (2026-05-25 to 2026-06-01), representing {analysis_top_pct:.1f}% of total revenue. This represents a change of SAR {top_cat_change:.2f} ({top_cat_pct_change:.1f}%) compared to SAR {previous_top_rev:.2f} ({previous_top_pct:.1f}% of total) in the previous week.",
            "finding_type": "category_mix_analysis",
            "metrics": {
                "analysis_top_category": {
                    "value": str(top_category_analysis),
                    "unit": "category",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "analysis_top_category_revenue": {
                    "value": round(analysis_top_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "analysis_top_category_percent": {
                    "value": round(analysis_top_pct, 2),
                    "unit": "%",
                    "numerator": round(analysis_top_rev, 2),
                    "denominator": round(analysis_total_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "previous_top_category_revenue": {
                    "value": round(previous_top_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "previous_top_category_percent": {
                    "value": round(previous_top_pct, 2),
                    "unit": "%",
                    "numerator": round(previous_top_rev, 2),
                    "denominator": round(previous_total_rev, 2),
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "category_revenue_change": {
                    "value": round(top_cat_change, 2),
                    "unit": "SAR",
                    "numerator": round(analysis_top_rev, 2),
                    "denominator": round(previous_top_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "category_revenue_change_percent": {
                    "value": round(top_cat_pct_change, 2),
                    "unit": "%",
                    "numerator": round(top_cat_change, 2),
                    "denominator": round(previous_top_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Analysis period: 2026-05-25 to 2026-06-01",
                "Previous period: 2026-05-18 to 2026-05-25",
                "Refunds excluded from category revenue calculations",
                "Revenue calculated using line_total_sar"
            ],
            "assumptions": [
                "Category field is accurately populated in POS data",
                "Refunds are correctly marked in is_refund column",
                "line_total_sar represents net revenue after discounts"
            ],
            "confidence": 0.92
        })

# Finding 3: Channel Mix Analysis
analysis_channel_revenue = analysis_data[analysis_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)
previous_channel_revenue = previous_data[previous_data['is_refund'] == False].groupby('channel')['line_total_sar'].sum().sort_values(ascending=False)

analysis_channel_txns = analysis_data[analysis_data['is_refund'] == False].groupby('channel')['transaction_id'].nunique()
previous_channel_txns = previous_data[previous_data['is_refund'] == False].groupby('channel')['transaction_id'].nunique()

if len(analysis_channel_revenue) > 0 and len(previous_channel_revenue) > 0:
    top_channel_analysis = analysis_channel_revenue.index[0] if len(analysis_channel_revenue) > 0 else None
    
    if top_channel_analysis and top_channel_analysis in previous_channel_revenue.index:
        analysis_top_ch_rev = float(analysis_channel_revenue[top_channel_analysis])
        previous_top_ch_rev = float(previous_channel_revenue[top_channel_analysis])
        top_ch_change = analysis_top_ch_rev - previous_top_ch_rev
        top_ch_pct_change = (top_ch_change / previous_top_ch_rev * 100) if previous_top_ch_rev != 0 else 0
        
        analysis_top_ch_txns = int(analysis_channel_txns.get(top_channel_analysis, 0))
        previous_top_ch_txns = int(previous_channel_txns.get(top_channel_analysis, 0))
        
        analysis_total_ch_rev = float(analysis_channel_revenue.sum())
        previous_total_ch_rev = float(previous_channel_revenue.sum())
        
        analysis_top_ch_pct = (analysis_top_ch_rev / analysis_total_ch_rev * 100) if analysis_total_ch_rev != 0 else 0
        previous_top_ch_pct = (previous_top_ch_rev / previous_total_ch_rev * 100) if previous_total_ch_rev != 0 else 0
        
        findings.append({
            "title": "Primary Channel Revenue and Transaction Performance",
            "claim": f"The {top_channel_analysis} channel generated SAR {analysis_top_ch_rev:.2f} across {analysis_top_ch_txns} transactions in the analysis period (2026-05-25 to 2026-06-01), representing {analysis_top_ch_pct:.1f}% of total revenue. This represents a change of SAR {top_ch_change:.2f} ({top_ch_pct_change:.1f}%) compared to SAR {previous_top_ch_rev:.2f} across {previous_top_ch_txns} transactions ({previous_top_ch_pct:.1f}% of total) in the previous week.",
            "finding_type": "channel_mix_analysis",
            "metrics": {
                "analysis_top_channel": {
                    "value": str(top_channel_analysis),
                    "unit": "channel",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "analysis_top_channel_revenue": {
                    "value": round(analysis_top_ch_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "analysis_top_channel_transactions": {
                    "value": analysis_top_ch_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "analysis_top_channel_percent": {
                    "value": round(analysis_top_ch_pct, 2),
                    "unit": "%",
                    "numerator": round(analysis_top_ch_rev, 2),
                    "denominator": round(analysis_total_ch_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "previous_top_channel_revenue": {
                    "value": round(previous_top_ch_rev, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "previous_top_channel_transactions": {
                    "value": previous_top_ch_txns,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "previous_top_channel_percent": {
                    "value": round(previous_top_ch_pct, 2),
                    "unit": "%",
                    "numerator": round(previous_top_ch_rev, 2),
                    "denominator": round(previous_total_ch_rev, 2),
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "channel_revenue_change": {
                    "value": round(top_ch_change, 2),
                    "unit": "SAR",
                    "numerator": round(analysis_top_ch_rev, 2),
                    "denominator": round(previous_top_ch_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "channel_revenue_change_percent": {
                    "value": round(top_ch_pct_change, 2),
                    "unit": "%",
                    "numerator": round(top_ch_change, 2),
                    "denominator": round(previous_top_ch_rev, 2),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                }
            },
            "source_names": ["pos"],
            "sample_size": len(analysis_data),
            "coverage_notes": [
                "Analysis period: 2026-05-25 to 2026-06-01",
                "Previous period: 2026-05-18 to 2026-05-25",
                "Refunds excluded from channel revenue and transaction counts",
                "Valid transactions identified by unique transaction_id",
                "Revenue calculated using line_total_sar"
            ],
            "assumptions": [
                "Channel field is accurately populated in POS data",
                "Refunds are correctly marked in is_refund column",
                "transaction_id uniquely identifies a basket",
                "line_total_sar represents net revenue after discounts"
            ],
            "confidence": 0.93
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
