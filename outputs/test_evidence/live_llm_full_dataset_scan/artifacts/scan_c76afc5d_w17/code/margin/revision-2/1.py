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
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = pd.Timestamp("2026-05-04T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-05-11T00:00:00+03:00")

# Convert POS timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS to analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()

# Exclude refunds from revenue/profit calculations
pos_sales = pos_analysis[~pos_analysis['is_refund']].copy()

# Merge POS with menu to get unit costs
pos_with_cost = pos_sales.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate line-level COGS and gross profit
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['line_cogs_sar']

# Item-level aggregation for menu engineering
item_metrics = pos_with_cost.groupby('sku').agg({
    'item_name_en': 'first',
    'category': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_metrics.columns = ['sku', 'item_name', 'category', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'transaction_count']
item_metrics['gross_margin_pct'] = (item_metrics['total_gross_profit'] / item_metrics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_metrics_sorted = item_metrics.sort_values('total_gross_profit', ascending=False)

# Finding 1: Top 3 items by absolute gross profit contribution
top_3_items = item_metrics_sorted.head(3)
top_3_gp_total = top_3_items['total_gross_profit'].sum()
top_3_transaction_count = top_3_items['transaction_count'].sum()

finding_1 = {
    "title": "Top 3 Items by Gross Profit Contribution",
    "claim": f"During the analysis week (May 4-11, 2026), the top 3 items by absolute gross profit contribution are {top_3_items.iloc[0]['item_name']}, {top_3_items.iloc[1]['item_name']}, and {top_3_items.iloc[2]['item_name']}. These items generated {top_3_gp_total:.2f} SAR in total gross profit.",
    "finding_type": "menu_engineering",
    "metrics": {
        "top_item_1_name": {
            "value": top_3_items.iloc[0]['item_name'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_1_gross_profit_sar": {
            "value": round(top_3_items.iloc[0]['total_gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(top_3_items.iloc[0]['total_gross_profit'], 2),
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_1_gross_margin_pct": {
            "value": round(top_3_items.iloc[0]['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(top_3_items.iloc[0]['total_gross_profit'], 2),
            "denominator": round(top_3_items.iloc[0]['total_revenue'], 2),
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_1_transaction_count": {
            "value": int(top_3_items.iloc[0]['transaction_count']),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_2_name": {
            "value": top_3_items.iloc[1]['item_name'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_2_gross_profit_sar": {
            "value": round(top_3_items.iloc[1]['total_gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(top_3_items.iloc[1]['total_gross_profit'], 2),
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_2_gross_margin_pct": {
            "value": round(top_3_items.iloc[1]['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(top_3_items.iloc[1]['total_gross_profit'], 2),
            "denominator": round(top_3_items.iloc[1]['total_revenue'], 2),
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_2_transaction_count": {
            "value": int(top_3_items.iloc[1]['transaction_count']),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_3_name": {
            "value": top_3_items.iloc[2]['item_name'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_3_gross_profit_sar": {
            "value": round(top_3_items.iloc[2]['total_gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(top_3_items.iloc[2]['total_gross_profit'], 2),
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_3_gross_margin_pct": {
            "value": round(top_3_items.iloc[2]['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(top_3_items.iloc[2]['total_gross_profit'], 2),
            "denominator": round(top_3_items.iloc[2]['total_revenue'], 2),
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_item_3_transaction_count": {
            "value": int(top_3_items.iloc[2]['transaction_count']),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_3_combined_gross_profit_sar": {
            "value": round(top_3_gp_total, 2),
            "unit": "SAR",
            "numerator": round(top_3_gp_total, 2),
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        },
        "top_3_combined_transaction_count": {
            "value": int(top_3_transaction_count),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start.isoformat(),
            "period_end": analysis_end.isoformat()
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": int(len(pos_sales)),
    "coverage_notes": [
        f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
        f"Total POS line items in analysis period: {len(pos_sales)}",
        f"Total transactions in analysis period: {pos_sales['transaction_id'].nunique()}",
        "Refunds excluded from revenue and profit calculations",
        "Unit costs sourced from menu.parquet",
        "All items with non-null unit_cost_sar included"
    ],
    "assumptions": [
        "Unit costs from menu are current and applicable to analysis period",
        "POS line_total_sar is net of discounts and represents actual revenue",
        "Gross profit = revenue - (quantity × unit_cost_sar)",
        "No recipe/BOM data available; analysis is at item level only"
    ],
    "confidence": 0.95
}

# Finding 2: Supplier price changes from emails
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter for price changes with both old and new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['old_price'] != 0)
].copy()

if len(price_changes) > 0:
    price_changes['price_delta'] = price_changes['new_price'] - price_changes['old_price']
    price_changes['pct_change'] = (price_changes['price_delta'] / price_changes['old_price'] * 100).round(2)
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    
    most_recent_change = price_changes_sorted.iloc[0]
    
    finding_2 = {
        "title": "Supplier Price Changes Detected",
        "claim": f"Email evidence identifies {len(price_changes)} supplier price changes. The most recent is for {most_recent_change['entity_or_ingredient']} with effective date {most_recent_change['effective_date'].strftime('%Y-%m-%d')}: old price {most_recent_change['old_price']} {most_recent_change['currency']}/{most_recent_change['unit']}, new price {most_recent_change['new_price']} {most_recent_change['currency']}/{most_recent_change['unit']}, representing a {most_recent_change['pct_change']:.2f}% increase.",
        "finding_type": "supplier_cost_change",
        "metrics": {
            "total_price_changes_detected": {
                "value": len(price_changes),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_ingredient": {
                "value": most_recent_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_old_price": {
                "value": round(most_recent_change['old_price'], 2),
                "unit": f"{most_recent_change['currency']}/{most_recent_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_new_price": {
                "value": round(most_recent_change['new_price'], 2),
                "unit": f"{most_recent_change['currency']}/{most_recent_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_price_delta": {
                "value": round(most_recent_change['price_delta'], 2),
                "unit": f"{most_recent_change['currency']}/{most_recent_change['unit']}",
                "numerator": round(most_recent_change['price_delta'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_pct_change": {
                "value": round(most_recent_change['pct_change'], 2),
                "unit": "%",
                "numerator": round(most_recent_change['pct_change'], 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "most_recent_effective_date": {
                "value": most_recent_change['effective_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Email extraction identified {len(price_changes)} price changes with both old and new prices",
            "Most recent effective date: " + most_recent_change['effective_date'].strftime('%Y-%m-%d'),
            "Price changes span multiple suppliers and ingredients",
            "No recipe/bill-of-materials data available to translate supplier prices to per-drink cost impact"
        ],
        "assumptions": [
            "Email extraction confidence and facts are as reported in emails.parquet",
            "Effective dates represent when price changes take effect",
            "Percentage change calculated as (new_price - old_price) / old_price × 100"
        ],
        "confidence": 0.85
    }
else:
    finding_2 = None

# Finding 3: Waste cost analysis
inventory_analysis = inventory_df.copy()
inventory_analysis['week_starting'] = pd.to_datetime(inventory_analysis['week_starting'])

# Filter inventory to analysis period (week starting May 4)
inv_analysis_week = inventory_analysis[
    inventory_analysis['week_starting'] == pd.Timestamp("2026-05-04")
].copy()

if len(inv_analysis_week) > 0:
    # Only include rows with non-null waste cost
    waste_items = inv_analysis_week[inv_analysis_week['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_waste_units = waste_items['units_wasted'].sum()
        
        waste_items_sorted = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        finding_3 = {
            "title": "Quantified Waste Cost in Analysis Week",
            "claim": f"During the week of May 4-11, 2026, {len(waste_items)} items with recorded waste generated a total known waste cost of {total_waste_cost:.2f} SAR, representing {total_waste_units:.0f} units wasted.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_waste_units": {
                    "value": round(total_waste_units, 2),
                    "unit": "units",
                    "numerator": round(total_waste_units, 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "items_with_waste_recorded": {
                    "value": len(waste_items),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "avg_waste_cost_per_item": {
                    "value": round(total_waste_cost / len(waste_items), 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": len(waste_items),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                f"Analysis week: May 4-11, 2026",
                f"Items with non-null known_waste_cost_sar: {len(waste_items)}",
                f"Total inventory records for week: {len(inv_analysis_week)}",
                "Waste cost only included where known_waste_cost_sar is non-null",
                "Blank waste values treated as missing, not zero"
            ],
            "assumptions": [
                "known_waste_cost_sar represents actual quantified waste cost",
                "Units wasted and waste cost are from inventory records",
                "Waste cost calculation methodology is as defined in inventory source"
            ],
            "confidence": 0.90
        }
    else:
        finding_3 = None
else:
    finding_3 = None

# Compile findings
findings = [f for f in [finding_1, finding_2, finding_3] if f is not None]

# Build output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
