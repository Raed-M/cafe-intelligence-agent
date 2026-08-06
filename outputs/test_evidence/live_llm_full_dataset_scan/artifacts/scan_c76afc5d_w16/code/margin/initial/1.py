import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-04-27T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-05-04T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-27T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis period
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp("2026-04-27", tz='UTC')].copy()
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp("2026-04-20", tz='UTC')].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
if len(pos_analysis) > 0 and len(menu_df) > 0:
    # Merge POS with menu to get unit costs
    pos_with_cost = pos_analysis.merge(
        menu_df[['sku', 'unit_cost_sar', 'price_sar']],
        on='sku',
        how='left'
    )
    
    # Filter out refunds for revenue calculation
    pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()
    
    # Calculate metrics by item
    item_metrics = pos_sales.groupby('item_name').agg({
        'quantity': 'sum',
        'line_total_sar': 'sum',
        'unit_cost_sar': 'first',
        'price_sar': 'first',
        'transaction_id': 'nunique'
    }).reset_index()
    
    item_metrics.columns = ['item_name', 'total_quantity', 'total_revenue', 'unit_cost', 'menu_price', 'basket_count']
    
    # Calculate COGS and gross profit
    item_metrics['total_cogs'] = item_metrics['total_quantity'] * item_metrics['unit_cost']
    item_metrics['gross_profit'] = item_metrics['total_revenue'] - item_metrics['total_cogs']
    item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['total_revenue'] * 100).round(2)
    
    # Sort by gross profit
    item_metrics_sorted = item_metrics.sort_values('gross_profit', ascending=False)
    
    # Top 3 items by gross profit
    top_items = item_metrics_sorted.head(3)
    
    if len(top_items) > 0:
        top_item = top_items.iloc[0]
        finding_1 = {
            "title": "Top Gross Profit Item - Analysis Period",
            "claim": f"{top_item['item_name']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin",
            "finding_type": "item_economics",
            "metrics": {
                "item_name": {
                    "value": top_item['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "total_revenue": {
                    "value": round(top_item['total_revenue'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['total_revenue'], 2),
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "total_cogs": {
                    "value": round(top_item['total_cogs'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['total_cogs'], 2),
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "gross_profit": {
                    "value": round(top_item['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "gross_margin_pct": {
                    "value": top_item['gross_margin_pct'],
                    "unit": "%",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": round(top_item['total_revenue'], 2),
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "total_quantity": {
                    "value": int(top_item['total_quantity']),
                    "unit": "units",
                    "numerator": int(top_item['total_quantity']),
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                },
                "basket_count": {
                    "value": int(top_item['basket_count']),
                    "unit": "transactions",
                    "numerator": int(top_item['basket_count']),
                    "denominator": None,
                    "period_start": "2026-04-27T00:00:00+03:00",
                    "period_end": "2026-05-04T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                "Analysis period: 2026-04-27 to 2026-05-04",
                "Excludes refund transactions",
                "Unit costs from menu_items.unit_cost_sar",
                "Revenue from line_total_sar (net of discounts)"
            ],
            "assumptions": [
                "Menu unit_cost_sar is current and applicable to analysis period",
                "No recipe/BOM available; using menu-level unit costs",
                "Discount amounts are deducted from revenue"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
# Calculate known waste costs from inventory data
if len(inv_analysis) > 0:
    inv_analysis_clean = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(inv_analysis_clean) > 0:
        total_waste_cost = inv_analysis_clean['known_waste_cost_sar'].sum()
        waste_items = inv_analysis_clean[['item', 'units_wasted', 'known_waste_cost_sar']].copy()
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        if total_waste_cost > 0:
            finding_2 = {
                "title": "Quantified Waste Cost - Analysis Period",
                "claim": f"Total known waste cost for week of 2026-04-27 is {total_waste_cost:.2f} SAR across {len(waste_items)} items with waste observations",
                "finding_type": "waste_cost",
                "metrics": {
                    "total_waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "items_with_waste": {
                        "value": len(waste_items),
                        "unit": "count",
                        "numerator": len(waste_items),
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "highest_waste_item": {
                        "value": waste_items.iloc[0]['item'] if len(waste_items) > 0 else None,
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "highest_waste_cost": {
                        "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                        "unit": "SAR",
                        "numerator": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_items),
                "coverage_notes": [
                    "Analysis period: week starting 2026-04-27",
                    "Only non-null waste_cost_sar values included",
                    "Blank waste values excluded per data quality rules"
                ],
                "assumptions": [
                    "known_waste_cost_sar reflects actual waste cost",
                    "Waste observations are complete for reported items"
                ],
                "confidence": 0.90
            }
            findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Analyze supplier emails for price changes
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Calculate percentage change
        price_changes['price_change_pct'] = (
            ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100)
        ).round(2)
        
        # Sort by absolute percentage change
        price_changes['abs_change_pct'] = price_changes['price_change_pct'].abs()
        price_changes_sorted = price_changes.sort_values('abs_change_pct', ascending=False)
        
        if len(price_changes_sorted) > 0:
            top_change = price_changes_sorted.iloc[0]
            
            finding_3 = {
                "title": "Supplier Price Change - Procurement Impact",
                "claim": f"{top_change['entity_or_ingredient']} price changed by {top_change['price_change_pct']:.1f}% (from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}) effective {top_change['effective_date'].strftime('%Y-%m-%d')}",
                "finding_type": "supplier_price_change",
                "metrics": {
                    "ingredient": {
                        "value": top_change['entity_or_ingredient'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "old_price": {
                        "value": round(top_change['old_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": round(top_change['old_price'], 2),
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "new_price": {
                        "value": round(top_change['new_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": round(top_change['new_price'], 2),
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "price_change_pct": {
                        "value": top_change['price_change_pct'],
                        "unit": "%",
                        "numerator": top_change['price_change_pct'],
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "effective_date": {
                        "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    },
                    "confidence_level": {
                        "value": top_change['confidence'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-27T00:00:00+03:00",
                        "period_end": "2026-05-04T00:00:00+03:00"
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(price_changes),
                "coverage_notes": [
                    "Analysis period: 2026-04-27 to 2026-05-04",
                    "Supplier price changes extracted from email communications",
                    "No recipe/BOM available; cannot calculate per-drink impact",
                    "Standing order quantities not confirmed in available emails"
                ],
                "assumptions": [
                    "Email extraction confidence reflects data quality",
                    "Price change applies to relevant products",
                    "Effective date marks when price change takes effect",
                    "No standing order volume data available for procurement scenario calculation"
                ],
                "confidence": float(top_change['confidence']) if pd.notna(top_change['confidence']) else 0.75
            }
            findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
