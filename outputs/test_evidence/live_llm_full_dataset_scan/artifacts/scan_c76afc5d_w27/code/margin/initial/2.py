import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis periods
analysis_start = "2026-07-13T00:00:00+03:00"
analysis_end = "2026-07-20T00:00:00+03:00"
previous_start = "2026-07-06T00:00:00+03:00"
previous_end = "2026-07-13T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Convert analysis periods to datetime
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start_dt) & (pos_df['timestamp'] < analysis_end_dt)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start_dt) & (pos_df['timestamp'] < previous_end_dt)].copy()

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
    
    # Find top contributor by gross profit
    top_item = item_metrics.loc[item_metrics['gross_profit'].idxmax()]
    
    if not pd.isna(top_item['gross_profit']) and top_item['gross_profit'] > 0:
        finding_1 = {
            "title": "Top Gross Profit Contributor - Item Economics",
            "claim": f"{top_item['item_name']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period, with {top_item['total_quantity']:.0f} units sold across {int(top_item['basket_count'])} transactions at a {top_item['gross_margin_pct']:.1f}% gross margin.",
            "finding_type": "item_economics",
            "metrics": {
                "item_name": {
                    "value": top_item['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_revenue_sar": {
                    "value": round(top_item['total_revenue'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['total_revenue'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_cogs_sar": {
                    "value": round(top_item['total_cogs'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['total_cogs'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "gross_profit_sar": {
                    "value": round(top_item['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "gross_margin_pct": {
                    "value": round(top_item['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": round(top_item['total_revenue'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "units_sold": {
                    "value": round(top_item['total_quantity'], 0),
                    "unit": "units",
                    "numerator": round(top_item['total_quantity'], 0),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "basket_count": {
                    "value": int(top_item['basket_count']),
                    "unit": "transactions",
                    "numerator": int(top_item['basket_count']),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                f"Analysis period: {analysis_start} to {analysis_end}",
                f"POS transactions analyzed: {len(pos_sales)} line items",
                f"Refunds excluded from revenue calculation",
                f"Menu unit costs matched to {len(item_metrics)} unique items"
            ],
            "assumptions": [
                "Unit costs from menu_items.unit_cost_sar are current and accurate",
                "Line totals in POS reflect actual revenue after discounts",
                "No recipe/BOM adjustments applied"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Quantify known waste cost only for non-null waste observations
if len(inventory_df) > 0:
    # Filter inventory for analysis week
    analysis_week_start = pd.to_datetime("2026-07-13")
    inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start].copy()
    
    if len(inventory_analysis) > 0:
        # Calculate total waste cost for items with known waste
        waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & (inventory_analysis['known_waste_cost_sar'] > 0)].copy()
        
        if len(waste_items) > 0:
            total_waste_cost = waste_items['known_waste_cost_sar'].sum()
            waste_item_count = len(waste_items)
            
            # Get top waste item
            top_waste_item = waste_items.loc[waste_items['known_waste_cost_sar'].idxmax()]
            
            finding_2 = {
                "title": "Quantified Waste Cost Impact",
                "claim": f"Known waste cost for the week of {analysis_week_start.strftime('%Y-%m-%d')} totaled {total_waste_cost:.2f} SAR across {waste_item_count} items, with {top_waste_item['item']} accounting for {top_waste_item['known_waste_cost_sar']:.2f} SAR ({top_waste_item['units_wasted']:.0f} units wasted).",
                "finding_type": "waste_cost",
                "metrics": {
                    "total_waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "waste_item_count": {
                        "value": waste_item_count,
                        "unit": "items",
                        "numerator": waste_item_count,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "top_waste_item": {
                        "value": top_waste_item['item'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "top_waste_cost_sar": {
                        "value": round(top_waste_item['known_waste_cost_sar'], 2),
                        "unit": "SAR",
                        "numerator": round(top_waste_item['known_waste_cost_sar'], 2),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "top_waste_units": {
                        "value": round(top_waste_item['units_wasted'], 0),
                        "unit": "units",
                        "numerator": round(top_waste_item['units_wasted'], 0),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_items),
                "coverage_notes": [
                    f"Analysis period: {analysis_start} to {analysis_end}",
                    f"Only non-null waste cost observations included",
                    f"Waste items with cost > 0: {waste_item_count}",
                    f"Total inventory records for period: {len(inventory_analysis)}"
                ],
                "assumptions": [
                    "known_waste_cost_sar values are accurate and complete for reported waste",
                    "Blank waste values are treated as unknown, not zero"
                ],
                "confidence": 0.90
            }
            findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Detect dated supplier price changes from emails
if len(emails_df) > 0:
    # Filter for price change emails with old and new prices
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
        
        # Find most significant price change
        price_changes['abs_change_pct'] = price_changes['price_change_pct'].abs()
        top_change = price_changes.loc[price_changes['abs_change_pct'].idxmax()]
        
        # Extract standing order quantity if available in facts
        standing_qty = None
        facts_value = top_change['facts']
        # Check if facts_value is a scalar (not an array)
        if isinstance(facts_value, str):
            # Look for quantity patterns in facts
            import re
            qty_match = re.search(r'(\d+)\s*(?:units?|bags?|boxes?|cartons?)', facts_value, re.IGNORECASE)
            if qty_match:
                standing_qty = int(qty_match.group(1))
        
        finding_3 = {
            "title": "Supplier Price Change Detection",
            "claim": f"Supplier price change detected for {top_change['entity_or_ingredient']}: {top_change['old_price']:.2f} {top_change['currency']}/{top_change['unit']} → {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']} ({top_change['price_change_pct']:+.1f}%), effective {pd.to_datetime(top_change['effective_date']).strftime('%Y-%m-%d')}.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "entity_or_ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "price_change_pct": {
                    "value": round(top_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "effective_date": {
                    "value": pd.to_datetime(top_change['effective_date']).strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "confidence_score": {
                    "value": round(top_change['confidence'], 2),
                    "unit": None,
                    "numerator": round(top_change['confidence'], 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Price change emails analyzed: {len(price_changes)}",
                f"Extraction confidence: {round(top_change['confidence'], 2)}",
                f"Email date: {pd.to_datetime(top_change['date']).strftime('%Y-%m-%d')}",
                f"Effective date: {pd.to_datetime(top_change['effective_date']).strftime('%Y-%m-%d')}"
            ],
            "assumptions": [
                "Email extraction confidence score reflects accuracy of price data",
                "No recipe/BOM exists to calculate per-drink impact",
                "Standing order quantity and payment terms are not confirmed in email data"
            ],
            "confidence": 0.85
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
