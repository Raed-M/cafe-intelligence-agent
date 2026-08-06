import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Parse timestamps
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Define analysis period
analysis_start = pd.to_datetime('2026-05-25T00:00:00+03:00')
analysis_end = pd.to_datetime('2026-06-01T00:00:00+03:00')
previous_start = pd.to_datetime('2026-05-18T00:00:00+03:00')
previous_end = pd.to_datetime('2026-05-25T00:00:00+03:00')

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis period
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-05-25')]
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-05-18')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
if len(pos_analysis) > 0 and len(menu_df) > 0:
    # Merge POS with menu to get unit costs
    pos_with_cost = pos_analysis.merge(
        menu_df[['sku', 'unit_cost_sar', 'price_sar']],
        on='sku',
        how='left'
    )
    
    # Filter out refunds for revenue calculation
    pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()
    
    # Calculate metrics per item
    item_metrics = []
    for sku in pos_sales['sku'].unique():
        sku_data = pos_sales[pos_sales['sku'] == sku]
        item_name = sku_data['item_name'].iloc[0] if len(sku_data) > 0 else 'Unknown'
        
        total_quantity = sku_data['quantity'].sum()
        total_revenue = sku_data['line_total_sar'].sum()
        
        # Get unit cost from menu
        unit_cost = menu_df[menu_df['sku'] == sku]['unit_cost_sar'].values
        if len(unit_cost) > 0 and pd.notna(unit_cost[0]):
            unit_cost = unit_cost[0]
            total_cogs = total_quantity * unit_cost
            gross_profit = total_revenue - total_cogs
            margin_rate = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            item_metrics.append({
                'sku': sku,
                'item_name': item_name,
                'quantity': total_quantity,
                'revenue': total_revenue,
                'unit_cost': unit_cost,
                'total_cogs': total_cogs,
                'gross_profit': gross_profit,
                'margin_rate': margin_rate
            })
    
    if len(item_metrics) > 0:
        item_df = pd.DataFrame(item_metrics)
        
        # Find top 3 by gross profit contribution
        top_items = item_df.nlargest(3, 'gross_profit')
        
        if len(top_items) > 0:
            top_item = top_items.iloc[0]
            
            finding_1 = {
                "title": "Top Gross Profit Item - Analysis Period",
                "claim": f"Item {top_item['item_name']} (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['quantity']:.0f} units sold at {top_item['margin_rate']:.1f}% margin rate during the analysis period.",
                "finding_type": "item_economics",
                "metrics": {
                    "item_name": {
                        "value": top_item['item_name'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "sku": {
                        "value": top_item['sku'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "quantity_sold": {
                        "value": top_item['quantity'],
                        "unit": "units",
                        "numerator": top_item['quantity'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "total_revenue": {
                        "value": top_item['revenue'],
                        "unit": "SAR",
                        "numerator": top_item['revenue'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "unit_cost_sar": {
                        "value": top_item['unit_cost'],
                        "unit": "SAR",
                        "numerator": top_item['unit_cost'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "total_cogs": {
                        "value": top_item['total_cogs'],
                        "unit": "SAR",
                        "numerator": top_item['total_cogs'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "gross_profit": {
                        "value": top_item['gross_profit'],
                        "unit": "SAR",
                        "numerator": top_item['gross_profit'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "margin_rate_percent": {
                        "value": top_item['margin_rate'],
                        "unit": "%",
                        "numerator": top_item['margin_rate'],
                        "denominator": 100,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": len(pos_sales),
                "coverage_notes": [
                    f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                    f"Total POS transactions analyzed: {len(pos_sales)}",
                    f"Unique items in analysis: {len(item_df)}",
                    "Refunds excluded from revenue calculation",
                    "Unit costs sourced from menu.parquet"
                ],
                "assumptions": [
                    "Menu unit_cost_sar values are current and accurate for the analysis period",
                    "POS line_total_sar reflects actual revenue after discounts",
                    "No recipe/BOM available; item-level economics based on declared menu costs"
                ],
                "confidence": 0.95
            }
            findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate known waste costs from inventory
if len(inv_analysis) > 0:
    inv_analysis_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(inv_analysis_with_waste) > 0:
        total_waste_cost = inv_analysis_with_waste['known_waste_cost_sar'].sum()
        waste_items = inv_analysis_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        if len(waste_items) > 0:
            top_waste = waste_items.iloc[0]
            
            finding_2 = {
                "title": "Highest Waste Cost Item - Analysis Week",
                "claim": f"Item {top_waste['item']} (SKU: {top_waste['sku']}) incurred the highest known waste cost of {top_waste['known_waste_cost_sar']:.2f} SAR with {top_waste['units_wasted']:.0f} units wasted during the analysis week.",
                "finding_type": "waste_cost",
                "metrics": {
                    "item_name": {
                        "value": top_waste['item'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "sku": {
                        "value": top_waste['sku'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "units_wasted": {
                        "value": top_waste['units_wasted'],
                        "unit": "units",
                        "numerator": top_waste['units_wasted'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "known_waste_cost_sar": {
                        "value": top_waste['known_waste_cost_sar'],
                        "unit": "SAR",
                        "numerator": top_waste['known_waste_cost_sar'],
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "total_waste_cost_all_items": {
                        "value": total_waste_cost,
                        "unit": "SAR",
                        "numerator": total_waste_cost,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inv_analysis_with_waste),
                "coverage_notes": [
                    f"Analysis week: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                    f"Items with known waste cost: {len(inv_analysis_with_waste)}",
                    "Only non-null waste_cost_sar values included",
                    "Blank waste values excluded per data quality rules"
                ],
                "assumptions": [
                    "known_waste_cost_sar values are accurate and complete for reported waste",
                    "Waste cost calculation based on unit_cost_sar from inventory records"
                ],
                "confidence": 0.90
            }
            findings.append(finding_2)

# FINDING 3: Supplier Price Change Impact Analysis
# Analyze supplier emails for price changes
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Calculate percentage change
        price_changes['percentage_change'] = (
            ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100)
        )
        
        # Sort by absolute percentage change
        price_changes['abs_pct_change'] = price_changes['percentage_change'].abs()
        price_changes = price_changes.sort_values('abs_pct_change', ascending=False)
        
        top_change = price_changes.iloc[0]
        
        # Extract facts about standing orders if available
        facts_text = str(top_change['facts']) if pd.notna(top_change['facts']) else ""
        
        finding_3 = {
            "title": "Significant Supplier Price Change Detected",
            "claim": f"Supplier {top_change['sender']} announced a price change for {top_change['entity_or_ingredient']} from {top_change['old_price']:.2f} {top_change['currency']}/{top_change['unit']} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].isoformat()}, representing a {top_change['percentage_change']:.1f}% change.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "supplier": {
                    "value": top_change['sender'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "old_price": {
                    "value": top_change['old_price'],
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": top_change['old_price'],
                    "denominator": None,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "new_price": {
                    "value": top_change['new_price'],
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": top_change['new_price'],
                    "denominator": None,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "percentage_change": {
                    "value": top_change['percentage_change'],
                    "unit": "%",
                    "numerator": top_change['percentage_change'],
                    "denominator": 100,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "effective_date": {
                    "value": top_change['effective_date'].isoformat(),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_change['effective_date'].isoformat(),
                    "period_end": top_change['effective_date'].isoformat()
                },
                "email_date": {
                    "value": top_change['date'].isoformat() if pd.notna(top_change['date']) else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": top_change['date'].isoformat() if pd.notna(top_change['date']) else None,
                    "period_end": top_change['date'].isoformat() if pd.notna(top_change['date']) else None
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Total supplier price changes detected: {len(price_changes)}",
                f"Email extraction confidence: {top_change['confidence']}",
                f"Extraction mode: {top_change['extraction_mode']}",
                "Price changes with both old and new prices and effective dates included"
            ],
            "assumptions": [
                "Email extraction accuracy as indicated by confidence score",
                "Effective date represents when price change takes effect",
                "No recipe/BOM available; impact on menu items cannot be calculated without standing order volumes",
                "Standing order quantities and payment terms not confirmed from email facts"
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
