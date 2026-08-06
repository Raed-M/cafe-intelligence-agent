import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load environment configuration
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
item_economics = {}

for _, menu_row in menu_df.iterrows():
    sku = menu_row['sku']
    unit_cost = menu_row['unit_cost_sar']
    menu_price = menu_row['price_sar']
    
    # Get POS data for this SKU in analysis period
    sku_pos = pos_analysis[pos_analysis['sku'] == sku].copy()
    
    if len(sku_pos) > 0:
        # Calculate totals (excluding refunds from net calculation)
        non_refund_pos = sku_pos[~sku_pos['is_refund']]
        
        total_quantity = non_refund_pos['quantity'].sum()
        total_revenue = non_refund_pos['line_total_sar'].sum()
        
        if total_quantity > 0:
            total_cogs = total_quantity * unit_cost
            total_gross_profit = total_revenue - total_cogs
            gross_margin_pct = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            item_economics[sku] = {
                'item_name': menu_row['item_en'],
                'quantity_sold': total_quantity,
                'revenue': total_revenue,
                'unit_cost': unit_cost,
                'total_cogs': total_cogs,
                'gross_profit': total_gross_profit,
                'gross_margin_pct': gross_margin_pct,
                'menu_price': menu_price,
                'transactions': sku_pos['transaction_id'].nunique()
            }

# Find top 3 items by gross profit contribution
if item_economics:
    sorted_items = sorted(item_economics.items(), key=lambda x: x[1]['gross_profit'], reverse=True)
    top_items = sorted_items[:3]
    
    total_cafe_revenue = pos_analysis[~pos_analysis['is_refund']]['line_total_sar'].sum()
    total_cafe_cogs = sum(item['total_cogs'] for item in item_economics.values())
    total_cafe_profit = total_cafe_revenue - total_cafe_cogs
    
    if len(top_items) > 0:
        top_item_sku, top_item_data = top_items[0]
        
        finding_1 = {
            "title": "Top Gross Profit Item: Item-Level Economics",
            "claim": f"The highest-margin item by absolute gross profit contribution is {top_item_data['item_name']} (SKU: {top_item_sku}), generating SAR {top_item_data['gross_profit']:.2f} gross profit from {int(top_item_data['quantity_sold'])} units sold at SAR {top_item_data['menu_price']:.2f} per unit, with unit cost of SAR {top_item_data['unit_cost']:.2f}, yielding {top_item_data['gross_margin_pct']:.1f}% gross margin.",
            "finding_type": "item_economics",
            "metrics": {
                "item_name": {
                    "value": top_item_data['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "quantity_sold": {
                    "value": int(top_item_data['quantity_sold']),
                    "unit": "units",
                    "numerator": int(top_item_data['quantity_sold']),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "revenue": {
                    "value": round(top_item_data['revenue'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item_data['revenue'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "unit_cost_sar": {
                    "value": round(top_item_data['unit_cost'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item_data['unit_cost'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_cogs": {
                    "value": round(top_item_data['total_cogs'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item_data['total_cogs'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "gross_profit": {
                    "value": round(top_item_data['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item_data['gross_profit'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "gross_margin_pct": {
                    "value": round(top_item_data['gross_margin_pct'], 1),
                    "unit": "%",
                    "numerator": round(top_item_data['gross_margin_pct'], 1),
                    "denominator": 100,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "transactions": {
                    "value": int(top_item_data['transactions']),
                    "unit": "baskets",
                    "numerator": int(top_item_data['transactions']),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(top_item_data['transactions']),
            "coverage_notes": [
                f"Analysis period: 2026-03-09 to 2026-03-16 (7 days)",
                f"POS data: {len(pos_analysis)} line items, {pos_analysis['transaction_id'].nunique()} transactions",
                f"Menu data: {len(menu_df)} items with unit costs",
                "Refunds excluded from net revenue and profit calculations",
                "Unit costs sourced from menu.unit_cost_sar (declared supplier costs)"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "POS line_total_sar is accurate and consistent",
                "No recipe/BOM data available; per-drink ingredient costs not calculated",
                "Waste costs not included in item-level COGS (only in inventory waste tracking)"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Impact
# Calculate known waste costs from inventory data
waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_analysis) > 0:
    total_waste_cost = waste_analysis['known_waste_cost_sar'].sum()
    total_waste_units = waste_analysis['units_wasted'].sum()
    
    # Get corresponding revenue for waste items
    waste_skus = waste_analysis['sku'].unique()
    waste_revenue = pos_analysis[pos_analysis['sku'].isin(waste_skus)][~pos_analysis['is_refund']]['line_total_sar'].sum()
    
    if waste_revenue > 0:
        waste_as_pct_revenue = (total_waste_cost / waste_revenue) * 100
        
        finding_2 = {
            "title": "Quantified Waste Cost Impact",
            "claim": f"Known waste cost in analysis period totals SAR {total_waste_cost:.2f} across {int(total_waste_units)} units, representing {waste_as_pct_revenue:.2f}% of revenue from affected items. This is a direct margin pressure from non-saleable inventory.",
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
                    "value": int(total_waste_units),
                    "unit": "units",
                    "numerator": int(total_waste_units),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_as_pct_revenue": {
                    "value": round(waste_as_pct_revenue, 2),
                    "unit": "%",
                    "numerator": round(waste_as_pct_revenue, 2),
                    "denominator": 100,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "affected_items": {
                    "value": len(waste_skus),
                    "unit": "SKUs",
                    "numerator": len(waste_skus),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory", "pos"],
            "sample_size": len(waste_analysis),
            "coverage_notes": [
                "Only non-null waste_cost_sar values included (known waste only)",
                f"Inventory records: {len(waste_analysis)} items with waste cost data",
                "Blank waste values treated as unknown, not zero",
                "Waste cost sourced from inventory.known_waste_cost_sar"
            ],
            "assumptions": [
                "Inventory waste_cost_sar accurately reflects disposal/loss value",
                "Waste occurred during analysis period (week_starting date used as proxy)",
                "No recipe/BOM data; cannot allocate waste to specific drink types"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Change Analysis
# Check for supplier price changes in emails with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Focus on price changes with clear entity/ingredient
    price_changes = price_changes[price_changes['entity_or_ingredient'].notna()]
    
    if len(price_changes) > 0:
        # Get the most recent price change
        price_changes = price_changes.sort_values('effective_date', ascending=False)
        latest_change = price_changes.iloc[0]
        
        old_price = latest_change['old_price']
        new_price = latest_change['new_price']
        ingredient = latest_change['entity_or_ingredient']
        unit = latest_change['unit'] if pd.notna(latest_change['unit']) else 'unit'
        effective_date = latest_change['effective_date']
        sender = latest_change['sender']
        
        # Calculate percentage change
        if old_price > 0:
            pct_change = ((new_price - old_price) / old_price) * 100
            
            # Check if effective date is within or after analysis period
            is_future = effective_date >= analysis_end
            temporal_note = "announced for future implementation" if is_future else "effective during analysis period"
            
            finding_3 = {
                "title": f"Supplier Price Change: {ingredient}",
                "claim": f"Email from {sender} announces {ingredient} price change from SAR {old_price:.2f} to SAR {new_price:.2f} per {unit} ({pct_change:+.2f}%), {temporal_note} on {effective_date.strftime('%Y-%m-%d')}. Email receipt status: unknown. Impact on cafe margins cannot be quantified without standing order volume data and recipe/BOM information.",
                "finding_type": "supplier_price_change",
                "metrics": {
                    "ingredient": {
                        "value": ingredient,
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "old_price": {
                        "value": round(old_price, 2),
                        "unit": f"SAR/{unit}",
                        "numerator": round(old_price, 2),
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "new_price": {
                        "value": round(new_price, 2),
                        "unit": f"SAR/{unit}",
                        "numerator": round(new_price, 2),
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "price_change_pct": {
                        "value": round(pct_change, 2),
                        "unit": "%",
                        "numerator": round(pct_change, 2),
                        "denominator": 100,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "effective_date": {
                        "value": effective_date.strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    }
                },
                "source_names": ["emails"],
                "sample_size": 1,
                "coverage_notes": [
                    "Email source: supplier communication",
                    "Supplier identity and standing order applicability not confirmed in structured data",
                    "No recipe/BOM data available to calculate per-drink cost impact",
                    "Standing order volume not available in email or structured data",
                    f"Effective date {effective_date.strftime('%Y-%m-%d')} is {'after' if is_future else 'within'} analysis period {analysis_start.strftime('%Y-%m-%d')} to {analysis_end.strftime('%Y-%m-%d')}"
                ],
                "assumptions": [
                    "Email sender is authoritative supplier (not independently verified)",
                    "Price change applies to cafe's standing orders (not confirmed)",
                    "Standing order volume remains constant (actual volume unknown)",
                    "No recipe/BOM data; cannot calculate per-drink margin impact",
                    "Email receipt and acknowledgment by cafe not confirmed"
                ],
                "confidence": 0.4
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
