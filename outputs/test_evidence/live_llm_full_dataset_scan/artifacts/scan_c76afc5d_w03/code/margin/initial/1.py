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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 2, 2, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 1, 19, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_analysis = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar', 'category']].copy()

# Merge with POS data for analysis period
pos_with_menu = pos_analysis.merge(menu_analysis, on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[pos_with_menu['is_refund'] == False].copy()

# Calculate metrics by item
item_metrics = []
for sku in pos_sales['sku'].unique():
    sku_data = pos_sales[pos_sales['sku'] == sku]
    
    if len(sku_data) == 0:
        continue
    
    item_name = sku_data['item_en'].iloc[0]
    unit_price = sku_data['unit_price_sar'].iloc[0]
    unit_cost = sku_data['unit_cost_sar'].iloc[0]
    
    # Skip if unit_cost is null
    if pd.isna(unit_cost):
        continue
    
    total_quantity = sku_data['quantity'].sum()
    total_revenue = sku_data['line_total_sar'].sum()
    total_cogs = total_quantity * unit_cost
    gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        gross_margin_pct = (gross_profit / total_revenue) * 100
    else:
        gross_margin_pct = 0
    
    item_metrics.append({
        'sku': sku,
        'item_name': item_name,
        'quantity': total_quantity,
        'revenue': total_revenue,
        'unit_cost': unit_cost,
        'cogs': total_cogs,
        'gross_profit': gross_profit,
        'gross_margin_pct': gross_margin_pct
    })

if item_metrics:
    item_df = pd.DataFrame(item_metrics)
    
    # Find top 3 items by gross profit
    top_items = item_df.nlargest(3, 'gross_profit')
    
    if len(top_items) > 0:
        top_item = top_items.iloc[0]
        
        finding_1 = {
            "title": "Top Gross Profit Item - Exact Item Economics",
            "claim": f"Item '{top_item['item_name']}' (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period, with {top_item['quantity']:.0f} units sold at {top_item['gross_margin_pct']:.1f}% gross margin.",
            "finding_type": "item_economics",
            "metrics": {
                "gross_profit_sar": {
                    "value": round(top_item['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['revenue'], 2),
                    "denominator": round(top_item['cogs'], 2),
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "units_sold": {
                    "value": int(top_item['quantity']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "gross_margin_percent": {
                    "value": round(top_item['gross_margin_pct'], 1),
                    "unit": "%",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": round(top_item['revenue'], 2),
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                "Analysis includes all non-refund POS transactions during 2026-01-26 to 2026-02-02",
                "Only items with non-null unit_cost_sar in menu are included",
                "Gross profit = Revenue - (Quantity × Unit Cost)"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "POS line_total_sar is accurate revenue after discounts",
                "No waste adjustments applied (waste data not linked to POS items)"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate known waste costs from inventory data
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-01-26'].copy()

if len(inventory_analysis) > 0:
    # Filter for non-null waste costs
    waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_units_wasted = waste_items['units_wasted'].sum()
        
        # Get corresponding revenue for waste items
        waste_skus = waste_items['sku'].unique()
        waste_revenue = pos_analysis[pos_analysis['sku'].isin(waste_skus) & (pos_analysis['is_refund'] == False)]['line_total_sar'].sum()
        
        if waste_revenue > 0:
            waste_impact_pct = (total_waste_cost / waste_revenue) * 100
        else:
            waste_impact_pct = 0
        
        finding_2 = {
            "title": "Quantified Waste Cost Impact",
            "claim": f"Known waste cost of {total_waste_cost:.2f} SAR was recorded for {int(total_units_wasted)} units during the week of 2026-01-26, representing {waste_impact_pct:.2f}% of revenue from affected items.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                },
                "waste_as_pct_of_revenue": {
                    "value": round(waste_impact_pct, 2),
                    "unit": "%",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": round(waste_revenue, 2),
                    "period_start": "2026-01-26T00:00:00+03:00",
                    "period_end": "2026-02-02T00:00:00+03:00"
                }
            },
            "source_names": ["inventory", "pos"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Only non-null known_waste_cost_sar values included",
                "Waste data from inventory week 2026-01-26",
                "Revenue calculated from POS for affected SKUs"
            ],
            "assumptions": [
                "known_waste_cost_sar accurately reflects waste value",
                "Waste items correspond to POS SKUs"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Change Impact Analysis
# Look for price changes in emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Get the most recent price change
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    recent_change = price_changes.sort_values('effective_date', ascending=False).iloc[0]
    
    old_price = float(recent_change['old_price'])
    new_price = float(recent_change['new_price'])
    price_delta = new_price - old_price
    price_delta_pct = (price_delta / old_price) * 100 if old_price != 0 else 0
    
    entity = recent_change['entity_or_ingredient']
    unit = recent_change['unit']
    effective_date = recent_change['effective_date']
    
    finding_3 = {
        "title": "Supplier Price Change Detection",
        "claim": f"Supplier price change detected for {entity}: {old_price:.2f} {recent_change['currency']}/{unit} → {new_price:.2f} {recent_change['currency']}/{unit} ({price_delta_pct:+.1f}%), effective {effective_date.strftime('%Y-%m-%d')}.",
        "finding_type": "supplier_cost",
        "metrics": {
            "old_price": {
                "value": round(old_price, 2),
                "unit": f"{recent_change['currency']}/{unit}",
                "numerator": None,
                "denominator": None,
                "period_start": None,
                "period_end": None
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": f"{recent_change['currency']}/{unit}",
                "numerator": None,
                "denominator": None,
                "period_start": None,
                "period_end": None
            },
            "price_change_percent": {
                "value": round(price_delta_pct, 1),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": round(old_price, 2),
                "period_start": None,
                "period_end": None
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Most recent price change from supplier emails",
            "No standing order quantities found to calculate procurement cost scenario",
            "Price change may not yet be reflected in current menu costs"
        ],
        "assumptions": [
            "Email extraction accurately captured price and effective date",
            "Price change applies to future purchases",
            "No recipe/BOM available to calculate per-drink impact"
        ],
        "confidence": 0.75
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
