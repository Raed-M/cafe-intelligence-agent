import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

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

# Define analysis period
analysis_start = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 29, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter for analysis period
analysis_pos = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
previous_pos = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_with_cost = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar']].copy()
menu_with_cost = menu_with_cost.dropna(subset=['unit_cost_sar'])

# Merge with analysis period POS
analysis_pos_merged = analysis_pos.merge(menu_with_cost, on='sku', how='inner')

# Calculate metrics for each item
item_economics = []
for sku in analysis_pos_merged['sku'].unique():
    sku_data = analysis_pos_merged[analysis_pos_merged['sku'] == sku]
    
    # Filter out refunds for revenue calculation
    non_refund_data = sku_data[~sku_data['is_refund']]
    
    total_quantity = non_refund_data['quantity'].sum()
    total_revenue = non_refund_data['line_total_sar'].sum()
    unit_cost = non_refund_data['unit_cost_sar'].iloc[0] if len(non_refund_data) > 0 else 0
    
    total_cogs = total_quantity * unit_cost
    gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        margin_rate = (gross_profit / total_revenue) * 100
    else:
        margin_rate = 0
    
    item_name = non_refund_data['item_name_en'].iloc[0] if len(non_refund_data) > 0 else sku
    
    item_economics.append({
        'sku': sku,
        'item_name': item_name,
        'quantity': total_quantity,
        'revenue': total_revenue,
        'unit_cost': unit_cost,
        'cogs': total_cogs,
        'gross_profit': gross_profit,
        'margin_rate': margin_rate
    })

item_econ_df = pd.DataFrame(item_economics)

# Find top 3 items by gross profit
top_items = item_econ_df.nlargest(3, 'gross_profit')

if len(top_items) > 0:
    finding1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Week)",
        "claim": f"During the analysis week (2026-06-22 to 2026-06-29), the top 3 items by gross profit contribution are: {', '.join(top_items['item_name'].tolist())}. Total gross profit from these items: {top_items['gross_profit'].sum():.2f} SAR.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_1_gross_profit": {
                "value": round(top_items.iloc[0]['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['revenue'], 2),
                "denominator": round(top_items.iloc[0]['cogs'], 2),
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_1_margin_rate": {
                "value": round(top_items.iloc[0]['margin_rate'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                "denominator": round(top_items.iloc[0]['revenue'], 2),
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_name'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_2_gross_profit": {
                "value": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['revenue'], 2) if len(top_items) > 1 else None,
                "denominator": round(top_items.iloc[1]['cogs'], 2) if len(top_items) > 1 else None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_3_name": {
                "value": top_items.iloc[2]['item_name'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_item_3_gross_profit": {
                "value": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['revenue'], 2) if len(top_items) > 2 else None,
                "denominator": round(top_items.iloc[2]['cogs'], 2) if len(top_items) > 2 else None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "total_top_3_gross_profit": {
                "value": round(top_items['gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['revenue'].sum(), 2),
                "denominator": round(top_items['cogs'].sum(), 2),
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_pos_merged),
        "coverage_notes": [
            "Analysis includes only POS transactions with matching menu items containing unit cost data",
            "Refunds excluded from revenue calculations",
            "Period: 2026-06-22 to 2026-06-29"
        ],
        "assumptions": [
            "Unit cost from menu applies uniformly across all sales in analysis period",
            "No recipe/BOM data available; per-unit COGS is menu unit_cost_sar",
            "Line totals are accurate and consistent"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Analysis
# Calculate waste costs from inventory data for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
analysis_week_inventory = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-06-22', tz='UTC')]

waste_analysis = []
total_waste_cost = 0
total_waste_units = 0

for _, row in analysis_week_inventory.iterrows():
    if pd.notna(row['known_waste_cost_sar']) and row['known_waste_cost_sar'] > 0:
        waste_analysis.append({
            'sku': row['sku'],
            'item': row['item'],
            'units_wasted': row['units_wasted'],
            'waste_cost': row['known_waste_cost_sar']
        })
        total_waste_cost += row['known_waste_cost_sar']
        total_waste_units += row['units_wasted'] if pd.notna(row['units_wasted']) else 0

if total_waste_cost > 0:
    waste_df = pd.DataFrame(waste_analysis)
    top_waste_items = waste_df.nlargest(3, 'waste_cost')
    
    finding2 = {
        "title": "Quantified Waste Cost (Analysis Week)",
        "claim": f"During the analysis week (2026-06-22 to 2026-06-29), quantified waste cost totaled {total_waste_cost:.2f} SAR across {len(waste_df)} items with known waste observations. Top waste contributor: {top_waste_items.iloc[0]['item']} ({top_waste_items.iloc[0]['waste_cost']:.2f} SAR).",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "items_with_waste": {
                "value": len(waste_df),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_waste_item_name": {
                "value": top_waste_items.iloc[0]['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            },
            "top_waste_item_cost": {
                "value": round(top_waste_items.iloc[0]['waste_cost'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+00:00",
                "period_end": "2026-06-29T00:00:00+00:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_df),
        "coverage_notes": [
            "Only non-null known_waste_cost_sar values included",
            "Blank waste observations excluded per methodology",
            "Period: week starting 2026-06-22"
        ],
        "assumptions": [
            "known_waste_cost_sar represents actual quantified waste cost",
            "Waste cost is independent of sales volume"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# FINDING 3: Supplier Price Changes and Margin Impact
# Extract supplier price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'], utc=True)
    
    # Filter for changes effective during or before analysis period
    analysis_period_end = pd.Timestamp('2026-06-29', tz='UTC')
    relevant_changes = price_changes[price_changes['effective_date'] <= analysis_period_end]
    
    if len(relevant_changes) > 0:
        # Calculate percentage changes
        relevant_changes['price_change_pct'] = (
            (relevant_changes['new_price'] - relevant_changes['old_price']) / 
            relevant_changes['old_price'] * 100
        )
        
        # Sort by absolute percentage change
        relevant_changes['abs_change_pct'] = relevant_changes['price_change_pct'].abs()
        top_changes = relevant_changes.nlargest(3, 'abs_change_pct')
        
        if len(top_changes) > 0:
            finding3 = {
                "title": "Supplier Price Changes and Margin Pressure",
                "claim": f"Supplier price changes detected for {len(relevant_changes)} ingredients/items effective through analysis week. Largest change: {top_changes.iloc[0]['entity_or_ingredient']} ({top_changes.iloc[0]['old_price']:.2f} → {top_changes.iloc[0]['new_price']:.2f} {top_changes.iloc[0]['currency']}, {top_changes.iloc[0]['price_change_pct']:+.1f}% effective {top_changes.iloc[0]['effective_date'].strftime('%Y-%m-%d')}). Impact on menu items requires recipe/BOM data to quantify.",
                "finding_type": "supplier_price_change",
                "metrics": {
                    "total_price_changes": {
                        "value": len(relevant_changes),
                        "unit": "count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    },
                    "largest_change_ingredient": {
                        "value": top_changes.iloc[0]['entity_or_ingredient'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    },
                    "largest_change_old_price": {
                        "value": round(top_changes.iloc[0]['old_price'], 2),
                        "unit": top_changes.iloc[0]['currency'],
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    },
                    "largest_change_new_price": {
                        "value": round(top_changes.iloc[0]['new_price'], 2),
                        "unit": top_changes.iloc[0]['currency'],
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    },
                    "largest_change_pct": {
                        "value": round(top_changes.iloc[0]['price_change_pct'], 2),
                        "unit": "%",
                        "numerator": round(top_changes.iloc[0]['new_price'] - top_changes.iloc[0]['old_price'], 2),
                        "denominator": round(top_changes.iloc[0]['old_price'], 2),
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    },
                    "largest_change_effective_date": {
                        "value": top_changes.iloc[0]['effective_date'].strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-22T00:00:00+00:00",
                        "period_end": "2026-06-29T00:00:00+00:00"
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(relevant_changes),
                "coverage_notes": [
                    "Price changes extracted from supplier emails",
                    "Only changes with old_price, new_price, and effective_date included",
                    "Effective through analysis week (2026-06-29)"
                ],
                "assumptions": [
                    "Email extraction confidence scores reflect reliability of price data",
                    "No recipe/BOM available; per-drink impact cannot be calculated",
                    "Standing order quantities and payment terms not confirmed in data"
                ],
                "confidence": 0.85
            }
            findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
