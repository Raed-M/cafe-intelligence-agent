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
analysis_start = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-03-16', tz=timezone.utc)
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu
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
    item_metrics = []
    for sku in pos_sales['sku'].unique():
        sku_data = pos_sales[pos_sales['sku'] == sku]
        item_name = sku_data['item_name_en'].iloc[0] if 'item_name_en' in sku_data.columns else sku_data['item_name'].iloc[0]
        
        total_quantity = sku_data['quantity'].sum()
        total_revenue = sku_data['line_total_sar'].sum()
        
        # Get unit cost from menu
        unit_cost = menu_df[menu_df['sku'] == sku]['unit_cost_sar'].values
        if len(unit_cost) > 0 and pd.notna(unit_cost[0]):
            unit_cost = unit_cost[0]
            total_cogs = total_quantity * unit_cost
            gross_profit = total_revenue - total_cogs
            gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            item_metrics.append({
                'sku': sku,
                'item_name': item_name,
                'quantity': total_quantity,
                'revenue': total_revenue,
                'unit_cost': unit_cost,
                'total_cogs': total_cogs,
                'gross_profit': gross_profit,
                'gross_margin_pct': gross_margin
            })
    
    if item_metrics:
        # Sort by gross profit
        item_metrics_sorted = sorted(item_metrics, key=lambda x: x['gross_profit'], reverse=True)
        
        # Top 5 items by gross profit
        top_items = item_metrics_sorted[:5]
        total_gp = sum([item['gross_profit'] for item in top_items])
        total_qty = sum([item['quantity'] for item in top_items])
        
        finding_1 = {
            "title": "Top 5 Items by Gross Profit (Week of 2026-03-16)",
            "claim": f"The top 5 items by gross profit generated SAR {total_gp:.2f} in gross profit from {int(total_qty)} units sold during the analysis week.",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items[0]['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+00:00",
                    "period_end": "2026-03-23T00:00:00+00:00"
                },
                "top_item_1_gross_profit": {
                    "value": round(top_items[0]['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_items[0]['gross_profit'], 2),
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+00:00",
                    "period_end": "2026-03-23T00:00:00+00:00"
                },
                "top_item_1_quantity": {
                    "value": int(top_items[0]['quantity']),
                    "unit": "units",
                    "numerator": int(top_items[0]['quantity']),
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+00:00",
                    "period_end": "2026-03-23T00:00:00+00:00"
                },
                "top_item_1_margin_pct": {
                    "value": round(top_items[0]['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_items[0]['gross_profit'], 2),
                    "denominator": round(top_items[0]['revenue'], 2),
                    "period_start": "2026-03-16T00:00:00+00:00",
                    "period_end": "2026-03-23T00:00:00+00:00"
                },
                "top_5_total_gross_profit": {
                    "value": round(total_gp, 2),
                    "unit": "SAR",
                    "numerator": round(total_gp, 2),
                    "denominator": None,
                    "period_start": "2026-03-16T00:00:00+00:00",
                    "period_end": "2026-03-23T00:00:00+00:00"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(total_qty),
            "coverage_notes": [
                "Analysis covers POS transactions from 2026-03-16 to 2026-03-23",
                "Only items with menu unit costs are included",
                "Refunds excluded from revenue calculation"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "POS line_total_sar is accurate revenue after discounts"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
if len(inventory_analysis) > 0:
    # Calculate waste costs
    waste_data = inventory_analysis[inventory_analysis['units_wasted'].notna() & (inventory_analysis['units_wasted'] > 0)].copy()
    
    if len(waste_data) > 0:
        total_waste_units = waste_data['units_wasted'].sum()
        total_waste_cost = waste_data['known_waste_cost_sar'].sum()
        
        waste_items = []
        for idx, row in waste_data.iterrows():
            if pd.notna(row['known_waste_cost_sar']) and row['known_waste_cost_sar'] > 0:
                waste_items.append({
                    'sku': row['sku'],
                    'item': row['item'],
                    'units_wasted': row['units_wasted'],
                    'waste_cost': row['known_waste_cost_sar'],
                    'unit_cost': row['unit_cost_sar']
                })
        
        if waste_items:
            waste_items_sorted = sorted(waste_items, key=lambda x: x['waste_cost'], reverse=True)
            
            finding_2 = {
                "title": "Waste Cost Impact (Week of 2026-03-16)",
                "claim": f"Quantified waste cost for the week totaled SAR {total_waste_cost:.2f} from {int(total_waste_units)} units wasted across {len(waste_items)} items.",
                "finding_type": "waste_economics",
                "metrics": {
                    "total_waste_units": {
                        "value": int(total_waste_units),
                        "unit": "units",
                        "numerator": int(total_waste_units),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "total_waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "items_with_waste": {
                        "value": len(waste_items),
                        "unit": "count",
                        "numerator": len(waste_items),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "highest_waste_item": {
                        "value": waste_items_sorted[0]['item'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "highest_waste_cost": {
                        "value": round(waste_items_sorted[0]['waste_cost'], 2),
                        "unit": "SAR",
                        "numerator": round(waste_items_sorted[0]['waste_cost'], 2),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_items),
                "coverage_notes": [
                    "Only non-null waste observations included",
                    "Waste cost calculated from known_waste_cost_sar field",
                    "Analysis covers inventory week starting 2026-03-16"
                ],
                "assumptions": [
                    "known_waste_cost_sar represents actual cost of wasted units",
                    "Waste data is complete for the inventory week"
                ],
                "confidence": 0.90
            }
            findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Impact
if len(emails_df) > 0:
    # Filter for price change emails with old and new prices
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Convert dates
        price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
        
        # Filter for changes effective during or before analysis period
        price_changes = price_changes[price_changes['effective_date'] <= analysis_end]
        
        if len(price_changes) > 0:
            # Calculate price deltas
            price_changes['price_delta'] = price_changes['new_price'] - price_changes['old_price']
            price_changes['pct_change'] = (price_changes['price_delta'] / price_changes['old_price'] * 100)
            
            # Sort by absolute price delta
            price_changes_sorted = price_changes.sort_values('price_delta', ascending=False, key=abs)
            
            top_change = price_changes_sorted.iloc[0]
            
            finding_3 = {
                "title": "Supplier Price Changes Detected",
                "claim": f"Supplier price change for {top_change['entity_or_ingredient']}: {top_change['old_price']} → {top_change['new_price']} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].strftime('%Y-%m-%d')}, representing a {top_change['pct_change']:.1f}% change.",
                "finding_type": "supplier_pricing",
                "metrics": {
                    "ingredient": {
                        "value": str(top_change['entity_or_ingredient']),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "old_price": {
                        "value": float(top_change['old_price']),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": float(top_change['old_price']),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "new_price": {
                        "value": float(top_change['new_price']),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": float(top_change['new_price']),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "price_delta": {
                        "value": round(float(top_change['price_delta']), 4),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": round(float(top_change['price_delta']), 4),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "pct_change": {
                        "value": round(float(top_change['pct_change']), 2),
                        "unit": "%",
                        "numerator": round(float(top_change['pct_change']), 2),
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    },
                    "effective_date": {
                        "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-16T00:00:00+00:00",
                        "period_end": "2026-03-23T00:00:00+00:00"
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(price_changes),
                "coverage_notes": [
                    f"Detected {len(price_changes)} supplier price changes",
                    "Only changes with old_price, new_price, and effective_date included",
                    "Showing highest absolute price delta"
                ],
                "assumptions": [
                    "Email extraction accurately captured supplier price changes",
                    "Effective dates are accurate",
                    "Price changes apply to relevant menu items (requires recipe/BOM for exact impact)"
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
    json.dump(output, f, indent=2)
