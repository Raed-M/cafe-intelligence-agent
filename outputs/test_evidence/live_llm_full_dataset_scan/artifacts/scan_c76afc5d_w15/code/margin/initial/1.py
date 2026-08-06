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
analysis_start = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)

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

# Join with POS data for analysis period
pos_with_menu = pos_analysis.merge(menu_analysis, on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[~pos_with_menu['is_refund']].copy()

# Calculate metrics by item
item_metrics = pos_sales.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_price_sar': 'first',
    'unit_cost_sar': 'first',
    'item_en': 'first',
    'category': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_metrics.columns = ['sku', 'total_quantity', 'total_revenue', 'unit_price', 'unit_cost', 'item_name', 'category', 'basket_count']

# Calculate COGS and gross profit
item_metrics['total_cogs'] = item_metrics['total_quantity'] * item_metrics['unit_cost']
item_metrics['gross_profit'] = item_metrics['total_revenue'] - item_metrics['total_cogs']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['total_revenue'] * 100).round(2)

# Filter items with meaningful sales
item_metrics_filtered = item_metrics[item_metrics['total_quantity'] > 0].sort_values('total_revenue', ascending=False)

if len(item_metrics_filtered) > 0:
    top_item = item_metrics_filtered.iloc[0]
    
    finding_1 = {
        "title": "Top Revenue Item Economics (Analysis Week)",
        "claim": f"Item '{top_item['item_name']}' (SKU: {top_item['sku']}) generated {top_item['total_revenue']:.2f} SAR revenue with {top_item['gross_profit']:.2f} SAR gross profit and {top_item['gross_margin_pct']:.1f}% margin during the analysis week.",
        "finding_type": "item_economics",
        "metrics": {
            "total_revenue_sar": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "total_cogs_sar": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "gross_profit_sar": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "gross_margin_pct": {
                "value": top_item['gross_margin_pct'],
                "unit": "%",
                "numerator": top_item['gross_margin_pct'],
                "denominator": 100,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "units_sold": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "unit_cost_sar": {
                "value": round(top_item['unit_cost'], 2),
                "unit": "SAR",
                "numerator": round(top_item['unit_cost'], 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            "Analysis period: 2026-04-20 to 2026-04-27",
            "Refunds excluded from revenue calculation",
            "Unit cost from menu.unit_cost_sar",
            "COGS calculated as total_quantity × unit_cost_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is accurate and constant during period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM available; using menu-level unit costs"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Filter inventory for analysis week
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-04-20'].copy()

# Calculate total waste cost for items with known waste
waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & (inventory_analysis['known_waste_cost_sar'] > 0)].copy()

if len(waste_items) > 0:
    total_waste_cost = waste_items['known_waste_cost_sar'].sum()
    waste_units = waste_items['units_wasted'].sum()
    
    # Get corresponding revenue for waste items
    waste_skus = waste_items['sku'].unique()
    waste_revenue = pos_sales[pos_sales['sku'].isin(waste_skus)]['line_total_sar'].sum()
    
    if waste_revenue > 0:
        waste_impact_pct = (total_waste_cost / waste_revenue * 100)
    else:
        waste_impact_pct = 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Analysis Week)",
        "claim": f"Known waste cost totaled {total_waste_cost:.2f} SAR across {int(waste_units)} units during the analysis week, representing {waste_impact_pct:.2f}% of related item revenue.",
        "finding_type": "waste_economics",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "waste_units": {
                "value": int(waste_units),
                "unit": "units",
                "numerator": int(waste_units),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "waste_impact_pct": {
                "value": round(waste_impact_pct, 2),
                "unit": "%",
                "numerator": round(waste_impact_pct, 2),
                "denominator": 100,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            },
            "related_revenue_sar": {
                "value": round(waste_revenue, 2),
                "unit": "SAR",
                "numerator": round(waste_revenue, 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00Z",
                "period_end": "2026-04-27T00:00:00Z"
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            "Only non-null waste cost observations included",
            "Analysis week: 2026-04-20",
            "Waste cost from inventory.known_waste_cost_sar",
            "Related revenue calculated from POS for waste item SKUs"
        ],
        "assumptions": [
            "known_waste_cost_sar accurately reflects waste value",
            "Waste items match POS SKUs for revenue correlation"
        ],
        "confidence": 0.90
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
    # Process first price change
    price_change = price_changes.iloc[0]
    
    old_price = float(price_change['old_price'])
    new_price = float(price_change['new_price'])
    price_delta = new_price - old_price
    price_delta_pct = (price_delta / old_price * 100) if old_price != 0 else 0
    
    ingredient = price_change['entity_or_ingredient']
    effective_date = price_change['effective_date']
    
    # Try to find standing order quantity from facts
    standing_qty = None
    if pd.notna(price_change['facts']):
        facts_text = str(price_change['facts']).lower()
        if 'standing order' in facts_text or 'monthly' in facts_text:
            # Extract number if possible
            import re
            numbers = re.findall(r'\d+', facts_text)
            if numbers:
                standing_qty = int(numbers[0])
    
    # Calculate procurement cost scenario if standing quantity available
    procurement_impact = None
    if standing_qty:
        procurement_impact = standing_qty * price_delta
    
    finding_3 = {
        "title": "Supplier Price Change Detection",
        "claim": f"Supplier price change for '{ingredient}': {old_price:.2f} {price_change['currency']} → {new_price:.2f} {price_change['currency']} per {price_change['unit']} (effective {effective_date}), representing {price_delta_pct:.1f}% change.",
        "finding_type": "supplier_pricing",
        "metrics": {
            "old_price": {
                "value": round(old_price, 2),
                "unit": price_change['currency'],
                "numerator": round(old_price, 2),
                "denominator": None,
                "period_start": effective_date,
                "period_end": effective_date
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": price_change['currency'],
                "numerator": round(new_price, 2),
                "denominator": None,
                "period_start": effective_date,
                "period_end": effective_date
            },
            "price_delta": {
                "value": round(price_delta, 2),
                "unit": price_change['currency'],
                "numerator": round(price_delta, 2),
                "denominator": None,
                "period_start": effective_date,
                "period_end": effective_date
            },
            "price_delta_pct": {
                "value": round(price_delta_pct, 2),
                "unit": "%",
                "numerator": round(price_delta_pct, 2),
                "denominator": 100,
                "period_start": effective_date,
                "period_end": effective_date
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Ingredient: {ingredient}",
            f"Unit: {price_change['unit']}",
            f"Effective date: {effective_date}",
            f"Confidence: {price_change['confidence']}"
        ],
        "assumptions": [
            "Email extraction accurately captured price and effective date",
            "Price change applies to specified ingredient only",
            "No recipe/BOM available; per-drink impact cannot be calculated"
        ] + (
            [f"Standing order quantity: {standing_qty} units (estimated from email facts)"] if standing_qty else []
        ),
        "confidence": float(price_change['confidence']) if pd.notna(price_change['confidence']) else 0.75
    }
    
    # Add procurement impact if available
    if procurement_impact:
        finding_3['metrics']['procurement_cost_impact_sar'] = {
            "value": round(procurement_impact, 2),
            "unit": "SAR",
            "numerator": round(procurement_impact, 2),
            "denominator": None,
            "period_start": effective_date,
            "period_end": effective_date
        }
        finding_3['assumptions'].append(
            f"Procurement scenario: {standing_qty} units × {price_delta:.2f} SAR/unit = {procurement_impact:.2f} SAR impact (assumes continued standing order volume and payment terms)"
        )
    
    findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
