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
analysis_start = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_with_costs = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar']].copy()

# Join POS with menu to get unit costs
pos_with_costs = pos_analysis.merge(menu_with_costs, on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_costs[~pos_with_costs['is_refund']].copy()

# Calculate metrics by item
item_metrics = []
for sku in pos_sales['sku'].unique():
    sku_data = pos_sales[pos_sales['sku'] == sku]
    
    if len(sku_data) == 0:
        continue
    
    item_name = sku_data['item_name_en'].iloc[0]
    unit_price = sku_data['unit_price_sar'].iloc[0]
    unit_cost = sku_data['unit_cost_sar'].iloc[0]
    
    # Skip if unit cost is null
    if pd.isna(unit_cost):
        continue
    
    total_quantity = sku_data['quantity'].sum()
    total_revenue = sku_data['line_total_sar'].sum()
    total_cogs = total_quantity * unit_cost
    total_gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        gross_margin_pct = (total_gross_profit / total_revenue) * 100
    else:
        gross_margin_pct = 0
    
    item_metrics.append({
        'sku': sku,
        'item_name': item_name,
        'quantity_sold': total_quantity,
        'unit_price': unit_price,
        'unit_cost': unit_cost,
        'total_revenue': total_revenue,
        'total_cogs': total_cogs,
        'gross_profit': total_gross_profit,
        'gross_margin_pct': gross_margin_pct,
        'transaction_count': sku_data['transaction_id'].nunique()
    })

# Sort by gross profit to identify top contributors
item_metrics_df = pd.DataFrame(item_metrics)
if len(item_metrics_df) > 0:
    item_metrics_df = item_metrics_df.sort_values('gross_profit', ascending=False)
    
    # Create finding for top 3 items by gross profit
    top_items = item_metrics_df.head(3)
    
    if len(top_items) > 0:
        finding_1 = {
            "title": "Top Gross Profit Contributors (Analysis Period)",
            "claim": f"The top 3 items by gross profit contribution during {analysis_start.date()} to {analysis_end.date()} are: {', '.join(top_items['item_name'].tolist())}. Combined they generated {top_items['gross_profit'].sum():.2f} SAR in gross profit from {top_items['quantity_sold'].sum():.0f} units sold.",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items.iloc[0]['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_1_gross_profit": {
                    "value": round(top_items.iloc[0]['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_1_quantity": {
                    "value": int(top_items.iloc[0]['quantity_sold']),
                    "unit": "units",
                    "numerator": int(top_items.iloc[0]['quantity_sold']),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_1_margin_pct": {
                    "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                    "denominator": 100,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "combined_top_3_gross_profit": {
                    "value": round(top_items['gross_profit'].sum(), 2),
                    "unit": "SAR",
                    "numerator": round(top_items['gross_profit'].sum(), 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "combined_top_3_quantity": {
                    "value": int(top_items['quantity_sold'].sum()),
                    "unit": "units",
                    "numerator": int(top_items['quantity_sold'].sum()),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(pos_sales['transaction_id'].nunique()),
            "coverage_notes": [
                f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                f"Includes {len(pos_sales)} POS line items from {pos_sales['transaction_id'].nunique()} transactions",
                f"Only items with non-null unit_cost_sar in menu included",
                f"Refunds excluded from revenue calculation"
            ],
            "assumptions": [
                "Unit costs from menu_items.unit_cost_sar are accurate and current",
                "POS line_total_sar reflects actual revenue after discounts",
                "No recipe/BOM available; unit cost is applied uniformly per item"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
# Calculate waste costs from inventory data
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-05-25'].copy()

if len(inventory_analysis) > 0:
    # Filter for non-null waste costs
    waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_units_wasted = waste_items['units_wasted'].sum()
        
        finding_2 = {
            "title": "Quantified Waste Cost (Week of 2026-05-25)",
            "claim": f"During the week of 2026-05-25, {int(total_units_wasted)} units were wasted with a known waste cost of {total_waste_cost:.2f} SAR across {len(waste_items)} items.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": int(total_units_wasted),
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "items_with_waste": {
                    "value": len(waste_items),
                    "unit": "count",
                    "numerator": len(waste_items),
                    "denominator": None,
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Only items with non-null known_waste_cost_sar included",
                f"Week of 2026-05-25 inventory data",
                f"Items with waste: {', '.join(waste_items['item'].tolist())}"
            ],
            "assumptions": [
                "known_waste_cost_sar values are accurate and complete for reported waste",
                "Null waste costs indicate unknown or unreported waste (excluded from calculation)"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Extract price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Calculate percentage changes
    price_changes['pct_change'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    
    # Sort by absolute percentage change
    price_changes['abs_pct_change'] = price_changes['pct_change'].abs()
    price_changes = price_changes.sort_values('abs_pct_change', ascending=False)
    
    # Get top price change
    top_change = price_changes.iloc[0]
    
    finding_3 = {
        "title": "Supplier Price Change Detected",
        "claim": f"Email evidence indicates {top_change['entity_or_ingredient']} price change from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']}/{top_change['unit']} (effective {top_change['effective_date']}), representing a {top_change['pct_change']:.2f}% change.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {
                "value": str(top_change['entity_or_ingredient']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": None,
                "period_end": None
            },
            "old_price": {
                "value": float(top_change['old_price']),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": float(top_change['old_price']),
                "denominator": None,
                "period_start": None,
                "period_end": None
            },
            "new_price": {
                "value": float(top_change['new_price']),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": float(top_change['new_price']),
                "denominator": None,
                "period_start": None,
                "period_end": None
            },
            "percentage_change": {
                "value": float(top_change['pct_change']),
                "unit": "%",
                "numerator": float(top_change['pct_change']),
                "denominator": 100,
                "period_start": None,
                "period_end": None
            },
            "effective_date": {
                "value": str(top_change['effective_date']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": None,
                "period_end": None
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price changes detected: {len(price_changes)}",
            f"Email sender: {top_change['sender']}",
            f"Email date: {top_change['date']}",
            f"Confidence level: {top_change['confidence']}"
        ],
        "assumptions": [
            "Email extraction accuracy depends on extraction_mode and confidence score",
            "Price change applies to specified ingredient/entity only",
            "No recipe/BOM available; cannot calculate per-drink impact without additional data"
        ],
        "confidence": float(top_change['confidence']) if pd.notna(top_change['confidence']) else 0.7
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
