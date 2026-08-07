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
analysis_start = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-06-15', tz=timezone.utc)
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
        total_revenue = sum([item['revenue'] for item in top_items])
        
        finding_1 = {
            "title": "Top 5 Items by Gross Profit (Analysis Week)",
            "claim": f"The top 5 items by gross profit contribution generated SAR {total_gp:.2f} in gross profit from SAR {total_revenue:.2f} in revenue during the analysis week (2026-06-15 to 2026-06-22).",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items[0]['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-15T00:00:00Z",
                    "period_end": "2026-06-22T00:00:00Z"
                },
                "top_item_1_gross_profit": {
                    "value": round(top_items[0]['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_items[0]['gross_profit'], 2),
                    "denominator": None,
                    "period_start": "2026-06-15T00:00:00Z",
                    "period_end": "2026-06-22T00:00:00Z"
                },
                "top_item_1_margin_pct": {
                    "value": round(top_items[0]['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_items[0]['gross_margin_pct'], 2),
                    "denominator": 100,
                    "period_start": "2026-06-15T00:00:00Z",
                    "period_end": "2026-06-22T00:00:00Z"
                },
                "top_5_total_gross_profit": {
                    "value": round(total_gp, 2),
                    "unit": "SAR",
                    "numerator": round(total_gp, 2),
                    "denominator": None,
                    "period_start": "2026-06-15T00:00:00Z",
                    "period_end": "2026-06-22T00:00:00Z"
                },
                "top_5_total_revenue": {
                    "value": round(total_revenue, 2),
                    "unit": "SAR",
                    "numerator": round(total_revenue, 2),
                    "denominator": None,
                    "period_start": "2026-06-15T00:00:00Z",
                    "period_end": "2026-06-22T00:00:00Z"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                f"Analysis period: 2026-06-15 to 2026-06-22",
                f"Total POS transactions in period: {len(pos_analysis)}",
                f"Sales lines (excluding refunds): {len(pos_sales)}",
                f"Unique items sold: {len(pos_sales['sku'].unique())}",
                f"Menu items with unit cost data: {len(menu_df[menu_df['unit_cost_sar'].notna()])}"
            ],
            "assumptions": [
                "Unit costs from menu_items.unit_cost_sar are current and applicable to all sales in the period",
                "Refunds are excluded from revenue and COGS calculations",
                "Line totals in POS are accurate and include discounts"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
if len(inventory_analysis) > 0:
    waste_data = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_data) > 0:
        total_waste_cost = waste_data['known_waste_cost_sar'].sum()
        total_units_wasted = waste_data['units_wasted'].sum()
        total_units_sold = waste_data['units_sold'].sum()
        
        waste_items = []
        for idx, row in waste_data.iterrows():
            if pd.notna(row['units_wasted']) and row['units_wasted'] > 0:
                waste_items.append({
                    'item': row['item'],
                    'units_wasted': row['units_wasted'],
                    'waste_cost': row['known_waste_cost_sar'],
                    'units_sold': row['units_sold']
                })
        
        if waste_items:
            waste_items_sorted = sorted(waste_items, key=lambda x: x['waste_cost'], reverse=True)
            
            finding_2 = {
                "title": "Quantified Waste Cost (Week of 2026-06-15)",
                "claim": f"Known waste cost for the week of 2026-06-15 totaled SAR {total_waste_cost:.2f} across {len(waste_items)} items with recorded waste observations.",
                "finding_type": "waste_cost",
                "metrics": {
                    "total_waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": None,
                        "period_start": "2026-06-15T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "total_units_wasted": {
                        "value": int(total_units_wasted),
                        "unit": "units",
                        "numerator": int(total_units_wasted),
                        "denominator": None,
                        "period_start": "2026-06-15T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "waste_to_sales_ratio": {
                        "value": round(total_units_wasted / (total_units_wasted + total_units_sold) * 100, 2) if (total_units_wasted + total_units_sold) > 0 else 0,
                        "unit": "%",
                        "numerator": round(total_units_wasted, 2),
                        "denominator": round(total_units_wasted + total_units_sold, 2),
                        "period_start": "2026-06-15T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "highest_waste_item": {
                        "value": waste_items_sorted[0]['item'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-06-15T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "highest_waste_cost": {
                        "value": round(waste_items_sorted[0]['waste_cost'], 2),
                        "unit": "SAR",
                        "numerator": round(waste_items_sorted[0]['waste_cost'], 2),
                        "denominator": None,
                        "period_start": "2026-06-15T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_data),
                "coverage_notes": [
                    f"Week starting: 2026-06-15",
                    f"Items with non-null waste cost: {len(waste_data)}",
                    f"Items with zero waste cost or null values are excluded from this analysis",
                    f"Waste cost is only calculated for items with known_waste_cost_sar values"
                ],
                "assumptions": [
                    "known_waste_cost_sar values in inventory are accurate and represent actual waste",
                    "Blank waste values are treated as unknown, not zero",
                    "Waste cost is independent of sales volume"
                ],
                "confidence": 0.85
            }
            findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Impact
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
        
        # Ensure effective_date is timezone-naive for comparison with naive timestamps
        if price_changes['effective_date'].dt.tz is not None:
            price_changes['effective_date'] = price_changes['effective_date'].dt.tz_localize(None)
        
        # Create naive versions of analysis dates for comparison
        analysis_end_naive = analysis_end.replace(tzinfo=None)
        comparison_start = pd.Timestamp('2026-05-01')
        
        # Filter for changes that might affect the analysis period
        relevant_changes = price_changes[
            (price_changes['effective_date'] <= analysis_end_naive) &
            (price_changes['effective_date'] >= comparison_start)
        ].copy()
        
        if len(relevant_changes) > 0:
            # Calculate percentage changes
            relevant_changes['price_change_pct'] = (
                (relevant_changes['new_price'] - relevant_changes['old_price']) / 
                relevant_changes['old_price'] * 100
            )
            
            # Sort by absolute price change
            relevant_changes['abs_price_change'] = abs(relevant_changes['new_price'] - relevant_changes['old_price'])
            relevant_changes_sorted = relevant_changes.sort_values('abs_price_change', ascending=False)
            
            top_change = relevant_changes_sorted.iloc[0]
            
            finding_3 = {
                "title": "Supplier Price Change Detection",
                "claim": f"Supplier email evidence shows {len(relevant_changes)} price changes effective between 2026-05-01 and 2026-06-22. The largest change is {top_change['entity_or_ingredient']}: {top_change['old_price']:.2f} → {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']} ({top_change['price_change_pct']:.1f}%), effective {top_change['effective_date'].strftime('%Y-%m-%d')}.",
                "finding_type": "supplier_price_change",
                "metrics": {
                    "total_price_changes_detected": {
                        "value": len(relevant_changes),
                        "unit": "count",
                        "numerator": len(relevant_changes),
                        "denominator": None,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "largest_change_ingredient": {
                        "value": top_change['entity_or_ingredient'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "largest_change_old_price": {
                        "value": round(top_change['old_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": round(top_change['old_price'], 2),
                        "denominator": None,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "largest_change_new_price": {
                        "value": round(top_change['new_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": round(top_change['new_price'], 2),
                        "denominator": None,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "largest_change_pct": {
                        "value": round(top_change['price_change_pct'], 2),
                        "unit": "%",
                        "numerator": round(top_change['price_change_pct'], 2),
                        "denominator": 100,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    },
                    "effective_date": {
                        "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-01T00:00:00Z",
                        "period_end": "2026-06-22T00:00:00Z"
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(relevant_changes),
                "coverage_notes": [
                    f"Price changes detected from supplier emails: {len(relevant_changes)}",
                    f"Analysis window: 2026-05-01 to 2026-06-22",
                    f"Only changes with both old_price and new_price values are included",
                    f"No recipe/BOM data available to calculate per-drink impact"
                ],
                "assumptions": [
                    "Supplier email extraction is accurate",
                    "Effective dates in emails are correct",
                    "Price changes apply to relevant menu items (not verified without recipe data)",
                    "Standing order quantities and payment terms are not confirmed in available data"
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
    json.dump(output, f, indent=2)
