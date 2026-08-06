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
analysis_start = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-04-13', tz=timezone.utc)
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
        
        # Top 3 items by gross profit
        top_items = item_metrics_sorted[:3]
        
        total_revenue_all = sum(m['revenue'] for m in item_metrics)
        total_cogs_all = sum(m['total_cogs'] for m in item_metrics)
        total_gp_all = sum(m['gross_profit'] for m in item_metrics)
        overall_margin = (total_gp_all / total_revenue_all * 100) if total_revenue_all > 0 else 0
        
        finding_1 = {
            "title": "Item-Level Gross Profit Analysis (Week of 2026-04-13)",
            "claim": f"Top 3 items by gross profit contribution: {', '.join([m['item_name'] for m in top_items])}. Overall cafe gross margin: {overall_margin:.1f}%.",
            "finding_type": "margin_analysis",
            "metrics": {
                "total_revenue_sar": {
                    "value": round(total_revenue_all, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "total_cogs_sar": {
                    "value": round(total_cogs_all, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "total_gross_profit_sar": {
                    "value": round(total_gp_all, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "overall_gross_margin_pct": {
                    "value": round(overall_margin, 1),
                    "unit": "%",
                    "numerator": round(total_gp_all, 2),
                    "denominator": round(total_revenue_all, 2),
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "top_item_1_name": {
                    "value": top_items[0]['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "top_item_1_gross_profit_sar": {
                    "value": round(top_items[0]['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "top_item_1_margin_pct": {
                    "value": round(top_items[0]['gross_margin_pct'], 1),
                    "unit": "%",
                    "numerator": round(top_items[0]['gross_profit'], 2),
                    "denominator": round(top_items[0]['revenue'], 2),
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                f"Analysis period: 2026-04-13 to 2026-04-20",
                f"Total POS transactions (non-refund): {len(pos_sales)}",
                f"Menu items with cost data: {len([m for m in item_metrics if m['unit_cost'] is not None])}",
                "Refunds excluded from revenue calculation"
            ],
            "assumptions": [
                "Unit costs from menu_items.unit_cost_sar applied uniformly across all sales",
                "Line totals reflect actual revenue after discounts",
                "No recipe/BOM data available; per-drink ingredient costs not calculated"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Impact
# Calculate known waste costs from inventory
if len(inventory_analysis) > 0:
    waste_items = inventory_analysis[inventory_analysis['units_wasted'].notna() & (inventory_analysis['units_wasted'] > 0)].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_units_wasted = waste_items['units_wasted'].sum()
        
        # Get corresponding revenue for waste items
        waste_skus = waste_items['sku'].unique()
        waste_revenue = pos_analysis[pos_analysis['sku'].isin(waste_skus) & ~pos_analysis['is_refund']]['line_total_sar'].sum()
        
        if waste_revenue > 0:
            waste_impact_pct = (total_waste_cost / waste_revenue * 100)
        else:
            waste_impact_pct = 0
        
        finding_2 = {
            "title": "Quantified Waste Cost Impact (Week of 2026-04-13)",
            "claim": f"Known waste cost of {total_waste_cost:.2f} SAR ({total_units_wasted:.0f} units) represents {waste_impact_pct:.1f}% of waste-item revenue.",
            "finding_type": "waste_analysis",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "total_units_wasted": {
                    "value": round(total_units_wasted, 0),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "waste_items_count": {
                    "value": len(waste_items),
                    "unit": "items",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                },
                "waste_impact_pct_of_revenue": {
                    "value": round(waste_impact_pct, 1),
                    "unit": "%",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": round(waste_revenue, 2),
                    "period_start": "2026-04-13T00:00:00Z",
                    "period_end": "2026-04-20T00:00:00Z"
                }
            },
            "source_names": ["inventory", "pos"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                f"Inventory week: 2026-04-13",
                f"Items with non-null waste observations: {len(waste_items)}",
                "Only known_waste_cost_sar values included (blank waste treated as unknown, not zero)"
            ],
            "assumptions": [
                "Waste cost reflects actual unit cost × units wasted",
                "Waste items correspond to POS sales in same period"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Analyze emails for price changes
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Convert dates - ensure both are timezone-naive for comparison
        price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'], utc=False)
        price_changes['date'] = pd.to_datetime(price_changes['date'], utc=False)
        
        # Create naive versions of analysis boundaries for comparison
        analysis_start_naive = analysis_start.replace(tzinfo=None)
        analysis_end_naive = analysis_end.replace(tzinfo=None)
        
        # Filter for changes relevant to analysis period
        relevant_changes = price_changes[
            (price_changes['effective_date'] >= analysis_start_naive) & 
            (price_changes['effective_date'] < analysis_end_naive)
        ].copy()
        
        if len(relevant_changes) > 0:
            # Calculate percentage changes
            relevant_changes['price_change_pct'] = (
                (relevant_changes['new_price'] - relevant_changes['old_price']) / 
                relevant_changes['old_price'] * 100
            )
            
            # Get the most significant change
            relevant_changes['abs_change_pct'] = relevant_changes['price_change_pct'].abs()
            top_change = relevant_changes.nlargest(1, 'abs_change_pct').iloc[0]
            
            entity = top_change['entity_or_ingredient']
            old_price = top_change['old_price']
            new_price = top_change['new_price']
            currency = top_change['currency']
            unit = top_change['unit']
            change_pct = top_change['price_change_pct']
            effective_date = top_change['effective_date']
            
            finding_3 = {
                "title": "Supplier Price Change Detected (Week of 2026-04-13)",
                "claim": f"{entity} price changed from {old_price} to {new_price} {currency}/{unit} (effective {effective_date.strftime('%Y-%m-%d')}), representing a {change_pct:+.1f}% change.",
                "finding_type": "supplier_cost_analysis",
                "metrics": {
                    "entity_or_ingredient": {
                        "value": entity,
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-13T00:00:00Z",
                        "period_end": "2026-04-20T00:00:00Z"
                    },
                    "old_price": {
                        "value": round(old_price, 2),
                        "unit": f"{currency}/{unit}",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-13T00:00:00Z",
                        "period_end": "2026-04-20T00:00:00Z"
                    },
                    "new_price": {
                        "value": round(new_price, 2),
                        "unit": f"{currency}/{unit}",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-13T00:00:00Z",
                        "period_end": "2026-04-20T00:00:00Z"
                    },
                    "price_change_pct": {
                        "value": round(change_pct, 1),
                        "unit": "%",
                        "numerator": round(new_price - old_price, 2),
                        "denominator": round(old_price, 2),
                        "period_start": "2026-04-13T00:00:00Z",
                        "period_end": "2026-04-20T00:00:00Z"
                    },
                    "effective_date": {
                        "value": effective_date.strftime('%Y-%m-%d'),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-04-13T00:00:00Z",
                        "period_end": "2026-04-20T00:00:00Z"
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(relevant_changes),
                "coverage_notes": [
                    f"Price changes effective during analysis period: {len(relevant_changes)}",
                    "Email extraction confidence varies by source",
                    "No standing order quantities available to calculate procurement cost scenario"
                ],
                "assumptions": [
                    "Price change applies to supplier invoices effective on stated date",
                    "No recipe/BOM data available; per-drink cost impact not calculated",
                    "Continued order volume and payment terms not confirmed"
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
