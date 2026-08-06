import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

def load_inputs():
    """Load input artifact paths from environment."""
    with open(os.environ['ANALYST_INPUTS_JSON']) as f:
        run_meta = json.load(f)
    return run_meta['inputs'], run_meta['output_path']

def parse_iso_datetime(dt_str):
    """Parse ISO datetime string to datetime object."""
    if isinstance(dt_str, str):
        return pd.to_datetime(dt_str)
    return dt_str

def calculate_findings(inputs: Dict[str, str], output_path: str):
    """Calculate margin and cost analysis findings."""
    
    # Load all artifacts
    pos_df = pd.read_parquet(inputs['pos'])
    inventory_df = pd.read_parquet(inputs['inventory'])
    menu_df = pd.read_parquet(inputs['menu'])
    emails_df = pd.read_parquet(inputs['emails'])
    
    # Parse dates
    analysis_start = parse_iso_datetime("2026-05-11T00:00:00+03:00")
    analysis_end = parse_iso_datetime("2026-05-18T00:00:00+03:00")
    prev_start = parse_iso_datetime("2026-05-04T00:00:00+03:00")
    prev_end = parse_iso_datetime("2026-05-11T00:00:00+03:00")
    
    # Convert POS timestamp to datetime
    pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
    
    # Filter POS data for analysis period
    pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & 
                          (pos_df['timestamp'] < analysis_end)].copy()
    pos_previous = pos_df[(pos_df['timestamp'] >= prev_start) & 
                          (pos_df['timestamp'] < prev_end)].copy()
    
    findings = []
    
    # FINDING 1: Item-level COGS and Gross Profit Analysis
    # Calculate exact item economics from menu and POS
    
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
        
        if len(sku_data) == 0:
            continue
            
        item_name = sku_data['item_name_en'].iloc[0]
        total_quantity = sku_data['quantity'].sum()
        total_revenue = sku_data['line_total_sar'].sum()
        
        # Get unit cost from menu
        unit_cost = sku_data['unit_cost_sar'].iloc[0]
        
        if pd.isna(unit_cost) or unit_cost == 0:
            continue
        
        total_cogs = total_quantity * unit_cost
        gross_profit = total_revenue - total_cogs
        gross_margin_pct = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        item_metrics.append({
            'sku': sku,
            'item_name': item_name,
            'quantity': total_quantity,
            'revenue': total_revenue,
            'unit_cost': unit_cost,
            'total_cogs': total_cogs,
            'gross_profit': gross_profit,
            'gross_margin_pct': gross_margin_pct
        })
    
    if item_metrics:
        item_df = pd.DataFrame(item_metrics)
        
        # Find top 3 items by gross profit
        top_items = item_df.nlargest(3, 'gross_profit')
        
        if len(top_items) > 0:
            top_item = top_items.iloc[0]
            
            finding1 = {
                "title": "Top Gross Profit Item - Exact Item Economics",
                "claim": f"Item '{top_item['item_name']}' (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period with {top_item['quantity']:.0f} units sold at {top_item['gross_margin_pct']:.1f}% gross margin.",
                "finding_type": "item_economics",
                "metrics": {
                    "gross_profit_sar": {
                        "value": round(top_item['gross_profit'], 2),
                        "unit": "SAR",
                        "numerator": round(top_item['revenue'], 2),
                        "denominator": round(top_item['total_cogs'], 2),
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "units_sold": {
                        "value": int(top_item['quantity']),
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "gross_margin_percent": {
                        "value": round(top_item['gross_margin_pct'], 1),
                        "unit": "%",
                        "numerator": round(top_item['gross_profit'], 2),
                        "denominator": round(top_item['revenue'], 2),
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "unit_cost_sar": {
                        "value": round(top_item['unit_cost'], 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": int(top_item['quantity']),
                "coverage_notes": [
                    "Analysis period: 2026-05-11 to 2026-05-18",
                    "Includes all non-refund transactions",
                    "Unit costs sourced from menu.parquet",
                    "Revenue calculated from line_total_sar"
                ],
                "assumptions": [
                    "Menu unit_cost_sar is current and accurate for the analysis period",
                    "All POS line_total_sar values are correct",
                    "No recipe/BOM adjustments applied"
                ],
                "confidence": 0.95
            }
            findings.append(finding1)
    
    # FINDING 2: Waste Cost Impact
    # Calculate known waste costs from inventory
    inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
    
    # Filter inventory for analysis week
    analysis_week_start = parse_iso_datetime("2026-05-11T00:00:00+03:00")
    inv_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start].copy()
    
    if len(inv_analysis) > 0:
        # Calculate total waste cost
        inv_analysis['known_waste_cost_sar'] = pd.to_numeric(
            inv_analysis['known_waste_cost_sar'], 
            errors='coerce'
        )
        
        total_waste_cost = inv_analysis['known_waste_cost_sar'].sum()
        total_units_wasted = inv_analysis['units_wasted'].sum()
        
        if total_waste_cost > 0 and not pd.isna(total_waste_cost):
            # Get total revenue for context
            total_revenue_analysis = pos_sales['line_total_sar'].sum()
            waste_pct_of_revenue = (total_waste_cost / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0
            
            finding2 = {
                "title": "Quantified Waste Cost Impact",
                "claim": f"Known waste cost during the analysis week totaled {total_waste_cost:.2f} SAR across {int(total_units_wasted)} units, representing {waste_pct_of_revenue:.2f}% of total revenue.",
                "finding_type": "waste_cost",
                "metrics": {
                    "total_waste_cost_sar": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "units_wasted": {
                        "value": int(total_units_wasted),
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    },
                    "waste_as_pct_of_revenue": {
                        "value": round(waste_pct_of_revenue, 2),
                        "unit": "%",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": round(total_revenue_analysis, 2),
                        "period_start": "2026-05-11T00:00:00+03:00",
                        "period_end": "2026-05-18T00:00:00+03:00"
                    }
                },
                "source_names": ["inventory", "pos"],
                "sample_size": len(inv_analysis),
                "coverage_notes": [
                    "Only non-null waste cost observations included",
                    "Week starting 2026-05-11",
                    "Waste cost from inventory.known_waste_cost_sar"
                ],
                "assumptions": [
                    "known_waste_cost_sar values are accurate",
                    "Waste occurred during the analysis period"
                ],
                "confidence": 0.90
            }
            findings.append(finding2)
    
    # FINDING 3: Supplier Price Changes and Margin Impact
    # Detect dated supplier price changes from emails
    emails_df['date'] = pd.to_datetime(emails_df['date'])
    emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])
    
    # Filter for price changes with old and new prices
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Get the most recent price change
        price_changes = price_changes.sort_values('effective_date', ascending=False)
        latest_change = price_changes.iloc[0]
        
        old_price = float(latest_change['old_price'])
        new_price = float(latest_change['new_price'])
        price_delta = new_price - old_price
        pct_change = (price_delta / old_price * 100) if old_price != 0 else 0
        
        entity = latest_change['entity_or_ingredient']
        unit = latest_change['unit'] if pd.notna(latest_change['unit']) else "unit"
        effective_date = latest_change['effective_date']
        
        finding3 = {
            "title": "Supplier Price Change Detected",
            "claim": f"Price change for {entity}: {old_price:.2f} SAR → {new_price:.2f} SAR per {unit} (effective {effective_date.strftime('%Y-%m-%d')}), representing a {pct_change:+.1f}% change. This is a supplier-level fact; impact on menu items depends on recipe/BOM and order volumes.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "old_price_sar": {
                    "value": round(old_price, 2),
                    "unit": f"SAR per {unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "new_price_sar": {
                    "value": round(new_price, 2),
                    "unit": f"SAR per {unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "price_delta_sar": {
                    "value": round(price_delta, 2),
                    "unit": f"SAR per {unit}",
                    "numerator": round(price_delta, 2),
                    "denominator": round(old_price, 2),
                    "period_start": None,
                    "period_end": None
                },
                "percent_change": {
                    "value": round(pct_change, 1),
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
                f"Most recent price change for {entity}",
                f"Effective date: {effective_date.strftime('%Y-%m-%d')}",
                "Supplier-level fact only; no recipe/BOM available for menu item impact calculation"
            ],
            "assumptions": [
                "Email extraction accurately captured old_price, new_price, and effective_date",
                "Price change applies to the specified entity/ingredient",
                "No recipe/BOM data available to calculate per-drink impact",
                "Standing order volumes and payment terms are unknown"
            ],
            "confidence": 0.85
        }
        findings.append(finding3)
    
    # Prepare output
    result = {
        "status": "success" if len(findings) > 0 else "insufficient_data",
        "findings": findings
    }
    
    # Write result to output path
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result

def main():
    inputs, output_path = load_inputs()
    result = calculate_findings(inputs, output_path)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
