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
analysis_start = "2026-02-16T00:00:00+03:00"
analysis_end = "2026-02-23T00:00:00+03:00"
previous_start = "2026-02-09T00:00:00+03:00"
previous_end = "2026-02-16T00:00:00+03:00"

# Convert timestamps to datetime for filtering
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting_dt'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date_dt'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date_dt'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter POS data for analysis period
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

pos_analysis = pos_df[(pos_df['timestamp_dt'] >= analysis_start_dt) & (pos_df['timestamp_dt'] < analysis_end_dt)].copy()
pos_previous = pos_df[(pos_df['timestamp_dt'] >= previous_start_dt) & (pos_df['timestamp_dt'] < previous_end_dt)].copy()

# Filter inventory for analysis week
inventory_analysis = inventory_df[inventory_df['week_starting_dt'] >= analysis_start_dt].copy()
inventory_previous = inventory_df[(inventory_df['week_starting_dt'] >= previous_start_dt) & (inventory_df['week_starting_dt'] < analysis_start_dt)].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
if len(pos_analysis) > 0 and len(menu_df) > 0:
    # Merge POS with menu to get unit costs
    pos_with_cost = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left')
    
    # Calculate COGS and gross profit for each line item
    pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
    pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
    
    # Exclude refunds from calculations
    pos_with_cost_no_refund = pos_with_cost[pos_with_cost['is_refund'] == False].copy()
    
    # Aggregate by item
    item_economics = pos_with_cost_no_refund.groupby('item_name_en').agg({
        'quantity': 'sum',
        'line_total_sar': 'sum',
        'cogs_sar': 'sum',
        'gross_profit_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    
    item_economics.columns = ['item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
    item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)
    
    # Sort by gross profit contribution
    item_economics = item_economics.sort_values('total_gross_profit', ascending=False)
    
    # Top 5 items by gross profit
    top_items = item_economics.head(5)
    
    if len(top_items) > 0:
        finding_1 = {
            "title": "Top 5 Items by Gross Profit Contribution (Analysis Week)",
            "claim": f"During the analysis week ({analysis_start} to {analysis_end}), the top 5 items by gross profit contribution generated {top_items['total_gross_profit'].sum():.2f} SAR in total gross profit, representing {(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100):.1f}% of total item-level gross profit.",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items.iloc[0]['item_name'] if len(top_items) > 0 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_item_1_gross_profit_sar": {
                    "value": round(top_items.iloc[0]['total_gross_profit'], 2) if len(top_items) > 0 else None,
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[0]['total_gross_profit'], 2) if len(top_items) > 0 else None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_item_1_margin_pct": {
                    "value": round(top_items.iloc[0]['gross_margin_pct'], 2) if len(top_items) > 0 else None,
                    "unit": "%",
                    "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2) if len(top_items) > 0 else None,
                    "denominator": 100,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_5_total_gross_profit_sar": {
                    "value": round(top_items['total_gross_profit'].sum(), 2),
                    "unit": "SAR",
                    "numerator": round(top_items['total_gross_profit'].sum(), 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_5_contribution_pct": {
                    "value": round(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100, 1),
                    "unit": "%",
                    "numerator": round(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100, 1),
                    "denominator": 100,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_items_analyzed": {
                    "value": len(item_economics),
                    "unit": None,
                    "numerator": len(item_economics),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_with_cost_no_refund),
            "coverage_notes": [
                "Analysis period: 2026-02-16 to 2026-02-23",
                "Refunds excluded from calculations",
                "Only items with known SKU and menu cost data included",
                f"Total non-refund transactions analyzed: {len(pos_with_cost_no_refund)}"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "Line totals are accurate and consistent",
                "No recipe/BOM data available; using menu-level unit costs only"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
# Calculate known waste costs from inventory data
if len(inventory_analysis) > 0:
    inventory_analysis['known_waste_cost_sar'] = pd.to_numeric(inventory_analysis['known_waste_cost_sar'], errors='coerce')
    
    # Filter for non-null waste costs
    waste_data = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_data) > 0:
        total_waste_cost = waste_data['known_waste_cost_sar'].sum()
        waste_by_item = waste_data.groupby('item').agg({
            'known_waste_cost_sar': 'sum',
            'units_wasted': 'sum'
        }).reset_index()
        waste_by_item = waste_by_item.sort_values('known_waste_cost_sar', ascending=False)
        
        finding_2 = {
            "title": "Quantified Waste Cost (Analysis Week)",
            "claim": f"During the analysis week, quantified waste cost totaled {total_waste_cost:.2f} SAR across {len(waste_data)} inventory records with non-null waste observations.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "waste_records_with_data": {
                    "value": len(waste_data),
                    "unit": None,
                    "numerator": len(waste_data),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_waste_item": {
                    "value": waste_by_item.iloc[0]['item'] if len(waste_by_item) > 0 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_waste_item_cost_sar": {
                    "value": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2) if len(waste_by_item) > 0 else None,
                    "unit": "SAR",
                    "numerator": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2) if len(waste_by_item) > 0 else None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_data),
            "coverage_notes": [
                "Analysis period: 2026-02-16 to 2026-02-23",
                "Only non-null waste cost observations included",
                f"Total inventory records in period: {len(inventory_analysis)}",
                f"Records with quantified waste: {len(waste_data)}"
            ],
            "assumptions": [
                "Blank waste values are treated as unknown, not zero",
                "known_waste_cost_sar represents actual waste cost incurred"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Extract supplier price changes with effective dates
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date_dt'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        price_changes['price_change_sar'] = price_changes['new_price'] - price_changes['old_price']
        price_changes['price_change_pct'] = (price_changes['price_change_sar'] / price_changes['old_price'] * 100).round(2)
        
        # Sort by effective date
        price_changes = price_changes.sort_values('effective_date_dt')
        
        # Get the most recent price change
        latest_change = price_changes.iloc[-1]
        
        finding_3 = {
            "title": "Supplier Price Change Evidence",
            "claim": f"Email evidence documents a price change for {latest_change['entity_or_ingredient']} effective {latest_change['effective_date']}: old price {latest_change['old_price']} {latest_change['currency']}/{latest_change['unit']} → new price {latest_change['new_price']} {latest_change['currency']}/{latest_change['unit']}, representing a {latest_change['price_change_pct']:.2f}% change.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient_or_entity": {
                    "value": latest_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "old_price": {
                    "value": round(latest_change['old_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['old_price'], 2),
                    "denominator": None,
                    "period_start": str(latest_change['date']),
                    "period_end": str(latest_change['effective_date'])
                },
                "new_price": {
                    "value": round(latest_change['new_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['new_price'], 2),
                    "denominator": None,
                    "period_start": str(latest_change['effective_date']),
                    "period_end": analysis_end
                },
                "price_change_sar": {
                    "value": round(latest_change['price_change_sar'], 2),
                    "unit": latest_change['currency'],
                    "numerator": round(latest_change['price_change_sar'], 2),
                    "denominator": None,
                    "period_start": str(latest_change['date']),
                    "period_end": str(latest_change['effective_date'])
                },
                "price_change_pct": {
                    "value": round(latest_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(latest_change['price_change_pct'], 2),
                    "denominator": 100,
                    "period_start": str(latest_change['date']),
                    "period_end": str(latest_change['effective_date'])
                },
                "effective_date": {
                    "value": str(latest_change['effective_date']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": str(latest_change['date']),
                    "period_end": analysis_end
                },
                "email_confidence": {
                    "value": latest_change['confidence'],
                    "unit": None,
                    "numerator": latest_change['confidence'],
                    "denominator": 1,
                    "period_start": str(latest_change['date']),
                    "period_end": analysis_end
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Total price change records in emails: {len(price_changes)}",
                f"Email extraction confidence: {latest_change['confidence']}",
                f"Email date: {latest_change['date']}",
                f"Effective date: {latest_change['effective_date']}"
            ],
            "assumptions": [
                "Email extraction is accurate for price and unit information",
                "No recipe/BOM data available; cannot calculate per-drink impact without standing order quantities",
                "Price change applies only to the specified ingredient/entity",
                "Continued order volume and payment terms are not confirmed"
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
