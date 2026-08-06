import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load run metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis periods
analysis_start = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-27T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-13T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-20T00:00:00+03:00")

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter POS for analysis period (exclude refunds for revenue, but include for volume analysis)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit for top revenue items
# ============================================================================

# Calculate item-level metrics for analysis period (excluding refunds from revenue)
pos_sales = pos_analysis[pos_analysis['is_refund'] == False].copy()

item_metrics = {}
for sku in pos_sales['sku'].unique():
    sku_data = pos_sales[pos_sales['sku'] == sku]
    item_name = sku_data['item_name_en'].iloc[0] if len(sku_data) > 0 else sku
    
    total_quantity = sku_data['quantity'].sum()
    total_revenue = sku_data['line_total_sar'].sum()
    
    # Get unit cost from menu
    menu_row = menu_df[menu_df['sku'] == sku]
    if len(menu_row) > 0:
        unit_cost = menu_row['unit_cost_sar'].iloc[0]
    else:
        unit_cost = None
    
    if unit_cost is not None and not pd.isna(unit_cost):
        cogs = total_quantity * unit_cost
        gross_profit = total_revenue - cogs
        gross_margin_pct = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
    else:
        cogs = None
        gross_profit = None
        gross_margin_pct = None
    
    item_metrics[sku] = {
        'item_name': item_name,
        'quantity': total_quantity,
        'revenue': total_revenue,
        'unit_cost': unit_cost,
        'cogs': cogs,
        'gross_profit': gross_profit,
        'gross_margin_pct': gross_margin_pct,
        'transaction_count': sku_data['transaction_id'].nunique()
    }

# Sort by revenue and get top 3
sorted_items = sorted(item_metrics.items(), key=lambda x: x[1]['revenue'], reverse=True)
top_items = sorted_items[:3]

# Build finding for top revenue item with complete COGS data
for sku, metrics in top_items:
    if metrics['cogs'] is not None and metrics['gross_profit'] is not None:
        finding = {
            "title": f"{metrics['item_name']}: {metrics['gross_margin_pct']:.2f}% gross margin",
            "claim": f"Item {metrics['item_name']} (SKU: {sku}) generated {metrics['revenue']:.2f} SAR revenue with {metrics['cogs']:.2f} SAR COGS and {metrics['gross_profit']:.2f} SAR gross profit ({metrics['gross_margin_pct']:.2f}% margin) during analysis period.",
            "finding_type": "item_economics",
            "metrics": {
                "revenue_sar": {
                    "value": round(metrics['revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "quantity_units": {
                    "value": int(metrics['quantity']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "unit_cost_sar": {
                    "value": round(metrics['unit_cost'], 2),
                    "unit": "SAR/unit",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "cogs_sar": {
                    "value": round(metrics['cogs'], 2),
                    "unit": "SAR",
                    "numerator": int(metrics['quantity']),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "gross_profit_sar": {
                    "value": round(metrics['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "gross_margin_pct": {
                    "value": round(metrics['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(metrics['gross_profit'], 2),
                    "denominator": round(metrics['revenue'], 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": metrics['transaction_count'],
            "coverage_notes": [
                f"Analysis period: 2026-04-20 to 2026-04-27 (7 days)",
                f"Item {sku} had {metrics['transaction_count']} transactions with {int(metrics['quantity'])} units sold",
                "Refunds excluded from revenue calculation",
                "Unit cost sourced from menu_items.unit_cost_sar"
            ],
            "assumptions": [
                "Analysis uses menu-defined unit costs (SAR per unit); actual COGS may vary due to portion control, ingredient waste, or supplier cost changes not captured in menu data",
                "No recipe/BOM available; using menu-level unit costs",
                "This is a menu-level margin analysis; item-level supplier cost optimization would require recipe/BOM evidence"
            ],
            "confidence": 0.95
        }
        findings.append(finding)
        break  # Only include top item with complete data

# ============================================================================
# FINDING 2: Supplier price changes from emails with impact scenario
# ============================================================================

# Filter emails with price changes and valid effective dates
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) & 
    (emails_df['effective_date'].notna())
].copy()

if len(price_change_emails) > 0:
    for idx, email_row in price_change_emails.iterrows():
        entity = email_row['entity_or_ingredient']
        old_price = email_row['old_price']
        new_price = email_row['new_price']
        currency = email_row['currency']
        unit = email_row['unit']
        effective_date = email_row['effective_date']
        
        # Calculate price delta
        price_delta = new_price - old_price
        price_delta_pct = (price_delta / old_price * 100) if old_price != 0 else 0
        
        # Check if effective date falls within or near analysis period
        # Use effective date as the period reference
        effective_iso = effective_date.isoformat()
        
        finding = {
            "title": f"Supplier price change: {entity} {price_delta_pct:+.1f}%",
            "claim": f"Supplier email dated {email_row['date'].strftime('%Y-%m-%d')} reports {entity} price change from {old_price} {currency}/{unit} to {new_price} {currency}/{unit} (effective {effective_date.strftime('%Y-%m-%d')}), representing a {price_delta_pct:+.2f}% change.",
            "finding_type": "supplier_cost_change",
            "metrics": {
                "old_price": {
                    "value": round(old_price, 2),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": effective_iso,
                    "period_end": effective_iso
                },
                "new_price": {
                    "value": round(new_price, 2),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": effective_iso,
                    "period_end": effective_iso
                },
                "price_delta": {
                    "value": round(price_delta, 2),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": effective_iso,
                    "period_end": effective_iso
                },
                "price_delta_pct": {
                    "value": round(price_delta_pct, 2),
                    "unit": "%",
                    "numerator": round(price_delta, 2),
                    "denominator": round(old_price, 2),
                    "period_start": effective_iso,
                    "period_end": effective_iso
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Supplier email dated {email_row['date'].strftime('%Y-%m-%d')}",
                f"Effective date: {effective_date.strftime('%Y-%m-%d')}",
                f"Entity/ingredient: {entity}",
                f"Extraction confidence: {email_row['confidence']}"
            ],
            "assumptions": [
                "Price change applies to future procurement; impact on current period depends on inventory timing and payment terms",
                "No standing order quantity or payment terms confirmed in email; actual margin impact requires volume and timing data"
            ],
            "confidence": email_row['confidence'] if pd.notna(email_row['confidence']) else 0.7
        }
        findings.append(finding)
        if len(findings) >= 3:
            break

# ============================================================================
# FINDING 3: Waste cost quantification for items with known waste
# ============================================================================

# Filter inventory for analysis period week
analysis_week = pd.Timestamp("2026-04-20", tz=timezone.utc)
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

if len(inventory_analysis) > 0:
    waste_items = inventory_analysis[
        (inventory_analysis['units_wasted'].notna()) & 
        (inventory_analysis['units_wasted'] > 0) &
        (inventory_analysis['known_waste_cost_sar'].notna()) &
        (inventory_analysis['known_waste_cost_sar'] > 0)
    ].copy()
    
    if len(waste_items) > 0:
        # Sort by waste cost and get top item
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        top_waste = waste_items.iloc[0]
        
        sku = top_waste['sku']
        item_name = top_waste['item']
        units_wasted = top_waste['units_wasted']
        waste_cost = top_waste['known_waste_cost_sar']
        unit_cost = top_waste['unit_cost_sar']
        units_sold = top_waste['units_sold']
        
        # Calculate waste as % of total units
        total_units = units_sold + units_wasted if pd.notna(units_sold) else units_wasted
        waste_pct = (units_wasted / total_units * 100) if total_units > 0 else 0
        
        finding = {
            "title": f"{item_name}: {waste_cost:.2f} SAR waste cost ({waste_pct:.1f}% of units)",
            "claim": f"Item {item_name} (SKU: {sku}) incurred {waste_cost:.2f} SAR in known waste cost during week of 2026-04-20, representing {units_wasted} wasted units at {unit_cost:.2f} SAR/unit ({waste_pct:.2f}% of total units handled).",
            "finding_type": "waste_cost",
            "metrics": {
                "units_wasted": {
                    "value": int(units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_cost_sar": {
                    "value": round(waste_cost, 2),
                    "unit": "SAR",
                    "numerator": int(units_wasted),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "unit_cost_sar": {
                    "value": round(unit_cost, 2),
                    "unit": "SAR/unit",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_pct_of_units": {
                    "value": round(waste_pct, 2),
                    "unit": "%",
                    "numerator": int(units_wasted),
                    "denominator": int(total_units),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": 1,
            "coverage_notes": [
                "Week starting 2026-04-20",
                "Known waste cost sourced from inventory.known_waste_cost_sar",
                "Waste units and unit cost from inventory records",
                "Only items with non-null waste cost included"
            ],
            "assumptions": [
                "Waste cost reflects actual observed waste, not estimated",
                "Unit cost is consistent with inventory records",
                "Waste occurred during the week; no timing detail available"
            ],
            "confidence": 0.9
        }
        findings.append(finding)

# ============================================================================
# Write output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Max 3 findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)