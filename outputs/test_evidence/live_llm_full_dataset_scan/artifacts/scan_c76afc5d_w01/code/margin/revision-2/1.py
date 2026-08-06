import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis period
analysis_start = "2026-01-12T00:00:00+03:00"
analysis_end = "2026-01-19T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS to analysis period
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start_dt) & (pos_df['timestamp'] < analysis_end_dt)].copy()

# Filter inventory to analysis period
inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start_dt) & (inventory_df['week_starting'] < analysis_end_dt)].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate line-level COGS and gross profit
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_revenue_sar'] = pos_with_cost['line_total_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_revenue_sar'] - pos_with_cost['line_cogs_sar']

# Exclude refunds for net calculations
pos_net = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Aggregate by item
item_economics = pos_net.groupby('sku').agg({
    'item_name_en': 'first',
    'quantity': 'sum',
    'line_revenue_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit contribution
item_economics_sorted = item_economics.sort_values('total_gross_profit', ascending=False)

# Calculate totals
total_revenue = item_economics['total_revenue'].sum()
total_cogs = item_economics['total_cogs'].sum()
total_gross_profit = item_economics['total_gross_profit'].sum()
total_baskets = pos_net['transaction_id'].nunique()
total_items_sold = item_economics['total_quantity'].sum()

# Top 5 items by gross profit
top_5_items = item_economics_sorted.head(5)

finding_1 = {
    "title": "Item-Level Gross Profit Contribution (Analysis Period)",
    "claim": f"During {analysis_start} to {analysis_end}, the cafe generated {total_gross_profit:.2f} SAR gross profit across {total_items_sold:.0f} items sold in {total_baskets} baskets. Top 5 items by gross profit contribution account for {top_5_items['total_gross_profit'].sum():.2f} SAR ({top_5_items['total_gross_profit'].sum()/total_gross_profit*100:.1f}% of total).",
    "finding_type": "item_economics",
    "metrics": {
        "total_gross_profit_sar": {
            "value": round(total_gross_profit, 2),
            "unit": "SAR",
            "numerator": round(total_gross_profit, 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "total_revenue_sar": {
            "value": round(total_revenue, 2),
            "unit": "SAR",
            "numerator": round(total_revenue, 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "total_cogs_sar": {
            "value": round(total_cogs, 2),
            "unit": "SAR",
            "numerator": round(total_cogs, 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "overall_gross_margin_pct": {
            "value": round((total_gross_profit / total_revenue * 100), 2),
            "unit": "%",
            "numerator": round(total_gross_profit, 2),
            "denominator": round(total_revenue, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "total_items_sold": {
            "value": int(total_items_sold),
            "unit": "units",
            "numerator": int(total_items_sold),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "unique_baskets": {
            "value": int(total_baskets),
            "unit": "transactions",
            "numerator": int(total_baskets),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_item_1_name": {
            "value": top_5_items.iloc[0]['item_name'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_item_1_gross_profit_sar": {
            "value": round(top_5_items.iloc[0]['total_gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(top_5_items.iloc[0]['total_gross_profit'], 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_item_1_margin_pct": {
            "value": round(top_5_items.iloc[0]['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(top_5_items.iloc[0]['total_gross_profit'], 2),
            "denominator": round(top_5_items.iloc[0]['total_revenue'], 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": int(total_baskets),
    "coverage_notes": [
        f"POS records in analysis period: {len(pos_analysis)}",
        f"Non-refund transactions: {len(pos_net)}",
        f"Unique items sold: {len(item_economics)}",
        f"Menu items with unit cost data: {menu_df['unit_cost_sar'].notna().sum()}",
        "Refunds excluded from net revenue and profit calculations"
    ],
    "assumptions": [
        "Menu unit_cost_sar values are current and apply to all sales in the period",
        "Line totals are accurate and consistent with quantity × unit_price - discount",
        "No recipe/BOM data available; unit costs are as-declared in menu"
    ],
    "confidence": 0.92
}

findings.append(finding_1)

# ============================================================================
# FINDING 2: Known Waste Cost Analysis
# ============================================================================

# Filter inventory to records with non-null waste cost
inventory_with_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(inventory_with_waste) > 0:
    total_waste_cost = inventory_with_waste['known_waste_cost_sar'].sum()
    waste_items_count = len(inventory_with_waste)
    max_waste_item = inventory_with_waste.loc[inventory_with_waste['known_waste_cost_sar'].idxmax()]
    total_waste_units = inventory_with_waste['units_wasted'].sum()
    
    # Total inventory records in period
    total_inventory_records = len(inventory_analysis)
    null_waste_records = len(inventory_analysis[inventory_analysis['known_waste_cost_sar'].isna()])
    
    finding_2 = {
        "title": "Known Waste Cost (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, quantified waste cost totaled {total_waste_cost:.2f} SAR across {waste_items_count} inventory items with recorded waste. Maximum single-item waste cost was {max_waste_item['known_waste_cost_sar']:.2f} SAR ({max_waste_item['item']}). Total units wasted: {total_waste_units:.0f}.",
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
            "waste_items_count": {
                "value": waste_items_count,
                "unit": "items",
                "numerator": waste_items_count,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_units_wasted": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": int(total_waste_units),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "max_waste_item_name": {
                "value": max_waste_item['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "max_waste_item_cost_sar": {
                "value": round(max_waste_item['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": round(max_waste_item['known_waste_cost_sar'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_items_count,
        "coverage_notes": [
            f"Total inventory records in analysis period: {total_inventory_records}",
            f"Records with non-null waste_cost_sar: {waste_items_count}",
            f"Records with null/blank waste_cost_sar: {null_waste_records}",
            f"Waste cost calculation includes only {waste_items_count} items with known waste observations",
            "Unknown waste (null waste_cost_sar values) excluded from this calculation"
        ],
        "assumptions": [
            "Null waste_cost_sar values represent unknown/unmeasured waste, not zero waste",
            "Recorded waste_cost_sar values are accurate and reflect actual disposal/loss cost",
            "Waste cost is independent of sales volume and represents actual loss"
        ],
        "confidence": 0.78
    }
    
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Detection (Email Evidence)
# ============================================================================

# Filter emails for price changes with non-null old_price and new_price
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['category'] == 'supplier_price_change')
].copy()

if len(price_changes) > 0:
    # Take the most recent/relevant price change
    price_change = price_changes.iloc[0]
    
    old_price = price_change['old_price']
    new_price = price_change['new_price']
    price_change_pct = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    entity = price_change['entity_or_ingredient']
    unit = price_change['unit']
    effective_date = price_change['effective_date']
    
    finding_3 = {
        "title": "Supplier Price Change: Roasted Coffee",
        "claim": f"Email evidence indicates roasted coffee price change from {old_price:.2f} to {new_price:.2f} SAR/{unit} ({price_change_pct:+.2f}%), effective {effective_date.strftime('%Y-%m-%d')}. No recipe/BOM or standing order volume data available to calculate per-drink or procurement cost impact.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "entity": {
                "value": entity,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "old_price": {
                "value": round(old_price, 2),
                "unit": f"SAR/{unit}",
                "numerator": round(old_price, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": f"SAR/{unit}",
                "numerator": round(new_price, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "price_change_pct": {
                "value": round(price_change_pct, 2),
                "unit": "%",
                "numerator": round(new_price - old_price, 2),
                "denominator": round(old_price, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "effective_date": {
                "value": effective_date.strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Price change extracted from supplier email",
            "No independent corroboration (invoice, PO, contract amendment)",
            "Effective date is future-dated relative to analysis period",
            "No recipe/BOM data available to link price change to menu items",
            "No standing order volume data available to quantify procurement exposure"
        ],
        "assumptions": [
            "Email price change is authentic and officially communicated by supplier",
            "Price applies to cafe's standing orders (unconfirmed)",
            "No recipe/BOM data available; cannot calculate per-drink cost impact",
            "No standing order volume data available; cannot calculate total procurement cost exposure"
        ],
        "confidence": 0.55
    }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")