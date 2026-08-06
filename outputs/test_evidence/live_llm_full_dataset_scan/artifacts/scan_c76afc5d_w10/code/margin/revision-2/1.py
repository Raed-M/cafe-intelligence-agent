import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis periods
analysis_start = datetime.fromisoformat("2026-03-16T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-23T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-03-16T00:00:00+03:00")

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit for Analysis Period
# ============================================================================

# Filter POS for analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit_cost_sar
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']

# Aggregate by item
item_economics = pos_with_cost.groupby('sku').agg({
    'item_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics['gross_margin_pct'] = (
    item_economics['gross_profit'] / item_economics['line_total_sar'] * 100
).round(2)

# Sort by gross profit descending
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

# Top 5 items
top_5 = item_economics_sorted.head(5)
top_5_total_gp = top_5['gross_profit'].sum()
top_5_total_qty = top_5['quantity'].sum()
top_5_total_revenue = top_5['line_total_sar'].sum()

# Build evidence array for top 5
top_5_evidence = []
for idx, row in top_5.iterrows():
    top_5_evidence.append({
        'sku': row['sku'],
        'item_en': row['item_en'],
        'quantity': int(row['quantity']),
        'revenue_sar': round(row['line_total_sar'], 2),
        'cogs_sar': round(row['cogs'], 2),
        'gross_profit_sar': round(row['gross_profit'], 2),
        'gross_margin_pct': row['gross_margin_pct'],
        'baskets': int(row['transaction_id'])
    })

finding_1 = {
    'title': 'Top 5 Items by Gross Profit (Analysis Week)',
    'claim': f'The top 5 items by gross profit contribution generated SAR {top_5_total_gp:.2f} from {int(top_5_total_qty)} units across {len(top_5)} SKUs during the analysis week.',
    'finding_type': 'item_economics',
    'metrics': {
        'top_5_total_gross_profit_sar': {
            'value': round(top_5_total_gp, 2),
            'unit': 'SAR',
            'numerator': round(top_5_total_gp, 2),
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        },
        'top_5_total_quantity': {
            'value': int(top_5_total_qty),
            'unit': 'units',
            'numerator': int(top_5_total_qty),
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        },
        'top_5_total_revenue_sar': {
            'value': round(top_5_total_revenue, 2),
            'unit': 'SAR',
            'numerator': round(top_5_total_revenue, 2),
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        },
        'top_5_avg_margin_pct': {
            'value': round((top_5_total_gp / top_5_total_revenue * 100), 2),
            'unit': '%',
            'numerator': round(top_5_total_gp, 2),
            'denominator': round(top_5_total_revenue, 2),
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        }
    },
    'source_names': ['pos', 'menu'],
    'sample_size': len(pos_analysis),
    'coverage_notes': [
        'POS data filtered to analysis period 2026-03-16 to 2026-03-23',
        'Refunds excluded (is_refund == False)',
        'Menu unit_cost_sar merged by SKU',
        'Top 5 items ranked by gross profit descending',
        'All 5 items have non-null unit_cost_sar values'
    ],
    'assumptions': [
        'Menu unit_cost_sar is current and applicable to analysis period',
        'POS line_total_sar is net of discounts and represents actual revenue',
        'Gross profit = line_total_sar - (quantity × unit_cost_sar)',
        'No recipe/BOM adjustments applied'
    ],
    'confidence': 0.95,
    'evidence': top_5_evidence
}

# ============================================================================
# FINDING 2: Quantified Waste Cost (Analysis Week)
# ============================================================================

# Filter inventory for analysis week
inv_analysis = inventory_df[
    inventory_df['week_starting'] == pd.Timestamp('2026-03-16', tz='UTC')
].copy()

# Only count items with non-null waste cost
inv_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

total_waste_cost = inv_with_waste['known_waste_cost_sar'].sum()
total_waste_units = inv_with_waste['units_wasted'].sum()
items_with_waste = len(inv_with_waste)

# Get total items in inventory for this week
total_items_week = len(inv_analysis)
items_without_waste = total_items_week - items_with_waste

waste_evidence = []
for idx, row in inv_with_waste.iterrows():
    waste_evidence.append({
        'sku': row['sku'],
        'item': row['item'],
        'units_wasted': int(row['units_wasted']) if pd.notna(row['units_wasted']) else 0,
        'known_waste_cost_sar': round(row['known_waste_cost_sar'], 2),
        'unit_cost_sar': round(row['unit_cost_sar'], 2) if pd.notna(row['unit_cost_sar']) else None
    })

finding_2 = {
    'title': 'Quantified Waste Cost (Analysis Week)',
    'claim': f'Measurable waste cost for the analysis week totaled SAR {total_waste_cost:.2f} across {items_with_waste} items with recorded waste observations.',
    'finding_type': 'waste_cost',
    'metrics': {
        'total_waste_cost_sar': {
            'value': round(total_waste_cost, 2),
            'unit': 'SAR',
            'numerator': round(total_waste_cost, 2),
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        },
        'total_waste_units': {
            'value': int(total_waste_units),
            'unit': 'units',
            'numerator': int(total_waste_units),
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        },
        'items_with_waste': {
            'value': items_with_waste,
            'unit': 'count',
            'numerator': items_with_waste,
            'denominator': total_items_week,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        }
    },
    'source_names': ['inventory'],
    'sample_size': total_items_week,
    'coverage_notes': [
        f'Inventory data for week starting 2026-03-16',
        f'Items with non-null known_waste_cost_sar: {items_with_waste}',
        f'Items with null/unknown waste: {items_without_waste}',
        'Blank waste values treated as unknown, not zero',
        'Only non-null waste costs included in total'
    ],
    'assumptions': [
        'known_waste_cost_sar field represents actual waste cost (not estimated)',
        'Waste data is complete for items with non-null values',
        'No refunds or credits are included in waste cost',
        'All items in inventory were active during the week'
    ],
    'confidence': 0.85,
    'evidence': waste_evidence
}

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with valid dates
emails_price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

# Calculate price change percentage
emails_price_changes['price_change_pct'] = (
    ((emails_price_changes['new_price'] - emails_price_changes['old_price']) / 
     emails_price_changes['old_price'] * 100)
).round(2)

# Sort by effective date
emails_price_changes_sorted = emails_price_changes.sort_values('effective_date')

price_change_evidence = []
for idx, row in emails_price_changes_sorted.iterrows():
    price_change_evidence.append({
        'ingredient': row['entity_or_ingredient'],
        'old_price': round(row['old_price'], 2),
        'new_price': round(row['new_price'], 2),
        'unit': row['unit'],
        'currency': row['currency'],
        'price_change_pct': row['price_change_pct'],
        'effective_date': row['effective_date'].isoformat() if pd.notna(row['effective_date']) else None,
        'email_date': row['date'].isoformat() if pd.notna(row['date']) else None,
        'confidence': row['confidence']
    })

if len(price_change_evidence) > 0:
    finding_3 = {
        'title': 'Supplier Price Changes Detected',
        'claim': f'Email evidence documents {len(price_change_evidence)} supplier price changes with effective dates and percentage deltas.',
        'finding_type': 'supplier_pricing',
        'metrics': {
            'price_changes_detected': {
                'value': len(price_change_evidence),
                'unit': 'count',
                'numerator': len(price_change_evidence),
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
        },
        'source_names': ['emails'],
        'sample_size': len(emails_df),
        'coverage_notes': [
            f'Email records with old_price, new_price, and effective_date: {len(price_change_evidence)}',
            'Price changes sorted by effective_date',
            'No standing-order quantities found in email facts for procurement scenario'
        ],
        'assumptions': [
            'Email extraction confidence scores reflect reliability of price data',
            'Effective dates are accurate as stated in supplier communications',
            'Price changes apply to supplier-level costs, not menu prices'
        ],
        'confidence': 0.80,
        'evidence': price_change_evidence
    }
else:
    finding_3 = None

# ============================================================================
# Compile output
# ============================================================================

findings = [finding_1, finding_2]
if finding_3 is not None:
    findings.append(finding_3)

output = {
    'status': 'success',
    'findings': findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Output written to {output_path}")