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

# Define analysis periods
analysis_period = {
    "start": "2026-06-15T00:00:00+03:00",
    "end": "2026-06-22T00:00:00+03:00"
}
previous_period = {
    "start": "2026-06-08T00:00:00+03:00",
    "end": "2026-06-15T00:00:00+03:00"
}

# Parse dates
analysis_start = pd.to_datetime(analysis_period["start"])
analysis_end = pd.to_datetime(analysis_period["end"])
previous_start = pd.to_datetime(previous_period["start"])
previous_end = pd.to_datetime(previous_period["end"])

# Convert POS timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Item-level revenue and quantity analysis (no COGS claims)
# ============================================================================

# Calculate item-level metrics for analysis period
item_metrics = pos_analysis[pos_analysis['is_refund'] == False].groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique',
    'unit_price_sar': 'mean'
}).reset_index()

item_metrics.columns = ['sku', 'total_quantity', 'total_revenue', 'basket_count', 'avg_unit_price']

# Merge with menu to get item names and unit costs
item_metrics = item_metrics.merge(menu_df[['sku', 'item_en', 'unit_cost_sar']], on='sku', how='left')

# Sort by revenue
item_metrics = item_metrics.sort_values('total_revenue', ascending=False)

# Get top 3 items by revenue
top_items = item_metrics.head(3)

if len(top_items) > 0:
    # Build metrics dictionary for top revenue items
    metrics_dict = {}
    
    for idx, row in top_items.iterrows():
        item_num = idx + 1
        metrics_dict[f'item_{item_num}_sku'] = {
            'value': row['sku'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict[f'item_{item_num}_name'] = {
            'value': row['item_en'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict[f'item_{item_num}_quantity'] = {
            'value': float(row['total_quantity']),
            'unit': 'units',
            'numerator': float(row['total_quantity']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict[f'item_{item_num}_revenue'] = {
            'value': float(row['total_revenue']),
            'unit': 'SAR',
            'numerator': float(row['total_revenue']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict[f'item_{item_num}_baskets'] = {
            'value': int(row['basket_count']),
            'unit': 'transactions',
            'numerator': int(row['basket_count']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict[f'item_{item_num}_avg_price'] = {
            'value': float(row['avg_unit_price']),
            'unit': 'SAR',
            'numerator': float(row['avg_unit_price']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
    
    finding_1 = {
        'title': 'Top Revenue-Generating Items (Analysis Week)',
        'claim': f'The three highest-revenue items in the analysis period (2026-06-15 to 2026-06-22) generated {float(top_items["total_revenue"].sum()):.2f} SAR across {int(top_items["total_quantity"].sum())} units sold in {int(top_items["basket_count"].sum())} transactions.',
        'finding_type': 'revenue_analysis',
        'metrics': metrics_dict,
        'source_names': ['pos'],
        'sample_size': int(pos_analysis[pos_analysis['is_refund'] == False].shape[0]),
        'coverage_notes': [
            'Refunds excluded from calculation',
            'Analysis period: 2026-06-15 to 2026-06-22',
            'Only non-refund transactions included'
        ],
        'assumptions': [
            'POS line_total_sar represents actual revenue received',
            'quantity field represents units sold',
            'transaction_id uniquely identifies baskets'
        ],
        'confidence': 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste cost quantification (only non-null waste)
# ============================================================================

# Filter inventory for analysis week
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-06-15'].copy()

# Only include rows with non-null waste cost
waste_data = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_data) > 0:
    total_waste_cost = waste_data['known_waste_cost_sar'].sum()
    waste_items = waste_data[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    metrics_dict_2 = {
        'total_waste_cost_sar': {
            'value': float(total_waste_cost),
            'unit': 'SAR',
            'numerator': float(total_waste_cost),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        },
        'waste_item_count': {
            'value': len(waste_data),
            'unit': 'items',
            'numerator': len(waste_data),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
    }
    
    # Add details for each waste item
    for idx, (_, row) in enumerate(waste_items.iterrows(), 1):
        metrics_dict_2[f'waste_item_{idx}_name'] = {
            'value': row['item'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_2[f'waste_item_{idx}_units'] = {
            'value': float(row['units_wasted']) if pd.notna(row['units_wasted']) else None,
            'unit': 'units',
            'numerator': float(row['units_wasted']) if pd.notna(row['units_wasted']) else None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_2[f'waste_item_{idx}_cost'] = {
            'value': float(row['known_waste_cost_sar']),
            'unit': 'SAR',
            'numerator': float(row['known_waste_cost_sar']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
    
    finding_2 = {
        'title': 'Quantified Waste Cost (Analysis Week)',
        'claim': f'Documented waste cost for the analysis week (2026-06-15 to 2026-06-22) totaled {float(total_waste_cost):.2f} SAR across {len(waste_data)} items with recorded waste observations.',
        'finding_type': 'waste_cost_analysis',
        'metrics': metrics_dict_2,
        'source_names': ['inventory'],
        'sample_size': len(waste_data),
        'coverage_notes': [
            'Only non-null waste_cost_sar values included',
            'Analysis period: week starting 2026-06-15',
            'Blank waste observations excluded per methodology'
        ],
        'assumptions': [
            'known_waste_cost_sar field represents actual waste cost',
            'Waste observations are complete for recorded items'
        ],
        'confidence': 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier price changes with effective dates
# ============================================================================

# Filter emails for price changes with valid dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Parse effective dates
    price_changes['effective_date_parsed'] = pd.to_datetime(price_changes['effective_date'], errors='coerce')
    
    # Calculate percentage change
    price_changes['pct_change'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    
    # Sort by absolute percentage change
    price_changes['abs_pct_change'] = price_changes['pct_change'].abs()
    price_changes = price_changes.sort_values('abs_pct_change', ascending=False)
    
    # Take top 3 price changes
    top_changes = price_changes.head(3)
    
    metrics_dict_3 = {
        'total_price_changes': {
            'value': len(price_changes),
            'unit': 'changes',
            'numerator': len(price_changes),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
    }
    
    # Add details for each price change
    for idx, (_, row) in enumerate(top_changes.iterrows(), 1):
        metrics_dict_3[f'price_change_{idx}_ingredient'] = {
            'value': row['entity_or_ingredient'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_old_price'] = {
            'value': float(row['old_price']),
            'unit': row['currency'],
            'numerator': float(row['old_price']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_new_price'] = {
            'value': float(row['new_price']),
            'unit': row['currency'],
            'numerator': float(row['new_price']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_pct_change'] = {
            'value': float(row['pct_change']),
            'unit': '%',
            'numerator': float(row['pct_change']),
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_unit'] = {
            'value': row['unit'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_effective_date'] = {
            'value': row['effective_date'],
            'unit': None,
            'numerator': None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
        metrics_dict_3[f'price_change_{idx}_confidence'] = {
            'value': float(row['confidence']) if pd.notna(row['confidence']) else None,
            'unit': None,
            'numerator': float(row['confidence']) if pd.notna(row['confidence']) else None,
            'denominator': None,
            'period_start': analysis_period['start'],
            'period_end': analysis_period['end']
        }
    
    # Build claim text
    claim_text = f'Supplier email analysis identified {len(price_changes)} price changes with effective dates. '
    if len(top_changes) > 0:
        largest_change = top_changes.iloc[0]
        claim_text += f'The largest change: {largest_change["entity_or_ingredient"]} from {largest_change["old_price"]} to {largest_change["new_price"]} {largest_change["currency"]}/{largest_change["unit"]} ({largest_change["pct_change"]:+.1f}%) effective {largest_change["effective_date"]}.'
    
    finding_3 = {
        'title': 'Supplier Price Changes (Email Evidence)',
        'claim': claim_text,
        'finding_type': 'supplier_price_change',
        'metrics': metrics_dict_3,
        'source_names': ['emails'],
        'sample_size': len(price_changes),
        'coverage_notes': [
            'Only price changes with both old_price, new_price, and effective_date included',
            'Analysis period: 2026-06-15 to 2026-06-22',
            'Email extraction confidence scores included where available'
        ],
        'assumptions': [
            'Email extraction accurately identified price changes',
            'Effective dates represent when price changes took effect',
            'Currency and unit fields are correctly populated'
        ],
        'confidence': 0.85
    }
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
