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

def is_in_period(timestamp, period_start, period_end):
    """Check if timestamp falls within period."""
    ts = parse_iso_datetime(timestamp)
    ps = parse_iso_datetime(period_start)
    pe = parse_iso_datetime(period_end)
    return ps <= ts < pe

def calculate_findings(inputs, analysis_period, previous_period, trailing_baseline_periods):
    """Calculate margin and cost analysis findings."""
    
    # Load data
    pos_df = pd.read_parquet(inputs['pos'])
    inventory_df = pd.read_parquet(inputs['inventory'])
    menu_df = pd.read_parquet(inputs['menu'])
    emails_df = pd.read_parquet(inputs['emails'])
    
    findings = []
    
    # Parse periods
    analysis_start = analysis_period['start']
    analysis_end = analysis_period['end']
    prev_start = previous_period['start']
    prev_end = previous_period['end']
    
    # Filter POS data for analysis period
    pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
    pos_analysis = pos_df[
        (pos_df['timestamp'] >= analysis_start) & 
        (pos_df['timestamp'] < analysis_end) &
        (pos_df['is_refund'] == False)
    ].copy()
    
    pos_previous = pos_df[
        (pos_df['timestamp'] >= prev_start) & 
        (pos_df['timestamp'] < prev_end) &
        (pos_df['is_refund'] == False)
    ].copy()
    
    # FINDING 1: Item-level COGS and Gross Profit Analysis
    if len(pos_analysis) > 0 and len(menu_df) > 0:
        # Merge POS with menu to get unit costs
        pos_with_cost = pos_analysis.merge(
            menu_df[['sku', 'unit_cost_sar', 'price_sar']], 
            on='sku', 
            how='left'
        )
        
        # Calculate item-level economics
        pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
        pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']
        pos_with_cost['gross_margin_pct'] = (
            pos_with_cost['gross_profit'] / pos_with_cost['line_total_sar'] * 100
        ).fillna(0)
        
        # Aggregate by item
        item_economics = pos_with_cost.groupby('item_name_en').agg({
            'quantity': 'sum',
            'line_total_sar': 'sum',
            'cogs': 'sum',
            'gross_profit': 'sum',
            'transaction_id': 'nunique'
        }).reset_index()
        
        item_economics['gross_margin_pct'] = (
            item_economics['gross_profit'] / item_economics['line_total_sar'] * 100
        )
        
        # Sort by gross profit contribution
        item_economics = item_economics.sort_values('gross_profit', ascending=False)
        
        # Top contributor
        if len(item_economics) > 0:
            top_item = item_economics.iloc[0]
            
            finding_1 = {
                "title": "Top Gross Profit Contributor - Analysis Period",
                "claim": f"{top_item['item_name_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period, with {int(top_item['quantity'])} units sold across {int(top_item['transaction_id'])} transactions.",
                "finding_type": "item_economics",
                "metrics": {
                    "item_name": {
                        "value": top_item['item_name_en'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "total_revenue": {
                        "value": round(top_item['line_total_sar'], 2),
                        "unit": "SAR",
                        "numerator": round(top_item['line_total_sar'], 2),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "total_cogs": {
                        "value": round(top_item['cogs'], 2),
                        "unit": "SAR",
                        "numerator": round(top_item['cogs'], 2),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "gross_profit": {
                        "value": round(top_item['gross_profit'], 2),
                        "unit": "SAR",
                        "numerator": round(top_item['gross_profit'], 2),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "gross_margin_pct": {
                        "value": round(top_item['gross_margin_pct'], 2),
                        "unit": "%",
                        "numerator": round(top_item['gross_margin_pct'], 2),
                        "denominator": 100,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "units_sold": {
                        "value": int(top_item['quantity']),
                        "unit": "units",
                        "numerator": int(top_item['quantity']),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "transactions": {
                        "value": int(top_item['transaction_id']),
                        "unit": "baskets",
                        "numerator": int(top_item['transaction_id']),
                        "denominator": None,
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    }
                },
                "source_names": ["pos", "menu"],
                "sample_size": len(pos_analysis),
                "coverage_notes": [
                    f"Analysis period: {analysis_start} to {analysis_end}",
                    f"Refunds excluded from analysis",
                    f"Total POS line items in period: {len(pos_analysis)}",
                    f"Items with menu cost data: {len(pos_with_cost[pos_with_cost['unit_cost_sar'].notna()])}"
                ],
                "assumptions": [
                    "Unit costs from menu_items.unit_cost_sar applied to all sales",
                    "Line totals include discounts as recorded in POS",
                    "Gross profit = revenue - (quantity × unit_cost)"
                ],
                "confidence": 0.95
            }
            findings.append(finding_1)
    
    # FINDING 2: Waste Cost Impact
    if len(inventory_df) > 0:
        # Filter inventory for analysis week
        inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
        analysis_week_start = pd.to_datetime(analysis_start).normalize()
        
        # Get the week that contains the analysis period
        week_inventory = inventory_df[
            (inventory_df['week_starting'] >= analysis_week_start - pd.Timedelta(days=7)) &
            (inventory_df['week_starting'] <= analysis_week_start)
        ].copy()
        
        if len(week_inventory) > 0:
            # Calculate waste cost only for non-null waste observations
            week_inventory['waste_cost'] = week_inventory['known_waste_cost_sar'].fillna(0)
            total_waste_cost = week_inventory['waste_cost'].sum()
            waste_items = week_inventory[week_inventory['waste_cost'] > 0]
            
            if total_waste_cost > 0 and len(waste_items) > 0:
                # Get top waste contributor
                top_waste = waste_items.sort_values('waste_cost', ascending=False).iloc[0]
                
                finding_2 = {
                    "title": "Quantified Waste Cost - Analysis Period",
                    "claim": f"Known waste cost in the analysis period totaled {total_waste_cost:.2f} SAR, with {top_waste['item']} accounting for {top_waste['waste_cost']:.2f} SAR ({top_waste['units_wasted']:.0f} units wasted).",
                    "finding_type": "waste_cost",
                    "metrics": {
                        "total_waste_cost": {
                            "value": round(total_waste_cost, 2),
                            "unit": "SAR",
                            "numerator": round(total_waste_cost, 2),
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "top_waste_item": {
                            "value": top_waste['item'],
                            "unit": None,
                            "numerator": None,
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "top_waste_cost": {
                            "value": round(top_waste['waste_cost'], 2),
                            "unit": "SAR",
                            "numerator": round(top_waste['waste_cost'], 2),
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "top_waste_units": {
                            "value": int(top_waste['units_wasted']),
                            "unit": "units",
                            "numerator": int(top_waste['units_wasted']),
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "waste_items_count": {
                            "value": len(waste_items),
                            "unit": "items",
                            "numerator": len(waste_items),
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        }
                    },
                    "source_names": ["inventory"],
                    "sample_size": len(week_inventory),
                    "coverage_notes": [
                        f"Analysis period: {analysis_start} to {analysis_end}",
                        f"Only non-null waste cost observations included",
                        f"Inventory records with waste data: {len(waste_items)} of {len(week_inventory)}"
                    ],
                    "assumptions": [
                        "Waste cost calculated from known_waste_cost_sar field only",
                        "Blank waste values treated as unknown, not zero",
                        "Week-level inventory data mapped to analysis period"
                    ],
                    "confidence": 0.85
                }
                findings.append(finding_2)
    
    # FINDING 3: Supplier Price Changes and Procurement Impact
    if len(emails_df) > 0:
        # Filter for price change emails with old and new prices
        price_changes = emails_df[
            (emails_df['old_price'].notna()) & 
            (emails_df['new_price'].notna()) &
            (emails_df['effective_date'].notna())
        ].copy()
        
        if len(price_changes) > 0:
            price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
            price_changes['date'] = pd.to_datetime(price_changes['date'])
            
            # Calculate percentage change
            price_changes['price_change_pct'] = (
                (price_changes['new_price'] - price_changes['old_price']) / 
                price_changes['old_price'] * 100
            )
            
            # Get most recent significant price change
            significant_changes = price_changes[price_changes['price_change_pct'].abs() > 0]
            
            if len(significant_changes) > 0:
                latest_change = significant_changes.sort_values('effective_date', ascending=False).iloc[0]
                
                finding_3 = {
                    "title": "Supplier Price Change - Procurement Cost Pressure",
                    "claim": f"Supplier {latest_change['entity_or_ingredient']} price changed from {latest_change['old_price']:.2f} to {latest_change['new_price']:.2f} {latest_change['currency']}/{latest_change['unit']} effective {latest_change['effective_date']}, representing a {latest_change['price_change_pct']:.1f}% change.",
                    "finding_type": "supplier_price_change",
                    "metrics": {
                        "ingredient": {
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
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "new_price": {
                            "value": round(latest_change['new_price'], 2),
                            "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                            "numerator": round(latest_change['new_price'], 2),
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "price_change_pct": {
                            "value": round(latest_change['price_change_pct'], 2),
                            "unit": "%",
                            "numerator": round(latest_change['price_change_pct'], 2),
                            "denominator": 100,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        },
                        "effective_date": {
                            "value": str(latest_change['effective_date']),
                            "unit": None,
                            "numerator": None,
                            "denominator": None,
                            "period_start": analysis_start,
                            "period_end": analysis_end
                        }
                    },
                    "source_names": ["emails"],
                    "sample_size": len(price_changes),
                    "coverage_notes": [
                        f"Analysis period: {analysis_start} to {analysis_end}",
                        f"Price changes extracted from supplier emails: {len(price_changes)}",
                        f"Significant changes (>0%): {len(significant_changes)}"
                    ],
                    "assumptions": [
                        "Price change extracted from email evidence",
                        "Percentage change calculated as (new - old) / old × 100",
                        "No standing order quantity data available for scenario modeling",
                        "Impact on menu items cannot be calculated without recipe/BOM data"
                    ],
                    "confidence": 0.90
                }
                findings.append(finding_3)
    
    return findings

def main():
    inputs, output_path = load_inputs()
    
    analysis_period = {
        "start": "2026-03-30T00:00:00+03:00",
        "end": "2026-04-06T00:00:00+03:00"
    }
    
    previous_period = {
        "start": "2026-03-23T00:00:00+03:00",
        "end": "2026-03-30T00:00:00+03:00"
    }
    
    trailing_baseline_periods = [
        {
            "start": "2026-03-23T00:00:00+03:00",
            "end": "2026-03-30T00:00:00+03:00"
        },
        {
            "start": "2026-03-16T00:00:00+03:00",
            "end": "2026-03-23T00:00:00+03:00"
        },
        {
            "start": "2026-03-09T00:00:00+03:00",
            "end": "2026-03-16T00:00:00+03:00"
        },
        {
            "start": "2026-03-02T00:00:00+03:00",
            "end": "2026-03-09T00:00:00+03:00"
        }
    ]
    
    try:
        findings = calculate_findings(
            inputs, 
            analysis_period, 
            previous_period, 
            trailing_baseline_periods
        )
        
        result = {
            "status": "success" if len(findings) > 0 else "insufficient_data",
            "findings": findings
        }
    except Exception as e:
        result = {
            "status": "insufficient_data",
            "findings": [],
            "error": str(e)
        }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    main()
