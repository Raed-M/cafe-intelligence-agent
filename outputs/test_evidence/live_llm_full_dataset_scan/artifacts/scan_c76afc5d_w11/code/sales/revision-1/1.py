import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define periods
analysis_start = pd.Timestamp("2026-03-23T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-03-30T00:00:00+03:00")
previous_start = pd.Timestamp("2026-03-16T00:00:00+03:00")
previous_end = pd.Timestamp("2026-03-23T00:00:00+03:00")

# Convert timestamp to datetime for filtering
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter data for analysis and previous periods
analysis_data = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
previous_data = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)]

# Trailing baseline: average of 4 weeks before analysis period
trailing_data = pos_df[(pos_df['timestamp'] >= pd.Timestamp("2026-02-23T00:00:00+03:00")) & 
                       (pos_df['timestamp'] < previous_start)]

findings = []

# ============================================================================
# FINDING 1: Revenue and Transaction Count Change (Analysis vs Previous Week)
# ============================================================================

# Calculate metrics for analysis period
analysis_revenue = analysis_data[~analysis_data['is_refund']]['line_total_sar'].sum()
analysis_refunds = analysis_data[analysis_data['is_refund']]['line_total_sar'].sum()
analysis_net_revenue = analysis_revenue + analysis_refunds  # refunds are negative
analysis_baskets = analysis_data['transaction_id'].nunique()
analysis_items = len(analysis_data)

# Calculate metrics for previous period
previous_revenue = previous_data[~previous_data['is_refund']]['line_total_sar'].sum()
previous_refunds = previous_data[previous_data['is_refund']]['line_total_sar'].sum()
previous_net_revenue = previous_revenue + previous_refunds
previous_baskets = previous_data['transaction_id'].nunique()
previous_items = len(previous_data)

# Calculate changes
revenue_change_sar = analysis_net_revenue - previous_net_revenue
revenue_change_pct = (revenue_change_sar / previous_net_revenue * 100) if previous_net_revenue != 0 else 0
basket_change = analysis_baskets - previous_baskets
basket_change_pct = (basket_change / previous_baskets * 100) if previous_baskets != 0 else 0

# AOV calculation
analysis_aov = analysis_net_revenue / analysis_baskets if analysis_baskets > 0 else 0
previous_aov = previous_net_revenue / previous_baskets if previous_baskets > 0 else 0
aov_change_sar = analysis_aov - previous_aov
aov_change_pct = (aov_change_sar / previous_aov * 100) if previous_aov != 0 else 0

finding1 = {
    "title": "Net Revenue and Transaction Volume Change (Week of 2026-03-23 vs 2026-03-16)",
    "claim": f"Net revenue decreased by SAR {abs(revenue_change_sar):.2f} ({revenue_change_pct:.2f}%) from SAR {previous_net_revenue:.2f} to SAR {analysis_net_revenue:.2f}. Valid transaction count decreased by {abs(basket_change)} baskets ({basket_change_pct:.2f}%) from {previous_baskets} to {analysis_baskets}. Average order value declined by SAR {abs(aov_change_sar):.2f} ({aov_change_pct:.2f}%) from SAR {previous_aov:.2f} to SAR {analysis_aov:.2f}.",
    "finding_type": "revenue_and_transaction_change",
    "metrics": {
        "analysis_net_revenue_sar": {
            "value": round(analysis_net_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "previous_net_revenue_sar": {
            "value": round(previous_net_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "revenue_change_sar": {
            "value": round(revenue_change_sar, 2),
            "unit": "SAR",
            "numerator": round(revenue_change_sar, 2),
            "denominator": round(previous_net_revenue, 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "revenue_change_pct": {
            "value": round(revenue_change_pct, 2),
            "unit": "%",
            "numerator": round(revenue_change_sar, 2),
            "denominator": round(previous_net_revenue, 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "analysis_baskets": {
            "value": analysis_baskets,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "previous_baskets": {
            "value": previous_baskets,
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "basket_change": {
            "value": basket_change,
            "unit": "count",
            "numerator": basket_change,
            "denominator": previous_baskets,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "basket_change_pct": {
            "value": round(basket_change_pct, 2),
            "unit": "%",
            "numerator": basket_change,
            "denominator": previous_baskets,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "analysis_aov_sar": {
            "value": round(analysis_aov, 2),
            "unit": "SAR",
            "numerator": round(analysis_net_revenue, 2),
            "denominator": analysis_baskets,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "previous_aov_sar": {
            "value": round(previous_aov, 2),
            "unit": "SAR",
            "numerator": round(previous_net_revenue, 2),
            "denominator": previous_baskets,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "aov_change_sar": {
            "value": round(aov_change_sar, 2),
            "unit": "SAR",
            "numerator": round(aov_change_sar, 2),
            "denominator": round(previous_aov, 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "aov_change_pct": {
            "value": round(aov_change_pct, 2),
            "unit": "%",
            "numerator": round(aov_change_sar, 2),
            "denominator": round(previous_aov, 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        f"Analysis period: {len(analysis_data)} line items across {analysis_baskets} unique transactions",
        f"Previous period: {len(previous_data)} line items across {previous_baskets} unique transactions",
        f"Refunds included in net revenue calculations: analysis refunds SAR {analysis_refunds:.2f}, previous refunds SAR {previous_refunds:.2f}"
    ],
    "assumptions": [
        "transaction_id uniqueness defines a basket",
        "line_total_sar includes all discounts and refunds",
        "is_refund flag correctly identifies refund transactions",
        "Refunds are negative values and included in net revenue"
    ],
    "confidence": 0.95
}

findings.append(finding1)

# ============================================================================
# FINDING 2: Category Mix Change (Beverages)
# ============================================================================

# Calculate category revenue for analysis period
analysis_category_revenue = analysis_data.groupby('category')['line_total_sar'].sum()
analysis_total_revenue = analysis_category_revenue.sum()
analysis_category_pct = (analysis_category_revenue / analysis_total_revenue * 100)

# Calculate category revenue for previous period
previous_category_revenue = previous_data.groupby('category')['line_total_sar'].sum()
previous_total_revenue = previous_category_revenue.sum()
previous_category_pct = (previous_category_revenue / previous_total_revenue * 100)

# Find the category with largest mix change
category_mix_changes = {}
for cat in analysis_category_pct.index:
    if cat in previous_category_pct.index:
        current_pct = analysis_category_pct[cat]
        prior_pct = previous_category_pct[cat]
        pct_point_change = current_pct - prior_pct
        category_mix_changes[cat] = {
            'current_pct': current_pct,
            'prior_pct': prior_pct,
            'pct_point_change': pct_point_change,
            'current_revenue': analysis_category_revenue[cat],
            'prior_revenue': previous_category_revenue[cat]
        }

# Find category with largest absolute change
largest_change_cat = max(category_mix_changes.items(), 
                         key=lambda x: abs(x[1]['pct_point_change']))
cat_name = largest_change_cat[0]
cat_metrics = largest_change_cat[1]

finding2 = {
    "title": f"Category Mix Shift: {cat_name} (Week of 2026-03-23 vs 2026-03-16)",
    "claim": f"The {cat_name} category declined from {cat_metrics['prior_pct']:.2f}% to {cat_metrics['current_pct']:.2f}% of total revenue, a {cat_metrics['pct_point_change']:.2f} percentage point shift. Revenue for {cat_name} decreased from SAR {cat_metrics['prior_revenue']:.2f} to SAR {cat_metrics['current_revenue']:.2f}, a decline of SAR {cat_metrics['current_revenue'] - cat_metrics['prior_revenue']:.2f} ({((cat_metrics['current_revenue'] - cat_metrics['prior_revenue']) / cat_metrics['prior_revenue'] * 100):.2f}%).",
    "finding_type": "category_mix_change",
    "metrics": {
        "analysis_category_pct": {
            "value": round(cat_metrics['current_pct'], 2),
            "unit": "%",
            "numerator": round(cat_metrics['current_revenue'], 2),
            "denominator": round(analysis_total_revenue, 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "previous_category_pct": {
            "value": round(cat_metrics['prior_pct'], 2),
            "unit": "%",
            "numerator": round(cat_metrics['prior_revenue'], 2),
            "denominator": round(previous_total_revenue, 2),
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "category_mix_change_pct_points": {
            "value": round(cat_metrics['pct_point_change'], 2),
            "unit": "percentage points",
            "numerator": round(cat_metrics['current_pct'] - cat_metrics['prior_pct'], 2),
            "denominator": None,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "analysis_category_revenue_sar": {
            "value": round(cat_metrics['current_revenue'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "previous_category_revenue_sar": {
            "value": round(cat_metrics['prior_revenue'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        },
        "category_revenue_change_sar": {
            "value": round(cat_metrics['current_revenue'] - cat_metrics['prior_revenue'], 2),
            "unit": "SAR",
            "numerator": round(cat_metrics['current_revenue'] - cat_metrics['prior_revenue'], 2),
            "denominator": round(cat_metrics['prior_revenue'], 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        },
        "category_revenue_change_pct": {
            "value": round(((cat_metrics['current_revenue'] - cat_metrics['prior_revenue']) / cat_metrics['prior_revenue'] * 100), 2),
            "unit": "%",
            "numerator": round(cat_metrics['current_revenue'] - cat_metrics['prior_revenue'], 2),
            "denominator": round(cat_metrics['prior_revenue'], 2),
            "period_start": "2026-03-23T00:00:00+03:00",
            "period_end": "2026-03-30T00:00:00+03:00"
        }
    },
    "source_names": ["pos"],
    "sample_size": len(analysis_data),
    "coverage_notes": [
        f"Analysis period {cat_name} revenue: SAR {cat_metrics['current_revenue']:.2f} from {len(analysis_data[analysis_data['category'] == cat_name])} line items",
        f"Previous period {cat_name} revenue: SAR {cat_metrics['prior_revenue']:.2f} from {len(previous_data[previous_data['category'] == cat_name])} line items",
        f"Total analysis period revenue: SAR {analysis_total_revenue:.2f}",
        f"Total previous period revenue: SAR {previous_total_revenue:.2f}"
    ],
    "assumptions": [
        "Category field is populated and consistent across both periods",
        "line_total_sar represents net revenue including refunds",
        "Percentage point change calculated as (current_pct - prior_pct)"
    ],
    "confidence": 0.92
}

findings.append(finding2)

# ============================================================================
# FINDING 3: Top SKU Performance Change
# ============================================================================

# Identify top SKUs by revenue in analysis period
analysis_sku_revenue = analysis_data.groupby('sku').agg({
    'line_total_sar': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).rename(columns={'transaction_id': 'baskets'})
analysis_sku_revenue = analysis_sku_revenue.sort_values('line_total_sar', ascending=False)

# Get top SKU
if len(analysis_sku_revenue) > 0:
    top_sku = analysis_sku_revenue.index[0]
    
    # Get metrics for top SKU in both periods
    analysis_top_sku = analysis_data[analysis_data['sku'] == top_sku]
    previous_top_sku = previous_data[previous_data['sku'] == top_sku]
    
    analysis_sku_revenue_val = analysis_top_sku['line_total_sar'].sum()
    analysis_sku_qty = analysis_top_sku['quantity'].sum()
    analysis_sku_baskets = analysis_top_sku['transaction_id'].nunique()
    
    previous_sku_revenue_val = previous_top_sku['line_total_sar'].sum()
    previous_sku_qty = previous_top_sku['quantity'].sum()
    previous_sku_baskets = previous_top_sku['transaction_id'].nunique()
    
    # Get SKU name from menu
    sku_menu = menu_df[menu_df['sku'] == top_sku]
    sku_name = sku_menu['item_en'].values[0] if len(sku_menu) > 0 else top_sku
    
    # Calculate changes
    sku_revenue_change = analysis_sku_revenue_val - previous_sku_revenue_val
    sku_revenue_change_pct = (sku_revenue_change / previous_sku_revenue_val * 100) if previous_sku_revenue_val != 0 else 0
    sku_qty_change = analysis_sku_qty - previous_sku_qty
    sku_qty_change_pct = (sku_qty_change / previous_sku_qty * 100) if previous_sku_qty != 0 else 0
    
    finding3 = {
        "title": f"Top SKU Performance: {sku_name} ({top_sku}) (Week of 2026-03-23 vs 2026-03-16)",
        "claim": f"The top-performing SKU {sku_name} ({top_sku}) generated SAR {analysis_sku_revenue_val:.2f} in the analysis week, compared to SAR {previous_sku_revenue_val:.2f} in the previous week, a change of SAR {sku_revenue_change:.2f} ({sku_revenue_change_pct:.2f}%). Units sold changed from {int(previous_sku_qty)} to {int(analysis_sku_qty)} ({sku_qty_change_pct:.2f}%), and basket penetration changed from {previous_sku_baskets} to {analysis_sku_baskets} baskets.",
        "finding_type": "sku_performance_change",
        "metrics": {
            "analysis_sku_revenue_sar": {
                "value": round(analysis_sku_revenue_val, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "previous_sku_revenue_sar": {
                "value": round(previous_sku_revenue_val, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "sku_revenue_change_sar": {
                "value": round(sku_revenue_change, 2),
                "unit": "SAR",
                "numerator": round(sku_revenue_change, 2),
                "denominator": round(previous_sku_revenue_val, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "sku_revenue_change_pct": {
                "value": round(sku_revenue_change_pct, 2),
                "unit": "%",
                "numerator": round(sku_revenue_change, 2),
                "denominator": round(previous_sku_revenue_val, 2),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_sku_quantity": {
                "value": int(analysis_sku_qty),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "previous_sku_quantity": {
                "value": int(previous_sku_qty),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "sku_quantity_change": {
                "value": int(sku_qty_change),
                "unit": "units",
                "numerator": int(sku_qty_change),
                "denominator": int(previous_sku_qty),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "sku_quantity_change_pct": {
                "value": round(sku_qty_change_pct, 2),
                "unit": "%",
                "numerator": int(sku_qty_change),
                "denominator": int(previous_sku_qty),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "analysis_sku_baskets": {
                "value": analysis_sku_baskets,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "previous_sku_baskets": {
                "value": previous_sku_baskets,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_top_sku),
        "coverage_notes": [
            f"Analysis period: {len(analysis_top_sku)} line items for SKU {top_sku}",
            f"Previous period: {len(previous_top_sku)} line items for SKU {top_sku}",
            f"SKU name resolved from menu artifact: {sku_name}",
            f"Top SKU identified by highest revenue in analysis period"
        ],
        "assumptions": [
            "SKU {top_sku} is the highest-revenue product in the analysis week",
            "Menu SKU reference provides authoritative product naming",
            "transaction_id uniqueness defines basket penetration",
            "line_total_sar includes all discounts and refunds"
        ],
        "confidence": 0.93
    }
    
    findings.append(finding3)

# Write output
output = {
    "status": "success",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)