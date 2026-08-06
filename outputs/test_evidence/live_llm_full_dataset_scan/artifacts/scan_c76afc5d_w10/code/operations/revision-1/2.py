import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
staff_df = pd.read_parquet(inputs['staff'])
inventory_df = pd.read_parquet(inputs['inventory'])

# Define periods (UTC+3)
tz = ZoneInfo('Asia/Riyadh')
analysis_start = datetime(2026, 3, 16, 0, 0, 0, tzinfo=tz)
analysis_end = datetime(2026, 3, 23, 0, 0, 0, tzinfo=tz)
previous_start = datetime(2026, 3, 9, 0, 0, 0, tzinfo=tz)
previous_end = datetime(2026, 3, 16, 0, 0, 0, tzinfo=tz)

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

# Filter POS for analysis and previous periods
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter traffic for analysis and previous periods
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start) & (traffic_df['date'] < analysis_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start) & (traffic_df['date'] < previous_end)].copy()

# Filter staff for analysis and previous periods
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()
staff_previous = staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())].copy()

# Filter inventory for analysis week
inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)].copy()
inventory_previous = inventory_df[(inventory_df['week_starting'] >= previous_start) & (inventory_df['week_starting'] < previous_end)].copy()

findings = []
result_metrics = {}

# ============================================================================
# FINDING 1: Conversion Rate Analysis
# ============================================================================

# Analysis period conversion
pos_analysis_valid = pos_analysis[~pos_analysis['is_refund']].copy()
transactions_analysis = pos_analysis_valid['transaction_id'].nunique()

# Exclude dead sensor days from traffic
traffic_analysis_valid = traffic_analysis[~traffic_analysis['is_dead_sensor_day']].copy()
footfall_analysis = traffic_analysis_valid['door_count'].sum()

if footfall_analysis > 0:
    conversion_analysis = transactions_analysis / footfall_analysis
else:
    conversion_analysis = None

# Previous period conversion
pos_previous_valid = pos_previous[~pos_previous['is_refund']].copy()
transactions_previous = pos_previous_valid['transaction_id'].nunique()

traffic_previous_valid = traffic_previous[~traffic_previous['is_dead_sensor_day']].copy()
footfall_previous = traffic_previous_valid['door_count'].sum()

if footfall_previous > 0:
    conversion_previous = transactions_previous / footfall_previous
else:
    conversion_previous = None

# Calculate change
if conversion_analysis is not None and conversion_previous is not None and conversion_previous > 0:
    conversion_change_pct = ((conversion_analysis - conversion_previous) / conversion_previous) * 100
    
    result_metrics['conversion_rate_analysis_period'] = {
        'value': round(conversion_analysis, 4),
        'unit': 'transactions per visitor',
        'numerator': transactions_analysis,
        'denominator': footfall_analysis,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    result_metrics['conversion_rate_previous_period'] = {
        'value': round(conversion_previous, 4),
        'unit': 'transactions per visitor',
        'numerator': transactions_previous,
        'denominator': footfall_previous,
        'period_start': previous_start.isoformat(),
        'period_end': previous_end.isoformat()
    }
    
    result_metrics['conversion_change_percent_analysis_vs_previous'] = {
        'value': round(conversion_change_pct, 2),
        'unit': 'percent',
        'numerator': None,
        'denominator': None,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    findings.append({
        'title': 'Conversion Rate Comparison',
        'claim': f'Conversion rate in analysis period ({analysis_start.date()} to {analysis_end.date()}) was {conversion_analysis:.4f} transactions per visitor, compared to {conversion_previous:.4f} in the previous period ({previous_start.date()} to {previous_end.date()}), representing a {conversion_change_pct:.2f}% change.',
        'finding_type': 'performance_metric',
        'metrics': result_metrics,
        'source_names': ['pos', 'traffic'],
        'sample_size': transactions_analysis,
        'coverage_notes': [
            f'Analysis period: {len(traffic_analysis_valid)} valid traffic days (excluded {len(traffic_analysis) - len(traffic_analysis_valid)} dead sensor days)',
            f'Previous period: {len(traffic_previous_valid)} valid traffic days (excluded {len(traffic_previous) - len(traffic_previous_valid)} dead sensor days)',
            f'POS transactions exclude refunds'
        ],
        'assumptions': [
            'Conversion = unique valid sales transactions / valid footfall',
            'Dead sensor days excluded from footfall denominator',
            'Refunds excluded from transaction count'
        ],
        'confidence': 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Demand
# ============================================================================

# Calculate total labour cost and demand for analysis period
labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
demand_analysis = pos_analysis_valid['line_total_sar'].sum()

# Calculate for previous period
labour_cost_previous = staff_previous['labour_cost_sar'].sum()
demand_previous = pos_previous_valid['line_total_sar'].sum()

if labour_cost_analysis > 0 and demand_analysis > 0:
    labour_to_demand_analysis = labour_cost_analysis / demand_analysis
    labour_to_demand_previous = labour_cost_previous / demand_previous if demand_previous > 0 else None
    
    result_metrics['labour_cost_analysis_period'] = {
        'value': round(labour_cost_analysis, 2),
        'unit': 'SAR',
        'numerator': None,
        'denominator': None,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    result_metrics['demand_analysis_period'] = {
        'value': round(demand_analysis, 2),
        'unit': 'SAR',
        'numerator': None,
        'denominator': None,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    result_metrics['labour_to_demand_ratio_analysis_period'] = {
        'value': round(labour_to_demand_analysis, 4),
        'unit': 'ratio',
        'numerator': labour_cost_analysis,
        'denominator': demand_analysis,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    if labour_to_demand_previous is not None:
        result_metrics['labour_to_demand_ratio_previous_period'] = {
            'value': round(labour_to_demand_previous, 4),
            'unit': 'ratio',
            'numerator': labour_cost_previous,
            'denominator': demand_previous,
            'period_start': previous_start.isoformat(),
            'period_end': previous_end.isoformat()
        }
        
        ratio_change = ((labour_to_demand_analysis - labour_to_demand_previous) / labour_to_demand_previous) * 100
        result_metrics['labour_to_demand_ratio_change_percent_analysis_vs_previous'] = {
            'value': round(ratio_change, 2),
            'unit': 'percent',
            'numerator': None,
            'denominator': None,
            'period_start': analysis_start.isoformat(),
            'period_end': analysis_end.isoformat()
        }
        
        findings.append({
            'title': 'Labour Cost Efficiency',
            'claim': f'Labour cost to demand ratio in analysis period was {labour_to_demand_analysis:.4f} (labour cost {labour_cost_analysis:.2f} SAR / demand {demand_analysis:.2f} SAR), compared to {labour_to_demand_previous:.4f} in previous period, a {ratio_change:.2f}% change.',
            'finding_type': 'efficiency_metric',
            'metrics': result_metrics,
            'source_names': ['staff', 'pos'],
            'sample_size': len(staff_analysis),
            'coverage_notes': [
                f'Analysis period: {len(staff_analysis)} staff records',
                f'Previous period: {len(staff_previous)} staff records',
                f'Labour cost includes computed duration hours and hourly rates',
                f'Demand calculated from non-refund POS transactions'
            ],
            'assumptions': [
                'Labour cost from staff shift records with computed_duration_hours',
                'Demand = sum of line_total_sar excluding refunds',
                'No imputation for missing staff records'
            ],
            'confidence': 0.80
        })

# ============================================================================
# FINDING 3: Known Waste Cost Analysis
# ============================================================================

if len(inventory_analysis) > 0:
    waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
    units_sold_analysis = inventory_analysis['units_sold'].sum()
    
    if len(inventory_previous) > 0:
        waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
        units_sold_previous = inventory_previous['units_sold'].sum()
        
        if units_sold_analysis > 0 and units_sold_previous > 0:
            waste_ratio_analysis = waste_cost_analysis / units_sold_analysis if units_sold_analysis > 0 else 0
            waste_ratio_previous = waste_cost_previous / units_sold_previous if units_sold_previous > 0 else 0
            
            result_metrics['known_waste_cost_analysis_period'] = {
                'value': round(waste_cost_analysis, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
            
            result_metrics['known_waste_cost_previous_period'] = {
                'value': round(waste_cost_previous, 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
            
            result_metrics['waste_ratio_analysis_period'] = {
                'value': round(waste_ratio_analysis, 4),
                'unit': 'SAR per unit sold',
                'numerator': waste_cost_analysis,
                'denominator': units_sold_analysis,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
            
            result_metrics['waste_ratio_previous_period'] = {
                'value': round(waste_ratio_previous, 4),
                'unit': 'SAR per unit sold',
                'numerator': waste_cost_previous,
                'denominator': units_sold_previous,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
            
            waste_ratio_change = ((waste_ratio_analysis - waste_ratio_previous) / waste_ratio_previous) * 100 if waste_ratio_previous > 0 else 0
            result_metrics['waste_ratio_change_percent_analysis_vs_previous'] = {
                'value': round(waste_ratio_change, 2),
                'unit': 'percent',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
            
            findings.append({
                'title': 'Known Waste Cost Efficiency',
                'claim': f'Known waste cost per unit sold in analysis period was {waste_ratio_analysis:.4f} SAR (total waste cost {waste_cost_analysis:.2f} SAR / {units_sold_analysis} units sold), compared to {waste_ratio_previous:.4f} SAR in previous period, a {waste_ratio_change:.2f}% change.',
                'finding_type': 'waste_metric',
                'metrics': result_metrics,
                'source_names': ['inventory'],
                'sample_size': len(inventory_analysis),
                'coverage_notes': [
                    f'Analysis period: {len(inventory_analysis)} inventory records',
                    f'Previous period: {len(inventory_previous)} inventory records',
                    f'Known waste cost only; unknown waste values excluded per schema',
                    f'Units sold from inventory records'
                ],
                'assumptions': [
                    'Known waste cost from inventory.known_waste_cost_sar column',
                    'Unknown waste values not included in calculations',
                    'Waste ratio = known_waste_cost / units_sold'
                ],
                'confidence': 0.75
            })

# ============================================================================
# Write output
# ============================================================================

output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings[:3]  # Max 3 findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
