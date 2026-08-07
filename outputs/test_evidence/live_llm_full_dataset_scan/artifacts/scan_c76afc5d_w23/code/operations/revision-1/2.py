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

# Define periods (all in UTC+3)
tz = ZoneInfo('Asia/Riyadh')
analysis_start = datetime(2026, 6, 15, 0, 0, 0, tzinfo=tz)
analysis_end = datetime(2026, 6, 22, 0, 0, 0, tzinfo=tz)
previous_start = datetime(2026, 6, 8, 0, 0, 0, tzinfo=tz)
previous_end = datetime(2026, 6, 15, 0, 0, 0, tzinfo=tz)

# Convert to UTC for comparison
analysis_start_utc = analysis_start.astimezone(ZoneInfo('UTC'))
analysis_end_utc = analysis_end.astimezone(ZoneInfo('UTC'))
previous_start_utc = previous_start.astimezone(ZoneInfo('UTC'))
previous_end_utc = previous_end.astimezone(ZoneInfo('UTC'))

# Ensure timestamp columns are datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Filter POS data for analysis and previous periods
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start_utc) & (pos_df['timestamp'] < analysis_end_utc)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start_utc) & (pos_df['timestamp'] < previous_end_utc)].copy()

# Filter traffic data
traffic_df['date_only'] = traffic_df['date'].dt.date
analysis_date_start = analysis_start.date()
analysis_date_end = (analysis_end - timedelta(days=1)).date()
previous_date_start = previous_start.date()
previous_date_end = (previous_end - timedelta(days=1)).date()

traffic_analysis = traffic_df[(traffic_df['date_only'] >= analysis_date_start) & (traffic_df['date_only'] <= analysis_date_end)].copy()
traffic_previous = traffic_df[(traffic_df['date_only'] >= previous_date_start) & (traffic_df['date_only'] <= previous_date_end)].copy()

# Filter staff data
staff_df['date_only'] = staff_df['date'].dt.date
staff_analysis = staff_df[(staff_df['date_only'] >= analysis_date_start) & (staff_df['date_only'] <= analysis_date_end)].copy()
staff_previous = staff_df[(staff_df['date_only'] >= previous_date_start) & (staff_df['date_only'] <= previous_date_end)].copy()

# Filter inventory data
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-06-15', tz='UTC')].copy()
inventory_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-06-08', tz='UTC')].copy()

findings = []
result_metrics = {}

# ============================================================================
# FINDING 1: Conversion Rate Analysis
# ============================================================================

# Analysis period conversion
valid_sales_analysis = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
valid_footfall_analysis = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['door_count'].sum()

if valid_footfall_analysis > 0:
    conversion_analysis = valid_sales_analysis / valid_footfall_analysis
else:
    conversion_analysis = None

# Previous period conversion
valid_sales_previous = pos_previous[pos_previous['is_refund'] == False]['transaction_id'].nunique()
valid_footfall_previous = traffic_previous[traffic_previous['is_dead_sensor_day'] == False]['door_count'].sum()

if valid_footfall_previous > 0:
    conversion_previous = valid_sales_previous / valid_footfall_previous
else:
    conversion_previous = None

# Calculate change
if conversion_analysis is not None and conversion_previous is not None and conversion_previous > 0:
    conversion_change_pct = ((conversion_analysis - conversion_previous) / conversion_previous) * 100
else:
    conversion_change_pct = None

result_metrics['conversion_rate_analysis'] = {
    'value': round(conversion_analysis, 4) if conversion_analysis else None,
    'unit': 'ratio',
    'numerator': valid_sales_analysis,
    'denominator': valid_footfall_analysis,
    'period_start': analysis_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

result_metrics['conversion_rate_previous'] = {
    'value': round(conversion_previous, 4) if conversion_previous else None,
    'unit': 'ratio',
    'numerator': valid_sales_previous,
    'denominator': valid_footfall_previous,
    'period_start': previous_start.isoformat(),
    'period_end': previous_end.isoformat()
}

result_metrics['conversion_change_pct'] = {
    'value': round(conversion_change_pct, 2) if conversion_change_pct else None,
    'unit': '%',
    'numerator': None,
    'denominator': None,
    'period_start': previous_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

if conversion_analysis is not None and conversion_previous is not None:
    findings.append({
        'title': 'Conversion Rate Comparison',
        'claim': f'Conversion rate in analysis period (Jun 15-22) was {conversion_analysis:.4f} vs {conversion_previous:.4f} in previous period (Jun 8-15), a change of {conversion_change_pct:.2f}%.',
        'finding_type': 'performance_metric',
        'metrics': {
            'conversion_rate_analysis': result_metrics['conversion_rate_analysis'],
            'conversion_rate_previous': result_metrics['conversion_rate_previous'],
            'conversion_change_pct': result_metrics['conversion_change_pct']
        },
        'source_names': ['pos', 'traffic'],
        'sample_size': valid_sales_analysis + valid_sales_previous,
        'coverage_notes': [
            f'Analysis period: {valid_footfall_analysis} valid footfall entries (dead sensor days excluded)',
            f'Previous period: {valid_footfall_previous} valid footfall entries (dead sensor days excluded)',
            f'Analysis period: {valid_sales_analysis} valid sales transactions (refunds excluded)',
            f'Previous period: {valid_sales_previous} valid sales transactions (refunds excluded)'
        ],
        'assumptions': [
            'Conversion = unique valid sales transactions / valid footfall',
            'Dead sensor days excluded from footfall denominator',
            'Refunds excluded from transaction count'
        ],
        'confidence': 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost and Revenue Relationship
# ============================================================================

# Analysis period
total_labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
total_revenue_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
labour_cost_ratio_analysis = total_labour_cost_analysis / total_revenue_analysis if total_revenue_analysis > 0 else None

# Previous period
total_labour_cost_previous = staff_previous['labour_cost_sar'].sum()
total_revenue_previous = pos_previous[pos_previous['is_refund'] == False]['line_total_sar'].sum()
labour_cost_ratio_previous = total_labour_cost_previous / total_revenue_previous if total_revenue_previous > 0 else None

# Calculate change
if labour_cost_ratio_analysis is not None and labour_cost_ratio_previous is not None and labour_cost_ratio_previous > 0:
    labour_ratio_change_pct = ((labour_cost_ratio_analysis - labour_cost_ratio_previous) / labour_cost_ratio_previous) * 100
else:
    labour_ratio_change_pct = None

result_metrics['total_labour_cost_analysis'] = {
    'value': round(total_labour_cost_analysis, 2),
    'unit': 'SAR',
    'numerator': None,
    'denominator': None,
    'period_start': analysis_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

result_metrics['total_revenue_analysis'] = {
    'value': round(total_revenue_analysis, 2),
    'unit': 'SAR',
    'numerator': None,
    'denominator': None,
    'period_start': analysis_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

result_metrics['labour_cost_ratio_analysis'] = {
    'value': round(labour_cost_ratio_analysis, 4) if labour_cost_ratio_analysis else None,
    'unit': 'ratio',
    'numerator': round(total_labour_cost_analysis, 2),
    'denominator': round(total_revenue_analysis, 2),
    'period_start': analysis_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

result_metrics['total_labour_cost_previous'] = {
    'value': round(total_labour_cost_previous, 2),
    'unit': 'SAR',
    'numerator': None,
    'denominator': None,
    'period_start': previous_start.isoformat(),
    'period_end': previous_end.isoformat()
}

result_metrics['total_revenue_previous'] = {
    'value': round(total_revenue_previous, 2),
    'unit': 'SAR',
    'numerator': None,
    'denominator': None,
    'period_start': previous_start.isoformat(),
    'period_end': previous_end.isoformat()
}

result_metrics['labour_cost_ratio_previous'] = {
    'value': round(labour_cost_ratio_previous, 4) if labour_cost_ratio_previous else None,
    'unit': 'ratio',
    'numerator': round(total_labour_cost_previous, 2),
    'denominator': round(total_revenue_previous, 2),
    'period_start': previous_start.isoformat(),
    'period_end': previous_end.isoformat()
}

result_metrics['labour_ratio_change_pct'] = {
    'value': round(labour_ratio_change_pct, 2) if labour_ratio_change_pct else None,
    'unit': '%',
    'numerator': None,
    'denominator': None,
    'period_start': previous_start.isoformat(),
    'period_end': analysis_end.isoformat()
}

if labour_cost_ratio_analysis is not None and labour_cost_ratio_previous is not None:
    findings.append({
        'title': 'Labour Cost to Revenue Ratio',
        'claim': f'Labour cost ratio in analysis period (Jun 15-22) was {labour_cost_ratio_analysis:.4f} vs {labour_cost_ratio_previous:.4f} in previous period (Jun 8-15), a change of {labour_ratio_change_pct:.2f}%.',
        'finding_type': 'operational_efficiency',
        'metrics': {
            'total_labour_cost_analysis': result_metrics['total_labour_cost_analysis'],
            'total_revenue_analysis': result_metrics['total_revenue_analysis'],
            'labour_cost_ratio_analysis': result_metrics['labour_cost_ratio_analysis'],
            'total_labour_cost_previous': result_metrics['total_labour_cost_previous'],
            'total_revenue_previous': result_metrics['total_revenue_previous'],
            'labour_cost_ratio_previous': result_metrics['labour_cost_ratio_previous'],
            'labour_ratio_change_pct': result_metrics['labour_ratio_change_pct']
        },
        'source_names': ['staff', 'pos'],
        'sample_size': len(staff_analysis) + len(staff_previous),
        'coverage_notes': [
            f'Analysis period: {len(staff_analysis)} staff records, {round(total_labour_cost_analysis, 2)} SAR total labour cost',
            f'Previous period: {len(staff_previous)} staff records, {round(total_labour_cost_previous, 2)} SAR total labour cost',
            f'Analysis period revenue (net of refunds): {round(total_revenue_analysis, 2)} SAR',
            f'Previous period revenue (net of refunds): {round(total_revenue_previous, 2)} SAR'
        ],
        'assumptions': [
            'Labour cost ratio = total labour cost / total revenue (refunds excluded)',
            'Staff labour_cost_sar field used as authoritative',
            'POS line_total_sar used for revenue calculation'
        ],
        'confidence': 0.80
    })

# ============================================================================
# FINDING 3: Known Waste Cost Analysis
# ============================================================================

if len(inventory_analysis) > 0 and len(inventory_previous) > 0:
    known_waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
    known_waste_cost_previous = inventory_previous['known_waste_cost_sar'].sum()
    
    # Calculate waste as percentage of revenue
    waste_ratio_analysis = known_waste_cost_analysis / total_revenue_analysis if total_revenue_analysis > 0 else None
    waste_ratio_previous = known_waste_cost_previous / total_revenue_previous if total_revenue_previous > 0 else None
    
    if waste_ratio_analysis is not None and waste_ratio_previous is not None and waste_ratio_previous > 0:
        waste_ratio_change_pct = ((waste_ratio_analysis - waste_ratio_previous) / waste_ratio_previous) * 100
    else:
        waste_ratio_change_pct = None
    
    result_metrics['known_waste_cost_analysis'] = {
        'value': round(known_waste_cost_analysis, 2),
        'unit': 'SAR',
        'numerator': None,
        'denominator': None,
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    result_metrics['known_waste_cost_previous'] = {
        'value': round(known_waste_cost_previous, 2),
        'unit': 'SAR',
        'numerator': None,
        'denominator': None,
        'period_start': previous_start.isoformat(),
        'period_end': previous_end.isoformat()
    }
    
    result_metrics['waste_ratio_analysis'] = {
        'value': round(waste_ratio_analysis, 4) if waste_ratio_analysis else None,
        'unit': 'ratio',
        'numerator': round(known_waste_cost_analysis, 2),
        'denominator': round(total_revenue_analysis, 2),
        'period_start': analysis_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    result_metrics['waste_ratio_previous'] = {
        'value': round(waste_ratio_previous, 4) if waste_ratio_previous else None,
        'unit': 'ratio',
        'numerator': round(known_waste_cost_previous, 2),
        'denominator': round(total_revenue_previous, 2),
        'period_start': previous_start.isoformat(),
        'period_end': previous_end.isoformat()
    }
    
    result_metrics['waste_ratio_change_pct'] = {
        'value': round(waste_ratio_change_pct, 2) if waste_ratio_change_pct else None,
        'unit': '%',
        'numerator': None,
        'denominator': None,
        'period_start': previous_start.isoformat(),
        'period_end': analysis_end.isoformat()
    }
    
    if waste_ratio_analysis is not None and waste_ratio_previous is not None:
        findings.append({
            'title': 'Known Waste Cost Trend',
            'claim': f'Known waste cost ratio in analysis period (Jun 15-22) was {waste_ratio_analysis:.4f} vs {waste_ratio_previous:.4f} in previous period (Jun 8-15), a change of {waste_ratio_change_pct:.2f}%.',
            'finding_type': 'cost_management',
            'metrics': {
                'known_waste_cost_analysis': result_metrics['known_waste_cost_analysis'],
                'known_waste_cost_previous': result_metrics['known_waste_cost_previous'],
                'waste_ratio_analysis': result_metrics['waste_ratio_analysis'],
                'waste_ratio_previous': result_metrics['waste_ratio_previous'],
                'waste_ratio_change_pct': result_metrics['waste_ratio_change_pct']
            },
            'source_names': ['inventory', 'pos'],
            'sample_size': len(inventory_analysis) + len(inventory_previous),
            'coverage_notes': [
                f'Analysis period: {len(inventory_analysis)} inventory records, {round(known_waste_cost_analysis, 2)} SAR known waste cost',
                f'Previous period: {len(inventory_previous)} inventory records, {round(known_waste_cost_previous, 2)} SAR known waste cost',
                'Unknown waste values preserved; only known_waste_cost_sar included in calculation',
                'Waste ratio calculated against net revenue (refunds excluded)'
            ],
            'assumptions': [
                'Waste ratio = known waste cost / total revenue',
                'Only known waste costs included; unknown waste excluded',
                'Inventory week_starting used to align with analysis periods'
            ],
            'confidence': 0.75
        })

# ============================================================================
# Write output
# ============================================================================

output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
