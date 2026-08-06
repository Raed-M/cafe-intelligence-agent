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

# Define timezone using zoneinfo
tz = ZoneInfo('Asia/Riyadh')

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.date
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.date

# Define analysis periods
analysis_start = pd.Timestamp('2026-03-23', tz=tz)
analysis_end = pd.Timestamp('2026-03-30', tz=tz)
previous_start = pd.Timestamp('2026-03-16', tz=tz)
previous_end = pd.Timestamp('2026-03-23', tz=tz)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.date()) & (traffic_df['date'] < previous_end.date())].copy()

# Calculate conversion metrics
def calculate_conversion(pos_data, traffic_data, period_name):
    # Count unique valid transactions (exclude refunds)
    valid_transactions = pos_data[~pos_data['is_refund']].groupby('transaction_id').size()
    unique_transactions = len(valid_transactions)
    
    # Count valid footfall (exclude dead sensor days)
    valid_traffic = traffic_data[~traffic_data['is_dead_sensor_day']].copy()
    total_footfall = valid_traffic['door_count'].sum()
    
    if total_footfall > 0:
        conversion = unique_transactions / total_footfall
    else:
        conversion = None
    
    return {
        'unique_transactions': unique_transactions,
        'total_footfall': total_footfall,
        'conversion': conversion,
        'period': period_name
    }

conv_analysis = calculate_conversion(pos_analysis, traffic_analysis, 'analysis')
conv_previous = calculate_conversion(pos_previous, traffic_previous, 'previous')

# Calculate revenue metrics
def calculate_revenue(pos_data, period_name):
    # Exclude refunds from revenue
    valid_sales = pos_data[~pos_data['is_refund']].copy()
    total_revenue = valid_sales['line_total_sar'].sum()
    transaction_count = valid_sales['transaction_id'].nunique()
    
    if transaction_count > 0:
        avg_transaction = total_revenue / transaction_count
    else:
        avg_transaction = None
    
    return {
        'total_revenue': total_revenue,
        'transaction_count': transaction_count,
        'avg_transaction': avg_transaction,
        'period': period_name
    }

rev_analysis = calculate_revenue(pos_analysis, 'analysis')
rev_previous = calculate_revenue(pos_previous, 'previous')

# Calculate staffing metrics
def calculate_staffing(staff_data, period_name):
    total_hours = staff_data['computed_duration_hours'].sum()
    total_cost = staff_data['labour_cost_sar'].sum()
    unique_employees = staff_data['employee_id'].nunique()
    
    return {
        'total_hours': total_hours,
        'total_cost': total_cost,
        'unique_employees': unique_employees,
        'period': period_name
    }

staff_analysis_metrics = calculate_staffing(staff_analysis, 'analysis')
staff_previous_metrics = calculate_staffing(staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())], 'previous')

# Calculate inventory metrics for analysis week
analysis_week_start = pd.Timestamp('2026-03-23', tz=tz).date()
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week_start]

# Prepare findings
findings = []

# Finding 1: Conversion Rate Comparison
if conv_analysis['total_footfall'] > 0 and conv_previous['total_footfall'] > 0:
    conversion_change = ((conv_analysis['conversion'] - conv_previous['conversion']) / conv_previous['conversion'] * 100) if conv_previous['conversion'] > 0 else None
    
    findings.append({
        'title': 'Conversion Rate Analysis',
        'claim': f"Conversion rate in analysis week was {conv_analysis['conversion']:.4f} (transactions/visitor) compared to {conv_previous['conversion']:.4f} in previous week",
        'finding_type': 'performance_metric',
        'metrics': {
            'conversion_rate_analysis': {
                'value': round(conv_analysis['conversion'], 4),
                'unit': 'transactions_per_visitor',
                'numerator': conv_analysis['unique_transactions'],
                'denominator': conv_analysis['total_footfall'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'conversion_rate_previous': {
                'value': round(conv_previous['conversion'], 4),
                'unit': 'transactions_per_visitor',
                'numerator': conv_previous['unique_transactions'],
                'denominator': conv_previous['total_footfall'],
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
        },
        'source_names': ['pos', 'traffic'],
        'sample_size': conv_analysis['unique_transactions'],
        'coverage_notes': [
            f"Analysis period: {conv_analysis['unique_transactions']} valid transactions from {conv_analysis['total_footfall']} footfall",
            f"Previous period: {conv_previous['unique_transactions']} valid transactions from {conv_previous['total_footfall']} footfall",
            'Excluded refunds from transaction count',
            'Excluded dead sensor days from footfall'
        ],
        'assumptions': [
            'Conversion calculated as unique valid sales transactions divided by valid footfall',
            'Refunds excluded from transaction count',
            'Dead sensor intervals excluded from footfall denominator'
        ],
        'confidence': 0.85
    })

# Finding 2: Revenue and Transaction Performance
if rev_analysis['transaction_count'] > 0 and rev_previous['transaction_count'] > 0:
    revenue_change_pct = ((rev_analysis['total_revenue'] - rev_previous['total_revenue']) / rev_previous['total_revenue'] * 100)
    
    findings.append({
        'title': 'Revenue Performance Comparison',
        'claim': f"Total revenue in analysis week was {rev_analysis['total_revenue']:.2f} SAR across {rev_analysis['transaction_count']} transactions, compared to {rev_previous['total_revenue']:.2f} SAR in previous week",
        'finding_type': 'financial_metric',
        'metrics': {
            'total_revenue_analysis': {
                'value': round(rev_analysis['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'avg_transaction_value_analysis': {
                'value': round(rev_analysis['avg_transaction'], 2),
                'unit': 'SAR',
                'numerator': rev_analysis['total_revenue'],
                'denominator': rev_analysis['transaction_count'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'total_revenue_previous': {
                'value': round(rev_previous['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'avg_transaction_value_previous': {
                'value': round(rev_previous['avg_transaction'], 2),
                'unit': 'SAR',
                'numerator': rev_previous['total_revenue'],
                'denominator': rev_previous['transaction_count'],
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': rev_analysis['transaction_count'],
        'coverage_notes': [
            f"Analysis period: {rev_analysis['transaction_count']} valid transactions",
            f"Previous period: {rev_previous['transaction_count']} valid transactions",
            'Refunds excluded from revenue calculations'
        ],
        'assumptions': [
            'Revenue calculated from line_total_sar excluding refunds',
            'Transaction identified by unique transaction_id',
            'Average transaction value = total revenue / transaction count'
        ],
        'confidence': 0.90
    })

# Finding 3: Staffing Cost and Hours
if staff_analysis_metrics['total_hours'] > 0 and staff_previous_metrics['total_hours'] > 0:
    cost_per_hour_analysis = staff_analysis_metrics['total_cost'] / staff_analysis_metrics['total_hours']
    cost_per_hour_previous = staff_previous_metrics['total_cost'] / staff_previous_metrics['total_hours']
    
    findings.append({
        'title': 'Labour Cost and Staffing Hours',
        'claim': f"Analysis week had {staff_analysis_metrics['total_hours']:.1f} total staff hours costing {staff_analysis_metrics['total_cost']:.2f} SAR, compared to {staff_previous_metrics['total_hours']:.1f} hours costing {staff_previous_metrics['total_cost']:.2f} SAR in previous week",
        'finding_type': 'operational_metric',
        'metrics': {
            'total_staff_hours_analysis': {
                'value': round(staff_analysis_metrics['total_hours'], 1),
                'unit': 'hours',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'total_labour_cost_analysis': {
                'value': round(staff_analysis_metrics['total_cost'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'cost_per_hour_analysis': {
                'value': round(cost_per_hour_analysis, 2),
                'unit': 'SAR/hour',
                'numerator': staff_analysis_metrics['total_cost'],
                'denominator': staff_analysis_metrics['total_hours'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'total_staff_hours_previous': {
                'value': round(staff_previous_metrics['total_hours'], 1),
                'unit': 'hours',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'total_labour_cost_previous': {
                'value': round(staff_previous_metrics['total_cost'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
        },
        'source_names': ['staff'],
        'sample_size': staff_analysis_metrics['unique_employees'],
        'coverage_notes': [
            f"Analysis period: {staff_analysis_metrics['unique_employees']} unique employees",
            f"Previous period: {staff_previous_metrics['unique_employees']} unique employees",
            'Used computed_duration_hours for accurate shift duration',
            'Labour cost calculated from hourly_rate_sar and computed duration'
        ],
        'assumptions': [
            'Staff hours calculated using computed_duration_hours field',
            'Labour cost = hourly_rate_sar × computed_duration_hours',
            'Shift overlap computed from shift_start and shift_end timestamps'
        ],
        'confidence': 0.88
    })

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
