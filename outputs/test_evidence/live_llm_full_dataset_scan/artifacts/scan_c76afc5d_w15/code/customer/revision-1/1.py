import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

# Load input/output paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Parse dates
analysis_start = datetime.fromisoformat("2026-04-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-27T00:00:00+03:00")
baseline_start = datetime.fromisoformat("2026-03-23T00:00:00+03:00")
baseline_end = datetime.fromisoformat("2026-04-20T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

baseline_reviews = reviews_df[
    (reviews_df['date'] >= baseline_start) & 
    (reviews_df['date'] < baseline_end)
].copy()

# Get unique sources in the data
all_sources = reviews_df['source'].unique().tolist()

findings = []

# Finding 1: Rating distribution and average for analysis period
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index().to_dict()
    avg_rating = analysis_reviews['rating'].mean()
    total_reviews_analysis = len(analysis_reviews)
    
    finding_1 = {
        "title": "Analysis Period Review Rating Distribution",
        "claim": f"During the analysis period (2026-04-20 to 2026-04-27), {total_reviews_analysis} reviews were collected with an average rating of {avg_rating:.2f} out of 5.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * total_reviews_analysis, 2),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "total_reviews": {
                "value": total_reviews_analysis,
                "unit": "count",
                "numerator": total_reviews_analysis,
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "rating_1_star": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "rating_2_star": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "rating_3_star": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "rating_4_star": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "rating_5_star": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": total_reviews_analysis,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            }
        },
        "source_names": all_sources,
        "sample_size": total_reviews_analysis,
        "coverage_notes": [
            f"Analysis period: 2026-04-20 to 2026-04-27",
            f"Total reviews in analysis period: {total_reviews_analysis}",
            f"Sources represented: {', '.join(all_sources)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are included without filtering"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Language distribution in analysis period
if len(analysis_reviews) > 0:
    lang_counts = analysis_reviews['language'].value_counts().to_dict()
    total_lang_reviews = len(analysis_reviews)
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"In the analysis period, {lang_counts.get('en', 0)} reviews were in English and {lang_counts.get('ar', 0)} reviews were in Arabic out of {total_lang_reviews} total reviews.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": lang_counts.get('en', 0),
                "unit": "count",
                "numerator": lang_counts.get('en', 0),
                "denominator": total_lang_reviews,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": lang_counts.get('ar', 0),
                "unit": "count",
                "numerator": lang_counts.get('ar', 0),
                "denominator": total_lang_reviews,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "other_language_reviews": {
                "value": total_lang_reviews - lang_counts.get('en', 0) - lang_counts.get('ar', 0),
                "unit": "count",
                "numerator": total_lang_reviews - lang_counts.get('en', 0) - lang_counts.get('ar', 0),
                "denominator": total_lang_reviews,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            }
        },
        "source_names": all_sources,
        "sample_size": total_lang_reviews,
        "coverage_notes": [
            f"Analysis period: 2026-04-20 to 2026-04-27",
            f"Language field populated for all {total_lang_reviews} reviews"
        ],
        "assumptions": [
            "Language classification is accurate",
            "Language values are limited to 'en', 'ar', or other ISO codes"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# Finding 3: Comparison of average rating between analysis and baseline periods
if len(analysis_reviews) > 0 and len(baseline_reviews) > 0:
    avg_rating_analysis = analysis_reviews['rating'].mean()
    avg_rating_baseline = baseline_reviews['rating'].mean()
    rating_change = avg_rating_analysis - avg_rating_baseline
    
    finding_3 = {
        "title": "Rating Trend: Analysis vs Baseline Period",
        "claim": f"Average rating in the analysis period (2026-04-20 to 2026-04-27) was {avg_rating_analysis:.2f}, compared to {avg_rating_baseline:.2f} in the baseline period (2026-03-23 to 2026-04-20), a change of {rating_change:+.2f} points.",
        "finding_type": "rating_comparison",
        "metrics": {
            "average_rating_analysis_period": {
                "value": round(avg_rating_analysis, 2),
                "unit": "stars",
                "numerator": round(avg_rating_analysis * len(analysis_reviews), 2),
                "denominator": len(analysis_reviews),
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            },
            "average_rating_baseline_period": {
                "value": round(avg_rating_baseline, 2),
                "unit": "stars",
                "numerator": round(avg_rating_baseline * len(baseline_reviews), 2),
                "denominator": len(baseline_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-04-20T00:00:00+03:00"
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": round(rating_change, 2),
                "denominator": None,
                "period_start": "2026-04-20T00:00:00+03:00",
                "period_end": "2026-04-27T00:00:00+03:00"
            }
        },
        "source_names": all_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period sample size: {len(analysis_reviews)} reviews",
            f"Baseline period sample size: {len(baseline_reviews)} reviews",
            f"Baseline period: 2026-03-23 to 2026-04-20 (4 weeks)"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate",
            "Baseline period represents typical performance for comparison"
        ],
        "confidence": 0.90
    }
    findings.append(finding_3)

# Construct output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
