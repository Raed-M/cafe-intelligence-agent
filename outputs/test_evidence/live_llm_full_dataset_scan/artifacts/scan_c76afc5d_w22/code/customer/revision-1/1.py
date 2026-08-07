import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define analysis period
analysis_start = "2026-06-08"
analysis_end = "2026-06-15"

# Convert review dates to datetime for filtering
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    total_reviews = len(reviews_analysis)
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, {total_reviews} reviews were collected with an average rating of {avg_rating:.2f}. Rating distribution: {rating_counts}. Language coverage: {language_dist}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * total_reviews, 2),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "rating_5_count": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "rating_4_count": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "rating_3_count": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "rating_2_count": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "rating_1_count": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": total_reviews,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language distribution: {language_dist}",
            f"Sources included: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are valid and complete",
            "Rating scale is 1-5 stars"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment/Topic Classification by Language
# ============================================================================

if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Identify positive (4-5 stars) and negative (1-2 stars) reviews
    positive_threshold = 4
    negative_threshold = 2
    
    english_positive = len(english_reviews[english_reviews['rating'] >= positive_threshold])
    english_negative = len(english_reviews[english_reviews['rating'] <= negative_threshold])
    english_total = len(english_reviews)
    
    arabic_positive = len(arabic_reviews[arabic_reviews['rating'] >= positive_threshold])
    arabic_negative = len(arabic_reviews[arabic_reviews['rating'] <= negative_threshold])
    arabic_total = len(arabic_reviews)
    
    if english_total > 0 or arabic_total > 0:
        finding_2 = {
            "title": "Sentiment Distribution by Language",
            "claim": f"English reviews (n={english_total}): {english_positive} positive (≥4 stars), {english_negative} negative (≤2 stars). Arabic reviews (n={arabic_total}): {arabic_positive} positive (≥4 stars), {arabic_negative} negative (≤2 stars).",
            "finding_type": "sentiment_distribution",
            "metrics": {
                "english_reviews_total": {
                    "value": english_total,
                    "unit": "count",
                    "numerator": english_total,
                    "denominator": None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                },
                "english_positive_count": {
                    "value": english_positive,
                    "unit": "count",
                    "numerator": english_positive,
                    "denominator": english_total if english_total > 0 else None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                },
                "english_negative_count": {
                    "value": english_negative,
                    "unit": "count",
                    "numerator": english_negative,
                    "denominator": english_total if english_total > 0 else None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                },
                "arabic_reviews_total": {
                    "value": arabic_total,
                    "unit": "count",
                    "numerator": arabic_total,
                    "denominator": None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                },
                "arabic_positive_count": {
                    "value": arabic_positive,
                    "unit": "count",
                    "numerator": arabic_positive,
                    "denominator": arabic_total if arabic_total > 0 else None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                },
                "arabic_negative_count": {
                    "value": arabic_negative,
                    "unit": "count",
                    "numerator": arabic_negative,
                    "denominator": arabic_total if arabic_total > 0 else None,
                    "period_start": analysis_start + "T00:00:00+03:00",
                    "period_end": analysis_end + "T00:00:00+03:00"
                }
            },
            "source_names": source_names,
            "sample_size": total_reviews,
            "coverage_notes": [
                f"English reviews: {english_total} ({100*english_total/total_reviews:.1f}% of total)",
                f"Arabic reviews: {arabic_total} ({100*arabic_total/total_reviews:.1f}% of total)",
                "Positive defined as rating ≥ 4 stars",
                "Negative defined as rating ≤ 2 stars"
            ],
            "assumptions": [
                "Language field is accurate",
                "Rating scale is 1-5 stars",
                "Positive/negative thresholds are appropriate for business context"
            ],
            "confidence": 1.0
        }
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Review Volume Comparison (Analysis vs Previous Period)
# ============================================================================

previous_start = "2026-06-01"
previous_end = "2026-06-08"

reviews_previous = reviews_df[
    (reviews_df['date'] >= previous_start) & 
    (reviews_df['date'] < previous_end)
].copy()

if len(reviews_analysis) > 0 and len(reviews_previous) > 0:
    previous_total = len(reviews_previous)
    analysis_total = len(reviews_analysis)
    volume_change = analysis_total - previous_total
    pct_change = (volume_change / previous_total * 100) if previous_total > 0 else 0
    
    finding_3 = {
        "title": "Review Volume Trend",
        "claim": f"Review volume in analysis period ({analysis_start} to {analysis_end}): {analysis_total} reviews. Previous period ({previous_start} to {previous_end}): {previous_total} reviews. Change: {volume_change} reviews ({pct_change:+.1f}%).",
        "finding_type": "volume_trend",
        "metrics": {
            "analysis_period_reviews": {
                "value": analysis_total,
                "unit": "count",
                "numerator": analysis_total,
                "denominator": None,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "previous_period_reviews": {
                "value": previous_total,
                "unit": "count",
                "numerator": previous_total,
                "denominator": None,
                "period_start": previous_start + "T00:00:00+03:00",
                "period_end": previous_end + "T00:00:00+03:00"
            },
            "volume_change": {
                "value": volume_change,
                "unit": "count",
                "numerator": volume_change,
                "denominator": previous_total,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            },
            "percent_change": {
                "value": round(pct_change, 1),
                "unit": "percent",
                "numerator": volume_change,
                "denominator": previous_total,
                "period_start": analysis_start + "T00:00:00+03:00",
                "period_end": analysis_end + "T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": analysis_total,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end} ({analysis_total} reviews)",
            f"Previous period: {previous_start} to {previous_end} ({previous_total} reviews)",
            "Both periods are 7 days"
        ],
        "assumptions": [
            "Review dates are accurate",
            "No systematic changes in review collection methodology between periods",
            "Comparison is valid across same day-of-week distribution"
        ],
        "confidence": 1.0
    }
    findings.append(finding_3)

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
