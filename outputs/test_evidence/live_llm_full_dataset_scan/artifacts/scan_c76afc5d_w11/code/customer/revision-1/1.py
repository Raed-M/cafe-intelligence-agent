import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-03-23T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-30T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} stars, with distribution: {dict(rating_counts)}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.isoformat().split('T')[0]}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the analysis period are included regardless of language"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "english_percentage": {
                "value": round(100 * language_counts.get('en', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": language_counts.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_percentage": {
                "value": round(100 * language_counts.get('ar', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": language_counts.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution across {len(reviews_analysis)} reviews",
            f"Bilingual review dataset with {len(reviews_df)} total reviews"
        ],
        "assumptions": [
            "Language field accurately reflects review language",
            "No reviews are in languages other than English and Arabic"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: High-Rating vs Low-Rating Review Counts
# ============================================================================

if len(reviews_analysis) > 0:
    high_rating_reviews = len(reviews_analysis[reviews_analysis['rating'] >= 4])
    low_rating_reviews = len(reviews_analysis[reviews_analysis['rating'] <= 2])
    
    finding_3 = {
        "title": "Sentiment Polarity: High vs Low Ratings",
        "claim": f"In the analysis period, {high_rating_reviews} reviews have ratings of 4-5 stars (positive), while {low_rating_reviews} reviews have ratings of 1-2 stars (negative), out of {len(reviews_analysis)} total reviews.",
        "finding_type": "sentiment_polarity",
        "metrics": {
            "high_rating_count": {
                "value": high_rating_reviews,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "low_rating_count": {
                "value": low_rating_reviews,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "high_rating_percentage": {
                "value": round(100 * high_rating_reviews / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": high_rating_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "low_rating_percentage": {
                "value": round(100 * low_rating_reviews / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": low_rating_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"High ratings defined as 4-5 stars",
            f"Low ratings defined as 1-2 stars",
            f"Neutral ratings (3 stars) excluded from this polarity analysis"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "4-5 stars represent positive sentiment, 1-2 stars represent negative sentiment"
        ],
        "confidence": 0.95
    }
    findings.append(finding_3)

# ============================================================================
# Construct output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Output written to {output_path}")
print(f"Findings generated: {len(findings)}")
