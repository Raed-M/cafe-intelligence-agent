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
analysis_start = "2026-01-05T00:00:00+03:00"
analysis_end = "2026-01-12T00:00:00+03:00"

# Convert to datetime for filtering, removing timezone info for comparison
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Filter reviews to analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    sources_in_period = sorted(reviews_analysis['source'].unique().tolist())
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5.0.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": reviews_analysis['rating'].sum(),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sources_in_period,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews filtered to analysis period {analysis_start} to {analysis_end}",
            f"Sources represented: {', '.join(sources_in_period)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1 and 5",
            "Date field is accurate and timezone-aware",
            "All reviews in artifact are from valid sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment/Topic Classification by Language
# ============================================================================

# Separate reviews by language
reviews_en = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
reviews_ar = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()

# Simple sentiment classification based on rating
def classify_sentiment(rating):
    if pd.isna(rating):
        return 'unknown'
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)

sentiment_counts = reviews_analysis['sentiment'].value_counts()

if len(reviews_analysis) > 0:
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, sentiment distribution shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. English reviews: {len(reviews_en)}, Arabic reviews: {len(reviews_ar)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_reviews": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": sentiment_counts.get('neutral', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": len(reviews_en),
                "unit": "count",
                "numerator": len(reviews_en),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": len(reviews_ar),
                "unit": "count",
                "numerator": len(reviews_ar),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sources_in_period,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classified by rating threshold: 4-5 stars = positive, 3 stars = neutral, 1-2 stars = negative",
            f"Language coverage: {len(reviews_en)} English, {len(reviews_ar)} Arabic",
            f"All reviews in analysis period {analysis_start} to {analysis_end}"
        ],
        "assumptions": [
            "Rating-based sentiment classification is appropriate for this dataset",
            "Language field is accurately populated",
            "No reviews have missing rating values in analysis period"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: High-Rating vs Low-Rating Review Frequency
# ============================================================================

high_rating_reviews = reviews_analysis[reviews_analysis['rating'] >= 4]
low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2]

if len(reviews_analysis) > 0:
    high_pct = (len(high_rating_reviews) / len(reviews_analysis)) * 100
    low_pct = (len(low_rating_reviews) / len(reviews_analysis)) * 100
    
    finding_3 = {
        "title": "High vs Low Rating Review Frequency",
        "claim": f"In the analysis period, {len(high_rating_reviews)} reviews ({high_pct:.1f}%) were rated 4-5 stars (high satisfaction), while {len(low_rating_reviews)} reviews ({low_pct:.1f}%) were rated 1-2 stars (low satisfaction).",
        "finding_type": "rating_frequency",
        "metrics": {
            "high_rating_count": {
                "value": len(high_rating_reviews),
                "unit": "count",
                "numerator": len(high_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "high_rating_percentage": {
                "value": round(high_pct, 1),
                "unit": "percent",
                "numerator": len(high_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "low_rating_count": {
                "value": len(low_rating_reviews),
                "unit": "count",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "low_rating_percentage": {
                "value": round(low_pct, 1),
                "unit": "percent",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sources_in_period,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"High rating defined as 4-5 stars, low rating as 1-2 stars",
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "High and low rating thresholds are appropriate for business context",
            "All reviews have valid rating values"
        ],
        "confidence": 0.95
    }
    findings.append(finding_3)

# ============================================================================
# Write output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
