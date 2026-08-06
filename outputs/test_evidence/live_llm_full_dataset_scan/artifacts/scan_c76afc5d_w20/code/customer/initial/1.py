import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-05-25"
analysis_end = "2026-06-01"

# Convert date columns to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding1 = {
        "title": "Review Rating Distribution (May 25 - Jun 1, 2026)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {len(reviews_analysis)} reviews in the analysis period.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1-5",
            "Date filtering uses review date field",
            "All reviews in artifact are included without exclusion"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
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
    
    if len(sentiment_counts) > 0:
        finding2 = {
            "title": "Sentiment Distribution by Rating (May 25 - Jun 1, 2026)",
            "claim": f"Sentiment analysis shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews based on rating thresholds.",
            "finding_type": "sentiment_classification",
            "metrics": {
                "positive_reviews": {
                    "value": int(sentiment_counts.get('positive', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('positive', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "neutral_reviews": {
                    "value": int(sentiment_counts.get('neutral', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('neutral', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                },
                "negative_reviews": {
                    "value": int(sentiment_counts.get('negative', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('negative', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-05-25T00:00:00+03:00",
                    "period_end": "2026-06-01T00:00:00+03:00"
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"English reviews: {len(english_reviews)}",
                f"Arabic reviews: {len(arabic_reviews)}",
                "Sentiment classification based on rating: 4-5=positive, 3=neutral, 1-2=negative"
            ],
            "assumptions": [
                "Rating-based sentiment classification is appropriate for this dataset",
                "All reviews have valid rating values",
                "Language field accurately identifies review language"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# Finding 3: Review Volume Comparison with Previous Period
previous_start = "2026-05-18"
previous_end = "2026-05-25"

reviews_previous = reviews_df[
    (reviews_df['date'] >= previous_start) & 
    (reviews_df['date'] < previous_end)
].copy()

if len(reviews_analysis) > 0 and len(reviews_previous) > 0:
    volume_change = len(reviews_analysis) - len(reviews_previous)
    pct_change = (volume_change / len(reviews_previous) * 100) if len(reviews_previous) > 0 else 0
    
    finding3 = {
        "title": "Review Volume Trend (Week-over-Week)",
        "claim": f"Review volume in the analysis period ({len(reviews_analysis)} reviews) compared to the previous week ({len(reviews_previous)} reviews) shows a change of {volume_change} reviews ({pct_change:+.1f}%).",
        "finding_type": "volume_trend",
        "metrics": {
            "current_period_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "previous_period_reviews": {
                "value": len(reviews_previous),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "volume_change": {
                "value": volume_change,
                "unit": "count",
                "numerator": volume_change,
                "denominator": len(reviews_previous),
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            },
            "percent_change": {
                "value": round(pct_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-25T00:00:00+03:00",
                "period_end": "2026-06-01T00:00:00+03:00"
            }
        },
        "source_names": list(set(list(reviews_analysis['source'].unique()) + list(reviews_previous['source'].unique()))),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: May 25 - Jun 1, 2026 ({len(reviews_analysis)} reviews)",
            f"Previous period: May 18 - May 25, 2026 ({len(reviews_previous)} reviews)",
            "Comparison is week-over-week"
        ],
        "assumptions": [
            "Date filtering uses review date field",
            "Both periods have complete data coverage",
            "Week boundaries are consistent"
        ],
        "confidence": 0.92
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
