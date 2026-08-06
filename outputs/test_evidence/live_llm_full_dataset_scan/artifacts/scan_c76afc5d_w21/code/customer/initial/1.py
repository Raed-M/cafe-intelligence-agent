import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-06-01T00:00:00+03:00"
analysis_end = "2026-06-08T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Language distribution
    language_counts = analysis_reviews['language'].value_counts()
    
    finding_1 = {
        "title": "Review Rating Distribution and Average (Jun 1-8, 2026)",
        "claim": f"During the analysis period (Jun 1-8, 2026), the cafe received {len(analysis_reviews)} reviews with an average rating of {avg_rating:.2f} out of 5. Rating distribution shows {rating_counts.get(5, 0)} five-star, {rating_counts.get(4, 0)} four-star, {rating_counts.get(3, 0)} three-star, {rating_counts.get(2, 0)} two-star, and {rating_counts.get(1, 0)} one-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: Jun 1-8, 2026",
            f"Total reviews in period: {len(analysis_reviews)}",
            f"Language distribution: {dict(language_counts)}",
            f"Sources: {', '.join(analysis_reviews['source'].unique())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "All reviews in the dataset have valid dates",
            "Reviews are independent observations"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Sentiment Classification by Language
if len(analysis_reviews) > 0:
    # Separate by language
    english_reviews = analysis_reviews[analysis_reviews['language'] == 'en'].copy()
    arabic_reviews = analysis_reviews[analysis_reviews['language'] == 'ar'].copy()
    
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
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    
    sentiment_counts = analysis_reviews['sentiment'].value_counts()
    
    # Language-specific sentiment
    if len(english_reviews) > 0:
        english_reviews['sentiment'] = english_reviews['rating'].apply(classify_sentiment)
        english_sentiment = english_reviews['sentiment'].value_counts()
    else:
        english_sentiment = {}
    
    if len(arabic_reviews) > 0:
        arabic_reviews['sentiment'] = arabic_reviews['rating'].apply(classify_sentiment)
        arabic_sentiment = arabic_reviews['sentiment'].value_counts()
    else:
        arabic_sentiment = {}
    
    finding_2 = {
        "title": "Sentiment Distribution by Language (Jun 1-8, 2026)",
        "claim": f"Sentiment analysis of {len(analysis_reviews)} reviews shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. English reviews ({len(english_reviews)}): {dict(english_sentiment)}. Arabic reviews ({len(arabic_reviews)}): {dict(arabic_sentiment)}.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "positive_reviews": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('positive', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('neutral', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('negative', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_review_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_review_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: Jun 1-8, 2026",
            f"Total reviews analyzed: {len(analysis_reviews)}",
            f"English reviews: {len(english_reviews)}",
            f"Arabic reviews: {len(arabic_reviews)}",
            f"Sentiment classification based on rating thresholds: 4-5 stars = positive, 3 stars = neutral, 1-2 stars = negative"
        ],
        "assumptions": [
            "Sentiment is determined by rating value",
            "Rating thresholds are: positive (4-5), neutral (3), negative (1-2)",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding_3 = {
        "title": "Review Source Distribution (Jun 1-8, 2026)",
        "claim": f"Reviews were collected from {len(source_counts)} sources during the analysis period. Distribution: {dict(source_counts)}. This represents {len(analysis_reviews)} total reviews with coverage across multiple platforms.",
        "finding_type": "source_coverage",
        "metrics": {
            "total_sources": {
                "value": len(source_counts),
                "unit": "count",
                "numerator": len(source_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: Jun 1-8, 2026",
            f"Total reviews: {len(analysis_reviews)}",
            f"Source breakdown: {dict(source_counts)}",
            f"All reviews in analysis period have valid source information"
        ],
        "assumptions": [
            "Source field accurately identifies review platform",
            "All reviews have been properly categorized by source"
        ],
        "confidence": 0.95
    }
    findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
