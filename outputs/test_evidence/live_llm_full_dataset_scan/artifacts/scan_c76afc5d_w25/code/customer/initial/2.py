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
analysis_start = "2026-06-29T00:00:00+03:00"
analysis_end = "2026-07-06T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Handle timezone awareness mismatch
if reviews_df['date'].dt.tz is None:
    # If reviews_df dates are tz-naive, localize them to UTC then convert to the analysis timezone
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC').dt.tz_convert(analysis_start_dt.tz)
else:
    # If reviews_df dates are tz-aware but different timezone, convert to analysis timezone
    reviews_df['date'] = reviews_df['date'].dt.tz_convert(analysis_start_dt.tz)

# Ensure comparison datetimes have matching timezone
analysis_start_dt = analysis_start_dt.tz_convert(reviews_df['date'].dt.tz[0] if reviews_df['date'].dt.tz is not None else None)
analysis_end_dt = analysis_end_dt.tz_convert(reviews_df['date'].dt.tz[0] if reviews_df['date'].dt.tz is not None else None)

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
        "title": "Review Rating Distribution and Language Coverage",
        "claim": f"During the analysis period (2026-06-29 to 2026-07-06), {len(analysis_reviews)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. English reviews comprised {language_counts.get('en', 0)} reviews and Arabic reviews comprised {language_counts.get('ar', 0)} reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": language_counts.get('en', 0),
                "unit": "count",
                "numerator": language_counts.get('en', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": language_counts.get('ar', 0),
                "unit": "count",
                "numerator": language_counts.get('ar', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-29 to 2026-07-06",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(analysis_reviews)}",
            f"Language distribution: {dict(language_counts)}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1-5",
            "Date field is accurate and timezone-aware",
            "Language classification is accurate"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Sentiment Classification by Language
if len(analysis_reviews) > 0:
    # Classify sentiment based on rating
    def classify_sentiment(rating):
        if pd.isna(rating):
            return "unknown"
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    
    # Count by sentiment and language
    sentiment_lang = analysis_reviews.groupby(['sentiment', 'language']).size().reset_index(name='count')
    
    positive_count = len(analysis_reviews[analysis_reviews['sentiment'] == 'positive'])
    negative_count = len(analysis_reviews[analysis_reviews['sentiment'] == 'negative'])
    neutral_count = len(analysis_reviews[analysis_reviews['sentiment'] == 'neutral'])
    
    if positive_count > 0 or negative_count > 0:
        finding_2 = {
            "title": "Sentiment Distribution by Language",
            "claim": f"Among {len(analysis_reviews)} reviews, {positive_count} were classified as positive (rating ≥4), {negative_count} as negative (rating <3), and {neutral_count} as neutral (rating=3). Positive sentiment reviews comprised {round(100*positive_count/len(analysis_reviews), 1)}% of all reviews.",
            "finding_type": "sentiment_classification",
            "metrics": {
                "positive_reviews": {
                    "value": positive_count,
                    "unit": "count",
                    "numerator": positive_count,
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews": {
                    "value": negative_count,
                    "unit": "count",
                    "numerator": negative_count,
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_reviews": {
                    "value": neutral_count,
                    "unit": "count",
                    "numerator": neutral_count,
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_percentage": {
                    "value": round(100*positive_count/len(analysis_reviews), 1),
                    "unit": "percent",
                    "numerator": positive_count,
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"Sentiment classification based on rating thresholds: positive (≥4), neutral (=3), negative (<3)",
                f"Language breakdown: {dict(analysis_reviews['language'].value_counts())}",
                f"Sentiment breakdown: {dict(analysis_reviews['sentiment'].value_counts())}"
            ],
            "assumptions": [
                "Rating-based sentiment classification is appropriate for this dataset",
                "All reviews have valid rating values",
                "No reviews were excluded due to missing data"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding_3 = {
        "title": "Review Source Distribution",
        "claim": f"Reviews were collected from {len(source_counts)} sources during the analysis period. The primary sources were: {', '.join([f'{source} ({count} reviews)' for source, count in source_counts.head(3).items()])}.",
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
            f"Source distribution: {dict(source_counts)}",
            f"Total reviews analyzed: {len(analysis_reviews)}",
            f"Analysis period: 2026-06-29 to 2026-07-06"
        ],
        "assumptions": [
            "Source field accurately identifies review origin",
            "All sources are equally reliable"
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
