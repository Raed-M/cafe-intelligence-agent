import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

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
analysis_start = datetime.fromisoformat("2026-03-23T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-30T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Get source names from the data
    source_names = analysis_reviews['source'].unique().tolist()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-03-23 to 2026-03-30), the average rating across {len(analysis_reviews)} reviews is {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star ratings and {int(rating_counts.get(1, 0))} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-03-23 to 2026-03-30",
            f"Total reviews in analysis period: {len(analysis_reviews)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC",
            "All reviews in the dataset are from the specified sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(analysis_reviews)} reviews in the analysis period, {int(language_counts.get('en', 0))} are in English and {int(language_counts.get('ar', 0))} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Language distribution across {len(analysis_reviews)} reviews",
            f"Bilingual coverage enables sentiment analysis in both languages"
        ],
        "assumptions": [
            "Language field accurately reflects review language",
            "Reviews are classified as either 'en' or 'ar'"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Sentiment Analysis by Language
if len(analysis_reviews) > 0:
    # Simple sentiment classification based on rating
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
    )
    
    # Separate by language
    en_reviews = analysis_reviews[analysis_reviews['language'] == 'en']
    ar_reviews = analysis_reviews[analysis_reviews['language'] == 'ar']
    
    en_positive = len(en_reviews[en_reviews['sentiment'] == 'positive'])
    ar_positive = len(ar_reviews[ar_reviews['sentiment'] == 'positive'])
    
    if len(en_reviews) > 0 or len(ar_reviews) > 0:
        finding3 = {
            "title": "Sentiment Distribution by Language",
            "claim": f"Among {len(en_reviews)} English reviews, {en_positive} are positive (rating ≥4). Among {len(ar_reviews)} Arabic reviews, {ar_positive} are positive (rating ≥4).",
            "finding_type": "sentiment_by_language",
            "metrics": {
                "english_positive_count": {
                    "value": en_positive,
                    "unit": "reviews",
                    "numerator": en_positive,
                    "denominator": len(en_reviews) if len(en_reviews) > 0 else None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "arabic_positive_count": {
                    "value": ar_positive,
                    "unit": "reviews",
                    "numerator": ar_positive,
                    "denominator": len(ar_reviews) if len(ar_reviews) > 0 else None,
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                }
            },
            "source_names": source_names,
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"English reviews: {len(en_reviews)}",
                f"Arabic reviews: {len(ar_reviews)}",
                "Sentiment classified as positive (rating ≥4), neutral (rating 3), or negative (rating ≤2)"
            ],
            "assumptions": [
                "Rating ≥4 indicates positive sentiment",
                "Rating ≤2 indicates negative sentiment",
                "Rating 3 indicates neutral sentiment",
                "Language classification is accurate"
            ],
            "confidence": 0.90
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
print(f"Output written to {output_path}")
