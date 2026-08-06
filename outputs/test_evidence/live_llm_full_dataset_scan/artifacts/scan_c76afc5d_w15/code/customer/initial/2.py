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
analysis_start = "2026-04-20T00:00:00+03:00"
analysis_end = "2026-04-27T00:00:00+03:00"

# Convert to datetime for filtering - use UTC and strip timezone info for comparison
analysis_start_dt = pd.to_datetime(analysis_start, utc=True).tz_convert(None)
analysis_end_dt = pd.to_datetime(analysis_end, utc=True).tz_convert(None)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True).dt.tz_convert(None)
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
        "title": "Review Rating Distribution and Average (Apr 20-27, 2026)",
        "claim": f"During the analysis period, the average review rating was {avg_rating:.2f} out of 5, based on {len(analysis_reviews)} reviews. Rating distribution shows {rating_counts.get(5, 0)} five-star, {rating_counts.get(4, 0)} four-star, {rating_counts.get(3, 0)} three-star, {rating_counts.get(2, 0)} two-star, and {rating_counts.get(1, 0)} one-star reviews.",
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
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language distribution: {dict(language_counts)}",
            f"Total reviews in dataset: {len(reviews_df)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews in the dataset are included without exclusion criteria"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Sentiment/Topic Analysis by Language
if len(analysis_reviews) > 0:
    # Separate by language
    english_reviews = analysis_reviews[analysis_reviews['language'] == 'en'].copy()
    arabic_reviews = analysis_reviews[analysis_reviews['language'] == 'ar'].copy()
    
    # Analyze sentiment by rating groups
    positive_reviews = analysis_reviews[analysis_reviews['rating'] >= 4]
    negative_reviews = analysis_reviews[analysis_reviews['rating'] <= 2]
    neutral_reviews = analysis_reviews[analysis_reviews['rating'] == 3]
    
    # Extract common words/topics from positive reviews (English)
    positive_en = english_reviews[english_reviews['rating'] >= 4]
    positive_ar = arabic_reviews[arabic_reviews['rating'] >= 4]
    
    # Count non-empty reviews
    non_empty_reviews = analysis_reviews[analysis_reviews['text'].notna() & (analysis_reviews['text'].str.len() > 0)]
    
    finding_2 = {
        "title": "Sentiment Distribution by Language (Apr 20-27, 2026)",
        "claim": f"Among {len(analysis_reviews)} reviews, {len(positive_reviews)} ({100*len(positive_reviews)/len(analysis_reviews):.1f}%) were positive (4-5 stars), {len(neutral_reviews)} ({100*len(neutral_reviews)/len(analysis_reviews):.1f}%) were neutral (3 stars), and {len(negative_reviews)} ({100*len(negative_reviews)/len(analysis_reviews):.1f}%) were negative (1-2 stars). English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_reviews_count": {
                "value": len(positive_reviews),
                "unit": "count",
                "numerator": len(positive_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_reviews_percentage": {
                "value": round(100*len(positive_reviews)/len(analysis_reviews), 1),
                "unit": "percent",
                "numerator": len(positive_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews_count": {
                "value": len(neutral_reviews),
                "unit": "count",
                "numerator": len(neutral_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_count": {
                "value": len(negative_reviews),
                "unit": "count",
                "numerator": len(negative_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "non_empty_reviews_count": {
                "value": len(non_empty_reviews),
                "unit": "count",
                "numerator": len(non_empty_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}",
            f"Reviews with text content: {len(non_empty_reviews)}",
            "Sentiment classification based on rating values only"
        ],
        "assumptions": [
            "Positive sentiment = rating 4-5 stars",
            "Neutral sentiment = rating 3 stars",
            "Negative sentiment = rating 1-2 stars",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding_3 = {
        "title": "Review Source Distribution (Apr 20-27, 2026)",
        "claim": f"Reviews were collected from {len(source_counts)} sources during the analysis period. Distribution: {', '.join([f'{source}: {count}' for source, count in source_counts.items()])}.",
        "finding_type": "source_distribution",
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
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Source breakdown: {dict(source_counts)}"
        ],
        "assumptions": [
            "Source field accurately identifies review platform/channel",
            "All reviews in the dataset are included"
        ],
        "confidence": 0.95
    }
    
    # Add source-specific metrics
    for source, count in source_counts.items():
        finding_3['metrics'][f'{source}_count'] = {
            "value": int(count),
            "unit": "count",
            "numerator": int(count),
            "denominator": len(analysis_reviews),
            "period_start": analysis_start,
            "period_end": analysis_end
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
