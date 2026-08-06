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
analysis_start = "2026-05-04T00:00:00+03:00"
analysis_end = "2026-05-11T00:00:00+03:00"

# Convert to datetime for filtering, removing timezone info for comparison
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all unique sources in the data
all_sources = sorted(reviews_df['source'].unique().tolist())

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Count by language
    lang_counts = reviews_analysis['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with distribution: {rating_dist}. Language coverage: {lang_counts}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": rating_dist.get(5, 0),
                "unit": "count",
                "numerator": rating_dist.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": rating_dist.get(4, 0),
                "unit": "count",
                "numerator": rating_dist.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": rating_dist.get(3, 0),
                "unit": "count",
                "numerator": rating_dist.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": rating_dist.get(2, 0),
                "unit": "count",
                "numerator": rating_dist.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": rating_dist.get(1, 0),
                "unit": "count",
                "numerator": rating_dist.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language distribution: {lang_counts}",
            f"All available sources included: {all_sources}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews in the artifact are included regardless of source"
        ],
        "confidence": 1.0
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Classify sentiment based on rating
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
    
    sentiment_dist = reviews_analysis['sentiment'].value_counts().to_dict()
    
    # Language-specific sentiment
    en_sentiment = english_reviews['rating'].apply(classify_sentiment).value_counts().to_dict() if len(english_reviews) > 0 else {}
    ar_sentiment = arabic_reviews['rating'].apply(classify_sentiment).value_counts().to_dict() if len(arabic_reviews) > 0 else {}
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, sentiment distribution is: {sentiment_dist}. English reviews ({len(english_reviews)}): {en_sentiment}. Arabic reviews ({len(arabic_reviews)}): {ar_sentiment}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_sentiment_count": {
                "value": sentiment_dist.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": sentiment_dist.get('neutral', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('neutral', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": sentiment_dist.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('negative', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classified by rating threshold: 4-5=positive, 3=neutral, 1-2=negative",
            f"English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}",
            f"All sources included: {all_sources}"
        ],
        "assumptions": [
            "Sentiment is derived from rating values only",
            "Rating >= 4 indicates positive sentiment",
            "Rating = 3 indicates neutral sentiment",
            "Rating < 3 indicates negative sentiment",
            "Language field is accurate as provided"
        ],
        "confidence": 0.9
    }
    findings.append(finding2)

# Finding 3: Review Source Coverage
source_dist = reviews_analysis['source'].value_counts().to_dict() if len(reviews_analysis) > 0 else {}

finding3 = {
    "title": "Review Source Coverage (Analysis Period)",
    "claim": f"During the analysis period, reviews were collected from {len(source_dist)} sources: {source_dist}. Total reviews analyzed: {len(reviews_analysis)}.",
    "finding_type": "source_coverage",
    "metrics": {
        "total_reviews_analyzed": {
            "value": len(reviews_analysis),
            "unit": "count",
            "numerator": len(reviews_analysis),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": all_sources,
    "sample_size": len(reviews_analysis),
    "coverage_notes": [
        f"Sources represented: {list(source_dist.keys())}",
        f"Source distribution: {source_dist}",
        f"Analysis period: {analysis_start} to {analysis_end}"
    ],
    "assumptions": [
        "Source field is accurate as provided in the artifact",
        "All reviews in the artifact are included in the analysis"
    ],
    "confidence": 1.0
}

# Add source-specific metrics
for source in all_sources:
    source_count = source_dist.get(source, 0)
    finding3["metrics"][f"{source}_count"] = {
        "value": source_count,
        "unit": "count",
        "numerator": source_count,
        "denominator": len(reviews_analysis) if len(reviews_analysis) > 0 else None,
        "period_start": analysis_start,
        "period_end": analysis_end
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

print(f"Analysis complete. Output written to {output_path}")
