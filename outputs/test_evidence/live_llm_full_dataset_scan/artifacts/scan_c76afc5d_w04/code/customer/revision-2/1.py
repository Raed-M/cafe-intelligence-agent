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
analysis_start = datetime.fromisoformat("2026-02-02T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-09T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

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
    
    # Get source names from the data
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (Feb 2-9, 2026), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5.0.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            }
        }
    }
    
    # Add rating distribution to metrics
    for rating in sorted(rating_counts.index):
        count = int(rating_counts[rating])
        finding1["metrics"][f"rating_{int(rating)}_count"] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": "2026-02-02T00:00:00+03:00",
            "period_end": "2026-02-09T00:00:00+03:00"
        }
    
    finding1["source_names"] = source_names
    finding1["sample_size"] = len(reviews_analysis)
    finding1["coverage_notes"] = [
        f"Reviews from {len(source_names)} source(s): {', '.join(source_names)}",
        f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
    ]
    finding1["assumptions"] = [
        "All reviews in the artifact are valid and represent genuine customer feedback",
        "Rating scale is 1-5 stars",
        "Analysis period is 2026-02-02 to 2026-02-09 (UTC+3)"
    ]
    finding1["confidence"] = 0.95
    
    findings.append(finding1)

# Finding 2: Sentiment/Topic Classification
if len(reviews_analysis) > 0:
    # Classify reviews by language and extract topics
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple topic extraction based on keywords
    topics_found = {}
    
    # English topics
    en_keywords = {
        'quality': ['quality', 'fresh', 'taste', 'flavor', 'delicious', 'excellent'],
        'service': ['service', 'staff', 'friendly', 'quick', 'slow', 'wait'],
        'price': ['price', 'expensive', 'cheap', 'value', 'cost'],
        'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary'],
        'atmosphere': ['atmosphere', 'ambiance', 'cozy', 'comfortable', 'noisy']
    }
    
    # Arabic topics
    ar_keywords = {
        'quality': ['جودة', 'طازة', 'طعم', 'لذيذ', 'ممتاز'],
        'service': ['خدمة', 'موظفين', 'ودود', 'سريع', 'بطيء'],
        'price': ['سعر', 'غالي', 'رخيص', 'قيمة'],
        'cleanliness': ['نظيف', 'وسخ', 'نظافة'],
        'atmosphere': ['أجواء', 'مريح', 'مزدحم']
    }
    
    topic_counts = {}
    
    for idx, row in english_reviews.iterrows():
        if pd.notna(row['text']) and len(str(row['text'])) > 0:
            text_lower = str(row['text']).lower()
            for topic, keywords in en_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    if topic not in topic_counts:
                        topic_counts[topic] = 0
                    topic_counts[topic] += 1
    
    for idx, row in arabic_reviews.iterrows():
        if pd.notna(row['text']) and len(str(row['text'])) > 0:
            text = str(row['text'])
            for topic, keywords in ar_keywords.items():
                if any(kw in text for kw in keywords):
                    if topic not in topic_counts:
                        topic_counts[topic] = 0
                    topic_counts[topic] += 1
    
    if topic_counts:
        finding2 = {
            "title": "Review Topics and Sentiment Themes",
            "claim": f"Analysis of {len(reviews_analysis)} reviews identified {len(topic_counts)} main topics. Quality-related comments appeared in {topic_counts.get('quality', 0)} reviews, service in {topic_counts.get('service', 0)}, and price in {topic_counts.get('price', 0)}.",
            "finding_type": "topic_analysis",
            "metrics": {
                "reviews_with_quality_mention": {
                    "value": topic_counts.get('quality', 0),
                    "unit": "count",
                    "numerator": topic_counts.get('quality', 0),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "reviews_with_service_mention": {
                    "value": topic_counts.get('service', 0),
                    "unit": "count",
                    "numerator": topic_counts.get('service', 0),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "reviews_with_price_mention": {
                    "value": topic_counts.get('price', 0),
                    "unit": "count",
                    "numerator": topic_counts.get('price', 0),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"English reviews: {len(english_reviews)}",
                f"Arabic reviews: {len(arabic_reviews)}",
                "Topics identified through keyword matching in original language"
            ],
            "assumptions": [
                "Topic keywords are representative of each category",
                "A review may mention multiple topics",
                "Keyword matching is case-insensitive for English, exact for Arabic"
            ],
            "confidence": 0.75
        }
        
        findings.append(finding2)

# Finding 3: High vs Low Rating Comparison
if len(reviews_analysis) > 0:
    high_ratings = reviews_analysis[reviews_analysis['rating'] >= 4]
    low_ratings = reviews_analysis[reviews_analysis['rating'] <= 2]
    
    if len(high_ratings) > 0 and len(low_ratings) > 0:
        finding3 = {
            "title": "High vs Low Rating Review Comparison",
            "claim": f"Of {len(reviews_analysis)} reviews, {len(high_ratings)} ({100*len(high_ratings)/len(reviews_analysis):.1f}%) gave ratings of 4-5 stars, while {len(low_ratings)} ({100*len(low_ratings)/len(reviews_analysis):.1f}%) gave ratings of 1-2 stars.",
            "finding_type": "rating_comparison",
            "metrics": {
                "high_rating_count": {
                    "value": len(high_ratings),
                    "unit": "count",
                    "numerator": len(high_ratings),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "high_rating_percentage": {
                    "value": round(100*len(high_ratings)/len(reviews_analysis), 1),
                    "unit": "percent",
                    "numerator": len(high_ratings),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "low_rating_count": {
                    "value": len(low_ratings),
                    "unit": "count",
                    "numerator": len(low_ratings),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "low_rating_percentage": {
                    "value": round(100*len(low_ratings)/len(reviews_analysis), 1),
                    "unit": "percent",
                    "numerator": len(low_ratings),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"High ratings (4-5 stars): {len(high_ratings)} reviews",
                f"Low ratings (1-2 stars): {len(low_ratings)} reviews",
                f"Neutral ratings (3 stars): {len(reviews_analysis) - len(high_ratings) - len(low_ratings)} reviews"
            ],
            "assumptions": [
                "Rating scale is 1-5 stars",
                "High ratings defined as 4-5 stars",
                "Low ratings defined as 1-2 stars"
            ],
            "confidence": 0.95
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
