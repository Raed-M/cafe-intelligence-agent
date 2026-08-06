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
analysis_start = datetime.fromisoformat("2026-05-25T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-06-01T00:00:00+03:00")

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
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star and {int(rating_counts.get(1, 0))} one-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC",
            "All reviews in the analysis period are included regardless of language"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {int(language_counts.get('en', 0))} are in English and {int(language_counts.get('ar', 0))} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "english_percentage": {
                "value": round(100 * int(language_counts.get('en', 0)) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Language field populated for all {len(reviews_analysis)} reviews"
        ],
        "assumptions": [
            "Language classification is accurate",
            "Language values are either 'en' or 'ar'"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# Finding 3: Sentiment Topics (High-level frequency analysis)
if len(reviews_analysis) > 0:
    # Simple keyword-based topic detection
    topics = {
        'quality': 0,
        'service': 0,
        'price': 0,
        'taste': 0,
        'speed': 0,
        'cleanliness': 0
    }
    
    quality_keywords = ['quality', 'good', 'excellent', 'poor', 'bad', 'جودة', 'ممتاز', 'سيء']
    service_keywords = ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف']
    price_keywords = ['price', 'expensive', 'cheap', 'cost', 'سعر', 'غالي', 'رخيص']
    taste_keywords = ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملح']
    speed_keywords = ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    cleanliness_keywords = ['clean', 'dirty', 'hygiene', 'neat', 'نظيف', 'قذر', 'صحة']
    
    for idx, row in reviews_analysis.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        
        if any(kw in text for kw in quality_keywords):
            topics['quality'] += 1
        if any(kw in text for kw in service_keywords):
            topics['service'] += 1
        if any(kw in text for kw in price_keywords):
            topics['price'] += 1
        if any(kw in text for kw in taste_keywords):
            topics['taste'] += 1
        if any(kw in text for kw in speed_keywords):
            topics['speed'] += 1
        if any(kw in text for kw in cleanliness_keywords):
            topics['cleanliness'] += 1
    
    # Find top topics
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if top_topics[0][1] > 0:  # Only report if there are mentions
        finding_3 = {
            "title": "Review Topic Frequency",
            "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, the most frequently mentioned topics are: {top_topics[0][0]} ({top_topics[0][1]} mentions), {top_topics[1][0]} ({top_topics[1][1]} mentions), and {top_topics[2][0]} ({top_topics[2][1]} mentions).",
            "finding_type": "topic_frequency",
            "metrics": {
                f"{top_topics[0][0]}_mentions": {
                    "value": top_topics[0][1],
                    "unit": "mentions",
                    "numerator": top_topics[0][1],
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                f"{top_topics[1][0]}_mentions": {
                    "value": top_topics[1][1],
                    "unit": "mentions",
                    "numerator": top_topics[1][1],
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                f"{top_topics[2][0]}_mentions": {
                    "value": top_topics[2][1],
                    "unit": "mentions",
                    "numerator": top_topics[2][1],
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                f"Topic detection based on keyword matching in review text",
                f"Reviews with non-empty text: {len(reviews_analysis[reviews_analysis['text'].notna()])}"
            ],
            "assumptions": [
                "Topic keywords are representative of each category",
                "A review can mention multiple topics",
                "Keyword matching is case-insensitive",
                "Both English and Arabic keywords are included"
            ],
            "confidence": 0.70
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
