import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

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
analysis_start = "2026-02-16T00:00:00+03:00"
analysis_end = "2026-02-23T00:00:00+03:00"

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
    rating_dist = analysis_reviews['rating'].value_counts().sort_index().to_dict()
    avg_rating = analysis_reviews['rating'].mean()
    
    finding1 = {
        "title": "Customer Rating Distribution (Feb 16-23, 2026)",
        "claim": f"Average customer rating is {avg_rating:.2f} out of 5, based on {len(analysis_reviews)} reviews during the analysis period.",
        "finding_type": "rating_analysis",
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
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(analysis_reviews)}",
            f"Sources represented: {', '.join(analysis_reviews['source'].unique())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the period are included regardless of language"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment and Topic Classification
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
        elif rating >= 3:
            return 'neutral'
        else:
            return 'negative'
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    
    sentiment_dist = analysis_reviews['sentiment'].value_counts().to_dict()
    
    # Extract topics from text (simple keyword matching)
    topics = {
        'quality': 0,
        'service': 0,
        'price': 0,
        'taste': 0,
        'speed': 0,
        'cleanliness': 0
    }
    
    quality_keywords = ['quality', 'جودة', 'excellent', 'great', 'good', 'bad', 'poor']
    service_keywords = ['service', 'خدمة', 'staff', 'friendly', 'rude', 'helpful']
    price_keywords = ['price', 'expensive', 'cheap', 'سعر', 'غالي']
    taste_keywords = ['taste', 'flavor', 'delicious', 'طعم', 'لذيذ', 'bitter']
    speed_keywords = ['fast', 'slow', 'quick', 'سريع', 'بطيء']
    cleanliness_keywords = ['clean', 'dirty', 'hygiene', 'نظيف', 'وسخ']
    
    for idx, row in analysis_reviews.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ''
        
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
    
    # Find most mentioned topic
    top_topic = max(topics, key=topics.get) if max(topics.values()) > 0 else None
    top_topic_count = topics.get(top_topic, 0) if top_topic else 0
    
    if top_topic and top_topic_count > 0:
        finding2 = {
            "title": "Sentiment Distribution and Topic Frequency",
            "claim": f"Among {len(analysis_reviews)} reviews, {sentiment_dist.get('positive', 0)} are positive (rating ≥4), {sentiment_dist.get('neutral', 0)} are neutral (rating 3), and {sentiment_dist.get('negative', 0)} are negative (rating <3). Most frequently mentioned topic is '{top_topic}' in {top_topic_count} reviews.",
            "finding_type": "sentiment_and_topic_analysis",
            "metrics": {
                "positive_reviews": {
                    "value": sentiment_dist.get('positive', 0),
                    "unit": "count",
                    "numerator": sentiment_dist.get('positive', 0),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_reviews": {
                    "value": sentiment_dist.get('neutral', 0),
                    "unit": "count",
                    "numerator": sentiment_dist.get('neutral', 0),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews": {
                    "value": sentiment_dist.get('negative', 0),
                    "unit": "count",
                    "numerator": sentiment_dist.get('negative', 0),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_topic": {
                    "value": top_topic,
                    "unit": "string",
                    "numerator": top_topic_count,
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"Language distribution: {len(english_reviews)} English, {len(arabic_reviews)} Arabic",
                f"Sentiment classification based on rating thresholds (positive ≥4, neutral 3, negative <3)",
                f"Topic detection uses keyword matching in both English and Arabic",
                f"Reviews with empty text: {analysis_reviews['text'].isna().sum()}"
            ],
            "assumptions": [
                "Rating-based sentiment classification is appropriate for this dataset",
                "Keyword matching captures topic mentions accurately",
                "A review can mention multiple topics",
                "Language field is accurate"
            ],
            "confidence": 0.85
        }
        findings.append(finding2)

# Finding 3: Language Coverage
if len(analysis_reviews) > 0:
    language_dist = analysis_reviews['language'].value_counts().to_dict()
    total_reviews = len(analysis_reviews)
    
    finding3 = {
        "title": "Review Language Coverage",
        "claim": f"Of {total_reviews} reviews in the analysis period, {language_dist.get('en', 0)} are in English and {language_dist.get('ar', 0)} are in Arabic, providing bilingual coverage.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": language_dist.get('en', 0),
                "unit": "count",
                "numerator": language_dist.get('en', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": language_dist.get('ar', 0),
                "unit": "count",
                "numerator": language_dist.get('ar', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Bilingual coverage enables sentiment analysis in both languages",
            f"Sources: {', '.join(analysis_reviews['source'].unique())}"
        ],
        "assumptions": [
            "Language field accurately identifies review language",
            "Both English and Arabic reviews are equally valid for analysis"
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
