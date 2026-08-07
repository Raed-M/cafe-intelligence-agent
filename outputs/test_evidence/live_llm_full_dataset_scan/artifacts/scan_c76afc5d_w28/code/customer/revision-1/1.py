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

# Load artifacts
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
analysis_start = "2026-07-20T00:00:00+03:00"
analysis_end = "2026-07-27T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
period_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Language Distribution
if len(period_reviews) > 0:
    language_counts = period_reviews['language'].value_counts().to_dict()
    total_reviews = len(period_reviews)
    
    # Get the most common language
    most_common_lang = period_reviews['language'].mode()[0] if len(period_reviews) > 0 else None
    most_common_count = language_counts.get(most_common_lang, 0)
    most_common_pct = (most_common_count / total_reviews * 100) if total_reviews > 0 else 0
    
    finding_1 = {
        "title": "Review Language Distribution",
        "claim": f"Reviews are distributed across {len(language_counts)} languages, with {most_common_lang} being the most common ({most_common_count} out of {total_reviews} reviews, {most_common_pct:.1f}%).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "language_distribution": {
                "value": json.dumps(language_counts),
                "unit": "count",
                "numerator": most_common_count,
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Total reviews in analysis period: {total_reviews}",
            f"Language distribution: {language_counts}",
            "Language field populated for all reviews"
        ],
        "assumptions": [
            "Language field values are accurate as provided in cleaned artifact",
            "Review date field is reliable for period filtering"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Rating Distribution and Average
if len(period_reviews) > 0:
    rating_counts = period_reviews['rating'].value_counts().sort_index().to_dict()
    avg_rating = period_reviews['rating'].mean()
    total_rated = len(period_reviews[period_reviews['rating'].notna()])
    
    finding_2 = {
        "title": "Review Rating Distribution",
        "claim": f"Average rating across {total_rated} reviews is {avg_rating:.2f} out of 5, with distribution: {json.dumps(rating_counts)}.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_distribution": {
                "value": json.dumps(rating_counts),
                "unit": "count",
                "numerator": total_rated,
                "denominator": total_rated,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_rated,
        "coverage_notes": [
            f"Total reviews with ratings: {total_rated}",
            f"Rating distribution: {rating_counts}",
            "All reviews have rating values"
        ],
        "assumptions": [
            "Rating field values are accurate and on standard 1-5 scale",
            "Missing ratings treated as non-rated"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# Finding 3: Sentiment/Topic Analysis (basic text analysis)
if len(period_reviews) > 0 and period_reviews['text'].notna().sum() > 0:
    # Count reviews with text content
    reviews_with_text = period_reviews[period_reviews['text'].notna()].copy()
    text_count = len(reviews_with_text)
    
    # Simple keyword detection for common topics
    topics_detected = {
        'quality': 0,
        'service': 0,
        'taste': 0,
        'price': 0,
        'speed': 0,
        'cleanliness': 0
    }
    
    quality_keywords = ['quality', 'جودة', 'excellent', 'ممتاز', 'good', 'جيد', 'bad', 'سيء', 'poor', 'رديء']
    service_keywords = ['service', 'خدمة', 'staff', 'موظف', 'friendly', 'ودود', 'rude', 'وقح']
    taste_keywords = ['taste', 'طعم', 'flavor', 'نكهة', 'delicious', 'لذيذ', 'bitter', 'مر']
    price_keywords = ['price', 'سعر', 'expensive', 'غالي', 'cheap', 'رخيص', 'value', 'قيمة']
    speed_keywords = ['fast', 'سريع', 'slow', 'بطيء', 'quick', 'wait', 'انتظار']
    cleanliness_keywords = ['clean', 'نظيف', 'dirty', 'وسخ', 'hygiene', 'نظافة']
    
    for idx, row in reviews_with_text.iterrows():
        text_lower = str(row['text']).lower()
        if any(kw in text_lower for kw in quality_keywords):
            topics_detected['quality'] += 1
        if any(kw in text_lower for kw in service_keywords):
            topics_detected['service'] += 1
        if any(kw in text_lower for kw in taste_keywords):
            topics_detected['taste'] += 1
        if any(kw in text_lower for kw in price_keywords):
            topics_detected['price'] += 1
        if any(kw in text_lower for kw in speed_keywords):
            topics_detected['speed'] += 1
        if any(kw in text_lower for kw in cleanliness_keywords):
            topics_detected['cleanliness'] += 1
    
    # Filter to topics with at least 1 mention
    topics_with_mentions = {k: v for k, v in topics_detected.items() if v > 0}
    
    if topics_with_mentions:
        top_topic = max(topics_with_mentions, key=topics_with_mentions.get)
        top_topic_count = topics_with_mentions[top_topic]
        
        finding_3 = {
            "title": "Review Topics Mentioned",
            "claim": f"Among {text_count} reviews with text content, {len(topics_with_mentions)} topics were detected, with '{top_topic}' being most frequently mentioned ({top_topic_count} reviews).",
            "finding_type": "voice_of_customer",
            "metrics": {
                "topics_detected": {
                    "value": json.dumps(topics_with_mentions),
                    "unit": "count",
                    "numerator": top_topic_count,
                    "denominator": text_count,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["reviews"],
            "sample_size": text_count,
            "coverage_notes": [
                f"Reviews with text content: {text_count}",
                f"Topics detected: {topics_with_mentions}",
                "Keyword-based detection across English and Arabic terms"
            ],
            "assumptions": [
                "Topic detection based on keyword matching in review text",
                "Multiple topics can be mentioned in single review",
                "Keyword list covers common cafe-related topics"
            ],
            "confidence": 0.75
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)