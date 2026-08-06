import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Analysis period
analysis_start = "2026-02-02T00:00:00+03:00"
analysis_end = "2026-02-09T00:00:00+03:00"

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
    
    # Sentiment classification based on rating
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
    
    finding1 = {
        "title": "Customer Rating Distribution and Average Sentiment",
        "claim": f"During the analysis period (Feb 2-9, 2026), customers provided {len(analysis_reviews)} reviews with an average rating of {avg_rating:.2f}/5. Positive reviews (4-5 stars) comprise {sentiment_counts.get('positive', 0)} reviews, neutral (3 stars) {sentiment_counts.get('neutral', 0)}, and negative (1-2 stars) {sentiment_counts.get('negative', 0)}.",
        "finding_type": "customer_sentiment",
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
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Reviews from {len(analysis_reviews['source'].unique())} source(s)",
            f"Language distribution: {dict(analysis_reviews['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Positive sentiment defined as 4-5 stars, neutral as 3 stars, negative as 1-2 stars"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution and Text Analysis
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    # Analyze text content for topics
    topics_found = {}
    
    # Common topic keywords in English and Arabic
    topic_keywords = {
        'quality': ['quality', 'fresh', 'good', 'excellent', 'bad', 'poor', 'جودة', 'طازة', 'ممتاز', 'سيء'],
        'service': ['service', 'staff', 'friendly', 'rude', 'slow', 'fast', 'خدمة', 'موظف', 'ودود', 'بطيء'],
        'price': ['price', 'expensive', 'cheap', 'value', 'cost', 'سعر', 'غالي', 'رخيص', 'قيمة'],
        'taste': ['taste', 'flavor', 'delicious', 'bitter', 'sweet', 'طعم', 'لذيذ', 'مر', 'حلو'],
        'temperature': ['hot', 'cold', 'warm', 'iced', 'temperature', 'ساخن', 'بارد', 'درجة حرارة']
    }
    
    for topic, keywords in topic_keywords.items():
        count = 0
        for text in analysis_reviews['text'].dropna():
            text_lower = str(text).lower()
            if any(keyword in text_lower for keyword in keywords):
                count += 1
        if count > 0:
            topics_found[topic] = count
    
    finding2 = {
        "title": "Review Language Distribution and Topic Frequency",
        "claim": f"Of {len(analysis_reviews)} reviews, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic. Common topics mentioned include: {', '.join([f'{topic} ({count} reviews)' for topic, count in sorted(topics_found.items(), key=lambda x: x[1], reverse=True)[:3]])}.",
        "finding_type": "customer_feedback_analysis",
        "metrics": {
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
            f"Language coverage: English {language_counts.get('en', 0)}/{len(analysis_reviews)}, Arabic {language_counts.get('ar', 0)}/{len(analysis_reviews)}",
            f"Topic analysis based on keyword matching in {len(analysis_reviews[analysis_reviews['text'].notna()])} non-empty reviews"
        ],
        "assumptions": [
            "Topic keywords are representative of customer concerns",
            "Single keyword match indicates topic mention"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# Finding 3: Rating by Language
if len(analysis_reviews) > 0:
    language_ratings = analysis_reviews.groupby('language')['rating'].agg(['mean', 'count', 'std']).round(2)
    
    if len(language_ratings) > 1:
        finding3 = {
            "title": "Rating Differences by Review Language",
            "claim": f"English reviews (n={int(language_ratings.loc['en', 'count'])}) averaged {language_ratings.loc['en', 'mean']:.2f} stars, while Arabic reviews (n={int(language_ratings.loc['ar', 'count'])}) averaged {language_ratings.loc['ar', 'mean']:.2f} stars.",
            "finding_type": "comparative_sentiment",
            "metrics": {
                "english_avg_rating": {
                    "value": float(language_ratings.loc['en', 'mean']),
                    "unit": "stars",
                    "numerator": int(language_ratings.loc['en', 'count']),
                    "denominator": int(language_ratings.loc['en', 'count']),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "arabic_avg_rating": {
                    "value": float(language_ratings.loc['ar', 'mean']),
                    "unit": "stars",
                    "numerator": int(language_ratings.loc['ar', 'count']),
                    "denominator": int(language_ratings.loc['ar', 'count']),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"English sample: {int(language_ratings.loc['en', 'count'])} reviews",
                f"Arabic sample: {int(language_ratings.loc['ar', 'count'])} reviews"
            ],
            "assumptions": [
                "Language classification is accurate",
                "Sample sizes are sufficient for comparison"
            ],
            "confidence": 0.80
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
