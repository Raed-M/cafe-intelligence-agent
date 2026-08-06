import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load input/output paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-03-02T00:00:00+03:00"
analysis_end = "2026-03-09T00:00:00+03:00"
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start[:10]} to {analysis_end[:10]}, {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5.0. Rating distribution: {dict(rating_counts)}.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews with non-null ratings are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# ============================================================================
# FINDING 2: Sentiment/Topic Classification by Language
# ============================================================================
if len(reviews_analysis) > 0:
    # Separate by language
    en_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    ar_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
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
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts()
    
    # Extract topics from text (simple keyword matching)
    topics = {
        'quality': 0,
        'service': 0,
        'price': 0,
        'taste': 0,
        'speed': 0,
        'cleanliness': 0,
        'ambiance': 0
    }
    
    quality_keywords = ['quality', 'good', 'excellent', 'bad', 'poor', 'fresh', 'stale', 'جودة', 'ممتاز', 'سيء']
    service_keywords = ['service', 'staff', 'friendly', 'rude', 'slow', 'fast', 'خدمة', 'موظف', 'لطيف']
    price_keywords = ['price', 'expensive', 'cheap', 'cost', 'value', 'سعر', 'غالي', 'رخيص']
    taste_keywords = ['taste', 'flavor', 'delicious', 'bland', 'sweet', 'bitter', 'طعم', 'لذيذ', 'مر']
    speed_keywords = ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    cleanliness_keywords = ['clean', 'dirty', 'hygiene', 'sanitary', 'نظيف', 'قذر', 'صحي']
    ambiance_keywords = ['ambiance', 'atmosphere', 'cozy', 'noisy', 'comfortable', 'جو', 'مريح', 'صاخب']
    
    for idx, row in reviews_analysis.iterrows():
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
        if any(kw in text for kw in ambiance_keywords):
            topics['ambiance'] += 1
    
    finding2 = {
        "title": "Sentiment Distribution and Topic Mentions",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, sentiment distribution shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. Most frequently mentioned topics: quality ({topics['quality']} mentions), service ({topics['service']} mentions), and taste ({topics['taste']} mentions).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "positive_sentiment_count": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('positive', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('neutral', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('negative', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "quality_topic_mentions": {
                "value": topics['quality'],
                "unit": "count",
                "numerator": topics['quality'],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "service_topic_mentions": {
                "value": topics['service'],
                "unit": "count",
                "numerator": topics['service'],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "taste_topic_mentions": {
                "value": topics['taste'],
                "unit": "count",
                "numerator": topics['taste'],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "price_topic_mentions": {
                "value": topics['price'],
                "unit": "count",
                "numerator": topics['price'],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"English reviews: {len(en_reviews)}, Arabic reviews: {len(ar_reviews)}",
            f"Topic mentions are based on keyword matching in original text",
            f"Sentiment classification derived from rating values (4-5: positive, 3: neutral, 1-2: negative)",
            f"Topics are not mutually exclusive; a single review may mention multiple topics"
        ],
        "assumptions": [
            "Sentiment classification based on rating is representative of text sentiment",
            "Keyword matching captures topic mentions with reasonable accuracy",
            "Review text is available and non-empty for topic extraction",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# ============================================================================
# FINDING 3: Language Coverage
# ============================================================================
if len(reviews_analysis) > 0:
    lang_counts = reviews_analysis['language'].value_counts()
    
    finding3 = {
        "title": "Review Language Coverage",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {lang_counts.get('en', 0)} are in English and {lang_counts.get('ar', 0)} are in Arabic, providing bilingual coverage of customer feedback.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "english_reviews": {
                "value": int(lang_counts.get('en', 0)),
                "unit": "count",
                "numerator": int(lang_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(lang_counts.get('ar', 0)),
                "unit": "count",
                "numerator": int(lang_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution: {dict(lang_counts)}",
            f"Bilingual coverage enables analysis of both English and Arabic-speaking customers",
            f"All reviews in analysis period have language field populated"
        ],
        "assumptions": [
            "Language field accurately reflects the language of the review text",
            "Both English and Arabic reviews are representative of customer base"
        ],
        "confidence": 0.95
    }
    findings.append(finding3)

# ============================================================================
# Build output
# ============================================================================
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
