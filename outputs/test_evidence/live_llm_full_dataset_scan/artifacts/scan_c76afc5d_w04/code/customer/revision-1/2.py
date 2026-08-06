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
analysis_start = "2026-02-02T00:00:00+03:00"
analysis_end = "2026-02-09T00:00:00+03:00"
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime and remove timezone info for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')

# Remove timezone info from both the data and the comparison boundaries
reviews_df['date_naive'] = reviews_df['date'].dt.tz_localize(None)
analysis_start_naive = analysis_start_dt.tz_localize(None)
analysis_end_naive = analysis_end_dt.tz_localize(None)

# Filter reviews in analysis period
reviews_analysis = reviews_df[
    (reviews_df['date_naive'] >= analysis_start_naive) & 
    (reviews_df['date_naive'] < analysis_end_naive)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names_list = sorted(reviews_analysis['source'].unique().tolist())
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start[:10]} to {analysis_end[:10]}, {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5.0.",
        "finding_type": "rating_distribution",
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
            "rating_1_star_count": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_star_count": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_list,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names_list)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are correctly parsed and timezone-aware",
            "All reviews in the artifact are genuine and unfiltered"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment/Topic Classification by Language
# ============================================================================

if len(reviews_analysis) > 0:
    # Separate by language
    reviews_en = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    reviews_ar = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
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
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts().to_dict()
    
    # Topic extraction: look for common keywords in non-empty text
    def extract_topics(text_series):
        topics = Counter()
        keywords = {
            'quality': ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء'],
            'service': ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف'],
            'price': ['price', 'expensive', 'cheap', 'value', 'سعر', 'غالي', 'رخيص'],
            'taste': ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملل'],
            'speed': ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'نظيف', 'قذر', 'نظافة']
        }
        
        for text in text_series:
            if pd.isna(text) or text == '':
                continue
            text_lower = str(text).lower()
            for topic, words in keywords.items():
                if any(word in text_lower for word in words):
                    topics[topic] += 1
        
        return dict(topics)
    
    topics_en = extract_topics(reviews_en['text']) if len(reviews_en) > 0 else {}
    topics_ar = extract_topics(reviews_ar['text']) if len(reviews_ar) > 0 else {}
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {sentiment_counts.get('positive', 0)} are positive (rating ≥4), {sentiment_counts.get('neutral', 0)} are neutral (rating=3), and {sentiment_counts.get('negative', 0)} are negative (rating <3). English reviews: {len(reviews_en)}, Arabic reviews: {len(reviews_ar)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_sentiment_count": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": sentiment_counts.get('neutral', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('neutral', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews_count": {
                "value": len(reviews_en),
                "unit": "count",
                "numerator": len(reviews_en),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": len(reviews_ar),
                "unit": "count",
                "numerator": len(reviews_ar),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_list,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classification based on rating scale: positive (≥4), neutral (=3), negative (<3)",
            f"English reviews: {len(reviews_en)}, Arabic reviews: {len(reviews_ar)}",
            f"Reviews with non-empty text: {len(reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'] != '')])}"
        ],
        "assumptions": [
            "Sentiment is derived from rating value, not text analysis",
            "Language field is accurately populated",
            "Rating scale is consistent (1-5)"
        ],
        "confidence": 1.0
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Topic Frequency in Reviews with Text
# ============================================================================

if len(reviews_analysis) > 0:
    reviews_with_text = reviews_analysis[
        reviews_analysis['text'].notna() & 
        (reviews_analysis['text'] != '')
    ].copy()
    
    if len(reviews_with_text) > 0:
        # Extract topics from all reviews with text
        keywords = {
            'quality': ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء'],
            'service': ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف'],
            'price': ['price', 'expensive', 'cheap', 'value', 'سعر', 'غالي', 'رخيص'],
            'taste': ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملل'],
            'speed': ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'نظيف', 'قذر', 'نظافة']
        }
        
        topic_counts = {topic: 0 for topic in keywords.keys()}
        
        for text in reviews_with_text['text']:
            if pd.isna(text):
                continue
            text_lower = str(text).lower()
            for topic, words in keywords.items():
                if any(word in text_lower for word in words):
                    topic_counts[topic] += 1
        
        # Find top topics
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        finding_3 = {
            "title": "Topic Frequency in Review Text",
            "claim": f"Among {len(reviews_with_text)} reviews with text content, the most frequently mentioned topics are: {top_topics[0][0]} ({top_topics[0][1]} mentions), {top_topics[1][0]} ({top_topics[1][1]} mentions), and {top_topics[2][0]} ({top_topics[2][1]} mentions).",
            "finding_type": "topic_frequency",
            "metrics": {
                "reviews_with_text_count": {
                    "value": len(reviews_with_text),
                    "unit": "count",
                    "numerator": len(reviews_with_text),
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "quality_mentions": {
                    "value": topic_counts['quality'],
                    "unit": "count",
                    "numerator": topic_counts['quality'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "service_mentions": {
                    "value": topic_counts['service'],
                    "unit": "count",
                    "numerator": topic_counts['service'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "price_mentions": {
                    "value": topic_counts['price'],
                    "unit": "count",
                    "numerator": topic_counts['price'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "taste_mentions": {
                    "value": topic_counts['taste'],
                    "unit": "count",
                    "numerator": topic_counts['taste'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "speed_mentions": {
                    "value": topic_counts['speed'],
                    "unit": "count",
                    "numerator": topic_counts['speed'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "cleanliness_mentions": {
                    "value": topic_counts['cleanliness'],
                    "unit": "count",
                    "numerator": topic_counts['cleanliness'],
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": source_names_list,
            "sample_size": len(reviews_with_text),
            "coverage_notes": [
                f"Topic extraction based on keyword matching in review text (bilingual)",
                f"Reviews with non-empty text: {len(reviews_with_text)} out of {len(reviews_analysis)} total",
                f"Keywords searched: quality, service, price, taste, speed, cleanliness (in English and Arabic)"
            ],
            "assumptions": [
                "Keyword matching is case-insensitive and language-agnostic",
                "A review may mention multiple topics",
                "Presence of keyword indicates topic relevance (no semantic validation)",
                "Empty or null text fields are excluded from topic analysis"
            ],
            "confidence": 0.7
        }
        findings.append(finding_3)

# ============================================================================
# Write output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
