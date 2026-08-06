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
analysis_start = "2026-03-16T00:00:00+03:00"
analysis_end = "2026-03-23T00:00:00+03:00"

# Convert to comparable datetime (strip timezone for comparison)
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Convert review dates to datetime
reviews_df['date_dt'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date_dt'] >= analysis_start_dt) & 
    (reviews_df['date_dt'] < analysis_end_dt)
].copy()

# Initialize result structure
result = {
    "status": "success",
    "findings": []
}

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    total_reviews = len(reviews_analysis)
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), the average rating across {total_reviews} reviews is {avg_rating:.2f} out of 5, with distribution: 1-star: {int(rating_counts.get(1, 0))}, 2-star: {int(rating_counts.get(2, 0))}, 3-star: {int(rating_counts.get(3, 0))}, 4-star: {int(rating_counts.get(4, 0))}, 5-star: {int(rating_counts.get(5, 0))}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": float(round(avg_rating, 2)),
                "unit": "stars",
                "numerator": float(round(reviews_analysis['rating'].sum(), 2)),
                "denominator": int(total_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": int(total_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_star": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_star": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": [str(s) for s in source_names],
        "sample_size": int(total_reviews),
        "coverage_notes": [
            f"Language distribution: {dict((k, int(v)) for k, v in language_dist.items())}",
            f"Sources included: {', '.join(str(s) for s in source_names)}",
            f"Analysis period: {analysis_start} to {analysis_end}"
        ],
        "assumptions": [
            "All reviews in the analysis period are included regardless of language",
            "Rating values are treated as numeric and complete",
            "No filtering applied for review quality or duplicates"
        ],
        "confidence": 1.0 if total_reviews > 0 else 0.0
    }
    result["findings"].append(finding1)

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
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts().to_dict()
    
    english_sentiment = english_reviews['rating'].apply(classify_sentiment).value_counts().to_dict()
    arabic_sentiment = arabic_reviews['rating'].apply(classify_sentiment).value_counts().to_dict()
    
    # Extract topics from non-empty text reviews
    topics_mentioned = []
    for idx, row in reviews_analysis.iterrows():
        if pd.notna(row['text']) and len(str(row['text']).strip()) > 0:
            text_lower = str(row['text']).lower()
            # Simple keyword detection
            if any(word in text_lower for word in ['coffee', 'espresso', 'latte', 'cappuccino', 'americano']):
                topics_mentioned.append('coffee_quality')
            if any(word in text_lower for word in ['service', 'staff', 'friendly', 'rude', 'slow']):
                topics_mentioned.append('service')
            if any(word in text_lower for word in ['price', 'expensive', 'cheap', 'cost']):
                topics_mentioned.append('pricing')
            if any(word in text_lower for word in ['clean', 'dirty', 'hygiene', 'atmosphere', 'ambiance']):
                topics_mentioned.append('ambiance')
    
    topic_counts = Counter(topics_mentioned)
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {total_reviews} reviews in the analysis period, sentiment distribution is: {int(sentiment_counts.get('positive', 0))} positive (rating ≥4), {int(sentiment_counts.get('neutral', 0))} neutral (rating=3), {int(sentiment_counts.get('negative', 0))} negative (rating <3). English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_sentiment_count": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews_count": {
                "value": int(len(english_reviews)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": int(len(arabic_reviews)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_positive": {
                "value": int(english_sentiment.get('positive', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_positive": {
                "value": int(arabic_sentiment.get('positive', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": [str(s) for s in source_names],
        "sample_size": int(total_reviews),
        "coverage_notes": [
            f"Sentiment classified by rating threshold: positive (≥4), neutral (=3), negative (<3)",
            f"English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}",
            f"Topics detected in {len(topics_mentioned)} review texts"
        ],
        "assumptions": [
            "Sentiment is derived deterministically from rating values",
            "Language field is accurate and complete",
            "Topic detection uses simple keyword matching on review text"
        ],
        "confidence": 1.0 if total_reviews > 0 else 0.0
    }
    result["findings"].append(finding2)

# Finding 3: High-Volume Topics (if sufficient evidence)
if len(topic_counts) > 0 and sum(topic_counts.values()) > 0:
    top_topics = topic_counts.most_common(3)
    
    finding3 = {
        "title": "Frequently Mentioned Topics in Reviews",
        "claim": f"Among {len(reviews_analysis)} reviews with text content in the analysis period, the most frequently mentioned topics are: {top_topics[0][0]} ({int(top_topics[0][1])} mentions)" + (f", {top_topics[1][0]} ({int(top_topics[1][1])} mentions)" if len(top_topics) > 1 else "") + (f", {top_topics[2][0]} ({int(top_topics[2][1])} mentions)" if len(top_topics) > 2 else ""),
        "finding_type": "topic_frequency",
        "metrics": {
            "total_reviews_with_text": {
                "value": int(len(reviews_analysis[reviews_analysis['text'].notna()])),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": [str(s) for s in source_names],
        "sample_size": int(len(reviews_analysis[reviews_analysis['text'].notna()])),
        "coverage_notes": [
            f"Topic detection based on keyword matching in review text",
            f"Total topic mentions: {sum(topic_counts.values())}",
            f"Unique topics identified: {len(topic_counts)}"
        ],
        "assumptions": [
            "Topic keywords are representative of actual customer concerns",
            "A single review may mention multiple topics",
            "Keyword matching is case-insensitive"
        ],
        "confidence": 0.7 if sum(topic_counts.values()) > 5 else 0.5
    }
    
    # Add topic-specific metrics
    for topic, count in top_topics:
        finding3["metrics"][f"{topic}_mentions"] = {
            "value": int(count),
            "unit": "count",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    result["findings"].append(finding3)

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
