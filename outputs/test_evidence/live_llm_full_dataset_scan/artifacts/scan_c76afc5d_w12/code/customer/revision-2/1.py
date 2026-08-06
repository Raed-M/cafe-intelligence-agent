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
analysis_start = datetime.fromisoformat("2026-03-30T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-06T00:00:00+03:00")

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
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star ratings and {int(rating_counts.get(1, 0))} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are from the specified sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Sentiment/Topic Classification
# Analyze reviews with text content
reviews_with_text = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)].copy()

if len(reviews_with_text) > 0:
    # Simple keyword-based sentiment analysis
    positive_keywords_en = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'fantastic', 'delicious', 'tasty']
    positive_keywords_ar = ['جيد', 'رائع', 'ممتاز', 'لذيذ', 'طعم', 'أحب', 'أفضل', 'رائعة', 'جميل']
    
    negative_keywords_en = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disgusting', 'cold', 'slow', 'rude', 'dirty']
    negative_keywords_ar = ['سيء', 'سيئة', 'رهيب', 'فظيع', 'بطيء', 'بطيئة', 'وقح', 'قذر', 'بارد']
    
    def classify_sentiment(text, language):
        if pd.isna(text) or len(str(text)) == 0:
            return 'neutral'
        text_lower = str(text).lower()
        
        if language == 'en':
            pos_count = sum(1 for kw in positive_keywords_en if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords_en if kw in text_lower)
        else:  # Arabic
            pos_count = sum(1 for kw in positive_keywords_ar if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords_ar if kw in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    reviews_with_text['sentiment'] = reviews_with_text.apply(
        lambda row: classify_sentiment(row['text'], row['language']), axis=1
    )
    
    sentiment_counts = reviews_with_text['sentiment'].value_counts()
    
    # Extract common topics
    topics = []
    for idx, row in reviews_with_text.iterrows():
        text = str(row['text']).lower()
        if 'coffee' in text or 'espresso' in text or 'latte' in text:
            topics.append('coffee_quality')
        if 'service' in text or 'staff' in text or 'waiter' in text:
            topics.append('service')
        if 'price' in text or 'expensive' in text or 'cost' in text:
            topics.append('pricing')
        if 'wait' in text or 'slow' in text or 'fast' in text:
            topics.append('speed')
        if 'clean' in text or 'dirty' in text or 'hygiene' in text:
            topics.append('cleanliness')
    
    topic_counts = Counter(topics)
    
    finding_2 = {
        "title": "Sentiment Distribution in Review Text",
        "claim": f"Among {len(reviews_with_text)} reviews with text content in the analysis period, {int(sentiment_counts.get('positive', 0))} are classified as positive sentiment, {int(sentiment_counts.get('negative', 0))} as negative, and {int(sentiment_counts.get('neutral', 0))} as neutral based on keyword analysis.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "reviews_with_text": {
                "value": len(reviews_with_text),
                "unit": "count",
                "numerator": len(reviews_with_text),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "positive_sentiment_count": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('positive', 0)),
                "denominator": len(reviews_with_text),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "negative_sentiment_count": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('negative', 0)),
                "denominator": len(reviews_with_text),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "neutral_sentiment_count": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('neutral', 0)),
                "denominator": len(reviews_with_text),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_with_text),
        "coverage_notes": [
            f"Only reviews with non-empty text content analyzed: {len(reviews_with_text)} out of {len(reviews_analysis)}",
            f"Sentiment classification based on keyword matching in original language",
            f"Language distribution in text reviews: {dict(reviews_with_text['language'].value_counts())}"
        ],
        "assumptions": [
            "Keyword-based sentiment classification is a proxy for actual sentiment",
            "Keywords are representative of positive/negative sentiment in both English and Arabic",
            "Review text is in the language specified in the 'language' column"
        ],
        "confidence": 0.70
    }
    findings.append(finding_2)

# Finding 3: High-rating vs Low-rating Review Comparison
high_rating_reviews = reviews_analysis[reviews_analysis['rating'] >= 4]
low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2]

if len(high_rating_reviews) > 0 and len(low_rating_reviews) > 0:
    high_with_text = high_rating_reviews[high_rating_reviews['text'].notna() & (high_rating_reviews['text'].str.len() > 0)]
    low_with_text = low_rating_reviews[low_rating_reviews['text'].notna() & (low_rating_reviews['text'].str.len() > 0)]
    
    finding_3 = {
        "title": "High vs Low Rating Review Volume",
        "claim": f"In the analysis period, {len(high_rating_reviews)} reviews have ratings of 4-5 stars, while {len(low_rating_reviews)} reviews have ratings of 1-2 stars. High-rating reviews with text content: {len(high_with_text)}, low-rating reviews with text content: {len(low_with_text)}.",
        "finding_type": "rating_comparison",
        "metrics": {
            "high_rating_count": {
                "value": len(high_rating_reviews),
                "unit": "count",
                "numerator": len(high_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "low_rating_count": {
                "value": len(low_rating_reviews),
                "unit": "count",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "high_rating_with_text": {
                "value": len(high_with_text),
                "unit": "count",
                "numerator": len(high_with_text),
                "denominator": len(high_rating_reviews),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "low_rating_with_text": {
                "value": len(low_with_text),
                "unit": "count",
                "numerator": len(low_with_text),
                "denominator": len(low_rating_reviews),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"High-rating defined as 4-5 stars: {len(high_rating_reviews)} reviews",
            f"Low-rating defined as 1-2 stars: {len(low_rating_reviews)} reviews",
            f"Neutral ratings (3 stars): {len(reviews_analysis) - len(high_rating_reviews) - len(low_rating_reviews)} reviews"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "High satisfaction threshold is 4+ stars",
            "Low satisfaction threshold is 1-2 stars"
        ],
        "confidence": 0.95
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
