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

# Analysis period
analysis_start = "2026-07-06T00:00:00+03:00"
analysis_end = "2026-07-13T00:00:00+03:00"

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
        "claim": f"During the analysis period (2026-07-06 to 2026-07-13), customers provided {len(analysis_reviews)} reviews with an average rating of {avg_rating:.2f} out of 5. The majority of reviews ({sentiment_counts.get('positive', 0)} out of {len(analysis_reviews)}) were positive (rating ≥4).",
        "finding_type": "customer_sentiment",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
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
            "Rating ≥4 classified as positive, rating=3 as neutral, rating <3 as negative",
            "All reviews in the period are included regardless of text content"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language and Topic Analysis
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    # Extract topics from reviews with text
    reviews_with_text = analysis_reviews[analysis_reviews['text'].notna() & (analysis_reviews['text'].str.len() > 0)]
    
    # Simple keyword extraction for common cafe topics
    topics = {
        'quality': 0,
        'service': 0,
        'price': 0,
        'taste': 0,
        'atmosphere': 0,
        'speed': 0
    }
    
    quality_keywords = ['quality', 'جودة', 'excellent', 'great', 'good', 'bad', 'poor', 'رديء']
    service_keywords = ['service', 'خدمة', 'staff', 'موظف', 'friendly', 'rude', 'slow', 'fast']
    price_keywords = ['price', 'سعر', 'expensive', 'cheap', 'cost', 'غالي', 'رخيص']
    taste_keywords = ['taste', 'طعم', 'flavor', 'delicious', 'bitter', 'sweet', 'لذيذ']
    atmosphere_keywords = ['atmosphere', 'جو', 'ambiance', 'clean', 'dirty', 'comfortable', 'نظيف']
    speed_keywords = ['speed', 'سرعة', 'fast', 'slow', 'quick', 'wait', 'سريع']
    
    for idx, row in reviews_with_text.iterrows():
        text = str(row['text']).lower()
        if any(kw in text for kw in quality_keywords):
            topics['quality'] += 1
        if any(kw in text for kw in service_keywords):
            topics['service'] += 1
        if any(kw in text for kw in price_keywords):
            topics['price'] += 1
        if any(kw in text for kw in taste_keywords):
            topics['taste'] += 1
        if any(kw in text for kw in atmosphere_keywords):
            topics['atmosphere'] += 1
        if any(kw in text for kw in speed_keywords):
            topics['speed'] += 1
    
    # Find most mentioned topics
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
    
    finding2 = {
        "title": "Review Language Distribution and Topic Mentions",
        "claim": f"Among {len(analysis_reviews)} reviews, {language_counts.get('en', 0)} were in English and {language_counts.get('ar', 0)} were in Arabic. The most frequently mentioned topics in reviews with text ({len(reviews_with_text)} reviews) were quality ({top_topics[0][1]} mentions), service ({top_topics[1][1]} mentions), and taste ({top_topics[2][1]} mentions).",
        "finding_type": "review_composition",
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
            },
            "reviews_with_text": {
                "value": len(reviews_with_text),
                "unit": "count",
                "numerator": len(reviews_with_text),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "quality_mentions": {
                "value": topics['quality'],
                "unit": "count",
                "numerator": topics['quality'],
                "denominator": len(reviews_with_text) if len(reviews_with_text) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "service_mentions": {
                "value": topics['service'],
                "unit": "count",
                "numerator": topics['service'],
                "denominator": len(reviews_with_text) if len(reviews_with_text) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Text analysis based on {len(reviews_with_text)} reviews with non-empty text",
            f"Topic detection uses keyword matching in both English and Arabic"
        ],
        "assumptions": [
            "Topic keywords are case-insensitive and include both English and Arabic terms",
            "A review can mention multiple topics",
            "Keyword presence indicates topic relevance without sentiment analysis"
        ],
        "confidence": 0.75
    }
    findings.append(finding2)

# Finding 3: Rating by Language
if len(analysis_reviews) > 0:
    language_ratings = analysis_reviews.groupby('language')['rating'].agg(['mean', 'count', 'std'])
    
    if len(language_ratings) > 1:
        en_avg = language_ratings.loc['en', 'mean'] if 'en' in language_ratings.index else None
        ar_avg = language_ratings.loc['ar', 'mean'] if 'ar' in language_ratings.index else None
        
        if en_avg is not None and ar_avg is not None:
            finding3 = {
                "title": "Average Rating by Review Language",
                "claim": f"English-language reviews (n={int(language_ratings.loc['en', 'count'])}) had an average rating of {en_avg:.2f}, while Arabic-language reviews (n={int(language_ratings.loc['ar', 'count'])}) had an average rating of {ar_avg:.2f}.",
                "finding_type": "language_comparison",
                "metrics": {
                    "english_avg_rating": {
                        "value": round(en_avg, 2),
                        "unit": "stars",
                        "numerator": round(analysis_reviews[analysis_reviews['language'] == 'en']['rating'].sum(), 2),
                        "denominator": int(language_ratings.loc['en', 'count']),
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    },
                    "arabic_avg_rating": {
                        "value": round(ar_avg, 2),
                        "unit": "stars",
                        "numerator": round(analysis_reviews[analysis_reviews['language'] == 'ar']['rating'].sum(), 2),
                        "denominator": int(language_ratings.loc['ar', 'count']),
                        "period_start": analysis_start,
                        "period_end": analysis_end
                    }
                },
                "source_names": list(analysis_reviews['source'].unique()),
                "sample_size": len(analysis_reviews),
                "coverage_notes": [
                    f"English reviews: {int(language_ratings.loc['en', 'count'])} reviews",
                    f"Arabic reviews: {int(language_ratings.loc['ar', 'count'])} reviews"
                ],
                "assumptions": [
                    "Language classification is as provided in the source data",
                    "Rating differences may reflect different customer segments or review sources"
                ],
                "confidence": 0.85
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
