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

# Analysis period
analysis_start = "2026-04-27T00:00:00+03:00"
analysis_end = "2026-05-04T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and handle timezone awareness
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Ensure both are timezone-aware or both are timezone-naive for comparison
if reviews_df['date'].dt.tz is None:
    # If reviews_df is naive, make comparison datetimes naive
    analysis_start_dt = analysis_start_dt.tz_localize(None)
    analysis_end_dt = analysis_end_dt.tz_localize(None)
else:
    # If reviews_df is aware, make comparison datetimes aware
    if analysis_start_dt.tz is None:
        analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])
        analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Language breakdown
    language_counts = reviews_analysis['language'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-04-27 to 2026-05-04), the average rating across {int(len(reviews_analysis))} reviews was {float(avg_rating):.2f} out of 5, with {int(language_counts.get('en', 0))} English and {int(language_counts.get('ar', 0))} Arabic reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": float(round(avg_rating, 2)),
                "unit": "stars",
                "numerator": float(round(reviews_analysis['rating'].sum(), 2)),
                "denominator": int(len(reviews_analysis)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": int(len(reviews_analysis)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Reviews from {int(len(reviews_analysis['source'].unique()))} source(s)",
            f"Language distribution: {dict(language_counts)}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1-5",
            "Date filtering uses UTC+3 timezone as specified"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Classification
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple sentiment classification based on rating
    positive_count = int(len(reviews_analysis[reviews_analysis['rating'] >= 4]))
    neutral_count = int(len(reviews_analysis[reviews_analysis['rating'] == 3]))
    negative_count = int(len(reviews_analysis[reviews_analysis['rating'] <= 2]))
    
    # Topic extraction from text (simple keyword-based)
    topics = {
        'quality': 0,
        'service': 0,
        'price': 0,
        'taste': 0,
        'atmosphere': 0,
        'speed': 0
    }
    
    quality_keywords = ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء']
    service_keywords = ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف']
    price_keywords = ['price', 'expensive', 'cheap', 'cost', 'سعر', 'غالي', 'رخيص']
    taste_keywords = ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملح']
    atmosphere_keywords = ['atmosphere', 'ambiance', 'clean', 'dirty', 'جو', 'نظيف', 'وسخ']
    speed_keywords = ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    
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
        if any(kw in text for kw in atmosphere_keywords):
            topics['atmosphere'] += 1
        if any(kw in text for kw in speed_keywords):
            topics['speed'] += 1
    
    # Filter topics with at least 1 mention
    mentioned_topics = {k: v for k, v in topics.items() if v > 0}
    
    if mentioned_topics:
        finding2 = {
            "title": "Sentiment Distribution and Topic Mentions",
            "claim": f"Among {int(len(reviews_analysis))} reviews in the analysis period, {positive_count} were positive (rating ≥4), {neutral_count} neutral (rating=3), and {negative_count} negative (rating ≤2). Key topics mentioned include: {', '.join([f'{k} ({int(v)} mentions)' for k, v in sorted(mentioned_topics.items(), key=lambda x: x[1], reverse=True)[:3]])}.",
            "finding_type": "sentiment_and_topics",
            "metrics": {
                "positive_reviews": {
                    "value": positive_count,
                    "unit": "count",
                    "numerator": positive_count,
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_reviews": {
                    "value": neutral_count,
                    "unit": "count",
                    "numerator": neutral_count,
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews": {
                    "value": negative_count,
                    "unit": "count",
                    "numerator": negative_count,
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": int(len(reviews_analysis)),
            "coverage_notes": [
                f"Sentiment based on rating scale (positive: ≥4, neutral: 3, negative: ≤2)",
                f"Topics identified through keyword matching in {int(len(english_reviews))} English and {int(len(arabic_reviews))} Arabic reviews",
                f"Topics mentioned: {dict(mentioned_topics)}"
            ],
            "assumptions": [
                "Sentiment classification based on rating thresholds",
                "Topic extraction uses simple keyword matching",
                "Empty or null review text treated as no topics mentioned"
            ],
            "confidence": 0.85
        }
        findings.append(finding2)

# Finding 3: High-rated vs Low-rated Review Comparison
if len(reviews_analysis) > 0:
    high_rated = reviews_analysis[reviews_analysis['rating'] >= 4]
    low_rated = reviews_analysis[reviews_analysis['rating'] <= 2]
    
    if len(high_rated) > 0 and len(low_rated) > 0:
        # Get sample review IDs for evidence
        high_sample_ids = high_rated['review_id'].head(3).tolist()
        low_sample_ids = low_rated['review_id'].head(3).tolist()
        
        finding3 = {
            "title": "High-Rated vs Low-Rated Review Comparison",
            "claim": f"During the analysis period, {int(len(high_rated))} reviews were highly rated (≥4 stars) compared to {int(len(low_rated))} low-rated reviews (≤2 stars), indicating a {float(round(len(high_rated)/len(reviews_analysis)*100, 1))}% positive sentiment rate.",
            "finding_type": "sentiment_comparison",
            "metrics": {
                "high_rated_count": {
                    "value": int(len(high_rated)),
                    "unit": "count",
                    "numerator": int(len(high_rated)),
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "low_rated_count": {
                    "value": int(len(low_rated)),
                    "unit": "count",
                    "numerator": int(len(low_rated)),
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_sentiment_rate": {
                    "value": float(round(len(high_rated)/len(reviews_analysis)*100, 1)),
                    "unit": "percent",
                    "numerator": int(len(high_rated)),
                    "denominator": int(len(reviews_analysis)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": int(len(reviews_analysis)),
            "coverage_notes": [
                f"High-rated sample IDs: {high_sample_ids}",
                f"Low-rated sample IDs: {low_sample_ids}",
                f"Total reviews analyzed: {int(len(reviews_analysis))}"
            ],
            "assumptions": [
                "High-rated defined as rating ≥4",
                "Low-rated defined as rating ≤2",
                "Positive sentiment rate calculated as high-rated / total"
            ],
            "confidence": 0.92
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
