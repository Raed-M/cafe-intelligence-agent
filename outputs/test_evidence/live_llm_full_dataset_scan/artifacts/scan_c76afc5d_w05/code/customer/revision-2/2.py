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
analysis_start = "2026-02-09T00:00:00+03:00"
analysis_end = "2026-02-16T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and ensure timezone-naive for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Convert comparison datetimes to UTC and make timezone-naive
analysis_start_dt = pd.to_datetime(analysis_start, utc=True).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end, utc=True).tz_localize(None)
reviews_df['date'] = reviews_df['date'].dt.tz_localize(None)

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all reviews for baseline comparison
all_reviews = reviews_df.copy()

# Initialize findings list
findings = []

# FINDING 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_dist = analysis_reviews['rating'].value_counts().sort_index().to_dict()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Get source names from analysis period
    source_names = sorted(analysis_reviews['source'].unique().tolist())
    
    # Language distribution
    lang_dist = analysis_reviews['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-02-09 to 2026-02-16), the average rating across {len(analysis_reviews)} reviews is {avg_rating:.2f} out of 5, with {rating_dist.get(5, 0)} five-star ratings and {rating_dist.get(1, 0)} one-star ratings.",
        "finding_type": "rating_distribution",
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
            "five_star_count": {
                "value": rating_dist.get(5, 0),
                "unit": "count",
                "numerator": rating_dist.get(5, 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": rating_dist.get(1, 0),
                "unit": "count",
                "numerator": rating_dist.get(1, 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-02-09 to 2026-02-16",
            f"Language distribution: {lang_dist}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews in the artifact are included without exclusion"
        ],
        "confidence": 1.0
    }
    findings.append(finding1)

# FINDING 2: Sentiment/Topic Classification by Language
if len(analysis_reviews) > 0:
    # Separate by language
    english_reviews = analysis_reviews[analysis_reviews['language'] == 'en'].copy()
    arabic_reviews = analysis_reviews[analysis_reviews['language'] == 'ar'].copy()
    
    # Simple sentiment classification based on rating
    def classify_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    
    sentiment_dist = analysis_reviews['sentiment'].value_counts().to_dict()
    
    # Extract topics from text (simple keyword matching)
    topics = []
    topic_keywords = {
        'quality': ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء'],
        'service': ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف'],
        'price': ['price', 'expensive', 'cheap', 'value', 'سعر', 'غالي', 'رخيص'],
        'taste': ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملل'],
        'speed': ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    }
    
    for idx, row in analysis_reviews.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
    
    topic_dist = Counter(topics)
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(analysis_reviews)} reviews in the analysis period, {sentiment_dist.get('positive', 0)} are positive (rating ≥4), {sentiment_dist.get('neutral', 0)} are neutral (rating=3), and {sentiment_dist.get('negative', 0)} are negative (rating <3). English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_sentiment_count": {
                "value": sentiment_dist.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('positive', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": sentiment_dist.get('neutral', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('neutral', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": sentiment_dist.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_dist.get('negative', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds",
            f"Language coverage: {len(english_reviews)} English, {len(arabic_reviews)} Arabic",
            f"Topics identified: {dict(topic_dist) if topic_dist else 'None'}"
        ],
        "assumptions": [
            "Sentiment classification: positive (≥4 stars), neutral (3 stars), negative (<3 stars)",
            "Topic extraction uses keyword matching in both English and Arabic",
            "Empty or null text fields are treated as having no identifiable topics"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# FINDING 3: Comparison with Previous Period
previous_start = "2026-02-02T00:00:00+03:00"
previous_end = "2026-02-09T00:00:00+03:00"

previous_start_dt = pd.to_datetime(previous_start, utc=True).tz_localize(None)
previous_end_dt = pd.to_datetime(previous_end, utc=True).tz_localize(None)

previous_reviews = reviews_df[
    (reviews_df['date'] >= previous_start_dt) & 
    (reviews_df['date'] < previous_end_dt)
].copy()

if len(previous_reviews) > 0 and len(analysis_reviews) > 0:
    prev_avg_rating = previous_reviews['rating'].mean()
    curr_avg_rating = analysis_reviews['rating'].mean()
    rating_change = curr_avg_rating - prev_avg_rating
    
    prev_sources = sorted(previous_reviews['source'].unique().tolist())
    curr_sources = sorted(analysis_reviews['source'].unique().tolist())
    
    finding3 = {
        "title": "Rating Trend: Analysis Period vs Previous Period",
        "claim": f"Average rating changed from {prev_avg_rating:.2f} (previous period: {len(previous_reviews)} reviews) to {curr_avg_rating:.2f} (analysis period: {len(analysis_reviews)} reviews), a change of {rating_change:+.2f} stars.",
        "finding_type": "rating_trend",
        "metrics": {
            "previous_period_avg_rating": {
                "value": round(prev_avg_rating, 2),
                "unit": "stars",
                "numerator": round(previous_reviews['rating'].sum(), 2),
                "denominator": len(previous_reviews),
                "period_start": previous_start,
                "period_end": previous_end
            },
            "analysis_period_avg_rating": {
                "value": round(curr_avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": round(rating_change, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_period_review_count": {
                "value": len(previous_reviews),
                "unit": "count",
                "numerator": len(previous_reviews),
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "analysis_period_review_count": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(set(prev_sources + curr_sources))),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Previous period: 2026-02-02 to 2026-02-09 ({len(previous_reviews)} reviews)",
            f"Analysis period: 2026-02-09 to 2026-02-16 ({len(analysis_reviews)} reviews)",
            f"Sources in previous period: {prev_sources}",
            f"Sources in analysis period: {curr_sources}"
        ],
        "assumptions": [
            "Periods are non-overlapping and consecutive",
            "Rating values are numeric and comparable across periods",
            "No data quality issues affect the comparison"
        ],
        "confidence": 0.9
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
