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
analysis_start = "2026-03-30T00:00:00+03:00"
analysis_end = "2026-04-06T00:00:00+03:00"

# Convert to comparable datetime (strip timezone for comparison)
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Convert review dates to datetime
reviews_df['date_dt'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)

# Filter reviews in analysis period
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
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source distribution
    source_counts = reviews_analysis['source'].value_counts()
    sources_list = source_counts.index.tolist()
    
    # Get language distribution
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {dict(rating_counts)}. Sources: {dict(source_counts)}. Languages: {dict(language_counts)}.",
        "finding_type": "descriptive_metric",
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
                "unit": "stars (1-5)",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
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
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
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
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
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
            }
        },
        "source_names": sources_list,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in artifact: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(language_counts)}",
            f"Source distribution: {dict(source_counts)}"
        ],
        "assumptions": [
            "Review dates are accurate and timezone-aware",
            "Rating values are integers from 1 to 5",
            "All reviews in the artifact are valid and complete"
        ],
        "confidence": 1.0 if len(reviews_analysis) > 0 else 0.0
    }
    result["findings"].append(finding_1)

# Finding 2: Sentiment/Topic Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Classify sentiment based on rating
    def classify_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts()
    
    # Language-specific sentiment
    en_sentiment = english_reviews['rating'].apply(classify_sentiment).value_counts() if len(english_reviews) > 0 else {}
    ar_sentiment = arabic_reviews['rating'].apply(classify_sentiment).value_counts() if len(arabic_reviews) > 0 else {}
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, sentiment distribution is: {dict(sentiment_counts)}. English reviews ({len(english_reviews)}): {dict(en_sentiment)}. Arabic reviews ({len(arabic_reviews)}): {dict(ar_sentiment)}.",
        "finding_type": "sentiment_classification",
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
            "english_reviews_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sources_list,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"English reviews: {len(english_reviews)}",
            f"Arabic reviews: {len(arabic_reviews)}",
            "Sentiment classified by rating: 4-5 stars = positive, 3 stars = neutral, 1-2 stars = negative"
        ],
        "assumptions": [
            "Sentiment classification based on rating alone (no text analysis performed)",
            "Language field is accurate",
            "Rating-to-sentiment mapping is appropriate for this domain"
        ],
        "confidence": 1.0 if len(reviews_analysis) > 0 else 0.0
    }
    result["findings"].append(finding_2)

# Finding 3: Review Text Availability and Non-Empty Review Count
if len(reviews_analysis) > 0:
    non_empty_reviews = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)].copy()
    empty_reviews = len(reviews_analysis) - len(non_empty_reviews)
    
    finding_3 = {
        "title": "Review Text Availability",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {len(non_empty_reviews)} contain text content and {empty_reviews} are empty or missing text.",
        "finding_type": "data_quality_metric",
        "metrics": {
            "reviews_with_text": {
                "value": len(non_empty_reviews),
                "unit": "count",
                "numerator": len(non_empty_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_without_text": {
                "value": empty_reviews,
                "unit": "count",
                "numerator": empty_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "text_coverage_percentage": {
                "value": round((len(non_empty_reviews) / len(reviews_analysis)) * 100, 1),
                "unit": "percent",
                "numerator": len(non_empty_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sources_list,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews: {len(reviews_analysis)}",
            f"Reviews with text: {len(non_empty_reviews)}",
            f"Reviews without text: {empty_reviews}",
            "Text availability affects sentiment analysis capability"
        ],
        "assumptions": [
            "Empty or null text fields indicate missing review content",
            "Text field presence is required for detailed sentiment/topic analysis"
        ],
        "confidence": 1.0 if len(reviews_analysis) > 0 else 0.0
    }
    result["findings"].append(finding_3)

# Write result to output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
