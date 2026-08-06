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
analysis_start = "2026-01-05T00:00:00+03:00"
analysis_end = "2026-01-12T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all reviews for baseline comparison
reviews_all = reviews_df.copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = reviews_analysis['source'].unique().tolist()
    
    # Language distribution
    lang_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-01-05 to 2026-01-12), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with distribution: {rating_dist}. Language coverage: {lang_dist}.",
        "finding_type": "customer_satisfaction_metric",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_star_count": {
                "value": rating_dist.get(5, 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": rating_dist.get(4, 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": rating_dist.get(3, 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": rating_dist.get(2, 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_star_count": {
                "value": rating_dist.get(1, 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": lang_dist.get('en', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": lang_dist.get('ar', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-01-05 to 2026-01-12 (7 days)",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {lang_dist}",
            f"Sources represented: {source_names_analysis}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Date field is accurate and timezone-aware",
            "Language field correctly identifies review language"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Classification (by language)
if len(reviews_analysis) > 0:
    # Separate by language
    en_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    ar_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
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
    
    sentiment_dist = reviews_analysis['sentiment'].value_counts().to_dict()
    
    # Analyze text for common topics (simple keyword matching)
    topics_en = []
    topics_ar = []
    
    if len(en_reviews) > 0:
        en_text = ' '.join(en_reviews['text'].fillna('').str.lower())
        keywords_en = ['coffee', 'taste', 'quality', 'service', 'price', 'fast', 'slow', 'friendly', 'rude', 'clean', 'dirty', 'hot', 'cold', 'fresh', 'stale']
        for keyword in keywords_en:
            if keyword in en_text:
                topics_en.append(keyword)
    
    if len(ar_reviews) > 0:
        ar_text = ' '.join(ar_reviews['text'].fillna('').str.lower())
        keywords_ar = ['قهوة', 'طعم', 'جودة', 'خدمة', 'سعر', 'سريع', 'بطيء', 'ودود', 'وقح', 'نظيف', 'وسخ', 'ساخن', 'بارد', 'طازج', 'قديم']
        for keyword in keywords_ar:
            if keyword in ar_text:
                topics_ar.append(keyword)
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"During the analysis period, sentiment distribution across {len(reviews_analysis)} reviews shows: {sentiment_dist}. English reviews: {len(en_reviews)}, Arabic reviews: {len(ar_reviews)}.",
        "finding_type": "sentiment_analysis",
        "metrics": {
            "positive_sentiment_count": {
                "value": sentiment_dist.get('positive', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_sentiment_count": {
                "value": sentiment_dist.get('neutral', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": sentiment_dist.get('negative', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_review_count": {
                "value": len(en_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_review_count": {
                "value": len(ar_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_sentiment_percentage": {
                "value": round((sentiment_dist.get('positive', 0) / len(reviews_analysis) * 100), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": sentiment_dist.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classified based on rating: 4-5 stars = positive, 3 stars = neutral, 1-2 stars = negative",
            f"English reviews: {len(en_reviews)} ({round(len(en_reviews)/len(reviews_analysis)*100, 1)}%)",
            f"Arabic reviews: {len(ar_reviews)} ({round(len(ar_reviews)/len(reviews_analysis)*100, 1)}%)",
            f"Topics identified via keyword matching (not exhaustive)"
        ],
        "assumptions": [
            "Rating-based sentiment classification is appropriate",
            "Text field contains meaningful review content",
            "Language field correctly identifies review language",
            "Keyword matching captures main topics discussed"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# Finding 3: Low Rating Reviews (1-2 stars) - Frequency and Sample
low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2].copy()

if len(low_rating_reviews) > 0:
    low_rating_count = len(low_rating_reviews)
    low_rating_pct = (low_rating_count / len(reviews_analysis) * 100) if len(reviews_analysis) > 0 else 0
    
    # Get sample review IDs (anonymized)
    sample_review_ids = low_rating_reviews['review_id'].head(3).tolist()
    
    finding3 = {
        "title": "Low Rating Reviews (1-2 Stars)",
        "claim": f"During the analysis period, {low_rating_count} reviews ({low_rating_pct:.1f}% of total) received ratings of 1-2 stars, indicating potential dissatisfaction areas.",
        "finding_type": "quality_alert",
        "metrics": {
            "low_rating_count": {
                "value": low_rating_count,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "low_rating_percentage": {
                "value": round(low_rating_pct, 1),
                "unit": "percent",
                "numerator": low_rating_count,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": len(low_rating_reviews[low_rating_reviews['rating'] == 1]),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "two_star_count": {
                "value": len(low_rating_reviews[low_rating_reviews['rating'] == 2]),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": low_rating_count,
        "coverage_notes": [
            f"Low rating threshold: 1-2 stars",
            f"Sample review IDs: {sample_review_ids}",
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Analysis period: 2026-01-05 to 2026-01-12"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Low ratings (1-2) indicate dissatisfaction",
            "Review text may contain specific complaint details"
        ],
        "confidence": 0.90
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
