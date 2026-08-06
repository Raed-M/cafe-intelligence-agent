import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-01-12T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-19T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-01-12 to 2026-01-19), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {rating_counts}. Language coverage: {language_dist}.",
        "finding_type": "descriptive_metric",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars (1-5)",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "rating_5_count": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "rating_4_count": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "rating_3_count": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "rating_2_count": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "rating_1_count": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "english_reviews": {
                "value": language_dist.get('en', 0),
                "unit": "count",
                "numerator": language_dist.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": language_dist.get('ar', 0),
                "unit": "count",
                "numerator": language_dist.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-01-12 to 2026-01-19 (7 days)",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: English={language_dist.get('en', 0)}, Arabic={language_dist.get('ar', 0)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are integers from 1 to 5",
            "All reviews in the dataset have valid rating and language fields",
            "Review dates are accurate and timezone-aware"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment Classification by Language
# ============================================================================

if len(reviews_analysis) > 0:
    # Simple sentiment classification based on rating
    # 5-4: positive, 3: neutral, 2-1: negative
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative')
    )
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts().to_dict()
    
    # Sentiment by language
    sentiment_by_lang = {}
    for lang in reviews_analysis['language'].unique():
        lang_reviews = reviews_analysis[reviews_analysis['language'] == lang]
        lang_sentiment = lang_reviews['sentiment'].value_counts().to_dict()
        sentiment_by_lang[lang] = lang_sentiment
    
    positive_count = sentiment_counts.get('positive', 0)
    neutral_count = sentiment_counts.get('neutral', 0)
    negative_count = sentiment_counts.get('negative', 0)
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {positive_count} are positive (rating ≥4), {neutral_count} are neutral (rating=3), and {negative_count} are negative (rating ≤2). English reviews: {sentiment_by_lang.get('en', {})}. Arabic reviews: {sentiment_by_lang.get('ar', {})}.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "positive_reviews": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "neutral_reviews": {
                "value": neutral_count,
                "unit": "count",
                "numerator": neutral_count,
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "positive_sentiment_pct": {
                "value": round(100 * positive_count / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": positive_count,
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "english_positive": {
                "value": sentiment_by_lang.get('en', {}).get('positive', 0),
                "unit": "count",
                "numerator": sentiment_by_lang.get('en', {}).get('positive', 0),
                "denominator": len(reviews_analysis[reviews_analysis['language'] == 'en']) if 'en' in reviews_analysis['language'].values else None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "arabic_positive": {
                "value": sentiment_by_lang.get('ar', {}).get('positive', 0),
                "unit": "count",
                "numerator": sentiment_by_lang.get('ar', {}).get('positive', 0),
                "denominator": len(reviews_analysis[reviews_analysis['language'] == 'ar']) if 'ar' in reviews_analysis['language'].values else None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds: positive (≥4), neutral (=3), negative (≤2)",
            f"English reviews: {len(reviews_analysis[reviews_analysis['language'] == 'en'])}",
            f"Arabic reviews: {len(reviews_analysis[reviews_analysis['language'] == 'ar'])}",
            f"All reviews in analysis period included"
        ],
        "assumptions": [
            "Rating-based sentiment classification is appropriate for this dataset",
            "Language field is accurately populated",
            "No reviews have missing rating values"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Low Rating Reviews (Potential Issues)
# ============================================================================

low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2].copy()

if len(low_rating_reviews) > 0:
    low_rating_by_lang = low_rating_reviews['language'].value_counts().to_dict()
    
    # Sample review IDs for evidence (anonymized)
    sample_ids = low_rating_reviews['review_id'].head(3).tolist()
    
    finding_3 = {
        "title": "Low Rating Reviews (1-2 Stars)",
        "claim": f"During the analysis period, {len(low_rating_reviews)} reviews received ratings of 1-2 stars, representing {round(100*len(low_rating_reviews)/len(reviews_analysis), 1)}% of all reviews. Language breakdown: English={low_rating_by_lang.get('en', 0)}, Arabic={low_rating_by_lang.get('ar', 0)}. Sample review IDs: {sample_ids}.",
        "finding_type": "issue_detection",
        "metrics": {
            "low_rating_count": {
                "value": len(low_rating_reviews),
                "unit": "count",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "low_rating_pct": {
                "value": round(100 * len(low_rating_reviews) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "low_rating_english": {
                "value": low_rating_by_lang.get('en', 0),
                "unit": "count",
                "numerator": low_rating_by_lang.get('en', 0),
                "denominator": len(reviews_analysis[reviews_analysis['language'] == 'en']) if 'en' in reviews_analysis['language'].values else None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "low_rating_arabic": {
                "value": low_rating_by_lang.get('ar', 0),
                "unit": "count",
                "numerator": low_rating_by_lang.get('ar', 0),
                "denominator": len(reviews_analysis[reviews_analysis['language'] == 'ar']) if 'ar' in reviews_analysis['language'].values else None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(low_rating_reviews),
        "coverage_notes": [
            f"Low rating threshold: 1-2 stars",
            f"Total low rating reviews: {len(low_rating_reviews)}",
            f"Percentage of total reviews: {round(100*len(low_rating_reviews)/len(reviews_analysis), 1)}%",
            f"Language distribution captured"
        ],
        "assumptions": [
            "Low ratings (1-2) indicate potential service or product issues",
            "Review text content not analyzed; classification based on rating only",
            "All reviews have valid rating values"
        ],
        "confidence": 0.95
    }
    findings.append(finding_3)

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
