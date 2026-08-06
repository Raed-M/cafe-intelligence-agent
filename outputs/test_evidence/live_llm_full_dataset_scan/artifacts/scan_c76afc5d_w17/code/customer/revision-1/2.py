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
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-05-04T00:00:00+03:00"
analysis_end = "2026-05-11T00:00:00+03:00"

# Convert to datetime for filtering, removing timezone info for comparison
analysis_start_dt = pd.to_datetime(analysis_start, utc=True).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end, utc=True).tz_localize(None)

# Filter reviews for analysis period - ensure date column is tz-naive
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True).dt.tz_localize(None)
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Initialize findings list
findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Language distribution
    language_dist = reviews_analysis['language'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution shows {rating_counts.to_dict()}. Language coverage: {language_dist.to_dict()}.",
        "finding_type": "voice_of_customer_metric",
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
                "numerator": round(avg_rating * len(reviews_analysis), 2),
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
            },
            "english_reviews": {
                "value": int(language_dist.get('en', 0)),
                "unit": "count",
                "numerator": int(language_dist.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_dist.get('ar', 0)),
                "unit": "count",
                "numerator": int(language_dist.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}",
            f"Language distribution: English={language_dist.get('en', 0)}, Arabic={language_dist.get('ar', 0)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "Language classification is accurate"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Analysis by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Analyze high vs low ratings
    high_rating_reviews = reviews_analysis[reviews_analysis['rating'] >= 4]
    low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2]
    
    finding2 = {
        "title": "Review Sentiment Distribution by Rating",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {len(high_rating_reviews)} reviews ({100*len(high_rating_reviews)/len(reviews_analysis):.1f}%) have ratings 4-5 (positive), while {len(low_rating_reviews)} reviews ({100*len(low_rating_reviews)/len(reviews_analysis):.1f}%) have ratings 1-2 (negative). English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "voice_of_customer_sentiment",
        "metrics": {
            "positive_reviews_4_5": {
                "value": len(high_rating_reviews),
                "unit": "count",
                "numerator": len(high_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_reviews_percentage": {
                "value": round(100*len(high_rating_reviews)/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": len(high_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_1_2": {
                "value": len(low_rating_reviews),
                "unit": "count",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_percentage": {
                "value": round(100*len(low_rating_reviews)/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_review_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_review_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Positive reviews (4-5 stars): {len(high_rating_reviews)} ({100*len(high_rating_reviews)/len(reviews_analysis):.1f}%)",
            f"Negative reviews (1-2 stars): {len(low_rating_reviews)} ({100*len(low_rating_reviews)/len(reviews_analysis):.1f}%)",
            f"Neutral reviews (3 stars): {len(reviews_analysis) - len(high_rating_reviews) - len(low_rating_reviews)}",
            f"Bilingual coverage: {len(english_reviews)} English, {len(arabic_reviews)} Arabic"
        ],
        "assumptions": [
            "Rating 4-5 indicates positive sentiment, 1-2 indicates negative, 3 is neutral",
            "Review text may be empty but rating is always present",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.92
    }
    findings.append(finding2)

# Finding 3: Review Volume Trend (Analysis vs Previous Period)
previous_start = "2026-04-27T00:00:00+03:00"
previous_end = "2026-05-04T00:00:00+03:00"

previous_start_dt = pd.to_datetime(previous_start, utc=True).tz_localize(None)
previous_end_dt = pd.to_datetime(previous_end, utc=True).tz_localize(None)

reviews_previous = reviews_df[
    (reviews_df['date'] >= previous_start_dt) & 
    (reviews_df['date'] < previous_end_dt)
].copy()

if len(reviews_analysis) > 0 and len(reviews_previous) > 0:
    volume_change = len(reviews_analysis) - len(reviews_previous)
    pct_change = (volume_change / len(reviews_previous)) * 100 if len(reviews_previous) > 0 else 0
    
    finding3 = {
        "title": "Review Volume Comparison: Analysis Period vs Previous Period",
        "claim": f"Review volume in the analysis period ({analysis_start} to {analysis_end}) was {len(reviews_analysis)} reviews, compared to {len(reviews_previous)} reviews in the previous period ({previous_start} to {previous_end}), representing a change of {volume_change} reviews ({pct_change:+.1f}%).",
        "finding_type": "voice_of_customer_volume",
        "metrics": {
            "analysis_period_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "previous_period_reviews": {
                "value": len(reviews_previous),
                "unit": "count",
                "numerator": len(reviews_previous),
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "volume_change": {
                "value": volume_change,
                "unit": "count",
                "numerator": volume_change,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "volume_change_percentage": {
                "value": round(pct_change, 1),
                "unit": "percent",
                "numerator": volume_change,
                "denominator": len(reviews_previous),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end} ({len(reviews_analysis)} reviews)",
            f"Previous period: {previous_start} to {previous_end} ({len(reviews_previous)} reviews)",
            f"Volume change: {volume_change} reviews ({pct_change:+.1f}%)"
        ],
        "assumptions": [
            "Review dates are accurate and in UTC+3 timezone",
            "Both periods have complete data coverage",
            "Review volume is a valid indicator of customer engagement"
        ],
        "confidence": 0.95
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
