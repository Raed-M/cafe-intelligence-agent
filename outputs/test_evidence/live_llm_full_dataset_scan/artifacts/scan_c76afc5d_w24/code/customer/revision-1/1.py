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
analysis_start = "2026-06-22T00:00:00+03:00"
analysis_end = "2026-06-29T00:00:00+03:00"
previous_start = "2026-06-15T00:00:00+03:00"
previous_end = "2026-06-22T00:00:00+03:00"

# Convert review dates to datetime for filtering
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)
analysis_start_dt = pd.to_datetime(analysis_start, utc=True)
analysis_end_dt = pd.to_datetime(analysis_end, utc=True)
previous_start_dt = pd.to_datetime(previous_start, utc=True)
previous_end_dt = pd.to_datetime(previous_end, utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[(reviews_df['date'] >= analysis_start_dt) & (reviews_df['date'] < analysis_end_dt)]
reviews_previous = reviews_df[(reviews_df['date'] >= previous_start_dt) & (reviews_df['date'] < previous_end_dt)]

findings = []
result = {
    "status": "success",
    "findings": []
}

# Finding 1: Rating distribution and average for analysis period
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = reviews_analysis['source'].unique().tolist()
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-06-22 to 2026-06-29), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {rating_counts}. Language coverage: {language_dist}.",
        "finding_type": "sentiment_distribution",
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
            "rating_1_count": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": language_dist.get('en', 0),
                "unit": "count",
                "numerator": language_dist.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": language_dist.get('ar', 0),
                "unit": "count",
                "numerator": language_dist.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-22 to 2026-06-29 (7 days)",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: English={language_dist.get('en', 0)}, Arabic={language_dist.get('ar', 0)}",
            f"Sources represented: {', '.join(source_names_analysis)}"
        ],
        "assumptions": [
            "Review dates are accurate and in UTC+3 timezone",
            "Rating values are integers from 1 to 5",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.95
    }
    result["findings"].append(finding1)

# Finding 2: Rating trend comparison (analysis vs previous period)
if len(reviews_analysis) > 0 and len(reviews_previous) > 0:
    avg_rating_analysis = reviews_analysis['rating'].mean()
    avg_rating_previous = reviews_previous['rating'].mean()
    rating_change = avg_rating_analysis - avg_rating_previous
    
    source_names_previous = reviews_previous['source'].unique().tolist()
    all_sources = list(set(source_names_analysis + source_names_previous))
    
    finding2 = {
        "title": "Average Rating Trend (Week-over-Week)",
        "claim": f"Average rating in analysis period (2026-06-22 to 2026-06-29) was {avg_rating_analysis:.2f}, compared to {avg_rating_previous:.2f} in the previous period (2026-06-15 to 2026-06-22), a change of {rating_change:+.2f} points.",
        "finding_type": "trend_comparison",
        "metrics": {
            "avg_rating_analysis_period": {
                "value": round(avg_rating_analysis, 2),
                "unit": "stars",
                "numerator": round(avg_rating_analysis * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "avg_rating_previous_period": {
                "value": round(avg_rating_previous, 2),
                "unit": "stars",
                "numerator": round(avg_rating_previous * len(reviews_previous), 2),
                "denominator": len(reviews_previous),
                "period_start": previous_start,
                "period_end": previous_end
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": round(rating_change, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_analysis_period": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_previous_period": {
                "value": len(reviews_previous),
                "unit": "count",
                "numerator": len(reviews_previous),
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-22 to 2026-06-29 ({len(reviews_analysis)} reviews)",
            f"Previous period: 2026-06-15 to 2026-06-22 ({len(reviews_previous)} reviews)",
            f"Sources in analysis period: {', '.join(source_names_analysis)}",
            f"Sources in previous period: {', '.join(source_names_previous)}"
        ],
        "assumptions": [
            "Review dates are accurate",
            "Periods are comparable in terms of business operations",
            "No major operational changes between periods"
        ],
        "confidence": 0.90
    }
    result["findings"].append(finding2)

# Finding 3: High-rating vs Low-rating review counts
if len(reviews_analysis) > 0:
    high_ratings = len(reviews_analysis[reviews_analysis['rating'] >= 4])
    low_ratings = len(reviews_analysis[reviews_analysis['rating'] <= 2])
    neutral_ratings = len(reviews_analysis[reviews_analysis['rating'] == 3])
    
    high_rating_pct = (high_ratings / len(reviews_analysis)) * 100 if len(reviews_analysis) > 0 else 0
    low_rating_pct = (low_ratings / len(reviews_analysis)) * 100 if len(reviews_analysis) > 0 else 0
    
    finding3 = {
        "title": "Sentiment Polarity Distribution (Analysis Period)",
        "claim": f"In the analysis period, {high_ratings} reviews ({high_rating_pct:.1f}%) were positive (4-5 stars), {low_ratings} reviews ({low_rating_pct:.1f}%) were negative (1-2 stars), and {neutral_ratings} reviews were neutral (3 stars).",
        "finding_type": "sentiment_polarity",
        "metrics": {
            "positive_reviews_4_5_stars": {
                "value": high_ratings,
                "unit": "count",
                "numerator": high_ratings,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_reviews_percentage": {
                "value": round(high_rating_pct, 1),
                "unit": "percent",
                "numerator": high_ratings,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_1_2_stars": {
                "value": low_ratings,
                "unit": "count",
                "numerator": low_ratings,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_percentage": {
                "value": round(low_rating_pct, 1),
                "unit": "percent",
                "numerator": low_ratings,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews_3_stars": {
                "value": neutral_ratings,
                "unit": "count",
                "numerator": neutral_ratings,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-22 to 2026-06-29",
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Positive (4-5 stars): {high_ratings} reviews",
            f"Negative (1-2 stars): {low_ratings} reviews",
            f"Neutral (3 stars): {neutral_ratings} reviews"
        ],
        "assumptions": [
            "Rating scale is 1-5 with 4-5 considered positive, 1-2 negative, 3 neutral",
            "All reviews have valid rating values",
            "Review ratings reflect customer satisfaction"
        ],
        "confidence": 0.95
    }
    result["findings"].append(finding3)

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
