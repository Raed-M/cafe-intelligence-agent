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
menu_df = pd.read_parquet(inputs['menu'])
pos_df = pd.read_parquet(inputs['pos'])

# Analysis periods
analysis_period_start = "2026-06-08T00:00:00+03:00"
analysis_period_end = "2026-06-15T00:00:00+03:00"
previous_period_start = "2026-06-01T00:00:00+03:00"
previous_period_end = "2026-06-08T00:00:00+03:00"

# Convert review dates to datetime for filtering
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Parse analysis period dates
analysis_start = pd.to_datetime(analysis_period_start, utc=True)
analysis_end = pd.to_datetime(analysis_period_end, utc=True)
previous_start = pd.to_datetime(previous_period_start, utc=True)
previous_end = pd.to_datetime(previous_period_end, utc=True)

# Filter reviews for analysis period
analysis_reviews = reviews_df[(reviews_df['date'] >= analysis_start) & (reviews_df['date'] < analysis_end)]
previous_reviews = reviews_df[(reviews_df['date'] >= previous_start) & (reviews_df['date'] < previous_end)]

findings = []

# Finding 1: Rating distribution and average for analysis period
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Get source names from analysis period
    source_names = analysis_reviews['source'].unique().tolist()
    
    # Get language distribution
    language_dist = analysis_reviews['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-06-08 to 2026-06-15), the average rating across {len(analysis_reviews)} reviews is {avg_rating:.2f} out of 5, with {rating_counts.get(5, 0)} five-star ratings, {rating_counts.get(4, 0)} four-star ratings, {rating_counts.get(3, 0)} three-star ratings, {rating_counts.get(2, 0)} two-star ratings, and {rating_counts.get(1, 0)} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "reviews",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-08 to 2026-06-15",
            f"Language distribution: {language_dist}",
            f"Sources included: {', '.join(source_names)}"
        ],
        "assumptions": [
            "All reviews in the dataset are valid and complete",
            "Rating values are on a 1-5 scale",
            "Review dates are accurate and in UTC+3 timezone"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language distribution in analysis period
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution (Analysis Period)",
        "claim": f"During the analysis period, {language_counts.get('en', 0)} reviews were in English and {language_counts.get('ar', 0)} reviews were in Arabic out of {len(analysis_reviews)} total reviews.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "english_percentage": {
                "value": round((language_counts.get('en', 0) / len(analysis_reviews)) * 100, 1) if len(analysis_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "arabic_percentage": {
                "value": round((language_counts.get('ar', 0) / len(analysis_reviews)) * 100, 1) if len(analysis_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            }
        },
        "source_names": source_names,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-08 to 2026-06-15",
            f"Total reviews analyzed: {len(analysis_reviews)}"
        ],
        "assumptions": [
            "Language field is accurately populated for all reviews",
            "Language values are either 'en' or 'ar'"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Rating comparison between analysis and previous period
if len(analysis_reviews) > 0 and len(previous_reviews) > 0:
    avg_rating_analysis = analysis_reviews['rating'].mean()
    avg_rating_previous = previous_reviews['rating'].mean()
    rating_change = avg_rating_analysis - avg_rating_previous
    
    finding3 = {
        "title": "Average Rating Comparison: Analysis vs Previous Period",
        "claim": f"The average rating in the analysis period (2026-06-08 to 2026-06-15) was {avg_rating_analysis:.2f} compared to {avg_rating_previous:.2f} in the previous period (2026-06-01 to 2026-06-08), representing a change of {rating_change:+.2f} stars.",
        "finding_type": "period_comparison",
        "metrics": {
            "analysis_period_avg_rating": {
                "value": round(avg_rating_analysis, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "previous_period_avg_rating": {
                "value": round(avg_rating_previous, 2),
                "unit": "stars",
                "numerator": round(previous_reviews['rating'].sum(), 2),
                "denominator": len(previous_reviews),
                "period_start": previous_period_start,
                "period_end": previous_period_end
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": round(rating_change, 2),
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "analysis_period_review_count": {
                "value": len(analysis_reviews),
                "unit": "reviews",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "previous_period_review_count": {
                "value": len(previous_reviews),
                "unit": "reviews",
                "numerator": len(previous_reviews),
                "denominator": None,
                "period_start": previous_period_start,
                "period_end": previous_period_end
            }
        },
        "source_names": list(set(analysis_reviews['source'].unique().tolist() + previous_reviews['source'].unique().tolist())),
        "sample_size": len(analysis_reviews) + len(previous_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-08 to 2026-06-15 ({len(analysis_reviews)} reviews)",
            f"Previous period: 2026-06-01 to 2026-06-08 ({len(previous_reviews)} reviews)"
        ],
        "assumptions": [
            "Both periods have sufficient review volume for comparison",
            "Rating scale is consistent across both periods",
            "No systematic differences in review collection methodology between periods"
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
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
