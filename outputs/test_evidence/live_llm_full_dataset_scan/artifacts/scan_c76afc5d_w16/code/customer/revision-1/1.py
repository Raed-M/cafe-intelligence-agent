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
analysis_start = "2026-04-27T00:00:00+03:00"
analysis_end = "2026-05-04T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews to analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with {rating_counts.get(5, 0)} five-star, {rating_counts.get(4, 0)} four-star, {rating_counts.get(3, 0)} three-star, {rating_counts.get(2, 0)} two-star, and {rating_counts.get(1, 0)} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language distribution: {language_dist}",
            f"Total reviews in analysis period: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "All reviews in the analysis period are included",
            "No filtering applied beyond date range"
        ],
        "confidence": 1.0
    }
    findings.append(finding1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Only 'en' and 'ar' language codes are present"
        ],
        "confidence": 1.0
    }
    findings.append(finding2)

# ============================================================================
# FINDING 3: Source Coverage
# ============================================================================

if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews were collected from {len(source_counts)} source(s): {', '.join([f'{src} ({count} reviews)' for src, count in source_counts.items()])}.",
        "finding_type": "source_coverage",
        "metrics": {
            "total_sources": {
                "value": len(source_counts),
                "unit": "sources",
                "numerator": len(source_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Source breakdown: {source_counts.to_dict()}",
            f"Total reviews: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Source field is accurately populated",
            "All reviews in the analysis period are included"
        ],
        "confidence": 1.0
    }
    findings.append(finding3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
