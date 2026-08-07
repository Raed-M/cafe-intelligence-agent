import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define analysis period
analysis_start = "2026-07-13T00:00:00+03:00"
analysis_end = "2026-07-20T00:00:00+03:00"

# Convert to comparable datetime (strip timezone for comparison)
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)

# Filter reviews for analysis period
reviews_period = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_period) > 0:
    rating_counts = reviews_period['rating'].value_counts().sort_index()
    avg_rating = reviews_period['rating'].mean()
    
    # Count by rating
    rating_dist = {}
    for rating in sorted(reviews_period['rating'].unique()):
        count = (reviews_period['rating'] == rating).sum()
        rating_dist[f"rating_{int(rating)}_count"] = count
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-07-13 to 2026-07-20), the cafe received {len(reviews_period)} reviews with an average rating of {avg_rating:.2f} out of 5.0.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_period),
                "unit": "count",
                "numerator": len(reviews_period),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_period['rating'].sum(), 2),
                "denominator": len(reviews_period),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_period),
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20 (7 days)",
            f"Total reviews in artifact: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_period)}",
            f"Language distribution: {dict(reviews_period['language'].value_counts())}"
        ],
        "assumptions": [
            "Review dates are accurate and timezone-normalized to +03:00",
            "Rating values are numeric and on a 1-5 scale",
            "All reviews in the artifact are valid and complete"
        ],
        "confidence": 0.95 if len(reviews_period) >= 10 else 0.70
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Platform Distribution and Sentiment by Platform
# ============================================================================

if len(reviews_period) > 0:
    platform_counts = reviews_period['source'].value_counts()
    
    # Identify platform with highest positive ratio
    platform_sentiment = {}
    for platform in reviews_period['source'].unique():
        platform_reviews = reviews_period[reviews_period['source'] == platform]
        positive_count = (platform_reviews['rating'] >= 4).sum()
        total_count = len(platform_reviews)
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        platform_sentiment[platform] = {
            'positive': positive_count,
            'total': total_count,
            'ratio': positive_ratio
        }
    
    # Find platform with highest positive ratio
    best_platform = max(platform_sentiment.items(), key=lambda x: x[1]['ratio'])
    best_platform_name = best_platform[0]
    best_platform_data = best_platform[1]
    
    finding_2 = {
        "title": "Platform with Highest Positive Review Ratio",
        "claim": f"Among the platforms analyzed, '{best_platform_name}' has the highest proportion of positive reviews (rating ≥4), with {best_platform_data['positive']} positive reviews out of {best_platform_data['total']} total reviews on the {best_platform_name} platform ({best_platform_data['ratio']*100:.1f}% positive ratio).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "best_platform": {
                "value": best_platform_name,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_ratio": {
                "value": round(best_platform_data['ratio'], 2),
                "unit": "proportion",
                "numerator": best_platform_data['positive'],
                "denominator": best_platform_data['total'],
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_count": {
                "value": best_platform_data['positive'],
                "unit": "count",
                "numerator": best_platform_data['positive'],
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_on_platform": {
                "value": best_platform_data['total'],
                "unit": "count",
                "numerator": best_platform_data['total'],
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_period),
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20",
            f"Platforms in analysis period: {list(platform_counts.index)}",
            f"Platform distribution: {dict(platform_counts)}",
            f"Positive defined as rating >= 4"
        ],
        "assumptions": [
            "Rating values are numeric and on a 1-5 scale",
            "Positive sentiment threshold is rating >= 4",
            "All platforms have equal validity and completeness"
        ],
        "confidence": 0.90 if best_platform_data['total'] >= 5 else 0.65
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Language Distribution
# ============================================================================

if len(reviews_period) > 0:
    language_counts = reviews_period['language'].value_counts()
    
    finding_3 = {
        "title": "Review Language Distribution",
        "claim": f"During the analysis period, reviews were submitted in {len(language_counts)} language(s). {language_counts.index[0]} reviews comprise {language_counts.iloc[0]} out of {len(reviews_period)} reviews ({language_counts.iloc[0]/len(reviews_period)*100:.1f}%).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_languages": {
                "value": len(language_counts),
                "unit": "count",
                "numerator": len(language_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "primary_language": {
                "value": language_counts.index[0],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "primary_language_count": {
                "value": language_counts.iloc[0],
                "unit": "count",
                "numerator": language_counts.iloc[0],
                "denominator": len(reviews_period),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_period),
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20",
            f"Language distribution: {dict(language_counts)}",
            f"Total reviews analyzed: {len(reviews_period)}"
        ],
        "assumptions": [
            "Language field is accurately populated in the reviews artifact",
            "Language classification is reliable and consistent"
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
    json.dump(output, f, indent=2)