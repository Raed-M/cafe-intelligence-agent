import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load input/output paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-07-06T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-13T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews to analysis period
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
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-07-06 to 2026-07-13), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {', '.join([f'{int(k)} stars: {v} reviews' for k, v in sorted(rating_counts.items())])}.",
        "finding_type": "voice_of_customer_metric",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars (1-5)",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "rating_5_stars": {
                "value": rating_counts.get(5, 0),
                "unit": "count",
                "numerator": rating_counts.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "rating_4_stars": {
                "value": rating_counts.get(4, 0),
                "unit": "count",
                "numerator": rating_counts.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "rating_3_stars": {
                "value": rating_counts.get(3, 0),
                "unit": "count",
                "numerator": rating_counts.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "rating_2_stars": {
                "value": rating_counts.get(2, 0),
                "unit": "count",
                "numerator": rating_counts.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "rating_1_stars": {
                "value": rating_counts.get(1, 0),
                "unit": "count",
                "numerator": rating_counts.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-07-06 to 2026-07-13 (7 days)",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(reviews_analysis['source'].unique())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the dataset are included without filtering by language or content quality"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts().to_dict()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic, representing {round(100*language_counts.get('en', 0)/len(reviews_analysis), 1)}% and {round(100*language_counts.get('ar', 0)/len(reviews_analysis), 1)}% respectively.",
        "finding_type": "voice_of_customer_coverage",
        "metrics": {
            "english_reviews": {
                "value": language_counts.get('en', 0),
                "unit": "count",
                "numerator": language_counts.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": language_counts.get('ar', 0),
                "unit": "count",
                "numerator": language_counts.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "english_percentage": {
                "value": round(100*language_counts.get('en', 0)/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": language_counts.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            },
            "arabic_percentage": {
                "value": round(100*language_counts.get('ar', 0)/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": language_counts.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Bilingual review coverage: {len(reviews_analysis)} reviews",
            f"Language distribution: {', '.join([f'{lang}: {count}' for lang, count in language_counts.items()])}"
        ],
        "assumptions": [
            "Language field is accurately populated in the cleaned reviews artifact",
            "Language codes are 'en' for English and 'ar' for Arabic"
        ],
        "confidence": 1.0
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Source Coverage
# ============================================================================

if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts().to_dict()
    
    finding_3 = {
        "title": "Review Source Distribution",
        "claim": f"Review sources in the analysis period: {', '.join([f'{source}: {count} reviews' for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)])}.",
        "finding_type": "voice_of_customer_coverage",
        "metrics": {
            "total_sources": {
                "value": len(source_counts),
                "unit": "count",
                "numerator": len(source_counts),
                "denominator": None,
                "period_start": "2026-07-06T00:00:00+03:00",
                "period_end": "2026-07-13T00:00:00+03:00"
            }
        },
        "source_names": list(source_counts.keys()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Unique sources: {len(source_counts)}",
            f"Source breakdown: {', '.join([f'{source}: {count}' for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)])}"
        ],
        "assumptions": [
            "Source field is accurately populated in the cleaned reviews artifact",
            "All source values are valid and represent distinct review platforms/channels"
        ],
        "confidence": 1.0
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
