import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read the reviews artifact
reviews_df = pd.read_parquet(inputs['reviews'])

# Convert date column to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-05-11T00:00:00+03:00").replace(tzinfo=None)
analysis_end = datetime.fromisoformat("2026-05-18T00:00:00+03:00").replace(tzinfo=None)

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

# Calculate rating distribution and average
rating_distribution = analysis_reviews['rating'].value_counts().sort_index().to_dict()
average_rating = float(analysis_reviews['rating'].mean())
total_reviews = int(len(analysis_reviews))

# Separate by language
english_reviews = analysis_reviews[analysis_reviews['language'] == 'en']
arabic_reviews = analysis_reviews[analysis_reviews['language'] == 'ar']

english_count = int(len(english_reviews))
arabic_count = int(len(arabic_reviews))

# Analyze sentiment and topics
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    finding1 = {
        "title": "Review Rating Distribution and Average (May 11-18, 2026)",
        "claim": f"During the analysis period, {total_reviews} reviews were collected with an average rating of {average_rating:.2f} out of 5. The distribution shows {int(rating_distribution.get(5, 0))} 5-star, {int(rating_distribution.get(4, 0))} 4-star, {int(rating_distribution.get(3, 0))} 3-star, {int(rating_distribution.get(2, 0))} 2-star, and {int(rating_distribution.get(1, 0))} 1-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(average_rating, 2),
                "unit": "stars",
                "numerator": round(float(analysis_reviews['rating'].sum()), 2),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_distribution.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_distribution.get(5, 0)),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "four_star_count": {
                "value": int(rating_distribution.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_distribution.get(4, 0)),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "three_star_count": {
                "value": int(rating_distribution.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_distribution.get(3, 0)),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "two_star_count": {
                "value": int(rating_distribution.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_distribution.get(2, 0)),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_distribution.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_distribution.get(1, 0)),
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: 2026-05-11 to 2026-05-18",
            f"Total reviews in period: {total_reviews}",
            f"English reviews: {english_count}",
            f"Arabic reviews: {arabic_count}",
            f"Sources represented: {', '.join(analysis_reviews['source'].unique())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in the specified timezone",
            "All reviews in the artifact are from the specified analysis period or earlier"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if total_reviews > 0:
    english_pct = (english_count / total_reviews * 100) if total_reviews > 0 else 0
    arabic_pct = (arabic_count / total_reviews * 100) if total_reviews > 0 else 0
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {total_reviews} reviews collected during the analysis period, {english_count} ({english_pct:.1f}%) were in English and {arabic_count} ({arabic_pct:.1f}%) were in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": english_count,
                "unit": "count",
                "numerator": english_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": arabic_count,
                "unit": "count",
                "numerator": arabic_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "english_percentage": {
                "value": round(english_pct, 1),
                "unit": "percent",
                "numerator": english_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "arabic_percentage": {
                "value": round(arabic_pct, 1),
                "unit": "percent",
                "numerator": arabic_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Bilingual review collection: {english_count} English + {arabic_count} Arabic",
            f"Total coverage: {total_reviews} reviews",
            f"Language distribution reflects customer base composition"
        ],
        "assumptions": [
            "Language field accurately reflects the language of each review",
            "All reviews have a valid language classification"
        ],
        "confidence": 0.98
    }
    findings.append(finding2)

# Finding 3: High-Rating vs Low-Rating Review Comparison
high_rating_reviews = analysis_reviews[analysis_reviews['rating'] >= 4]
low_rating_reviews = analysis_reviews[analysis_reviews['rating'] <= 2]

high_rating_count = int(len(high_rating_reviews))
low_rating_count = int(len(low_rating_reviews))

if high_rating_count > 0 or low_rating_count > 0:
    high_rating_pct = (high_rating_count / total_reviews * 100) if total_reviews > 0 else 0
    low_rating_pct = (low_rating_count / total_reviews * 100) if total_reviews > 0 else 0
    
    finding3 = {
        "title": "High vs Low Rating Review Prevalence",
        "claim": f"During the analysis period, {high_rating_count} reviews ({high_rating_pct:.1f}%) were rated 4-5 stars (positive), while {low_rating_count} reviews ({low_rating_pct:.1f}%) were rated 1-2 stars (negative).",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "high_rating_reviews": {
                "value": high_rating_count,
                "unit": "count",
                "numerator": high_rating_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "low_rating_reviews": {
                "value": low_rating_count,
                "unit": "count",
                "numerator": low_rating_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "high_rating_percentage": {
                "value": round(high_rating_pct, 1),
                "unit": "percent",
                "numerator": high_rating_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            },
            "low_rating_percentage": {
                "value": round(low_rating_pct, 1),
                "unit": "percent",
                "numerator": low_rating_count,
                "denominator": total_reviews,
                "period_start": "2026-05-11T00:00:00+03:00",
                "period_end": "2026-05-18T00:00:00+03:00"
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Positive reviews (4-5 stars): {high_rating_count}",
            f"Negative reviews (1-2 stars): {low_rating_count}",
            f"Neutral reviews (3 stars): {int(rating_distribution.get(3, 0))}",
            f"Total sample: {total_reviews}"
        ],
        "assumptions": [
            "Rating scale is 1-5 with 4-5 considered positive and 1-2 considered negative",
            "All reviews have valid rating values"
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

print(f"Analysis complete. Output written to {output_path}")
print(f"Total findings: {len(findings)}")
