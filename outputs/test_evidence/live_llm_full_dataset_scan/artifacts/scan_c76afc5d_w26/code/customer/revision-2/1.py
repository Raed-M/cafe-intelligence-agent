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
analysis_start = "2026-07-06T00:00:00+03:00"
analysis_end = "2026-07-13T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Calculate metrics
findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Sentiment classification based on rating
    positive_count = len(reviews_analysis[reviews_analysis['rating'] >= 4])
    neutral_count = len(reviews_analysis[reviews_analysis['rating'] == 3])
    negative_count = len(reviews_analysis[reviews_analysis['rating'] < 3])
    
    finding1 = {
        "title": "Customer Rating Distribution and Sentiment",
        "claim": f"During the analysis period (2026-07-06 to 2026-07-13), the cafe received {len(reviews_analysis)} reviews with an average rating of {avg_rating:.2f}/5. {positive_count} reviews were positive (4-5 stars), {neutral_count} were neutral (3 stars), and {negative_count} were negative (1-2 stars).",
        "finding_type": "voice_of_customer",
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
            "positive_reviews_count": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews_count": {
                "value": neutral_count,
                "unit": "count",
                "numerator": neutral_count,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews_count": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis covers {len(reviews_analysis)} reviews from {len(reviews_analysis['source'].unique())} review platforms",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Positive sentiment defined as 4-5 stars, neutral as 3 stars, negative as 1-2 stars",
            "All reviews in the artifact are valid and complete"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Review Platform Distribution
if len(reviews_analysis) > 0:
    platform_counts = reviews_analysis['source'].value_counts()
    
    finding2 = {
        "title": "Review Source Platform Distribution",
        "claim": f"Reviews during the analysis period came from {len(platform_counts)} different platforms. The distribution shows {', '.join([f'{platform}: {count} reviews' for platform, count in platform_counts.items()])}.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_review_platforms": {
                "value": len(platform_counts),
                "unit": "count",
                "numerator": len(platform_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Platform breakdown: {dict(platform_counts)}"
        ],
        "assumptions": [
            "Platform names in 'source' column represent distinct review channels",
            "All reviews are properly attributed to their source platform"
        ],
        "confidence": 0.95
    }
    
    # Add platform-specific metrics
    for platform, count in platform_counts.items():
        platform_key = f"{platform}_review_count"
        finding2["metrics"][platform_key] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding2)

# Finding 3: Language Distribution in Reviews
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding3 = {
        "title": "Review Language Distribution",
        "claim": f"Customer reviews were submitted in {len(language_counts)} languages. {', '.join([f'{lang}: {count} reviews' for lang, count in language_counts.items()])}.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_languages": {
                "value": len(language_counts),
                "unit": "count",
                "numerator": len(language_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language breakdown: {dict(language_counts)}"
        ],
        "assumptions": [
            "Language field accurately reflects the language of each review",
            "Reviews are classified as either Arabic or English"
        ],
        "confidence": 0.95
    }
    
    # Add language-specific metrics
    for language, count in language_counts.items():
        lang_key = f"{language}_review_count"
        finding3["metrics"][lang_key] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": analysis_start,
            "period_end": analysis_end
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
print(f"Output written to {output_path}")
