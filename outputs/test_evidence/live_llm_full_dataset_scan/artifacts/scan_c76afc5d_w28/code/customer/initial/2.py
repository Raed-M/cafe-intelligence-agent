import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-27T00:00:00+03:00")

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
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding_1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5.0 across {len(reviews_analysis)} reviews in the analysis period (2026-07-20 to 2026-07-27).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "average_rating": {
                "value": round(float(avg_rating), 2),
                "unit": "stars",
                "numerator": round(float(avg_rating * len(reviews_analysis)), 2),
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "total_reviews": {
                "value": int(len(reviews_analysis)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Analysis period: 2026-07-20 to 2026-07-27",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Rating distribution: {dict((int(k), int(v)) for k, v in rating_counts.items())}"
        ],
        "assumptions": [
            "Review dates are accurate and in UTC+03:00 timezone",
            "All ratings are valid numeric values between 1 and 5"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Reviews are distributed across {len(language_counts)} languages, with {language_counts.index[0]} being the most common ({int(language_counts.iloc[0])} reviews, {100*float(language_counts.iloc[0])/len(reviews_analysis):.1f}%).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "language_distribution": {
                "value": dict((str(k), int(v)) for k, v in language_counts.items()),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Languages detected: {list(language_counts.index)}"
        ],
        "assumptions": [
            "Language field is accurately populated in the reviews dataset"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Review Platform Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    platform_counts = reviews_analysis['source'].value_counts()
    
    finding_3 = {
        "title": "Review Source Platform Distribution",
        "claim": f"Customer reviews originate from {len(platform_counts)} platforms, with {platform_counts.index[0]} accounting for {int(platform_counts.iloc[0])} reviews ({100*float(platform_counts.iloc[0])/len(reviews_analysis):.1f}% of total).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "platform_distribution": {
                "value": dict((str(k), int(v)) for k, v in platform_counts.items()),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Platforms represented: {list(platform_counts.index)}"
        ],
        "assumptions": [
            "Source field accurately identifies the platform from which each review was collected"
        ],
        "confidence": 0.95
    }
    findings.append(finding_3)

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
