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
analysis_start = "2026-03-09T00:00:00+03:00"
analysis_end = "2026-03-16T00:00:00+03:00"

# Convert to datetime for comparison, removing timezone info for comparison with tz-naive data
analysis_start_dt = pd.to_datetime(analysis_start).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end).tz_localize(None)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.tz_localize(None)
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Compute rating distribution and average
rating_counts = analysis_reviews['rating'].value_counts().sort_index()
avg_rating = analysis_reviews['rating'].mean()
total_reviews = len(analysis_reviews)

# Get source names from reviews
source_names = sorted(analysis_reviews['source'].unique().tolist())

# Classify sentiment and topics
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    rating_dist = {}
    for rating in sorted(analysis_reviews['rating'].unique()):
        count = len(analysis_reviews[analysis_reviews['rating'] == rating])
        rating_dist[f"rating_{int(rating)}_count"] = count
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), {total_reviews} reviews were collected with an average rating of {avg_rating:.2f} out of 5.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {total_reviews}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Reviews are filtered by date within the analysis period",
            "Rating values are numeric and valid",
            "All reviews in the dataset have complete rating information"
        ],
        "confidence": 0.95 if total_reviews > 10 else 0.7
    }
    
    # Add rating distribution to metrics
    for rating in sorted(analysis_reviews['rating'].unique()):
        count = len(analysis_reviews[analysis_reviews['rating'] == rating])
        finding1["metrics"][f"rating_{int(rating)}_count"] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": total_reviews,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding1)

# Finding 2: Language Distribution
if total_reviews > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {total_reviews} reviews in the analysis period, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": language_counts.get('en', 0),
                "unit": "count",
                "numerator": language_counts.get('en', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": language_counts.get('ar', 0),
                "unit": "count",
                "numerator": language_counts.get('ar', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Language distribution: {dict(language_counts)}",
            f"Bilingual review coverage: {len(analysis_reviews[analysis_reviews['language'].notna()])} reviews with language specified"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Reviews are classified as either 'en' or 'ar'"
        ],
        "confidence": 0.9 if total_reviews > 5 else 0.6
    }
    
    findings.append(finding2)

# Finding 3: Source Distribution
if total_reviews > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"Reviews in the analysis period come from {len(source_counts)} sources: {', '.join([f'{source} ({count})' for source, count in source_counts.items()])}.",
        "finding_type": "source_coverage",
        "metrics": {},
        "source_names": source_names,
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Source distribution: {dict(source_counts)}",
            f"Total unique sources: {len(source_counts)}"
        ],
        "assumptions": [
            "Source field is accurately populated",
            "All reviews have a source attribution"
        ],
        "confidence": 0.95
    }
    
    # Add source counts to metrics
    for source, count in source_counts.items():
        finding3["metrics"][f"source_{source}_count"] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": total_reviews,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if total_reviews > 0 else "insufficient_data",
    "findings": findings[:3]  # Return at most 3 findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Output written to {output_path}")
print(f"Total reviews analyzed: {total_reviews}")
print(f"Average rating: {avg_rating:.2f}")
print(f"Sources: {source_names}")
