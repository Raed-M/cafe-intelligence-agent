import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read the reviews artifact
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
analysis_start = "2026-07-20T00:00:00+03:00"
analysis_end = "2026-07-27T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime and remove timezone info for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
# Remove timezone awareness from both the data and the comparison values
reviews_df['date'] = reviews_df['date'].dt.tz_localize(None)
analysis_start_dt = analysis_start_dt.tz_localize(None)
analysis_end_dt = analysis_end_dt.tz_localize(None)

# Filter reviews for analysis period
reviews_period = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Language Distribution
if len(reviews_period) > 0:
    language_counts = reviews_period['language'].value_counts().to_dict()
    total_reviews = len(reviews_period)
    
    # Calculate percentages
    language_dist = {}
    for lang, count in language_counts.items():
        percentage = (count / total_reviews) * 100
        language_dist[lang] = {"count": count, "percentage": round(percentage, 1)}
    
    # Sort by count descending
    sorted_langs = sorted(language_dist.items(), key=lambda x: x[1]['count'], reverse=True)
    most_common_lang = sorted_langs[0][0]
    most_common_count = sorted_langs[0][1]['count']
    most_common_pct = sorted_langs[0][1]['percentage']
    
    finding1 = {
        "title": "Review Language Distribution",
        "claim": f"Reviews are distributed across {len(language_dist)} languages, with {most_common_lang} being the most common ({most_common_count} out of {total_reviews} reviews, {most_common_pct}%).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "language_distribution": {
                "value": json.dumps(language_dist),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Total reviews in analysis period: {total_reviews}",
            f"Languages detected: {', '.join(language_dist.keys())}",
            f"Language distribution: {json.dumps(language_dist)}"
        ],
        "assumptions": [
            "Language field in reviews artifact is accurate and complete",
            "Review dates are correctly parsed"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Rating Distribution and Average
if len(reviews_period) > 0:
    # Get rating distribution
    rating_counts = reviews_period['rating'].value_counts().sort_index().to_dict()
    
    # Ensure all ratings 1-5 are represented
    complete_rating_dist = {}
    for i in range(1, 6):
        complete_rating_dist[i] = rating_counts.get(i, 0)
    
    # Calculate average rating
    total_rating_sum = sum(rating * count for rating, count in complete_rating_dist.items())
    avg_rating = total_rating_sum / total_reviews if total_reviews > 0 else 0
    avg_rating = round(avg_rating, 2)
    
    # Find most common rating
    most_common_rating = max(complete_rating_dist.items(), key=lambda x: x[1])[0]
    most_common_rating_count = complete_rating_dist[most_common_rating]
    
    # Build coverage notes with explicit mention of missing ratings
    coverage_notes = [
        f"Total reviews analyzed: {total_reviews}",
        f"Complete rating distribution (1-5 scale): {json.dumps(complete_rating_dist)}"
    ]
    
    # Add explicit note if any ratings are missing
    missing_ratings = [i for i in range(1, 6) if complete_rating_dist[i] == 0]
    if missing_ratings:
        coverage_notes.append(f"No reviews received ratings of {', '.join(map(str, missing_ratings))}")
    
    finding2 = {
        "title": "Review Rating Distribution and Average",
        "claim": f"Average rating across {total_reviews} reviews is {avg_rating} out of 5, with complete distribution across 1-5 scale: {{1: {complete_rating_dist[1]}, 2: {complete_rating_dist[2]}, 3: {complete_rating_dist[3]}, 4: {complete_rating_dist[4]}, 5: {complete_rating_dist[5]}}}. Rating {most_common_rating} is most common with {most_common_rating_count} reviews.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "average_rating": {
                "value": avg_rating,
                "unit": "out of 5",
                "numerator": total_rating_sum,
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_distribution": {
                "value": json.dumps(complete_rating_dist),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": coverage_notes,
        "assumptions": [
            "Rating field contains valid 1-5 scale values",
            "All reviews have a rating value",
            "Rating scale is consistent across all review sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Review Platforms/Sources
if len(reviews_period) > 0:
    platform_counts = reviews_period['source'].value_counts().to_dict()
    total_platforms = len(platform_counts)
    
    # Sort by count descending
    sorted_platforms = sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_platform = sorted_platforms[0][0]
    most_common_platform_count = sorted_platforms[0][1]
    most_common_platform_pct = round((most_common_platform_count / total_reviews) * 100, 1)
    
    finding3 = {
        "title": "Review Platform Distribution",
        "claim": f"Reviews come from {total_platforms} platforms, with {most_common_platform} being the primary source ({most_common_platform_count} out of {total_reviews} reviews, {most_common_platform_pct}%).",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_reviews_by_platform": {
                "value": json.dumps(platform_counts),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "primary_platform": {
                "value": most_common_platform,
                "unit": "platform",
                "numerator": most_common_platform_count,
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Total reviews in analysis period: {total_reviews}",
            f"Platforms detected: {', '.join(platform_counts.keys())}",
            f"Platform distribution: {json.dumps(platform_counts)}"
        ],
        "assumptions": [
            "Source field in reviews artifact accurately identifies review platform",
            "All reviews have a source/platform value"
        ],
        "confidence": 0.95
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output file
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
