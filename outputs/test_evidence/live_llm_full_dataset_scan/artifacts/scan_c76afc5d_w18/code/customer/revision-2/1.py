import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Analysis period
analysis_start = "2026-05-11T00:00:00+03:00"
analysis_end = "2026-05-18T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all unique sources in the data
all_sources = reviews_df['source'].unique().tolist()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {dict(sorted(rating_dist.items()))}",
        "finding_type": "rating_distribution",
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
            "rating_5_count": {
                "value": rating_dist.get(5, 0),
                "unit": "count",
                "numerator": rating_dist.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": rating_dist.get(4, 0),
                "unit": "count",
                "numerator": rating_dist.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": rating_dist.get(3, 0),
                "unit": "count",
                "numerator": rating_dist.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": rating_dist.get(2, 0),
                "unit": "count",
                "numerator": rating_dist.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": rating_dist.get(1, 0),
                "unit": "count",
                "numerator": rating_dist.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Data sources represented: {', '.join(all_sources)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews with non-null ratings are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    lang_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"During the analysis period, reviews were submitted in {len(lang_dist)} language(s). English: {lang_dist.get('English', 0)} reviews, Arabic: {lang_dist.get('Arabic', 0)} reviews.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": lang_dist.get('English', 0),
                "unit": "count",
                "numerator": lang_dist.get('English', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": lang_dist.get('Arabic', 0),
                "unit": "count",
                "numerator": lang_dist.get('Arabic', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews_with_language": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution computed from {len(reviews_analysis)} reviews",
            f"Languages identified: {', '.join(lang_dist.keys())}",
            f"All reviews in analysis period included"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Language values are standardized (English/Arabic)"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Source Distribution
if len(reviews_analysis) > 0:
    source_dist = reviews_analysis['source'].value_counts().to_dict()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, {len(reviews_analysis)} reviews were collected from {len(source_dist)} source(s): {', '.join([f'{src}: {count}' for src, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True)])}",
        "finding_type": "source_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sources identified: {', '.join(all_sources)}",
            f"Total reviews across all sources: {len(reviews_analysis)}",
            f"Source distribution: {source_dist}"
        ],
        "assumptions": [
            "Source field is accurately populated",
            "All review sources are represented in the data"
        ],
        "confidence": 0.95
    }
    
    # Add source-specific metrics
    for source in all_sources:
        source_count = source_dist.get(source, 0)
        finding3["metrics"][f"{source}_count"] = {
            "value": source_count,
            "unit": "count",
            "numerator": source_count,
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

print(f"Analysis complete. Output written to {output_path}")
print(f"Findings generated: {len(findings)}")
