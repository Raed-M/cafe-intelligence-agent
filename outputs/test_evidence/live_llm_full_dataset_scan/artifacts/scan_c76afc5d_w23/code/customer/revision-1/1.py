import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import re

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
analysis_start = pd.Timestamp("2026-06-15T00:00:00+03:00")
analysis_end = pd.Timestamp("2026-06-22T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding_1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-06-15 to 2026-06-22), customer ratings averaged {avg_rating:.2f} out of 5.0 across {len(reviews_analysis)} reviews. The distribution shows: 5-star ({int(rating_counts.get(5, 0))} reviews), 4-star ({int(rating_counts.get(4, 0))} reviews), 3-star ({int(rating_counts.get(3, 0))} reviews), 2-star ({int(rating_counts.get(2, 0))} reviews), 1-star ({int(rating_counts.get(1, 0))} reviews).",
        "finding_type": "customer_sentiment",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars (1-5)",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-15T00:00:00+03:00",
                "period_end": "2026-06-22T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22 (7 days)",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid integers from 1 to 5",
            "All reviews in the analysis period are included",
            "No filtering applied based on review text quality"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment and Topic Classification
# ============================================================================

# Define topic keywords in English and Arabic
topic_keywords = {
    "taste_quality": {
        "en": ["taste", "flavor", "delicious", "good", "bad", "quality", "fresh", "stale"],
        "ar": ["طعم", "نكهة", "لذيذ", "جيد", "سيء", "جودة", "طازج", "قديم"]
    },
    "temperature": {
        "en": ["hot", "cold", "warm", "temperature", "ice", "iced", "cool"],
        "ar": ["ساخن", "بارد", "دافئ", "درجة حرارة", "ثلج", "مثلج", "بارد"]
    },
    "service_speed": {
        "en": ["fast", "slow", "quick", "wait", "service", "speed", "rushed", "delayed"],
        "ar": ["سريع", "بطيء", "خدمة", "انتظار", "سرعة", "مسرع", "متأخر"]
    },
    "price_value": {
        "en": ["price", "expensive", "cheap", "cost", "value", "worth", "overpriced"],
        "ar": ["سعر", "غالي", "رخيص", "تكلفة", "قيمة", "يستحق", "مبالغ"]
    },
    "cleanliness": {
        "en": ["clean", "dirty", "hygiene", "sanitary", "mess", "tidy"],
        "ar": ["نظيف", "قذر", "نظافة", "صحي", "فوضى", "مرتب"]
    }
}

def classify_topics(text, language):
    """Classify topics in a review text."""
    if pd.isna(text) or text == "":
        return []
    
    text_lower = str(text).lower()
    found_topics = []
    
    for topic, keywords in topic_keywords.items():
        lang_key = "ar" if language == "Arabic" else "en"
        if lang_key in keywords:
            for keyword in keywords[lang_key]:
                if keyword.lower() in text_lower:
                    found_topics.append(topic)
                    break
    
    return found_topics

# Classify topics in analysis period reviews
reviews_analysis['topics'] = reviews_analysis.apply(
    lambda row: classify_topics(row['text'], row['language']), 
    axis=1
)

# Count topic mentions
topic_mention_counts = Counter()
reviews_with_topics = 0

for topics_list in reviews_analysis['topics']:
    if len(topics_list) > 0:
        reviews_with_topics += 1
        for topic in topics_list:
            topic_mention_counts[topic] += 1

# Get top topics
top_topics = topic_mention_counts.most_common(3)

if len(top_topics) > 0:
    # Build metrics for top topics
    metrics_dict = {}
    for idx, (topic, count) in enumerate(top_topics):
        metrics_dict[f"topic_{idx+1}_name"] = {
            "value": topic,
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": "2026-06-15T00:00:00+03:00",
            "period_end": "2026-06-22T00:00:00+03:00"
        }
        metrics_dict[f"topic_{idx+1}_mentions"] = {
            "value": count,
            "unit": "mentions",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": "2026-06-15T00:00:00+03:00",
            "period_end": "2026-06-22T00:00:00+03:00"
        }
    
    metrics_dict["reviews_with_topics"] = {
        "value": reviews_with_topics,
        "unit": "reviews",
        "numerator": reviews_with_topics,
        "denominator": len(reviews_analysis),
        "period_start": "2026-06-15T00:00:00+03:00",
        "period_end": "2026-06-22T00:00:00+03:00"
    }
    
    metrics_dict["total_topic_mentions"] = {
        "value": sum(topic_mention_counts.values()),
        "unit": "mentions",
        "numerator": sum(topic_mention_counts.values()),
        "denominator": len(reviews_analysis),
        "period_start": "2026-06-15T00:00:00+03:00",
        "period_end": "2026-06-22T00:00:00+03:00"
    }
    
    # Build claim with complete accounting
    top_topics_str = ", ".join([f"{topic} ({count} mentions)" for topic, count in top_topics])
    reviews_without_topics = len(reviews_analysis) - reviews_with_topics
    
    claim_text = f"During the analysis period, {reviews_with_topics} out of {len(reviews_analysis)} reviews contained identifiable topics. The most frequently mentioned topics were: {top_topics_str}. {reviews_without_topics} reviews contained no identified topics or other topics not in the primary classification set."
    
    finding_2 = {
        "title": "Customer Review Topics and Sentiment Themes",
        "claim": claim_text,
        "finding_type": "topic_analysis",
        "metrics": metrics_dict,
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Reviews with identified topics: {reviews_with_topics}",
            f"Reviews without identified topics: {reviews_without_topics}",
            f"Total topic mentions: {sum(topic_mention_counts.values())}",
            "Topic classification based on keyword matching in original review text",
            "Reviews may mention multiple topics; each mention is counted separately",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Topic keywords are representative of customer concerns",
            "Keyword matching in original language (English or Arabic) is reliable",
            "A single review may contain multiple topics",
            "Absence of keywords indicates absence of topic mention"
        ],
        "confidence": 0.70
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Language Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    metrics_dict_lang = {}
    for lang in language_counts.index:
        count = language_counts[lang]
        metrics_dict_lang[f"{lang.lower()}_reviews"] = {
            "value": int(count),
            "unit": "reviews",
            "numerator": int(count),
            "denominator": len(reviews_analysis),
            "period_start": "2026-06-15T00:00:00+03:00",
            "period_end": "2026-06-22T00:00:00+03:00"
        }
    
    lang_dist_str = ", ".join([f"{lang}: {int(count)} reviews" for lang, count in language_counts.items()])
    
    finding_3 = {
        "title": "Review Language Distribution",
        "claim": f"During the analysis period, reviews were submitted in {len(language_counts)} language(s): {lang_dist_str}.",
        "finding_type": "data_coverage",
        "metrics": metrics_dict_lang,
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Total reviews: {len(reviews_analysis)}",
            f"Languages represented: {list(language_counts.index)}"
        ],
        "assumptions": [
            "Language field is accurately populated in source data",
            "All reviews are included regardless of language"
        ],
        "confidence": 1.0
    }
    findings.append(finding_3)

# ============================================================================
# Build Output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)