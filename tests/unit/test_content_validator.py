from src.validation.content_validator import validate_content_ideas

OPENING_HOURS = {"default": "07:00-23:00"}

FINDING = {
    "finding_id": "F1", "analyst_name": "sales", "title": "t", "claim": "c", "finding_type": "trend",
    "evidence": [{
        "metric_name": "net_revenue", "value": 1000, "unit": "SAR", "numerator": None, "denominator": None,
        "period_start": "2026-01-05", "period_end": "2026-01-12", "comparison_period_start": None,
        "comparison_period_end": None, "source_names": ["pos"], "result_path": "x", "result_key": "net_revenue",
    }],
    "source_names": ["pos"], "code_artifact": {}, "result_artifact": {}, "sample_size": 10,
    "coverage_notes": [], "assumptions": [], "confidence": 0.8, "business_impact_score": 0.6,
    "actionability_score": 0.6, "revision_count": 0, "execution_metadata": {},
}

CONTEXT_BUNDLE = {
    "recommendation_period_start": "2026-01-12T00:00:00+03:00",
    "recommendation_period_end": "2026-01-19T00:00:00+03:00",
    "search_queries": [],
    "evidence": [
        {"context_id": "ev1", "kind": "event", "title": "Local market", "date_start": None, "date_end": None,
         "location": None, "source": "tavily", "source_url_or_artifact": None, "retrieved_at": "now", "summary": ""},
        {"context_id": "cal1", "kind": "calendar", "title": "Ramadan", "date_start": None, "date_end": None,
         "location": None, "source": "hijridate", "source_url_or_artifact": None, "retrieved_at": "now", "summary": ""},
    ],
    "posting_windows": [
        {"window_id": "pw1", "post_date": "2026-01-13", "start_time_local": "18:00", "end_time_local": "19:00",
         "busy_metric_keys": ["hourly_valid_transaction_count"], "demand_score": 10.0, "prayer_relation": None,
         "event_context_ids": [], "rationale": "busy hour"},
    ],
    "search_status": "success", "warnings": [],
}


def _idea(**overrides):
    base = {
        "idea_id": "I1", "hook_ar": "عرض خاص", "hook_en": "Try our new drink",
        "format": "reel", "product_sku": "ICE-001", "product_name_ar": "لاتيه",
        "product_name_en": "Iced Latte", "finding_id": "F1", "cited_metric_keys": ["net_revenue"],
        "local_context_ids": ["ev1"], "calendar_context_ids": ["cal1"], "posting_window_id": "pw1",
        "timing_metric_keys": ["hourly_valid_transaction_count"], "rationale_ar": "لأن",
        "rationale_en": "because sales are strong at this hour", "post_date": "2026-01-13",
        "post_time_local": "18:30", "timing_reason": "peak hour", "inventory_suitability": "unknown",
    }
    base.update(overrides)
    return base


def _three_distinct_ideas():
    return [
        _idea(idea_id="I1", hook_en="Try our new drink", rationale_en="Sales are strong this week."),
        _idea(idea_id="I2", hook_en="Cool down with iced latte", rationale_en="Peak afternoon demand observed."),
        _idea(idea_id="I3", hook_en="Weekend treat time", rationale_en="Weekend footfall is high."),
    ]


def test_hook_with_number_not_matching_cited_metric_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["rationale_en"] = "Sales rose 55% this week according to our data."
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]
    assert any("do not match any cited metric" in i for i in out["issues"]["I1"])


def test_hook_with_number_matching_cited_metric_accepted():
    ideas = _three_distinct_ideas()
    ideas[0]["rationale_en"] = "Net revenue reached SAR 1000 this week."
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert out["valid"], out["issues"]


def test_valid_ideas_pass():
    out = validate_content_ideas(_three_distinct_ideas(), [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert out["valid"], out["issues"]


def test_wrong_finding_id_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["finding_id"] = "F-does-not-exist"
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]
    assert "I1" in out["issues"]


def test_wrong_metric_key_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["cited_metric_keys"] = ["made_up_metric"]
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]


def test_closed_time_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["post_time_local"] = "03:00"
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]
    assert any("outside opening hours" in i for i in out["issues"]["I1"])


def test_inactive_sku_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["product_sku"] = "RETIRED-999"
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]


def test_stock_risk_item_flagged_unless_addressed():
    ideas = _three_distinct_ideas()
    ideas[0]["inventory_suitability"] = "supported"
    out = validate_content_ideas(
        ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS, skus_with_stock_risk={"ICE-001"},
    )
    assert not out["valid"]


def test_nonexistent_context_id_rejected():
    ideas = _three_distinct_ideas()
    ideas[0]["local_context_ids"] = ["not-a-real-context-id"]
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]


def test_duplicate_hooks_rejected():
    ideas = _three_distinct_ideas()
    ideas[1]["hook_en"] = ideas[0]["hook_en"]
    out = validate_content_ideas(ideas, [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]
    assert "_distinctness" in out["issues"]


def test_wrong_count_rejected():
    out = validate_content_ideas(_three_distinct_ideas()[:2], [FINDING], CONTEXT_BUNDLE, {"ICE-001"}, OPENING_HOURS)
    assert not out["valid"]
    assert "_count" in out["issues"]
