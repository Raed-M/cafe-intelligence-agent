from src.reducers import merge_source_results, upsert_findings


def _finding(fid, rev):
    return {
        "finding_id": fid, "analyst_name": "sales", "title": "t", "claim": "c",
        "finding_type": "trend", "evidence": [], "source_names": [], "code_artifact": {},
        "result_artifact": {}, "sample_size": 1, "coverage_notes": [], "assumptions": [],
        "confidence": 0.9, "business_impact_score": 0.5, "actionability_score": 0.5,
        "revision_count": rev, "execution_metadata": {},
    }


def _source(name, status):
    return {
        "source_name": name, "status": status, "raw_row_count": 1, "accepted_row_count": 1,
        "rejected_row_count": 0, "artifact": None, "schema_version": "1.0",
        "date_min": None, "date_max": None, "warnings": [], "error": None,
    }


def test_merge_source_results_upserts_by_name():
    left = [_source("pos", "failed")]
    right = [_source("pos", "success"), _source("menu", "success")]
    merged = merge_source_results(left, right)
    by_name = {r["source_name"]: r for r in merged}
    assert by_name["pos"]["status"] == "success"
    assert by_name["menu"]["status"] == "success"
    assert len(merged) == 2


def test_upsert_findings_replaces_lower_revision():
    left = [_finding("F1", 0)]
    right = [_finding("F1", 1)]
    merged = upsert_findings(left, right)
    assert len(merged) == 1
    assert merged[0]["revision_count"] == 1


def test_upsert_findings_does_not_regress_to_lower_revision():
    left = [_finding("F1", 2)]
    right = [_finding("F1", 1)]
    merged = upsert_findings(left, right)
    assert merged[0]["revision_count"] == 2


def test_upsert_findings_keeps_distinct_ids():
    merged = upsert_findings([_finding("F1", 0)], [_finding("F2", 0)])
    assert {f["finding_id"] for f in merged} == {"F1", "F2"}
