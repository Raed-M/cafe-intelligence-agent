from __future__ import annotations

from src.schemas.sources import DataQualitySummary, SourceQuality


def build_data_quality_summary(
    source_qualities: list[SourceQuality],
    sources_successful: list[str],
    sources_partial: list[str],
    sources_failed: list[str],
    critical_dependencies_missing: list[str],
    warnings: list[str],
) -> DataQualitySummary:
    return DataQualitySummary(
        source_summaries=source_qualities,
        sources_successful=sources_successful,
        sources_partial=sources_partial,
        sources_failed=sources_failed,
        critical_dependencies_missing=critical_dependencies_missing,
        total_rows_in=sum(s["rows_in"] for s in source_qualities),
        total_rows_dropped=sum(s["rows_dropped"] for s in source_qualities),
        total_rows_repaired=sum(s["rows_repaired"] for s in source_qualities),
        warnings=warnings,
    )
