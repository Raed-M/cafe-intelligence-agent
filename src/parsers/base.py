from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.config.runtime_config import RuntimeCafeConfig
from src.config.source_registry import SourceConfig
from src.schemas.sources import ErrorRecord, SourceResult


@dataclass
class RunContext:
    run_id: str
    config: RuntimeCafeConfig

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def artifact_root(self) -> Path:
        return Path(self.config.artifact_root, self.run_id)


class SourceParser(Protocol):
    def __call__(self, source: SourceConfig, ctx: RunContext) -> SourceResult: ...


def failed_result(source_name: str, exc: Exception) -> SourceResult:
    return SourceResult(
        source_name=source_name,
        status="failed",
        raw_row_count=0,
        accepted_row_count=0,
        rejected_row_count=0,
        artifact=None,
        schema_version="1.0",
        date_min=None,
        date_max=None,
        warnings=[],
        error=ErrorRecord(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        ),
    )


def run_parser_safely(fn, source: SourceConfig, ctx: RunContext) -> SourceResult:
    """One failed source must not crash unrelated branches."""
    try:
        return fn(source, ctx)
    except Exception as exc:  # noqa: BLE001
        return failed_result(source.name, exc)
