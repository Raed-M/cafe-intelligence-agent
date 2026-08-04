"""Canonical artifact writer/reader. Large data is written once to disk under
outputs/artifacts/<run_id>/...; graph state only ever holds ArtifactRef metadata.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas.artifacts import ArtifactRef


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_ref(path: Path, media_type: str, schema_version: str, row_count: int | None) -> ArtifactRef:
    return ArtifactRef(
        path=str(path),
        media_type=media_type,
        sha256=_sha256_of_file(path),
        schema_version=schema_version,
        row_count=row_count,
        byte_size=path.stat().st_size,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_dataframe(
    df: pd.DataFrame, path: Path, schema_version: str = "1.0"
) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return _make_ref(path, "application/x-parquet", schema_version, len(df))


def read_dataframe(ref: ArtifactRef) -> pd.DataFrame:
    return pd.read_parquet(ref["path"])


def write_json(obj: Any, path: Path, schema_version: str = "1.0", row_count: int | None = None) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return _make_ref(path, "application/json", schema_version, row_count)


def read_json(ref: ArtifactRef) -> Any:
    return json.loads(Path(ref["path"]).read_text(encoding="utf-8"))


def artifact_dir(artifact_root: Path, run_id: str, *parts: str) -> Path:
    return Path(artifact_root, run_id, *parts)
