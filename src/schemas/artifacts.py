"""Artifact reference contracts. Large data never enters graph state directly;
only these lightweight references do."""
from __future__ import annotations

from typing import TypedDict


class ArtifactRef(TypedDict):
    path: str
    media_type: str
    sha256: str
    schema_version: str
    row_count: int | None
    byte_size: int
    created_at: str
