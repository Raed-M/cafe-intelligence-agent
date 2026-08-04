from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class SourceConfig(BaseModel):
    name: str
    parser: str
    path: str
    required_for: list[str]
    sheet: str | None = None


class SourceRegistry(BaseModel):
    sources: list[SourceConfig]

    def by_name(self, name: str) -> SourceConfig:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(name)


def load_source_registry(path: Path) -> SourceRegistry:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(data)
