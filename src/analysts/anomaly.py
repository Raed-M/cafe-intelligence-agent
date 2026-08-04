from pathlib import Path

from src.analysts.base import AnalystSpec

SPEC = AnalystSpec(
    name="anomaly",
    prompt_path=Path("prompts/analysts/anomaly.md"),
    required_artifacts=["pos"],
    optional_artifacts=["traffic", "inventory", "reviews"],
)
