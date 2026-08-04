from pathlib import Path

from src.analysts.base import AnalystSpec

SPEC = AnalystSpec(
    name="margin",
    prompt_path=Path("prompts/analysts/margin.md"),
    required_artifacts=["pos", "menu"],
    optional_artifacts=["inventory", "emails"],
)
