from pathlib import Path

from src.analysts.base import AnalystSpec

SPEC = AnalystSpec(
    name="operations",
    prompt_path=Path("prompts/analysts/operations.md"),
    required_artifacts=["pos", "traffic", "staff"],
    optional_artifacts=["inventory"],
)
