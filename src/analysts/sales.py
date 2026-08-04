from pathlib import Path

from src.analysts.base import AnalystSpec

SPEC = AnalystSpec(
    name="sales",
    prompt_path=Path("prompts/analysts/sales.md"),
    required_artifacts=["pos", "menu"],
)
