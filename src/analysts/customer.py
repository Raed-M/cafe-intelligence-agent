from pathlib import Path

from src.analysts.base import AnalystSpec

SPEC = AnalystSpec(
    name="customer",
    prompt_path=Path("prompts/analysts/customer.md"),
    required_artifacts=["reviews"],
    optional_artifacts=["pos", "menu"],
)
