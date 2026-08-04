"""Deterministic charts rendered only from structured, validated result
artifacts -- never from a number a chart itself invented.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas.findings import AnalystFinding


def render_menu_engineering_chart(final_findings: list[AnalystFinding], out_dir: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quadrant_data: list[dict[str, Any]] | None = None
    for f in final_findings:
        if f["analyst_name"] != "margin":
            continue
        try:
            result = json.loads(Path(f["result_artifact"]["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for finding_obj in result.get("findings", []):
            items = finding_obj.get("menu_engineering_items")
            if items:
                quadrant_data = items
                break
        if quadrant_data:
            break

    if not quadrant_data:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "menu_engineering.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    for item in quadrant_data:
        ax.scatter(item.get("popularity", 0), item.get("contribution", 0))
        ax.annotate(item.get("sku", ""), (item.get("popularity", 0), item.get("contribution", 0)), fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Contribution margin")
    ax.set_title("Menu Engineering Quadrants")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)
