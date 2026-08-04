"""Exports the compiled main graph's structure as a Mermaid diagram."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.main_graph import build_main_graph


def main() -> None:
    graph = build_main_graph()
    mermaid = graph.get_graph().draw_mermaid()
    out_path = Path("outputs") / "graph.mmd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(mermaid)


if __name__ == "__main__":
    main()
