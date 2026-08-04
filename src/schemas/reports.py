from __future__ import annotations

from typing import Any, TypedDict


class ReportOutput(TypedDict):
    html_path: str
    pdf_path: str | None
    pdf_warning: str | None
    whatsapp_summary: str
    whatsapp_char_count: int
    generated_at: str
    context: dict[str, Any]
