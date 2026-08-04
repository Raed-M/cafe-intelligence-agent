"""Deterministic WhatsApp-length bilingual summary (Arabic-first per profile
notes, English follows). No LLM involvement -- every number here is read
directly from final_findings/content_ideas, so nothing can drift from what was
actually approved.
"""
from __future__ import annotations

from typing import Any

from src.schemas.content import ContentIdea
from src.schemas.findings import AnalystFinding


def count_unicode_chars(text: str) -> int:
    return len(text)


def build_whatsapp_summary(
    cafe_name: str,
    run_status: str,
    analysis_period: dict[str, str],
    final_findings: list[AnalystFinding],
    content_ideas: list[ContentIdea],
    report_reference: str,
    max_chars: int,
) -> tuple[str, int]:
    period_label = f"{analysis_period['start'][:10]} - {analysis_period['end'][:10]}"

    if run_status in ("failed",) or not final_findings:
        body_ar = f"تقرير {cafe_name} ({period_label}): لا توجد أدلة كافية لهذا الأسبوع."
        body_en = f"{cafe_name} report ({period_label}): insufficient evidence this week."
        text = f"{body_ar}\n{body_en}\n{report_reference}"
        return text[:max_chars], count_unicode_chars(text[:max_chars])

    lines_ar = [f"تقرير {cafe_name} ({period_label}):"]
    lines_en = [f"{cafe_name} report ({period_label}):"]
    for f in final_findings[:3]:
        lines_ar.append(f"- {f['title']}")
        lines_en.append(f"- {f['title']}: {f['claim']}")

    if content_ideas:
        lines_en.append(f"{len(content_ideas)} content ideas ready for review.")

    lines_en.append(f"Full report: {report_reference}")

    text = "\n".join(lines_ar) + "\n\n" + "\n".join(lines_en)
    if count_unicode_chars(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text, count_unicode_chars(text)
