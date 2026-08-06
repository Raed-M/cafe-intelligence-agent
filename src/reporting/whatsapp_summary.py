"""Deterministic WhatsApp-length bilingual summary (Arabic-first per profile
notes, English follows) -- the template the plan calls "preferred"
(prompts/report_summary.md, plan section 16.12). Every number here is read
directly from final_findings/content_ideas, so nothing can drift from what was
actually approved.

An optional constrained-LLM compression pass can shorten this deterministic
text further (never given raw data, only the already-built text). It is
disabled unless a caller explicitly opts in, and its output must still pass a
deterministic post-check -- every number and finding_id in the compressed text
must already appear in the deterministic text, and it must fit the character
limit -- or the deterministic text is used unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from src.schemas.content import ContentIdea
from src.schemas.findings import AnalystFinding

Compressor = Callable[[str, str], str]

REPORT_SUMMARY_PROMPT_PATH = Path("prompts/report_summary.md")

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def count_unicode_chars(text: str) -> int:
    return len(text)


def _default_llm_compressor(model_name: str) -> Compressor:
    def _compress(system_prompt: str, deterministic_text: str) -> str:
        from src.tools.llm_factory import extract_text, get_chat_model

        llm = get_chat_model(model_name, temperature=0)
        user_prompt = f"Text to compress:\n{deterministic_text}"
        resp = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return extract_text(resp)

    return _compress


def _compressed_summary_passes_post_check(
    deterministic_text: str, compressed_text: str, max_chars: int, allowed_finding_ids: set[str]
) -> bool:
    if count_unicode_chars(compressed_text) > max_chars:
        return False
    allowed_numbers = set(_NUMBER_RE.findall(deterministic_text))
    if set(_NUMBER_RE.findall(compressed_text)) - allowed_numbers:
        return False
    mentioned_ids = {fid for fid in allowed_finding_ids if fid in deterministic_text}
    disallowed_ids = allowed_finding_ids - mentioned_ids
    if any(fid in compressed_text for fid in disallowed_ids):
        return False
    return True


def build_whatsapp_summary(
    cafe_name: str,
    run_status: str,
    analysis_period: dict[str, str],
    final_findings: list[AnalystFinding],
    content_ideas: list[ContentIdea],
    report_reference: str,
    max_chars: int,
    use_llm_compression: bool = False,
    model_name: str | None = None,
    compressor: Compressor | None = None,
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

    if use_llm_compression and (compressor is not None or model_name):
        try:
            system_prompt = REPORT_SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")
            compress = compressor or _default_llm_compressor(model_name)  # type: ignore[arg-type]
            compressed = compress(system_prompt, text)
            allowed_ids = {f["finding_id"] for f in final_findings}
            if _compressed_summary_passes_post_check(text, compressed, max_chars, allowed_ids):
                return compressed, count_unicode_chars(compressed)
        except Exception:  # noqa: BLE001
            pass  # compression unavailable or failed its post-check -- use the deterministic text

    return text, count_unicode_chars(text)
