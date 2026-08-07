"""Supplier email ingestion.

Headers (From/Date/Subject) are parsed deterministically. Body facts are extracted
using an LLM (see prompts/email_extractor.md) when the active LLM_PROVIDER's API key
is configured; otherwise a deterministic regex-based fallback extractor runs so the
system still produces evidence-backed facts offline. The fallback is always disclosed
via a warning and a lower confidence band -- never silently treated as equal quality.
"""
from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.llm_factory import has_provider_key

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe
from src.tools.concurrency import bounded_map

_HEADER_RE = re.compile(r"^(From|Date|Subject):\s*(.*)$")
_PRICE_CHANGE_RE = re.compile(
    r"(?P<entity>[A-Za-z0-9 /\-]+?)\s+moves from\s+(?P<currency>[A-Z]{3})\s*(?P<old>[\d.]+)"
    r"(?:/(?P<unit>[A-Za-z]+))?\s+to\s+[A-Z]{3}\s*(?P<new>[\d.]+)"
    r"(?:/[A-Za-z]+)?(?:\s+effective\s+(?P<effective>[\w ]+?)(?:\.|$))?",
    re.IGNORECASE,
)
_NOISE_KEYWORDS = ["maintenance", "scheduled maintenance", "reporting module"]
_EVENT_KEYWORDS = ["market", "footfall", "sponsorship", "vendor", "programme", "corniche"]
_DELAY_KEYWORDS = ["delay", "late", "under repair"]
_QUOTE_KEYWORDS = ["quotation", "cups", "lids", "per unit", "sar 0."]
_OFFER_KEYWORDS = ["sample kit", "lead time", "minimum order", "allocate"]


def _parse_email_file(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers = {"from": None, "date": None, "subject": None}
    body_start = 0
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            headers[m.group(1).lower()] = m.group(2).strip()
            body_start = i + 1
        elif line.strip() == "" and headers["subject"] is not None:
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return {"sender": headers["from"], "date": headers["date"], "subject": headers["subject"], "body": body}


def _classify_noise_delay_event_quote_offer(subject: str, body: str) -> str:
    text = f"{subject}\n{body}".lower()
    if any(k in text for k in _NOISE_KEYWORDS):
        return "maintenance"
    if any(k in text for k in _DELAY_KEYWORDS):
        return "delivery_delay"
    if any(k in text for k in _EVENT_KEYWORDS):
        return "event"
    if any(k in text for k in _QUOTE_KEYWORDS):
        return "quote"
    if any(k in text for k in _OFFER_KEYWORDS):
        return "product_offer"
    return "noise"


def _extract_deterministic(email_file: str, parsed: dict[str, Any]) -> list[dict[str, Any]]:
    subject = parsed["subject"] or ""
    body = parsed["body"] or ""
    facts: list[dict[str, Any]] = []

    price_matches = list(_PRICE_CHANGE_RE.finditer(body))
    if price_matches:
        for m in price_matches:
            facts.append({
                "email_file": email_file,
                "sender": parsed["sender"],
                "date": parsed["date"],
                "subject": subject,
                "category": "price_change",
                "entity_or_ingredient": m.group("entity").strip(),
                "old_price": float(m.group("old")),
                "new_price": float(m.group("new")),
                "currency": m.group("currency"),
                "unit": m.group("unit"),
                "effective_date": m.group("effective"),
                "event_start": None,
                "event_end": None,
                "location": None,
                "facts": [m.group(0).strip()],
                "confidence": 0.7,
                "evidence_text": m.group(0).strip(),
                "extraction_mode": "deterministic_fallback",
            })
    else:
        category = _classify_noise_delay_event_quote_offer(subject, body)
        facts.append({
            "email_file": email_file,
            "sender": parsed["sender"],
            "date": parsed["date"],
            "subject": subject,
            "category": category,
            "entity_or_ingredient": None,
            "old_price": None,
            "new_price": None,
            "currency": None,
            "unit": None,
            "effective_date": None,
            "event_start": None,
            "event_end": None,
            "location": None,
            "facts": [body[:280]],
            "confidence": 0.4 if category != "noise" else 0.9,
            "evidence_text": body[:280],
            "extraction_mode": "deterministic_fallback",
        })
    return facts


def _normalize_facts_field(value: Any) -> list[str]:
    """The model is instructed to return facts as list[str], but in practice
    sometimes returns a bare string or a list of {"fact": ...} objects. This
    normalizes either shape to list[str] so every row has a uniform column
    type -- pyarrow otherwise raises on a mixed str/list "facts" column."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("fact") or item.get("evidence_text") or item))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _extract_with_llm(email_file: str, parsed: dict[str, Any], model_name: str) -> list[dict[str, Any]] | None:
    if not has_provider_key():
        return None
    try:
        from src.schemas.emails import EmailExtractionOutput
        from src.tools.llm_factory import get_chat_model

        prompt_path = Path("prompts/email_extractor.md")
        system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else (
            "Extract structured supplier-email facts."
        )
        llm = get_chat_model(model_name, temperature=0)
        structured_llm = llm.with_structured_output(EmailExtractionOutput)
        user_content = (
            f"email_file: {email_file}\nsender: {parsed['sender']}\ndate: {parsed['date']}\n"
            f"subject: {parsed['subject']}\nbody:\n{parsed['body']}\n\n"
            "Extract one or more facts into `items`, each matching the fact schema "
            "(email_file, sender, date, subject, category, entity_or_ingredient, old_price, "
            "new_price, currency, unit, effective_date, event_start, event_end, location, "
            "facts, confidence, evidence_text). Use null for absent fields."
        )
        result = structured_llm.invoke([("system", system_prompt), ("user", user_content)])
        raw = result["items"] if isinstance(result, dict) else result.items  # type: ignore[union-attr]
        for r in raw:
            r["extraction_mode"] = "llm"
            r["facts"] = _normalize_facts_field(r.get("facts"))  # belt-and-suspenders, see docstring above
        return raw
    except Exception:  # noqa: BLE001
        return None


def parse_emails(source: SourceConfig, ctx: RunContext) -> SourceResult:
    files = sorted(glob.glob(str(ctx.data_dir / source.path)))
    raw_row_count = len(files)
    warnings: list[str] = []
    all_facts: list[dict[str, Any]] = []
    failed_files: list[str] = []

    model_name = ctx.config.app_settings.models.email_extractor

    # Point-in-time cutoff: an email dated after the current analysis period
    # hasn't "arrived" yet from the system's perspective, so it must not be
    # visible to any analyst -- excluding it here also means never spending
    # an LLM call (or even the deterministic fallback) extracting facts from
    # data that shouldn't be seen yet. Only applied when the date header
    # parses as an unambiguous ISO date; anything else is included rather
    # than silently dropped on a parsing guess.
    cutoff_date = ctx.config.analysis_period["end"][:10]
    future_files: list[str] = []

    parsed_by_file: dict[str, dict[str, Any]] = {}
    for fpath in files:
        try:
            parsed = _parse_email_file(Path(fpath))
        except Exception as e:  # noqa: BLE001
            failed_files.append(fpath)
            warnings.append(f"failed to parse {fpath}: {e}")
            continue
        email_date = parsed.get("date")
        try:
            is_future = bool(email_date) and datetime.strptime(email_date, "%Y-%m-%d").date().isoformat() > cutoff_date
        except ValueError:
            is_future = False  # unparseable date header -- fail open, include it
        if is_future:
            future_files.append(fpath)
            continue
        parsed_by_file[fpath] = parsed

    if future_files:
        warnings.append(
            f"{len(future_files)} of {raw_row_count} emails dated after the current analysis period "
            f"(after {cutoff_date}) excluded as not-yet-visible at this point in time -- skipped "
            f"entirely, no LLM call spent on them: {[Path(f).name for f in future_files]}"
        )

    def _extract_one(fpath: str) -> tuple[str, list[dict[str, Any]], bool]:
        facts = _extract_with_llm(fpath, parsed_by_file[fpath], model_name)
        used_fallback = facts is None
        if used_fallback:
            facts = _extract_deterministic(fpath, parsed_by_file[fpath])
        return fpath, facts, used_fallback

    # Each email's extraction is an independent LLM call -- run them concurrently
    # (bounded, and still throttled centrally by llm_factory's rate limiter)
    # instead of one at a time, see src/tools/concurrency.py.
    for fpath, facts, used_fallback in bounded_map(_extract_one, parsed_by_file.keys()):
        if used_fallback:
            warnings.append(f"LLM extraction unavailable for {Path(fpath).name}; used deterministic fallback")
        all_facts.extend(facts)

    if failed_files:
        warnings.append(f"{len(failed_files)} of {raw_row_count} emails failed to parse; directory not invalidated")

    df = pd.DataFrame(all_facts)
    out_path = ctx.artifact_root / "parsed" / "emails.parquet"
    artifact = write_dataframe(df, out_path)

    dates = [f["date"] for f in all_facts if f.get("date")]
    date_min = min(dates) if dates else None
    date_max = max(dates) if dates else None

    return SourceResult(
        source_name="emails",
        status="success" if not failed_files else "partial",
        raw_row_count=raw_row_count,
        accepted_row_count=raw_row_count - len(failed_files),
        rejected_row_count=len(failed_files),
        artifact=artifact,
        schema_version="1.0",
        date_min=date_min,
        date_max=date_max,
        warnings=warnings,
        error=None,
    )
