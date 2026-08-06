from __future__ import annotations

from typing import TypedDict


class EmailFact(TypedDict):
    email_file: str
    sender: str | None
    date: str | None
    subject: str | None
    category: str
    entity_or_ingredient: str | None
    old_price: float | None
    new_price: float | None
    currency: str | None
    unit: str | None
    effective_date: str | None
    event_start: str | None
    event_end: str | None
    location: str | None
    facts: list[str]
    confidence: float
    evidence_text: str


class EmailExtractionOutput(TypedDict):
    """Plan section 16.8's named output contract. Wrapped in a top-level object
    because .with_structured_output() constrains one schema per call and a
    single email can yield more than one fact (e.g. two separate price
    changes) -- the model fills `items`, never a bare top-level array."""
    items: list[EmailFact]
