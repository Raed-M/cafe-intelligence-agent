from __future__ import annotations

import json
import re

import pandas as pd

from src.config.source_registry import SourceConfig
from src.parsers.base import RunContext
from src.schemas.sources import SourceResult
from src.tools.artifact_io import write_dataframe

REQUIRED_FIELDS = ["review_id", "date", "source", "rating", "text"]

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _detect_language(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    return "ar" if _ARABIC_RE.search(text) else "en"


def parse_reviews(source: SourceConfig, ctx: RunContext) -> SourceResult:
    path = ctx.data_dir / source.path
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_row_count = len(data)
    warnings: list[str] = []

    if raw_row_count == 0:
        empty = pd.DataFrame(columns=[*REQUIRED_FIELDS, "language"])
        out_path = ctx.artifact_root / "parsed" / "reviews.parquet"
        artifact = write_dataframe(empty, out_path)
        return SourceResult(
            source_name="reviews", status="success", raw_row_count=0, accepted_row_count=0,
            rejected_row_count=0, artifact=artifact, schema_version="1.0", date_min=None, date_max=None,
            warnings=["customer_reviews.json is a valid empty list; no reviews available this period"],
            error=None,
        )

    df = pd.DataFrame(data)
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"customer_reviews.json missing required fields: {missing_cols}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    bad = (
        df["review_id"].isna() | (df["date_parsed"] == "NaT")
        | df["rating"].isna() | (df["rating"] < 1) | (df["rating"] > 5)
        | ~df["rating"].apply(lambda v: float(v).is_integer() if pd.notna(v) else False)
        | df["text"].isna()
    )
    rejected = df[bad]
    accepted = df[~bad].copy()
    if len(rejected):
        warnings.append(f"{len(rejected)} review rows rejected for invalid id/date/rating/text")

    dup_id = accepted["review_id"].duplicated()
    if dup_id.any():
        warnings.append(f"{dup_id.sum()} duplicate review_id rows dropped")
        accepted = accepted[~dup_id]

    accepted["rating"] = accepted["rating"].astype(int)
    accepted["language"] = accepted["text"].map(_detect_language)
    accepted = accepted.drop(columns=["date"]).rename(columns={"date_parsed": "date"})

    date_min = accepted["date"].min() if len(accepted) else None
    date_max = accepted["date"].max() if len(accepted) else None

    out_path = ctx.artifact_root / "parsed" / "reviews.parquet"
    artifact = write_dataframe(accepted, out_path)

    return SourceResult(
        source_name="reviews",
        status="success" if not len(rejected) else "partial",
        raw_row_count=raw_row_count,
        accepted_row_count=len(accepted),
        rejected_row_count=len(rejected),
        artifact=artifact,
        schema_version="1.0",
        date_min=date_min,
        date_max=date_max,
        warnings=warnings,
        error=None,
    )
